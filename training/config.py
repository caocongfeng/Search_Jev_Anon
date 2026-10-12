"""Configuration handling for SearchJev training.

The single source of hyper-parameters is ``config.yaml`` in the project root.
This module embeds the very same defaults so that a missing key never crashes,
loads the YAML file, applies the per-size preset, ``--set a.b=c`` overrides and
the convenience CLI flags, validates enums, resolves relative paths against the
project root and exposes a small attribute-accessible :class:`Config` object.

Resolution order (later wins)::

    embedded defaults -> config.yaml -> presets[model.size] -> --set overrides -> CLI flags

The preset is chosen from the *final* ``model.size`` (i.e. after taking
``--model``/``--debug``/``--set model.size=`` into account) so that
``--model 27B`` picks up the 27B batch-size preset while an explicit
``--set training.per_device_train_batch_size=2`` still wins over the preset.

This module is deliberately light: it imports neither torch nor transformers.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

#: Project root (``$P``): the parent directory of the ``training`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default YAML file (``$P/config.yaml``).
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Model size table
# ---------------------------------------------------------------------------

#: Canonical size key -> HF repo id of the *instruct* checkpoint.
#: The ``base`` variant appends ``-Base`` (e.g. ``Qwen/Qwen3.5-4B-Base``).
MODEL_SIZES: dict[str, str] = {
    "0.8B": "Qwen/Qwen3.5-0.8B",
    "2B": "Qwen/Qwen3.5-2B",
    "4B": "Qwen/Qwen3.5-4B",
    "9B": "Qwen/Qwen3.5-9B",
    "27B": "Qwen/Qwen3.5-27B",
}

#: Every task family the pipeline knows (whitelist for data.families, data.val_only_families
#: and data.family_modes). Each family is a directory ``<data.root>/<family>/`` of per-dataset
#: jsonl files:
#:
#: * ``unified_v2`` - SearchDecision-Bench (the converted search_jev_unified_v2 release; train/val/test);
#: * ``classification`` / ``ranking`` / ``search`` - SearchJev-V1 (validation only in the paper recipe).
FAMILIES: tuple[str, ...] = ("classification", "ranking", "search", "unified_v2")

#: The three original SearchJev-V1 families.
LEGACY_FAMILIES: tuple[str, ...] = ("classification", "ranking", "search")

# ---------------------------------------------------------------------------
# Embedded defaults (must stay identical to config.yaml)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "run": {
        "name": "searchjev-{size}",  # "{size}" -> lower-case model.size (0.8b / 4b / 9b ...)
        "output_root": "outputs",
        "seed": 42,
        "hf_home": None,
        "report_to": "none",
        "wandb_project": "SearchJev",
        "wandb_entity": None,
    },
    "model": {
        "size": "4B",
        "variant": "instruct",
        "name": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "load_in_4bit": False,
        "gradient_checkpointing": True,
    },
    "lora": {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        ],
        "init_adapter": None,
    },
    "labels": {
        "alphabet": "extended",
        "max_options": None,
        "over_limit_policy": "drop",
    },
    "prompt": {
        "style": "auto",
        "language": "bilingual",
        "max_length": 2048,
    },
    # Paper recipe: train on unified_v2 (SearchDecision-Bench); the SearchJev-V1 families are val-only.
    "data": {
        "root": "data/SearchJev-V2",
        "mode": "sampled",
        "families": ["unified_v2"],
        "family_modes": {"unified_v2": "full"},
        "val_only_families": ["classification", "ranking", "search"],
        "processed_dir": "data/processed",
        "train_splits": ["train"],
        "val_split": "val",
        "val_fallback_per_dataset": 300,
        "max_val_per_dataset": 500,
        "max_train_per_dataset": None,
        "dedup_val_against_train": True,
        # drop train/val examples whose state also occurs in a test row of any prepared family
        "exclude_test_overlap": True,
        "include_datasets": None,
        "exclude_datasets": [],
        "permutation": {"choice": True, "noul": True, "score": False, "copies": 1},
        "prepare_workers": 8,
        "rebuild": False,
    },
    "training": {
        "epochs": 1,
        # paper: 22 per device x 6 GPUs = effective batch 132
        "per_device_train_batch_size": 22,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "bf16": True,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 3,
        "max_steps": -1,
        "max_train_examples": None,
        "max_eval_examples": 2000,
        "dataloader_num_workers": 8,
        # batch examples of similar prompt length together (cached character-based estimates); less padding
        "group_by_length": True,
        "optim": "adamw_torch_fused",
        "resume_from_checkpoint": None,
    },
    "loss": {
        "type": "ce",
        "lambda_brier": 0.5,
        # per-example training-loss weights from row meta (label reliability); None = unweighted.
        # Paper: LLM-assigned labels, demonstrations and constructed rewrites (meta.weak / llm_teacher) get 0.5.
        "sample_weights": {"weak": 0.5, "label_tier": {"llm_teacher": 0.5}},
    },
    "calibration": {
        "enabled": True,
        "fraction": 0.5,
        "per_question_type": True,
        "max_examples": 5000,
    },
    "evaluation": {
        # held-out half of the validation data (the other half fits the temperatures)
        "max_examples": 5000,
    },
    "distributed": {
        "backend": "auto",
        "num_gpus": "all",
        "gpus": None,
        "gpu_selection": "free",
        "min_free_gb": 16,
    },
    "presets": {
        "0.8B": {"training": {"per_device_train_batch_size": 22, "gradient_accumulation_steps": 1}},
        "2B": {"training": {"per_device_train_batch_size": 22, "gradient_accumulation_steps": 1}},
        "4B": {"training": {"per_device_train_batch_size": 22, "gradient_accumulation_steps": 1}},
        "9B": {"training": {"per_device_train_batch_size": 22, "gradient_accumulation_steps": 1}},
        "27B": {"training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 32,
                             "per_device_eval_batch_size": 4}},
    },
}

#: Debug preset applied by ``--debug`` (after everything else).
DEBUG_OVERRIDES: dict[str, Any] = {
    "model": {"size": "0.8B", "variant": "base", "name": None},
    "training": {
        "max_train_examples": 256,
        "max_eval_examples": 64,
        "max_steps": 20,
        "eval_steps": 10,
        "save_steps": 10,
        "logging_steps": 1,
    },
    "calibration": {"max_examples": 256},
    "evaluation": {"max_examples": 256},
    "distributed": {"num_gpus": 1},
}

#: Allowed enum values, keyed by dotted path.
ENUMS: dict[str, tuple[str, ...]] = {
    "run.report_to": ("none", "wandb", "tensorboard"),
    "model.variant": ("instruct", "base"),
    "model.dtype": ("bfloat16", "float16", "float32"),
    "model.attn_implementation": ("sdpa", "eager", "flash_attention_2", "flex_attention"),
    "labels.over_limit_policy": ("drop", "subsample"),
    "prompt.style": ("auto", "chat", "plain"),
    "prompt.language": ("en", "zh", "bilingual"),
    "data.mode": ("sampled", "full"),
    "training.eval_strategy": ("no", "steps", "epoch"),
    "loss.type": ("ce", "kl"),
    "distributed.backend": ("auto", "none", "ddp"),
    "distributed.gpu_selection": ("free", "first"),
}

#: Dotted paths whose values are filesystem paths resolved against the project root.
PATH_KEYS: tuple[str, ...] = ("run.output_root", "data.root", "data.processed_dir")


class ConfigError(ValueError):
    """Raised for an invalid configuration value or override."""


# ---------------------------------------------------------------------------
# Config container
# ---------------------------------------------------------------------------


class Config(dict):
    """Attribute-accessible nested dict (``cfg.data.mode``).

    Nested mappings are wrapped recursively on construction and assignment, so
    ``cfg.training.learning_rate`` works at any depth.  :meth:`to_dict`
    converts back to plain ``dict`` objects (needed for YAML/JSON dumping).
    """

    __slots__ = ()

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        merged: dict[str, Any] = dict(data or {})
        merged.update(kwargs)
        for key, value in merged.items():
            self[key] = value

    # -- mapping protocol ---------------------------------------------------
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def __deepcopy__(self, memo: dict[int, Any]) -> "Config":
        return Config(copy.deepcopy(self.to_dict(), memo))

    # -- conversion -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain, recursively converted ``dict`` copy."""
        return _unwrap(self)

    def to_yaml(self) -> str:
        """Render the configuration as a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False)

    def dump(self, path: str | os.PathLike[str]) -> Path:
        """Write the (resolved) configuration as YAML to ``path`` and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_yaml(), encoding="utf-8")
        logger.info("Wrote resolved config to %s", out)
        return out

    # -- dotted access ------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return the value at ``"a.b.c"`` or ``default`` if any level is missing."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set the value at ``"a.b.c"``, creating intermediate mappings."""
        _set_dotted(self, dotted, value)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Merging / dotted helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], update: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge ``update`` into ``base`` **in place** and return ``base``.

    Mappings are merged key by key; any other value (including lists) replaces
    the previous one.  A ``None`` in ``update`` is a real value (it clears the
    key), which is how ``max_train_per_dataset: null`` works.
    """
    if not update:
        return base
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(_unwrap(value)) if isinstance(value, (Mapping, list)) else value
    return base


def _set_dotted(node: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Mapping):
            node[part] = {} if not isinstance(node, Config) else Config()
        node = node[part]
    node[parts[-1]] = value


def _get_dotted(node: Any, dotted: str, default: Any = None) -> Any:
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def parse_override(item: str) -> tuple[str, Any]:
    """Parse one ``a.b.c=value`` override; the value is YAML-parsed.

    ``training.max_steps=100`` -> ``("training.max_steps", 100)``;
    ``data.families=[classification,search]`` -> list;
    ``data.mode=full`` -> ``"full"``; ``x=null`` -> ``None``.
    """
    if "=" not in item:
        raise ConfigError(f"override must look like a.b=c, got {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_.\-]+", key):
        raise ConfigError(f"invalid override key {key!r} in {item!r}")
    raw = raw.strip()
    try:
        value = yaml.safe_load(raw) if raw != "" else ""
    except yaml.YAMLError:
        value = raw
    # YAML 1.1 reads "no"/"yes"/"on"/"off" as booleans and "4" as an int; keep the
    # literal text for string-valued keys (training.eval_strategy=no,
    # distributed.num_gpus=4 -> normalised later in _finalize).
    default = _get_dotted(DEFAULTS, key, default=None)
    if isinstance(default, str) and value is not None and not isinstance(value, str):
        value = raw
    return key, value


def parse_overrides(items: Iterable[str] | None) -> list[tuple[str, Any]]:
    """Parse a sequence of ``a.b=c`` strings (see :func:`parse_override`)."""
    return [parse_override(it) for it in (items or [])]


# ---------------------------------------------------------------------------
# Model size helpers
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"(?i)(?:^|[^0-9.])(0\.8|2|4|9|27)\s*b(?:$|[^a-z0-9])")


def try_normalize_size(value: Any) -> str | None:
    """Return the canonical size key (``"4B"``) for ``4b``/``Qwen3.5-4B``/... or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    direct = text.upper()
    if direct in MODEL_SIZES:
        return direct
    # Accept things like "qwen3.5-4b", "Qwen/Qwen3.5-0.8B-Base", "4b", "27B-instruct".
    match = _SIZE_RE.search(" " + text + " ")
    if match:
        key = match.group(1) + "B"
        if key in MODEL_SIZES:
            return key
    return None


def normalize_size(value: Any) -> str:
    """Like :func:`try_normalize_size` but raises :class:`ConfigError` when unknown."""
    key = try_normalize_size(value)
    if key is None:
        raise ConfigError(f"unknown model size {value!r}; expected one of {list(MODEL_SIZES)}")
    return key


def resolve_model_name(cfg: Mapping[str, Any]) -> str:
    """Return the HF id / local path of the base model for ``cfg``.

    ``model.name`` (explicit id or path) wins; otherwise the size+variant table
    is used: ``Qwen/Qwen3.5-<size>`` for ``instruct`` and ``...-Base`` for ``base``.
    """
    model = cfg.get("model", {})
    name = model.get("name")
    if name:
        return str(name)
    size = normalize_size(model.get("size", "4B"))
    variant = str(model.get("variant", "instruct")).lower()
    if variant not in ENUMS["model.variant"]:
        raise ConfigError(f"model.variant must be one of {ENUMS['model.variant']}, got {variant!r}")
    base = MODEL_SIZES[size]
    return base + "-Base" if variant == "base" else base


def effective_batch_size(cfg: Mapping[str, Any], world_size: int = 1) -> int:
    """Global optimisation batch size = per-device batch x grad-accum x world size."""
    training = cfg.get("training", {})
    per_device = int(training.get("per_device_train_batch_size", 1))
    accum = int(training.get("gradient_accumulation_steps", 1))
    return per_device * accum * max(1, int(world_size))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    unknown = sorted(set(loaded) - set(DEFAULTS))
    if unknown:
        logger.warning("%s: unknown top-level keys %s (kept, but unused)", path, unknown)
    return dict(loaded)


def _split_list(value: Any) -> Any:
    """Turn ``"a,b"`` into ``["a", "b"]``; lists pass through; ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() if isinstance(v, str) else v for v in value]
    return value


def _coerce_scalars(node: dict[str, Any], defaults: Mapping[str, Any], prefix: str = "") -> None:
    """Coerce string values to the numeric/bool type of the matching default.

    Handles YAML quirks such as ``5e-5`` (parsed as a string by YAML 1.1) and
    ``--set training.bf16=false`` when the caller passed the raw string.
    """
    for key, default in defaults.items():
        if key not in node:
            continue
        value = node[key]
        path = f"{prefix}{key}"
        if isinstance(default, Mapping):
            if isinstance(value, Mapping):
                if not isinstance(value, dict):
                    value = dict(value)
                    node[key] = value
                _coerce_scalars(value, default, prefix=path + ".")
            continue
        if isinstance(default, list) and isinstance(value, str):
            node[key] = _split_list(value)
            continue
        if not isinstance(value, str):
            continue
        text = value.strip()
        if isinstance(default, bool):
            low = text.lower()
            if low in ("true", "yes", "on", "1"):
                node[key] = True
            elif low in ("false", "no", "off", "0"):
                node[key] = False
            else:
                raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        elif isinstance(default, int) and not isinstance(default, bool):
            try:
                node[key] = int(text)
            except ValueError:
                try:
                    node[key] = float(text)
                except ValueError:
                    raise ConfigError(f"{path}: expected a number, got {value!r}") from None
        elif isinstance(default, float):
            try:
                node[key] = float(text)
            except ValueError:
                raise ConfigError(f"{path}: expected a number, got {value!r}") from None


def _effective_size(cfg: Mapping[str, Any], overrides: list[tuple[str, Any]], cli: Mapping[str, Any]) -> str:
    """Determine the final ``model.size`` before presets are applied."""
    size: Any = _get_dotted(cfg, "model.size", "4B")
    for key, value in overrides:
        if key == "model.size":
            size = value
    cli_model = cli.get("model")
    if cli.get("debug") and cli_model is None and "model.size" not in {k for k, _ in overrides}:
        size = DEBUG_OVERRIDES["model"]["size"]
    if cli_model is not None and try_normalize_size(cli_model) is not None:
        size = cli_model
    return normalize_size(size)


def _flatten_keys(node: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted leaf keys of a nested mapping."""
    out: list[str] = []
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.extend(_flatten_keys(value, path + "."))
        else:
            out.append(path)
    return out


def _apply_cli(cfg: dict[str, Any], cli: Mapping[str, Any], explicit: Iterable[str] = ()) -> None:
    """Apply the convenience CLI flags (``--model``, ``--full``, ...).

    ``explicit`` lists the dotted keys given via ``--set``; the ``--debug``
    preset never overrides one of those (``--debug --set training.max_steps=8``
    runs 8 steps), while the other debug values still apply.
    """
    explicit = set(explicit)
    model = cli.get("model")
    if model is not None:
        size = try_normalize_size(model)
        if size is not None and "/" not in str(model) and not os.path.exists(str(model)):
            _set_dotted(cfg, "model.size", size)
            _set_dotted(cfg, "model.name", None)
        else:
            # An explicit HF id or local path.
            _set_dotted(cfg, "model.name", str(model))
            if size is not None:
                _set_dotted(cfg, "model.size", size)
    if cli.get("variant") is not None:
        _set_dotted(cfg, "model.variant", str(cli["variant"]).lower())
        explicit.add("model.variant")
    if cli.get("full"):
        _set_dotted(cfg, "data.mode", "full")
    if cli.get("families") is not None:
        _set_dotted(cfg, "data.families", _split_list(cli["families"]))
    if cli.get("gpus") is not None:
        gpus = cli["gpus"]
        if isinstance(gpus, (list, tuple)):
            gpus = ",".join(str(g) for g in gpus)
        _set_dotted(cfg, "distributed.gpus", str(gpus))
    if cli.get("num_gpus") is not None:
        _set_dotted(cfg, "distributed.num_gpus", cli["num_gpus"])
        explicit.add("distributed.num_gpus")
    if cli.get("gpus") is not None:
        explicit.add("distributed.num_gpus")  # explicit ids win over the debug 1-GPU default
    if cli.get("wandb"):
        _set_dotted(cfg, "run.report_to", "wandb")
        _set_dotted(cfg, "run.wandb_project", str(cli["wandb"]))
    if cli.get("name") is not None:
        _set_dotted(cfg, "run.name", str(cli["name"]))
    if cli.get("resume") is not None:
        _set_dotted(cfg, "training.resume_from_checkpoint", str(cli["resume"]))
    if cli.get("debug"):
        for dotted in _flatten_keys(DEBUG_OVERRIDES):
            if dotted in explicit or (dotted.startswith("model.") and cli.get("model") is not None):
                continue  # an explicit --set / --model wins over the debug preset
            _set_dotted(cfg, dotted, _get_dotted(DEBUG_OVERRIDES, dotted))
        name = str(_get_dotted(cfg, "run.name", "jev"))
        if not name.endswith("-debug"):
            _set_dotted(cfg, "run.name", name + "-debug")


def _normalize_family_modes(value: Any) -> dict[str, str]:
    """Validate ``data.family_modes`` (``{family: sampled|full}``; ``None`` -> ``{}``)."""
    if value in (None, "", "null", "None", "none"):
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"data.family_modes must be a mapping family -> sampled|full, got {value!r}")
    out: dict[str, str] = {}
    for fam, mode in value.items():
        fam = str(fam).strip()
        if fam not in FAMILIES:
            raise ConfigError(f"data.family_modes: unknown family {fam!r}; allowed {list(FAMILIES)}")
        mode = str(mode).strip().lower()
        if mode not in ENUMS["data.mode"]:
            raise ConfigError(f"data.family_modes.{fam} must be one of {ENUMS['data.mode']}, got {mode!r}")
        out[fam] = mode
    return out


def family_mode(cfg: Mapping[str, Any], family: str) -> str:
    """Effective source mode of ``family``: ``data.family_modes[family]`` or else ``data.mode``.

    It only decides which source files are read (``sampled_300k/<family>_300k.jsonl`` vs every
    per-dataset file); the processed cache stays at ``<processed_dir>/<data.mode>/<family>``.
    """
    data = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    modes = data.get("family_modes") or {}
    return str(modes.get(family) or data.get("mode") or "sampled").lower()


def prepared_families(cfg: Mapping[str, Any]) -> list[str]:
    """Families ``prepare_data`` handles: ``data.families`` (train+val) then ``data.val_only_families`` (val only)."""
    data = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    out = list(data.get("families") or [])
    return out + [f for f in (data.get("val_only_families") or []) if f not in out]


def _resolve_path(value: str) -> str:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _finalize(cfg: dict[str, Any]) -> None:
    """Normalise, validate and resolve paths in place."""
    _coerce_scalars(cfg, DEFAULTS)

    # model
    model = cfg["model"]
    model["size"] = normalize_size(model.get("size", "4B"))
    model["variant"] = str(model.get("variant", "instruct")).lower()
    if model.get("name") in ("", "null", "None"):
        model["name"] = None
    if model.get("name"):
        candidate = Path(os.path.expanduser(str(model["name"])))
        if not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            model["name"] = str(candidate.resolve())

    # lists that may come as comma-separated strings
    data = cfg["data"]
    data["families"] = _split_list(data.get("families")) or []
    data["val_only_families"] = _split_list(data.get("val_only_families")) or []
    data["family_modes"] = _normalize_family_modes(data.get("family_modes"))
    data["train_splits"] = _split_list(data.get("train_splits")) or []
    data["exclude_datasets"] = _split_list(data.get("exclude_datasets")) or []
    data["include_datasets"] = _split_list(data.get("include_datasets"))
    cfg["lora"]["target_modules"] = _split_list(cfg["lora"].get("target_modules")) or []
    if isinstance(cfg["labels"].get("alphabet"), (list, tuple)):
        cfg["labels"]["alphabet"] = [str(a) for a in cfg["labels"]["alphabet"]]
    if cfg["distributed"].get("gpus") is not None:
        gpus = cfg["distributed"]["gpus"]
        if isinstance(gpus, (list, tuple)):
            gpus = ",".join(str(g) for g in gpus)
        cfg["distributed"]["gpus"] = str(gpus).replace(" ", "")
        if cfg["distributed"]["gpus"].lower() in ("", "null", "none", "all"):
            cfg["distributed"]["gpus"] = None
    dist = cfg["distributed"]
    num = dist.get("num_gpus", "all")
    if isinstance(num, bool):
        raise ConfigError("distributed.num_gpus must be 'all' or a positive integer")
    if num is None or (isinstance(num, str) and num.strip().lower() in ("", "all", "auto", "null", "none", "max")):
        dist["num_gpus"] = "all"
    else:
        try:
            n_gpus = int(num)
        except (TypeError, ValueError):
            raise ConfigError(f"distributed.num_gpus must be 'all' or a positive integer, got {num!r}") from None
        if n_gpus < 1:
            raise ConfigError("distributed.num_gpus must be >= 1 (or 'all')")
        dist["num_gpus"] = n_gpus
    if dist.get("min_free_gb") in ("", "null", "None", "none"):
        dist["min_free_gb"] = None
    if dist.get("min_free_gb") is not None:
        dist["min_free_gb"] = float(dist["min_free_gb"])
    dist["gpu_selection"] = str(dist.get("gpu_selection") or "free").lower()
    for section in ("calibration", "evaluation"):
        node = cfg.setdefault(section, {})
        max_examples = node.get("max_examples", DEFAULTS[section]["max_examples"])
        if max_examples in ("", "null", "None", "none"):
            max_examples = None
        node["max_examples"] = None if max_examples is None else int(max_examples)
        if node["max_examples"] is not None and node["max_examples"] < 1:
            raise ConfigError(f"{section}.max_examples must be >= 1 or null")
    lora_cfg = cfg["lora"]
    if lora_cfg.get("init_adapter") in ("", "null", "None", "none"):
        lora_cfg["init_adapter"] = None
    if lora_cfg.get("init_adapter"):
        adapter_path = Path(os.path.expanduser(str(lora_cfg["init_adapter"])))
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        if not adapter_path.exists():
            raise ConfigError(f"lora.init_adapter not found: {adapter_path}")
        lora_cfg["init_adapter"] = str(adapter_path.resolve())

    # enums
    for dotted, allowed in ENUMS.items():
        value = _get_dotted(cfg, dotted)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.lower() if dotted != "model.attn_implementation" else value
            _set_dotted(cfg, dotted, value)
        if value not in allowed:
            raise ConfigError(f"{dotted} must be one of {allowed}, got {value!r}")

    alphabet = cfg["labels"].get("alphabet")
    if isinstance(alphabet, str):
        if alphabet not in ("basic", "extended"):
            raise ConfigError(f"labels.alphabet must be 'basic', 'extended' or a list, got {alphabet!r}")
    elif not isinstance(alphabet, list) or not alphabet:
        raise ConfigError("labels.alphabet must be 'basic', 'extended' or a non-empty list of labels")

    unknown_families = [f for f in data["families"] if f not in FAMILIES]
    if unknown_families:
        raise ConfigError(f"data.families contains unknown families {unknown_families}; allowed {list(FAMILIES)}")
    if not data["families"]:
        raise ConfigError("data.families must not be empty")
    unknown_val_only = [f for f in data["val_only_families"] if f not in FAMILIES]
    if unknown_val_only:
        raise ConfigError(f"data.val_only_families contains unknown families {unknown_val_only}; allowed {list(FAMILIES)}")
    overlap = [f for f in data["val_only_families"] if f in data["families"]]
    if overlap:
        raise ConfigError(
            f"data.val_only_families {overlap} are also in data.families {data['families']}; a family is either "
            "trained on (train+val) or val-only - e.g. --set data.val_only_families=[] when training on them"
        )
    if "test" in data["train_splits"]:
        raise ConfigError("data.train_splits must never contain 'test'")
    if data.get("val_split") == "test":
        raise ConfigError("data.val_split must not be 'test'")
    if data.get("val_split") in data["train_splits"]:
        raise ConfigError("data.val_split must not also be in data.train_splits")

    perm = data.get("permutation") or {}
    if not isinstance(perm, Mapping):
        raise ConfigError("data.permutation must be a mapping")
    copies = int(perm.get("copies", 1) or 1)
    if copies < 1:
        raise ConfigError("data.permutation.copies must be >= 1")
    perm = dict(perm)
    perm["copies"] = copies
    for qtype in ("choice", "noul", "score"):
        perm[qtype] = bool(perm.get(qtype, False))
    data["permutation"] = perm

    training = cfg["training"]
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps"):
        if int(training.get(key, 1)) < 1:
            raise ConfigError(f"training.{key} must be >= 1")
    if float(training.get("learning_rate", 0)) <= 0:
        raise ConfigError("training.learning_rate must be > 0")
    if not 0.0 <= float(training.get("warmup_ratio", 0)) <= 1.0:
        raise ConfigError("training.warmup_ratio must be in [0, 1]")
    if training.get("resume_from_checkpoint") in ("", "null", "None", "none"):
        training["resume_from_checkpoint"] = None

    sw = cfg["loss"].get("sample_weights")
    if sw in ("", "null", "None", "none", {}, False):
        sw = None
    if sw is not None:
        if not isinstance(sw, Mapping):
            raise ConfigError(f"loss.sample_weights must be a mapping (e.g. {{weak: 0.5, label_tier: {{llm_teacher: 0.5}}}}) or null, got {sw!r}")
        clean: dict[str, Any] = {}
        for key, rule in sw.items():
            if isinstance(rule, Mapping):
                clean[str(key)] = {str(v): float(w) for v, w in rule.items()}
                bad = [w for w in clean[str(key)].values() if not 0.0 <= w <= 10.0]
            else:
                clean[str(key)] = float(rule)
                bad = [] if 0.0 <= clean[str(key)] <= 10.0 else [clean[str(key)]]
            if bad:
                raise ConfigError(f"loss.sample_weights.{key}: weights must be in [0, 10], got {bad}")
        sw = clean
    cfg["loss"]["sample_weights"] = sw
    lam = float(cfg["loss"].get("lambda_brier", 0.0))
    if lam < 0:
        raise ConfigError("loss.lambda_brier must be >= 0")
    frac = float(cfg["calibration"].get("fraction", 0.5))
    if not 0.0 < frac < 1.0:
        raise ConfigError("calibration.fraction must be strictly between 0 and 1")
    if cfg["labels"].get("max_options") is not None and int(cfg["labels"]["max_options"]) < 2:
        raise ConfigError("labels.max_options must be >= 2 (or null)")
    if int(cfg["prompt"].get("max_length", 0)) < 64:
        raise ConfigError("prompt.max_length is unreasonably small (< 64)")
    r = int(cfg["lora"].get("r", 0))
    if cfg["lora"].get("enabled") and r < 1:
        raise ConfigError("lora.r must be >= 1 when lora.enabled")

    # paths
    for dotted in PATH_KEYS:
        value = _get_dotted(cfg, dotted)
        if value is not None:
            _set_dotted(cfg, dotted, _resolve_path(value))
    run = cfg["run"]
    hf_home = run.get("hf_home")
    run["hf_home"] = None if hf_home in (None, "", "null", "None", "none") else str(Path(os.path.expanduser(str(hf_home))))
    run["name"] = str(run.get("name") or DEFAULTS["run"]["name"]).strip().replace("{size}", str(model["size"]).lower())
    if not run["name"] or "/" in run["name"] or run["name"] in (".", ".."):
        raise ConfigError(f"run.name must be a plain directory name, got {run['name']!r}")
    run["output_dir"] = str(Path(run["output_root"]) / run["name"])
    run["seed"] = int(run.get("seed", 42))
    run["model_name"] = resolve_model_name(cfg)


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] = (),
    cli: Mapping[str, Any] | None = None,
) -> Config:
    """Load, merge, validate and resolve the configuration.

    Args:
        path: YAML file; ``None`` uses ``$P/config.yaml`` when it exists
            (embedded defaults alone otherwise).
        overrides: ``["a.b=c", ...]`` strings from ``--set`` (YAML-parsed values).
        cli: convenience flags; recognised keys are ``model``, ``variant``,
            ``full``, ``families``, ``gpus``, ``name``, ``resume`` and ``debug``.

    Returns:
        A :class:`Config` with, in addition to the YAML keys, ``run.output_dir``
        (``output_root/name``) and ``run.model_name`` (resolved base model).
    """
    cfg: dict[str, Any] = copy.deepcopy(DEFAULTS)

    yaml_path: Path | None
    if path is None:
        yaml_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
    else:
        yaml_path = Path(path).expanduser()
        if not yaml_path.is_absolute() and not yaml_path.exists():
            yaml_path = PROJECT_ROOT / yaml_path
        if not yaml_path.exists():
            raise ConfigError(f"config file not found: {path}")
    if yaml_path is not None:
        logger.info("Loading config from %s", yaml_path)
        deep_merge(cfg, _read_yaml(yaml_path))
    else:
        logger.info("No config.yaml found; using embedded defaults")

    cli = dict(cli or {})
    parsed = parse_overrides(overrides)

    # presets for the final model size (deep-merge over defaults+yaml)
    size = _effective_size(cfg, parsed, cli)
    presets = cfg.get("presets") or {}
    preset = None
    for key, value in presets.items():
        if try_normalize_size(key) == size:
            preset = value
            break
    if preset:
        logger.info("Applying preset for model size %s: %s", size, preset)
        deep_merge(cfg, preset)
    _set_dotted(cfg, "model.size", size)

    # --set overrides
    for key, value in parsed:
        current = _get_dotted(cfg, key, default=None)
        if isinstance(current, list) and isinstance(value, str):
            value = _split_list(value)
        _set_dotted(cfg, key, value)

    # convenience CLI flags (win over everything; --debug does not touch --set keys)
    _apply_cli(cfg, cli, explicit=[key for key, _ in parsed])

    _finalize(cfg)
    return Config(cfg)


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULTS",
    "DEBUG_OVERRIDES",
    "DEFAULT_CONFIG_PATH",
    "ENUMS",
    "FAMILIES",
    "LEGACY_FAMILIES",
    "MODEL_SIZES",
    "family_mode",
    "prepared_families",
    "PROJECT_ROOT",
    "deep_merge",
    "effective_batch_size",
    "load_config",
    "normalize_size",
    "parse_override",
    "parse_overrides",
    "resolve_model_name",
    "try_normalize_size",
]
