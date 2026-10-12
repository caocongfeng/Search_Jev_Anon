"""Model utilities for Qwen3.5-JEV.

This module owns everything that touches the backbone:

* the size table (``0.8B`` .. ``27B`` -> Hugging Face model ids),
* tokenizer loading (left padding, pad token),
* model loading in bf16 (optionally 4-bit NF4 for QLoRA),
* LoRA injection (matched by module leaf name, covering both the
  Gated-DeltaNet ``linear_attn.*`` projections and the full-attention
  ``self_attn.*`` projections),
* the *restricted first-token readout* ``restricted_menu_logits`` which is
  the heart of the JEV adaptation: with left padding and ``logits_to_keep=1``
  we take the logits of the very next token and gather only the label-token
  columns (``A``, ``B``, ...) so that the model becomes a K-way classifier over
  a dynamic option menu without adding any head,
* ``load_for_inference`` for run directories produced by ``start_training.py``.

Nothing here loads weights at import time; the module is importable without a
GPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Model-size table. The ``base`` variant appends ``-Base`` to the id.
SIZES: dict[str, str] = {
    "0.8B": "Qwen/Qwen3.5-0.8B",
    "2B": "Qwen/Qwen3.5-2B",
    "4B": "Qwen/Qwen3.5-4B",
    "9B": "Qwen/Qwen3.5-9B",
    "27B": "Qwen/Qwen3.5-27B",
}

#: Default LoRA targets: full attention, Gated-DeltaNet linear attention, MLP.
DEFAULT_LORA_TARGETS: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def to_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Map a dtype name (``"bfloat16"``, ``"bf16"``, ...) or ``torch.dtype`` to ``torch.dtype``."""
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[key]


def normalize_size(size: str) -> str:
    """Normalise a size spelling (``4b``, ``4B``, ``Qwen3.5-4B``, ``qwen/qwen3.5-4b-base``) to a ``SIZES`` key.

    Raises:
        ValueError: if the size cannot be matched.
    """
    s = str(size).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", s)
    if m is None:
        raise ValueError(f"Cannot parse model size from {size!r}; expected one of {sorted(SIZES)}")
    num = m.group(1)
    if num.endswith(".0"):
        num = num[:-2]
    key = f"{num}B"
    if key not in SIZES:
        raise ValueError(f"Unsupported model size {size!r}; expected one of {sorted(SIZES)}")
    return key


def resolve_model_id(size: str, variant: str = "instruct") -> str:
    """Return the Hugging Face model id for ``size`` and ``variant`` (``instruct`` | ``base``)."""
    key = normalize_size(size)
    v = str(variant).lower().strip()
    if v in {"instruct", "it", "chat", ""}:
        return SIZES[key]
    if v in {"base", "pretrained", "pt"}:
        return SIZES[key] + "-Base"
    raise ValueError(f"Unknown model variant {variant!r}; expected 'instruct' or 'base'")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or attribute-style config object."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _set_config_flag(config: Any, name: str, value: Any) -> None:
    """Set ``name`` on a model config and on its ``text_config`` when present (multimodal configs)."""
    if config is None:
        return
    if hasattr(config, name):
        setattr(config, name, value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        setattr(text_config, name, value)


def _local_rank_device() -> str:
    """Device string for this process: ``cuda:<LOCAL_RANK>`` when CUDA is available, else ``cpu``."""
    if torch.cuda.is_available():
        return f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    return "cpu"


# --------------------------------------------------------------------------- #
# tokenizer / model loading
# --------------------------------------------------------------------------- #
def load_tokenizer(name_or_path: str) -> Any:
    """Load the tokenizer with **left** padding and a pad token.

    Left padding is mandatory for the restricted readout: the answer position
    must be the last column of the batch for every row.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; cannot pad batches")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token missing; using eos_token %r as pad", tokenizer.eos_token)
    logger.info(
        "Loaded tokenizer %s (%s): vocab=%d pad=%r(%d) padding_side=%s chat_template=%s",
        name_or_path, type(tokenizer).__name__, len(tokenizer), tokenizer.pad_token,
        tokenizer.pad_token_id, tokenizer.padding_side, bool(getattr(tokenizer, "chat_template", None)),
    )
    return tokenizer


def load_model(
    name_or_path: str,
    *,
    dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
    device_map: Any | None = None,
) -> Any:
    """Load a causal LM backbone (``Qwen3_5ForCausalLM`` for Qwen3.5 checkpoints).

    Args:
        name_or_path: HF id or local directory.
        dtype: compute / weight dtype (``bfloat16`` by default; transformers 5.x ``dtype`` kwarg).
        attn_implementation: ``sdpa`` (default) or ``eager``.
        load_in_4bit: QLoRA loading with bitsandbytes NF4 + double quantisation and
            bf16 compute; ``peft.prepare_model_for_kbit_training`` is applied.
        gradient_checkpointing: enable non-reentrant activation checkpointing and
            disable the KV cache.
        device_map: passed through to ``from_pretrained``. ``None`` leaves the
            weights on CPU (the Trainer moves them), except for 4-bit loading where
            the current local-rank GPU is used because bitsandbytes needs CUDA.

    Returns:
        The loaded model (a ``PreTrainedModel``).
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = to_torch_dtype(dtype)
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if device_map is None:
            device_map = {"": _local_rank_device()}
    if device_map is not None:
        kwargs["device_map"] = device_map

    logger.info(
        "Loading model %s (dtype=%s attn=%s 4bit=%s device_map=%s)",
        name_or_path, torch_dtype, attn_implementation, load_in_4bit, device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(str(name_or_path), **kwargs)
    logger.info("Loaded %s", type(model).__name__)

    gc_kwargs = {"use_reentrant": False}
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=gc_kwargs,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    if gradient_checkpointing:
        _set_config_flag(model.config, "use_cache", False)
        logger.info("Gradient checkpointing enabled (non-reentrant); use_cache=False")

    stats = count_parameters(model)
    logger.info("Parameters: total=%.3fB", stats["total"] / 1e9)
    return model


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def linear_leaf_names(model: torch.nn.Module) -> set[str]:
    """Leaf names of all linear-like modules (``nn.Linear`` and bitsandbytes ``Linear4bit``/``Linear8bitLt``)."""
    names: set[str] = set()
    for name, module in model.named_modules():
        cls = type(module).__name__
        if isinstance(module, torch.nn.Linear) or cls in {"Linear4bit", "Linear8bitLt"}:
            names.add(name.rsplit(".", 1)[-1])
    return names


def resolve_lora_targets(model: torch.nn.Module, targets: list[str] | str | None) -> tuple[list[str], list[str]]:
    """Split requested LoRA targets into ``(present, missing)`` by matching module leaf names."""
    if targets is None:
        targets = list(DEFAULT_LORA_TARGETS)
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    leafs = linear_leaf_names(model)
    present = [t for t in targets if t in leafs]
    missing = [t for t in targets if t not in leafs]
    return present, missing


def apply_lora(model: torch.nn.Module, lora_cfg: Any) -> Any:
    """Wrap ``model`` in a PEFT LoRA adapter.

    Args:
        model: backbone from :func:`load_model`.
        lora_cfg: mapping or attribute object with ``r``, ``alpha``, ``dropout``,
            ``target_modules`` (list of module leaf names, e.g. ``q_proj``,
            ``in_proj_qkv``); missing keys fall back to ``r=16, alpha=32,
            dropout=0.05`` and :data:`DEFAULT_LORA_TARGETS`.

    Returns:
        A ``peft.PeftModel`` with ``task_type="CAUSAL_LM"``.
    """
    from peft import LoraConfig, get_peft_model

    r = int(_cfg_get(lora_cfg, "r", 16))
    alpha = int(_cfg_get(lora_cfg, "alpha", _cfg_get(lora_cfg, "lora_alpha", 2 * r)))
    dropout = float(_cfg_get(lora_cfg, "dropout", _cfg_get(lora_cfg, "lora_dropout", 0.05)))
    requested = _cfg_get(lora_cfg, "target_modules", None)
    present, missing = resolve_lora_targets(model, requested)
    if missing:
        logger.warning("LoRA targets not found in model and skipped: %s", missing)
    if not present:
        raise ValueError(
            f"None of the LoRA target modules {requested!r} exist in the model; "
            f"available linear leaf names: {sorted(linear_leaf_names(model))}"
        )

    if getattr(model, "is_gradient_checkpointing", False):
        # Required so that activations flowing into checkpointed blocks carry
        # grad even though the (frozen) embedding output does not.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(present),
    )
    peft_model = get_peft_model(model, config)
    stats = count_parameters(peft_model)
    logger.info(
        "LoRA applied: r=%d alpha=%d dropout=%.3f targets=%s | trainable=%.2fM / total=%.2fM (%.3f%%)",
        r, alpha, dropout, present, stats["trainable"] / 1e6, stats["total"] / 1e6, stats["trainable_percent"],
    )
    return peft_model


def load_lora_for_training(model: torch.nn.Module, adapter_path: str) -> Any:
    """Load an existing LoRA adapter as a *trainable* initialisation (stage-2 training).

    Args:
        model: backbone from :func:`load_model` (same base as the adapter).
        adapter_path: directory produced by a previous run (``<run>/adapter``).

    Returns:
        A trainable ``peft.PeftModel`` whose LoRA weights start from ``adapter_path``.
    """
    from peft import PeftModel

    if getattr(model, "is_gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    stats = count_parameters(peft_model)
    logger.info(
        "LoRA adapter loaded for continued training from %s | trainable=%.2fM / total=%.2fM (%.3f%%)",
        adapter_path, stats["trainable"] / 1e6, stats["total"] / 1e6, stats["trainable_percent"],
    )
    return peft_model


def count_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    """Return ``{"total", "trainable", "frozen", "trainable_percent"}`` parameter counts."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        # bitsandbytes 4-bit params store packed values; report unpacked count when known.
        if type(p).__name__ == "Params4bit" and hasattr(p, "quant_storage"):
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / total if total else 0.0
    return {"total": total, "trainable": trainable, "frozen": total - trainable, "trainable_percent": pct}


# --------------------------------------------------------------------------- #
# restricted readout
# --------------------------------------------------------------------------- #
def restricted_menu_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Restricted first-token readout.

    Runs the backbone on a **left-padded** batch, keeps only the logits of the
    last position (``logits_to_keep=1``) and gathers the columns of the label
    tokens, giving ``z[b, j] = logit(label_j | prompt_b)``.

    Args:
        model: backbone or PEFT model (may be DDP-wrapped).
        input_ids: ``[B, T]`` left-padded token ids.
        attention_mask: ``[B, T]``; its last column must be all ones.
        label_token_ids: ``[B, K]`` token ids of the displayed labels (padded
            slots may hold any valid id; the caller masks them in the loss).

    Returns:
        ``[B, K]`` float32 logits over the menu.
    """
    if attention_mask is not None and attention_mask.shape[-1] > 0:
        if not bool(attention_mask[:, -1].all()):
            raise ValueError(
                "restricted_menu_logits requires LEFT padding: the last attention_mask column must be all ones"
            )
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=1,
        use_cache=False,
    )
    last = out.logits[:, -1, :].float()  # [B, V]
    ids = label_token_ids.to(device=last.device, dtype=torch.long)
    return last.gather(1, ids)


# --------------------------------------------------------------------------- #
# inference loading
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _base_from_resolved_config(run_dir: Path) -> str | None:
    """Recover the base model id from ``<run>/config.resolved.yaml`` if present."""
    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = _read_yaml(cfg_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not parse %s: %s", cfg_path, exc)
        return None
    model_cfg = cfg.get("model", {}) or {}
    if model_cfg.get("name"):
        return str(model_cfg["name"])
    size = model_cfg.get("size")
    if size:
        try:
            return resolve_model_id(str(size), str(model_cfg.get("variant", "instruct")))
        except ValueError:
            return None
    return None


def resolve_run_layout(run_dir_or_name: str) -> dict[str, Any]:
    """Inspect a path / HF id and describe how to load it.

    Returns a dict with keys ``kind`` (``"adapter"`` | ``"local_model"`` | ``"hub"``),
    ``run_dir`` (directory holding ``config.resolved.yaml`` / ``calibration.json`` or
    ``None``), ``weights`` (adapter dir, model dir or HF id) and ``adapter_base``
    (from ``adapter_config.json`` when applicable).
    """
    p = Path(str(run_dir_or_name)).expanduser()
    layout: dict[str, Any] = {"kind": "hub", "run_dir": None, "weights": str(run_dir_or_name), "adapter_base": None}
    if not p.exists():
        return layout
    candidates_adapter = [p, p / "adapter"]
    for cand in candidates_adapter:
        if (cand / "adapter_config.json").exists():
            layout["kind"] = "adapter"
            layout["weights"] = str(cand)
            layout["run_dir"] = str(p if cand == p / "adapter" else p.parent)
            try:
                with (cand / "adapter_config.json").open("r", encoding="utf-8") as fh:
                    layout["adapter_base"] = json.load(fh).get("base_model_name_or_path")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not read adapter_config.json in %s: %s", cand, exc)
            return layout
    for cand, run_dir in [(p / "merged", p), (p / "model", p), (p, None)]:
        if (cand / "config.json").exists():
            layout["kind"] = "local_model"
            layout["weights"] = str(cand)
            if run_dir is None and (p.parent / "config.resolved.yaml").exists():
                run_dir = p.parent  # e.g. <run>/merged passed directly
            layout["run_dir"] = str(run_dir) if run_dir is not None else None
            return layout
    # A run dir that exists but has no weights yet.
    layout["run_dir"] = str(p) if (p / "config.resolved.yaml").exists() else None
    return layout


def load_for_inference(
    run_dir_or_name: str,
    *,
    base_model: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
    device: str = "cuda",
    merge: bool = False,
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = False,
) -> tuple[Any, Any, dict | None]:
    """Load a trained run (or a plain model id) for inference.

    Accepted inputs:

    * a run directory produced by ``start_training.py`` (``<run>/adapter`` or
      ``<run>/model``; reads ``<run>/config.resolved.yaml`` and
      ``<run>/calibration.json``),
    * an adapter directory itself (``adapter_config.json`` inside),
    * a local model directory or a Hugging Face id (zero-shot base model).

    The base model for an adapter is resolved from ``base_model``, then from
    ``adapter_config.json``, then from ``config.resolved.yaml``.

    Returns:
        ``(model.eval(), tokenizer, calibration_dict_or_None)``.
    """
    layout = resolve_run_layout(run_dir_or_name)
    run_dir = Path(layout["run_dir"]) if layout["run_dir"] else None
    device_map = {"": device} if device else None

    if layout["kind"] == "adapter":
        from peft import PeftModel

        adapter_dir = Path(layout["weights"])
        base = base_model or layout["adapter_base"] or (_base_from_resolved_config(run_dir) if run_dir else None)
        if not base:
            raise ValueError(
                f"Cannot determine the base model for adapter {adapter_dir}; pass base_model=..."
            )
        tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base
        tokenizer = load_tokenizer(str(tok_src))
        base_m = load_model(
            base,
            dtype=dtype,
            attn_implementation=attn_implementation,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=False,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_m, str(adapter_dir), is_trainable=False)
        logger.info("Loaded LoRA adapter %s on base %s", adapter_dir, base)
        if merge:
            model = model.merge_and_unload()
            logger.info("Merged LoRA weights into the base model")
    else:
        weights = layout["weights"]
        tokenizer = load_tokenizer(str(weights))
        model = load_model(
            weights,
            dtype=dtype,
            attn_implementation=attn_implementation,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=False,
            device_map=device_map,
        )
        if merge:
            logger.info("merge=True ignored: %s has no adapter", weights)

    _set_config_flag(model.config if hasattr(model, "config") else None, "use_cache", False)
    model.eval()

    calibration: dict | None = None
    if run_dir is not None:
        calib_path = run_dir / "calibration.json"
        if calib_path.exists():
            with calib_path.open("r", encoding="utf-8") as fh:
                calibration = json.load(fh)
            logger.info("Loaded calibration from %s", calib_path)
    return model, tokenizer, calibration


__all__ = [
    "SIZES",
    "DEFAULT_LORA_TARGETS",
    "to_torch_dtype",
    "normalize_size",
    "resolve_model_id",
    "load_tokenizer",
    "load_model",
    "linear_leaf_names",
    "resolve_lora_targets",
    "apply_lora",
    "count_parameters",
    "restricted_menu_logits",
    "resolve_run_layout",
    "load_for_inference",
]
