#!/usr/bin/env python
"""One-click Qwen3.5-JEV training entry point.

Flow (main process)::

    resolve config -> export HF_HOME / CUDA_VISIBLE_DEVICES -> run dir + config.resolved.yaml
    -> tokenizer + LabelSpace -> prepare data (cached)            [--prepare-only stops here]
    -> (>1 visible GPU: re-exec under torchrun, else in-process) train_worker
    -> adapter saved under <run>/adapter (or <run>/model, optional <run>/merged)
    -> temperature calibration on the calibration half of val    -> <run>/calibration.json
    -> final evaluation on the held-out half (raw vs calibrated)  -> <run>/eval_val/metrics.json
    -> <run>/summary.json

``--dry-run`` never touches the data or the model: it prints the resolved
configuration, the base model name, the discovered source files (with sizes
and row estimates), the effective batch size and exits.

``--full`` is exactly ``--set data.mode=full``.  ``--debug``
switches to the 0.8B Base model with tiny caps so the whole pipeline finishes
in minutes on a single GPU.

Examples::

    python start_training.py --model 4B               # SearchJev-4B with the recipe of config.yaml
    python start_training.py --models 0.8B,4B,9B      # the three paper models, one after another
    python start_training.py --model 9B --gpus 0,1    # 9B on two GPUs (DDP via torchrun)
    python start_training.py --debug                  # smoke test
    python start_training.py --dry-run --model 4B     # inspect the resolved config and data sources
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bootstrap import ensure_environment  # noqa: E402  (stdlib only)

ensure_environment()  # re-exec with ./.venv when this interpreter has no torch

from training.config import (  # noqa: E402  (after sys.path fix; light module, no torch)
    FAMILIES,
    Config,
    ConfigError,
    effective_batch_size,
    family_mode,
    load_config,
    normalize_size,
    prepared_families,
    try_normalize_size,
)

logger = logging.getLogger("jev.start_training")

#: Question-type ids shared with the dataset / trainer.
QTYPE_IDS: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}

#: Maximum number of val examples used for calibration and for the held-out evaluation.
CALIB_CAP = 5000
HELDOUT_CAP = 5000

#: Loggers of third-party libraries that are too chatty at INFO.
_QUIET_LOGGERS = ("urllib3", "filelock", "fsspec", "huggingface_hub", "httpx", "httpcore",
                  "torch.distributed", "datasets", "PIL", "matplotlib")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


#: Backbones trained by a bare ``--models`` (one after another).
SEQUENCE_MODELS: tuple[str, ...] = ("0.8B", "4B", "9B")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (see the module docstring for the flags)."""
    parser = argparse.ArgumentParser(
        prog="start_training.py",
        description="One-click Qwen3.5-JEV training: prepare -> train (LoRA) -> calibrate -> eval -> summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML config (default: config.yaml in the project root)")
    parser.add_argument("--model", default=None, metavar="SIZE|HF_ID|PATH",
                        help="model size (0.8B|2B|4B|9B|27B) or an explicit HF id / local path")
    parser.add_argument("--models", default=None, nargs="?", const=",".join(SEQUENCE_MODELS), metavar="LIST",
                        help="train several backbones one after another, each in its own process and run dir "
                             "(e.g. --models 0.8B,4B,9B; a bare --models means %s)" % ",".join(SEQUENCE_MODELS))
    parser.add_argument("--stop-on-error", action="store_true",
                        help="with --models: stop the sequence at the first model that fails (default: continue)")
    parser.add_argument("--variant", default=None, choices=["instruct", "base"], help="checkpoint variant")
    parser.add_argument("--full", action="store_true",
                        help="train on the full per-dataset files (== --set data.mode=full)")
    parser.add_argument("--families", default=None, metavar="LIST",
                        help="training families, comma-separated subset of " + ",".join(FAMILIES)
                             + " (== --set data.families=...)")
    parser.add_argument("--gpus", default=None, metavar="IDS",
                        help='explicit GPU ids, e.g. "0,1" (sets CUDA_VISIBLE_DEVICES; wins over --num-gpus)')
    parser.add_argument("--num-gpus", dest="num_gpus", default=None, metavar="N|all",
                        help="how many GPUs to use (default: config distributed.num_gpus = all visible); "
                             "N < visible picks the GPUs with the most free memory")
    parser.add_argument("--wandb", default=None, metavar="PROJECT",
                        help="log to Weights & Biases under PROJECT "
                             "(== --set run.report_to=wandb --set run.wandb_project=PROJECT)")
    parser.add_argument("--name", default=None, help="run name (output dir = run.output_root/NAME)")
    parser.add_argument("--set", dest="set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config key, e.g. --set training.learning_rate=1e-4 (repeatable)")
    parser.add_argument("--prepare-only", action="store_true", help="stop after data preparation")
    parser.add_argument("--skip-prepare", action="store_true", help="reuse processed data without checking it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print resolved config, model, source files and batch size; do not process anything")
    parser.add_argument("--debug", action="store_true",
                        help="0.8B-Base, 256 train / 64 eval examples, 20 steps: end-to-end smoke run")
    parser.add_argument("--resume", default=None, metavar="auto|PATH",
                        help="resume training from the latest (auto) or a given checkpoint")
    parser.add_argument("--merge", action="store_true", help="also save merged full weights to <run>/merged")
    parser.add_argument("--_worker", dest="worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def cli_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convenience flags handed to :func:`training.config.load_config`."""
    return {
        "model": args.model,
        "variant": args.variant,
        "full": args.full,
        "families": args.families,
        "gpus": args.gpus,
        "num_gpus": args.num_gpus,
        "wandb": args.wandb,
        "name": args.name,
        "resume": args.resume,
        "debug": args.debug,
    }


# ---------------------------------------------------------------------------
# Logging / environment helpers
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path | None, rank: int = 0) -> None:
    """Configure console (+ optional file) logging with timestamps.

    Non-zero ranks only log warnings to the console so a torchrun launch does
    not repeat every line N times.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    level = logging.INFO if rank == 0 else logging.WARNING
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | %(levelname)-7s | r{rank} | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def export_environment(cfg: Config) -> str:
    """Export ``HF_HOME`` / ``CUDA_VISIBLE_DEVICES`` (must run before importing torch).

    GPU selection (``distributed.num_gpus`` / ``distributed.gpus``) happens only
    in the launcher process; torchrun workers inherit ``CUDA_VISIBLE_DEVICES``.
    Returns a one-line description of the selected GPUs for logging.
    """
    if cfg.run.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(cfg.run.hf_home))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Temp files (DataLoader worker sockets under pymp-*, triton/tilelang caches) go next to the outputs instead of
    # /tmp: a full /tmp makes DataLoader workers fail with ENOSPC and the DDP ranks hang.
    if not os.environ.get("TMPDIR"):
        tmp = Path(str(cfg.run.output_root)).resolve().parent / "tmp"
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(tmp)
        except OSError:
            pass
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        return (f"torchrun worker rank {os.environ.get('RANK')} "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
    from training.gpus import GPUSelectionError, describe, select_gpu_ids

    dist = cfg.distributed
    num_gpus = dist.get("num_gpus", "all")
    if str(dist.get("backend", "auto")).lower() == "none":
        num_gpus = 1  # single process == single GPU (never let Trainer fall back to nn.DataParallel)
    try:
        ids, infos = select_gpu_ids(
            num_gpus,
            dist.get("gpus"),
            selection=str(dist.get("gpu_selection", "free")),
            min_free_gb=dist.get("min_free_gb"),
        )
    except GPUSelectionError as exc:
        raise SystemExit(f"GPU selection failed: {exc}") from exc
    if ids and str(dist.get("backend", "auto")).lower() == "none" and len(ids) > 1:
        ids, infos = ids[:1], infos[:1]
    if ids and any(i < 0 for i in ids):
        return (f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} contains opaque device names; "
                f"kept as-is ({len(ids)} device(s))")
    if ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
        return (f"{len(ids)} GPU(s) selected [num_gpus={dist.get('num_gpus')}, gpus={dist.get('gpus')}, "
                f"selection={dist.get('gpu_selection')}]: {describe(infos)} "
                f"-> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return f"no GPU selected (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})"


def count_visible_gpus() -> int:
    """Number of GPUs this process may use (CUDA_VISIBLE_DEVICES aware)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return len([x for x in env.split(",") if x.strip() != ""])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - torch missing / broken
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
    except Exception:  # pragma: no cover
        return 0


def seed_everything(seed: int) -> None:
    """Seed python/numpy (torch is seeded in the worker via ``transformers.set_seed``)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass


def free_gpu_memory() -> None:
    """Release cached GPU memory between the training and inference phases."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.0f} KB"


def _fmt_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"


# ---------------------------------------------------------------------------
# Source discovery (dry-run; never reads the data beyond a few lines)
# ---------------------------------------------------------------------------


def family_roles(cfg: Config) -> dict[str, str]:
    """``family -> "train"`` (train + val) for ``data.families``, ``"val-only"`` for ``data.val_only_families``."""
    val_only = set(cfg.data.get("val_only_families") or [])
    return {f: ("val-only" if f in val_only else "train") for f in prepared_families(cfg)}


def discover_sources(cfg: Config) -> list[tuple[str, Path]]:
    """Return ``(family, path)`` pairs of the raw source files of every prepared family.

    Covers ``data.families`` and ``data.val_only_families`` (see :func:`family_roles`), each in
    its effective mode (``data.family_modes[family]`` or ``data.mode``).  Uses
    :func:`training.data.list_sources` when available and falls back to the same rule inline:
    ``sampled`` -> ``<root>/<family>/sampled_300k/<family>_300k.jsonl``; ``full`` -> every
    ``<root>/<family>/*.jsonl`` not starting with ``all_``.  A family whose sources are missing
    yields its expected path (``<root>/<family>`` in full mode) so the dry run can flag it.
    """
    root = Path(cfg.data.root)
    try:
        from training.data import list_sources  # type: ignore
    except Exception:  # module not written yet / import error -> inline rule
        list_sources = None  # type: ignore[assignment]
    out: list[tuple[str, Path]] = []
    for family in family_roles(cfg):
        mode = family_mode(cfg, family)
        paths: list[Path]
        if list_sources is not None:
            try:
                paths = [Path(p) for p in list_sources(root, family, mode)]
            except FileNotFoundError:
                paths = [root / family / "sampled_300k" / f"{family}_300k.jsonl" if mode == "sampled" else root / family]
        elif mode == "sampled":
            paths = [root / family / "sampled_300k" / f"{family}_300k.jsonl"]
        else:
            paths = sorted(p for p in (root / family).glob("*.jsonl") if not p.name.startswith("all_")) or [root / family]
        out.extend((family, p) for p in paths)
    return out


def estimate_rows(path: Path, sample_lines: int = 256) -> int | None:
    """Estimate the number of JSONL rows from the file size and the first lines."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return 0
    total = 0
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            total += len(line)
            n += 1
            if n >= sample_lines:
                break
    if n == 0:
        return 0
    if n < sample_lines:
        return n
    return int(round(size / (total / n)))


def processed_layout(cfg: Config) -> dict[str, dict[str, Path]]:
    """Expected processed files of every prepared family (training + val-only, see :func:`family_roles`).

    Always ``<processed_dir>/<data.mode>/<family>/`` - also for a family whose sources are read in
    another mode through ``data.family_modes``.
    """
    base = Path(cfg.data.processed_dir) / str(cfg.data.mode)
    return {
        family: {
            "train": base / family / "train.jsonl",
            "val": base / family / "val.jsonl",
            "stats": base / family / "stats.json",
        }
        for family in family_roles(cfg)
    }


def run_dry_run(cfg: Config, args: argparse.Namespace, gpu_desc: str = "") -> int:
    """Print what a real run would do and return 0."""
    n_gpus = count_visible_gpus()
    world_size = n_gpus if (n_gpus > 1 and cfg.distributed.backend in ("auto", "ddp")) else 1
    print("=" * 78)
    print("DRY RUN - nothing is processed, trained or written")
    print("=" * 78)
    print("\n### Resolved config\n")
    print(cfg.to_yaml().rstrip())
    print("\n### Model\n")
    print(f"base model         : {cfg.run.model_name}")
    print(f"size / variant     : {cfg.model.size} / {cfg.model.variant}")
    print(f"dtype / attn       : {cfg.model.dtype} / {cfg.model.attn_implementation}")
    print(f"LoRA               : {'r=%s alpha=%s' % (cfg.lora.r, cfg.lora.alpha) if cfg.lora.enabled else 'off (full fine-tuning)'}")
    print(f"4-bit (QLoRA)      : {cfg.model.load_in_4bit}")
    print(f"HF_HOME            : {os.environ.get('HF_HOME')}")
    roles = family_roles(cfg)
    overrides = {f: m for f, m in (cfg.data.get("family_modes") or {}).items() if f in roles}
    print("\n### Data sources (data.mode = %s%s; root = %s)\n" % (
        cfg.data.mode, ", per-family: " + ", ".join(f"{f}={m}" for f, m in overrides.items()) if overrides else "",
        cfg.data.root))
    print(f"train families     : {','.join(f for f, r in roles.items() if r == 'train')}  (train + val)")
    print(f"val-only families  : {','.join(f for f, r in roles.items() if r == 'val-only') or '-'}  (val only; never trained on)")
    print()
    total_rows = 0
    total_bytes = 0
    n_files: dict[str, int] = {}
    print(f"{'family':<15} {'role / mode':<16} {'file':<58} {'size':>10} {'rows(≈)':>12}")
    for family, path in discover_sources(cfg):
        tag = f"{roles.get(family, '?')}/{family_mode(cfg, family)}"
        if not path.exists() or path.is_dir():
            print(f"{family:<15} {tag:<16} {str(path):<58} {'MISSING':>10} {'-':>12}")
            continue
        n_files[family] = n_files.get(family, 0) + 1
        size = path.stat().st_size
        rows = estimate_rows(path)
        total_rows += rows or 0
        total_bytes += size
        try:
            shown = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            shown = str(path)
        if len(shown) > 58:
            shown = "…" + shown[-57:]
        print(f"{family:<15} {tag:<16} {shown:<58} {_fmt_bytes(size):>10} {rows if rows is not None else '-':>12}")
    print(f"{'TOTAL':<15} {'':<16} {'':<58} {_fmt_bytes(total_bytes):>10} {total_rows:>12,}")
    print("source files       : " + ", ".join(f"{f}={n_files.get(f, 0)}" for f in roles))
    print(f"\ntrain splits: {list(cfg.data.train_splits)}   val split: {cfg.data.val_split}   (split=test is never used)")
    print(f"caps: max_train_per_dataset={cfg.data.max_train_per_dataset}  max_val_per_dataset={cfg.data.max_val_per_dataset}"
          f"  val_fallback_per_dataset={cfg.data.val_fallback_per_dataset} (sampled mode only)")
    print(f"\n### Processed cache ({Path(cfg.data.processed_dir) / str(cfg.data.mode)}/<family>)\n")
    try:
        from training.data import cache_status

        status = cache_status(cfg)
    except Exception as exc:  # pragma: no cover - dry run must not fail on the cache check
        logger.warning("could not check the processed cache: %s", exc)
        status = {}
    for family, files in processed_layout(cfg).items():
        info = status.get(family) or {}
        state = info.get("state") or ("cached" if files["train"].exists() else "to build")
        extra = ""
        if files["train"].exists():
            extra = f"  train={_fmt_bytes(files['train'].stat().st_size)}"
            if files["val"].exists():
                extra += f"  val={_fmt_bytes(files['val'].stat().st_size)}"
        if info.get("diff"):
            extra += f"  (key changed: {','.join(info['diff'])})"
        if info.get("error"):
            extra += f"  ({info['error']})"
        print(f"{family:<15} {roles.get(family, '?'):<9} {state:<11} {files['train'].parent}{extra}")
    if cfg.data.rebuild:
        print("data.rebuild=true -> caches will be rebuilt")
    print("\n### Training\n")
    print(f"visible GPUs       : {n_gpus} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
    print(f"GPU selection      : {gpu_desc}")
    print(f"distributed        : backend={cfg.distributed.backend} -> world_size={world_size}"
          + ("  (torchrun DDP)" if world_size > 1 else "  (single process)"))
    print(f"per-device batch   : {cfg.training.per_device_train_batch_size} x accum {cfg.training.gradient_accumulation_steps}")
    print(f"effective batch    : {effective_batch_size(cfg, world_size)}")
    print(f"epochs / max_steps : {cfg.training.epochs} / {cfg.training.max_steps}")
    print(f"lr / sched / warmup: {cfg.training.learning_rate} / {cfg.training.lr_scheduler_type} / {cfg.training.warmup_ratio}")
    print(f"loss               : {cfg.loss.type} + {cfg.loss.lambda_brier} * brier; sample weights: "
          f"{dict(cfg.loss.sample_weights) if cfg.loss.get('sample_weights') else 'none'}")
    print(f"batching           : group_by_length={bool(cfg.training.get('group_by_length'))}  "
          f"dataloader_num_workers={cfg.training.dataloader_num_workers}  "
          f"gradient_checkpointing={bool(cfg.model.gradient_checkpointing)}")
    print(f"data hygiene       : exclude_test_overlap={bool(cfg.data.get('exclude_test_overlap'))}  "
          f"dedup_val_against_train={bool(cfg.data.dedup_val_against_train)}")
    print(f"calibration        : enabled={cfg.calibration.enabled} fraction={cfg.calibration.fraction}"
          f" per_question_type={cfg.calibration.per_question_type}")
    print(f"resume             : {cfg.training.resume_from_checkpoint}")
    print(f"merge weights      : {args.merge}")
    print(f"\nrun dir            : {cfg.run.output_dir}")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def write_label_space(label_space: Any, cfg: Config, path: Path) -> None:
    """Persist the label alphabet used by this run (``label_space.json``)."""
    payload = {
        "alphabet": cfg.labels.alphabet,
        "labels": list(label_space.labels),
        "token_ids": [int(t) for t in label_space.token_ids],
        "max_options": int(label_space.max_options),
        "over_limit_policy": cfg.labels.over_limit_policy,
        "tokenizer": cfg.run.model_name,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("label space: %d labels, max_options=%d -> %s", len(payload["labels"]), payload["max_options"], path)


def locate_prepared(cfg: Config) -> Any:
    """Rebuild a ``PreparedData`` from the processed directory without re-processing."""
    from training.data import PreparedData

    train_files: list[Path] = []
    val_files: list[Path] = []
    stats: dict[str, Any] = {}
    roles = family_roles(cfg)
    for family, files in processed_layout(cfg).items():
        if roles[family] == "train" and files["train"].exists():
            train_files.append(files["train"])
        if files["val"].exists():
            val_files.append(files["val"])
        if files["stats"].exists():
            try:
                stats[family] = json.loads(files["stats"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("could not read %s", files["stats"])
    if not train_files:
        raise FileNotFoundError(
            f"no processed training data under {Path(cfg.data.processed_dir) / str(cfg.data.mode)}; "
            "run once without --skip-prepare"
        )
    logger.info("using processed data: %d train file(s), %d val file(s)", len(train_files), len(val_files))
    return PreparedData(train_files=train_files, val_files=val_files, stats=stats,
                        families=[f for f, r in roles.items() if r == "train"],
                        val_only_families=[f for f, r in roles.items() if r == "val-only"])


def _find_int(node: Any, *keys: str) -> int | None:
    """Best-effort lookup of a count in a stats dict (``keys`` are tried in order, one level deep)."""
    if not isinstance(node, dict):
        return None
    for key in keys:
        value = node.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            for sub in ("total", "n", "count", "examples"):
                if isinstance(value.get(sub), int):
                    return value[sub]
    return None


def print_prepare_stats(stats: Any) -> None:
    """Print a compact table of the preparation statistics (format-tolerant)."""
    print("\n### Prepared data\n")
    if not isinstance(stats, dict) or not stats:
        print(json.dumps(stats, ensure_ascii=False)[:2000])
        return
    families = stats.get("families") if isinstance(stats.get("families"), dict) else stats
    val_only = set(stats.get("val_only_families") or [])
    rows: list[tuple[str, str, str, str]] = []
    for family, node in families.items():
        if not isinstance(node, dict):
            continue
        n_rows = _find_int(node, "rows_read", "rows", "n_rows")
        n_train = _find_int(node, "train", "n_train", "train_examples")
        n_val = _find_int(node, "val", "n_val", "val_examples")
        if n_train is None and isinstance(node.get("examples"), dict):
            n_train = _find_int(node["examples"], "train")
            n_val = _find_int(node["examples"], "val")
        if family in val_only:
            n_train = None  # prepared (cached) but never trained on
        rows.append((str(family) + (" (val only)" if family in val_only else ""), str(n_rows if n_rows is not None else "-"),
                     str(n_train if n_train is not None else "-"), str(n_val if n_val is not None else "-")))
    if rows:
        print(f"{'family':<27} {'rows read':>12} {'train ex.':>12} {'val ex.':>10}")
        for r in rows:
            print(f"{r[0]:<27} {r[1]:>12} {r[2]:>12} {r[3]:>10}")
        if isinstance(stats.get("n_train"), int):
            print(f"{'total':<27} {'':>12} {stats['n_train']:>12} {stats.get('n_val', '-'):>10}")
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=1)[:4000])
    print()


# ---------------------------------------------------------------------------
# Distributed launch
# ---------------------------------------------------------------------------


def relaunch_with_torchrun(argv: Sequence[str], n_gpus: int) -> int:
    """Re-exec this script under ``torchrun`` with the same arguments and wait."""
    passthrough = [a for a in argv if a not in ("--dry-run", "--prepare-only", "--skip-prepare", "--_worker")]
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={n_gpus}",
        str(Path(__file__).resolve()), *passthrough, "--skip-prepare", "--_worker",
    ]
    logger.info("launching DDP on %d GPUs: %s", n_gpus, " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    # start_new_session: torchrun installs its own SIGHUP handler (overriding nohup), so a
    # closing terminal / IDE session would kill the whole DDP job. A new session has no
    # controlling terminal and never receives that hangup. SIGINT/SIGTERM sent to this
    # launcher are forwarded to the torchrun process group below.
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, start_new_session=True)
    _CHILD_GROUPS.add(proc.pid)
    try:
        rc = proc.wait()
    finally:
        _CHILD_GROUPS.discard(proc.pid)
    if rc != 0:
        logger.error("torchrun exited with code %s", rc)
    return int(rc)


#: Process-group ids of detached children (torchrun) to forward termination signals to.
_CHILD_GROUPS: set[int] = set()


def install_signal_handlers() -> None:
    """Launcher: ignore SIGHUP (terminal hang-up); forward SIGINT/SIGTERM to detached children."""
    import signal

    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def _forward(signum: int, _frame: Any) -> None:
        for pgid in list(_CHILD_GROUPS):
            try:
                os.killpg(pgid, signum)
            except OSError:
                pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------


def _resolve_resume(value: Any, checkpoint_dir: Path) -> str | None:
    """Translate ``training.resume_from_checkpoint`` into a path (or ``None``)."""
    if value in (None, False, "", "null", "none", "None"):
        return None
    if value is True or str(value).lower() == "auto":
        if not checkpoint_dir.exists():
            logger.info("resume=auto but %s does not exist yet; starting fresh", checkpoint_dir)
            return None
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(str(checkpoint_dir))
        if last is None:
            logger.info("resume=auto: no checkpoint under %s; starting fresh", checkpoint_dir)
        else:
            logger.info("resuming from %s", last)
        return last
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    logger.info("resuming from %s", path)
    return str(path)


def _save_merged(model: Any, tokenizer: Any, out_dir: Path, four_bit: bool) -> bool:
    """Merge LoRA into the base weights and save to ``out_dir``; returns success."""
    if four_bit:
        logger.warning("--merge skipped: merging into 4-bit (bitsandbytes) weights is not supported")
        return False
    if not hasattr(model, "merge_and_unload"):
        logger.warning("--merge skipped: model has no LoRA adapter to merge")
        return False
    logger.info("merging LoRA weights into the base model -> %s", out_dir)
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    return True


def train_worker(cfg: Config, *, prepared: Any = None, merge: bool = False) -> dict[str, Any]:
    """Build everything and run the Trainer (one process; rank-aware under torchrun).

    Returns the training metrics dict (empty on non-zero ranks).  Rank 0 writes
    ``<run>/adapter`` (or ``<run>/model``), ``<run>/train_metrics.json`` and,
    with ``merge``, ``<run>/merged``.
    """
    import torch
    from transformers import set_seed

    from training.collator import JEVDataCollator
    from training.data import build_train_dataset, build_val_dataset
    from training.labels import LabelSpace
    from training.modeling import apply_lora, count_parameters, load_model, load_tokenizer
    from training.prompting import PromptBuilder
    from training.trainer import JEVTrainer, build_training_arguments, compute_metrics_fn

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_dir = Path(cfg.run.output_dir)
    model_name = str(cfg.run.model_name)
    set_seed(int(cfg.run.seed))

    if prepared is None:
        prepared = locate_prepared(cfg)

    logger.info("train_worker: rank %d/%d, model %s, run dir %s", rank, world_size, model_name, run_dir)
    tokenizer = load_tokenizer(model_name)
    label_space = LabelSpace(tokenizer, alphabet=cfg.labels.alphabet, max_options=cfg.labels.max_options)
    prompt_builder = PromptBuilder(
        tokenizer, label_space, style=cfg.prompt.style, language=cfg.prompt.language,
        max_length=int(cfg.prompt.max_length),
    )
    train_ds = build_train_dataset(cfg, prepared, prompt_builder, label_space)
    n_train = len(train_ds)
    if n_train == 0:
        raise RuntimeError("the training set is empty; check data.families / include_datasets / caps")
    val_ds = build_val_dataset(cfg, prepared, prompt_builder, label_space) if prepared.val_files else None
    n_val = len(val_ds) if val_ds is not None else 0
    if n_val == 0:
        val_ds = None
        if cfg.training.eval_strategy != "no":
            logger.warning("no validation examples -> in-training evaluation disabled")
            cfg.training.eval_strategy = "no"
    logger.info("datasets: %d train examples (x%d permutation copies incl.), %d val examples",
                n_train, int(cfg.data.permutation.copies), n_val)

    # pad_to_multiple_of=128: see JEVDataCollator.SAFE_PAD_MULTIPLE (fla backward NaN at L % 128 == 64)
    collator = JEVDataCollator(tokenizer, pad_to_multiple_of=JEVDataCollator.SAFE_PAD_MULTIPLE)
    device_map = {"": local_rank} if cfg.model.load_in_4bit else None
    model = load_model(
        model_name,
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
        load_in_4bit=bool(cfg.model.load_in_4bit),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        device_map=device_map,
    )
    if cfg.lora.enabled:
        init_adapter = cfg.lora.get("init_adapter")
        if init_adapter:
            from training.modeling import load_lora_for_training

            model = load_lora_for_training(model, str(init_adapter))
        else:
            model = apply_lora(model, cfg.lora)
    params = count_parameters(model)
    logger.info("parameters: %s", params)

    checkpoint_dir = run_dir / "checkpoints"
    targs = build_training_arguments(cfg, str(run_dir), n_train, world_size)  # checkpoints -> <run>/checkpoints
    trainer = JEVTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn(label_space) if val_ds is not None else None,
        label_space=label_space,
        lambda_brier=float(cfg.loss.lambda_brier),
        loss_type=str(cfg.loss.type),
        group_by_length=bool(cfg.training.get("group_by_length", False)),
        max_length=int(cfg.prompt.max_length),
    )
    resume = _resolve_resume(cfg.training.resume_from_checkpoint, checkpoint_dir)

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    peak_gb = float(torch.cuda.max_memory_allocated() / (1 << 30)) if torch.cuda.is_available() else None
    peak_reserved_gb = float(torch.cuda.max_memory_reserved() / (1 << 30)) if torch.cuda.is_available() else None
    samples_per_sec = float(getattr(result, "metrics", {}).get("train_samples_per_second", float("nan")))
    logger.info("training finished in %s (%d steps): %.2f samples/s, peak GPU memory %s GB allocated / %s GB reserved",
                _fmt_seconds(elapsed), trainer.state.global_step, samples_per_sec,
                f"{peak_gb:.2f}" if peak_gb is not None else "n/a",
                f"{peak_reserved_gb:.2f}" if peak_reserved_gb is not None else "n/a")

    save_dir = run_dir / ("adapter" if cfg.lora.enabled else "model")
    trainer.save_model(str(save_dir))  # rank-aware: only the main process writes
    metrics: dict[str, Any] = {}
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(save_dir))
        metrics = {
            "model": model_name,
            "world_size": world_size,
            "n_train_examples": n_train,
            "n_val_examples": n_val,
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch or 0.0),
            "train_loss": float(getattr(result, "training_loss", float("nan"))),
            "train_runtime_sec": elapsed,
            "effective_batch_size": effective_batch_size(cfg, world_size),
            "train_samples_per_second": samples_per_sec,
            "peak_gpu_memory_allocated_gb": peak_gb,
            "peak_gpu_memory_reserved_gb": peak_reserved_gb,
            "parameters": params,
            "resumed_from": resume,
            "saved_to": str(save_dir),
            "log_history": list(trainer.state.log_history)[-200:],
        }
        if isinstance(getattr(result, "metrics", None), dict):
            metrics["trainer_metrics"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                          for k, v in result.metrics.items()}
        if merge and cfg.lora.enabled:
            metrics["merged"] = _save_merged(trainer.model, tokenizer, run_dir / "merged", bool(cfg.model.load_in_4bit))
        (run_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("saved %s and train_metrics.json", save_dir)

    # Close the trainer's wandb run explicitly so a later launch can resume it right away
    # (otherwise the server may still report it "in use").
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:  # pragma: no cover - wandb not installed / not used
        pass

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    del trainer, model
    free_gpu_memory()
    return metrics


# ---------------------------------------------------------------------------
# Validation split, calibration and final evaluation
# ---------------------------------------------------------------------------


def _id_unit(example_id: str, seed: int) -> float:
    """Uniform number in [0, 1) for an example id; identical to ``training.data.calibration_unit``."""
    from training.data import calibration_unit

    return calibration_unit(example_id, seed)


def hash_split_ids(ids: Iterable[str], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministically split ids into (calibration, held-out); same rule as ``split_val_for_calibration``."""
    calib: set[str] = set()
    held: set[str] = set()
    for example_id in ids:
        (calib if _id_unit(example_id, seed) < fraction else held).add(example_id)
    return calib, held


def load_val_examples(val_files: Sequence[Path], fraction: float, seed: int,
                      calib_cap: int = CALIB_CAP, heldout_cap: int = HELDOUT_CAP) -> tuple[list[Any], list[Any]]:
    """Load the calibration and held-out subsets of the processed val files.

    Pass 1 records ``(hash, file, offset)`` per example; the ``calib_cap`` /
    ``heldout_cap`` smallest hashes of each side are then read back in pass 2, so
    the selection is deterministic and memory stays small for huge val sets.
    """
    from training.schema import DecisionExample

    from training.data import calibration_unit

    calib_keys: list[tuple[float, int, int]] = []
    held_keys: list[tuple[float, int, int]] = []
    for file_idx, path in enumerate(val_files):
        with open(path, "rb") as fh:
            offset = 0
            for line in fh:
                length = len(line)
                stripped = line.strip()
                if stripped:
                    try:
                        example_id = str(json.loads(stripped)["id"])
                    except (ValueError, KeyError, TypeError):
                        offset += length
                        continue
                    u = calibration_unit(example_id, seed)
                    (calib_keys if u < fraction else held_keys).append((u, file_idx, offset))
                offset += length
    calib_keys.sort()
    held_keys.sort()
    selected = {"calib": calib_keys[:calib_cap], "held": held_keys[:heldout_cap]}
    logger.info("val examples: %d calibration (using %d), %d held-out (using %d)",
                len(calib_keys), len(selected["calib"]), len(held_keys), len(selected["held"]))

    out: dict[str, list[Any]] = {"calib": [], "held": []}
    for side, keys in selected.items():
        by_file: dict[int, list[int]] = {}
        for _, file_idx, offset in keys:
            by_file.setdefault(file_idx, []).append(offset)
        for file_idx, offsets in by_file.items():
            with open(val_files[file_idx], "rb") as fh:
                for offset in sorted(offsets):
                    fh.seek(offset)
                    row = json.loads(fh.readline())
                    out[side].append(DecisionExample.from_dict(row))
    return out["calib"], out["held"]


def collect_menu_logits(predictor: Any, examples: Sequence[Any], chunk: int = 1000) -> dict[str, Any]:
    """Run the predictor and pack raw canonical-order menu logits into padded arrays."""
    import numpy as np

    logits_rows: list[list[float]] = []
    for start in range(0, len(examples), chunk):
        batch = examples[start:start + chunk]
        logits_rows.extend(predictor.raw_menu_logits(batch))
        logger.info("  logits %d/%d", min(start + chunk, len(examples)), len(examples))
    kmax = max((len(r) for r in logits_rows), default=1)
    n = len(examples)
    logits = np.full((n, kmax), -np.inf, dtype=np.float32)
    target_idx = np.zeros(n, dtype=np.int64)
    qtype_id = np.zeros(n, dtype=np.int64)
    num_options = np.zeros(n, dtype=np.int64)
    datasets: list[str] = []
    for i, (ex, row) in enumerate(zip(examples, logits_rows)):
        k = len(ex.options)
        if len(row) != k:
            raise RuntimeError(f"predictor returned {len(row)} logits for {k} options (example {ex.id})")
        logits[i, :k] = np.asarray(row, dtype=np.float32)
        target_idx[i] = int(ex.target_idx if ex.target_idx is not None else 0)
        qtype_id[i] = QTYPE_IDS.get(ex.qtype, 0)
        num_options[i] = k
        datasets.append(ex.dataset)
    return {"logits": logits, "target_idx": target_idx, "qtype_id": qtype_id,
            "num_options": num_options, "datasets": datasets}


def softmax_with_temperature(logits: Any, temps: Any) -> Any:
    """Row-wise softmax of ``logits / T`` (``-inf`` entries become exactly 0)."""
    import numpy as np

    z = logits.astype(np.float64) / np.asarray(temps, dtype=np.float64).reshape(-1, 1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)


def run_calibration(cfg: Config, run_dir: Path, predictor: Any, calib_examples: Sequence[Any]) -> dict[str, Any] | None:
    """Fit temperature(s) on the calibration half and write ``calibration.json``."""
    import torch

    from training.calibration import fit_calibration, save_calibration

    if not calib_examples:
        logger.warning("calibration skipped: no calibration examples")
        return None
    logger.info("calibration: collecting menu logits on %d examples", len(calib_examples))
    packed = collect_menu_logits(predictor, calib_examples)
    calib = fit_calibration(
        torch.from_numpy(packed["logits"]),
        torch.from_numpy(packed["target_idx"]),
        torch.from_numpy(packed["qtype_id"]),
        per_question_type=bool(cfg.calibration.per_question_type),
    )
    calib = dict(calib)
    calib.setdefault("n_examples", len(calib_examples))
    calib.setdefault("fraction", float(cfg.calibration.fraction))
    path = run_dir / "calibration.json"
    save_calibration(calib, path)
    logger.info("calibration written to %s: %s", path,
                {k: v for k, v in calib.items() if isinstance(v, (int, float))})
    return calib


def run_final_eval(cfg: Config, run_dir: Path, predictor: Any, heldout: Sequence[Any],
                   calib: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate on the held-out val half with and without temperature; write metrics.json."""
    import numpy as np

    from training import metrics as jev_metrics

    out_dir = run_dir / "eval_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not heldout:
        logger.warning("final evaluation skipped: no held-out examples")
        report = {"n": 0, "before_temperature": None, "after_temperature": None, "temperatures": None}
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    logger.info("final eval: collecting menu logits on %d held-out examples", len(heldout))
    packed = collect_menu_logits(predictor, heldout)
    ones = np.ones(len(heldout), dtype=np.float64)
    before = jev_metrics.summarize(
        softmax_with_temperature(packed["logits"], ones), packed["target_idx"], packed["qtype_id"],
        packed["num_options"], datasets=packed["datasets"],
    )
    report: dict[str, Any] = {"n": len(heldout), "before_temperature": before, "after_temperature": None,
                              "temperatures": None}
    if calib is not None:
        from training.calibration import temperature_for

        qtypes = {v: k for k, v in QTYPE_IDS.items()}
        temps = np.array([temperature_for(calib, qtypes.get(int(q), "choice")) for q in packed["qtype_id"]],
                         dtype=np.float64)
        after = jev_metrics.summarize(
            softmax_with_temperature(packed["logits"], temps), packed["target_idx"], packed["qtype_id"],
            packed["num_options"], datasets=packed["datasets"],
        )
        report["after_temperature"] = after
        report["temperatures"] = {q: temperature_for(calib, q) for q in QTYPE_IDS}
        report["temperatures"]["global"] = temperature_for(calib, "global")
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
                                          encoding="utf-8")
    logger.info("held-out metrics written to %s", out_dir / "metrics.json")
    return report


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _overall(block: Any) -> dict[str, Any]:
    """Pick the overall numbers out of a ``metrics.summarize`` result (format tolerant)."""
    if not isinstance(block, dict):
        return {}
    node = block.get("overall", block)
    keys = ("n", "accuracy", "nll", "brier", "ece", "mean_confidence")
    return {k: node[k] for k in keys if k in node}


def write_summary(cfg: Config, run_dir: Path, *, train_metrics: dict[str, Any], eval_report: dict[str, Any],
                  calib: dict[str, Any] | None, n_calib: int, n_heldout: int, world_size: int,
                  t_start: float, merge: bool) -> dict[str, Any]:
    """Write ``<run>/summary.json`` and print a short human summary."""
    summary = {
        "run_name": cfg.run.name,
        "run_dir": str(run_dir),
        "model": cfg.run.model_name,
        "model_size": cfg.model.size,
        "variant": cfg.model.variant,
        "lora": bool(cfg.lora.enabled),
        "load_in_4bit": bool(cfg.model.load_in_4bit),
        "data_mode": cfg.data.mode,
        "data_root": str(cfg.data.root),
        "families": list(cfg.data.families),
        "val_only_families": list(cfg.data.get("val_only_families") or []),
        "family_modes": {f: family_mode(cfg, f) for f in prepared_families(cfg)},
        "n_train_examples": train_metrics.get("n_train_examples"),
        "n_val_examples": train_metrics.get("n_val_examples"),
        "n_calibration_examples": n_calib,
        "n_heldout_examples": n_heldout,
        "global_step": train_metrics.get("global_step"),
        "epoch": train_metrics.get("epoch"),
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime_sec": train_metrics.get("train_runtime_sec"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "peak_gpu_memory_allocated_gb": train_metrics.get("peak_gpu_memory_allocated_gb"),
        "total_runtime_sec": time.time() - t_start,
        "effective_batch_size": effective_batch_size(cfg, world_size),
        "world_size": world_size,
        "temperatures": eval_report.get("temperatures"),
        "calibration": {k: v for k, v in (calib or {}).items() if isinstance(v, (int, float, str))} or None,
        "final_metrics": {
            "before_temperature": _overall(eval_report.get("before_temperature")),
            "after_temperature": _overall(eval_report.get("after_temperature")),
        },
        "artifacts": {
            "weights": str(run_dir / ("adapter" if cfg.lora.enabled else "model")),
            "merged": str(run_dir / "merged") if merge and train_metrics.get("merged") else None,
            "config": str(run_dir / "config.resolved.yaml"),
            "label_space": str(run_dir / "label_space.json"),
            "calibration": str(run_dir / "calibration.json") if calib is not None else None,
            "eval_val": str(run_dir / "eval_val" / "metrics.json"),
            "train_log": str(run_dir / "train.log"),
        },
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
                                          encoding="utf-8")
    before = summary["final_metrics"]["before_temperature"]
    after = summary["final_metrics"]["after_temperature"]

    def _fmt(block: dict[str, Any]) -> str:
        if not block:
            return "n/a"
        return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in block.items())

    print("\n" + "=" * 78)
    print(f"JEV run finished: {cfg.run.name}")
    print("=" * 78)
    print(f"model            : {summary['model']}  (LoRA={'on' if summary['lora'] else 'off'})")
    print(f"data             : mode={summary['data_mode']} families={','.join(summary['families'])}"
          + (f" val-only={','.join(summary['val_only_families'])}" if summary["val_only_families"] else ""))
    print(f"examples         : train={summary['n_train_examples']} val={summary['n_val_examples']} "
          f"calib={n_calib} held-out={n_heldout}")
    print(f"steps / loss     : {summary['global_step']} / {summary['train_loss']}")
    print(f"train time       : {_fmt_seconds(summary['train_runtime_sec'] or 0)}  "
          f"(total {_fmt_seconds(summary['total_runtime_sec'])})")
    print(f"temperatures     : {summary['temperatures']}")
    print(f"held-out (T=1)   : {_fmt(before)}")
    print(f"held-out (T*)    : {_fmt(after)}")
    print(f"weights          : {summary['artifacts']['weights']}")
    print(f"summary          : {run_dir / 'summary.json'}")
    print("=" * 78)
    return summary


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------


def wandb_enabled(cfg: Config) -> bool:
    """True when ``run.report_to`` contains ``wandb``."""
    report_to = cfg.run.get("report_to") or "none"
    items = report_to if isinstance(report_to, (list, tuple)) else str(report_to).split(",")
    return "wandb" in [str(x).strip() for x in items]


def export_wandb_env(cfg: Config) -> None:
    """Project / entity / group for every wandb run of this training run (trainer workers inherit it)."""
    if not wandb_enabled(cfg):
        return
    os.environ.setdefault("WANDB_PROJECT", str(cfg.run.get("wandb_project") or "SearchJev"))
    if cfg.run.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(cfg.run.wandb_entity))
    os.environ.setdefault("WANDB_RUN_GROUP", str(cfg.run.name))


_WANDB_ID_FILE = "wandb_run_id.txt"


def training_wandb_run_id(cfg: Config, run_dir: Path, *, create: bool) -> str | None:
    """Id of this run's wandb training run: ``<run>/wandb_run_id.txt``, or a new id (``create=True`` only)."""
    id_path = run_dir / _WANDB_ID_FILE
    if id_path.exists():
        rid = id_path.read_text(encoding="utf-8").strip()
        if rid:
            return rid
    if not create:
        return None
    rid = __import__("uuid").uuid4().hex[:8]
    run_dir.mkdir(parents=True, exist_ok=True)
    id_path.write_text(rid, encoding="utf-8")
    return rid


def bind_training_wandb_run(cfg: Config, run_dir: Path) -> str | None:
    """Fix the training run's wandb id before anything is logged; all ranks inherit ``WANDB_RUN_ID`` +
    ``WANDB_RESUME=allow``, and a ``--resume`` launch continues the same wandb run."""
    if not wandb_enabled(cfg):
        return None
    rid = training_wandb_run_id(cfg, run_dir, create=True)
    os.environ["WANDB_RUN_ID"] = str(rid)
    os.environ["WANDB_RESUME"] = "allow"
    return rid


def start_wandb_run_early(cfg: Config, run_dir: Path, status: str) -> None:
    """Create / resume the training wandb run right away so it is visible (with its config) from the start.
    Never raises."""
    if not wandb_enabled(cfg):
        return
    try:
        import wandb

        run = wandb.init(project=str(cfg.run.get("wandb_project") or "SearchJev"),
                         entity=str(cfg.run.wandb_entity) if cfg.run.get("wandb_entity") else None,
                         id=training_wandb_run_id(cfg, run_dir, create=True), resume="allow", name=str(cfg.run.name),
                         group=str(cfg.run.name), dir=str(run_dir), config={"jev": cfg.to_dict()})
        run.define_metric("train/global_step")
        run.summary["status"] = status
        logger.info("wandb run ready: %s (status: %s)", run.url, status)
        run.finish()
    except Exception as exc:  # pragma: no cover - never break training because of logging
        logger.warning("could not create the wandb run early: %s", exc)


# ---------------------------------------------------------------------------
# --models: several backbones in sequence
# ---------------------------------------------------------------------------


def _strip_sequence_flags(argv: Sequence[str]) -> list[str]:
    """``argv`` without ``--models [LIST]``, ``--stop-on-error`` and ``--name X`` (the sequence sets them per model)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--models", "--name"):
            i += 1
            if i < len(argv) and (tok == "--name" or not argv[i].startswith("-")):
                i += 1
            continue
        if tok.startswith(("--models=", "--name=")) or tok == "--stop-on-error":
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def run_model_sequence(argv: Sequence[str], args: argparse.Namespace, models: Sequence[str]) -> int:
    """Run the whole pipeline (prepare -> train -> calibrate) for each backbone in turn, one child process each.

    Every model gets its own run dir: ``run.name`` with ``{size}`` substituted (config.yaml default
    ``qwen35-{size}-jev``), or ``<--name>-<size>`` when the names would otherwise collide.
    """
    overrides = list(args.set)
    names: list[str] = []
    for m in models:
        cli = cli_dict(args)
        cli["model"] = m
        try:
            names.append(str(load_config(args.config, overrides=overrides, cli=cli).run.name))
        except ConfigError as exc:
            print(f"config error for --models entry {m!r}: {exc}", file=sys.stderr)
            return 2
    if len(set(names)) < len(names):
        names = [f"{n}-{normalize_size(m).lower() if try_normalize_size(m) else Path(m).name}" for n, m in zip(names, models)]
    base = _strip_sequence_flags(argv)
    script = str(Path(__file__).resolve())
    results: list[dict[str, Any]] = []
    t_all = time.time()
    print(f"[models] training {len(models)} backbone(s) in sequence: " + ", ".join(f"{m} -> {n}" for m, n in zip(models, names)),
          flush=True)
    for m, n in zip(models, names):
        cmd = [sys.executable, script, *base, "--model", m, "--name", n]
        print(f"[models] ==== {m} ({n}) ====\n[models] $ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        try:
            rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        except KeyboardInterrupt:
            print("[models] interrupted; stopping the sequence", flush=True)
            results.append({"model": m, "run": n, "rc": 130, "seconds": round(time.time() - t0)})
            break
        results.append({"model": m, "run": n, "rc": rc, "seconds": round(time.time() - t0)})
        print(f"[models] {m} finished with exit code {rc} after {_fmt_seconds(time.time() - t0)}", flush=True)
        if rc != 0 and args.stop_on_error:
            print("[models] --stop-on-error: stopping the sequence", flush=True)
            break
    print("[models] summary (total %s):" % _fmt_seconds(time.time() - t_all), flush=True)
    for r in results:
        print(f"[models]   {r['model']:<6} {r['run']:<28} rc={r['rc']}  {_fmt_seconds(r['seconds'])}", flush=True)
    if not args.dry_run:
        out = Path(load_config(args.config, overrides=overrides, cli={**cli_dict(args), "model": models[0]}).run.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"sequence-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"models": list(models), "argv": list(argv), "results": results}, indent=2), encoding="utf-8")
    failed = [r for r in results if r["rc"] != 0]
    return 0 if not failed and len(results) == len(models) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.models and not args.worker:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.model:
            print("use either --model or --models, not both", file=sys.stderr)
            return 2
        if args.resume and len(models) > 1:
            print("--resume with --models: resume each model with --model", file=sys.stderr)
            return 2
        if len(models) > 1:
            return run_model_sequence(argv, args, models)
        args.model = models[0] if models else None
    try:
        overrides = list(args.set)
        cfg = load_config(args.config, overrides=overrides, cli=cli_dict(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    gpu_desc = export_environment(cfg)
    run_dir = Path(cfg.run.output_dir)
    t_start = time.time()

    # ---- torchrun worker ----------------------------------------------------
    launched_by_torchrun = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    if args.worker or launched_by_torchrun:
        rank = int(os.environ.get("RANK", "0"))
        setup_logging(run_dir / "train.log" if rank == 0 else None, rank=rank)
        seed_everything(int(cfg.run.seed))
        train_worker(cfg, merge=args.merge)
        return 0

    install_signal_handlers()  # launcher only (workers keep torch's default handling)
    export_wandb_env(cfg)

    # ---- dry run --------------------------------------------------------------
    if args.dry_run:
        setup_logging(None)
        return run_dry_run(cfg, args, gpu_desc)

    # ---- (1) run dir, logging, resolved config ------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    seed_everything(int(cfg.run.seed))
    logger.info("command: %s", " ".join([sys.executable, *sys.argv]))
    logger.info("run dir: %s", run_dir)
    logger.info("GPUs: %s", gpu_desc)
    logger.info("base model: %s  data mode: %s  families: %s  val-only: %s  source modes: %s", cfg.run.model_name,
                cfg.data.mode, ",".join(cfg.data.families), ",".join(cfg.data.get("val_only_families") or []) or "-",
                {f: family_mode(cfg, f) for f in prepared_families(cfg)})
    cfg.dump(run_dir / "config.resolved.yaml")

    # ---- (2) tokenizer, label space, data preparation -------------------------
    from training.labels import LabelSpace
    from training.modeling import load_tokenizer

    tokenizer = load_tokenizer(cfg.run.model_name)
    label_space = LabelSpace(tokenizer, alphabet=cfg.labels.alphabet, max_options=cfg.labels.max_options)
    write_label_space(label_space, cfg, run_dir / "label_space.json")

    if args.skip_prepare:
        prepared = locate_prepared(cfg)
    else:
        from training.data import prepare_data

        t0 = time.time()
        logger.info("preparing data (mode=%s) ...", cfg.data.mode)
        prepared = prepare_data(cfg, label_space)
        logger.info("data preparation done in %s", _fmt_seconds(time.time() - t0))
        print_prepare_stats(prepared.stats)
        (run_dir / "prepare_stats.json").write_text(
            json.dumps(prepared.stats, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    del tokenizer
    if args.prepare_only:
        logger.info("--prepare-only: stopping after data preparation")
        return 0

    # ---- (2b) wandb run ----------------------------------------------------------
    wandb_id = bind_training_wandb_run(cfg, run_dir)
    if wandb_id:
        logger.info("wandb: training run id %s", wandb_id)
    start_wandb_run_early(cfg, run_dir, "training")

    # ---- (3) training: DDP re-exec or in-process -------------------------------
    n_gpus = count_visible_gpus()
    if n_gpus == 0:
        logger.warning("no GPU visible; training will run on CPU (only sensible with --debug)")
    use_ddp = n_gpus > 1 and cfg.distributed.backend in ("auto", "ddp") and "WORLD_SIZE" not in os.environ
    world_size = n_gpus if use_ddp else 1
    logger.info("visible GPUs: %d -> %s, effective batch size %d", n_gpus,
                f"DDP world_size={world_size}" if use_ddp else "single process",
                effective_batch_size(cfg, world_size))
    if use_ddp:
        rc = relaunch_with_torchrun(sys.argv[1:], n_gpus)
        if rc != 0:
            return rc
    else:
        train_worker(cfg, prepared=prepared, merge=args.merge)
    metrics_path = run_dir / "train_metrics.json"
    if not metrics_path.exists():
        logger.error("training did not produce %s", metrics_path)
        return 1
    train_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    free_gpu_memory()

    # ---- (4)/(5)/(6) calibration + held-out evaluation -----------------------
    calib_examples, heldout = load_val_examples(
        list(prepared.val_files), float(cfg.calibration.fraction), int(cfg.run.seed),
        calib_cap=int(cfg.calibration.max_examples or 10**9),
        heldout_cap=int(cfg.evaluation.max_examples or 10**9))
    calib: dict[str, Any] | None = None
    eval_report: dict[str, Any] = {"n": 0, "before_temperature": None, "after_temperature": None,
                                   "temperatures": None}
    if calib_examples or heldout:
        from training.inference import JEVPredictor

        predictor = JEVPredictor(
            str(run_dir),
            device="cuda" if n_gpus > 0 else "cpu",
            dtype=cfg.model.dtype,
            batch_size=int(cfg.training.per_device_eval_batch_size),
            max_length=int(cfg.prompt.max_length),
            calibration=False,
            label_alphabet=cfg.labels.alphabet,
            prompt_style=cfg.prompt.style,
            prompt_language=cfg.prompt.language,
        )
        if cfg.calibration.enabled:
            calib = run_calibration(cfg, run_dir, predictor, calib_examples)
        else:
            logger.info("calibration disabled by config")
        eval_report = run_final_eval(cfg, run_dir, predictor, heldout, calib)
        del predictor
        free_gpu_memory()
    else:
        logger.warning("no validation data: skipping calibration and final evaluation")

    # ---- (7) summary -----------------------------------------------------------
    write_summary(cfg, run_dir, train_metrics=train_metrics, eval_report=eval_report, calib=calib,
                  n_calib=len(calib_examples), n_heldout=len(heldout), world_size=world_size,
                  t_start=t_start, merge=args.merge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
