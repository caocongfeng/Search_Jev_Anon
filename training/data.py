"""Data pipeline for SearchJev.

This module covers everything between the raw JEV-format jsonl files (SearchJev-V1/V2) and
the tensors consumed by the trainer:

* **source discovery** (:func:`list_sources`, :func:`full_source_files`):
  ``sampled`` mode uses the ``sampled_300k/<family>_300k.jsonl`` subsets,
  ``full`` mode uses every per-dataset ``<family>/*.jsonl`` file.  The mode is
  chosen per family: ``data.family_modes[family]`` if set, else ``data.mode``
  (e.g. ``unified_v2`` has no sampled subset and runs ``full``
  while the SearchJev-V1 families stay ``sampled``).  The processed cache of every
  family stays at ``<processed_dir>/<data.mode>/<family>`` (``data.mode`` = the
  global mode); the per-family source mode is part of the family's cache key.
* **val-only families** (``data.val_only_families``): prepared exactly like the
  training families (same cache), but only their ``val.jsonl`` is used (validation,
  in-training eval, calibration and held-out evaluation); their ``train.jsonl`` is
  never trained on.
* **prepare** (:func:`prepare_data`): a streaming, memory-light conversion of
  JEV rows into :class:`~training.schema.DecisionExample` json lines, split
  into ``train.jsonl`` / ``val.jsonl`` per family.  Rows whose ``split`` is
  ``test`` are never written.  The work is distributed over a
  ``multiprocessing.Pool`` (one task per source file); per-dataset caps are
  enforced with hash-keyed reservoirs so the output is deterministic no
  matter how the pool schedules the files.  Families without any ``val``
  rows in their sources (the ranking 300k subset) fall back to streaming a
  small number of ``val`` rows from the family's full per-dataset files.
* **dataset** (:class:`JEVDataset`): an offset-indexed jsonl reader (index
  cached at ``<path>.idx.npy``) with lazily opened per-process file handles,
  deterministic option permutation and deterministic sub-sampling.
* helpers used by the scripts: :func:`build_train_dataset`,
  :func:`build_val_dataset`, :func:`split_val_for_calibration`,
  :func:`read_examples`, :func:`format_prepare_table`.
"""

from __future__ import annotations

import enum
import hashlib
import heapq
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from .labels import LabelSpace
from .prompting import PromptBuilder
from .schema import DecisionExample, example_weight, jev_row_to_examples, render_state

try:  # the policy type lives in schema.py; string names are accepted as well
    from .schema import OverLimitPolicy  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the schema module's API
    OverLimitPolicy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "QTYPES",
    "QTYPE_IDS",
    "PREPARE_VERSION",
    "PreparedData",
    "JEVDataset",
    "list_sources",
    "full_source_files",
    "prepare_data",
    "build_train_dataset",
    "build_val_dataset",
    "split_val_for_calibration",
    "calibration_unit",
    "read_examples",
    "count_lines",
    "load_or_build_line_index",
    "format_prepare_table",
    "cache_status",
]

#: Question types in canonical order and their integer ids (used by the
#: trainer / metrics to group examples).
QTYPES: tuple[str, ...] = ("choice", "noul", "score")
QTYPE_IDS: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}

#: Bump when the on-disk format of processed files changes; part of the cache key.
PREPARE_VERSION = 2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SPLITS: tuple[str, str] = ("train", "val")
_TEST_SPLIT = "test"
_INDEX_SUFFIX = ".idx.npy"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from an attribute-accessible nested dict (or plain dicts)."""
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            if part not in node:
                return default
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return default
    return default if node is None else node


def _resolve_path(p: str | os.PathLike[str]) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _stable_hash(text: str) -> int:
    """Deterministic 64-bit hash of a string (first 8 bytes of sha1)."""
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def _hash_unit(text: str) -> float:
    """Deterministic uniform number in ``[0, 1)`` derived from ``text``."""
    return _stable_hash(text) / float(1 << 64)


def _as_name_set(value: Any) -> frozenset[str] | None:
    """Normalise ``include/exclude_datasets`` config values to a frozenset."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    else:
        items = [str(v).strip() for v in value]
    items = [v for v in items if v]
    return frozenset(items) if items else None


def _resolve_policy(name: str | None) -> Any:
    """Map the config string to the schema module's ``OverLimitPolicy`` (if it is an Enum)."""
    if name is None:
        return None
    if isinstance(name, enum.Enum):
        return name
    text = str(name).strip().lower()
    if OverLimitPolicy is not None and isinstance(OverLimitPolicy, type) and issubclass(OverLimitPolicy, enum.Enum):
        try:
            return OverLimitPolicy(text)
        except ValueError:
            for member in OverLimitPolicy:  # match by member name as a fallback
                if member.name.lower() == text:
                    return member
            raise ValueError(f"unknown over_limit_policy {name!r}; valid: {[m.value for m in OverLimitPolicy]}")
    return text


def _label_space_max_options(label_space: LabelSpace | None) -> int | None:
    return None if label_space is None else int(getattr(label_space, "max_options"))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (frozenset, set, tuple)):
        return sorted(str(v) for v in value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #
def full_source_files(root: str | os.PathLike[str], family: str) -> list[Path]:
    """All per-dataset jsonl files of ``root/family`` (sorted), excluding ``all_*`` files.

    The ``sampled_300k`` directory is not a direct child file, so it is naturally excluded.
    """
    fam_dir = _resolve_path(root) / family
    if not fam_dir.is_dir():
        raise FileNotFoundError(f"family directory not found: {fam_dir}")
    return sorted(p for p in fam_dir.glob("*.jsonl") if p.is_file() and not p.name.startswith("all_"))


def list_sources(root: str | os.PathLike[str], family: str, mode: str) -> list[Path]:
    """Source files for ``family`` in the given data ``mode`` (``sampled`` | ``full``)."""
    mode = str(mode).lower()
    if mode == "sampled":
        path = _resolve_path(root) / family / "sampled_300k" / f"{family}_300k.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"sampled source not found: {path}")
        return [path]
    if mode == "full":
        files = full_source_files(root, family)
        if not files:
            raise FileNotFoundError(f"no per-dataset jsonl files under {_resolve_path(root) / family}")
        return files
    raise ValueError(f"unknown data mode {mode!r} (expected 'sampled' or 'full')")


# --------------------------------------------------------------------------- #
# Prepare: per-source-file worker
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FileTask:
    """One unit of prepare work: a single source file of one family."""

    family: str
    source: str
    part_dir: str
    part_index: int
    train_splits: tuple[str, ...]
    val_split: str
    include: frozenset[str] | None
    exclude: frozenset[str]
    max_train: int | None
    max_val: int | None
    val_only_quota: int | None = None  # fallback mode: collect at most this many val rows per dataset
    default_dataset: str = ""


@dataclass
class _FileResult:
    """What a worker hands back to the parent process (small: no example content)."""

    part_index: int
    source: str
    uncapped_parts: dict[str, str | None]
    capped_parts: dict[str, str | None]
    capped_meta: dict[str, list[tuple[int, str, str, int]]]  # split -> [(hash, dataset, qtype, K)] in file order
    stats: dict[str, Any]
    seconds: float


# Pool workers receive the label space through the initializer (no pickling of
# the tokenizer through the task queue; with the default ``fork`` start method
# nothing is pickled at all).
_WORKER_LABEL_SPACE: LabelSpace | None = None
_WORKER_POLICY: Any = None


def _init_worker(label_space: LabelSpace | None, policy: Any) -> None:
    global _WORKER_LABEL_SPACE, _WORKER_POLICY
    _WORKER_LABEL_SPACE = label_space
    _WORKER_POLICY = policy
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _new_stats() -> dict[str, Any]:
    return {
        "rows_read": 0,
        "rows_by_split": {"train": 0, "val": 0, "test": 0, "other": 0},
        "rows_malformed": 0,
        "rows_excluded_by_dataset": 0,
        "rows_adapter_error": 0,
        "examples": {"train": 0, "val": 0},
        "capped_candidates": {"train": 0, "val": 0},
        "per_dataset": {},
        "per_qtype": {"train": {}, "val": {}},
        "n_options_hist": {"train": {}, "val": {}},
        "drops": {},
    }


def _stats_add_example(stats: dict[str, Any], split: str, dataset: str, qtype: str, k: int) -> None:
    stats["examples"][split] += 1
    ds = stats["per_dataset"].setdefault(dataset, {"train": 0, "val": 0})
    ds[split] += 1
    qt = stats["per_qtype"][split]
    qt[qtype] = qt.get(qtype, 0) + 1
    hist = stats["n_options_hist"][split]
    hist[str(k)] = hist.get(str(k), 0) + 1


def _merge_counts(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively add integer counters of ``src`` into ``dst``."""
    for key, value in src.items():
        if isinstance(value, dict):
            _merge_counts(dst.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            dst[key] = dst.get(key, 0) + value
        else:
            dst.setdefault(key, value)


def _process_source(task: _FileTask) -> _FileResult:
    """Stream one source file and write its train/val candidates as part files.

    Uncapped datasets are written directly (``part-<n>.<split>.jsonl``).  For
    datasets with a per-dataset cap a bounded reservoir keeps the ``cap``
    examples with the smallest ``sha1(example.id)``; they are written to
    ``part-<n>.<split>.capped.jsonl`` together with their hashes so that the
    parent process can perform the *global* selection deterministically.
    """
    t0 = time.time()
    label_space = _WORKER_LABEL_SPACE
    policy = _WORKER_POLICY
    source = Path(task.source)
    part_dir = Path(task.part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    stats = _new_stats()
    caps = {"train": task.max_train, "val": task.max_val}
    writers: dict[str, Any] = {}
    uncapped_parts: dict[str, str | None] = {"train": None, "val": None}
    capped_parts: dict[str, str | None] = {"train": None, "val": None}
    reservoirs: dict[tuple[str, str], list[tuple[int, str, str, tuple[str, str, int]]]] = {}
    fallback_rows: dict[str, int] = {}
    fallback_datasets_seen: set[str] = set()
    quota = task.val_only_quota
    train_splits = set(task.train_splits)
    warned = 0

    def writer_for(split: str):
        fh = writers.get(split)
        if fh is None:
            path = part_dir / f"part-{task.part_index:05d}.{split}.jsonl"
            fh = open(path, "w", encoding="utf-8")
            writers[split] = fh
            uncapped_parts[split] = str(path)
        return fh

    try:
        with open(source, "r", encoding="utf-8", errors="replace") as fin:
            for raw in fin:
                if not raw.strip():
                    continue
                stats["rows_read"] += 1
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError("row is not a JSON object")
                except ValueError:
                    stats["rows_malformed"] += 1
                    continue
                split = str(row.get("split", "") or "")
                if split == _TEST_SPLIT:
                    stats["rows_by_split"]["test"] += 1
                    continue  # test rows are never used for training or validation
                if quota is not None:
                    # fallback mode: only val rows, bounded per dataset, early stop
                    if split != task.val_split:
                        stats["rows_by_split"]["train" if split in train_splits else "other"] += 1
                        continue
                    target_split = "val"
                else:
                    if split in train_splits:
                        target_split = "train"
                    elif split == task.val_split:
                        target_split = "val"
                    else:
                        stats["rows_by_split"]["other"] += 1
                        continue
                stats["rows_by_split"][target_split] += 1
                dataset = str(row.get("dataset") or task.default_dataset or source.stem)
                if not row.get("dataset"):
                    row["dataset"] = dataset
                if (task.include is not None and dataset not in task.include) or dataset in task.exclude:
                    stats["rows_excluded_by_dataset"] += 1
                    continue
                if quota is not None:
                    fallback_datasets_seen.add(dataset)
                    if fallback_rows.get(dataset, 0) >= quota:
                        if all(fallback_rows.get(d, 0) >= quota for d in fallback_datasets_seen):
                            break  # every dataset seen in this file has reached its quota
                        continue
                    fallback_rows[dataset] = fallback_rows.get(dataset, 0) + 1
                sample_id = str(row.get("sample_id") or f"{dataset}:{stats['rows_read']}")
                rng = random.Random(f"jev-prepare:{sample_id}")
                try:
                    examples, drops = jev_row_to_examples(row, family=task.family, label_space=label_space, policy=policy, rng=rng)
                except Exception as exc:  # noqa: BLE001 - a bad row must not kill the whole prepare
                    stats["rows_adapter_error"] += 1
                    if warned < 5:
                        warned += 1
                        logger.warning("%s: adapter failed on row %s: %s", source.name, sample_id, exc)
                    continue
                for name, count in (drops or {}).items():
                    stats["drops"][name] = stats["drops"].get(name, 0) + int(count)
                cap = caps[target_split]
                for ex in examples:
                    line = json.dumps(ex.to_dict(), ensure_ascii=False)
                    k = len(ex.options)
                    if cap is None:
                        writer_for(target_split).write(line + "\n")
                        _stats_add_example(stats, target_split, ex.dataset, ex.qtype, k)
                    else:
                        stats["capped_candidates"][target_split] += 1
                        heap = reservoirs.setdefault((target_split, ex.dataset), [])
                        h = _stable_hash(ex.id)
                        item = (-h, ex.id, line, (ex.dataset, ex.qtype, k))
                        if len(heap) < cap:
                            heapq.heappush(heap, item)
                        elif -h > heap[0][0]:  # smaller hash than the current largest -> replace
                            heapq.heapreplace(heap, item)
                if quota is not None and fallback_rows and all(fallback_rows.get(d, 0) >= quota for d in fallback_datasets_seen):
                    break
    finally:
        for fh in writers.values():
            fh.close()

    capped_meta: dict[str, list[tuple[int, str, str, int]]] = {"train": [], "val": []}
    for split in _SPLITS:
        keys = sorted(key for key in reservoirs if key[0] == split)
        if not keys:
            continue
        path = part_dir / f"part-{task.part_index:05d}.{split}.capped.jsonl"
        with open(path, "w", encoding="utf-8") as fout:
            for key in keys:
                for neg_h, _id, line, (dataset, qtype, k) in sorted(reservoirs[key], key=lambda it: (-it[0], it[1])):
                    fout.write(line + "\n")
                    capped_meta[split].append((-neg_h, dataset, qtype, k))
        capped_parts[split] = str(path)
    reservoirs.clear()
    if quota is not None:
        stats["val_fallback_rows_per_dataset"] = dict(sorted(fallback_rows.items()))
    return _FileResult(
        part_index=task.part_index,
        source=str(source),
        uncapped_parts=uncapped_parts,
        capped_parts=capped_parts,
        capped_meta=capped_meta,
        stats=stats,
        seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# Prepare: orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Result of :func:`prepare_data`."""

    train_files: list[Path]
    val_files: list[Path]
    stats: dict[str, Any]
    families: list[str] = field(default_factory=list)
    processed_dir: Path | None = None
    manifest_path: Path | None = None
    #: families whose val.jsonl is in ``val_files`` but whose train.jsonl is NOT in ``train_files``
    val_only_families: list[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return sum(int(s.get("examples", {}).get("train", 0)) for f, s in self.stats.get("families", {}).items()
                   if f not in self.val_only_families)

    @property
    def n_val(self) -> int:
        return sum(int(s.get("examples", {}).get("val", 0)) for s in self.stats.get("families", {}).values())


@dataclass(frozen=True)
class _PrepareSettings:
    """The config subset that determines the processed files (part of the cache key)."""

    root: Path
    mode: str
    families: tuple[str, ...]
    processed_dir: Path
    train_splits: tuple[str, ...]
    val_split: str
    val_fallback_per_dataset: int
    max_val_per_dataset: int | None
    max_train_per_dataset: int | None
    include: frozenset[str] | None
    exclude: frozenset[str]
    over_limit_policy: str
    workers: int
    rebuild: bool
    alphabet_size: int | None
    max_options: int | None
    dedup_val_against_train: bool = True
    #: families prepared for validation only (their train.jsonl is never used)
    val_only_families: tuple[str, ...] = ()
    #: effective source mode of every prepared family: ((family, "sampled"|"full"), ...)
    family_modes: tuple[tuple[str, str], ...] = ()
    #: test files whose states must never appear in train/val (data.exclude_test_overlap); () = off
    test_sources: tuple[Path, ...] = ()

    def mode_for(self, family: str) -> str:
        """Effective source mode of ``family`` (``data.family_modes`` override or the global ``data.mode``)."""
        return dict(self.family_modes).get(family, self.mode)

    @property
    def all_families(self) -> tuple[str, ...]:
        """Training families followed by the val-only families (the order they are prepared in)."""
        return self.families + tuple(f for f in self.val_only_families if f not in self.families)

    @classmethod
    def from_config(cls, cfg: Any, label_space: LabelSpace | None) -> _PrepareSettings:
        families = _cfg_get(cfg, "data.families", ["classification", "ranking", "search"])
        if isinstance(families, str):
            families = [f.strip() for f in families.split(",") if f.strip()]
        val_only = _cfg_get(cfg, "data.val_only_families", []) or []
        if isinstance(val_only, str):
            val_only = [f.strip() for f in val_only.split(",") if f.strip()]
        val_only = [str(f) for f in val_only]
        overlap = sorted(set(val_only) & {str(f) for f in families})
        if overlap:
            raise ValueError(f"data.val_only_families {overlap} must not also be in data.families")
        train_splits = _cfg_get(cfg, "data.train_splits", ["train"])
        if isinstance(train_splits, str):
            train_splits = [train_splits]
        train_splits = tuple(str(s) for s in train_splits if str(s) != _TEST_SPLIT)
        if not train_splits:
            raise ValueError("data.train_splits must contain at least one non-test split")
        val_split = str(_cfg_get(cfg, "data.val_split", "val"))
        if val_split == _TEST_SPLIT:
            raise ValueError("data.val_split may not be 'test'")
        mode = str(_cfg_get(cfg, "data.mode", "sampled")).lower()
        if mode not in ("sampled", "full"):
            raise ValueError(f"data.mode must be 'sampled' or 'full', got {mode!r}")
        overrides = _cfg_get(cfg, "data.family_modes", {}) or {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"data.family_modes must be a mapping family -> sampled|full, got {overrides!r}")
        fam_modes: list[tuple[str, str]] = []
        for fam in [str(f) for f in families] + val_only:
            fam_mode = str(overrides.get(fam) or mode).lower()
            if fam_mode not in ("sampled", "full"):
                raise ValueError(f"data.family_modes.{fam} must be 'sampled' or 'full', got {fam_mode!r}")
            fam_modes.append((fam, fam_mode))
        max_train = _cfg_get(cfg, "data.max_train_per_dataset")
        max_val = _cfg_get(cfg, "data.max_val_per_dataset")
        labels = getattr(label_space, "labels", None) if label_space is not None else None
        return cls(
            root=_resolve_path(_cfg_get(cfg, "data.root", "data/SearchJev-V1")),
            mode=mode,
            families=tuple(str(f) for f in families),
            processed_dir=_resolve_path(_cfg_get(cfg, "data.processed_dir", "data/processed")),
            train_splits=train_splits,
            val_split=val_split,
            val_fallback_per_dataset=int(_cfg_get(cfg, "data.val_fallback_per_dataset", 300) or 0),
            max_val_per_dataset=None if max_val is None else int(max_val),
            max_train_per_dataset=None if max_train is None else int(max_train),
            include=_as_name_set(_cfg_get(cfg, "data.include_datasets")),
            exclude=_as_name_set(_cfg_get(cfg, "data.exclude_datasets")) or frozenset(),
            over_limit_policy=str(_cfg_get(cfg, "labels.over_limit_policy", "drop")).lower(),
            workers=max(1, int(_cfg_get(cfg, "data.prepare_workers", 8) or 1)),
            rebuild=bool(_cfg_get(cfg, "data.rebuild", False)),
            alphabet_size=None if labels is None else len(labels),
            max_options=_label_space_max_options(label_space),
            dedup_val_against_train=bool(_cfg_get(cfg, "data.dedup_val_against_train", True)),
            val_only_families=tuple(val_only),
            family_modes=tuple(fam_modes),
            test_sources=_test_sources(cfg) if bool(_cfg_get(cfg, "data.exclude_test_overlap", False)) else (),
        )

    def family_key_payload(self, family: str, sources: Sequence[Path]) -> dict[str, Any]:
        """Everything that determines a family's processed files (hashed into its cache key).

        ``mode`` is the global ``data.mode`` (= the processed sub-directory).  A family whose
        effective source mode differs (``data.family_modes``) additionally records
        ``family_mode``; the key is only extended in that case so the keys of families that
        follow the global mode - and therefore their existing caches - stay unchanged.
        Whether a family is trained on or val-only is NOT part of the key: the processed
        files are identical, so a val-only family reuses the cache built when it was trained.
        """
        fam_mode = self.mode_for(family)
        payload = {
            "version": PREPARE_VERSION,
            "mode": self.mode,
            "family": family,
            "train_splits": list(self.train_splits),
            "val_split": self.val_split,
            "max_train_per_dataset": self.max_train_per_dataset,
            "max_val_per_dataset": self.max_val_per_dataset,
            "include_datasets": _to_jsonable(self.include),
            "exclude_datasets": _to_jsonable(self.exclude),
            "alphabet_size": self.alphabet_size,
            "max_options": self.max_options,
            "over_limit_policy": self.over_limit_policy,
            "val_fallback_per_dataset": self.val_fallback_per_dataset if fam_mode == "sampled" else None,
            "dedup_val_against_train": self.dedup_val_against_train,
            "sources": [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in sources],
        }
        if fam_mode != self.mode:
            payload["family_mode"] = fam_mode
        if self.test_sources:  # only when enabled, so caches built without the filter keep their keys
            payload["exclude_test_overlap"] = hashlib.sha1(json.dumps(
                [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in self.test_sources]).encode()).hexdigest()
        return payload

    def family_key(self, family: str, sources: Sequence[Path]) -> str:
        blob = json.dumps(self.family_key_payload(family, sources), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("could not read manifest %s; ignoring it", path)
        return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=_to_jsonable)
    os.replace(tmp, path)


def _family_files(fam_dir: Path) -> dict[str, Path]:
    return {"train": fam_dir / "train.jsonl", "val": fam_dir / "val.jsonl", "stats": fam_dir / "stats.json"}


def _clear_family_dir(fam_dir: Path) -> None:
    """Delete stale processed files of a family (outputs, offset indexes, parts)."""
    if not fam_dir.exists():
        return
    for p in fam_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.suffix in (".jsonl", ".npy", ".json") or p.name.endswith(_INDEX_SUFFIX):
            p.unlink(missing_ok=True)


def _run_tasks(tasks: list[_FileTask], label_space: LabelSpace | None, policy: Any, workers: int) -> list[_FileResult]:
    """Run the per-file workers (Pool or in-process) and return results sorted by part index."""
    results: list[_FileResult] = []
    n = len(tasks)
    if n == 0:
        return results
    workers = max(1, min(workers, n))
    started = time.time()

    def on_result(res: _FileResult) -> None:
        results.append(res)
        s = res.stats
        cand = s.get("capped_candidates", {})
        logger.info(
            "[prepare %3d/%d] %s: rows=%d train_ex=%d val_ex=%d (capped candidates: train=%d val=%d) "
            "test_rows=%d (%.1fs, elapsed %.0fs)",
            len(results), n, Path(res.source).name, s["rows_read"], s["examples"]["train"], s["examples"]["val"],
            cand.get("train", 0), cand.get("val", 0), s["rows_by_split"]["test"], res.seconds, time.time() - started,
        )

    if workers == 1:
        _init_worker(label_space, policy)
        for task in tasks:
            on_result(_process_source(task))
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # pragma: no cover - non-POSIX
            ctx = mp.get_context()
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(label_space, policy)) as pool:
            for res in pool.imap_unordered(_process_source, tasks):
                on_result(res)
    results.sort(key=lambda r: r.part_index)
    return results


def _merge_parts(results: list[_FileResult], out_path: Path, split: str, cap: int | None, stats: dict[str, Any]) -> int:
    """Concatenate part files in deterministic order into ``out_path``; return the line count.

    Uncapped parts are copied verbatim; capped parts are filtered by the global
    per-dataset selection (smallest ``cap`` hashes across all parts).
    """
    keep: dict[str, set[int]] = {}
    if cap is not None:
        by_dataset: dict[str, list[int]] = {}
        for res in results:
            for h, dataset, _qtype, _k in res.capped_meta.get(split, []):
                by_dataset.setdefault(dataset, []).append(h)
        for dataset, hashes in by_dataset.items():
            hashes.sort()
            keep[dataset] = set(hashes[:cap])
    n_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fout:
        for res in results:
            part = res.uncapped_parts.get(split)
            if part and os.path.exists(part):
                with open(part, "rb") as fin:
                    shutil.copyfileobj(fin, fout, length=1 << 20)
                n_written += res.stats["examples"][split]
            part = res.capped_parts.get(split)
            if part and os.path.exists(part):
                meta = res.capped_meta.get(split, [])
                with open(part, "rb") as fin:
                    for line, (h, dataset, qtype, k) in zip(fin, meta, strict=True):
                        if h in keep.get(dataset, ()):
                            fout.write(line)
                            _stats_add_example(stats, split, dataset, qtype, k)
                            n_written += 1
    os.replace(tmp, out_path)
    return n_written


def state_overlap_key(state: Any) -> int:
    """Leakage key of a state: rendered text, whitespace-collapsed, lower-cased."""
    text = " ".join(render_state(state).split()).lower()
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def _test_sources(cfg: Any) -> tuple[Path, ...]:
    """Every per-dataset file (``<root>/<family>/*.jsonl``) of the prepared families; their ``test`` rows are
    the states that must never appear in train/val."""
    root = _resolve_path(_cfg_get(cfg, "data.root", "data/SearchJev-V2"))
    fams = list(_cfg_get(cfg, "data.families", []) or [])
    fams += [f for f in (_cfg_get(cfg, "data.val_only_families") or []) if f not in fams]
    out: list[Path] = []
    for fam in fams:
        out += sorted(p for p in (root / str(fam)).glob("*.jsonl") if not p.name.startswith("all_"))
    return tuple(out)


def _test_keys_of_file(path: str) -> np.ndarray:
    keys = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if '"test"' not in line:          # cheap pre-filter; the split is checked after parsing
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and str(row.get("split", "")) == _TEST_SPLIT:
                keys.append(state_overlap_key(row.get("state")))
    return np.asarray(keys, dtype=np.uint64)


def load_test_keys(settings: "_PrepareSettings") -> np.ndarray:
    """Sorted unique leakage keys of all test rows of ``settings.test_sources`` (cached under processed_dir)."""
    fp = hashlib.sha1(json.dumps([[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in settings.test_sources]).encode()).hexdigest()
    cache = settings.processed_dir / "_test_keys" / f"{fp}.npy"
    if cache.is_file():
        return np.load(cache, allow_pickle=False)
    t0 = time.time()
    with mp.get_context("fork").Pool(max(1, min(settings.workers, len(settings.test_sources)))) as pool:
        parts = pool.map(_test_keys_of_file, [str(p) for p in settings.test_sources], chunksize=1)
    keys = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, keys)
    os.replace(tmp, cache)
    logger.info("[prepare] %d test-state keys from %d test files in %.0fs", keys.size, len(settings.test_sources), time.time() - t0)
    return keys


def _drop_test_overlap(path: Path, split: str, test_keys: np.ndarray, stats: dict[str, Any]) -> dict[str, int]:
    """Remove examples of ``path`` whose state occurs in a test row; returns dropped counts per dataset."""
    dropped: dict[str, int] = {}
    if not path.is_file() or test_keys.size == 0:
        return dropped
    tmp = path.with_name(path.name + f".testfilter{os.getpid()}")
    with open(path, "rb") as fin, open(tmp, "wb") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            k = np.uint64(state_overlap_key(row.get("state", "")))
            i = int(np.searchsorted(test_keys, k))
            if i < test_keys.size and test_keys[i] == k:
                ds = str(row.get("dataset", ""))
                dropped[ds] = dropped.get(ds, 0) + 1
                per_ds = stats["per_dataset"].get(ds)
                if per_ds is not None:
                    per_ds[split] = max(0, per_ds.get(split, 0) - 1)
                qt = str(row.get("qtype", ""))
                if qt in stats["per_qtype"][split]:
                    stats["per_qtype"][split][qt] = max(0, stats["per_qtype"][split][qt] - 1)
                kk = str(len(row.get("options") or []))
                if kk in stats["n_options_hist"][split]:
                    stats["n_options_hist"][split][kk] = max(0, stats["n_options_hist"][split][kk] - 1)
                continue
            fout.write(line)
    os.replace(tmp, path)
    return dropped


def _content_key(row: Mapping[str, Any]) -> int:
    """64-bit content fingerprint of a decision example (state, question, qtype, option keys)."""
    keys = tuple(str(o.get("key", "")) for o in (row.get("options") or []))
    text = json.dumps([row.get("state", ""), row.get("question", ""), row.get("qtype", ""), keys], ensure_ascii=False)
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def _drop_val_overlapping_train(train_path: Path, val_path: Path, stats: dict[str, Any]) -> int:
    """Remove val examples whose content fingerprint also occurs in the train file.

    Upstream corpora share items across datasets (e.g. afqmc/atec, NER corpora), so
    a val example can be byte-identical to a train example even though the split
    labels differ.  Dropping them keeps validation/calibration honest.  Returns the
    number of dropped val examples and fixes the per-dataset/per-qtype counters.
    """
    seen: set[int] = set()
    with open(train_path, "rb") as fh:
        for line in fh:
            if line.strip():
                seen.add(_content_key(json.loads(line)))
    dropped = 0
    tmp = val_path.with_name(val_path.name + f".dedup{os.getpid()}")
    with open(val_path, "rb") as fin, open(tmp, "wb") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            if _content_key(row) in seen:
                dropped += 1
                ds = str(row.get("dataset", ""))
                qt = str(row.get("qtype", ""))
                k = str(len(row.get("options") or []))
                per_ds = stats["per_dataset"].get(ds)
                if per_ds is not None:
                    per_ds["val"] = max(0, per_ds.get("val", 0) - 1)
                if qt in stats["per_qtype"]["val"]:
                    stats["per_qtype"]["val"][qt] = max(0, stats["per_qtype"]["val"][qt] - 1)
                if k in stats["n_options_hist"]["val"]:
                    stats["n_options_hist"]["val"][k] = max(0, stats["n_options_hist"]["val"][k] - 1)
                continue
            fout.write(line)
    os.replace(tmp, val_path)
    return dropped


def _make_tasks(family: str, sources: Sequence[Path], part_dir: Path, settings: _PrepareSettings, *, start_index: int = 0, val_only_quota: int | None = None) -> list[_FileTask]:
    return [
        _FileTask(
            family=family,
            source=str(src),
            part_dir=str(part_dir),
            part_index=start_index + i,
            train_splits=settings.train_splits,
            val_split=settings.val_split,
            include=settings.include,
            exclude=settings.exclude,
            max_train=settings.max_train_per_dataset,
            max_val=settings.max_val_per_dataset,
            val_only_quota=val_only_quota,
            default_dataset=src.stem,
        )
        for i, src in enumerate(sources)
    ]


def _prepare_family(family: str, sources: list[Path], settings: _PrepareSettings, label_space: LabelSpace | None, policy: Any) -> dict[str, Any]:
    """Build ``train.jsonl`` / ``val.jsonl`` / ``stats.json`` for one family; return the stats."""
    fam_dir = settings.processed_dir / settings.mode / family
    fam_mode = settings.mode_for(family)
    _clear_family_dir(fam_dir)
    fam_dir.mkdir(parents=True, exist_ok=True)
    part_dir = fam_dir / "parts"
    files = _family_files(fam_dir)
    t0 = time.time()
    logger.info("[prepare] family=%s mode=%s (sources: %s) sources=%d workers=%d", family, settings.mode, fam_mode,
                len(sources), settings.workers)

    tasks = _make_tasks(family, sources, part_dir, settings)
    results = _run_tasks(tasks, label_space, policy, settings.workers)

    stats = _new_stats()
    for res in results:  # row-level counters and uncapped example counters are exact per worker
        _merge_counts(stats, res.stats)
    stats["examples"] = {"train": 0, "val": 0}  # recomputed exactly during the merge
    for split in _SPLITS:
        for res in results:
            stats["examples"][split] += res.stats["examples"][split]
    n_train = _merge_parts(results, files["train"], "train", settings.max_train_per_dataset, stats)
    n_val = _merge_parts(results, files["val"], "val", settings.max_val_per_dataset, stats)
    stats["examples"] = {"train": n_train, "val": n_val}
    stats["val_fallback"] = {"used": False, "files": [], "rows_per_dataset": {}}

    if n_val == 0 and fam_mode == "sampled" and settings.val_fallback_per_dataset > 0:
        full_files = [p for p in full_source_files(settings.root, family) if p.resolve() not in {s.resolve() for s in sources}]
        if settings.include is not None:
            full_files = [p for p in full_files if p.stem in settings.include] or full_files
        full_files = [p for p in full_files if p.stem not in settings.exclude]
        logger.info(
            "[prepare] family=%s has no '%s' rows in its sources; streaming up to %d val rows per dataset from %d full files",
            family, settings.val_split, settings.val_fallback_per_dataset, len(full_files),
        )
        fb_tasks = _make_tasks(family, full_files, part_dir, settings, start_index=len(tasks), val_only_quota=settings.val_fallback_per_dataset)
        fb_results = _run_tasks(fb_tasks, label_space, policy, settings.workers)
        fb_stats = _new_stats()
        for res in fb_results:
            _merge_counts(fb_stats, res.stats)
        for res in fb_results:
            fb_stats["examples"]["val"] += res.stats["examples"]["val"]
        fb_stats["examples"]["train"] = 0
        n_val = _merge_parts(fb_results, files["val"], "val", settings.max_val_per_dataset, fb_stats)
        fb_stats["examples"] = {"train": 0, "val": n_val}
        rows_per_dataset: dict[str, int] = {}
        for res in fb_results:
            for ds, cnt in res.stats.get("val_fallback_rows_per_dataset", {}).items():
                rows_per_dataset[ds] = rows_per_dataset.get(ds, 0) + int(cnt)
        # fold the fallback val examples into the family stats
        stats["examples"]["val"] = n_val
        for ds, counts in fb_stats["per_dataset"].items():
            stats["per_dataset"].setdefault(ds, {"train": 0, "val": 0})["val"] += counts["val"]
        for qt, cnt in fb_stats["per_qtype"]["val"].items():
            stats["per_qtype"]["val"][qt] = stats["per_qtype"]["val"].get(qt, 0) + cnt
        for k, cnt in fb_stats["n_options_hist"]["val"].items():
            stats["n_options_hist"]["val"][k] = stats["n_options_hist"]["val"].get(k, 0) + cnt
        for name, cnt in fb_stats["drops"].items():
            stats["drops"][name] = stats["drops"].get(name, 0) + cnt
        stats["val_fallback"] = {
            "used": True,
            "quota_rows_per_dataset": settings.val_fallback_per_dataset,
            "files": [p.name for p in full_files],
            "rows_read": fb_stats["rows_read"],
            "rows_per_dataset": dict(sorted(rows_per_dataset.items())),
            "examples": n_val,
        }
        contributed = [ds for ds, c in rows_per_dataset.items() if c > 0]
        logger.info("[prepare] family=%s val fallback: %d val examples from datasets %s", family, n_val, contributed)
        if n_val == 0:
            logger.warning("[prepare] family=%s still has no val examples after the fallback", family)

    stats["test_overlap"] = {"checked": False}
    if settings.test_sources:
        test_keys = load_test_keys(settings)
        dropped_tr = _drop_test_overlap(files["train"], "train", test_keys, stats)
        dropped_va = _drop_test_overlap(files["val"], "val", test_keys, stats)
        n_train -= sum(dropped_tr.values())
        n_val -= sum(dropped_va.values())
        stats["examples"] = {"train": n_train, "val": n_val}
        stats["test_overlap"] = {"checked": True, "test_files": len(settings.test_sources),
                                 "dropped_train": dropped_tr, "dropped_val": dropped_va}
        if dropped_tr or dropped_va:
            logger.info("[prepare] family=%s dropped %d train / %d val examples whose state appears in a test set",
                        family, sum(dropped_tr.values()), sum(dropped_va.values()))

    stats["val_overlap_with_train"] = {"checked": False, "dropped": 0}
    if settings.dedup_val_against_train and n_train > 0 and n_val > 0:
        dropped = _drop_val_overlapping_train(files["train"], files["val"], stats)
        n_val -= dropped
        stats["examples"]["val"] = n_val
        stats["val_overlap_with_train"] = {"checked": True, "dropped": dropped}
        if dropped:
            logger.info("[prepare] family=%s dropped %d val examples whose content also appears in train", family, dropped)

    shutil.rmtree(part_dir, ignore_errors=True)
    stats.pop("val_fallback_rows_per_dataset", None)
    stats["per_dataset"] = dict(sorted(stats["per_dataset"].items()))
    for split in _SPLITS:
        stats["n_options_hist"][split] = dict(sorted(stats["n_options_hist"][split].items(), key=lambda kv: int(kv[0])))
    stats.update({
        "family": family,
        "mode": settings.mode,
        "source_mode": fam_mode,
        "sources": [str(p) for p in sources],
        "train_file": str(files["train"]),
        "val_file": str(files["val"]),
        "seconds": round(time.time() - t0, 1),
    })
    _write_json(files["stats"], stats)
    if n_train == 0:
        logger.warning("[prepare] family=%s produced 0 training examples", family)
    logger.info("[prepare] family=%s done: train=%d val=%d rows=%d in %.0fs", family, n_train, n_val, stats["rows_read"], stats["seconds"])
    return stats


def prepare_data(cfg: Any, label_space: LabelSpace | None) -> PreparedData:
    """Convert the configured JEV sources into processed train/val example files.

    Output layout: ``processed_dir/<data.mode>/<family>/{train.jsonl,val.jsonl,stats.json}``
    and ``processed_dir/<data.mode>/manifest.json`` (the directory always follows the
    global ``data.mode``, also for a family read in another mode via ``data.family_modes``).
    A family is rebuilt only when its cache key (mode / family source mode, splits, caps,
    filters, alphabet size / max options, policy, fallback quota, source names and sizes)
    changed, its files are missing, or ``data.rebuild`` is set.

    ``data.families`` contribute their train.jsonl AND val.jsonl; ``data.val_only_families``
    are prepared (or reused from the cache) the same way but contribute only val.jsonl.
    """
    settings = _PrepareSettings.from_config(cfg, label_space)
    policy = _resolve_policy(settings.over_limit_policy)
    mode_dir = settings.processed_dir / settings.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = mode_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    fam_entries: dict[str, Any] = dict(manifest.get("families", {}))

    train_files: list[Path] = []
    val_files: list[Path] = []
    all_stats: dict[str, Any] = {}
    for family in settings.all_families:
        sources = list_sources(settings.root, family, settings.mode_for(family))
        key = settings.family_key(family, sources)
        fam_dir = mode_dir / family
        files = _family_files(fam_dir)
        entry = fam_entries.get(family) or {}
        cached = (
            not settings.rebuild
            and entry.get("key") == key
            and all(files[k].is_file() for k in ("train", "val", "stats"))
        )
        if cached:
            with open(files["stats"], "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            logger.info("[prepare] family=%s: reusing cached files in %s (train=%d val=%d)%s", family, fam_dir,
                        stats["examples"]["train"], stats["examples"]["val"],
                        " - val only" if family in settings.val_only_families else "")
        else:
            stats = _prepare_family(family, sources, settings, label_space, policy)
            fam_entries[family] = {
                "key": key,
                "key_payload": settings.family_key_payload(family, sources),
                "source_mode": settings.mode_for(family),
                "train": str(files["train"]),
                "val": str(files["val"]),
                "stats": str(files["stats"]),
                "n_train": stats["examples"]["train"],
                "n_val": stats["examples"]["val"],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            manifest = {"version": PREPARE_VERSION, "mode": settings.mode, "processed_dir": str(settings.processed_dir), "families": fam_entries}
            _write_json(manifest_path, manifest)
        all_stats[family] = stats
        if family not in settings.val_only_families:
            train_files.append(files["train"])
        val_files.append(files["val"])

    combined = {
        "mode": settings.mode,
        "processed_dir": str(settings.processed_dir),
        "families": all_stats,
        "train_families": list(settings.families),
        "val_only_families": list(settings.val_only_families),
        "family_modes": dict(settings.family_modes),
        "n_train": sum(s["examples"]["train"] for f, s in all_stats.items() if f not in settings.val_only_families),
        "n_val": sum(s["examples"]["val"] for s in all_stats.values()),
    }
    return PreparedData(
        train_files=train_files,
        val_files=val_files,
        stats=combined,
        families=list(settings.families),
        processed_dir=settings.processed_dir,
        manifest_path=manifest_path,
        val_only_families=list(settings.val_only_families),
    )


def cache_status(cfg: Any) -> dict[str, dict[str, Any]]:
    """Dry-run helper: would :func:`prepare_data` reuse each family's cache?  (No tokenizer needed.)

    For every prepared family (``data.families`` + ``data.val_only_families``) the current
    cache-key payload is compared with the one recorded in ``<processed_dir>/<mode>/manifest.json``;
    the label-space fields (alphabet size / max options) are taken from the manifest because
    they need the tokenizer.  Returns ``{family: {"state": cached|stale|to build|no sources,
    "role": train|val-only, "source_mode": ..., "dir": ..., "diff": [changed payload keys],
    "error": ...}}``.
    """
    import dataclasses

    settings = _PrepareSettings.from_config(cfg, None)
    manifest = _load_manifest(settings.processed_dir / settings.mode / "manifest.json")
    entries = manifest.get("families", {}) if isinstance(manifest.get("families"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for family in settings.all_families:
        fam_dir = settings.processed_dir / settings.mode / family
        info: dict[str, Any] = {"role": "val-only" if family in settings.val_only_families else "train",
                                "source_mode": settings.mode_for(family), "dir": str(fam_dir), "diff": []}
        out[family] = info
        try:
            sources = list_sources(settings.root, family, settings.mode_for(family))
        except (FileNotFoundError, ValueError) as exc:
            info.update(state="no sources", error=str(exc))
            continue
        entry = entries.get(family) or {}
        recorded = entry.get("key_payload") or {}
        files = _family_files(fam_dir)
        if settings.rebuild or not recorded or not all(files[k].is_file() for k in ("train", "val", "stats")):
            info["state"] = "to build"
            continue
        probe = dataclasses.replace(settings, alphabet_size=recorded.get("alphabet_size"),
                                    max_options=recorded.get("max_options"))
        current = json.loads(json.dumps(probe.family_key_payload(family, sources), default=_to_jsonable))
        info["diff"] = sorted(k for k in set(current) | set(recorded) if current.get(k) != recorded.get(k))
        info["state"] = "stale" if info["diff"] else "cached"
    return out


def format_prepare_table(stats: dict[str, Any]) -> str:
    """Render a compact text table of :attr:`PreparedData.stats` (scripts print it)."""
    fam_stats = stats.get("families", stats)
    val_only = set(stats.get("val_only_families") or [])
    lines = [f"{'family':<16}{'rows':>12}{'train':>12}{'val':>10}{'test(skip)':>12}  qtypes(train)            drops"]
    for family, s in fam_stats.items():
        qt = ",".join(f"{k}={v}" for k, v in sorted(s.get("per_qtype", {}).get("train", {}).items()))
        drops = ",".join(f"{k}={v}" for k, v in sorted(s.get("drops", {}).items())) or "-"
        fb = " (val via fallback)" if s.get("val_fallback", {}).get("used") else ""
        n_train = "-" if family in val_only else f"{s['examples']['train']:,}"
        if family in val_only:
            fb += " (val only: train not used)"
        lines.append(
            f"{family:<16}{s.get('rows_read', 0):>12,}{n_train:>12}{s['examples']['val']:>10,}"
            f"{s.get('rows_by_split', {}).get('test', 0):>12,}  {qt:<25} {drops}{fb}"
        )
    lines.append(f"{'total':<16}{'':>12}{stats.get('n_train', 0):>12,}{stats.get('n_val', 0):>10,}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Offset index + example readers
# --------------------------------------------------------------------------- #
def build_line_index(path: str | os.PathLike[str], chunk_size: int = 1 << 24) -> np.ndarray:
    """Byte offsets (int64) of every non-empty line of ``path``."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    starts = [np.zeros(1, dtype=np.int64)]
    base = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.uint8)
            nl = np.flatnonzero(arr == 10).astype(np.int64) + (base + 1)
            if nl.size:
                starts.append(nl)
            base += len(chunk)
    offsets = np.concatenate(starts)
    ends = np.append(offsets[1:], size)
    lengths = ends - offsets
    # drop the phantom line after a trailing newline and blank ("\n") lines
    return np.ascontiguousarray(offsets[lengths > 1])


def load_or_build_line_index(path: str | os.PathLike[str]) -> np.ndarray:
    """Load ``<path>.idx.npy`` if it is fresh, otherwise (re)build and cache it atomically."""
    path = Path(path)
    idx_path = path.with_name(path.name + _INDEX_SUFFIX)
    meta_path = idx_path.with_name(idx_path.name + ".meta.json")
    st = path.stat()
    signature = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if idx_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("size") == signature["size"] and meta.get("mtime_ns") == signature["mtime_ns"]:
                offsets = np.load(idx_path, allow_pickle=False)
                if offsets.dtype == np.int64 and offsets.ndim == 1 and (offsets.size == 0 or offsets[-1] < st.st_size):
                    return offsets
            logger.info("offset index %s is stale (jsonl size/mtime changed); rebuilding", idx_path)
        except (OSError, ValueError) as exc:
            logger.warning("could not load offset index %s (%s); rebuilding", idx_path, exc)
    t0 = time.time()
    offsets = build_line_index(path)
    tmp = idx_path.with_name(idx_path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:  # np.save(path) would append ".npy" to the tmp name
            np.save(fh, offsets)
        os.replace(tmp, idx_path)
        meta_path.write_text(json.dumps(signature), encoding="utf-8")
    except OSError as exc:  # read-only location: keep the in-memory index
        logger.warning("could not write offset index %s: %s", idx_path, exc)
        tmp.unlink(missing_ok=True)
    logger.info("built offset index for %s: %d lines in %.1fs", path.name, offsets.size, time.time() - t0)
    return offsets


_LENGTH_SUFFIX = ".len.npy"
_LENGTH_VERSION = 1


def _est_tokens(text: str) -> float:
    """Rough Qwen token count: ~1 token per CJK character, ~3.6 characters per token otherwise."""
    cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" or "\uff00" <= ch <= "\uffef")
    return cjk + (len(text) - cjk) / 3.6


def estimate_prompt_tokens(ex: DecisionExample, max_length: int | None = None) -> int:
    """Cheap prompt-length estimate (no tokenizer) used to group training batches by length."""
    est = 80.0 + _est_tokens(ex.state) + _est_tokens(ex.question) + sum(_est_tokens(o.text) + 4.0 for o in ex.options)
    n = int(est)
    return min(n, int(max_length)) if max_length else n


def load_or_build_length_index(path: str | os.PathLike[str], max_length: int | None = None) -> np.ndarray:
    """Per-line prompt-length estimates of a processed jsonl, cached as ``<path>.len.npy`` (int32)."""
    path = Path(path)
    len_path = path.with_name(path.name + _LENGTH_SUFFIX)
    meta_path = len_path.with_name(len_path.name + ".meta.json")
    st = path.stat()
    signature = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "max_length": max_length, "version": _LENGTH_VERSION}
    if len_path.is_file() and meta_path.is_file():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == signature:
                return np.load(len_path, allow_pickle=False)
        except (OSError, ValueError):
            pass
    t0 = time.time()
    out: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(estimate_prompt_tokens(DecisionExample.from_dict(json.loads(line)), max_length))
    arr = np.asarray(out, dtype=np.int32)
    tmp = len_path.with_name(len_path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, len_path)
        meta_path.write_text(json.dumps(signature), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write length index %s: %s", len_path, exc)
        tmp.unlink(missing_ok=True)
    logger.info("built length index for %s: %d lines in %.1fs", path.name, arr.size, time.time() - t0)
    return arr


def dataset_lengths(ds: Any, max_length: int | None = None) -> list[int] | None:
    """Per-item length estimates of a :class:`JEVDataset` or a ``ConcatDataset`` of them (``None`` if unsupported)."""
    parts = getattr(ds, "datasets", None)
    if parts is None:
        return ds.lengths(max_length) if hasattr(ds, "lengths") else None
    out: list[int] = []
    for d in parts:
        if not hasattr(d, "lengths"):
            return None
        out.extend(d.lengths(max_length))
    return out


def count_lines(path: str | os.PathLike[str]) -> int:
    """Number of non-empty lines of a jsonl file (uses / creates the cached offset index)."""
    return int(load_or_build_line_index(path).size)


def read_examples(path: str | os.PathLike[str], *, ids: Iterable[str] | None = None, limit: int | None = None) -> Iterator[DecisionExample]:
    """Stream :class:`DecisionExample` objects from a processed jsonl file.

    ``ids`` restricts the output to those example ids; ``limit`` stops after
    that many yielded examples.
    """
    wanted = None if ids is None else set(ids)
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if wanted is not None and d.get("id") not in wanted:
                continue
            yield DecisionExample.from_dict(d)
            n += 1
            if limit is not None and n >= limit:
                break


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser (uint64 -> uint64)."""
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_subset(n: int, k: int, seed: int) -> np.ndarray:
    """First ``k`` indices of a deterministic hash-shuffled order of ``range(n)`` (sorted ascending)."""
    if k >= n:
        return np.arange(n, dtype=np.int64)
    with np.errstate(over="ignore"):
        keys = _splitmix64(np.arange(n, dtype=np.uint64) ^ np.uint64((int(seed) * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF))
    chosen = np.argsort(keys, kind="stable")[: max(0, int(k))]
    return np.sort(chosen).astype(np.int64)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class JEVDataset(Dataset):
    """Offset-indexed reader over a processed ``*.jsonl`` file of decision examples.

    Item ``i`` maps to ``(copy, j) = divmod(i, n)``; line ``j`` is decoded into
    a :class:`DecisionExample`, its options are (optionally) permuted with a
    permutation that depends only on ``(seed, copy, line index)`` and the
    prompt is encoded with the :class:`PromptBuilder`.  The returned dict:

    ``input_ids``, ``attention_mask`` (lists of ints), ``label_token_ids``
    (first ``K`` label ids), ``target_dist`` (length ``K``, display order,
    one-hot when the example has no soft target), ``target_idx`` (display
    index, ``-1`` if unlabeled), ``qtype_id``, ``num_options``, ``perm``
    (display slot -> canonical index) and ``example_id``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        prompt_builder: PromptBuilder,
        label_space: LabelSpace,
        permutation: Mapping[str, Any] | None,
        seed: int = 42,
        copies: int = 1,
        max_examples: int | None = None,
        sample_weights: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.sample_weights = ({str(k): (dict(v) if isinstance(v, Mapping) else v) for k, v in sample_weights.items()}
                               if sample_weights else None)  # plain dicts: picklable for DataLoader workers
        self.prompt_builder = prompt_builder
        self.label_space = label_space
        perm_cfg = dict(permutation or {})
        self.permutation: dict[str, bool] = {qt: bool(perm_cfg.get(qt, False)) for qt in QTYPES}
        self.seed = int(seed)
        self.copies = max(1, int(copies))
        self._offsets = load_or_build_line_index(self.path)
        n_total = int(self._offsets.size)
        self._selected: np.ndarray | None = None
        if max_examples is not None and int(max_examples) < n_total:
            self._selected = hash_subset(n_total, int(max_examples), self.seed)
        self.n = n_total if self._selected is None else int(self._selected.size)
        self._fh = None
        self._pid: int | None = None

    # -- pickling / per-process handles -------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_pid"] = None
        return state

    def _handle(self):
        if self._fh is None or self._pid != os.getpid():
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = open(self.path, "rb")
            self._pid = os.getpid()
        return self._fh

    def __del__(self) -> None:  # pragma: no cover - best effort
        fh = getattr(self, "_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass

    # -- basic access ---------------------------------------------------------
    def __len__(self) -> int:
        return self.n * self.copies

    @property
    def n_examples(self) -> int:
        """Number of distinct examples (before ``copies``)."""
        return self.n

    def line_index(self, j: int) -> int:
        """Underlying file line number of the ``j``-th (possibly subsampled) example."""
        return int(j) if self._selected is None else int(self._selected[j])

    def read_line(self, j: int) -> str:
        if not 0 <= j < self.n:
            raise IndexError(j)
        fh = self._handle()
        fh.seek(int(self._offsets[self.line_index(j)]))
        return fh.readline().decode("utf-8")

    def example(self, j: int) -> DecisionExample:
        """Decode the ``j``-th example (no permutation, no encoding)."""
        return DecisionExample.from_dict(json.loads(self.read_line(j)))

    def examples(self) -> Iterator[DecisionExample]:
        """Iterate over all (subsampled) examples in order."""
        for j in range(self.n):
            yield self.example(j)

    def lengths(self, max_length: int | None = None) -> list[int]:
        """Estimated prompt length of every item (``copies`` repeat the file order), for length-grouped batching."""
        per_line = load_or_build_length_index(self.path, max_length)
        base = per_line if self._selected is None else per_line[self._selected]
        return [int(x) for x in base.tolist()] * self.copies

    def perm_for(self, ex: DecisionExample, copy: int, j: int) -> list[int]:
        """Display permutation for example ``j`` in ``copy`` (identity if disabled for its qtype)."""
        k = len(ex.options)
        perm = list(range(k))
        if k > 1 and self.permutation.get(ex.qtype, False):
            rng = random.Random(f"jev-perm:{self.seed}:{copy}:{self.line_index(j)}")
            rng.shuffle(perm)
        return perm

    def __getitem__(self, i: int) -> dict[str, Any]:
        total = len(self)
        if i < 0:
            i += total
        if not 0 <= i < total:
            raise IndexError(i)
        copy, j = divmod(int(i), self.n)
        ex = self.example(j)
        perm = self.perm_for(ex, copy, j)
        feat = encode_example(ex, perm, self.prompt_builder, self.label_space)
        if self.sample_weights:
            feat["weight"] = example_weight(ex.meta, self.sample_weights)
        return feat


def encode_example(ex: DecisionExample, perm: Sequence[int] | None, prompt_builder: PromptBuilder, label_space: LabelSpace) -> dict[str, Any]:
    """Encode one example under a display permutation into the trainer's feature dict."""
    k = len(ex.options)
    perm_list = list(range(k)) if perm is None else [int(p) for p in perm]
    if sorted(perm_list) != list(range(k)):
        raise ValueError(f"perm {perm_list} is not a permutation of range({k}) for example {ex.id}")
    enc = prompt_builder.encode(ex, None if perm is None else perm_list)
    if ex.target_dist is not None:
        canon = [float(p) for p in ex.target_dist]
        if len(canon) != k:
            raise ValueError(f"target_dist length {len(canon)} != {k} options for example {ex.id}")
    elif ex.target_idx is not None:
        canon = [0.0] * k
        canon[int(ex.target_idx)] = 1.0
    else:
        canon = [0.0] * k
    display_dist = [canon[c] for c in perm_list]
    target_idx = -1 if ex.target_idx is None else perm_list.index(int(ex.target_idx))
    qtype_id = QTYPE_IDS.get(ex.qtype)
    if qtype_id is None:
        raise ValueError(f"unknown qtype {ex.qtype!r} for example {ex.id}")
    return {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
        "label_token_ids": list(label_space.ids_for(k)),
        "target_dist": display_dist,
        "target_idx": int(target_idx),
        "qtype_id": int(qtype_id),
        "num_options": int(k),
        "perm": perm_list,
        "example_id": ex.id,
    }


# --------------------------------------------------------------------------- #
# Dataset builders and calibration split
# --------------------------------------------------------------------------- #
def _allocate(total: int | None, sizes: Sequence[int]) -> list[int | None]:
    """Split a global example cap over files proportionally to their sizes (largest remainder)."""
    if total is None:
        return [None] * len(sizes)
    total = max(0, int(total))
    n_all = sum(sizes)
    if n_all <= total:
        return [None] * len(sizes)
    raw = [total * s / n_all for s in sizes]
    alloc = [int(r) for r in raw]
    remainder = total - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: (raw[i] - alloc[i], -sizes[i]), reverse=True)
    for i in order[:remainder]:
        alloc[i] += 1
    return [min(a, s) for a, s in zip(alloc, sizes, strict=True)]


def _concat(datasets: list[JEVDataset], what: str) -> Dataset:
    if not datasets:
        raise ValueError(f"no {what} files to build a dataset from")
    total = sum(len(d) for d in datasets)
    logger.info("%s dataset: %d items over %d files (%s)", what, total, len(datasets), ", ".join(f"{d.path.parent.name}={len(d)}" for d in datasets))
    return ConcatDataset(datasets)


def build_train_dataset(cfg: Any, prepared: PreparedData, prompt_builder: PromptBuilder, label_space: LabelSpace) -> Dataset:
    """ConcatDataset over the families' ``train.jsonl`` files with permutation/copies from the config."""
    perm_cfg = dict(_cfg_get(cfg, "data.permutation", {"choice": True, "noul": True, "score": False, "copies": 1}) or {})
    copies = max(1, int(perm_cfg.get("copies", 1) or 1))
    seed = int(_cfg_get(cfg, "run.seed", 42))
    max_train = _cfg_get(cfg, "training.max_train_examples")
    files = [Path(p) for p in prepared.train_files if Path(p).is_file()]
    sizes = [count_lines(p) for p in files]
    caps = _allocate(None if max_train is None else int(max_train), sizes)
    datasets = [
        JEVDataset(p, prompt_builder, label_space, perm_cfg, seed=seed, copies=copies, max_examples=cap,
                   sample_weights=_cfg_get(cfg, "loss.sample_weights"))
        for p, size, cap in zip(files, sizes, caps, strict=True)
        if size > 0
    ]
    return _concat(datasets, "train")


def build_val_dataset(
    cfg: Any,
    prepared: PreparedData,
    prompt_builder: PromptBuilder,
    label_space: LabelSpace,
    max_examples: int | None = None,
    *,
    cap_from_config: bool = True,
    ids: Iterable[str] | None = None,
) -> Dataset:
    """ConcatDataset over the families' ``val.jsonl`` files (identity permutation, one copy).

    ``max_examples`` caps the total (deterministic hash subset); when it is
    ``None`` and ``cap_from_config`` is true, ``training.max_eval_examples``
    is used.  Pass ``cap_from_config=False`` for the full validation set.
    ``ids`` is currently informational only (filtering by id is done by the
    caller with :func:`read_examples`).
    """
    if max_examples is None and cap_from_config:
        value = _cfg_get(cfg, "training.max_eval_examples")
        max_examples = None if value is None else int(value)
    seed = int(_cfg_get(cfg, "run.seed", 42))
    files = [Path(p) for p in prepared.val_files if Path(p).is_file()]
    sizes = [count_lines(p) for p in files]
    caps = _allocate(max_examples, sizes)
    identity = {qt: False for qt in QTYPES}
    datasets = [
        JEVDataset(p, prompt_builder, label_space, identity, seed=seed, copies=1, max_examples=cap)
        for p, size, cap in zip(files, sizes, caps, strict=True)
        if size > 0
    ]
    return _concat(datasets, "val")


def calibration_unit(example_id: str, seed: int = 42) -> float:
    """Uniform number in ``[0, 1)`` that decides the calibration/held-out side of an example id.

    Shared by :func:`split_val_for_calibration` and ``start_training.py`` so that
    the split is identical wherever it is computed.
    """
    return _hash_unit(f"jev-calib:{int(seed)}:{example_id}")


def split_val_for_calibration(
    examples_or_file: str | os.PathLike[str] | Iterable[DecisionExample | str] | Iterable[str | os.PathLike[str]],
    fraction: float = 0.5,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split validation example ids into ``(calibration_ids, heldout_ids)`` by ``sha1(seed:id)``.

    Accepts a processed jsonl path, a list of such paths, or an iterable of
    :class:`DecisionExample` / id strings.  The split is stable across runs
    and processes because it depends only on the id and the seed.
    """
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")

    def iter_ids() -> Iterator[str]:
        src = examples_or_file
        if isinstance(src, (str, os.PathLike)):
            for ex in read_examples(src):
                yield ex.id
            return
        for item in src:
            if isinstance(item, DecisionExample):
                yield item.id
            elif isinstance(item, os.PathLike) or (isinstance(item, str) and item.endswith(".jsonl") and os.path.isfile(item)):
                for ex in read_examples(item):
                    yield ex.id
            else:
                yield str(item)

    calib: set[str] = set()
    held: set[str] = set()
    for ex_id in iter_ids():
        (calib if calibration_unit(ex_id, seed) < fraction else held).add(ex_id)
    return calib, held


