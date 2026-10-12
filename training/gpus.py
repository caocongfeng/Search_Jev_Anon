"""GPU discovery and selection for the one-click launcher.

``distributed.num_gpus`` in ``config.yaml`` (or ``--num-gpus``) says *how many*
GPUs a run may use: ``all`` (default) takes every GPU visible to the process,
an integer ``N`` picks ``N`` of them.  ``distributed.gpus`` (or ``--gpus``)
names explicit device ids and wins over the count.  When picking ``N`` out of
more candidates the devices with the most free memory are preferred
(``distributed.gpu_selection: free``) so that a shared machine does not hand
the run a GPU that another job already fills; ``first`` keeps the natural
order instead.  ``distributed.min_free_gb`` optionally drops devices with less
free memory than that.

The module deliberately avoids importing torch: it must run before
``CUDA_VISIBLE_DEVICES`` is exported and torch is initialised.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SELECTION_MODES: tuple[str, ...] = ("free", "first")


@dataclass(frozen=True)
class GPUInfo:
    """One physical GPU as reported by ``nvidia-smi``."""

    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    @property
    def memory_free_gb(self) -> float:
        return self.memory_free_mb / 1024.0


class GPUSelectionError(RuntimeError):
    """Raised when the requested GPUs cannot be satisfied."""


def query_gpus(timeout: float = 20.0) -> list[GPUInfo]:
    """Return all physical GPUs via ``nvidia-smi`` (empty list when unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - no driver / binary
        logger.warning("nvidia-smi unavailable (%s); assuming no GPUs", exc)
        return []
    if out.returncode != 0:
        logger.warning("nvidia-smi failed (rc=%s): %s", out.returncode, out.stderr.strip()[:200])
        return []
    gpus: list[GPUInfo] = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GPUInfo(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3]))))
        except ValueError:
            continue
    return gpus


def parse_gpu_ids(value: Any) -> list[int] | None:
    """``"0,1"`` / ``[0, 1]`` / ``"0"`` -> ``[0, 1]``; ``None``/empty -> ``None``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        text = str(value).strip()
        if text == "" or text.lower() in ("null", "none", "all"):
            return None
        items = [t.strip() for t in text.split(",")]
    ids: list[int] = []
    for item in items:
        if item == "":
            continue
        try:
            ids.append(int(item))
        except ValueError:
            raise GPUSelectionError(f"invalid GPU id {item!r} (expected integers such as '0,1')") from None
    return ids or None


def parse_num_gpus(value: Any) -> int | None:
    """``all``/``None``/``-1``/``0`` -> ``None`` (= every visible GPU); otherwise a positive int."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "all", "auto", "null", "none", "max"):
            return None
        try:
            value = int(text)
        except ValueError:
            raise GPUSelectionError(f"distributed.num_gpus must be 'all' or an integer, got {value!r}") from None
    n = int(value)
    if n <= 0:
        return None
    return n


def visible_candidates(gpus: Sequence[GPUInfo]) -> list[GPUInfo]:
    """GPUs the process may use: honours an already exported ``CUDA_VISIBLE_DEVICES``."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        return list(gpus)
    try:
        wanted = parse_gpu_ids(env) or []
    except GPUSelectionError:
        # UUID-style entries (GPU-xxxx) or other opaque values: index -1 tells the
        # launcher to leave CUDA_VISIBLE_DEVICES untouched.
        return [GPUInfo(-1, str(tok.strip()), 0, 0) for tok in env.split(",") if tok.strip()]
    by_index = {g.index: g for g in gpus}
    picked = [by_index[i] for i in wanted if i in by_index]
    if not picked and wanted:
        # ids that nvidia-smi does not know (container remapping): keep them verbatim
        return [GPUInfo(i, "unknown", 0, 0) for i in wanted]
    return picked


def select_gpu_ids(
    num_gpus: Any = None,
    explicit_ids: Any = None,
    *,
    selection: str = "free",
    min_free_gb: float | None = None,
    gpus: Sequence[GPUInfo] | None = None,
) -> tuple[list[int], list[GPUInfo]]:
    """Decide which physical GPU ids a run uses.

    Args:
        num_gpus: ``all``/``None`` or a positive count.
        explicit_ids: explicit ids (``"0,1"``); when given they are used as-is
            (after validation) and ``num_gpus`` is ignored.
        selection: ``free`` (most free memory first) or ``first`` (natural order).
        min_free_gb: drop candidates with less free memory (ignored for explicit ids).
        gpus: pre-queried GPU list (tests); defaults to :func:`query_gpus`.

    Returns:
        ``(ids, infos)`` – the chosen physical ids in the order to export as
        ``CUDA_VISIBLE_DEVICES`` and their :class:`GPUInfo` records.
    """
    if selection not in SELECTION_MODES:
        raise GPUSelectionError(f"distributed.gpu_selection must be one of {SELECTION_MODES}, got {selection!r}")
    all_gpus = list(gpus) if gpus is not None else query_gpus()
    by_index = {g.index: g for g in all_gpus}

    ids = parse_gpu_ids(explicit_ids)
    if ids:
        missing = [i for i in ids if i not in by_index]
        if missing and all_gpus:
            raise GPUSelectionError(f"GPU ids {missing} do not exist (available: {sorted(by_index)})")
        infos = [by_index.get(i, GPUInfo(i, "unknown", 0, 0)) for i in ids]
        return list(ids), infos

    candidates = visible_candidates(all_gpus)
    if not candidates:
        return [], []
    if min_free_gb is not None and all(g.memory_total_mb > 0 for g in candidates):
        kept = [g for g in candidates if g.memory_free_gb >= float(min_free_gb)]
        if not kept:
            raise GPUSelectionError(
                f"no GPU has >= {min_free_gb} GB free (free GB: "
                f"{[(g.index, round(g.memory_free_gb, 1)) for g in candidates]})"
            )
        candidates = kept

    n = parse_num_gpus(num_gpus)
    if n is None or n >= len(candidates):
        if n is not None and n > len(candidates):
            logger.warning("distributed.num_gpus=%d but only %d GPU(s) available; using %d",
                           n, len(candidates), len(candidates))
        chosen = list(candidates)
    elif selection == "first":
        chosen = list(candidates[:n])
    else:
        ranked = sorted(candidates, key=lambda g: (-g.memory_free_mb, g.index))
        chosen = sorted(ranked[:n], key=lambda g: g.index)
    return [g.index for g in chosen], chosen


def describe(infos: Sequence[GPUInfo]) -> str:
    """Human-readable one-liner for logs."""
    if not infos:
        return "no GPU"
    return ", ".join(
        f"GPU{g.index}({g.memory_free_gb:.0f}/{g.memory_total_mb / 1024:.0f} GB free)" if g.memory_total_mb else f"GPU{g.index}"
        for g in infos
    )


__all__ = [
    "GPUInfo",
    "GPUSelectionError",
    "SELECTION_MODES",
    "describe",
    "parse_gpu_ids",
    "parse_num_gpus",
    "query_gpus",
    "select_gpu_ids",
    "visible_candidates",
]
