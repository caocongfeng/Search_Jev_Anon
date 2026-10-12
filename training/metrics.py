"""Numpy-only evaluation metrics for JEV decisions.

All functions take plain ``numpy`` arrays. ``summarize`` is the single entry
point used by the trainer's ``compute_metrics`` and the final evaluation. Probabilities
passed to it must be in **canonical** option order (index 0 == ``"true"`` for
``noul``; index ``i`` == level ``i`` for ``score``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

#: Question-type ids used by the dataset / collator.
QTYPE_IDS: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}
QTYPE_NAMES: dict[int, str] = {v: k for k, v in QTYPE_IDS.items()}

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties replaced by their average rank."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = (cum - counts + 1 + cum) / 2.0
    return avg_rank[inverse.reshape(-1)]


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    """Macro-averaged F1 over the classes that appear in ``gold``.

    A class's F1 is ``2TP / (2TP + FP + FN)``; predictions of classes never seen
    in ``gold`` only count as false positives / false negatives of the gold classes.
    """
    pred = np.asarray(pred).reshape(-1)
    gold = np.asarray(gold).reshape(-1)
    if gold.size == 0:
        return float("nan")
    scores = []
    for c in np.unique(gold):
        tp = float(np.sum((pred == c) & (gold == c)))
        fp = float(np.sum((pred == c) & (gold != c)))
        fn = float(np.sum((pred != c) & (gold == c)))
        denom = 2.0 * tp + fp + fn
        scores.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged).

    Returns ``nan`` when only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with ``n_bins`` equal-width confidence bins on ``[0, 1]``."""
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = conf.size
    if n == 0:
        return float("nan")
    bins = np.minimum((np.clip(conf, 0.0, 1.0) * n_bins).astype(np.int64), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        m = int(sel.sum())
        if m == 0:
            continue
        total += abs(correct[sel].mean() - conf[sel].mean()) * m / n
    return float(total)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); ``nan`` if either input is constant."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _finite_or_none(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _prepare_probs(probs: np.ndarray, num_options: np.ndarray) -> np.ndarray:
    """Clean a ``[N, Kmax]`` probability matrix: drop padding (negative values and
    slots beyond ``num_options``), then renormalise each row."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    p = np.where(p < 0.0, 0.0, p)
    kmax = p.shape[1]
    valid = np.arange(kmax)[None, :] < np.asarray(num_options).reshape(-1, 1)
    p = np.where(valid, p, 0.0)
    row_sum = p.sum(axis=1, keepdims=True)
    p = np.where(row_sum > 0.0, p / np.maximum(row_sum, _EPS), p)
    return p


def _per_example(probs: np.ndarray, target_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-example statistics shared by every block."""
    n, kmax = probs.shape
    rows = np.arange(n)
    gold = np.asarray(target_idx, dtype=np.int64).reshape(-1)
    gold_c = np.clip(gold, 0, kmax - 1)
    p_gold = probs[rows, gold_c]
    onehot = np.zeros_like(probs)
    onehot[rows, gold_c] = 1.0
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    if kmax >= 2:
        top2 = np.partition(probs, kmax - 2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = conf.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, _EPS)), 0.0), axis=1)
    return {
        "gold": gold,
        "pred": pred,
        "correct": (pred == gold).astype(np.float64),
        "nll": -np.log(np.clip(p_gold, _EPS, None)),
        "brier": np.sum((probs - onehot) ** 2, axis=1),
        "conf": conf,
        "entropy": entropy,
        "margin": margin,
    }


def _block(stats: dict[str, np.ndarray], idx: np.ndarray, n_bins: int) -> dict[str, float | int | None]:
    """Overall-style metrics for the subset ``idx``."""
    n = int(idx.size)
    if n == 0:
        return {"n": 0}
    correct = stats["correct"][idx]
    conf = stats["conf"][idx]
    return {
        "n": n,
        "accuracy": _finite_or_none(correct.mean()),
        "nll": _finite_or_none(stats["nll"][idx].mean()),
        "brier": _finite_or_none(stats["brier"][idx].mean()),
        "ece": _finite_or_none(ece(conf, correct, n_bins)),
        "mean_confidence": _finite_or_none(conf.mean()),
        "mean_entropy": _finite_or_none(stats["entropy"][idx].mean()),
        "mean_margin": _finite_or_none(stats["margin"][idx].mean()),
    }


def _noul_extras(probs: np.ndarray, stats: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float | None]:
    p_true = probs[idx, 0]
    is_true = stats["gold"][idx] == 0
    return {"auroc": _finite_or_none(auroc(p_true, is_true))}


def _score_extras(probs: np.ndarray, stats: dict[str, np.ndarray], idx: np.ndarray, num_options: np.ndarray) -> dict[str, float | None]:
    sub = probs[idx]
    levels = np.arange(sub.shape[1], dtype=np.float64)[None, :]
    expected = (sub * levels).sum(axis=1)
    gold = stats["gold"][idx].astype(np.float64)
    pred = stats["pred"][idx].astype(np.float64)
    k = np.asarray(num_options)[idx].astype(np.float64)
    within1 = np.abs(pred - gold) <= 1.0
    return {
        "mae": _finite_or_none(np.abs(expected - gold).mean()),
        "mae_argmax": _finite_or_none(np.abs(pred - gold).mean()),
        "exact_acc": _finite_or_none(stats["correct"][idx].mean()),
        "within_1_acc": _finite_or_none(within1.mean()),
        "spearman": _finite_or_none(spearman(expected, gold)),
        "mean_expected_level": _finite_or_none(expected.mean()),
        "mean_levels": _finite_or_none(k.mean()),
    }


def summarize(
    probs: np.ndarray,
    target_idx: np.ndarray,
    qtype_id: np.ndarray,
    num_options: np.ndarray,
    datasets: Sequence[str] | None = None,
    n_bins: int = 15,
) -> dict:
    """Summarise predictions.

    Args:
        probs: ``[N, Kmax]`` probabilities in canonical option order (slots beyond
            ``num_options[i]`` are ignored; negative padding values are ignored).
        target_idx: ``[N]`` canonical gold index.
        qtype_id: ``[N]`` question-type id (``0`` choice, ``1`` noul, ``2`` score).
        num_options: ``[N]`` number of valid options per example.
        datasets: optional ``[N]`` dataset names for a per-dataset breakdown.
        n_bins: bins for ECE.

    Returns:
        ``{"overall": {...}, "per_qtype": {name: {...}}, "per_dataset": {name: {...}}}``.
        ``overall`` holds ``n, accuracy, nll, brier, ece, mean_confidence,
        mean_entropy, mean_margin``; ``noul`` adds ``auroc``; ``score`` adds
        ``mae, mae_argmax, exact_acc, within_1_acc, spearman``; per-dataset blocks
        add ``macro_f1`` and ``qtype`` (plus the type extras when the dataset is
        homogeneous in question type). Non-finite values are reported as ``None``.
    """
    target_idx = np.asarray(target_idx, dtype=np.int64).reshape(-1)
    qtype_id = np.asarray(qtype_id, dtype=np.int64).reshape(-1)
    num_options = np.asarray(num_options, dtype=np.int64).reshape(-1)
    n = int(target_idx.size)
    if n == 0:
        return {"overall": {"n": 0}, "per_qtype": {}, "per_dataset": {}}
    p = _prepare_probs(probs, num_options)
    if p.shape[0] != n or qtype_id.size != n or num_options.size != n:
        raise ValueError(
            f"Inconsistent lengths: probs {p.shape[0]}, target_idx {n}, qtype_id {qtype_id.size}, num_options {num_options.size}"
        )
    if datasets is not None and len(datasets) != n:
        raise ValueError(f"datasets has {len(datasets)} entries, expected {n}")

    stats = _per_example(p, target_idx)
    all_idx = np.arange(n)
    out: dict = {"overall": _block(stats, all_idx, n_bins), "per_qtype": {}, "per_dataset": {}}

    for qid in sorted(set(int(q) for q in qtype_id.tolist())):
        idx = np.nonzero(qtype_id == qid)[0]
        name = QTYPE_NAMES.get(qid, f"qtype{qid}")
        block = _block(stats, idx, n_bins)
        block.update(_type_extras(name, p, stats, idx, num_options))
        out["per_qtype"][name] = block

    if datasets is not None:
        ds = np.asarray([str(d) for d in datasets], dtype=object)
        for name in sorted(set(ds.tolist())):
            idx = np.nonzero(ds == name)[0]
            block = _block(stats, idx, n_bins)
            block["macro_f1"] = _finite_or_none(macro_f1(stats["pred"][idx], stats["gold"][idx]))
            qtypes = sorted(set(int(q) for q in qtype_id[idx].tolist()))
            if len(qtypes) == 1:
                qname = QTYPE_NAMES.get(qtypes[0], f"qtype{qtypes[0]}")
                block["qtype"] = qname
                block.update(_type_extras(qname, p, stats, idx, num_options))
            else:
                block["qtype"] = "mixed"
                block["qtypes"] = [QTYPE_NAMES.get(q, f"qtype{q}") for q in qtypes]
            out["per_dataset"][name] = block
    return out


def _type_extras(name: str, p: np.ndarray, stats: dict[str, np.ndarray], idx: np.ndarray, num_options: np.ndarray) -> dict:
    if idx.size == 0:
        return {}
    if name == "noul":
        return _noul_extras(p, stats, idx)
    if name == "score":
        return _score_extras(p, stats, idx, num_options)
    return {}


def flatten_summary(summary: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a ``summarize`` result into ``{key: number}`` (for Trainer logging).

    ``overall`` keys are emitted bare, per-type keys as ``<qtype>_<key>`` and
    per-dataset keys as ``ds_<dataset>_<key>``. ``None`` / non-numeric values are dropped.
    """
    flat: dict[str, float] = {}

    def _put(key: str, value: object) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            flat[prefix + key] = float(value)

    for k, v in (summary.get("overall") or {}).items():
        _put(k, v)
    for qname, block in (summary.get("per_qtype") or {}).items():
        for k, v in block.items():
            _put(f"{qname}_{k}", v)
    for dname, block in (summary.get("per_dataset") or {}).items():
        for k, v in block.items():
            _put(f"ds_{dname}_{k}", v)
    return flat


__all__ = [
    "QTYPE_IDS",
    "QTYPE_NAMES",
    "macro_f1",
    "auroc",
    "ece",
    "spearman",
    "summarize",
    "flatten_summary",
]
