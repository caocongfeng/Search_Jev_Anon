"""Post-hoc temperature scaling for JEV restricted-menu logits.

The JEV read-out produces a vector of raw logits ``z`` over the option menu
(canonical order, masked slots set to ``-inf``).  Temperature scaling replaces
``softmax(z)`` by ``softmax(z / T)`` where the scalar ``T > 0`` is fitted on a
held-out calibration split by minimising the negative log-likelihood of the
gold option.  ``T`` does not change the arg-max, only the confidence, so
accuracy is unchanged while NLL / Brier / ECE usually improve.

The fit is performed on ``log T`` with L-BFGS, optionally one temperature per
question type (``choice`` / ``noul`` / ``score``) on top of a global one.  The
result is a small JSON-serialisable dict written to ``<run>/calibration.json``.

All functions accept ``torch.Tensor`` or ``numpy.ndarray`` inputs and run on
CPU; no GPU or model weights are required.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Question-type ids used by the dataset / collator (mirrors ``training.data``).
QTYPE_TO_ID: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}
ID_TO_QTYPE: dict[int, str] = {v: k for k, v in QTYPE_TO_ID.items()}
QTYPES: tuple[str, ...] = ("choice", "noul", "score")

#: Admissible temperature range (clamped after the fit).
T_MIN: float = 0.05
T_MAX: float = 50.0

#: Minimum number of examples required to fit a type-specific temperature;
#: below this the global temperature is reused for that type.
MIN_EXAMPLES_PER_TYPE: int = 10


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float_tensor(x: Any) -> torch.Tensor:
    """Convert a tensor / array / nested list of logits to a float32 CPU tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach().to("cpu", dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2:
        raise ValueError(f"logits must be [N, K]; got shape {tuple(t.shape)}")
    return t


def _as_long_tensor(x: Any) -> torch.Tensor:
    """Convert a tensor / array / list of integer ids to an int64 CPU tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(np.asarray(x).reshape(-1), dtype=torch.long)


def nll_at_temperature(logits: Any, target_idx: Any, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold option at a given temperature.

    Args:
        logits: ``[N, K]`` raw menu logits, masked slots ``-inf``.
        target_idx: ``[N]`` gold option index (canonical order).
        temperature: scalar temperature ``T``; probabilities are ``softmax(z / T)``.

    Returns:
        Mean NLL in nats (``nan`` when ``N == 0``).
    """
    z = _as_float_tensor(logits)
    y = _as_long_tensor(target_idx)
    if z.shape[0] == 0:
        return float("nan")
    logp = torch.log_softmax(z / float(temperature), dim=-1)
    return float(-logp.gather(1, y.view(-1, 1)).mean())


def _valid_rows(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Boolean mask of rows whose gold slot is finite and within range."""
    n, k = z.shape
    in_range = (y >= 0) & (y < k)
    ok = torch.zeros(n, dtype=torch.bool)
    if in_range.any():
        gold = z[in_range].gather(1, y[in_range].view(-1, 1)).squeeze(1)
        ok[in_range] = torch.isfinite(gold)
    # every row also needs at least one finite logit (otherwise softmax is nan)
    ok &= torch.isfinite(z).any(dim=1)
    return ok


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_temperature(
    logits: Any,
    target_idx: Any,
    init: float = 1.0,
    max_iter: int = 200,
) -> float:
    """Fit a single scalar temperature by minimising NLL with L-BFGS on ``log T``.

    Args:
        logits: ``[N, K]`` raw menu logits (canonical order, masked ``-inf``).
        target_idx: ``[N]`` gold indices.
        init: initial temperature.
        max_iter: maximum L-BFGS iterations.

    Returns:
        The fitted temperature, clamped to ``[T_MIN, T_MAX]``.  If the data are
        empty or the optimiser fails to improve on ``init``, ``init`` (clamped)
        is returned.
    """
    init = float(min(max(init, T_MIN), T_MAX))
    z = _as_float_tensor(logits)
    y = _as_long_tensor(target_idx)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"logits rows ({z.shape[0]}) != targets ({y.shape[0]})")
    keep = _valid_rows(z, y)
    n_bad = int((~keep).sum())
    if n_bad:
        logger.warning("fit_temperature: dropping %d rows with masked/invalid gold slot", n_bad)
        z, y = z[keep], y[keep]
    if z.shape[0] == 0:
        logger.warning("fit_temperature: no valid examples; returning T=%.3f", init)
        return init

    z = z.double()
    y = y.view(-1, 1)
    # masked (-inf) slots are zeroed before scaling and re-masked afterwards so
    # that no -inf enters the autograd graph (``-inf * exp(-log_t)`` -> nan grad)
    finite = torch.isfinite(z)
    z_finite = z.masked_fill(~finite, 0.0)
    log_t = torch.full((), math.log(init), dtype=torch.float64, requires_grad=True)
    lo, hi = math.log(T_MIN), math.log(T_MAX)

    def nll_of(lt: torch.Tensor) -> torch.Tensor:
        scaled = (z_finite * torch.exp(-lt)).masked_fill(~finite, float("-inf"))
        logp = torch.log_softmax(scaled, dim=-1)
        return -logp.gather(1, y).mean()

    optimiser = torch.optim.LBFGS(
        [log_t],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = nll_of(log_t)
        loss.backward()
        return loss

    with torch.enable_grad():
        try:
            optimiser.step(closure)
        except RuntimeError as exc:  # pragma: no cover - numerical corner cases
            logger.warning("fit_temperature: L-BFGS failed (%s); keeping T=%.3f", exc, init)
            return init

    with torch.no_grad():
        lt = float(log_t.clamp(lo, hi))
        if not math.isfinite(lt):
            logger.warning("fit_temperature: non-finite result; keeping T=%.3f", init)
            return init
        fitted = float(nll_of(torch.tensor(lt, dtype=torch.float64)))
        baseline = float(nll_of(torch.tensor(math.log(init), dtype=torch.float64)))
    if not math.isfinite(fitted) or fitted > baseline + 1e-9:
        logger.warning(
            "fit_temperature: fitted T=%.4f did not improve NLL (%.5f vs %.5f); keeping T=%.3f",
            math.exp(lt), fitted, baseline, init,
        )
        return init
    t = math.exp(lt)
    logger.info("fit_temperature: n=%d  T=%.4f  nll %.5f -> %.5f", z.shape[0], t, baseline, fitted)
    return float(t)


def fit_calibration(
    logits: Any,
    target_idx: Any,
    qtype_id: Any,
    per_question_type: bool = True,
) -> dict[str, Any]:
    """Fit global and (optionally) per-question-type temperatures.

    Args:
        logits: ``[N, K]`` raw menu logits, canonical order, masked ``-inf``.
        target_idx: ``[N]`` gold indices (canonical order).
        qtype_id: ``[N]`` question-type ids (``0=choice, 1=noul, 2=score``) or
            question-type names.
        per_question_type: fit a separate temperature per type.  Types with
            fewer than ``MIN_EXAMPLES_PER_TYPE`` examples reuse the global ``T``.

    Returns:
        ``{"global": T, "choice": T, "noul": T, "score": T,
        "n": {"total": N, "choice": n, ...}, "nll_before": .., "nll_after": ..,
        "per_type": {qtype: {"n", "nll_before", "nll_after"}},
        "per_question_type": bool, "t_min": .., "t_max": ..}``.
    """
    z = _as_float_tensor(logits)
    y = _as_long_tensor(target_idx)
    if isinstance(qtype_id, (list, tuple)) and qtype_id and isinstance(qtype_id[0], str):
        q = torch.tensor([QTYPE_TO_ID[str(s)] for s in qtype_id], dtype=torch.long)
    else:
        q = _as_long_tensor(qtype_id)
    n = int(z.shape[0])
    if not (y.shape[0] == n and q.shape[0] == n):
        raise ValueError("logits, target_idx and qtype_id must have the same length")

    t_global = fit_temperature(z, y) if n else 1.0
    out: dict[str, Any] = {
        "global": float(t_global),
        "n": {"total": n},
        "per_question_type": bool(per_question_type),
        "per_type": {},
        "t_min": T_MIN,
        "t_max": T_MAX,
    }

    # per-type temperatures
    for name in QTYPES:
        mask = q == QTYPE_TO_ID[name]
        n_t = int(mask.sum())
        out["n"][name] = n_t
        if per_question_type and n_t >= MIN_EXAMPLES_PER_TYPE:
            t_type = fit_temperature(z[mask], y[mask], init=t_global)
        else:
            if per_question_type and 0 < n_t < MIN_EXAMPLES_PER_TYPE:
                logger.info("calibration: only %d %s examples; using global T", n_t, name)
            t_type = t_global
        out[name] = float(t_type)
        if n_t:
            out["per_type"][name] = {
                "n": n_t,
                "temperature": float(t_type),
                "nll_before": nll_at_temperature(z[mask], y[mask], 1.0),
                "nll_after": nll_at_temperature(z[mask], y[mask], t_type),
            }

    # overall NLL before / after using the temperatures that will be applied
    if n:
        t_vec = torch.ones(n, dtype=torch.float32)
        for name in QTYPES:
            t_vec[q == QTYPE_TO_ID[name]] = float(out[name])
        keep = _valid_rows(z, y)
        zk, yk, tk = z[keep], y[keep].view(-1, 1), t_vec[keep].view(-1, 1)
        if zk.shape[0]:
            before = float(-torch.log_softmax(zk, -1).gather(1, yk).mean())
            after = float(-torch.log_softmax(zk / tk, -1).gather(1, yk).mean())
        else:
            before = after = float("nan")
    else:
        before = after = float("nan")
    out["nll_before"] = before
    out["nll_after"] = after
    logger.info(
        "calibration: n=%d global T=%.4f choice=%.4f noul=%.4f score=%.4f  nll %.5f -> %.5f",
        n, out["global"], out["choice"], out["noul"], out["score"], before, after,
    )
    return out


# --------------------------------------------------------------------------- #
# persistence / lookup
# --------------------------------------------------------------------------- #
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain Python types."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_calibration(d: Mapping[str, Any], path: str | Path) -> Path:
    """Write a calibration dict to ``path`` as JSON (parents created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(d), fh, ensure_ascii=False, indent=2)
    logger.info("saved calibration to %s", p)
    return p


def load_calibration(path: str | Path) -> dict[str, Any] | None:
    """Load a calibration dict; returns ``None`` if the file is missing or invalid."""
    p = Path(path)
    if p.is_dir():
        p = p / "calibration.json"
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read calibration file %s: %s", p, exc)
        return None
    if not isinstance(d, dict):
        logger.warning("calibration file %s does not hold a dict; ignoring", p)
        return None
    return d


def _positive_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def temperature_for(calib: Mapping[str, Any] | None, qtype: str) -> float:
    """Temperature to apply for ``qtype``: type-specific, else global, else ``1.0``."""
    if not calib:
        return 1.0
    for key in (qtype, "global"):
        t = _positive_float(calib.get(key))
        if t is not None:
            return t
    return 1.0


def apply_temperature(logits: Any, temperature: float) -> np.ndarray:
    """Return ``softmax(z / T)`` as a float64 numpy array (masked ``-inf`` -> 0)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
