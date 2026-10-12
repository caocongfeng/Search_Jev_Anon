"""Inference API for JEV models: :class:`JEVPredictor` and :class:`DecisionResult`.

A JEV decision is a single forward pass: the prompt (state + question + option
menu) is encoded with the same :class:`~training.prompting.PromptBuilder` used
during training, the model's last-position logits are restricted to the label
tokens ``A, B, C, ...`` (:func:`~training.modeling.restricted_menu_logits`),
mapped back to the canonical option order and turned into a probability
distribution with ``softmax(z / T)`` where ``T`` is the fitted temperature for
the question type (see :mod:`training.calibration`).

``JEVPredictor`` accepts a training run directory (adapter + ``calibration.json``
+ ``label_space.json`` + ``config.resolved.yaml``), a local model directory or a
plain Hugging Face model id (zero-shot).  Heavy work (loading weights) happens
only in ``__init__``; the module itself imports without a GPU.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import random

import numpy as np
import torch

from training.calibration import load_calibration, temperature_for
from training.labels import LabelSpace
from training.modeling import load_for_inference, restricted_menu_logits
from training.prompting import PromptBuilder
from training.schema import (
    NOUL_FALSE_TEXT,
    NOUL_TRUE_TEXT,
    DecisionExample,
    Option,
    jev_row_to_examples,
    render_state,
)

logger = logging.getLogger(__name__)

RUN_DIR_MARKERS: tuple[str, ...] = ("config.resolved.yaml", "label_space.json", "calibration.json")
_ENUMERATOR_KEY = re.compile(r"[A-Za-z]|\d{1,3}")


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class DecisionResult:
    """Outcome of one JEV decision (all option-indexed fields in canonical order)."""

    choice: str
    """Canonical key of the arg-max option."""
    choice_index: int
    """Canonical index of the arg-max option."""
    probabilities: dict[str, float]
    """Temperature-scaled probability per canonical option key (insertion order = canonical)."""
    confidence: float
    """Probability of the chosen option (``max p``)."""
    margin: float
    """``p_top1 - p_top2`` (``p_top1`` if there is a single option)."""
    entropy: float
    """Shannon entropy of the distribution in nats."""
    logits: list[float]
    """Raw restricted logits, canonical order (before temperature)."""
    temperature: float
    """Temperature used to produce ``probabilities``."""
    expected_score: float | None = None
    """``sum_i p_i * i`` for ``score`` questions, otherwise ``None``."""
    truncated: bool = False
    """Whether the state was truncated to fit the prompt budget."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form additional information (e.g. ``qtype``, ``example_id``)."""

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable dict."""
        return asdict(self)

    @property
    def prob_list(self) -> list[float]:
        """Probabilities as a list in canonical order."""
        return list(self.probabilities.values())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def locate_run_dir(path: str | Path) -> Path | None:
    """Return the training run directory for ``path`` if it looks like one.

    ``path`` may be the run dir itself or one of its sub-directories
    (``adapter/``, ``model/``, ``merged/``); anything else returns ``None``.
    """
    p = Path(path).expanduser()
    if not p.is_dir():
        return None
    if any((p / m).is_file() for m in RUN_DIR_MARKERS):
        return p
    if any((p.parent / m).is_file() for m in RUN_DIR_MARKERS):
        return p.parent
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # local import: optional at module level
    except ImportError:  # pragma: no cover
        logger.warning("pyyaml not available; ignoring %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    return d if isinstance(d, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    import json

    try:
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    return d if isinstance(d, dict) else {}


def softmax_with_temperature(logits: Sequence[float], temperature: float) -> np.ndarray:
    """Numerically stable ``softmax(z / T)`` in float64 (``-inf`` slots -> 0)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def build_options(
    options: Sequence[str] | Mapping[str, Any] | Sequence[Option] | None,
    question_type: str = "choice",
) -> list[Option]:
    """Normalise ad-hoc option specs into canonical :class:`Option` objects.

    Mirrors the rendering rules of :func:`training.schema.jev_row_to_examples`:

    * ``noul``: options are fixed (``true`` / ``false``); ``options`` is ignored.
    * ``score``: ``options`` is the ordered list of level descriptions; keys
      are ``"0".."K-1"`` and texts ``"{i}: {desc}"``.
    * ``choice``: a ``dict`` maps key -> description (text = ``key`` when the
      description is empty/equal to the key, = description when the key is a
      pure enumerator such as ``A`` or ``12``, else ``"key：desc"``); a list
      of strings uses each string as both key and text; :class:`Option`
      objects pass through unchanged.
    """
    qt = question_type.lower()
    if qt == "noul":
        return [Option("true", NOUL_TRUE_TEXT), Option("false", NOUL_FALSE_TEXT)]
    if options is None:
        raise ValueError(f"options are required for question_type={question_type!r}")
    if qt == "score":
        if isinstance(options, Mapping):
            levels = [str(v) for _, v in sorted(options.items(), key=lambda kv: int(kv[0]))]
        else:
            levels = [o.text if isinstance(o, Option) else str(o) for o in options]
        if len(levels) < 2:
            raise ValueError("score questions need at least two levels")
        return [Option(str(i), f"{i}: {desc}") for i, desc in enumerate(levels)]
    if qt != "choice":
        raise ValueError(f"unknown question_type {question_type!r}")
    out: list[Option] = []
    if isinstance(options, Mapping):
        for k, d in options.items():
            key = str(k)
            desc = "" if d is None else str(d)
            if desc == "" or desc == key:
                text = key
            elif _ENUMERATOR_KEY.fullmatch(key):
                text = desc
            else:
                text = f"{key}：{desc}"
            out.append(Option(key, text))
    else:
        for o in options:
            if isinstance(o, Option):
                out.append(o)
            else:
                s = str(o)
                out.append(Option(s, s))
    if not out:
        raise ValueError("empty option list")
    keys = [o.key for o in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate option keys: {keys}")
    return out


def _qid_of(example_id: str, sample_id: str) -> str:
    prefix = f"{sample_id}#"
    if example_id.startswith(prefix):
        return example_id[len(prefix):]
    return example_id.rsplit("#", 1)[-1]


# --------------------------------------------------------------------------- #
# predictor
# --------------------------------------------------------------------------- #
class JEVPredictor:
    """Batched JEV inference on a trained run, a local model dir or an HF model id.

    Args:
        model_or_run: run directory produced by ``start_training.py`` (or its
            ``adapter/`` sub-dir), a local HF model directory, or an HF id
            such as ``Qwen/Qwen3.5-4B`` (zero-shot).
        base_model: base model to attach a LoRA adapter to (defaults to the
            adapter's recorded base).
        device: ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``; falls back to CPU when
            CUDA is unavailable.
        dtype: weight dtype name (``bfloat16`` recommended).
        batch_size: examples per forward pass.
        max_length: total prompt token budget (state is truncated to fit).
        calibration: ``True`` -> use ``<run>/calibration.json`` if present,
            ``False`` -> ``T = 1``, or an explicit calibration dict.
        label_alphabet: label alphabet name or explicit list (overridden by
            ``<run>/label_space.json`` when present).
        prompt_style / prompt_language: prompt options; a run's
            ``config.resolved.yaml`` supplies them when the caller keeps the
            defaults.
        merge_lora: merge the adapter into the base weights after loading.
    """

    _DEFAULTS = {"prompt_style": "auto", "prompt_language": "bilingual", "label_alphabet": "extended"}

    def __init__(
        self,
        model_or_run: str,
        *,
        base_model: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 16,
        max_length: int = 2048,
        calibration: bool | Mapping[str, Any] = True,
        label_alphabet: str | list[str] = "extended",
        prompt_style: str = "auto",
        prompt_language: str = "bilingual",
        merge_lora: bool = False,
        max_batch_tokens: int | None = None,
    ) -> None:
        self.model_or_run = str(model_or_run)
        self.run_dir = locate_run_dir(model_or_run)
        self.batch_size = max(1, int(batch_size))
        # token budget per forward pass (padded tokens); None = fixed ``batch_size`` batches. With a budget,
        # ``batch_size`` is the upper bound on examples per batch and short prompts are packed into bigger batches.
        self.max_batch_tokens = None if not max_batch_tokens else max(1, int(max_batch_tokens))
        self.max_length = int(max_length)
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available; running JEVPredictor on CPU")
            device = "cpu"

        # ---- settings recorded by the training run --------------------------
        run_cfg: dict[str, Any] = {}
        label_space_meta: dict[str, Any] = {}
        if self.run_dir is not None:
            cfg_path = self.run_dir / "config.resolved.yaml"
            if cfg_path.is_file():
                run_cfg = _load_yaml(cfg_path)
            ls_path = self.run_dir / "label_space.json"
            if ls_path.is_file():
                label_space_meta = _load_json(ls_path)
        prompt_cfg = run_cfg.get("prompt") or {}
        labels_cfg = run_cfg.get("labels") or {}
        if prompt_style == self._DEFAULTS["prompt_style"] and prompt_cfg.get("style"):
            prompt_style = str(prompt_cfg["style"])
        if prompt_language == self._DEFAULTS["prompt_language"] and prompt_cfg.get("language"):
            prompt_language = str(prompt_cfg["language"])
        max_options: int | None = None
        if label_space_meta.get("labels"):
            label_alphabet = [str(x) for x in label_space_meta["labels"]]
            max_options = label_space_meta.get("max_options")
        elif label_alphabet == self._DEFAULTS["label_alphabet"] and labels_cfg.get("alphabet"):
            label_alphabet = labels_cfg["alphabet"]
            max_options = labels_cfg.get("max_options")
        self.prompt_style = prompt_style
        self.prompt_language = prompt_language

        # ---- model / tokenizer ----------------------------------------------
        logger.info("loading JEV model from %s (device=%s, dtype=%s)", model_or_run, device, dtype)
        self.model, self.tokenizer, run_calib = load_for_inference(
            self.model_or_run, base_model=base_model, dtype=dtype, device=device, merge=merge_lora
        )
        self.model.eval()
        if getattr(self.tokenizer, "padding_side", "left") != "left":
            self.tokenizer.padding_side = "left"
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            self.device = torch.device(device)

        # ---- calibration -----------------------------------------------------
        if isinstance(calibration, Mapping):
            self.calibration: dict[str, Any] | None = dict(calibration)
        elif calibration:
            self.calibration = dict(run_calib) if run_calib else None
            if self.calibration is None and self.run_dir is not None:
                self.calibration = load_calibration(self.run_dir / "calibration.json")
            if self.calibration is None:
                logger.info("no calibration found; using T=1.0")
        else:
            self.calibration = None

        # ---- label space / prompt builder -----------------------------------
        self.label_space = LabelSpace(self.tokenizer, alphabet=label_alphabet, max_options=max_options)
        self.prompt_builder = PromptBuilder(
            self.tokenizer,
            self.label_space,
            style=prompt_style,
            language=prompt_language,
            max_length=self.max_length,
        )
        self.pad_token_id: int = int(
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.n_truncated = 0
        logger.info(
            "JEVPredictor ready: labels=%d max_options=%d style=%s language=%s calibration=%s",
            len(self.label_space.labels), self.label_space.max_options, prompt_style, prompt_language,
            "yes" if self.calibration else "no",
        )

    # ------------------------------------------------------------------ core
    def _encode(self, ex: DecisionExample, perm: list[int] | None) -> dict[str, Any]:
        enc = self.prompt_builder.encode(ex, perm)
        if enc.get("truncated"):
            self.n_truncated += 1
        return enc

    @torch.inference_mode()
    def _forward(self, encoded: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        """Run one padded batch; returns per-example display-order logits (length ``K``)."""
        b = len(encoded)
        max_len = max(len(e["input_ids"]) for e in encoded)
        ks = [int(e["num_options"]) for e in encoded]
        k_max = max(ks)
        input_ids = torch.full((b, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        label_ids = torch.zeros((b, k_max), dtype=torch.long)
        for i, e in enumerate(encoded):
            ids = e["input_ids"]
            n = len(ids)
            input_ids[i, max_len - n:] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - n:] = 1
            label_ids[i, : ks[i]] = torch.as_tensor(self.label_space.ids_for(ks[i]), dtype=torch.long)
        menu = restricted_menu_logits(
            self.model,
            input_ids.to(self.device),
            attention_mask.to(self.device),
            label_ids.to(self.device),
        )
        menu = menu.detach().float().cpu()
        return [menu[i, : ks[i]].tolist() for i in range(b)]

    def _display_logits(
        self, examples: Sequence[DecisionExample], perms: Sequence[list[int] | None] | None
    ) -> tuple[list[list[float]], list[bool]]:
        """Display-order logits (and truncation flags) for every example, batched by prompt length."""
        n = len(examples)
        if n == 0:
            return [], []
        if perms is None:
            perms = [None] * n
        if len(perms) != n:
            raise ValueError("perms must have one entry per example")
        encoded = [self._encode(ex, perm) for ex, perm in zip(examples, perms)]
        truncated = [bool(e.get("truncated", False)) for e in encoded]
        # longest first: an OOM surfaces on the first batch, and padding waste is minimal
        order = sorted(range(n), key=lambda i: -len(encoded[i]["input_ids"]))
        out: list[list[float] | None] = [None] * n
        for idx in self._batches(order, [len(e["input_ids"]) for e in encoded]):
            logits = self._forward([encoded[i] for i in idx])
            for i, lg in zip(idx, logits):
                out[i] = lg
        return [lg for lg in out if lg is not None], truncated

    def _batches(self, order: Sequence[int], lengths: Sequence[int]) -> Iterator[list[int]]:
        """Split length-sorted (longest first) indices into batches: fixed ``batch_size``, or packed up to
        ``max_batch_tokens`` padded tokens (the first item of a batch is its longest) and at most ``batch_size``."""
        if not self.max_batch_tokens:
            for start in range(0, len(order), self.batch_size):
                yield list(order[start : start + self.batch_size])
            return
        batch: list[int] = []
        width = 0
        for i in order:
            w = max(width, int(lengths[i]))
            if batch and (len(batch) >= self.batch_size or w * (len(batch) + 1) > self.max_batch_tokens):
                yield batch
                batch, w = [], int(lengths[i])
            batch.append(i)
            width = w
        if batch:
            yield batch

    @staticmethod
    def _unpermute(display: Sequence[float], perm: list[int] | None) -> list[float]:
        """Map display-order values back to canonical order (``perm[j]`` = canonical index at slot ``j``)."""
        if perm is None:
            return list(display)
        canon = [0.0] * len(display)
        for j, c in enumerate(perm):
            canon[c] = float(display[j])
        return canon

    # ------------------------------------------------------------------ API
    def raw_menu_logits(
        self, examples: Sequence[DecisionExample], perms: Sequence[list[int] | None] | None = None
    ) -> list[list[float]]:
        """Raw restricted logits per example in **canonical** option order (no temperature)."""
        return self._canonical_logits(examples, perms)[0]

    def _canonical_logits(
        self, examples: Sequence[DecisionExample], perms: Sequence[list[int] | None] | None
    ) -> tuple[list[list[float]], list[bool]]:
        display, truncated = self._display_logits(examples, perms)
        if perms is None:
            return display, truncated
        return [self._unpermute(d, p) for d, p in zip(display, perms)], truncated

    def _temperature(self, ex: DecisionExample, temperature: float | None,
                     type_temperature: Mapping[str, float] | None = None) -> float:
        """Effective temperature of ``ex``: explicit ``temperature``, else the calibrated one for its qtype,
        multiplied by ``type_temperature[qtype]`` when given (a factor < 1 sharpens that question type)."""
        t = float(temperature) if temperature is not None else temperature_for(self.calibration, ex.qtype)
        if type_temperature and ex.qtype in type_temperature:
            t *= float(type_temperature[ex.qtype])
        return t

    def _make_result(
        self, ex: DecisionExample, logits_canon: Sequence[float], temperature: float | None, truncated: bool = False,
        type_temperature: Mapping[str, float] | None = None, probs: Sequence[float] | None = None,
    ) -> DecisionResult:
        t = self._temperature(ex, temperature, type_temperature)
        probs = softmax_with_temperature(logits_canon, t) if probs is None else np.asarray(probs, dtype=np.float64)
        idx = int(np.argmax(probs))
        srt = np.sort(probs)[::-1]
        margin = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        expected = float(np.dot(probs, np.arange(len(probs)))) if ex.qtype == "score" else None
        return DecisionResult(
            choice=ex.options[idx].key,
            choice_index=idx,
            probabilities={o.key: float(p) for o, p in zip(ex.options, probs)},
            confidence=float(probs[idx]),
            margin=margin,
            entropy=entropy,
            logits=[float(x) for x in logits_canon],
            temperature=t,
            expected_score=expected,
            truncated=truncated,
            extra={"example_id": ex.id, "qtype": ex.qtype, "dataset": ex.dataset},
        )

    def decide_batch(
        self,
        examples: Sequence[DecisionExample],
        perms: Sequence[list[int] | None] | None = None,
        temperature: float | None = None,
        type_temperature: Mapping[str, float] | None = None,
    ) -> list[DecisionResult]:
        """Decide many examples; results are aligned with ``examples`` and canonical-ordered.

        Prompts are sorted by length for efficient padding, run in batches of
        ``batch_size`` and the original order is restored.  ``perms`` (one
        display permutation per example, ``None`` = identity) only affects the
        menu shown to the model; probabilities are un-permuted. ``type_temperature``
        multiplies the calibrated temperature per question type (``{"noul": 0.25}``
        sharpens binary answers, e.g. for a 0.2/0.8 abstention rule).
        """
        logits, truncated = self._canonical_logits(examples, perms)
        return [
            self._make_result(ex, lg, temperature, truncated=tr, type_temperature=type_temperature)
            for ex, lg, tr in zip(examples, logits, truncated)
        ]

    def decide_batch_tta(
        self,
        examples: Sequence[DecisionExample],
        trials: int = 4,
        seed: int = 0,
        temperature: float | None = None,
        type_temperature: Mapping[str, float] | None = None,
    ) -> list[DecisionResult]:
        """Test-time augmentation: average the probabilities over the identity menu plus ``trials`` random
        option permutations (``choice`` / ``noul`` questions; ``score`` menus are ordered and never permuted).

        Averaging in canonical order removes position bias of the displayed menu; ``trials`` extra forward
        passes per example. Deterministic for a given ``seed``. ``logits`` of the result is the mean canonical logit.
        """
        n = len(examples)
        base_logits, truncated = self._canonical_logits(examples, None)
        sums = [softmax_with_temperature(lg, self._temperature(ex, temperature, type_temperature))
                for ex, lg in zip(examples, base_logits)]
        logit_sums = [np.asarray(lg, dtype=np.float64) for lg in base_logits]
        counts = [1] * n
        for trial in range(1, int(trials) + 1):
            perms: list[list[int] | None] = []
            idx: list[int] = []
            for i, ex in enumerate(examples):
                k = len(ex.options)
                if ex.qtype == "score" or k < 2:
                    perms.append(None)
                    continue
                rng = random.Random(f"jev-tta:{seed}:{trial}:{ex.id}")
                p = list(range(k))
                rng.shuffle(p)
                perms.append(p)
                idx.append(i)
            if not idx:
                break
            sub = [examples[i] for i in idx]
            sub_perms = [perms[i] for i in idx]
            logits, _ = self._canonical_logits(sub, sub_perms)
            for i, lg in zip(idx, logits):
                sums[i] = sums[i] + softmax_with_temperature(lg, self._temperature(examples[i], temperature, type_temperature))
                logit_sums[i] = logit_sums[i] + np.asarray(lg, dtype=np.float64)
                counts[i] += 1
        out = []
        for i, ex in enumerate(examples):
            probs = sums[i] / counts[i]
            out.append(self._make_result(ex, (logit_sums[i] / counts[i]).tolist(), temperature, truncated=truncated[i],
                                         type_temperature=type_temperature, probs=probs))
            out[-1].extra["tta_trials"] = counts[i] - 1
        return out

    def decide(
        self,
        state: Any,
        question: str,
        options: Sequence[str] | Mapping[str, Any] | Sequence[Option] | None = None,
        question_type: str = "choice",
        temperature: float | None = None,
    ) -> DecisionResult:
        """Decide a single ad-hoc question.

        Args:
            state: raw state (string, dict, list or JSON string); rendered with
                :func:`training.schema.render_state`.
            question: the instructions text.
            options: see :func:`build_options` (ignored for ``noul``).
            question_type: ``choice`` | ``noul`` | ``score``.
            temperature: override the calibrated temperature.
        """
        qt = question_type.lower()
        if qt not in ("choice", "noul", "score"):
            raise ValueError(f"unknown question_type {question_type!r}")
        opts = build_options(options, qt)
        if len(opts) > self.label_space.max_options:
            raise ValueError(f"{len(opts)} options exceed the label space ({self.label_space.max_options})")
        state_text = render_state(state)
        if not state_text:
            raise ValueError("empty state")
        ex = DecisionExample(
            id="adhoc#q",
            dataset="adhoc",
            family="",
            task="",
            split="",
            qtype=qt,
            state=state_text,
            question=str(question),
            options=opts,
            target_idx=None,
            target_dist=None,
            meta={},
        )
        return self.decide_batch([ex], temperature=temperature)[0]

    def answer_jev(self, row: Mapping[str, Any], family: str = "") -> dict[str, Any]:
        """Answer every question of one JEV API row.

        Returns ``{"answers": {qid: TypeSafe-style answer}, "results": {qid:
        DecisionResult}, "dropped": {reason: n}}``.  Answer formats:

        * ``noul``  -> ``{"type": "noul", "noul": P(true)}``
        * ``choice`` -> ``{"type": "choice", "choice": key, "confidence": c, "probabilities": {key: p}}``
        * ``score`` -> ``{"type": "score", "score": E[level], "confidence": c,
          "legend": {"i": desc}, "probabilities": {"i": p}}``
        """
        row = dict(row)
        examples, drops = jev_row_to_examples(
            row, family=family, label_space=self.label_space, require_answer=False
        )
        results = self.decide_batch(examples) if examples else []
        sample_id = str(row.get("sample_id", ""))
        questions = row.get("questions") or {}
        answers: dict[str, Any] = {}
        by_qid: dict[str, DecisionResult] = {}
        for ex, res in zip(examples, results):
            qid = _qid_of(ex.id, sample_id)
            by_qid[qid] = res
            if ex.qtype == "noul":
                answers[qid] = {"type": "noul", "noul": float(res.probabilities.get("true", 0.0))}
            elif ex.qtype == "choice":
                answers[qid] = {
                    "type": "choice",
                    "choice": res.choice,
                    "confidence": res.confidence,
                    "probabilities": dict(res.probabilities),
                }
            else:  # score
                criteria = (questions.get(qid) or {}).get("criteria")
                if isinstance(criteria, list):
                    legend = {str(i): str(d) for i, d in enumerate(criteria)}
                else:
                    legend = {o.key: o.text for o in ex.options}
                answers[qid] = {
                    "type": "score",
                    "score": float(res.expected_score if res.expected_score is not None else res.choice_index),
                    "confidence": res.confidence,
                    "legend": legend,
                    "probabilities": {str(k): float(v) for k, v in res.probabilities.items()},
                }
        if drops and any(drops.values()):
            logger.debug("answer_jev(%s): dropped questions %s", sample_id, drops)
        return {"answers": answers, "results": by_qid, "dropped": dict(drops)}

    def iter_decide(
        self, examples: Iterable[DecisionExample], chunk: int = 1024
    ) -> Iterable[tuple[DecisionExample, DecisionResult]]:
        """Stream ``(example, result)`` pairs, deciding ``chunk`` examples at a time."""
        buf: list[DecisionExample] = []
        for ex in examples:
            buf.append(ex)
            if len(buf) >= chunk:
                yield from zip(buf, self.decide_batch(buf))
                buf = []
        if buf:
            yield from zip(buf, self.decide_batch(buf))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"JEVPredictor(model={self.model_or_run!r}, device={self.device}, "
            f"labels={len(self.label_space.labels)}, calibrated={bool(self.calibration)})"
        )


__all__ = [
    "DecisionResult",
    "JEVPredictor",
    "build_options",
    "locate_run_dir",
    "softmax_with_temperature",
]
