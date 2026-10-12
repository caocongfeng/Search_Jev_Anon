"""Core representation: options, decision examples and the JEV-row adapter.

One JEV API row (``sample_id``, ``state``, ``questions``, ``answers``) can hold
several questions that share one state.  Each question becomes its own
:class:`DecisionExample` with a *menu* of :class:`Option` objects in canonical
order, an optional gold index and an optional soft target distribution.

The three TypeSafe question types are mapped as follows:

* ``choice``: options are the ``criteria`` dict entries in insertion order.
* ``noul`` (binary proposition): a fixed two-option menu ``true`` / ``false``.
* ``score`` (ordered scale): one option per level, key ``"0"``..``"K-1"``.

This module has no torch/transformers dependency.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .labels import LabelSpace

logger = logging.getLogger(__name__)

QTYPES: tuple[str, ...] = ("choice", "noul", "score")
QTYPE_IDS: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}
#: Informational copy; the authoritative whitelist is ``training.config.FAMILIES``.
FAMILIES: tuple[str, ...] = ("classification", "ranking", "search", "unified_v2")

NOUL_TRUE_KEY = "true"
NOUL_FALSE_KEY = "false"
NOUL_TRUE_TEXT = "成立 / True"
NOUL_FALSE_TEXT = "不成立 / False"

DROP_REASONS: tuple[str, ...] = ("over_limit", "bad_answer", "empty_state", "bad_question", "bad_row")

_ENUMERATOR_RE = re.compile(r"[A-Za-z]|\d{1,3}")
_LONG_VALUE_CHARS = 80
_META_STR_MAX = 200
_META_LIST_MAX = 8


class OverLimitPolicy(str, enum.Enum):
    """What to do with a question whose menu is larger than ``LabelSpace.max_options``."""

    DROP = "drop"
    SUBSAMPLE = "subsample"

    @classmethod
    def coerce(cls, value: "OverLimitPolicy | str | None") -> "OverLimitPolicy":
        """Accept an enum member, its string value or ``None`` (-> ``DROP``)."""
        if value is None:
            return cls.DROP
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown over_limit_policy {value!r}; expected one of {[p.value for p in cls]}"
            ) from exc


@dataclass
class Option:
    """One menu entry: canonical ``key`` from the source and the ``text`` shown in the prompt."""

    key: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "text": self.text}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Option":
        return cls(key=str(d["key"]), text=str(d.get("text", d["key"])))


@dataclass
class DecisionExample:
    """A single decision: rendered state, question, canonical option menu and target.

    ``target_dist`` (when not ``None``) is a probability vector over ``options``
    in canonical order; ``None`` means a one-hot target at ``target_idx``.
    ``target_idx`` is ``None`` for unlabeled examples (inference).
    """

    id: str
    dataset: str
    family: str
    task: str
    split: str
    qtype: str
    state: str
    question: str
    options: list[Option]
    target_idx: int | None = None
    target_dist: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def keys(self) -> list[str]:
        """Canonical option keys."""
        return [o.key for o in self.options]

    @property
    def target_key(self) -> str | None:
        """Canonical key of the gold option, if labeled."""
        if self.target_idx is None:
            return None
        return self.options[self.target_idx].key

    def dense_target(self) -> list[float] | None:
        """Target distribution as a dense list (one-hot if only ``target_idx`` is set)."""
        if self.target_dist is not None:
            return list(self.target_dist)
        if self.target_idx is None:
            return None
        vec = [0.0] * len(self.options)
        vec[self.target_idx] = 1.0
        return vec

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["options"] = [o.to_dict() for o in self.options]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DecisionExample":
        target_idx = d.get("target_idx")
        target_dist = d.get("target_dist")
        return cls(
            id=str(d["id"]),
            dataset=str(d.get("dataset", "")),
            family=str(d.get("family", "")),
            task=str(d.get("task", "")),
            split=str(d.get("split", "")),
            qtype=str(d["qtype"]),
            state=str(d.get("state", "")),
            question=str(d.get("question", "")),
            options=[Option.from_dict(o) for o in d["options"]],
            target_idx=None if target_idx is None else int(target_idx),
            target_dist=None if target_dist is None else [float(x) for x in target_dist],
            meta=dict(d.get("meta") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "DecisionExample":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------- state text
def _is_chat_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(m, Mapping) and "role" in m and "content" in m for m in value)
    )


def _render_chat(messages: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"[{m.get('role')}] {content.strip()}")
    return "\n".join(lines)


def _render_value(value: Any) -> str:
    """Render a nested value as text (strings verbatim, chat lists as lines, else JSON)."""
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=False)
    if _is_chat_list(value):
        return _render_chat(value)
    return json.dumps(value, ensure_ascii=False)


def _render_dict(d: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for key, value in d.items():
        text = _render_value(value)
        if "\n" in text or len(text) > _LONG_VALUE_CHARS:
            lines.append(f"{key}:\n{text}")
        else:
            lines.append(f"{key}: {text}")
    return "\n".join(lines)


def _render_list(items: Sequence[Any]) -> str:
    if _is_chat_list(items):
        return _render_chat(items)
    return "\n".join(_render_value(x) for x in items)


def render_state(state: Any) -> str:
    """Render a JEV ``state`` (str / dict / list / JSON-encoded string) as prompt text.

    * ``str``: if it starts with ``{`` or ``[`` and parses as JSON into a dict or
      list, that structure is rendered; otherwise the stripped string is used.
    * ``dict``: one ``key: value`` line per entry (insertion order); multi-line
      or long (> 80 chars) values go on the following lines.
    * chat lists (``[{"role", "content"}, ...]``) render as ``[role] content`` lines.
    * other nested values are JSON-encoded.

    Returns an empty string for empty/None states (callers drop those).
    """
    if state is None:
        return ""
    if isinstance(state, str):
        text = state.strip()
        if text[:1] in ("{", "["):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return render_state(parsed)
        return text
    if isinstance(state, Mapping):
        return _render_dict(state).strip()
    if isinstance(state, (list, tuple)):
        return _render_list(state).strip()
    return _render_value(state).strip()


# ---------------------------------------------------------------- options
def option_text(key: str, description: Any) -> str:
    """Display text for a ``choice`` option following the spec rules.

    * empty description or description == key -> the key;
    * pure enumerator key (single letter or 1-3 digits) -> the description;
    * otherwise ``"{key}：{description}"``.
    """
    k = str(key)
    if description is None:
        d = ""
    elif isinstance(description, str):
        d = description.strip()
    else:
        d = json.dumps(description, ensure_ascii=False) if isinstance(description, (dict, list)) else str(description)
    if d == "" or d == k:
        return k
    if _ENUMERATOR_RE.fullmatch(k):
        return d
    return f"{k}：{d}"


def noul_options(criteria: Any = None) -> list[Option]:
    """The two-option menu for binary propositions (index 0 == true).

    Keys are always ``true`` / ``false``. When the question carries ``criteria`` ``{"true": desc, "false": desc}``
    (e.g. the unified_v2 sentiment sets), the descriptions are appended to the option texts so the
    prompt says what "true" means; without criteria the menu is the fixed ``成立 / True`` / ``不成立 / False``.
    """
    t_desc = f_desc = ""
    if isinstance(criteria, Mapping):
        low = {str(k).strip().lower(): v for k, v in criteria.items()}
        t_desc = _render_value(low.get(NOUL_TRUE_KEY)).strip() if low.get(NOUL_TRUE_KEY) not in (None, "") else ""
        f_desc = _render_value(low.get(NOUL_FALSE_KEY)).strip() if low.get(NOUL_FALSE_KEY) not in (None, "") else ""
    return [Option(NOUL_TRUE_KEY, f"{NOUL_TRUE_TEXT}：{t_desc}" if t_desc else NOUL_TRUE_TEXT),
            Option(NOUL_FALSE_KEY, f"{NOUL_FALSE_TEXT}：{f_desc}" if f_desc else NOUL_FALSE_TEXT)]


def example_weight(meta: Mapping[str, Any] | None, rules: Mapping[str, Any] | None) -> float:
    """Training-loss weight of an example from its row meta (label reliability); 1.0 without rules.

    ``rules`` (``loss.sample_weights``): ``weak: w`` applies to ``meta.weak == true``; any other key maps a meta
    field to ``{value: w}`` (e.g. ``label_tier: {llm_teacher: 0.5}``). The smallest matching weight wins.
    """
    if not rules or not meta:
        return 1.0
    w = 1.0
    for key, rule in rules.items():
        if key == "weak":
            if meta.get("weak") is True and rule is not None:
                w = min(w, float(rule))
        elif isinstance(rule, Mapping):
            val = meta.get(key)
            if val is not None and str(val) in rule:
                w = min(w, float(rule[str(val)]))
    return w


def choice_options(criteria: Mapping[str, Any]) -> list[Option]:
    """Menu for a ``choice`` question from its criteria dict (insertion order)."""
    return [Option(str(k), option_text(str(k), v)) for k, v in criteria.items()]


def score_options(criteria: Sequence[Any]) -> list[Option]:
    """Menu for a ``score`` question: ``"i"`` -> ``"i: <level description>"``."""
    return [Option(str(i), f"{i}: {_render_value(desc)}") for i, desc in enumerate(criteria)]


def options_from_spec(options: Sequence[str] | Mapping[str, Any] | Sequence[Option], qtype: str) -> list[Option]:
    """Build a menu from user-provided options (used by the inference API).

    ``choice``: a dict ``{key: description}`` or a list of keys/texts;
    ``score``: a list of level descriptions; ``noul``: options are ignored.
    """
    if qtype == "noul":
        return noul_options()
    if isinstance(options, Mapping):
        if qtype == "score":
            return score_options([options[k] for k in options])
        return choice_options(options)
    opts = list(options)
    if opts and all(isinstance(o, Option) for o in opts):
        return list(opts)  # type: ignore[arg-type]
    if qtype == "score":
        return score_options(opts)
    return [Option(str(o), str(o)) for o in opts]


def make_example(
    state: Any,
    question: str,
    options: Sequence[str] | Mapping[str, Any] | Sequence[Option] | None,
    qtype: str = "choice",
    *,
    example_id: str = "",
    dataset: str = "custom",
    family: str = "",
    task: str = "",
    split: str = "",
    target_key: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> DecisionExample:
    """Convenience constructor for ad-hoc decisions (see ``JEVPredictor.decide``)."""
    if qtype not in QTYPES:
        raise ValueError(f"unknown question type {qtype!r}; expected one of {QTYPES}")
    opts = options_from_spec(options or [], qtype)
    if len(opts) < 1:
        raise ValueError("a decision needs at least one option")
    state_text = render_state(state)
    if not example_id:
        digest = hashlib.sha1((state_text + "\x00" + question).encode("utf-8")).hexdigest()[:12]
        example_id = f"{dataset}:{digest}#q"
    target_idx: int | None = None
    if target_key is not None:
        keys = [o.key for o in opts]
        if str(target_key) not in keys:
            raise ValueError(f"target_key {target_key!r} not among options {keys}")
        target_idx = keys.index(str(target_key))
    return DecisionExample(
        id=example_id,
        dataset=dataset,
        family=family,
        task=task,
        split=split,
        qtype=qtype,
        state=state_text,
        question=str(question).strip(),
        options=opts,
        target_idx=target_idx,
        target_dist=None,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------- targets
def soft_distribution(probabilities: Any, keys: Sequence[str]) -> list[float] | None:
    """Return a normalised soft target over ``keys`` if ``probabilities`` is a genuine soft label.

    A label is considered soft only when at least one value lies strictly in
    (0, 1); one-hot ``{key: 0.0/1.0}`` dicts return ``None`` (dense one-hot is
    reconstructed from ``target_idx`` later, which keeps the jsonl small).
    """
    if not isinstance(probabilities, Mapping) or not probabilities:
        return None
    try:
        values = {str(k): float(v) for k, v in probabilities.items()}
    except (TypeError, ValueError):
        return None
    if not any(0.0 < v < 1.0 for v in values.values()):
        return None
    dist = [max(values.get(k, 0.0), 0.0) for k in keys]
    total = sum(dist)
    if total <= 0.0:
        return None
    return [v / total for v in dist]


def _compact_meta(meta: Any) -> dict[str, Any]:
    """Keep only small JSON-serialisable scalars / short scalar lists from a row's meta."""
    out: dict[str, Any] = {}
    if not isinstance(meta, Mapping):
        return out
    for key, value in meta.items():
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            out[str(key)] = value
        elif isinstance(value, str):
            out[str(key)] = value[:_META_STR_MAX]
        elif isinstance(value, (list, tuple)) and len(value) <= _META_LIST_MAX and all(
            isinstance(x, (str, int, float, bool)) or x is None for x in value
        ):
            out[str(key)] = [x[:_META_STR_MAX] if isinstance(x, str) else x for x in value]
    return out


def _default_rng(example_id: str) -> random.Random:
    seed = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _subsample_options(ex: DecisionExample, max_options: int, rng: random.Random) -> None:
    """In place: keep the gold option plus a random subset of the others (canonical order kept)."""
    assert ex.target_idx is not None
    k = len(ex.options)
    others = [i for i in range(k) if i != ex.target_idx]
    chosen = sorted(rng.sample(others, max_options - 1) + [ex.target_idx])
    dist = ex.target_dist
    ex.options = [ex.options[i] for i in chosen]
    if dist is not None:
        new_dist = [dist[i] for i in chosen]
        total = sum(new_dist)
        ex.target_dist = [v / total for v in new_dist] if total > 0 else None
    ex.target_idx = chosen.index(ex.target_idx)
    ex.meta["subsampled_from"] = k


def _build_question(
    qid: str,
    q: Any,
    answer: Any,
    require_answer: bool = True,
) -> tuple[str, str, list[Option], int | None, list[float] | None] | str:
    """Return ``(qtype, instructions, options, target_idx, target_dist)`` or a drop reason.

    A missing answer is a ``bad_answer`` drop when ``require_answer`` is true
    (training / evaluation data); otherwise the question is kept unlabeled
    (``target_idx=None``), which is what inference on new rows needs.
    """
    if not isinstance(q, Mapping):
        return "bad_question"
    qtype = q.get("type")
    if qtype not in QTYPES:
        return "bad_question"
    instructions = q.get("instructions", "")
    instructions = "" if instructions is None else str(instructions).strip()
    criteria = q.get("criteria")
    has_answer = isinstance(answer, Mapping)
    if require_answer and not has_answer:
        return "bad_answer"
    target_idx: int | None = None
    target_dist: list[float] | None = None

    if qtype == "choice":
        if not isinstance(criteria, Mapping) or len(criteria) < 1:
            return "bad_question"
        options = choice_options(criteria)
        keys = [o.key for o in options]
        if has_answer:
            choice = answer.get("choice")
            if choice is None or str(choice) not in keys:
                return "bad_answer"
            target_idx = keys.index(str(choice))
            target_dist = soft_distribution(answer.get("probabilities"), keys)
    elif qtype == "noul":
        options = noul_options(criteria)
        if has_answer:
            try:
                p = float(answer.get("noul"))
            except (TypeError, ValueError):
                return "bad_answer"
            if not (0.0 <= p <= 1.0):
                return "bad_answer"
            target_idx = 0 if p >= 0.5 else 1
            target_dist = [p, 1.0 - p] if 0.0 < p < 1.0 else None
    else:  # score
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 1:
            return "bad_question"
        options = score_options(criteria)
        keys = [o.key for o in options]
        if has_answer:
            raw = answer.get("score")
            try:
                level = int(raw)
            except (TypeError, ValueError):
                return "bad_answer"
            if isinstance(raw, float) and not float(raw).is_integer():
                return "bad_answer"
            if not (0 <= level < len(options)):
                return "bad_answer"
            target_idx = level
            target_dist = soft_distribution(answer.get("probabilities"), keys)
    return qtype, instructions, options, target_idx, target_dist


def jev_row_to_examples(
    row: Mapping[str, Any],
    family: str = "",
    label_space: LabelSpace | None = None,
    policy: OverLimitPolicy | str | None = None,
    rng: random.Random | None = None,
    *,
    default_dataset: str = "",
    require_answer: bool = True,
) -> tuple[list[DecisionExample], dict[str, int]]:
    """Expand one JEV API row into decision examples (one per question).

    Args:
        row: parsed JEV row (``sample_id``, ``dataset``, ``task``, ``split``,
            ``state``, ``questions``, ``answers``, ``meta``).
        family: task family (``classification`` / ``ranking`` / ``search`` / ``""``),
            normally the data directory name.
        label_space: when given, menus larger than ``label_space.max_options``
            are handled by ``policy``.
        policy: ``drop`` (default) or ``subsample`` (keep gold + random others).
        rng: random source for subsampling; ``None`` -> deterministic per example id.
        default_dataset: used when the row carries no ``dataset`` (e.g. file stem).
        require_answer: drop questions without an answer (``bad_answer``); pass
            ``False`` at inference time to keep them as unlabeled examples.

    Returns:
        ``(examples, drops)`` where ``drops`` counts skipped questions by reason
        (keys: ``over_limit``, ``bad_answer``, ``empty_state``, ``bad_question``, ``bad_row``).
    """
    drops: dict[str, int] = {r: 0 for r in DROP_REASONS}
    policy_enum = OverLimitPolicy.coerce(policy)

    if not isinstance(row, Mapping):
        drops["bad_row"] += 1
        return [], drops
    questions = row.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        drops["bad_row"] += 1
        return [], drops
    answers = row.get("answers")
    if not isinstance(answers, Mapping):
        answers = {}

    dataset = str(row.get("dataset") or default_dataset or "")
    sample_id = row.get("sample_id")
    if sample_id is None or str(sample_id) == "":
        digest = hashlib.sha1(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        sample_id = f"{dataset or 'row'}:{digest}"
    sample_id = str(sample_id)
    task = str(row.get("task") or "")
    split = str(row.get("split") or "")
    meta = _compact_meta(row.get("meta"))

    state_text = render_state(row.get("state"))
    if not state_text:
        drops["empty_state"] += len(questions)
        return [], drops

    examples: list[DecisionExample] = []
    for qid, q in questions.items():
        built = _build_question(str(qid), q, answers.get(qid), require_answer=require_answer)
        if isinstance(built, str):
            drops[built] += 1
            continue
        qtype, instructions, options, target_idx, target_dist = built
        ex = DecisionExample(
            id=f"{sample_id}#{qid}",
            dataset=dataset,
            family=family,
            task=task,
            split=split,
            qtype=qtype,
            state=state_text,
            question=instructions,
            options=options,
            target_idx=target_idx,
            target_dist=target_dist,
            meta=dict(meta),
        )
        if label_space is not None and len(ex.options) > label_space.max_options:
            if policy_enum is OverLimitPolicy.SUBSAMPLE and ex.target_idx is not None:
                _subsample_options(ex, label_space.max_options, rng or _default_rng(ex.id))
            else:
                drops["over_limit"] += 1
                continue
        examples.append(ex)
    return examples, drops


def merge_drop_counts(target: dict[str, int], drops: Mapping[str, int]) -> dict[str, int]:
    """Accumulate ``drops`` into ``target`` (in place) and return it."""
    for k, v in drops.items():
        target[k] = target.get(k, 0) + int(v)
    return target


def iter_examples(
    rows: Iterable[Mapping[str, Any]],
    family: str = "",
    label_space: LabelSpace | None = None,
    policy: OverLimitPolicy | str | None = None,
    *,
    default_dataset: str = "",
    drops: dict[str, int] | None = None,
) -> Iterable[DecisionExample]:
    """Yield examples for an iterable of rows, accumulating drop counts into ``drops``."""
    for row in rows:
        exs, d = jev_row_to_examples(row, family, label_space, policy, default_dataset=default_dataset)
        if drops is not None:
            merge_drop_counts(drops, d)
        yield from exs
