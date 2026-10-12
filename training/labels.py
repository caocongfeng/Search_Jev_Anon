"""Label alphabets for the restricted first-token readout.

A :class:`LabelSpace` maps the *displayed* option slots (slot 0, slot 1, ...)
to single-token labels (``A``, ``B``, ...) and to their tokenizer ids.  The
model answers a question by emitting exactly one of these tokens directly
after the generation prompt, so every label **must** be a single token and
the tokenisation must not depend on the preceding context (no space-prefixed
variants).  Both properties are verified at construction time.

Two built-in alphabets exist:

* ``basic``: ``A``-``Z`` (26 labels, all must be single tokens).
* ``extended``: Latin upper/lower, digits, Greek lower/upper and Cyrillic
  lower/upper (174 candidates, ~173 single-token labels for the Qwen3.5
  tokenizer; non-single-token candidates are skipped with a warning).

This module deliberately does not import torch or transformers: any object
with an ``encode(text, add_special_tokens=False) -> list[int]`` method works
as the ``tokenizer`` argument.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


def _char_range(start: str, stop: str, *, skip: Iterable[str] = ()) -> list[str]:
    """Return the inclusive list of characters from ``start`` to ``stop``."""
    skipped = set(skip)
    return [chr(c) for c in range(ord(start), ord(stop) + 1) if chr(c) not in skipped]


LATIN_UPPER: list[str] = _char_range("A", "Z")
LATIN_LOWER: list[str] = _char_range("a", "z")
DIGITS: list[str] = _char_range("0", "9")
# Greek lower alpha..omega without the final sigma (U+03C2) -> 24 letters.
GREEK_LOWER: list[str] = _char_range("α", "ω", skip=("ς",))
# Greek upper Alpha..Omega; U+03A2 is unassigned -> 24 letters.
GREEK_UPPER: list[str] = _char_range("Α", "Ω", skip=("΢",))
CYRILLIC_LOWER: list[str] = _char_range("а", "я")  # 32 letters
CYRILLIC_UPPER: list[str] = _char_range("А", "Я")  # 32 letters

ALPHABETS: dict[str, list[str]] = {
    "basic": list(LATIN_UPPER),
    "extended": (
        LATIN_UPPER
        + LATIN_LOWER
        + DIGITS
        + GREEK_LOWER
        + GREEK_UPPER
        + CYRILLIC_LOWER
        + CYRILLIC_UPPER
    ),
}
"""Named alphabets: ``basic`` (26) and ``extended`` (174 candidates)."""

_CONTEXT_CHECK_COUNT = 3


class LabelSpaceError(ValueError):
    """Raised when an alphabet cannot be represented by single, distinct tokens."""


class LabelSpace:
    """Ordered single-token labels used as the answer menu.

    Args:
        tokenizer: any object exposing ``encode(text, add_special_tokens=False)``
            (a Hugging Face tokenizer in practice).
        alphabet: ``"basic"``, ``"extended"`` or an explicit list of label
            strings.  For ``"extended"`` labels that do not tokenise to a single
            token (or collide with an earlier label) are skipped with a
            warning; for ``"basic"`` and explicit lists this is an error.
        max_options: maximum number of options a decision may have; defaults to
            the number of usable labels and may not exceed it.

    Attributes:
        labels: the usable label strings in slot order.
        token_ids: the corresponding token ids (same order).
        max_options: the largest menu size supported.
        alphabet_name: ``"basic"``, ``"extended"`` or ``"custom"``.
    """

    def __init__(
        self,
        tokenizer: Any,
        alphabet: str | Sequence[str] = "extended",
        max_options: int | None = None,
    ) -> None:
        if isinstance(alphabet, str):
            if alphabet not in ALPHABETS:
                raise LabelSpaceError(
                    f"unknown alphabet {alphabet!r}; expected one of {sorted(ALPHABETS)} or a list"
                )
            candidates = list(ALPHABETS[alphabet])
            name = alphabet
        else:
            candidates = [str(x) for x in alphabet]
            name = "custom"
            if not candidates:
                raise LabelSpaceError("an explicit label alphabet must not be empty")
        lenient = name == "extended"

        self.tokenizer = tokenizer
        self.alphabet_name = name
        self.labels: list[str] = []
        self.token_ids: list[int] = []
        self.skipped: list[str] = []
        seen_ids: set[int] = set()

        for label in candidates:
            ids = list(tokenizer.encode(label, add_special_tokens=False))
            problem: str | None = None
            if len(ids) != 1:
                problem = f"label {label!r} tokenises to {len(ids)} tokens {ids}"
            elif ids[0] in seen_ids:
                problem = f"label {label!r} shares token id {ids[0]} with an earlier label"
            elif label in self.labels:
                problem = f"label {label!r} is listed twice"
            if problem is not None:
                if lenient:
                    logger.warning("LabelSpace: skipping %s", problem)
                    self.skipped.append(label)
                    continue
                raise LabelSpaceError(problem)
            self.labels.append(label)
            self.token_ids.append(int(ids[0]))
            seen_ids.add(int(ids[0]))

        if not self.labels:
            raise LabelSpaceError("no usable single-token labels in the alphabet")

        self._verify_context_rule()

        if max_options is None:
            max_options = len(self.labels)
        max_options = int(max_options)
        if max_options < 2:
            raise LabelSpaceError(f"max_options must be >= 2, got {max_options}")
        if max_options > len(self.labels):
            raise LabelSpaceError(
                f"max_options={max_options} exceeds the {len(self.labels)} usable labels of "
                f"alphabet {name!r}"
            )
        self.max_options: int = max_options
        self._index: dict[str, int] = {lab: i for i, lab in enumerate(self.labels)}
        self._id_index: dict[int, int] = {tid: i for i, tid in enumerate(self.token_ids)}
        logger.info(
            "LabelSpace(%s): %d labels, max_options=%d, skipped=%d",
            name,
            len(self.labels),
            self.max_options,
            len(self.skipped),
        )

    # ------------------------------------------------------------------ checks
    def _verify_context_rule(self) -> None:
        """Ensure a label after a newline yields the same token id (no space-prefixed variant)."""
        for label, tid in list(zip(self.labels, self.token_ids))[:_CONTEXT_CHECK_COUNT]:
            ids = list(self.tokenizer.encode("\n" + label, add_special_tokens=False))
            if not ids or ids[-1] != tid:
                raise LabelSpaceError(
                    f"context rule violated for label {label!r}: encode('\\n'+label) -> {ids}, "
                    f"expected last id {tid}. The prompt must end with a newline and labels must be "
                    "context-independent single tokens."
                )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.labels)

    def ids_for(self, k: int) -> list[int]:
        """Token ids of the first ``k`` slots (raises if ``k`` exceeds ``max_options``)."""
        self._check_k(k)
        return self.token_ids[:k]

    def labels_for(self, k: int) -> list[str]:
        """Label strings of the first ``k`` slots."""
        self._check_k(k)
        return self.labels[:k]

    def _check_k(self, k: int) -> None:
        if k < 1 or k > self.max_options:
            raise ValueError(f"k={k} outside [1, max_options={self.max_options}]")

    def index_of_label(self, label: str) -> int:
        """Slot index of a label string (``ValueError`` if unknown)."""
        try:
            return self._index[label]
        except KeyError as exc:
            raise ValueError(f"unknown label {label!r}") from exc

    def index_of_token(self, token_id: int) -> int:
        """Slot index of a token id (``ValueError`` if unknown)."""
        try:
            return self._id_index[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"token id {token_id} is not a label token") from exc

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description written to ``<run>/label_space.json``."""
        return {
            "alphabet": self.alphabet_name,
            "labels": list(self.labels),
            "token_ids": list(self.token_ids),
            "max_options": self.max_options,
            "skipped": list(self.skipped),
        }

    def __repr__(self) -> str:
        return (
            f"LabelSpace(alphabet={self.alphabet_name!r}, n_labels={len(self.labels)}, "
            f"max_options={self.max_options})"
        )
