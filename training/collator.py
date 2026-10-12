"""Batch collation for Qwen3.5-JEV.

:class:`JEVDataCollator` turns the per-example feature dicts produced by
:class:`training.data.JEVDataset` into padded tensors.  Prompts are
**left**-padded (the restricted read-out looks at the last position, so the
last column of ``attention_mask`` must be all ones) and the per-example
option "menus" are padded to the widest menu in the batch.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["JEVDataCollator"]


class JEVDataCollator:
    """Collate JEV features into tensors.

    Output keys:

    * ``input_ids``, ``attention_mask`` -- ``[B, L]`` long, left-padded via
      ``tokenizer.pad`` (optionally to a multiple of ``pad_to_multiple_of``).
    * ``label_token_ids`` -- ``[B, Kmax]`` long, the vocabulary ids of the
      label tokens shown for each example (pad ``0``).
    * ``label_mask`` -- ``[B, Kmax]`` bool, ``True`` for real menu slots.
    * ``target_dist`` -- ``[B, Kmax]`` float32 target distribution in display
      order (pad ``0``).
    * ``target_idx`` -- ``[B]`` long display index of the gold option (``-1``
      when unlabeled).
    * ``qtype_id`` -- ``[B]`` long (choice=0, noul=1, score=2).
    * ``num_options`` -- ``[B]`` long.
    * ``perm`` -- ``[B, Kmax]`` long, display slot -> canonical index (pad ``-1``);
      lets the evaluation path map probabilities back to canonical order.
    """

    #: Default padding multiple.  128 is REQUIRED for training on Qwen3.5: the
    #: flash-linear-attention (tilelang) backward kernel used by the Gated-DeltaNet
    #: layers returns NaN gradients whenever the padded sequence length is an odd
    #: multiple of 64 (L % 128 == 64), verified on Qwen3.5-0.8B and -4B with fla 0.5.2.
    #: Padding every batch to a multiple of 128 avoids those lengths entirely.
    SAFE_PAD_MULTIPLE: int = 128

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = SAFE_PAD_MULTIPLE) -> None:
        if getattr(tokenizer, "padding_side", None) != "left":
            raise ValueError(
                f"JEVDataCollator requires tokenizer.padding_side == 'left' (got {getattr(tokenizer, 'padding_side', None)!r}); "
                "use training.modeling.load_tokenizer or set tokenizer.padding_side = 'left'"
            )
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer has no pad token; set tokenizer.pad_token before collating")
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = None if pad_to_multiple_of is None or int(pad_to_multiple_of) <= 1 else int(pad_to_multiple_of)
        # transformers >= 4.46 accepts an explicit padding_side in pad(); pass it when available
        try:
            self._pad_accepts_side = "padding_side" in inspect.signature(tokenizer.pad).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic tokenizer objects
            self._pad_accepts_side = False

    def _pad_prompts(self, features: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer.padding_side != "left":
            raise AssertionError("tokenizer.padding_side changed after collator construction; must be 'left'")
        encoded = {
            "input_ids": [list(f["input_ids"]) for f in features],
            "attention_mask": [list(f["attention_mask"]) for f in features],
        }
        kwargs: dict[str, Any] = {"padding": True, "pad_to_multiple_of": self.pad_to_multiple_of, "return_tensors": "pt"}
        if self._pad_accepts_side:
            kwargs["padding_side"] = "left"
        batch = self.tokenizer.pad(encoded, **kwargs)
        input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.long)
        if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
            raise AssertionError(f"unexpected padded shapes {tuple(input_ids.shape)} / {tuple(attention_mask.shape)}")
        if input_ids.shape[0] and not bool(attention_mask[:, -1].all()):
            raise AssertionError("left padding violated: last attention column is not all ones")
        return input_ids, attention_mask

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("cannot collate an empty batch")
        input_ids, attention_mask = self._pad_prompts(features)
        bsz = len(features)
        num_options = [int(f["num_options"]) for f in features]
        kmax = max(1, max(num_options))
        label_token_ids = torch.zeros(bsz, kmax, dtype=torch.long)
        label_mask = torch.zeros(bsz, kmax, dtype=torch.bool)
        target_dist = torch.zeros(bsz, kmax, dtype=torch.float32)
        perm = torch.full((bsz, kmax), -1, dtype=torch.long)
        target_idx = torch.empty(bsz, dtype=torch.long)
        qtype_id = torch.empty(bsz, dtype=torch.long)
        for b, f in enumerate(features):
            k = num_options[b]
            ids = list(f["label_token_ids"])
            dist = list(f["target_dist"])
            if len(ids) < k or len(dist) != k:
                raise ValueError(
                    f"example {f.get('example_id', b)}: num_options={k} but {len(ids)} label ids and {len(dist)} target entries"
                )
            label_token_ids[b, :k] = torch.as_tensor(ids[:k], dtype=torch.long)
            label_mask[b, :k] = True
            target_dist[b, :k] = torch.as_tensor(dist, dtype=torch.float32)
            p = list(f.get("perm") or range(k))
            if len(p) != k:
                raise ValueError(f"example {f.get('example_id', b)}: perm length {len(p)} != num_options {k}")
            perm[b, :k] = torch.as_tensor(p, dtype=torch.long)
            target_idx[b] = int(f["target_idx"])
            qtype_id[b] = int(f["qtype_id"])
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_token_ids": label_token_ids,
            "label_mask": label_mask,
            "target_dist": target_dist,
            "target_idx": target_idx,
            "qtype_id": qtype_id,
            "num_options": torch.as_tensor(num_options, dtype=torch.long),
            "perm": perm,
        }
        if any("weight" in f for f in features):
            batch["weight"] = torch.as_tensor([float(f.get("weight", 1.0)) for f in features], dtype=torch.float32)
        return batch
