"""``JEVTrainer``: a ``transformers.Trainer`` that trains the restricted first-token readout.

The trainer never passes the JEV-specific batch fields (``label_token_ids``,
``label_mask``, ``target_dist``, ``target_idx``, ``qtype_id``, ``num_options``,
``perm``, ``example_id``) to the backbone. ``compute_loss`` runs the restricted
readout (:func:`training.modeling.restricted_menu_logits`) and
:func:`training.loss.jev_loss`; ``prediction_step`` returns menu probabilities
mapped back to **canonical** option order (using the batch ``perm`` tensor) so
that :func:`training.metrics.summarize` sees canonical indices.

Also provided: :func:`build_training_arguments` (transformers 5.x:
``warmup_steps`` computed from ``warmup_ratio``, ``eval_strategy``) and
:func:`compute_metrics_fn`.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from .loss import jev_loss, menu_probs
from .metrics import flatten_summary, summarize
from .modeling import restricted_menu_logits

if TYPE_CHECKING:  # pragma: no cover
    from .labels import LabelSpace

logger = logging.getLogger(__name__)

#: Keys that are forwarded to the backbone; everything else is a JEV extra.
MODEL_INPUT_KEYS: tuple[str, ...] = ("input_ids", "attention_mask", "position_ids")

#: JEV batch fields produced by the collator (documentation; anything not in
#: ``MODEL_INPUT_KEYS`` is treated as an extra).
JEV_EXTRA_KEYS: tuple[str, ...] = (
    "label_token_ids", "label_mask", "target_dist", "target_idx", "qtype_id", "num_options", "perm", "example_id",
    "weight",
)

_RUNNING_KEYS: tuple[str, ...] = ("ce", "kl", "brier", "acc")


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from a nested mapping / attribute-style config."""
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def _pad_perm(perm: Any, batch_size: int, width: int, device: torch.device) -> torch.Tensor | None:
    """Normalise ``perm`` (tensor ``[B, K]`` padded with -1, or list of lists) to a long tensor."""
    if perm is None:
        return None
    if isinstance(perm, torch.Tensor):
        return perm.to(device=device, dtype=torch.long)
    out = torch.full((batch_size, width), -1, dtype=torch.long)
    for b, row in enumerate(perm):
        row_t = torch.as_tensor(list(row), dtype=torch.long)
        out[b, : row_t.numel()] = row_t
    return out.to(device)


class JEVTrainer(Trainer):
    """Trainer for JEV restricted-readout fine-tuning.

    Args:
        label_space: the :class:`training.labels.LabelSpace` (needed for ``max_options``,
            the fixed width of the probability matrix returned by ``prediction_step``).
        lambda_brier: weight of the Brier term (see :func:`training.loss.jev_loss`).
        loss_type: ``"ce"`` or ``"kl"``.

    ``args`` must have ``remove_unused_columns=False`` (the collator emits
    non-model fields) and ``label_names=["target_idx"]``.
    """

    def __init__(
        self,
        *args: Any,
        label_space: LabelSpace,
        lambda_brier: float = 0.5,
        loss_type: str = "ce",
        group_by_length: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.group_by_length = bool(group_by_length)
        self.max_prompt_length = max_length
        super().__init__(*args, **kwargs)
        self.label_space = label_space
        self.lambda_brier = float(lambda_brier)
        self.loss_type = str(loss_type)
        # We normalise the loss ourselves (mean over the batch); Trainer then divides
        # by the gradient-accumulation steps, which is the correct behaviour.
        self.model_accepts_loss_kwargs = False
        self._jev_running: dict[str, torch.Tensor | float] = {}
        self._reset_running()
        if self.args.remove_unused_columns:
            raise ValueError("JEVTrainer requires TrainingArguments(remove_unused_columns=False)")
        if not self.args.label_names or "target_idx" not in self.args.label_names:
            logger.warning(
                "TrainingArguments.label_names should contain 'target_idx' (got %r)", self.args.label_names
            )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def menu_width(self) -> int:
        """Fixed width of the probability matrix returned by ``prediction_step``."""
        return int(getattr(self.label_space, "max_options", 0) or 0)

    @staticmethod
    def split_inputs(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a collated batch into ``(model_inputs, jev_extras)``."""
        model_inputs = {k: v for k, v in inputs.items() if k in MODEL_INPUT_KEYS}
        extras = {k: v for k, v in inputs.items() if k not in MODEL_INPUT_KEYS}
        for required in ("label_token_ids", "label_mask", "target_dist"):
            if required not in extras:
                raise KeyError(f"Batch is missing JEV field {required!r}; got keys {sorted(inputs)}")
        return model_inputs, extras

    def _reset_running(self) -> None:
        self._jev_running = {k: 0.0 for k in _RUNNING_KEYS}
        self._jev_running["count"] = 0.0

    def _accumulate(self, comps: Mapping[str, torch.Tensor], batch_size: int) -> None:
        run = self._jev_running
        for k in _RUNNING_KEYS:
            if k in comps:
                run[k] = run[k] + comps[k].detach().float() * batch_size
        run["count"] = run["count"] + float(batch_size)

    def _menu_forward(
        self, model: torch.nn.Module, inputs: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Run the restricted readout and the loss; returns ``(loss, menu_logits, comps, extras)``."""
        model_inputs, extras = self.split_inputs(inputs)
        menu = restricted_menu_logits(
            model, model_inputs["input_ids"], model_inputs.get("attention_mask"), extras["label_token_ids"]
        )
        loss, comps = jev_loss(
            menu, extras["target_dist"], extras["label_mask"], lambda_brier=self.lambda_brier, loss_type=self.loss_type,
            weights=extras.get("weight") if model.training else None,
        )
        return loss, menu, comps, extras

    # ------------------------------------------------------------------ #
    # Trainer overrides
    # ------------------------------------------------------------------ #
    def _get_train_sampler(self, train_dataset: Any = None) -> Any:
        """Length-grouped sampling (``training.group_by_length``) with cached length estimates.

        Consecutive ``per_device_batch x grad_accum`` indices have similar prompt lengths (mega-batches of 50 such
        groups are shuffled, then sorted), so far less padding is computed. Falls back to the default sampler.
        """
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if self.group_by_length and ds is not None:
            from transformers.trainer_pt_utils import LengthGroupedSampler

            from .data import dataset_lengths

            lengths = dataset_lengths(ds, self.max_prompt_length)
            if lengths is not None and len(lengths) == len(ds):
                logger.info("length-grouped sampling over %d items (median est. %d tokens)", len(lengths),
                            sorted(lengths)[len(lengths) // 2] if lengths else 0)
                return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                            lengths=lengths)
            logger.warning("group_by_length: dataset has no length index; using the default sampler")
        return super()._get_train_sampler(train_dataset) if train_dataset is not None else super()._get_train_sampler()
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Restricted-readout loss for one batch (JEV fields are never passed to the model)."""
        loss, menu, comps, _ = self._menu_forward(model, inputs)
        if model.training:
            self._accumulate(comps, int(menu.shape[0]))
        if return_outputs:
            return loss, {"menu_logits": menu}
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Inject running averages of the loss components into training logs."""
        count = float(self._jev_running.get("count", 0.0))
        is_train_log = "loss" in logs and not any(k.startswith("eval_") for k in logs)
        if is_train_log and count > 0:
            for k in _RUNNING_KEYS:
                val = self._jev_running.get(k, 0.0)
                val_f = float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                logs[f"jev/{k}"] = round(val_f / count, 6)
            self._reset_running()
        super().log(logs, start_time)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Evaluate one batch.

        Returns ``(loss, probs, labels)`` where ``probs`` is ``[B, max_options]``
        in **canonical** option order (un-permuted with the batch ``perm``, zero
        padded) and ``labels`` is ``[B, 3] = (canonical target_idx, qtype_id, num_options)``.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, menu, _comps, extras = self._menu_forward(model, inputs)
            loss = loss.detach()
            if prediction_loss_only:
                return loss, None, None
            label_mask = extras["label_mask"].bool()
            probs = menu_probs(menu, label_mask)  # [B, K] display order, 0 at masked slots
            bsz, k = probs.shape
            device = probs.device

            target_idx = extras.get("target_idx")
            if target_idx is None:
                target_idx = extras["target_dist"].argmax(-1)
            target_idx = target_idx.to(device=device, dtype=torch.long)
            qtype_id = extras.get("qtype_id")
            if qtype_id is None:
                qtype_id = torch.zeros(bsz, dtype=torch.long, device=device)
            num_options = extras.get("num_options")
            if num_options is None:
                num_options = label_mask.sum(-1)
            num_options = num_options.to(device=device, dtype=torch.long)

            perm = _pad_perm(extras.get("perm"), bsz, k, device)
            width = max(self.menu_width, k)
            canon = torch.zeros(bsz, width, dtype=torch.float32, device=device)
            if perm is None:
                canon[:, :k] = probs
                target_canon = target_idx
            else:
                valid = perm >= 0
                idx = perm.clamp(min=0, max=width - 1)
                canon.scatter_add_(1, idx, probs.masked_fill(~valid, 0.0))
                # an identity for slots the perm does not cover (defensive; perm covers all K)
                uncovered = valid.sum(-1) == 0
                if bool(uncovered.any()):
                    canon[uncovered, :k] = probs[uncovered]
                gathered = perm.gather(1, target_idx.clamp(min=0, max=k - 1).unsqueeze(1)).squeeze(1)
                target_canon = torch.where(gathered >= 0, gathered, target_idx)
            labels_out = torch.stack([target_canon, qtype_id.to(device), num_options], dim=1)
        return loss, canon, labels_out


# ---------------------------------------------------------------------- #
# TrainingArguments / metrics factories
# ---------------------------------------------------------------------- #
def compute_training_schedule(
    n_train_examples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    epochs: float,
    max_steps: int = -1,
    warmup_ratio: float = 0.05,
) -> dict[str, int]:
    """Compute ``effective_batch_size``, ``steps_per_epoch``, ``total_steps`` and ``warmup_steps``."""
    world_size = max(int(world_size), 1)
    effective = max(int(per_device_batch_size), 1) * max(int(gradient_accumulation_steps), 1) * world_size
    steps_per_epoch = max(1, math.ceil(max(int(n_train_examples), 0) / effective)) if n_train_examples else 0
    if max_steps is not None and int(max_steps) > 0:
        total = int(max_steps)
    else:
        total = int(math.ceil(steps_per_epoch * float(epochs)))
    warmup = int(round(max(float(warmup_ratio), 0.0) * total)) if total > 0 else 0
    return {
        "effective_batch_size": effective,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total,
        "warmup_steps": warmup,
    }


def build_training_arguments(
    cfg: Any,
    output_dir: str,
    n_train_examples: int,
    world_size: int,
    *,
    has_eval: bool = True,
) -> TrainingArguments:
    """Build ``TrainingArguments`` from the ``training``/``run`` sections of ``cfg``.

    * ``warmup_steps`` is computed from ``training.warmup_ratio`` (transformers 5.x
      has no ``warmup_ratio``), using ``n_train_examples`` and ``world_size``.
    * gradient checkpointing is handled on the model (``args.gradient_checkpointing=False``).
    * ``remove_unused_columns=False``, ``label_names=["target_idx"]``,
      ``ddp_find_unused_parameters=False``, ``logging_first_step=True``, ``eval_on_start=False``.
    * ``save_only_model`` is left at its default so optimizer state is kept for resuming.

    Args:
        cfg: nested config (mapping or attribute style).
        output_dir: run directory (checkpoints go to ``<output_dir>/checkpoints``).
        n_train_examples: length of the training dataset (across all ranks).
        world_size: number of data-parallel processes.
        has_eval: ``False`` disables in-training evaluation (no validation set).
    """
    t = lambda key, default=None: _cfg_get(cfg, f"training.{key}", default)  # noqa: E731

    epochs = float(t("epochs", 1))
    max_steps = int(t("max_steps", -1) or -1)
    bs = int(t("per_device_train_batch_size", 8))
    ga = int(t("gradient_accumulation_steps", 4))
    schedule = compute_training_schedule(
        n_train_examples, bs, ga, world_size, epochs, max_steps, float(t("warmup_ratio", 0.05))
    )
    logger.info(
        "Schedule: n_train=%d world_size=%d effective_batch=%d steps/epoch=%d total_steps=%d warmup_steps=%d",
        n_train_examples, world_size, schedule["effective_batch_size"], schedule["steps_per_epoch"],
        schedule["total_steps"], schedule["warmup_steps"],
    )

    report_to = _cfg_get(cfg, "run.report_to", "none") or "none"
    if isinstance(report_to, str):
        report_list = [r.strip() for r in report_to.split(",") if r.strip()] or ["none"]
    else:
        report_list = [str(r) for r in report_to] or ["none"]
    run_name = str(_cfg_get(cfg, "run.name", os.path.basename(os.path.normpath(output_dir))))
    if "wandb" in report_list:
        # project / entity (team or user); the run name is run.name via TrainingArguments.run_name.
        # An already exported WANDB_PROJECT / WANDB_ENTITY in the environment wins.
        os.environ.setdefault("WANDB_PROJECT", str(_cfg_get(cfg, "run.wandb_project", "JEV")))
        entity = _cfg_get(cfg, "run.wandb_entity", None)
        if entity:
            os.environ.setdefault("WANDB_ENTITY", str(entity))

    eval_strategy = str(t("eval_strategy", "steps")) if has_eval else "no"
    eval_steps = int(t("eval_steps", 500))
    save_steps = int(t("save_steps", 500))
    seed = int(_cfg_get(cfg, "run.seed", 42))

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        run_name=run_name,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=int(t("per_device_eval_batch_size", 16)),
        gradient_accumulation_steps=ga,
        learning_rate=float(t("learning_rate", 5e-5)),
        weight_decay=float(t("weight_decay", 0.01)),
        warmup_steps=schedule["warmup_steps"],
        lr_scheduler_type=str(t("lr_scheduler_type", "cosine")),
        max_grad_norm=float(t("max_grad_norm", 1.0)),
        bf16=bool(t("bf16", True)),
        logging_steps=int(t("logging_steps", 10)),
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(t("save_total_limit", 3)) if t("save_total_limit", 3) is not None else None,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(t("dataloader_num_workers", 4)),
        dataloader_pin_memory=True,
        optim=str(t("optim", "adamw_torch_fused")),
        report_to=report_list,
        remove_unused_columns=False,
        label_names=["target_idx"],
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        eval_on_start=False,
        disable_tqdm=bool(t("disable_tqdm", False)),
    )
    return args


def compute_metrics_fn(label_space: LabelSpace | None = None) -> Callable[[EvalPrediction], dict[str, float]]:
    """Return a ``compute_metrics`` callable for :class:`JEVTrainer`.

    ``EvalPrediction.predictions`` is the ``[N, Kmax]`` canonical probability
    matrix; ``label_ids`` is ``[N, 3] = (target_idx, qtype_id, num_options)``.
    Output keys are flat (``accuracy``, ``nll``, ``brier``, ``ece``, ``choice_accuracy``,
    ``noul_auroc``, ``score_mae`` ...) so the Trainer prefixes them with ``eval_``.
    """
    max_options = int(getattr(label_space, "max_options", 0) or 0) if label_space is not None else 0

    def _compute(ep: EvalPrediction) -> dict[str, float]:
        preds = ep.predictions
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        probs = np.asarray(preds, dtype=np.float64)
        labels = np.asarray(ep.label_ids)
        if labels.ndim != 2 or labels.shape[1] < 3:
            raise ValueError(f"Expected label_ids of shape [N, 3], got {labels.shape}")
        if probs.ndim == 1:
            probs = probs[:, None]
        if max_options and probs.shape[1] < max_options:
            probs = np.pad(probs, ((0, 0), (0, max_options - probs.shape[1])))
        summary = summarize(probs, labels[:, 0], labels[:, 1], labels[:, 2])
        flat = flatten_summary(summary)
        return {k: v for k, v in flat.items() if not k.startswith("ds_")}

    return _compute


__all__ = [
    "JEVTrainer",
    "MODEL_INPUT_KEYS",
    "JEV_EXTRA_KEYS",
    "compute_training_schedule",
    "build_training_arguments",
    "compute_metrics_fn",
]
