"""JEV loss: masked cross-entropy / KL over the option menu plus a Brier term.

Given restricted menu logits ``z`` (``[B, K]``, one column per displayed
option, invalid slots masked), the model's distribution over options is

    p_j = softmax(z)_j   (over valid slots only)

and the loss for a (possibly soft) target ``t`` is

    L = L_main + lambda * L_Brier
    L_CE    = -sum_j t_j log p_j
    L_KL    = L_CE - H(t) = sum_j t_j log(t_j / p_j)
    L_Brier = sum_j (p_j - t_j)^2

``ce`` and ``kl`` have identical gradients; ``kl`` merely subtracts the target
entropy so that a perfectly matched soft target gives zero loss. The Brier
term is a proper scoring rule that directly rewards calibrated probabilities.
"""

from __future__ import annotations

import torch

_LOSS_TYPES = ("ce", "kl")


def masked_log_softmax(menu_logits: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid menu slots; masked slots get ``-inf``.

    Args:
        menu_logits: ``[B, K]`` restricted logits.
        label_mask: ``[B, K]`` bool, ``True`` for valid option slots.
    """
    mask = label_mask.bool()
    masked = menu_logits.float().masked_fill(~mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def menu_probs(
    menu_logits: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Probabilities over the menu (masked slots are exactly 0), optionally temperature-scaled.

    Args:
        menu_logits: ``[B, K]`` restricted logits.
        label_mask: ``[B, K]`` bool validity mask.
        temperature: scalar ``T`` (or ``[B]`` / ``[B, 1]`` tensor); logits are divided by ``T``.

    Returns:
        ``[B, K]`` float32 probabilities summing to 1 over valid slots.
    """
    logits = menu_logits.float()
    if isinstance(temperature, torch.Tensor):
        t = temperature.to(logits.device, torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        logits = logits / t
    elif float(temperature) != 1.0:
        logits = logits / float(temperature)
    mask = label_mask.bool()
    logp = masked_log_softmax(logits, mask)
    return logp.exp().masked_fill(~mask, 0.0)


def jev_loss(
    menu_logits: torch.Tensor,
    target_dist: torch.Tensor,
    label_mask: torch.Tensor,
    lambda_brier: float = 0.5,
    loss_type: str = "ce",
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the JEV loss and its components.

    Args:
        menu_logits: ``[B, K]`` restricted logits (display order).
        target_dist: ``[B, K]`` target distribution (one-hot or soft, display order;
            padded slots may hold anything, they are zeroed by ``label_mask``).
        label_mask: ``[B, K]`` bool, ``True`` for valid option slots.
        lambda_brier: weight of the Brier term.
        loss_type: ``"ce"`` or ``"kl"``.
        weights: optional ``[B]`` per-example weights (label reliability); the loss is ``mean(w * l)`` so
            down-weighted examples contribute proportionally less. ``None`` = all ones.

    Returns:
        ``(loss, components)`` where ``loss`` is a scalar tensor and ``components``
        holds detached scalar tensors ``ce``, ``kl``, ``brier``, ``acc`` (argmax
        agreement with the target argmax) and ``loss``.
    """
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {_LOSS_TYPES}, got {loss_type!r}")
    if menu_logits.dim() != 2:
        raise ValueError(f"menu_logits must be [B, K], got shape {tuple(menu_logits.shape)}")
    if menu_logits.shape != target_dist.shape or menu_logits.shape != label_mask.shape:
        raise ValueError(
            "menu_logits, target_dist and label_mask must share shape [B, K]; got "
            f"{tuple(menu_logits.shape)}, {tuple(target_dist.shape)}, {tuple(label_mask.shape)}"
        )

    mask = label_mask.bool()
    t = target_dist.to(menu_logits.device, torch.float32).masked_fill(~mask, 0.0)

    logp = masked_log_softmax(menu_logits, mask)
    logp_safe = logp.masked_fill(~mask, 0.0)           # avoid 0 * -inf = nan
    p = logp_safe.exp().masked_fill(~mask, 0.0)        # exp(0) = 1 at masked slots -> zero them

    ce = -(t * logp_safe).sum(-1)                       # [B]
    target_entropy = -torch.special.xlogy(t, t).sum(-1)  # 0 log 0 = 0
    kl = ce - target_entropy
    brier = ((p - t) ** 2).sum(-1)

    main = ce if loss_type == "ce" else kl
    if weights is None:
        loss = main.mean() + float(lambda_brier) * brier.mean()
    else:
        w = weights.to(menu_logits.device, torch.float32).reshape(-1)
        if w.shape[0] != main.shape[0]:
            raise ValueError(f"weights must be [B]={main.shape[0]}, got {tuple(weights.shape)}")
        loss = (w * main).mean() + float(lambda_brier) * (w * brier).mean()

    with torch.no_grad():
        acc = (p.argmax(-1) == t.argmax(-1)).float().mean()
    components = {
        "loss": loss.detach(),
        "ce": ce.mean().detach(),
        "kl": kl.mean().detach(),
        "brier": brier.mean().detach(),
        "acc": acc,
    }
    return loss, components


__all__ = ["jev_loss", "masked_log_softmax", "menu_probs"]
