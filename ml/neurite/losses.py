#!/usr/bin/env python3
"""Topology-aware losses for thin-structure (neurite) segmentation.

The headline loss is **soft-clDice** (Shit et al., "clDice — a Novel Topology-
Preserving Loss Function for Tubular Structure Segmentation", CVPR 2021). Plain
Dice/BCE reward pixel overlap, so a network can score well while leaving thin
neurites broken — exactly the failure mode of the CellProfiler baseline this
module replaces. clDice instead measures overlap between each mask and the
*skeleton* of the other, so a single-pixel gap in a process is penalised
heavily. That directly targets connectivity/topology, which is what the
downstream tol=3 F1 metric and the per-cell length measurements care about.

Soft skeletonization (Shit et al., Sec. 3.1) is a differentiable stand-in for
morphological thinning built from iterated min-/max-pooling::

    soft_erode(I)  = -maxpool(-I)                       # grayscale erosion
    soft_dilate(I) =  maxpool(I)                        # grayscale dilation
    soft_open(I)   =  soft_dilate(soft_erode(I))
    skel += relu(I - soft_open(I))                      # accumulate ridge
    I    =  soft_erode(I)                               # peel one layer
    ... repeat k times

Everything here is self-contained PyTorch — no ``monai`` required. If ``monai``
is installed you may prefer ``monai.losses.SoftclDiceLoss`` /
``SoftDiceclDiceLoss``; ``monai_cldice()`` returns it when available, else
raises a clear ImportError. The reference/default path is the pure-torch
``SoftDiceClDiceBCELoss`` below.

Run ``python losses.py`` for a sanity check on random tensors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Differentiable grayscale erosion via min-pooling (3x3 cross).

    Args:
        img: Tensor of shape (N, C, H, W) with values in [0, 1].

    Returns:
        Eroded tensor, same shape.
    """
    # min-pool == -maxpool(-x); use separable 3x1 / 1x3 as in the paper.
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Differentiable grayscale dilation via 3x3 max-pooling.

    Args:
        img: Tensor of shape (N, C, H, W) with values in [0, 1].

    Returns:
        Dilated tensor, same shape.
    """
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Morphological opening = erosion followed by dilation.

    Args:
        img: Tensor of shape (N, C, H, W).

    Returns:
        Opened tensor, same shape.
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Iterative differentiable skeletonization (Shit et al., 2021).

    Args:
        img: Probability tensor of shape (N, C, H, W) in [0, 1].
        iters: Number of thinning iterations. Should exceed the maximum
            expected half-width of a structure in pixels.

    Returns:
        Soft skeleton tensor, same shape, values in [0, 1].
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        # skel + delta - skel*delta  == fuzzy union, kept in [0,1]
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    iters: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft clDice loss (1 - clDice) on probabilities.

    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens), where
    Tprec = |skel(pred) . true| / |skel(pred)| and
    Tsens = |skel(true) . pred| / |skel(true)|.

    Args:
        y_pred: Predicted probabilities, shape (N, C, H, W) in [0, 1].
        y_true: Ground-truth mask, same shape, in {0, 1}.
        iters: Soft-skeletonization iterations.
        smooth: Numerical stabiliser.

    Returns:
        Scalar loss tensor (1 - clDice), lower is better.
    """
    skel_pred = soft_skeletonize(y_pred, iters)
    skel_true = soft_skeletonize(y_true, iters)
    tprec = (torch.sum(skel_pred * y_true) + smooth) / (
        torch.sum(skel_pred) + smooth
    )
    tsens = (torch.sum(skel_true * y_pred) + smooth) / (
        torch.sum(skel_true) + smooth
    )
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
    return 1.0 - cl_dice


def soft_dice_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Standard soft Dice loss on probabilities.

    Args:
        y_pred: Predicted probabilities, shape (N, C, H, W) in [0, 1].
        y_true: Ground-truth mask, same shape, in {0, 1}.
        smooth: Numerical stabiliser.

    Returns:
        Scalar Dice loss (1 - Dice).
    """
    inter = torch.sum(y_pred * y_true)
    denom = torch.sum(y_pred) + torch.sum(y_true)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


class SoftDiceClDiceBCELoss(nn.Module):
    """Combined BCE + soft-Dice + soft-clDice loss.

    The composite balances three complementary signals:

    * **BCE** — per-pixel calibration / stable early gradients.
    * **soft-Dice** — region overlap, robust to the heavy foreground/background
      imbalance of sparse neurites.
    * **soft-clDice** — topology/connectivity (no broken processes).

    The loss operates on raw logits; it applies ``sigmoid`` internally for the
    Dice/clDice terms and uses ``binary_cross_entropy_with_logits`` for BCE.
    """

    def __init__(
        self,
        iters: int = 10,
        alpha_bce: float = 0.5,
        alpha_dice: float = 0.5,
        alpha_cldice: float = 0.5,
        pos_weight: Optional[float] = None,
    ) -> None:
        """Initialise the composite loss.

        Args:
            iters: Soft-skeletonization iterations for the clDice term.
            alpha_bce: Weight on the BCE term.
            alpha_dice: Weight on the soft-Dice term.
            alpha_cldice: Weight on the soft-clDice term.
            pos_weight: Optional positive-class weight for BCE, useful for the
                extreme class imbalance of thin structures.
        """
        super().__init__()
        self.iters = iters
        self.alpha_bce = alpha_bce
        self.alpha_dice = alpha_dice
        self.alpha_cldice = alpha_cldice
        self.pos_weight = pos_weight

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            logits: Raw model outputs, shape (N, 1, H, W).
            target: Ground-truth mask, same shape, in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        target = target.to(logits.dtype)
        pw = (
            torch.tensor(self.pos_weight, device=logits.device)
            if self.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        prob = torch.sigmoid(logits)
        dice = soft_dice_loss(prob, target)
        cldice = soft_cldice_loss(prob, target, iters=self.iters)
        return (
            self.alpha_bce * bce
            + self.alpha_dice * dice
            + self.alpha_cldice * cldice
        )


def monai_cldice(**kwargs) -> nn.Module:
    """Return a MONAI clDice loss if ``monai`` is importable.

    Args:
        **kwargs: Forwarded to ``monai.losses.SoftDiceclDiceLoss``.

    Returns:
        A MONAI loss module.

    Raises:
        ImportError: If ``monai`` is not installed.
    """
    try:
        from monai.losses import SoftDiceclDiceLoss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "monai is not installed; use SoftDiceClDiceBCELoss (pure torch) "
            "instead, or `pip install monai`."
        ) from exc
    return SoftDiceclDiceLoss(**kwargs)


def _sanity_check() -> None:
    """Smoke-test the losses on random tensors and a trivial optimisation."""
    torch.manual_seed(0)
    n, c, h, w = 2, 1, 64, 64
    target = (torch.rand(n, c, h, w) > 0.9).float()

    loss_fn = SoftDiceClDiceBCELoss(iters=5)

    # Random logits -> finite, positive loss.
    logits = torch.randn(n, c, h, w, requires_grad=True)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    assert logits.grad is not None, "no gradient produced"
    print(f"random-input loss = {loss.item():.4f}  (grad ok)")

    # Overfit a single learnable logit field to the target -> loss decreases.
    param = torch.zeros(n, c, h, w, requires_grad=True)
    opt = torch.optim.Adam([param], lr=0.5)
    first = last = None
    for step in range(40):
        opt.zero_grad()
        l = loss_fn(param, target)
        l.backward()
        opt.step()
        if step == 0:
            first = l.item()
        last = l.item()
    print(f"overfit loss: {first:.4f} -> {last:.4f}")
    assert last < first, "loss did not decrease on trivial overfit"
    print("losses.py sanity check PASSED")


if __name__ == "__main__":
    _sanity_check()
