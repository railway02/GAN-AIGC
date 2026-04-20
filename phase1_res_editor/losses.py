# losses.py
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def phase1_loss(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    bounded_r: torch.Tensor,
    score_after: torch.Tensor,
    lambda_rec: float = 1.0,
    lambda_det: float = 0.5,
    lambda_tv: float = 0.01,
    lambda_res: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    score_after: larger => more AI/fake
    We minimize it directly.
    """
    loss_rec = F.l1_loss(x_adv, x)
    loss_det = score_after.mean()
    loss_tv = total_variation_loss(bounded_r)
    loss_res = bounded_r.abs().mean()

    total = (
        lambda_rec * loss_rec
        + lambda_det * loss_det
        + lambda_tv * loss_tv
        + lambda_res * loss_res
    )

    stats = {
        "loss_total": total.detach(),
        "loss_rec": loss_rec.detach(),
        "loss_det": loss_det.detach(),
        "loss_tv": loss_tv.detach(),
        "loss_res": loss_res.detach(),
    }
    return total, stats