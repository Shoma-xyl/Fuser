"""BCE loss for the reliability head.

Given uq maps (B, 1, H, W) and per-sample corruption plans, compute a
binary cross-entropy loss that trains uq_cam to go high wherever the
camera modality is corrupted AND the camera actually contributes to
that BEV cell (likewise for uq_lid).

Support masks (v2) — quantile-based
-----------------------------------
v1 used a fixed-eps magnitude threshold on the BEV features. For the
LSS camera features this effectively passed EVERY cell (the downsample
Conv2d + BN in DepthLSSTransform leaves a nonzero floor everywhere),
so the cam support mask was ~1.0 everywhere. That diluted the cam loss
across 180×180 = 32,400 cells and matched the LiDAR support (~21%
coverage) poorly — the cam head effectively never learned.

v2 uses a PER-SAMPLE QUANTILE threshold. We pick the top-K% of cells
by magnitude within each sample, so cam support and lid support are
both concentrated on the cells where the modality actually carries
useful information. Defaults: 50% for cam, absolute-eps (=v1 behavior)
for lid, since lid is naturally sparse enough.

Target encoding
---------------
For each sample b in the batch:
    plan[b].kind == "clean" :
        label_cam[b] = 0,  label_lid[b] = 0
    plan[b].kind == "cam"   :
        label_cam[b] = 1 * support_cam[b]
        label_lid[b] = 0
    plan[b].kind == "lid"   :
        label_cam[b] = 0
        label_lid[b] = 1 * support_lid[b]
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


__all__ = ["compute_uq_bce"]


def _support_mask_eps(feat_bev: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Legacy v1 mask: cells with any feature magnitude > eps. Kept for lid,
    where it works well (lid BEV feat is naturally sparse).
    """
    with torch.no_grad():
        magn = feat_bev.abs().sum(dim=1, keepdim=True)            # (B, 1, H, W)
        mask = (magn > eps).float()
    return mask


def _support_mask_quantile(
    feat_bev: torch.Tensor, keep_frac: float = 0.5
) -> torch.Tensor:
    """Per-sample top-K% mask: 1 where feature magnitude is in the top
    `keep_frac` fraction of cells within that sample.

    We use quantile-per-sample rather than a global threshold to stay
    robust to per-batch magnitude drift (e.g., when cam is corrupted
    globally, overall magnitude shrinks; we still want to monitor the
    relatively-strongest cells).

    Note: torch.quantile requires float32/float64; we cast fp16 inputs
    to fp32 for the quantile computation only. The mask itself is float.
    """
    with torch.no_grad():
        B = feat_bev.shape[0]
        # Cast to fp32 for quantile compatibility; magn is also fp32.
        magn = feat_bev.float().abs().sum(dim=1, keepdim=True)    # (B, 1, H, W)
        flat = magn.view(B, -1)                                   # (B, H*W)
        q = torch.quantile(flat, 1.0 - keep_frac, dim=1, keepdim=True)
        thr = q.view(B, 1, 1, 1)
        mask = (magn > thr).float()
    return mask


def compute_uq_bce(
    logit_cam: torch.Tensor,             # (B, 1, H, W)   RAW LOGITS
    logit_lid: torch.Tensor,             # (B, 1, H, W)   RAW LOGITS
    feat_cam_bev: torch.Tensor,          # (B, C_cam, H, W) — for support mask
    feat_lid_bev: torch.Tensor,          # (B, C_lid, H, W) — for support mask
    plan_kinds: Sequence[str],           # len B, each in {"clean","cam","lid"}
    label_smoothing: float = 0.0,
    loss_cam_weight: float = 1.0,
    loss_lid_weight: float = 1.0,
    cam_support_keep_frac: float = 0.5,
) -> dict:
    """BCE loss with logits, per-modality weight, and quantile-based cam support.

    Numerically stable in fp16 via F.binary_cross_entropy_with_logits
    (uses log-sum-exp trick internally; no log(0) risk).

    Parameters
    ----------
    logit_cam, logit_lid : RAW logits (pre-sigmoid), shape (B, 1, H, W).
    feat_cam_bev, feat_lid_bev : pre-fuser BEV features for support masks.
    plan_kinds : per-sample corruption kind.
    loss_cam_weight, loss_lid_weight : per-modality weights.
    cam_support_keep_frac : top-fraction of cells retained for cam loss.
    """
    assert logit_cam.shape == logit_lid.shape, (logit_cam.shape, logit_lid.shape)
    B = logit_cam.shape[0]
    assert len(plan_kinds) == B, (len(plan_kinds), B)

    # Support masks (no grad).
    support_cam = _support_mask_quantile(feat_cam_bev,
                                          keep_frac=cam_support_keep_frac)
    support_lid = _support_mask_eps(feat_lid_bev)

    # Build per-sample label maps (float, in [0, 1]).
    label_cam = torch.zeros_like(logit_cam)
    label_lid = torch.zeros_like(logit_lid)
    for b, k in enumerate(plan_kinds):
        if k == "cam":
            label_cam[b] = support_cam[b]
        elif k == "lid":
            label_lid[b] = support_lid[b]
        elif k == "clean":
            pass
        else:
            raise ValueError(f"bad plan kind: {k}")

    # Label smoothing.
    if label_smoothing > 0:
        label_cam = label_cam * (1 - 2 * label_smoothing) + label_smoothing
        label_lid = label_lid * (1 - 2 * label_smoothing) + label_smoothing

    # Numerically stable BCE from logits. Compute per-cell with
    # reduction='none' so we can apply the support-weight mask ourselves.
    # IMPORTANT: cast to float32 so that even if fp16 is on, the BCE math
    # doesn't under/overflow for very confident predictions.
    logit_cam_f = logit_cam.float()
    logit_lid_f = logit_lid.float()
    label_cam_f = label_cam.float()
    label_lid_f = label_lid.float()

    bce_cam_per_cell = F.binary_cross_entropy_with_logits(
        logit_cam_f, label_cam_f, reduction="none"
    )
    bce_lid_per_cell = F.binary_cross_entropy_with_logits(
        logit_lid_f, label_lid_f, reduction="none"
    )

    w_cam = support_cam
    w_lid = support_lid
    n_cam = w_cam.sum().clamp_min(1.0)
    n_lid = w_lid.sum().clamp_min(1.0)
    loss_cam_raw = (bce_cam_per_cell * w_cam).sum() / n_cam
    loss_lid_raw = (bce_lid_per_cell * w_lid).sum() / n_lid

    # Weighted combination.
    w_sum = float(loss_cam_weight + loss_lid_weight)
    loss_cam = loss_cam_weight * loss_cam_raw
    loss_lid = loss_lid_weight * loss_lid_raw
    loss = (loss_cam + loss_lid) / max(w_sum, 1e-6)

    # Logging
    with torch.no_grad():
        pred_cam = (logit_cam_f > 0).float()       # sigmoid>0.5 iff logit>0
        pred_lid = (logit_lid_f > 0).float()
        acc_cam = ((pred_cam == (label_cam_f > 0.5).float()).float() * w_cam).sum() / n_cam
        acc_lid = ((pred_lid == (label_lid_f > 0.5).float()).float() * w_lid).sum() / n_lid
        frac_cam = w_cam.mean()
        frac_lid = w_lid.mean()

    return {
        "loss": loss,
        "loss_cam": loss_cam.detach(),
        "loss_lid": loss_lid.detach(),
        "acc_cam": acc_cam,
        "acc_lid": acc_lid,
        "support_cam": frac_cam,
        "support_lid": frac_lid,
    }
