"""UQ-aware fusion: per-modality reliability drives fusion strategy.

Supported strategies
====================

1. ``gate`` (Phase-1-compatible, ablation baseline)
    feat_cam_gated = feat_cam * (1 - alpha * uq_cam)
    feat_lid_gated = feat_lid * (1 - alpha * uq_lid)
    out = base_fuser(cat([gated_cam, gated_lid]))

2. ``ot_attn`` (Phase-2 main method)
    UQ-OT Attention: cross-modal fusion where attention weights are
    computed via Sinkhorn-UOT with source marginals = similarity ×
    reliability. Replaces the old compose + MLP + base_fuser pipeline.
    See uq_ot_attention.py for details.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from mmdet3d.models.builder import FUSERS

from .reliability_head import ReliabilityHead
from .uq_ot_attention import UQOTAttention


__all__ = ["UQFuser"]


@FUSERS.register_module()
class UQFuser(nn.Module):
    """Fuser with learned per-cell reliability and UQ-guided fusion.

    Parameters
    ----------
    in_channels : list[int]
        [camera_in_ch, lidar_in_ch]. Camera first.
    out_channels : int
        Output feature channels (256).
    fusion_strategy : str
        ``"gate"`` or ``"ot_attn"``.
    reliability : dict | None
        Kwargs for ReliabilityHead.
    stash_uq : bool
        Store uq tensors for outer BCE loss computation.
    gate_ramp_steps : int
        Linearly ramp alpha from 0→1 over this many training steps.
    ot_attn : dict | None
        Kwargs for UQOTAttention (embed_dim, window_size, ot_eps, etc.).
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        fusion_strategy: str = "gate",
        reliability: dict | None = None,
        stash_uq: bool = True,
        gate_ramp_steps: int = 0,
        ot_attn: dict | None = None,
    ) -> None:
        super().__init__()
        assert len(in_channels) == 2, f"expected [cam_ch, lid_ch], got {in_channels}"
        assert fusion_strategy in ("gate", "ot_attn"), fusion_strategy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_strategy = fusion_strategy
        self.stash_uq = stash_uq
        self.gate_ramp_steps = int(gate_ramp_steps)

        cam_ch, lid_ch = in_channels

        # Reliability head (shared across strategies)
        rh_kwargs = dict(reliability) if reliability is not None else {}
        rh_kwargs.setdefault("in_ch_cam", cam_ch)
        rh_kwargs.setdefault("in_ch_lid", lid_ch)
        self.reliability_head = ReliabilityHead(**rh_kwargs)

        # Strategy-specific modules
        if fusion_strategy == "gate":
            self.base_fuser = nn.Sequential(
                nn.Conv2d(cam_ch + lid_ch, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_strategy == "ot_attn":
            attn_kwargs = dict(ot_attn) if ot_attn is not None else {}
            attn_kwargs.setdefault("cam_channels", cam_ch)
            attn_kwargs.setdefault("lid_channels", lid_ch)
            attn_kwargs.setdefault("out_channels", out_channels)
            self.ot_attention = UQOTAttention(**attn_kwargs)

        # Stash for outer BCE loss
        self.last_uq_cam_logit: torch.Tensor | None = None
        self.last_uq_lid_logit: torch.Tensor | None = None
        self.last_uq_cam: torch.Tensor | None = None
        self.last_uq_lid: torch.Tensor | None = None

        # Gate ramp counter (persistent for checkpoint resume)
        self.register_buffer(
            "_gate_step", torch.zeros(1, dtype=torch.long), persistent=True
        )

    def _current_alpha(self) -> float:
        if not self.training or self.gate_ramp_steps <= 0:
            return 1.0
        step = int(self._gate_step.item())
        return min(1.0, step / float(self.gate_ramp_steps))

    def _step_counter(self) -> None:
        if self.training and self.gate_ramp_steps > 0:
            self._gate_step.add_(1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == 2, "UQFuser expects [cam_bev, lid_bev]"
        feat_cam, feat_lid = inputs
        assert feat_cam.shape[1] == self.in_channels[0]
        assert feat_lid.shape[1] == self.in_channels[1]

        # Reliability head (detached inputs: BCE loss won't reach encoders,
        # but detection loss still reaches head weights via compose/attention)
        logit_cam, logit_lid = self.reliability_head(
            feat_cam.detach(), feat_lid.detach()
        )
        uq_cam = torch.sigmoid(logit_cam)
        uq_lid = torch.sigmoid(logit_lid)

        if self.stash_uq:
            self.last_uq_cam_logit = logit_cam
            self.last_uq_lid_logit = logit_lid
            self.last_uq_cam = uq_cam
            self.last_uq_lid = uq_lid

        alpha = self._current_alpha()
        self._step_counter()

        if self.fusion_strategy == "gate":
            return self._forward_gate(feat_cam, feat_lid, uq_cam, uq_lid, alpha)
        elif self.fusion_strategy == "ot_attn":
            return self._forward_ot_attn(feat_cam, feat_lid, uq_cam, uq_lid, alpha)
        else:
            raise ValueError(self.fusion_strategy)

    def _forward_gate(
        self, feat_cam: torch.Tensor, feat_lid: torch.Tensor,
        uq_cam: torch.Tensor, uq_lid: torch.Tensor, alpha: float,
    ) -> torch.Tensor:
        gated_cam = feat_cam * (1.0 - alpha * uq_cam)
        gated_lid = feat_lid * (1.0 - alpha * uq_lid)
        return self.base_fuser(torch.cat([gated_cam, gated_lid], dim=1))

    def _forward_ot_attn(
        self, feat_cam: torch.Tensor, feat_lid: torch.Tensor,
        uq_cam: torch.Tensor, uq_lid: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        return self.ot_attention(feat_cam, feat_lid, uq_cam, uq_lid, alpha=alpha)
