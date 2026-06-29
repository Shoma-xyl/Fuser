"""Reliability head for learned per-BEV-cell uncertainty.

Predicts two scalar maps in [0, 1]:
    uq_cam : high where camera BEV feature is unreliable
    uq_lid : high where LiDAR BEV feature is unreliable

Input
-----
Concatenation of feat_cam_bev (B, C_cam, H, W) and feat_lid_bev
(B, C_lid, H, W). In the default convfuser config these are 80 and 256
channels respectively at resolution 180x180.

Design choices
--------------
1. Capacity ~1.7M params. The user explicitly asked for non-trivial
   capacity ("don't under-parameterize, it won't learn").
2. Dilated convolutions so the BEV receptive field covers ~12 m (~31
   BEV cells at 0.4 m/cell), which is about one object-scale of context.
3. Identity-at-init: the final conv has a strongly negative bias so
   sigmoid(bias) ~ 0.02 at initialization. This makes (1 - uq) ~ 1
   at initialization, so inserting the gate into a pretrained
   BEVFusion does NOT perturb its output until training moves uq.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ReliabilityHead"]


class _ConvBlock(nn.Module):
    """Conv3x3 (dilated) + BN + GELU + optional Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = dilation  # to keep spatial size with 3x3 conv
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class ReliabilityHead(nn.Module):
    """Small CNN that maps concatenated BEV features to per-cell
    per-modality reliability.

    Parameters
    ----------
    in_ch_cam : int
        Number of camera BEV feature channels (e.g. 80).
    in_ch_lid : int
        Number of LiDAR BEV feature channels (e.g. 256).
    hidden_ch : int
        Base hidden width (default 256).
    init_bias : float
        Bias initialization of the final layer. Strong negative =
        (sigmoid ~= 0) at init. Default -4.0 gives sigmoid(-4) = 0.0180.
    """

    def __init__(
        self,
        in_ch_cam: int = 80,
        in_ch_lid: int = 256,
        hidden_ch: int = 256,
        init_bias: float = -4.0,
    ) -> None:
        super().__init__()
        in_ch = in_ch_cam + in_ch_lid
        # Dilated stack: 1, 2, 4, 1 dilation schedule -> RF ~31 cells ~12 m BEV
        self.block1 = _ConvBlock(in_ch, hidden_ch, dilation=1, dropout=0.1)
        self.block2 = _ConvBlock(hidden_ch, hidden_ch, dilation=2, dropout=0.1)
        self.block3 = _ConvBlock(hidden_ch, hidden_ch // 2, dilation=4, dropout=0.1)
        self.block4 = _ConvBlock(hidden_ch // 2, hidden_ch // 4, dilation=1, dropout=0.0)
        # final head: 2 outputs (cam, lid)
        self.head = nn.Conv2d(hidden_ch // 4, 2, kernel_size=1)
        # identity-at-init: output ~ sigmoid(-4) ~ 0.018
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(
        self, feat_cam_bev: torch.Tensor, feat_lid_bev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logit_cam, logit_lid), both (B, 1, H, W) — RAW LOGITS.

        v2: We return logits (NOT sigmoid-ed probabilities) so downstream
        can use binary_cross_entropy_with_logits for numerical stability
        in fp16. Callers that need probabilities should apply sigmoid
        themselves.
        """
        assert feat_cam_bev.dim() == 4 and feat_lid_bev.dim() == 4
        assert feat_cam_bev.shape[0] == feat_lid_bev.shape[0]
        assert feat_cam_bev.shape[2:] == feat_lid_bev.shape[2:], (
            feat_cam_bev.shape, feat_lid_bev.shape
        )
        x = torch.cat([feat_cam_bev, feat_lid_bev], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.head(x)                       # (B, 2, H, W)
        logit_cam = logits[:, 0:1, :, :]             # (B, 1, H, W)
        logit_lid = logits[:, 1:2, :, :]             # (B, 1, H, W)
        return logit_cam, logit_lid

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
