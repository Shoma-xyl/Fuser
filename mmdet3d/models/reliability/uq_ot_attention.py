"""UQ-OT Attention: cross-modal fusion via Sinkhorn-UOT attention.

Replaces standard softmax attention with an unbalanced optimal transport
plan for computing attention weights. The key insight: source marginals
are defined by per-cell reliability (1 - uq), so unreliable positions
contribute less mass to the transport plan, while reliable positions
contribute more — providing a principled, UQ-driven attention mechanism.

Mathematical formulation
========================
Given a BEV grid of N = H*W positions, for each position j we collect
K/V pairs from a local R×R window of both modalities (2*R² sources).

Standard attention:  w = softmax(Q·K^T / √D)
UOT attention:       w = sinkhorn(mu=reliability, nu=1, cost=spatial_dist)
                     applied to V weighted by feature similarity Q·K^T

Concretely, the source marginal at neighbor i is:
    mu_i = sim(Q_j, K_i) * reliability_i

where reliability_i = (1 - uq) of the modality that K_i belongs to,
and sim is a scaled dot-product similarity passed through a smooth
positive activation.

This means UOT considers BOTH feature relevance (Q·K similarity) AND
source reliability (UQ) when computing the transport plan, unlike
standard attention which only considers feature relevance.

The transport cost is the squared spatial distance within the local
window, normalized to [0, 1], encouraging nearby sources.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn_uot import local_sinkhorn_uot, make_local_cost


__all__ = ["UQOTAttention"]


class UQOTAttention(nn.Module):
    """Cross-modal fusion via UQ-guided optimal transport attention.

    For each BEV position, queries from the fused perspective attend to
    K/V pairs from both camera and lidar in a local window. Attention
    weights are computed via Sinkhorn-UOT where source marginals encode
    both feature relevance and per-cell reliability.

    Parameters
    ----------
    cam_channels : int
        Camera BEV feature channels (80).
    lid_channels : int
        LiDAR BEV feature channels (256).
    out_channels : int
        Output feature channels (256, matching decoder input).
    embed_dim : int
        Internal Q/K/V embedding dimension. Default 64.
    window_size : int
        Local attention window (must be odd). Default 5.
    ot_eps : float
        Sinkhorn entropy regularization.
    ot_lambda : float
        UOT KL-marginal relaxation weight.
    ot_iters : int
        Sinkhorn iterations.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lid_channels: int = 256,
        out_channels: int = 256,
        embed_dim: int = 64,
        window_size: int = 5,
        ot_eps: float = 0.1,
        ot_lambda: float = 1.0,
        ot_iters: int = 15,
        residual_uq_thr: float = 0.5,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        assert window_size % 2 == 1
        self.D = embed_dim
        self.R = window_size
        self.ot_eps = ot_eps
        self.ot_lambda = ot_lambda
        self.ot_iters = ot_iters
        self.residual_uq_thr = float(residual_uq_thr)

        # Q/K/V projections
        self.q_proj = nn.Conv2d(cam_channels + lid_channels, embed_dim, 1)
        self.k_cam = nn.Conv2d(cam_channels, embed_dim, 1)
        self.v_cam = nn.Conv2d(cam_channels, embed_dim, 1)
        self.k_lid = nn.Conv2d(lid_channels, embed_dim, 1)
        self.v_lid = nn.Conv2d(lid_channels, embed_dim, 1)

        # Output: embed_dim → out_channels. Keep a direct projection so the
        # transport path can learn amplitude instead of being re-normalized.
        self.out_proj = nn.Conv2d(embed_dim, out_channels, 1, bias=False)

        # Spatial cost buffer (not saved in checkpoint)
        cost = make_local_cost(window_size)
        self.register_buffer("ot_cost", cost, persistent=False)

        # Skip path: preserves pretrained BEVFusion behavior.
        # Initialized from base_fuser weights at load time.
        self.skip = nn.Sequential(
            nn.Conv2d(cam_channels + lid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Learnable gate controlling how much OT-attention contributes.
        # The outer alpha-ramp keeps identity-at-start; this bias just stops
        # the residual from overpowering the pretrained skip branch.
        self.attn_gate = nn.Parameter(torch.tensor(float(gate_init_bias)))

        # Diagnostics for training logs.
        self.last_attn_gate: torch.Tensor | None = None
        self.last_target_gate_mean: torch.Tensor | None = None
        self.last_res_rms: torch.Tensor | None = None
        self.last_skip_rms: torch.Tensor | None = None
        self.last_res_skip_ratio: torch.Tensor | None = None

    def forward(
        self,
        feat_cam: torch.Tensor,   # (B, 80, H, W)
        feat_lid: torch.Tensor,   # (B, 256, H, W)
        uq_cam: torch.Tensor,     # (B, 1, H, W)
        uq_lid: torch.Tensor,     # (B, 1, H, W)
        alpha: float = 1.0,
    ) -> torch.Tensor:
        B, _, H, W = feat_cam.shape
        N = H * W
        R = self.R
        R2 = R * R
        D = self.D
        pad = R // 2

        # --- Skip path (pretrained baseline before ReLU) ---
        skip_pre = self.skip(torch.cat([feat_cam, feat_lid], dim=1))

        # --- OT-Attention path (all in fp32 for Sinkhorn stability) ---
        with torch.cuda.amp.autocast(enabled=False):
            feat_cam_f = feat_cam.float()
            feat_lid_f = feat_lid.float()
            uq_cam_f = uq_cam.float()
            uq_lid_f = uq_lid.float()

            # Project Q, K, V
            Q = self.q_proj(torch.cat([feat_cam_f, feat_lid_f], dim=1))  # (B, D, H, W)
            K_c = self.k_cam(feat_cam_f)   # (B, D, H, W)
            V_c = self.v_cam(feat_cam_f)
            K_l = self.k_lid(feat_lid_f)
            V_l = self.v_lid(feat_lid_f)

            # Unfold K, V into local R×R windows
            def _unfold(x: torch.Tensor) -> torch.Tensor:
                return F.unfold(x, R, padding=pad).view(B, D, R2, N)

            K_c_u = _unfold(K_c)   # (B, D, R², N)
            V_c_u = _unfold(V_c)
            K_l_u = _unfold(K_l)
            V_l_u = _unfold(V_l)

            # Concatenate cam and lid sources: 2*R² sources per position
            K_all = torch.cat([K_c_u, K_l_u], dim=2)   # (B, D, 2R², N)
            V_all = torch.cat([V_c_u, V_l_u], dim=2)   # (B, D, 2R², N)

            # Compute feature similarity. Softplus keeps the source mass
            # positive without hard-thresholding negative matches to zero.
            Q_flat = Q.flatten(2)                        # (B, D, N)
            sim = torch.einsum("bdn, bdkn -> bkn", Q_flat, K_all)  # (B, 2R², N)
            sim = F.softplus(sim / (D ** 0.5))

            # Unfold UQ maps for source reliability
            uq_cam_u = F.unfold(uq_cam_f, R, padding=pad)   # (B, R², N)
            uq_lid_u = F.unfold(uq_lid_f, R, padding=pad)   # (B, R², N)
            reliability = torch.cat([
                1.0 - uq_cam_u,   # cam neighbor reliability
                1.0 - uq_lid_u,   # lid neighbor reliability
            ], dim=1)                                         # (B, 2R², N)

            # Source marginal = similarity × reliability
            # High similarity + high reliability → strong source
            # Low reliability → weak source regardless of similarity
            mu = sim * reliability                            # (B, 2R², N)
            nu = torch.ones(B, 1, N, dtype=torch.float32, device=feat_cam.device)

            # Spatial cost: replicate for both modalities
            cost_2r2 = torch.cat([self.ot_cost, self.ot_cost])  # (2R²,)

            # Sinkhorn-UOT → transport plan
            pi = local_sinkhorn_uot(
                mu, nu, cost_2r2,
                eps=self.ot_eps, lam=self.ot_lambda, n_iter=self.ot_iters,
            )  # (B, 2R², N)

            # Apply transport plan to values
            attended = torch.einsum("bkn, bdkn -> bdn", pi, V_all)  # (B, D, N)
            attended = attended.view(B, D, H, W)

        # Project transport output into the fused feature space.
        attn_out = self.out_proj(attended.to(feat_cam.dtype))

        # Only high-UQ camera cells should receive a strong corrective residual.
        target_gate = (
            (uq_cam - self.residual_uq_thr) / max(1e-6, 1.0 - self.residual_uq_thr)
        ).clamp(0.0, 1.0)
        gate = float(alpha) * torch.sigmoid(self.attn_gate)
        effective_res = gate * target_gate.to(attn_out.dtype) * attn_out
        fused = F.relu(skip_pre + effective_res)

        with torch.no_grad():
            self.last_attn_gate = gate.detach()
            self.last_target_gate_mean = target_gate.detach().float().mean()
            self.last_res_rms = effective_res.detach().float().pow(2).mean().sqrt()
            self.last_skip_rms = skip_pre.detach().float().pow(2).mean().sqrt()
            self.last_res_skip_ratio = (
                self.last_res_rms / self.last_skip_rms.clamp_min(1e-6)
            )

        return fused
