"""UQ-conditioned Flow Matching fuser.

This fuser keeps Phase-1 uncertainty as a weak condition and learns a
few-step velocity update in the 256-channel fused BEV latent space.

The main training path is:
    x0 = source_proj(cat(degraded_cam, degraded_lid))
    x1 = teacher_proj(cat(clean_cam, clean_lid)).detach()
    x_t = (1 - t) * stopgrad(x0) + t * stopgrad(x1)
    L_fm = mse(v_theta(x_t, t, cond), stopgrad(x1 - x0))

Detection uses the non-detached path:
    x_out = x0 + v_theta(x0, t=0, cond)
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSERS

from .reliability_head import ReliabilityHead


__all__ = ["UQFlowFuser"]


def _make_convfuser_proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        assert dim % 2 == 0, dim
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float().view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class AdaGNResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None].to(dtype=x.dtype)
        shift = shift[:, :, None, None].to(dtype=x.dtype)

        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class VelocityNet(nn.Module):
    def __init__(
        self,
        latent_ch: int = 256,
        cond_ch: int = 256,
        hidden_ch: int = 256,
        time_dim: int = 128,
        num_blocks: int = 4,
        groups: int = 8,
        max_velocity: float | None = 1.0,
        use_time_condition: bool = True,
    ) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        self.use_time_condition = bool(use_time_condition)
        if self.use_time_condition:
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )
            self.static_time_emb = None
        else:
            self.time_embed = None
            self.time_mlp = None
            self.static_time_emb = nn.Parameter(torch.zeros(time_dim))
        self.in_proj = nn.Conv2d(latent_ch + cond_ch, hidden_ch, 1)
        self.blocks = nn.ModuleList(
            [AdaGNResBlock(hidden_ch, time_dim, groups=groups) for _ in range(num_blocks)]
        )
        self.out_norm = nn.GroupNorm(groups, hidden_ch)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor | float, cond: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), float(t), dtype=torch.float32, device=x.device)
        else:
            t = t.to(device=x.device)
            if t.dim() == 0:
                t = t.expand(x.shape[0])
        if self.use_time_condition:
            emb = self.time_embed(t).to(dtype=self.time_mlp[0].weight.dtype)
            t_emb = self.time_mlp(emb)
        else:
            time_dtype = self.blocks[0].time_mlp[1].weight.dtype
            t_emb = self.static_time_emb.to(device=x.device, dtype=time_dtype)
            t_emb = t_emb.unsqueeze(0).expand(x.shape[0], -1)

        h = self.in_proj(torch.cat([x, cond], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_proj(self.out_act(self.out_norm(h)))
        if self.max_velocity is not None and self.max_velocity > 0:
            max_v = float(self.max_velocity)
            h = torch.nan_to_num(h.float(), nan=0.0, posinf=max_v, neginf=-max_v)
            h = max_v * torch.tanh(h / max_v)
            h = h.to(dtype=x.dtype)
        return h


@FUSERS.register_module()
class UQFlowFuser(nn.Module):
    """Flow Matching fuser with Phase-1 UQ maps as weak conditions."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        reliability: dict | None = None,
        cond_hidden_ch: int = 256,
        flow: dict | None = None,
        stash_uq: bool = True,
        use_uq_condition: bool = True,
        inference_steps: int = 1,
    ) -> None:
        super().__init__()
        assert len(in_channels) == 2, f"expected [cam_ch, lid_ch], got {in_channels}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stash_uq = stash_uq
        self.use_uq_condition = bool(use_uq_condition)
        self.inference_steps = int(inference_steps)

        cam_ch, lid_ch = in_channels
        raw_ch = cam_ch + lid_ch

        rh_kwargs = dict(reliability) if reliability is not None else {}
        rh_kwargs.setdefault("in_ch_cam", cam_ch)
        rh_kwargs.setdefault("in_ch_lid", lid_ch)
        self.reliability_head = ReliabilityHead(**rh_kwargs)

        self.source_proj = _make_convfuser_proj(raw_ch, out_channels)
        self.teacher_proj = _make_convfuser_proj(raw_ch, out_channels)
        self.teacher_proj.load_state_dict(self.source_proj.state_dict())
        for p in self.teacher_proj.parameters():
            p.requires_grad = False
        self.teacher_proj.eval()

        self.cond_proj = nn.Sequential(
            nn.Conv2d(raw_ch + 2, cond_hidden_ch, 1),
            nn.GroupNorm(8, cond_hidden_ch),
            nn.SiLU(),
            nn.Conv2d(cond_hidden_ch, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

        flow_kwargs = dict(flow) if flow is not None else {}
        flow_kwargs.setdefault("latent_ch", out_channels)
        flow_kwargs.setdefault("cond_ch", out_channels)
        self.velocity_net = VelocityNet(**flow_kwargs)

        self.last_uq_cam_logit: torch.Tensor | None = None
        self.last_uq_lid_logit: torch.Tensor | None = None
        self.last_uq_cam: torch.Tensor | None = None
        self.last_uq_lid: torch.Tensor | None = None
        self.last_flow_stats: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_proj.eval()
        return self

    def _compute_uq(
        self, feat_cam: torch.Tensor, feat_lid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return uq_cam, uq_lid

    def _get_uq_for_condition(
        self, feat_cam: torch.Tensor, feat_lid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_uq_condition:
            return self._compute_uq(feat_cam, feat_lid)

        self.last_uq_cam_logit = None
        self.last_uq_lid_logit = None
        self.last_uq_cam = None
        self.last_uq_lid = None

        b, _, h, w = feat_cam.shape
        uq_cam = feat_cam.new_zeros((b, 1, h, w))
        uq_lid = feat_lid.new_zeros((b, 1, h, w))
        return uq_cam, uq_lid

    def _make_cond(
        self,
        feat_cam: torch.Tensor,
        feat_lid: torch.Tensor,
        uq_cam: torch.Tensor,
        uq_lid: torch.Tensor,
    ) -> torch.Tensor:
        cond_raw = torch.cat(
            [feat_cam.detach(), feat_lid.detach(), uq_cam.detach(), uq_lid.detach()],
            dim=1,
        )
        return self.cond_proj(cond_raw)

    def _integrate(
        self, x0: torch.Tensor, cond: torch.Tensor, num_steps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_steps = max(int(num_steps), 1)
        if num_steps == 1:
            v0 = self.velocity_net(x0, 0.0, cond)
            return x0 + v0, v0

        x = x0
        dt = 1.0 / float(num_steps)
        last_v = None
        for k in range(num_steps):
            t = torch.full((x0.shape[0],), k * dt, dtype=torch.float32, device=x0.device)
            last_v = self.velocity_net(x, t, cond)
            x = x + dt * last_v
        assert last_v is not None
        return x, last_v

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == 2, "UQFlowFuser expects [cam_bev, lid_bev]"
        feat_cam, feat_lid = inputs
        uq_cam, uq_lid = self._get_uq_for_condition(feat_cam, feat_lid)
        raw = torch.cat([feat_cam, feat_lid], dim=1)
        x0 = self.source_proj(raw)
        cond = self._make_cond(feat_cam, feat_lid, uq_cam, uq_lid)
        x_out, monitor_v = self._integrate(
            x0, cond, num_steps=self.inference_steps
        )
        self._stash_stats(x0, monitor_v, loss_fm=None)
        return x_out

    def forward_train_flow(
        self,
        deg_inputs: List[torch.Tensor],
        clean_inputs: List[torch.Tensor] | None = None,
        plan_kinds: Sequence[str] | None = None,
        enable_fm: bool = True,
        num_steps: int = 1,
        fm_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert len(deg_inputs) == 2, "expected degraded [cam_bev, lid_bev]"
        feat_cam, feat_lid = deg_inputs
        uq_cam, uq_lid = self._get_uq_for_condition(feat_cam, feat_lid)

        raw_deg = torch.cat([feat_cam, feat_lid], dim=1)
        x0 = self.source_proj(raw_deg)
        cond = self._make_cond(feat_cam, feat_lid, uq_cam, uq_lid)
        x_out, monitor_v = self._integrate(x0, cond, num_steps=num_steps)

        zero_stat = x0.detach().float().new_zeros(())
        aux: dict[str, torch.Tensor] = {
            # Keep DDP log keys identical across ranks. Missing conditional
            # diagnostics must still participate in the same all_reduce calls.
            "stats/flow/clean_l2": zero_stat,
            "stats/flow/corr_l2": zero_stat,
            "stats/flow/clean_l2_shortcut": zero_stat,
            "stats/flow/v_pred_nonfinite": zero_stat,
            "stats/flow/u_t_nonfinite": zero_stat,
            "stats/flow/u_t_rms": zero_stat,
            "stats/flow/v_pred_rms": zero_stat,
            "stats/flow/u_t_abs_max": zero_stat,
            "stats/flow/v_pred_abs_max": zero_stat,
        }
        loss_fm = None
        x1 = None
        if enable_fm:
            if clean_inputs is None:
                x1 = x0.detach()
            else:
                with torch.no_grad():
                    raw_clean = torch.cat(clean_inputs, dim=1)
                    x1 = self.teacher_proj(raw_clean).detach()
                if plan_kinds is not None and any(k == "clean" for k in plan_kinds):
                    clean_mask = torch.tensor(
                        [k == "clean" for k in plan_kinds],
                        dtype=torch.bool,
                        device=x0.device,
                    )
                    if clean_mask.any():
                        with torch.no_grad():
                            clean_l2_diag = (
                                x1[clean_mask] - x0.detach()[clean_mask]
                            ).float().pow(2).mean(dim=(1, 2, 3)).sqrt().mean()
                            aux["stats/flow/clean_l2"] = clean_l2_diag
                        x1 = x1.clone()
                        x1[clean_mask] = x0.detach()[clean_mask]

            x0_sg = x0.detach()
            x1_sg = x1.detach()
            t = torch.rand(x0.shape[0], dtype=torch.float32, device=x0.device)
            t_view = t.view(-1, 1, 1, 1).to(dtype=x0.dtype)
            x_t = (1.0 - t_view) * x0_sg + t_view * x1_sg
            u_t = (x1_sg - x0_sg).detach()
            v_pred_t = self.velocity_net(x_t, t, cond)
            v_pred_t_f = v_pred_t.float()
            u_t_f = u_t.float()
            aux["stats/flow/v_pred_nonfinite"] = (
                ~torch.isfinite(v_pred_t_f)
            ).float().mean()
            aux["stats/flow/u_t_nonfinite"] = (~torch.isfinite(u_t_f)).float().mean()
            clip_v = float(getattr(self.velocity_net, "max_velocity", 1.0) or 1.0)
            v_pred_t_f = torch.nan_to_num(
                v_pred_t_f, nan=0.0, posinf=clip_v, neginf=-clip_v
            )
            u_t_f = torch.nan_to_num(u_t_f, nan=0.0, posinf=clip_v, neginf=-clip_v)
            if fm_loss_type == "mse":
                loss_fm = F.mse_loss(v_pred_t_f, u_t_f)
            elif fm_loss_type == "smooth_l1":
                loss_fm = F.smooth_l1_loss(
                    v_pred_t_f,
                    u_t_f,
                    beta=float(smooth_l1_beta),
                )
            else:
                raise ValueError(f"unsupported fm_loss_type: {fm_loss_type}")
            aux["loss_fm"] = loss_fm

            with torch.no_grad():
                l2 = (x1_sg - x0_sg).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
                aux["stats/flow/u_t_rms"] = u_t_f.pow(2).mean().sqrt()
                aux["stats/flow/v_pred_rms"] = v_pred_t_f.pow(2).mean().sqrt()
                aux["stats/flow/u_t_abs_max"] = u_t_f.abs().amax()
                aux["stats/flow/v_pred_abs_max"] = v_pred_t_f.abs().amax()
                if plan_kinds is not None:
                    corr_mask = torch.tensor(
                        [k != "clean" for k in plan_kinds],
                        dtype=torch.bool,
                        device=x0.device,
                    )
                    if corr_mask.any():
                        aux["stats/flow/corr_l2"] = l2[corr_mask].mean()
                    if (~corr_mask).any():
                        aux["stats/flow/clean_l2_shortcut"] = l2[~corr_mask].mean()

        self._stash_stats(x0, monitor_v, loss_fm=loss_fm)
        aux.update(self.last_flow_stats)
        return x_out, aux

    def _stash_stats(
        self,
        x0: torch.Tensor,
        monitor_v: torch.Tensor,
        loss_fm: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            x0_rms = x0.detach().float().pow(2).mean().sqrt()
            v_rms = monitor_v.detach().float().pow(2).mean().sqrt()
            stats = {
                "stats/flow/x0_rms": x0_rms,
                "stats/flow/v_rms": v_rms,
                "stats/flow/v_x0_ratio": v_rms / x0_rms.clamp_min(1e-6),
            }
            if self.last_uq_cam is not None:
                stats["stats/flow/uq_cam_mean"] = self.last_uq_cam.detach().float().mean()
            if self.last_uq_lid is not None:
                stats["stats/flow/uq_lid_mean"] = self.last_uq_lid.detach().float().mean()
            stats["stats/flow/use_uq_condition"] = x0.detach().float().new_tensor(
                1.0 if self.use_uq_condition else 0.0
            )
            if loss_fm is not None:
                stats["stats/flow/loss_fm_raw"] = loss_fm.detach()
            self.last_flow_stats = stats
