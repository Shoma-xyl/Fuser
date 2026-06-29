"""Synthetic corruption for training the reliability head.

During training, we apply one of six corruption types to either the
camera or the LiDAR modality with a known probability. The same
corruption decision is returned alongside the corrupted inputs so the
training loop knows which modality is unreliable and can supervise the
reliability head accordingly.

Per-sample schedule (independent per batch element) — v2:
    clean               with prob 0.30
    camera corruption   with prob 0.45   (C1/C2/C3/C6 uniform)
    LiDAR corruption    with prob 0.25   (C4/C5 uniform)

Corruptions — v2 (strengthened)
-------------------------------
C1 "night"     : multiply pixel values by u ~ U(0.02, 0.08) per-sample
                 (was U(0.10, 0.25) in v1 — now much darker)
C2 "rain/fog"  : Gaussian blur with sigma ~ U(4.0, 7.0), kernel 15
                 (was U(2.0, 4.0) kernel 7 in v1 — now heavy blur)
C3 "cam drop"  : zero k views, k ~ {2, 3, 4}
                 (was {1, 2, 3} in v1 — now at least 2 dropped)
C4 "lid sparse": keep u ~ U(0.20, 0.35) fraction of points
C5 "lid noise" : add N(0, sigma) to xyz, sigma ~ U(0.2, 0.5) m
C6 "cam occl"  : per-view, paste 2-4 dark rectangles (local spatial
                 corruption — simulates dirt / rain drops / partial
                 occlusion). Provides LOCAL (not global) cam signal
                 that complements C1/C2/C3.

All corruptions are applied AFTER normal data loading but BEFORE the
model forward. No distribution shift is introduced in eval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F


__all__ = ["CorruptionPlan", "make_corruption_plan", "apply_corruption"]


@dataclass
class CorruptionPlan:
    """Plan per batch element.

    kind : "clean" | "cam" | "lid"
    sub  : optional sub-corruption identifier (C1..C5) or None
    """
    kind: str
    sub: str | None = None


def make_corruption_plan(
    batch_size: int,
    generator: torch.Generator | None = None,
    clean_prob: float = 0.30,
    cam_prob: float = 0.45,
) -> List[CorruptionPlan]:
    """Sample a plan for each of `batch_size` items independently.

    Probabilities:
        clean               : clean_prob       (default 0.30)
        camera corruption   : cam_prob         (default 0.45, C1/C2/C3/C6 uniform)
        LiDAR corruption    : 1 - clean - cam  (default 0.25, C4/C5 uniform)

    Phase 1 uses defaults (30/45/25). Phase 2 should use higher clean_prob
    (e.g. 0.70) to avoid contaminating BN running stats with corrupted data.
    """
    rng = generator if generator is not None else torch.Generator()
    cam_threshold = clean_prob + cam_prob
    plans: List[CorruptionPlan] = []
    for _ in range(batch_size):
        r = torch.rand(1, generator=rng).item()
        if r < clean_prob:
            plans.append(CorruptionPlan(kind="clean", sub=None))
        elif r < cam_threshold:
            c = torch.randint(0, 4, (1,), generator=rng).item()
            plans.append(CorruptionPlan(kind="cam", sub=["C1", "C2", "C3", "C6"][c]))
        else:
            c = torch.randint(0, 2, (1,), generator=rng).item()
            plans.append(CorruptionPlan(kind="lid", sub=["C4", "C5"][c]))
    return plans


# -----------------------------------------------------------------------------
# Per-sample camera corruption
# -----------------------------------------------------------------------------

def _rand_uniform(lo: float, hi: float, rng: torch.Generator) -> float:
    """Single uniform scalar in [lo, hi) using a CPU generator, regardless
    of device. Kept on CPU to avoid the 'generator on device' API pitfalls.
    """
    return lo + (hi - lo) * float(torch.rand(1, generator=rng).item())


def _corrupt_cam_C1(img: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Brightness reduction (night).

    STRENGTHENED: was U(0.10, 0.25). Now U(0.02, 0.08) — much darker,
    harder for BN to normalize away since signal magnitude is ~5x lower.

    img: (N_cam, 3, H, W), already normalized (ImageNormalize), so a
    multiplicative factor here still represents an overall scale change
    that propagates through the frozen backbone.
    """
    scale = _rand_uniform(0.02, 0.08, rng)
    return img * scale


def _corrupt_cam_C2(img: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Gaussian blur (rain/fog).

    STRENGTHENED: was sigma U(2.0, 4.0) kernel 7. Now sigma U(4.0, 7.0)
    kernel 15 — heavy blur that obliterates fine texture.
    """
    sigma = _rand_uniform(4.0, 7.0, rng)
    ksize = 15
    # Build 1D gaussian kernel
    coords = torch.arange(ksize, dtype=img.dtype, device=img.device) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k1d = g.view(1, 1, ksize)
    # Apply separable blur per channel
    N, C, H, W = img.shape
    x = img.reshape(N * C, 1, H, W)
    x = F.conv2d(x, k1d.view(1, 1, 1, ksize), padding=(0, ksize // 2))
    x = F.conv2d(x, k1d.view(1, 1, ksize, 1), padding=(ksize // 2, 0))
    return x.reshape(N, C, H, W)


def _corrupt_cam_C3(img: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Camera dropout: zero out k randomly chosen views.

    Mildly strengthened: was k in [1, 3]. Now k in [2, 4] (at least 2
    views dropped). Ensures a strong local signal per BEV region.
    """
    N_cam = img.shape[0]
    # k in [2, min(4, N_cam)]
    k_min = 2
    k_max = min(4, N_cam)
    if k_max < k_min:
        k = k_max
    else:
        k = int(torch.randint(k_min, k_max + 1, (1,), generator=rng).item())
    perm = torch.randperm(N_cam, generator=rng)
    drop_idx = perm[:k]
    out = img.clone()
    out[drop_idx] = 0.0
    return out


def _corrupt_cam_C6(img: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Random rectangular occlusion per view (simulates dirt/rain drops).

    For each of the N_cam views independently, paste 2-4 dark rectangles
    at random locations covering ~10-25% of the image area each. This
    provides LOCAL (spatial) cam-corruption signal that C1/C2/C3 (all
    global) lack.
    """
    out = img.clone()
    N_cam, C, H, W = out.shape
    for v in range(N_cam):
        n_rect = int(torch.randint(2, 5, (1,), generator=rng).item())
        for _ in range(n_rect):
            # Rect of size 25-45% H x 25-45% W
            h_frac = _rand_uniform(0.25, 0.45, rng)
            w_frac = _rand_uniform(0.25, 0.45, rng)
            rh = max(1, int(H * h_frac))
            rw = max(1, int(W * w_frac))
            top = int(torch.randint(0, max(1, H - rh + 1), (1,), generator=rng).item())
            left = int(torch.randint(0, max(1, W - rw + 1), (1,), generator=rng).item())
            # Dark (close to normalized-black) rectangle
            out[v, :, top:top + rh, left:left + rw] = _rand_uniform(-2.0, -1.5, rng)
    return out


def corrupt_camera(
    img: torch.Tensor, sub: str, rng: torch.Generator
) -> torch.Tensor:
    """img: (N_cam, 3, H, W). Returns corrupted image."""
    if sub == "C1":
        return _corrupt_cam_C1(img, rng)
    elif sub == "C2":
        return _corrupt_cam_C2(img, rng)
    elif sub == "C3":
        return _corrupt_cam_C3(img, rng)
    elif sub == "C6":
        return _corrupt_cam_C6(img, rng)
    else:
        raise ValueError(f"unknown cam sub: {sub}")


# -----------------------------------------------------------------------------
# Per-sample LiDAR corruption
# -----------------------------------------------------------------------------

def _corrupt_lid_C4(pts: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Random subsampling."""
    N = pts.shape[0]
    keep_frac = _rand_uniform(0.20, 0.35, rng)
    n_keep = max(1000, int(N * keep_frac))
    if n_keep >= N:
        return pts
    # Use CPU permutation then move indices to points' device.
    perm = torch.randperm(N, generator=rng)[:n_keep]
    if pts.is_cuda:
        perm = perm.to(pts.device)
    return pts[perm]


def _corrupt_lid_C5(pts: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """XYZ Gaussian noise."""
    sigma = _rand_uniform(0.2, 0.5, rng)
    noise = torch.randn(pts.shape[0], 3, generator=rng) * sigma
    if pts.is_cuda:
        noise = noise.to(pts.device)
    out = pts.clone()
    out[:, :3] = out[:, :3] + noise
    return out


def corrupt_lidar(
    pts: torch.Tensor, sub: str, rng: torch.Generator
) -> torch.Tensor:
    """pts: (N, >=3). Returns corrupted points."""
    if sub == "C4":
        return _corrupt_lid_C4(pts, rng)
    elif sub == "C5":
        return _corrupt_lid_C5(pts, rng)
    else:
        raise ValueError(f"unknown lid sub: {sub}")


# -----------------------------------------------------------------------------
# Dispatch: apply plan to a batch
# -----------------------------------------------------------------------------

def apply_corruption(
    img: torch.Tensor,                      # (B, N_cam, 3, H, W)
    points: Sequence[torch.Tensor],         # list of (N_pts_b, >=3), len B
    plans: Sequence[CorruptionPlan],
    generator: torch.Generator | None = None,
):
    """Apply corruptions in-place on copies.

    Returns
    -------
    img_out : (B, N_cam, 3, H, W)     corrupted image tensor (same shape)
    points_out : list of tensors       corrupted points list
    """
    rng = generator if generator is not None else torch.Generator()
    B = img.shape[0]
    assert len(plans) == B, (len(plans), B)
    assert len(points) == B, (len(points), B)

    img_out = img.clone()
    points_out: List[torch.Tensor] = [p for p in points]

    for b, plan in enumerate(plans):
        if plan.kind == "cam":
            img_out[b] = corrupt_camera(img_out[b], plan.sub, rng)
        elif plan.kind == "lid":
            points_out[b] = corrupt_lidar(points_out[b], plan.sub, rng)
        # clean: do nothing
    return img_out, points_out
