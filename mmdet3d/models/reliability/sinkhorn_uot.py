"""Local Sinkhorn-UOT for cross-modal BEV feature completion.

Problem statement
-----------------
For each BEV cell j in a grid of N = H*W cells, we want to transport
information from a local (k x k) neighborhood of a reliable modality
to j, where the transport is "gated" by the reliability of j in the
unreliable modality.

Concretely, to complete camera features at position j using nearby
LiDAR features:

    source (per neighbor i in window around j):  mu_i = 1 - uq_lid[i]
    sink   (at position j):                      nu_j = 1   (constant)
    cost   (neighbor i to center j):             C_i  = ||pos_i - pos_j||^2
                                                        normalized to [0, 1]

This is an unbalanced (UOT) transport because sum(mu) != sum(nu) in
general.  We use Sinkhorn-Knopp iteration with entropy regularization
and KL-divergence relaxation on both marginals (Chizat et al. 2018).

Critical: everything below runs in fp32 regardless of outer autocast
context. Sinkhorn involves exp/log and divisions that are numerically
fragile in fp16.

Return shape: transport plan pi of shape (B, k*k, N) where each
pi[b, i, j] tells us how much mass flows from the i-th neighbor of
center j to center j.
"""
from __future__ import annotations

import torch


__all__ = ["make_local_cost", "local_sinkhorn_uot"]


def make_local_cost(k: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Squared L2 distance from the center of a (k x k) window to each
    cell, normalized so the maximum cost is 1.  Shape: (k*k,), fp32.

    For k=7 the center cell has cost 0; the four corners have cost 1.
    All costs are distances in "cell" units (not meters) — that's fine
    because we only ever use them divided by a Sinkhorn epsilon.
    """
    assert k % 2 == 1, "window size must be odd"
    half = (k - 1) // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    d2 = ys * ys + xs * xs                     # (k, k)
    d2 = d2 / d2.max().clamp_min(1e-6)         # normalize to [0, 1]
    return d2.flatten().to(device)              # (k*k,)


def local_sinkhorn_uot(
    mu: torch.Tensor,                      # (B, k2, N) — source marginal per neighbor
    nu: torch.Tensor,                      # (B, 1,  N) — sink marginal at center
    cost: torch.Tensor,                    # (k2,)      — window-local cost
    eps: float = 0.1,
    lam: float = 1.0,
    n_iter: int = 20,
    safe_eps: float = 1e-8,
) -> torch.Tensor:
    """Batched local Sinkhorn-UOT.

    Parameters
    ----------
    mu, nu : source and sink marginals. Both non-negative floats.
    cost   : (k*k,) cost vector for the local window. Treated as constant
             across the batch and across spatial positions.
    eps    : entropy regularization. Larger => smoother plan.
    lam    : KL-divergence weight on marginal constraints.
    n_iter : number of Sinkhorn iterations. 20 is plenty for k*k <= 49.
    safe_eps : numerical floor to prevent div-by-zero inside the loop.

    Returns
    -------
    pi : (B, k2, N) transport plan, fp32. pi[b, i, j] is the mass
         transported from neighbor i into center j.

    Notes
    -----
    - Runs entirely in fp32. Caller can pass fp16 inputs; we cast up.
    - All operations are batched; no Python loops over B or N.
    """
    # Always compute in fp32 for stability — Sinkhorn is numerically
    # fragile in fp16 especially with entropy regularization.
    mu32 = mu.float()
    nu32 = nu.float()
    cost32 = cost.float()

    # K = exp(-cost / eps), shape (k2,). All positive.
    # Clamp cost/eps to prevent K from underflowing to 0; at most ~1/eps.
    K = torch.exp(-cost32 / eps)                 # (k2,)
    # Broadcast to (1, k2, 1) for batched ops.
    K_b = K.view(1, -1, 1)                       # (1, k2, 1)

    # Unbalanced Sinkhorn scaling exponent.  tau = lam / (lam + eps).
    tau = lam / (lam + eps)

    # Initialize dual/scaling variables to 1.
    # u: (B, k2, N), v: (B, 1, N)
    B, k2, N = mu32.shape
    # Keep a tiny floor for numerical safety, but do not wash out the UQ
    # weighting by forcing weak sources up to a large common baseline.
    mu32 = mu32.clamp_min(1e-6)

    u = torch.ones_like(mu32)
    v = torch.ones_like(nu32)

    for _ in range(n_iter):
        # u <- (mu / (K * v))^tau      (source update)
        Kv = K_b * v                              # (B, k2, N)
        u = (mu32 / (Kv + safe_eps)).clamp_min(safe_eps).pow(tau)
        # v <- (nu / (K^T * u))^tau    (sink update)
        # K^T * u = sum_i K_i * u_i    (for each batch, spatial loc)
        Ktu = (K_b * u).sum(dim=1, keepdim=True)  # (B, 1, N)
        v = (nu32 / (Ktu + safe_eps)).clamp_min(safe_eps).pow(tau)

    pi = u * K_b * v                              # (B, k2, N)
    return pi
