"""Conditional flow matching: AdaLN-Zero velocity net + distributional source prior.

Source ``x_0 = mu + sigma * eps`` with ``(mu, sigma)`` learned per sample.
Training is I-CFM (linear path, no OT) with
``L = MSE(v, x_1 - x_0) + lambda * beta_NLL(mu, sigma, x_1)``.
Inference integrates explicit Euler from ``x_0`` for ``n_steps`` steps.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for ``t in [0, 1]``."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1))
        )
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNZeroBlock(nn.Module):
    """DiT-style AdaLN-Zero block.

    ``h' = (1 + gamma) * LayerNorm(h) + beta``,
    ``h <- h + alpha * FFN(h')``. The Linear that produces
    ``(gamma, beta, alpha)`` is zero-init so the block starts as identity.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta, alpha = self.adaLN_modulation(cond).chunk(3, dim=-1)
        h = (1 + gamma) * self.norm(x) + beta
        return x + alpha * self.ffn(h)


class AdaLNVelocityNet(nn.Module):
    """``v(x_t, t, z_eeg)`` with AdaLN-Zero modulation. Output is zero-init."""

    def __init__(
        self,
        flow_dim: int = 64,
        cond_dim: int = 512,
        hidden: int = 512,
        n_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.eeg_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.input_proj = nn.Linear(flow_dim, hidden)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(hidden) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(hidden)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.output_proj = nn.Linear(hidden, flow_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, z_eeg: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.time_proj(self.time_embed(t)) + self.eeg_proj(z_eeg)
        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        gamma, beta = self.final_adaLN(cond).chunk(2, dim=-1)
        h = (1 + gamma) * self.final_norm(h) + beta
        return self.output_proj(h)


class DistributionalPrior(nn.Module):
    """Learned per-sample Gaussian over the flow source.

    ``sigma = softplus(raw) + sigma_floor``; the floor prevents posterior
    collapse. ``sigma_head.bias`` is initialised so ``sigma(t=0) ~= init_sigma``.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        flow_dim: int = 64,
        hidden_1: int = 256,
        hidden_2: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 0.05,
        init_sigma: float = 0.2,
    ):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2), nn.GELU(), nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_2, flow_dim)
        self.sigma_head = nn.Linear(hidden_2, flow_dim)

        # softplus^{-1}(init_sigma - sigma_floor) so sigma starts near init_sigma.
        start_raw = max(init_sigma - sigma_floor, 1e-4)
        init_bias = math.log(math.expm1(start_raw)) if start_raw > 1e-6 else math.log(start_raw)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, init_bias)

    def forward(self, z_eeg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z_eeg)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + self.sigma_floor
        return mu, sigma


def beta_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Beta-NLL loss (Seitzer et al., ICLR 2022).

    Detaches ``sigma^{2 beta}`` in the weight to decouple mean and variance
    gradients. ``beta=0`` recovers MSE, ``beta=1`` naive NLL, 0.5 is default.
    """
    var = sigma.pow(2).clamp(min=eps)
    residual_sq = (target - mu).pow(2)
    nll = 0.5 * (residual_sq / var + var.log())
    if beta == 0.0:
        return nll.mean()
    weight = var.detach().pow(beta)
    return (weight * nll).mean()


@torch.no_grad()
def euler_integrate(
    velocity_net: nn.Module,
    x0: torch.Tensor,
    z_eeg: torch.Tensor,
    n_steps: int = 50,
) -> torch.Tensor:
    """Integrate ``dx/dt = v(x, t, z_eeg)`` from t=0 to t=1 with explicit Euler."""
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        x = x + velocity_net(x, t, z_eeg) * dt
    return x
