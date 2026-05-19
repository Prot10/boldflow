"""Ablation variants of BOLDFlow used in the paper's Tables 2-3.

The full BOLDFlow uses a learned per-sample distributional prior. This
module ships the **point-prior** ablation, which replaces the distributional
prior with a deterministic ``mu`` and a noise-annealed source. This is the
"+ AdaLN-Zero CFM, detached prior" row of Table 2 (T.Corr=0.321, FC=0.442).

For paper Tables 1, 3 and the rest of the ablation rows we point readers
back to the relevant configuration knobs (e.g. ``model.embed_dim``,
``model.n_inference_steps``, ``data.tmin``) and to the exposition in the
paper appendices; the ablations are mechanical sweeps over the headline
architecture rather than independent codepaths.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from boldflow.encoders import REVEEncoder, MSSEncoder
from boldflow.flow import AdaLNVelocityNet, euler_integrate
from boldflow.model import BoldFlow


def _point_prior_mlp(
    cond_dim: int = 512, flow_dim: int = 64,
    hidden_1: int = 256, hidden_2: int = 128, dropout: float = 0.1,
) -> nn.Sequential:
    """Deterministic prior MLP. Layout matches the released p28c checkpoint.

    Keys: ``detached_prior_net.{0,3,6}.weight`` (three ``nn.Linear`` layers
    at positions 0, 3, 6 inside the ``Sequential``).
    """
    return nn.Sequential(
        nn.Linear(cond_dim, hidden_1), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_1, hidden_2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_2, flow_dim),
    )


class BoldFlowPointPrior(nn.Module):
    """Ablation: deterministic prior + sigma-annealed source + OT coupling.

    Differences vs. :class:`BoldFlow`:
      * ``self.detached_prior_net`` replaces ``self.distributional_prior_head``.
      * Source is ``x_0 = mu.detach() + sigma * eps`` with ``sigma`` annealed
        from ``sigma_anneal_start`` to ``sigma_anneal_end`` over the first
        ``sigma_anneal_epochs`` epochs.
      * Auxiliary loss is plain MSE on ``mu`` (no beta-NLL).
      * Inference draws ``x_0 = mu`` (no ensembling).

    Two-term loss, matching the ``boldflow.py`` reference implementation:
    ``L = MSE(v, x1 - x0) + MSE(mu, x1)`` (CFM term + auxiliary prior MSE).

    This reproduces the Table 2 ablation row "+ AdaLN-Zero CFM, detached
    prior" with paper headline T.Corr=0.321 +/- 0.011, FC Corr=0.442.
    """

    def __init__(
        self,
        n_channels: int = 26,
        input_length: int = 6400,
        n_rois: int = 64,
        embed_dim: int = 512,
        velocity_layers: int = 4,
        n_inference_steps: int = 50,
        sigma_anneal_start: float = 0.5,
        sigma_anneal_end: float = 0.1,
        sigma_anneal_epochs: int = 10,
    ):
        super().__init__()
        self.n_inference_steps = n_inference_steps
        self.sigma_anneal_start = sigma_anneal_start
        self.sigma_anneal_end = sigma_anneal_end
        self.sigma_anneal_epochs = sigma_anneal_epochs
        self.current_sigma = sigma_anneal_start

        d = BoldFlow.DEFAULTS
        self.encoder = REVEEncoder(
            embed_dim=embed_dim,
            depth=int(d["encoder_depth"]),
            heads=int(d["encoder_heads"]),
            head_dim=int(d["encoder_head_dim"]),
            mlp_ratio=float(d["encoder_mlp_ratio"]),
            patch_size=int(d["encoder_patch_size"]),
            patch_stride=int(d["encoder_patch_stride"]),
            n_fourier_freqs=int(d["n_fourier_freqs"]),
        )
        self.spectral_encoder = MSSEncoder(
            embed_dim=embed_dim,
            input_length=input_length,
            n_channels=n_channels,
            scales=d["spectral_scales"],
            depth=int(d["spectral_depth"]),
            heads=int(d["spectral_heads"]),
            dropout=float(d["spectral_dropout"]),
        )
        self.head_activation = nn.GELU()
        self.velocity_net = AdaLNVelocityNet(
            flow_dim=n_rois,
            cond_dim=embed_dim,
            hidden=int(d["velocity_hidden"]),
            n_layers=velocity_layers,
            time_embed_dim=int(d["velocity_time_dim"]),
        )
        # Attribute name and inner key layout match the released p28c checkpoint.
        self.detached_prior_net = _point_prior_mlp(
            cond_dim=embed_dim, flow_dim=n_rois,
            hidden_1=int(d["prior_hidden_1"]),
            hidden_2=int(d["prior_hidden_2"]),
            dropout=float(d["prior_dropout"]),
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the noise schedule. Call once per epoch from the trainer."""
        if self.sigma_anneal_epochs <= 0:
            self.current_sigma = self.sigma_anneal_end
            return
        progress = min(1.0, epoch / max(1, self.sigma_anneal_epochs))
        self.current_sigma = (
            self.sigma_anneal_start
            + (self.sigma_anneal_end - self.sigma_anneal_start) * progress
        )

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head_activation(self.encoder(eeg) + self.spectral_encoder(eeg))

    def load_pretrained_encoder(self, weights_path, strict: bool = False):
        """Reuse :meth:`BoldFlow.load_pretrained_encoder` (same encoder layout)."""
        return BoldFlow.load_pretrained_encoder(self, weights_path, strict=strict)

    def forward(
        self, eeg: torch.Tensor, fmri_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_eeg = self.encode_eeg(eeg)
        mu = self.detached_prior_net(z_eeg)

        if self.training and fmri_target is not None:
            x1 = fmri_target

            # CFM matching loss (mu is detached so the flow gradient does not
            # update the prior net; aux MSE on mu provides that signal).
            x0 = mu.detach() + self.current_sigma * torch.randn_like(mu)
            t = torch.rand(x0.shape[0], device=eeg.device).clamp(1e-5, 1 - 1e-5)
            xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            ut = x1 - x0
            vt = self.velocity_net(xt, t, z_eeg)
            flow_loss = F.mse_loss(vt, ut)

            # Auxiliary MSE on the prior mean (replaces the beta-NLL term;
            # the prior net only learns through this channel).
            aux_loss = F.mse_loss(mu, x1)

            return flow_loss + aux_loss

        with torch.no_grad():
            return euler_integrate(self.velocity_net, mu, z_eeg, self.n_inference_steps)
