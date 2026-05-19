"""BOLDFlow model: REVE + MSS + AdaLN-Zero CFM with a distributional source prior.

Architecture (~96.4 M parameters at the default ``embed_dim=512``)::

    EEG (B, 26, 6400)                 # 32 s at 200 Hz, z-scored, clipped to [-15, 15]
       |
       +-- REVEEncoder      (~70.2 M, fine-tuned from pretrained weights)
       +-- MSSEncoder       (~13.0 M, multi-scale STFT + linear-attention pooling)
                |
                v
          additive fusion + GELU
                |
              z_eeg (B, 512)
                |
       +--------+------------+
       |                     |
       v                     v
    DistributionalPrior   AdaLNVelocityNet (B, D) <- (B, D), t, z_eeg
       (mu, sigma)
       |
       v
    x_0 = mu (+ sigma * eps  during training, or for ensemble UQ)
       |
       v
    Euler ODE integration (n_inference_steps) -> predicted fMRI (B, D)

The flow dimension is ``D = n_rois * n_out_timesteps``. With the default
seq2seq horizon ``n_out_timesteps=4`` the model predicts the block of 4
consecutive DiFuMo volumes ending at the anchor TR (D = 4*64 = 256); set
``n_out_timesteps=1`` for the seq2one variant (D = n_rois).

Training loss (I-CFM, matches the paper, Eq. 4-5)::

    L = MSE(v_theta(x_t, t, z_eeg),  x_1 - x_0)        # CFM term  (Eq. 4)
      + lambda * beta_NLL(mu, sigma, x_1; beta=0.5)    # prior term (Eq. 5)

with ``x_0 = mu + sigma * eps``, ``lambda = 1``, ``beta = 0.5``. I-CFM (no OT).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from boldflow.encoders import (DEFAULT_CHANNEL_ORDER, REVEEncoder, MSSEncoder,
                               SLEEP_CHANNEL_ORDER)
from boldflow.flow import (
    AdaLNVelocityNet,
    DistributionalPrior,
    beta_nll,
    euler_integrate,
)


class BoldFlow(nn.Module):
    """The BOLDFlow model.

    Parameters
    ----------
    n_channels
        EEG channels in the input. 26 for NeuroBOLT, 30 for OpenNeuroSleep.
    input_length
        EEG samples per window (= sampling_rate * window_seconds). Default
        6400 = 32 s at 200 Hz, the paper headline.
    n_rois
        fMRI parcels per TR (DiFuMo-64 by default; 256 / 512 supported).
    n_out_timesteps
        Seq2seq horizon ``T_out``: number of consecutive DiFuMo volumes
        predicted per EEG window (default 4, the paper headline). The flow
        dimension is ``D = n_rois * n_out_timesteps``; ``n_out_timesteps=1``
        recovers the seq2one variant.
    embed_dim
        Width of the EEG embedding space.
    velocity_layers
        AdaLN-Zero blocks in the velocity net.
    n_inference_steps
        Explicit Euler steps used at inference (default 50).
    prior_loss_weight
        Weight ``lambda`` on the beta-NLL prior term (default 1.0, paper Eq. 5).
        Setting it to 0 removes the prior supervision (mu, sigma stay at their
        initialisation).
    prior_beta, prior_sigma_floor, prior_init_sigma
        Hyperparameters of the distributional prior and beta-NLL loss.
    """

    # Architectural defaults matching the paper headline (T.Corr=0.326, FC Corr=0.584).
    DEFAULTS: Dict[str, float] = dict(
        n_channels=26,
        input_length=6400,
        n_rois=64,
        embed_dim=512,
        encoder_depth=22,
        encoder_heads=8,
        encoder_head_dim=64,
        encoder_mlp_ratio=2.66,
        encoder_patch_size=200,
        encoder_patch_stride=180,
        n_fourier_freqs=4,
        spectral_scales=(100, 200, 400, 800),
        spectral_depth=4,
        spectral_heads=8,
        spectral_dropout=0.2,
        velocity_hidden=512,
        velocity_layers=4,
        velocity_time_dim=64,
        n_inference_steps=50,
        prior_hidden_1=256,
        prior_hidden_2=128,
        prior_dropout=0.1,
        prior_beta=0.5,
        prior_loss_weight=1.0,
        prior_sigma_floor=0.05,
        prior_init_sigma=0.2,
    )

    def __init__(
        self,
        n_channels: int = 26,
        input_length: int = 6400,
        n_rois: int = 64,
        n_out_timesteps: int = 4,
        embed_dim: int = 512,
        velocity_layers: int = 4,
        n_inference_steps: int = 50,
        prior_beta: float = 0.5,
        prior_loss_weight: float = 1.0,
        prior_sigma_floor: float = 0.05,
        prior_init_sigma: float = 0.2,
        channel_order: Optional[tuple[str, ...]] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.input_length = input_length
        self.n_rois = n_rois
        self.n_out_timesteps = max(1, int(n_out_timesteps))
        # Flow dimension: T_out consecutive DiFuMo volumes, flattened.
        self.flow_dim = n_rois * self.n_out_timesteps
        self.embed_dim = embed_dim
        self.n_inference_steps = n_inference_steps
        self.prior_beta = prior_beta
        self.prior_loss_weight = prior_loss_weight

        d = self.DEFAULTS

        # Pick the right channel layout. Pass an explicit ``channel_order`` to
        # use a custom montage; otherwise we default to NeuroBOLT (26) or
        # OpenNeuroSleep (30) based on ``n_channels``.
        if channel_order is None:
            if n_channels == len(DEFAULT_CHANNEL_ORDER):
                channel_order = DEFAULT_CHANNEL_ORDER
            elif n_channels == len(SLEEP_CHANNEL_ORDER):
                channel_order = SLEEP_CHANNEL_ORDER
            else:
                raise ValueError(
                    f"no built-in channel_order for n_channels={n_channels}; "
                    f"pass channel_order=(...,) explicitly."
                )

        self.encoder = REVEEncoder(
            embed_dim=embed_dim,
            depth=int(d["encoder_depth"]),
            heads=int(d["encoder_heads"]),
            head_dim=int(d["encoder_head_dim"]),
            mlp_ratio=float(d["encoder_mlp_ratio"]),
            patch_size=int(d["encoder_patch_size"]),
            patch_stride=int(d["encoder_patch_stride"]),
            n_fourier_freqs=int(d["n_fourier_freqs"]),
            channel_order=channel_order,
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
            flow_dim=self.flow_dim,
            cond_dim=embed_dim,
            hidden=int(d["velocity_hidden"]),
            n_layers=velocity_layers,
            time_embed_dim=int(d["velocity_time_dim"]),
        )
        # Attribute name kept stable for compatibility with released checkpoints.
        self.distributional_prior_head = DistributionalPrior(
            cond_dim=embed_dim,
            flow_dim=self.flow_dim,
            hidden_1=int(d["prior_hidden_1"]),
            hidden_2=int(d["prior_hidden_2"]),
            dropout=float(d["prior_dropout"]),
            sigma_floor=prior_sigma_floor,
            init_sigma=prior_init_sigma,
        )

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Run both encoder branches; return the fused EEG embedding ``z_eeg``."""
        temporal = self.encoder(eeg)
        spectral = self.spectral_encoder(eeg)
        return self.head_activation(temporal + spectral)

    def forward(
        self,
        eeg: torch.Tensor,
        fmri_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Train (with ``fmri_target``) or run inference.

        Training returns the two-term loss ``L = L_CFM + lambda*L_prior``
        (paper Eq. 4-5); inference returns the deterministic point estimate
        ``Euler(mu)``.
        """
        z_eeg = self.encode_eeg(eeg)
        mu, sigma = self.distributional_prior_head(z_eeg)

        if self.training and fmri_target is not None:
            x1 = fmri_target

            # CFM matching loss (Eq. 4). I-CFM: index-wise pairing, no OT.
            eps = torch.randn_like(mu)
            x0 = mu + sigma * eps
            t = torch.rand(x0.shape[0], device=eeg.device).clamp(1e-5, 1 - 1e-5)
            xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            ut = x1 - x0
            vt = self.velocity_net(xt, t, z_eeg)
            flow_loss = F.mse_loss(vt, ut)

            # beta-NLL on the per-sample Gaussian prior (Eq. 5, Seitzer 2022).
            prior_loss = beta_nll(mu, sigma, x1, beta=self.prior_beta)

            return flow_loss + self.prior_loss_weight * prior_loss

        with torch.no_grad():
            return euler_integrate(self.velocity_net, mu, z_eeg, self.n_inference_steps)

    @torch.no_grad()
    def sample_ensemble(
        self,
        eeg: torch.Tensor,
        n_samples: int = 50,
        inference_sigma: float = 1.0,
    ) -> torch.Tensor:
        """Draw ``n_samples`` flow trajectories. Returns ``(n_samples, B, n_rois)``.

        ``inference_sigma`` is a temperature on the learned ``sigma``: 0
        collapses to the point estimate; 1 matches training-time sampling.
        """
        z_eeg = self.encode_eeg(eeg)
        mu, sigma = self.distributional_prior_head(z_eeg)
        outputs = []
        for _ in range(n_samples):
            if inference_sigma > 0:
                eps = torch.randn_like(mu)
                x0 = mu + inference_sigma * sigma * eps
            else:
                x0 = mu
            outputs.append(euler_integrate(self.velocity_net, x0, z_eeg, self.n_inference_steps))
        return torch.stack(outputs, dim=0)

    @torch.no_grad()
    def prior_sigma_stats(self, eeg: torch.Tensor) -> Dict[str, float]:
        """Mean/min/max of learned sigma over a batch, for collapse monitoring."""
        z_eeg = self.encode_eeg(eeg)
        _, sigma = self.distributional_prior_head(z_eeg)
        return {"mean": sigma.mean().item(), "min": sigma.min().item(), "max": sigma.max().item()}

    def load_pretrained_encoder(
        self,
        weights_path: Union[str, Path],
        strict: bool = False,
    ) -> Tuple[list[str], list[str]]:
        """Load REVE pretrained weights into ``self.encoder``.

        Accepts ``.safetensors`` (preferred) or ``.pt``. Strips ``module.``
        and ``encoder.`` prefixes from keys before loading.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"REVE checkpoint not found: {weights_path}")

        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            cleaned[k] = v

        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=strict)
        return list(missing), list(unexpected)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = False,
        **kwargs,
    ) -> "BoldFlow":
        """Instantiate the model and load a full BoldFlow checkpoint.

        Accepts a ``best.pt`` written by ``train.py`` (with ``model_state_dict``)
        or a bare ``state_dict``.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        model = cls(**kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[BoldFlow] {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"[BoldFlow] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        return model.to(device).eval()

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Total parameter count, optionally restricted to trainable params."""
        params = self.parameters()
        if only_trainable:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
