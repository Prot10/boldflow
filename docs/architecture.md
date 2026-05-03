# BOLDFlow Architecture

This document expands the architecture summary from the README into
component-by-component detail, with shapes, parameter counts, and pointers to
the implementing module.

## Input

* **EEG**: `(B, C, T)` raw EEG with `C = 26` channels (NeuroBOLT) or `C = 30`
  (OpenNeuroSleep) and `T = 6400` samples (= 32 s at 200 Hz). Z-scored per
  channel, clipped to `[-15, 15]`.
* **fMRI**: `(B, R)` parcellated BOLD signal at the trigger volume, with
  `R = 64` for DiFuMo-64. The data loader normalises each ROI by per-scan
  absolute 95th percentile.

## Temporal encoder: REVE (`boldflow.encoders.REVEEncoder`)

* 22 transformer blocks, hidden dim 512, 8 heads, head dim 64, MLP ratio 2.66
  (FFN hidden = 1361). Pre-norm with `RMSNorm`, GEGLU FFN, no projection bias.
* Patch embedding: overlapping 1 s patches with 0.9 s stride yield 35 patches
  per channel; flattened to `(B, 26 * 35, 200)` and linearly projected to
  `(B, 910, 512)`.
* Positional encoding: sum of a frozen 4D Fourier feature
  (`FourierEmb4D`, `cos`/`sin` over `(x, y, z, time)`) and a learnable MLP
  over the same coordinates, then `LayerNorm`.
* Pooling: a learnable cross-attention query reduces the 910 tokens to a
  single `(B, 512)` vector.
* Approx. 70.2 M parameters; initialised from the public `brain-bzh/reve-base`
  weights and fine-tuned end to end.

## Spectral encoder: MSS (`boldflow.encoders.MSSEncoder`)

* Per channel, take the magnitude STFT at four scales `[100, 200, 400, 800]`
  with no overlap and a rectangular window (matches the NeuroBOLT MSS).
* Frequency bins are projected to 512; time bins are projected to a fixed
  length `T_base = 6400 / 100 = 64`. The four scales are summed to produce
  `(B, 64, 512)` per channel.
* A learnable channel token plus sinusoidal positional encoding tags each
  channel; 26 channels are concatenated to `(B, 1664, 512)` and pooled by a
  4-layer linear-attention transformer (mean pool over tokens).
* Approx. 13.0 M parameters; trained from scratch. (The previous BOLDFlow
  prototypes used a 200-dim variant at ~4.6 M parameters; the released
  configuration uses the wider 512-dim variant for a stronger spectral
  pathway, at the cost of more parameters.)

## Fusion

`z_eeg = GELU(temporal + spectral)` -- the simplest fusion that matches the
paper. No learnable parameters.

## Distributional prior (`boldflow.flow.DistributionalPrior`)

* `MLP(z_eeg) -> (mu, sigma)` where `sigma = softplus(raw) + sigma_floor`.
* `mu_head` and `sigma_head` are linear probes on top of a shared two-layer
  MLP trunk (256 -> 128).
* `sigma_head.bias` is initialised so `sigma(t = 0) ~ init_sigma = 0.2`.
* Approx. 0.18 M parameters.

## Velocity network (`boldflow.flow.AdaLNVelocityNet`)

* 4 AdaLN-Zero blocks, hidden dim 512.
* Conditioning vector: `cond = time_proj(t) + eeg_proj(z_eeg)`.
* Each block computes `h' = (1 + gamma) * LayerNorm(h) + beta`, updates
  `h <- h + alpha * FFN(h')` where `(gamma, beta, alpha)` come from a
  zero-init `Linear` so the block is identity at init.
* Output projection: zero-init `Linear` so the velocity is zero at init,
  matching DiT's "AdaLN-Zero" recipe.
* Approx. 13.0 M parameters at `hidden=512` and 4 blocks (mostly the FFN
  inside each block, which is `Linear(512, 2048) -> Linear(2048, 512)`).

## Training objective

```
x_0 = mu + sigma * eps     (eps ~ N(0, I))
x_1 = fmri_target
x_t = (1 - t) * x_0 + t * x_1     (I-CFM linear path)
v_target = x_1 - x_0

L = MSE(v_theta(x_t, t, z_eeg), v_target) + beta_NLL(mu, sigma, x_1; beta=0.5)
```

`beta_NLL` is the Seitzer (2022) loss; `beta = 0.5` decouples mean and
variance gradients without losing all NLL effects, and is the
recommended default.

## Inference

Point estimate: `x_0 = mu`, integrate 50 explicit Euler steps with
`v_theta(x, t, z_eeg)`.

Ensemble: draw `K` source samples `x_0_k = mu + sigma * eps_k`, integrate
each, stack outputs to `(K, B, R)`. Per-sample per-ROI uncertainty is the
standard deviation across ensemble members; the ensemble mean is a strong
free point estimate.

## Total parameter count

At the default `embed_dim=512`, the model has 96,403,264 trainable
parameters, distributed as:

```
Component                       Parameters
REVE encoder + attention pool    70.2 M
MSS spectral encoder             13.0 M
AdaLN velocity network           13.0 M
Distributional prior head         0.18 M
                                 ------
Total                            96.4 M
```

