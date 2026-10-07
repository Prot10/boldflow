# BOLDFlow Architecture

This document expands the architecture summary from the README into
component-by-component detail, with shapes, parameter counts, and pointers to
the implementing module.

## Input

* **EEG**: `(B, C, T)` raw EEG with `C = 26` channels (NeuroBOLT) or `C = 30`
  (OpenNeuroSleep) and `T = 6400` samples (= 32 s at 200 Hz). Z-scored per
  channel, clipped to `[-15, 15]`.
* **fMRI**: parcellated BOLD signal with `R = 64` for DiFuMo-64. BOLDFlow is
  sequence-to-sequence: each EEG window predicts the block of
  `T_out = n_out_timesteps` consecutive volumes ending at the anchor TR, so
  the target is `(B, T_out, R)`, flattened to `(B, T_out * R)` for the flow.
  The paper headline uses `T_out = 4` (flow dimension `D = 256`); `T_out = 1`
  recovers the seq2one variant. The data loader normalises each ROI by
  per-scan absolute 95th percentile.

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
* Approx. 13.0 M parameters; trained from scratch.

## Fusion

`z_eeg = GELU(temporal + spectral)` -- the simplest fusion that matches the
paper. No learnable parameters.

## Distributional prior (`boldflow.flow.DistributionalPrior`)

* `MLP(z_eeg) -> (mu, sigma)` where `sigma = softplus(raw) + sigma_floor`.
  `mu` and `sigma` have the flow dimension `D = n_rois * n_out_timesteps`.
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
* Input/output projections map the flow vector `D = n_rois * n_out_timesteps`
  to/from the hidden width. The output projection is zero-init so the velocity
  is zero at init, matching DiT's "AdaLN-Zero" recipe.
* Approx. 13.0 M parameters at `hidden=512` and 4 blocks (mostly the
  FFN inside each block, `Linear(512, 2048) -> Linear(2048, 512)`).

## Training objective

```
x_0 = mu + sigma * eps     (eps ~ N(0, I))
x_1 = fmri_target
x_t = (1 - t) * x_0 + t * x_1     (I-CFM linear path)
v_target = x_1 - x_0

L = MSE(v_theta(x_t, t, z_eeg), v_target) + lambda * beta_NLL(mu, sigma, x_1; beta=0.5)
```

with `lambda = 1`. `beta_NLL` is the Seitzer (2022) loss: the Gaussian NLL
multiplied by `stopgrad(sigma^(2 beta))`, which stops gradients through the
variance-dependent weight while the NLL term updates mean and variance. The
source sample is reparameterised without detaching `mu` or `sigma`, so the
flow objective also updates the source.

## Inference

Every prediction draws a source `x_0 = mu + sigma * eps` and integrates 50
explicit Euler steps with `v_theta(x, t, z_eeg)`. The raw output is the
flattened `(B, T_out * R)` block; reshape to `(B, T_out, R)` for the per-TR
volumes.

Overlap-averaging (seq2seq, `T_out > 1`): neighbouring EEG windows are
stride-1 in TR, so a TR is covered by `K_t <= T_out` blocks. The evaluator
(`boldflow.training.evaluate`, `aggregate=True`) averages those estimates per
scan into the per-TR trajectory (paper Eq. 8).

Sampled trajectory (`M = 1`): one independent source draw per anchor,
overlap-averaged. This is the readout used for FC and for the main
comparison.

Ensemble (`M > 1`): repeat the full procedure independently `M` times. The
ensemble mean averages the trajectories before any metric is computed; it
improves pointwise accuracy while attenuating the residual covariance by
`1/M`. UQ uses the per-TR, per-component mean and Bessel-corrected standard
deviation of `M = 50` trajectories, with a validation-fitted scalar
recalibration of the latter.

Deterministic readout: `x_0 = mu` (`model(eeg, sample=False)`).

## Evaluation metrics

* MSE: all TRs and all output components, in normalised target units.
* T. Corr.: Pearson correlation per component within each scan, averaged over
  components and scans.
* FC Corr.: Pearson correlation between the upper triangles of the predicted
  and measured FC matrices, computed within each scan and averaged across
  scans. FC uses the components assigned to a cortical network in the atlas
  metadata (`boldflow.difumo.cortical_network_indices`; 55 of the 64 DiFuMo-64
  components).

## Total parameter count

At the default `embed_dim=512`, the model has ~96.4 M trainable parameters,
distributed as:

```
Component                       Parameters
REVE encoder + attention pool    70.2 M
MSS spectral encoder             13.0 M
AdaLN velocity network           13.0 M
Distributional prior head         0.18 M
                                 ------
Total                            96.4 M
```

