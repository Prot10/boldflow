# Reproducing the paper results

## 0. Data

Two preprocessed simultaneous EEG/fMRI datasets:

* **NeuroBOLT** -- 22 subjects, 29 resting-state scans, 26 EEG channels at
  200 Hz, fMRI TR = 2.1 s.
* **OpenNeuroSleep** -- 33 subjects, 229 scans (rest + sleep), 30 EEG
  channels, TR = 2.1 s.

Expected on-disk layout::

    data_root/
      EEG/<scan_name>_eeg.set
      EEG/<scan_name>_eeg.fdt
      fMRI_difumo_64/<scan_name>_difumo64_roi.pkl

`<scan_name>` is `sub01-scan01` for NeuroBOLT and `sub-01_task-rest_run-1`
for OpenNeuroSleep. The fMRI pickle is a pandas DataFrame whose columns are
DiFuMo region labels plus two `global signal` columns (auto-dropped). The
EEG fMRI trigger is `R149` for NeuroBOLT and `R128` for OpenNeuroSleep.

## 1. Pretrained encoder

```bash
python scripts/download_pretrained.py
```

Pulls `model.safetensors` from `brain-bzh/reve-base` into
`./checkpoints/reve-base.safetensors` (or `$BOLDFLOW_CHECKPOINTS_DIR`).

## 2. Train (paper protocol)

```bash
# Headline: 5-fold CV, 30 epochs/fold, 3 seeds = 15 runs
python scripts/train.py --config configs/neurobolt.yaml --seeds 12345 22345 32345

# OpenNeuroSleep
python scripts/train.py --config configs/sleep.yaml --seeds 12345 22345 32345
```

Per fold takes ~2.5 h on an A100 40 GB; full 5-fold x 3-seed sweep is
~38 GPU-hours. Output structure:

```
outputs/boldflow_neurobolt/
  seed_12345/   results.json + fold_<i>/best.pt   (only if --seeds is given)
  seed_22345/   ...
  seed_32345/   ...
  results.json  aggregated mean/std across seeds
```

## 3. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/neurobolt.yaml \
    --checkpoint outputs/boldflow_neurobolt/seed_12345/fold_1/best.pt \
    --fold 1 \
    --save-predictions outputs/.../predictions_fold1.pt
```

`evaluate.py` follows the protocol of the main comparison: one sampled
trajectory per scan (`M = 1`, one source draw per anchor, overlap-averaged),
MSE and T. Corr. over all 64 components, FC Corr. within each scan on the
55-component cortical mask. Pass `--deterministic` for the source-mean
readout. Paper values (Table 1, NeuroBOLT row, mean over 5 folds x 3 seeds):

```
mean_test_mse            = 0.239
mean_test_pearson_r      = 0.326
mean_test_fc_correlation = 0.584
```

Because the readout is stochastic, single-run values vary with the seed.

## 4. Uncertainty quantification

```bash
python scripts/run_uncertainty.py \
    --config configs/neurobolt.yaml \
    --checkpoint outputs/.../fold_1/best.pt \
    --fold 1 --output uq_fold_1.json
```

50-member native ensemble with a validation-fitted scalar recalibration.
Paper values for the recalibrated native ensemble (Table 2, NeuroBOLT):

```
Spearman residual/std = 0.155
Calibration Error     = 0.011
Coverage@95           = 0.948
```

The script additionally reports AUSE and split-conformal coverage, which are
not part of the paper tables.

## 5. Figures

After training, reproduce the bar-plot figures (ablation, parcellation,
context length) from a list of `results.json`:

```bash
python scripts/make_figures.py --metric pearson_r \
    --results outputs/run16s/results.json \
              outputs/run24s/results.json \
              outputs/run32s/results.json \
    --labels 16s 24s 32s \
    --output figures/context_length.pdf
```

After `evaluate.py --save-predictions`, reproduce the qualitative figures
(predicted vs. ground-truth time courses + FC matrices):

```bash
python scripts/make_qualitative.py \
    --predictions outputs/.../predictions_fold1.pt \
    --output-dir figures/qualitative
```

## 6. Ablations

The release ships one ablation variant via `boldflow.ablations`:

* **Point-prior (fixed-sigma AdaLN-Zero, row L4 of Table 6, FC Corr 0.442)**
  -- `boldflow.ablations.BoldFlowPointPrior`. Train it with
  `python scripts/train.py --config configs/ablation_point_prior.yaml`.

The other Table 1 baselines (NeuroBOLT joint, NeuroBOLT†, REVE-NoFT, REVE-FT) are
reimplementations of independently-published architectures; we point readers
to the original NeuroBOLT and REVE repositories for those baselines.

The context-length sweep, parcellation sweep (64/256/512), and seq2seq
operating-point ablation are reproduced by changing config knobs:

| Ablation                  | Knob                                |
| ------------------------- | ----------------------------------- |
| Context length            | `data.tmin` (and `model.input_length`) |
| Parcellation              | `data.n_rois` and `model.n_rois`    |
| Seq2seq horizon T_out      | `model.n_out_timesteps` (1 = seq2one, 4 = headline) |
| Without spectral encoder  | (not exposed; see `boldflow/model.py`) |

## 7. Per-fold reproducibility

Default seed is 12345. Determinism is not perfect because some flow-matching
kernels lack deterministic implementations on GPU; fold-to-fold scatter from
re-runs is well below the reported per-seed standard deviation.

## 8. Sanity check (no GPU, no real data)

```bash
pytest tests/                          # ~70 s on CPU
python scripts/train.py --config configs/neurobolt.yaml \
    --folds 1 --epochs 1 --device cpu \
    --data-root /path/to/your/data
```

The 1-fold / 1-epoch run completes in ~10-30 min on CPU and writes a
complete `outputs/.../fold_1/best.pt` plus `results.json`. The metrics will
be far below the paper numbers because there are no positive epochs of
training; the value is in verifying that the entire pipeline (data loader,
model, training step, AMP off-path, eval, save) runs end to end.
