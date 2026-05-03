# BOLDFlow

**Network-Level EEG-to-fMRI Synthesis via Conditional Flow Matching**

BOLDFlow predicts parcellated fMRI from concurrent EEG, training end-to-end
with a conditional flow matching decoder whose source distribution is a
learned per-sample Gaussian. The same model produces a strong point estimate
and well-calibrated, sample-level uncertainty through native ODE ensembling.

| Dataset         | MSE                | T. Corr.            | FC Corr.            |
| --------------- | ------------------ | ------------------- | ------------------- |
| NeuroBOLT       | 0.239 (±0.002)     | **0.326** (±0.008)  | **0.584** (±0.017)  |
| OpenNeuroSleep  | 0.255 (±0.001)     | **0.212** (±0.001)  | **0.528** (±0.017)  |

5-fold inter-subject CV with 3 seeds per fold, DiFuMo-64.

## Architecture

```
EEG (B, 26, 6400)              # 32 s at 200 Hz, z-scored, clipped to [-15, 15]
   |
   +-- REVE encoder         (~70.2 M, fine-tuned from pretrained weights)
   +-- MSS spectral encoder (~13.0 M, multi-scale STFT + linear-attention pooling)
        |
        v
     additive fusion + GELU  ->  z_eeg  (B, 512)
        |
        +-- DistributionalPrior(z_eeg) -> (mu, sigma)        # ~0.18 M, learned Gaussian
        +-- AdaLN-Zero velocity net  v(x_t, t, z_eeg)        # ~13.0 M, 4 blocks
        |
        v
     50 explicit Euler steps  ->  predicted fMRI (B, 64) over DiFuMo-64

Total: ~96.4 M parameters at the default embed_dim=512.
```

Training loss: `MSE(v, x_1 - x_0) + lambda * beta_NLL(mu, sigma, x_1)` with
`lambda = 1`, `beta = 0.5`, I-CFM (no OT rematching).

Inference (point): `Euler(mu)`. Inference (ensemble UQ): draw `K = 50` samples
`x_0_k = mu + sigma * eps_k`, integrate each, return `(mean, std)`.

## Installation

```bash
git clone <REPO_URL>
cd boldflow

# uv (recommended)
uv sync --extra all

# or pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Download pretrained REVE encoder weights:

```bash
python scripts/download_pretrained.py
# saves ./checkpoints/reve-base.safetensors  (~265 MB)
```

## Configuration without editing files

The release is path-agnostic. You can run any script with no edits to the
YAML configs by setting environment variables:

| Variable                    | Purpose                                            |
| --------------------------- | -------------------------------------------------- |
| `BOLDFLOW_DATA_ROOT`        | Override `data.data_root` in any config.           |
| `BOLDFLOW_CHECKPOINTS_DIR`  | Where `download_pretrained.py` writes weights, and where `train.py` looks for `reve-base.safetensors` if the config points at a non-existent path. |
| `BOLDFLOW_OUTPUT_DIR`       | Override `output_dir` in any config.               |

CLI flags `--data-root`, `--output-dir`, `--checkpoints-dir` take precedence.

## Quickstart

```python
import torch
from boldflow import BoldFlow

model = BoldFlow()                                 # ~96 M parameters
eeg = torch.randn(1, 26, 6400).clamp(-15, 15)      # 32 s @ 200 Hz, z-scored
prediction = model(eeg)                            # (1, 64) DiFuMo-64

# Native ensemble UQ:
samples = model.sample_ensemble(eeg, n_samples=50) # (50, 1, 64)
mean, std = samples.mean(0), samples.std(0)
```

Load a trained checkpoint:

```python
model = BoldFlow.from_pretrained("checkpoints/boldflow_neurobolt_fold1.pt", device="cuda")
```

## Training

```bash
# Full 5-fold CV reproducing the NeuroBOLT row of Table 1
python scripts/train.py --config configs/neurobolt.yaml

# Single fold, 2 epochs (sanity check)
python scripts/train.py --config configs/neurobolt.yaml --folds 1 --epochs 2

# Path overrides (no YAML edits)
BOLDFLOW_DATA_ROOT=/scratch/neurobolt python scripts/train.py --config configs/neurobolt.yaml
python scripts/train.py --config configs/neurobolt.yaml --data-root /scratch/neurobolt

# OpenNeuroSleep
python scripts/train.py --config configs/sleep.yaml
```

`train.py` reads a YAML config, builds a `SubjectLevelCVSplitter` over the
data root, and trains a fresh `BoldFlow` per fold with cosine-warmup LR. Each
fold writes `outputs/<run_name>/fold_<i>/best.pt`; after all folds it writes
`outputs/<run_name>/results.json` with per-fold and aggregated metrics.

### Data layout

Both NeuroBOLT and OpenNeuroSleep should be preprocessed into:

```
<data_root>/
  EEG/<scan_name>_eeg.set      # EEGLAB .set + .fdt
  EEG/<scan_name>_eeg.fdt
  fMRI_difumo_64/<scan_name>_difumo64_roi.pkl   # pandas DataFrame, T x ROIs
```

Scan name pattern is `sub01-scan01` for NeuroBOLT and
`sub-01_task-rest_run-1` for OpenNeuroSleep.

## Inference / Evaluation

```bash
# Evaluate a fold's test split
python scripts/evaluate.py \
    --config configs/neurobolt.yaml \
    --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \
    --fold 1

# Predict on a single preprocessed EEG window (.npy of shape (C, T))
python scripts/predict.py \
    --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \
    --eeg sample_eeg.npy \
    --ensemble 50      # optional: 50-member ensemble for UQ
```

## Uncertainty Quantification

Native ensemble + scalar recalibration + split conformal on a held-out fold:

```bash
python scripts/run_uncertainty.py \
    --config configs/neurobolt.yaml \
    --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \
    --fold 1 \
    --output uq_fold_1.json
```

Reports Coverage@95, AUSE, Spearman residual / std, and Expected Calibration
Error after recalibration.

## Repository Layout

```
boldflow/
  boldflow/                    python package
    __init__.py                exports BoldFlow
    model.py                   BoldFlow class
    encoders.py                REVE temporal + MSS spectral encoders
    flow.py                    AdaLN-Zero velocity, distributional prior, beta-NLL, Euler ODE
    data.py                    NeuroBOLT + OpenNeuroSleep loaders
    splits.py                  subject-level K-fold CV
    difumo.py                  DiFuMo labels and non-neural masks
    metrics.py                 MSE, MAE, R2, T.Corr, Spearman, FC Corr
    schedulers.py              cosine warmup + layer-wise LR decay
    training.py                per-fold loop + K-fold runner + evaluate
    uncertainty.py             native ensemble + ScalarRecalibration + SplitConformal + AUSE/ECE
    utils.py                   logging, seeding, IO, env-var path resolution
  configs/{neurobolt,sleep}.yaml
  scripts/
    train.py                   K-fold (and multi-seed) training entry point
    evaluate.py                evaluate a checkpoint on a fold's test split
    predict.py                 single-window inference + ensemble UQ from .npy
    run_uncertainty.py         post-hoc UQ pipeline + conformal calibration
    download_pretrained.py     download REVE-base weights from HuggingFace
    make_figures.py            ablation/scaling bar plots from results.json
    make_qualitative.py        time-course + FC-matrix plots from saved predictions
  tests/                       pytest smoke tests
  checkpoints/                 placeholder, see checkpoints/README.md
  docs/                        architecture.md, reproducing.md
```

The release also exposes the **point-prior ablation** as
`boldflow.ablations.BoldFlowPointPrior` with config
`configs/ablation_point_prior.yaml`. This loads the original
`p28c_adaln_32s` checkpoint with zero key mismatches and reproduces the
"+ AdaLN-Zero CFM, detached prior" row of Table 2 (T.Corr=0.321, FC=0.442).

## Tests

```bash
pytest tests/
```

Covers model instantiation, forward/backward shapes, metric correctness
against numpy references, UQ recalibration recovering known scaling factors,
and conformal coverage hitting the nominal level. Runs in seconds on CPU.

## License

MIT.
