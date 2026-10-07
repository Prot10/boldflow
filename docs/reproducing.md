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
# Subject partitions come from the config seed and are shared across --seeds.

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
  results.json  mean/std over all fold x seed runs
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
for split in val test; do
  python scripts/analysis/sample_trajectories.py \
      --config configs/neurobolt.yaml --checkpoint outputs/.../fold_1/best.pt \
      --fold 1 --split $split --n-samples 50 \
      --output-dir outputs/trajectories_$split
done
python scripts/analysis/uq_calibration.py \
    --val-dir outputs/trajectories_val --test-dir outputs/trajectories_test
```

50-trajectory native ensemble with a validation-fitted scalar recalibration.
Paper values for the recalibrated native ensemble (Table 2, NeuroBOLT):

```
Spearman residual/std = 0.155
Calibration Error     = 0.011
Coverage@95           = 0.948
```

## 5. Figures

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

Retrained controls and ablations with their own config:

| Experiment                              | Config                              |
| --------------------------------------- | ----------------------------------- |
| Constant-input generator (Appendix D)   | `configs/control_constant_eeg.yaml` |
| Reduced montage, 19 channels (Table 7)  | `configs/montage_19.yaml`           |
| Reduced montage, 6 channels (Table 7)   | `configs/montage_6.yaml`            |
| Fixed-sigma source (Table 6, L4)        | `configs/ablation_point_prior.yaml` |

## 7. Appendix analyses

The analyses of Appendices C-G work on cached sampled trajectories. Run
`sample_trajectories.py` once per checkpoint and fold; every other script in
`scripts/analysis/` reads the cache and writes a JSON summary.

```bash
# M trajectories per held-out scan (one source draw per anchor, overlap-averaged)
python scripts/analysis/sample_trajectories.py \
    --config configs/neurobolt.yaml \
    --checkpoint outputs/boldflow_neurobolt/seed_12345/fold_1/best.pt \
    --fold 1 --n-samples 50 --output-dir outputs/trajectories
# add --split val for the recalibration set, --n-samples 200 for the audit
```

| Paper item                                             | Script                              |
| ------------------------------------------------------ | ----------------------------------- |
| Table 14, trajectory averaging over M                  | `trajectory_averaging.py`           |
| Appendix D, constant-input generator                   | `constant_input_fc.py`              |
| Appendix D, pairing beyond population structure        | `subject_pairing.py`                |
| Appendix D, within-scan temporal specificity           | `dynamic_fc_alignment.py`           |
| Table 9, held-out trajectory audit                     | `trajectory_audit.py`               |
| Appendix D, filtering control                          | `bandlimited_fc.py`                 |
| Appendix C, measured-fMRI effective rank               | `effective_rank.py`                 |
| Table 2, native ensemble and scalar recalibration      | `uq_calibration.py`                 |
| Appendix E, edge-error selection                       | `uq_edge_selection.py`              |
| Appendix E, regional and temporal associations         | `uq_structure.py`                   |
| Table 10, source-spread preservation                   | `source_spread.py`                  |
| Figure 1, strongest-edge recovery                      | `edge_recovery.py`                  |
| Table 11, community structure                          | `community_structure.py`            |
| Table 12, subject identification                       | `fingerprinting.py`                 |
| Table 13, per-component accuracy and thalamus          | `per_component_report.py`           |
| Appendix F, cross-session personalization              | `cross_session_personalization.py`  |
| Appendix A, checkpoint audit                           | `checkpoint_audit.py`               |

`source_spread.py` and `cross_session_personalization.py` run the model and
take `--config`/`--checkpoint` directly. `community_structure.py` needs
`networkx` (`pip install -e ".[analysis]"`).

## 8. Per-fold reproducibility

Default seed is 12345. Determinism is not perfect because some flow-matching
kernels lack deterministic implementations on GPU; fold-to-fold scatter from
re-runs is well below the reported per-seed standard deviation.

## 9. Sanity check (no GPU, no real data)

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
