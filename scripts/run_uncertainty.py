#!/usr/bin/env python
"""Post-hoc uncertainty quantification for a BOLDFlow checkpoint.

Pipeline: native ensemble on validation -> fit ScalarRecalibration and
SplitConformal -> apply to test set -> report Coverage, AUSE, Spearman
residual/std, and Expected Calibration Error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from boldflow.data import create_cv_dataloaders
from boldflow.model import BoldFlow
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.uncertainty import (ScalarRecalibration, SplitConformal, ause,
                                  expected_calibration_error, native_ensemble,
                                  spearman_residual_std)
from boldflow.utils import (ENV_DATA_ROOT, autodetect_device,
                            load_yaml_config, resolve_path, save_json,
                            set_seed, setup_logging)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run post-hoc UQ on a BOLDFlow checkpoint.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=50, help="Ensemble size.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Conformal miscoverage rate (0.05 = 95-percent intervals).")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON path for the UQ summary.")
    return p.parse_args()


def collect_ensemble(model, loader, *, n_samples, device):
    means, stds, targets = [], [], []
    for eeg, fmri in loader:
        eeg = eeg.to(device)
        m, s = native_ensemble(model, eeg, n_samples=n_samples)
        means.append(m.cpu().numpy())
        stds.append(s.cpu().numpy())
        targets.append(fmri.numpy())
    return np.concatenate(means), np.concatenate(stds), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    cfg = load_yaml_config(args.config)
    cfg["data"]["data_root"] = resolve_path(
        args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"),
    )
    if not cfg["data"]["data_root"] or cfg["data"]["data_root"].startswith("/path/to/"):
        raise SystemExit(
            f"data_root not set. Pass --data-root or set ${ENV_DATA_ROOT}."
        )
    set_seed(int(cfg.get("seed", 12345)))
    device = autodetect_device(args.device or cfg.get("device", "cuda"))

    splitter = SubjectLevelCVSplitter(
        data_root=cfg["data"]["data_root"],
        k_folds=int(cfg.get("k_folds", 5)),
        seed=int(cfg.get("seed", 12345)),
        dataset=cfg["data"]["dataset"],
        n_rois=int(cfg["data"]["n_rois"]),
    )
    fold = splitter.get_fold(args.fold)

    _, val_loader, test_loader, _ = create_cv_dataloaders(
        cfg["data"]["data_root"], fold,
        dataset=cfg["data"]["dataset"],
        n_rois=int(cfg["data"]["n_rois"]),
        target_roi=cfg["data"].get("target_roi"),
        multi_roi=bool(cfg["data"].get("multi_roi", True)),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        apply_eeg_filter=bool(cfg["data"].get("apply_eeg_filter", True)),
        apply_fmri_filter=bool(cfg["data"].get("apply_fmri_filter", True)),
        normalize_eeg=bool(cfg["data"].get("normalize_eeg", True)),
        eeg_lowpass=cfg["data"].get("eeg_lowpass"),
        exclude_non_neural=bool(cfg["data"].get("exclude_non_neural", False)),
        tr=float(cfg["data"].get("tr", 2.1)),
        tmin=float(cfg["data"].get("tmin", -32.0)),
        tmax=float(cfg["data"].get("tmax", 0.0)),
        crop=int(cfg["model"]["input_length"]),
    )

    model = BoldFlow.from_pretrained(args.checkpoint, device=device)

    print("Validation ensemble for calibration...")
    val_mean, val_std, val_target = collect_ensemble(model, val_loader, n_samples=args.n_samples, device=device)
    print("Test ensemble for evaluation...")
    test_mean, test_std, test_target = collect_ensemble(model, test_loader, n_samples=args.n_samples, device=device)

    val_residual = (val_target - val_mean).reshape(-1)
    recal = ScalarRecalibration().fit(val_residual, val_std.reshape(-1))
    conformal = SplitConformal(alpha=args.alpha).fit(val_target, val_mean, recal(val_std))

    test_std_recal = recal(test_std).reshape(test_std.shape)
    summary = {
        "fold": args.fold,
        "n_ensemble": args.n_samples,
        "alpha": args.alpha,
        "scalar_recalibration_alpha": recal.alpha,
        "conformal_q": conformal.q,
        "test_coverage": conformal.coverage(test_target, test_mean, test_std_recal),
        "test_ause": ause(test_target, test_mean, test_std_recal),
        "test_spearman_residual_std": spearman_residual_std(test_target, test_mean, test_std_recal),
        "test_calibration_error": expected_calibration_error(test_target, test_mean, test_std_recal),
    }
    print()
    print("Uncertainty quantification summary:")
    for k, v in summary.items():
        line = f"  {k:>30s} = {v:.4f}" if isinstance(v, float) else f"  {k:>30s} = {v}"
        print(line)

    if args.output:
        save_json(summary, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
