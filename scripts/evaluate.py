#!/usr/bin/env python
"""Evaluate a trained BOLDFlow checkpoint on a fold's test split.

Examples
--------
    python scripts/evaluate.py \\
        --config configs/neurobolt.yaml \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \\
        --fold 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from boldflow.data import create_cv_dataloaders
from boldflow.model import BoldFlow
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.training import evaluate
from boldflow.utils import (ENV_DATA_ROOT, autodetect_device,
                            load_yaml_config, resolve_path, set_seed,
                            setup_logging)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained BOLDFlow checkpoint.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fold", type=int, default=1, help="1-indexed fold number.")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-predictions", type=str, default=None,
                   help="Optional .pt file to dump predictions and targets.")
    return p.parse_args()


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
    print(f"Evaluating fold {args.fold}: {fold.n_test_scans} test scans.")

    _, _, test_loader, meta = create_cv_dataloaders(
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
        n_out_timesteps=int(cfg["model"].get("n_out_timesteps", 4)),
    )

    model = BoldFlow.from_pretrained(
        args.checkpoint, device=device,
        n_channels=int(cfg["model"]["n_channels"]),
        input_length=int(cfg["model"]["input_length"]),
        n_rois=int(cfg["model"]["n_rois"]),
        n_out_timesteps=int(cfg["model"].get("n_out_timesteps", 4)),
        embed_dim=int(cfg["model"]["embed_dim"]),
        velocity_layers=int(cfg["model"]["velocity_layers"]),
        n_inference_steps=int(cfg["model"]["n_inference_steps"]),
    )

    # Seq2seq: report metrics on the per-scan overlap-averaged trajectory.
    out = evaluate(model, test_loader, device,
                   scan_sizes=meta.get("test_scan_sizes"), aggregate=True)
    print()
    print(f"Test metrics on fold {args.fold}:")
    for k, v in out["metrics"].items():
        print(f"  {k:>14s} = {v:.4f}")

    if args.save_predictions is not None:
        torch.save({
            "predictions": out["predictions"],
            "targets": out["targets"],
            "metrics": out["metrics"],
        }, args.save_predictions)
        print(f"saved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
