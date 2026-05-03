#!/usr/bin/env python
"""Run BOLDFlow on a single EEG window and print the predicted fMRI ROIs.

The input must be a `.npy` file of shape ``(n_channels, n_samples)`` containing
preprocessed EEG (z-scored per channel, clipped to [-15, 15]). Default config
expects ``n_channels=26`` and ``n_samples=6400`` (32 s @ 200 Hz).

Examples
--------
    # Single deterministic prediction
    python scripts/predict.py \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \\
        --eeg sample_eeg.npy

    # Ensemble UQ (50 ODE integrations from sampled priors)
    python scripts/predict.py \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \\
        --eeg sample_eeg.npy --ensemble 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from boldflow.model import BoldFlow
from boldflow.uncertainty import native_ensemble
from boldflow.utils import autodetect_device, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict fMRI ROIs from a single EEG window.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eeg", type=str, required=True, help="Path to a .npy EEG file.")
    p.add_argument("--ensemble", type=int, default=0,
                   help="If >0, run an ensemble of this many ODE samples.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Optional .npz path to save predictions and uncertainty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    device = autodetect_device(args.device or "cuda")

    eeg_np = np.load(args.eeg)
    if eeg_np.ndim == 2:
        eeg_np = eeg_np[None, ...]                   # add batch dim
    eeg = torch.from_numpy(eeg_np).float().to(device)
    eeg = eeg.clamp(-15, 15)

    model = BoldFlow.from_pretrained(args.checkpoint, device=device)
    model.eval()

    if args.ensemble > 0:
        mean, std = native_ensemble(model, eeg, n_samples=args.ensemble)
        print(f"ensemble mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")
        print(f"mean range: [{mean.min().item():.3f}, {mean.max().item():.3f}]")
        print(f"std  range: [{std.min().item():.3f}, {std.max().item():.3f}]")
        if args.output:
            np.savez(args.output, mean=mean.cpu().numpy(), std=std.cpu().numpy())
            print(f"saved to {args.output}")
    else:
        with torch.no_grad():
            pred = model(eeg)
        print(f"prediction shape: {tuple(pred.shape)}")
        print(f"range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        if args.output:
            np.savez(args.output, prediction=pred.cpu().numpy())
            print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
