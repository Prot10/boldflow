#!/usr/bin/env python
"""Run BOLDFlow on a single EEG window and print the predicted fMRI block.

The input must be a `.npy` file of shape ``(n_channels, n_samples)`` containing
preprocessed EEG (z-scored per channel, clipped to [-15, 15]). The default
model expects ``n_channels=26`` and ``n_samples=6400`` (32 s at 200 Hz) and
predicts a block of ``T_out`` consecutive DiFuMo volumes ending at the window.

With ``--ensemble M`` the script draws ``M`` predictions of the block and
reports their block-level mean and Bessel-corrected standard deviation. This
is the spread of single blocks; the per-TR ensemble spread of the uncertainty
analyses is computed on overlap-averaged trajectories
(``scripts/analysis/sample_trajectories.py``).

Examples
--------
    # Single sampled prediction (one source draw)
    python scripts/predict.py \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \\
        --eeg sample_eeg.npy

    # Block-level mean and spread of 50 draws
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
from boldflow.utils import autodetect_device, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict an fMRI block from a single EEG window.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eeg", type=str, required=True, help="Path to a .npy EEG file.")
    p.add_argument("--ensemble", type=int, default=0,
                   help="If >1, draw this many predictions and report their "
                        "block-level mean and standard deviation.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Optional .npz path to save the prediction (or the "
                        "block-level mean and standard deviation).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    if args.ensemble == 1 or args.ensemble < 0:
        raise SystemExit("--ensemble needs at least 2 draws (0 for a single prediction)")
    device = autodetect_device(args.device or "cuda")

    eeg_np = np.load(args.eeg)
    if eeg_np.ndim == 2:
        eeg_np = eeg_np[None, ...]                   # add batch dim
    eeg = torch.from_numpy(eeg_np).float().to(device)
    eeg = eeg.clamp(-15, 15)

    model = BoldFlow.from_pretrained(args.checkpoint, device=device)
    model.eval()

    # Seq2seq models flow in T_out*R space; reshape outputs to (B, T_out, R)
    # so each predicted block of consecutive DiFuMo volumes is explicit.
    t_out, n_rois = model.n_out_timesteps, model.n_rois
    shape_note = f"  (B, T_out={t_out}, R={n_rois})" if t_out > 1 else ""

    def _as_blocks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], t_out, n_rois) if t_out > 1 else x

    if args.ensemble > 0:
        samples = model.sample_ensemble(eeg, n_samples=args.ensemble)
        mean, std = _as_blocks(samples.mean(dim=0)), _as_blocks(samples.std(dim=0))
        print(f"block-level mean shape: {tuple(mean.shape)}, "
              f"std shape: {tuple(std.shape)}{shape_note}")
        print(f"block-level mean range: [{mean.min().item():.3f}, {mean.max().item():.3f}]")
        print(f"block-level std  range: [{std.min().item():.3f}, {std.max().item():.3f}]")
        if args.output:
            np.savez(args.output, mean=mean.cpu().numpy(), std=std.cpu().numpy())
            print(f"saved to {args.output}")
    else:
        with torch.no_grad():
            pred = _as_blocks(model(eeg))
        print(f"prediction shape: {tuple(pred.shape)}{shape_note}")
        print(f"range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        if args.output:
            np.savez(args.output, prediction=pred.cpu().numpy())
            print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
