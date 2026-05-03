#!/usr/bin/env python
"""Qualitative paper figures: predicted vs ground-truth time courses + FC matrices.

Inputs
------
A ``predictions.pt`` saved by ``scripts/evaluate.py --save-predictions``,
i.e. a dict with ``predictions``, ``targets`` and ``metrics``.

Outputs
-------
* ``timeseries.pdf``: per-ROI predicted vs. ground-truth time course for the
  ROIs given by ``--rois`` (defaults to three primary sensory cortices).
* ``fc_matrices.pdf``: side-by-side ground-truth and predicted functional
  connectivity matrices, plus their difference.

These reproduce the structure of Figs. ``qualitative_timeseries`` and
``fc_matrix_comparison`` in the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from boldflow.difumo import DIFUMO_64_LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative time-course + FC figures.")
    p.add_argument("--predictions", type=str, required=True,
                   help="Path to a .pt file with `predictions` and `targets` tensors.")
    p.add_argument("--output-dir", type=str, default="figures/qualitative")
    p.add_argument("--rois", type=str, nargs="+",
                   default=["Calcarine cortex posterior", "Central sulcus", "Heschl’s gyrus"],
                   help="DiFuMo-64 region names to plot.")
    p.add_argument("--max-tr", type=int, default=200,
                   help="Truncate time courses to this many TRs for the plot.")
    return p.parse_args()


def _find_roi(name: str, labels: List[str]) -> int:
    norm = name.replace("’", "'").replace("‘", "'").lower()
    for i, label in enumerate(labels):
        if label.replace("’", "'").lower() == norm:
            return i
    # Loose substring fallback.
    for i, label in enumerate(labels):
        if norm in label.replace("’", "'").lower():
            return i
    raise ValueError(f"ROI {name!r} not found in DiFuMo-64.")


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.predictions, map_location="cpu", weights_only=False)
    pred = payload["predictions"].numpy()
    target = payload["targets"].numpy()
    if pred.shape != target.shape:
        raise SystemExit(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    n_rois = pred.shape[1]
    labels = DIFUMO_64_LABELS if n_rois == 64 else [f"ROI {i}" for i in range(n_rois)]

    # ---- timeseries ----
    n_to_plot = len(args.rois)
    fig, axes = plt.subplots(n_to_plot, 1, figsize=(10, 2.4 * n_to_plot), sharex=True)
    if n_to_plot == 1:
        axes = [axes]
    end = min(args.max_tr, pred.shape[0])
    xs = np.arange(end)
    for ax, roi_name in zip(axes, args.rois, strict=True):
        idx = _find_roi(roi_name, labels)
        gt = target[:end, idx]
        pr = pred[:end, idx]
        r = np.corrcoef(pr, gt)[0, 1]
        ax.plot(xs, gt, "--", label="ground truth", color="black", linewidth=1)
        ax.plot(xs, pr, "-", label="prediction", color="C0", linewidth=1)
        ax.fill_between(xs, gt, pr, alpha=0.15, color="C0")
        ax.set_ylabel(roi_name, fontsize=9)
        ax.set_title(f"{roi_name}  (r = {r:.3f})", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("TR")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "timeseries.pdf", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'timeseries.pdf'}")

    # ---- FC matrices ----
    if n_rois >= 2:
        fc_t = np.corrcoef(target, rowvar=False)
        fc_p = np.corrcoef(pred, rowvar=False)
        diff = fc_p - fc_t
        vmax = max(np.abs(fc_t).max(), np.abs(fc_p).max())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, mat, title in zip(
            axes, [fc_t, fc_p, diff],
            ["Ground truth FC", "Predicted FC", "Predicted - Ground truth"],
            strict=True,
        ):
            limit = vmax if "Predicted -" not in title else np.abs(diff).max()
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_title(title)
            ax.set_xlabel("ROI")
            ax.set_ylabel("ROI")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / "fc_matrices.pdf", dpi=150)
        plt.close(fig)
        print(f"wrote {out / 'fc_matrices.pdf'}")


if __name__ == "__main__":
    main()
