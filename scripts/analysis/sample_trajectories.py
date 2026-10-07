#!/usr/bin/env python
"""Cache sampled trajectories for the held-out scans of one fold.

For every scan in the chosen split this draws ``M`` independent sampled
trajectories (one source draw per anchor, overlap-averaged per scan) and
writes ``<output-dir>/fold_<k>/<scan>.npz`` with the ``(M, L, R)`` samples and
the ``(L, R)`` measured BOLD. All scripts in ``scripts/analysis`` read these
files, so inference runs once per checkpoint.

Examples
--------
    # 50 trajectories per test scan (UQ, trajectory averaging)
    python scripts/analysis/sample_trajectories.py \\
        --config configs/neurobolt.yaml \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt \\
        --fold 1 --n-samples 50 --output-dir outputs/trajectories

    # 200 trajectories per scan for the held-out trajectory audit
    python scripts/analysis/sample_trajectories.py ... --n-samples 200

    # validation split, used to fit the scalar recalibration
    python scripts/analysis/sample_trajectories.py ... --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from boldflow.analysis import (ScanTrajectories, model_kwargs, sample_scan_trajectories,
                               save_scan, scan_load_kwargs, subject_of)
from boldflow.data import load_scan
from boldflow.model import BoldFlow
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.utils import (ENV_DATA_ROOT, autodetect_device, load_yaml_config,
                            resolve_path, set_seed, setup_logging)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fold", type=int, default=1, help="1-indexed fold number.")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Independent sampled trajectories per scan (M).")
    p.add_argument("--deterministic", action="store_true",
                   help="Store the single source-mean trajectory instead.")
    p.add_argument("--zero-eeg", action="store_true",
                   help="Replace the EEG input by zeros (constant-input control).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=None,
                   help="Sampling seed (default: the config seed).")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    cfg = load_yaml_config(args.config)
    data_root = resolve_path(args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"))
    if not data_root or data_root.startswith("/path/to/"):
        raise SystemExit(f"data_root not set. Pass --data-root or set ${ENV_DATA_ROOT}.")
    seed = int(cfg.get("seed", 12345))
    device = autodetect_device(args.device or cfg.get("device", "cuda"))
    dataset = cfg["data"]["dataset"]

    fold = SubjectLevelCVSplitter(
        data_root=data_root, k_folds=int(cfg.get("k_folds", 5)), seed=seed,
        dataset=dataset, n_rois=int(cfg["data"]["n_rois"]),
    ).get_fold(args.fold)
    scans = fold.test_scans if args.split == "test" else fold.val_scans

    model = BoldFlow.from_pretrained(args.checkpoint, device=device, **model_kwargs(cfg))
    set_seed(seed if args.seed is None else args.seed)
    transform = torch.zeros_like if args.zero_eeg else None
    out_dir = Path(args.output_dir) / f"fold_{args.fold}"

    for scan in scans:
        eeg, fmri, meta = load_scan(data_root, scan, **scan_load_kwargs(cfg))
        if not eeg:
            continue
        samples, target = sample_scan_trajectories(
            model, torch.from_numpy(np.stack(eeg)).float(), np.stack(fmri),
            n_samples=args.n_samples, device=device, batch_size=args.batch_size,
            deterministic=args.deterministic, eeg_transform=transform,
        )
        path = save_scan(out_dir, ScanTrajectories(
            scan=scan, subject=subject_of(scan, dataset), fold=args.fold,
            target=target, samples=samples, tr=float(cfg["data"].get("tr", 2.1)),
        ))
        print(f"{scan}: samples {samples.shape} -> {path}")


if __name__ == "__main__":
    main()
