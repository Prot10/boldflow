#!/usr/bin/env python
"""FC recovery of the model versus a constant-input generator (Appendix D).

The constant-input generator is the same architecture retrained with the EEG
replaced by zero, so it can only reproduce population structure. This script
compares two cached sets of trajectories covering the same held-out scans:

* sampled-trajectory FC Corr: FC of one sampled trajectory against the
  measured FC of the same scan (averaged over ``--n-trajectories`` trajectories
  when more than one is requested);
* ensemble-mean FC Corr: FC of the pointwise mean of the first
  ``--ensemble-size`` trajectories (all cached ones by default).

FC is computed within scan on the cortical component mask. The paired
model-minus-constant difference is bootstrapped over subjects.

Examples
--------
    python scripts/analysis/constant_input_fc.py \\
        --trajectory-dir outputs/trajectories \\
        --constant-input-dir outputs/trajectories_constant_input \\
        --output outputs/analysis/constant_input_fc.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import (ScanTrajectories, fc_components, fc_matrix, fc_similarity,
                               load_scans, subject_bootstrap)
from boldflow.utils import save_json

READOUTS = ("sampled", "ensemble_mean")


def fc_corr_scores(
    items: Sequence[ScanTrajectories],
    n_trajectories: int = 1,
    ensemble_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Scan-level FC Corr of the sampled-trajectory and ensemble-mean readouts."""
    comps = fc_components(items[0].target.shape[1])
    rows = []
    for item in items:
        target = fc_matrix(item.target, comps)
        sampled = [fc_similarity(fc_matrix(traj, comps), target)
                   for traj in item.samples[:n_trajectories]]
        mean_fc = fc_matrix(item.ensemble_mean(ensemble_size), comps)
        rows.append({
            "scan": item.scan, "subject": item.subject, "fold": item.fold,
            "sampled": float(np.nanmean(sampled)),
            "ensemble_mean": fc_similarity(mean_fc, target),
            "ensemble_size": int(min(ensemble_size or item.n_samples, item.n_samples)),
        })
    return rows


def _fold_mean(rows: Sequence[Dict[str, Any]], field: str) -> float:
    folds = sorted({r["fold"] for r in rows})
    return float(np.mean([np.nanmean([r[field] for r in rows if r["fold"] == k])
                          for k in folds]))


def compare(
    model_rows: Sequence[Dict[str, Any]],
    constant_rows: Sequence[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-readout FC Corr of both generators and their paired difference."""
    constant = {r["scan"]: r for r in constant_rows}
    if set(constant) != {r["scan"] for r in model_rows}:
        raise ValueError("model and constant-input caches do not cover the same scans")
    paired = [constant[r["scan"]] for r in model_rows]
    subjects = [r["subject"] for r in model_rows]
    out: Dict[str, Any] = {}
    for field in READOUTS:
        a = [r[field] for r in model_rows]
        b = [r[field] for r in paired]
        kw = dict(n_boot=n_boot, seed=seed)
        out[field] = {
            "model": {**subject_bootstrap(a, subjects, **kw),
                      "scan_mean": float(np.nanmean(a)),
                      "fold_mean": _fold_mean(model_rows, field)},
            "constant_input": {**subject_bootstrap(b, subjects, **kw),
                               "scan_mean": float(np.nanmean(b)),
                               "fold_mean": _fold_mean(paired, field)},
            "difference": subject_bootstrap(np.subtract(a, b), subjects, **kw),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory-dir", type=str, nargs="+", required=True,
                   help="Cached trajectories of the model.")
    p.add_argument("--constant-input-dir", type=str, nargs="+", required=True,
                   help="Cached trajectories of the constant-input generator.")
    p.add_argument("--n-trajectories", type=int, default=1,
                   help="Sampled trajectories whose FC Corr is averaged per scan.")
    p.add_argument("--ensemble-size", type=int, default=None,
                   help="Trajectories in the ensemble mean (default: all cached).")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_rows = fc_corr_scores(load_scans(args.trajectory_dir),
                                args.n_trajectories, args.ensemble_size)
    constant_rows = fc_corr_scores(load_scans(args.constant_input_dir),
                                   args.n_trajectories, args.ensemble_size)
    summary = compare(model_rows, constant_rows, args.n_boot, args.seed)
    result = {
        "n_scans": len(model_rows), "n_subjects": len({r["subject"] for r in model_rows}),
        "n_trajectories": args.n_trajectories,
        "ensemble_size": {"model": sorted({r["ensemble_size"] for r in model_rows}),
                          "constant_input": sorted({r["ensemble_size"] for r in constant_rows})},
        "fc_corr": summary,
        "scans": {"model": model_rows, "constant_input": constant_rows},
    }
    print(f"FC Corr over {result['n_scans']} scans, {result['n_subjects']} subjects "
          "(fold mean; subject-bootstrap interval of the difference)")
    for field in READOUTS:
        s = summary[field]
        d = s["difference"]
        print(f"  {field:<14s} model {s['model']['fold_mean']:.3f}  "
              f"constant-input {s['constant_input']['fold_mean']:.3f}  "
              f"difference {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    save_json(result, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
