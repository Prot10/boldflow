#!/usr/bin/env python
"""Effective rank of the measured fMRI targets (paper Appendix C).

The effective rank reported here is the participation ratio
``(sum l)^2 / sum l^2`` of the eigenvalues ``l`` of the measured
component-by-component correlation matrix of a held-out scan, over all output
components. Scan values are averaged within subject and the mean is reported
with a subject-bootstrap interval. ``--exclude-non-neural`` repeats the
computation after dropping the predominantly non-neural components of the
parcellation; it requires a component count with a non-neural list (64, 256
or 512).

Only the measured targets stored in the trajectory cache are used, so a cache
written with ``--n-samples 1`` is enough.

Examples
--------
    python scripts/analysis/effective_rank.py --trajectories outputs/trajectories_512 \\
        --output outputs/analysis/effective_rank_512.json

    python scripts/analysis/effective_rank.py --trajectories outputs/trajectories_512 \\
        --exclude-non-neural
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import (ScanTrajectories, effective_rank, fc_matrix, load_scans,
                               subject_bootstrap)
from boldflow.difumo import non_neural_indices
from boldflow.utils import save_json


def neural_components(n_rois: int) -> np.ndarray:
    """Component indices left after removing the non-neural ones."""
    excluded = non_neural_indices(n_rois)
    if not excluded:
        raise ValueError(f"no non-neural component list for {n_rois} components")
    return np.array([i for i in range(n_rois) if i not in excluded])


def measured_effective_rank(target: np.ndarray, exclude_non_neural: bool = False) -> Dict[str, float]:
    """Participation ratio of one scan's ``(L, R)`` correlation matrix."""
    series = np.asarray(target, dtype=np.float64)
    if exclude_non_neural:
        series = series[:, neural_components(series.shape[1])]
    series = series[:, series.std(axis=0) > 1e-10]
    return {"effective_rank": effective_rank(fc_matrix(series)),
            "n_components": series.shape[1]}


def summarize(scans: Sequence[ScanTrajectories], *, exclude_non_neural: bool = False,
              n_boot: int = 10000, seed: int = 0) -> Dict[str, object]:
    """Per-scan ranks and their subject-level mean with a bootstrap interval."""
    seen, rows = set(), []
    for scan in scans:          # the same scan may sit in several cache directories
        if scan.scan in seen:
            continue
        seen.add(scan.scan)
        rows.append({"scan": scan.scan, "subject": scan.subject,
                     **measured_effective_rank(scan.target, exclude_non_neural)})
    subjects: List[str] = [r["subject"] for r in rows]
    summary = {"effective_rank": subject_bootstrap(
        [r["effective_rank"] for r in rows], subjects, n_boot=n_boot, seed=seed)}
    return {"definition": "participation ratio",
            "exclude_non_neural": exclude_non_neural, "n_scans": len(rows),
            "n_components": rows[0]["n_components"], "summary": summary, "per_scan": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Directories written by sample_trajectories.py.")
    p.add_argument("--exclude-non-neural", action="store_true",
                   help="Drop the non-neural components before computing the rank.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # the same scans may sit in several cache directories: load each on its own
    scans = [scan for directory in args.trajectories for scan in load_scans(directory)]
    results = [summarize(scans, exclude_non_neural=False, n_boot=args.n_boot, seed=args.seed)]
    if args.exclude_non_neural:
        results.append(summarize(scans, exclude_non_neural=True,
                                 n_boot=args.n_boot, seed=args.seed))
    print(f"{'Components':<28}{'R':>6}{'Participation ratio [95% CI]':>30}{'scans/subjects':>18}")
    for res in results:
        s = res["summary"]["effective_rank"]
        label = "neural only" if res["exclude_non_neural"] else "all"
        interval = f"{s['mean']:.1f} [{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        print(f"{label:<28}{res['n_components']:>6}{interval:>30}"
              f"{res['n_scans']:>12}/{s['n_subjects']}")
    if args.output:
        save_json({"results": results}, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
