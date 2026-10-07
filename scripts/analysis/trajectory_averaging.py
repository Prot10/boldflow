#!/usr/bin/env python
"""Effect of trajectory averaging at fixed checkpoints (paper Table 14).

For every ensemble size ``M`` the pointwise mean of the first ``M`` cached
trajectories of each held-out scan is scored with the scan-level protocol:
MSE pools all TRs and components of a run, T.Corr is the per-component
Pearson r within each scan, and FC Corr compares within-scan FC matrices on
the cortical component mask; the last two are averaged across scans. ``M = 1``
is a single sampled trajectory. FC is computed from the averaged trajectory,
not by averaging per-trajectory FC estimates.

Every ``fold_<k>`` found under a cache directory is one run (one checkpoint);
the table reports mean and sample standard deviation (ddof=1) across all runs
given. Each
directory is loaded separately, so the same scans may appear in several
directories (one per seed) but not twice within one run.

Examples
--------
    # one cache directory per seed, each holding fold_1 ... fold_5
    python scripts/analysis/trajectory_averaging.py \\
        --trajectories outputs/trajectories_seed0 outputs/trajectories_seed1 \\
        --output outputs/analysis/trajectory_averaging.json

    # custom ensemble sizes
    python scripts/analysis/trajectory_averaging.py --trajectories outputs/trajectories \\
        --m 1 2 5 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import ScanTrajectories, fc_components, load_scans
from boldflow.metrics import fc_correlation_per_scan, pearson_r_per_scan
from boldflow.utils import save_json

M_GRID = (1, 5, 10, 25, 50, 200)
METRICS = ("fc_corr", "t_corr", "mse")


def readout_metrics(scans: Sequence[ScanTrajectories], m: int) -> Dict[str, float]:
    """Scan-level metrics of the ``m``-trajectory mean over one run's scans."""
    preds = [s.ensemble_mean(m) for s in scans]
    targets = [s.target for s in scans]
    components = fc_components(targets[0].shape[1])
    sq_err = np.concatenate([((p.astype(np.float64) - t) ** 2).ravel()
                             for p, t in zip(preds, targets)])
    return {
        "fc_corr": fc_correlation_per_scan(preds, targets, components),
        "t_corr": pearson_r_per_scan(preds, targets),
        "mse": float(sq_err.mean()),
    }


def group_runs(directories: Sequence[str | Path]) -> Dict[str, List[ScanTrajectories]]:
    """Split the cached scans into runs, one per ``(directory, fold)``.

    Each distinct directory is one run (labelled ``run_1``, ``run_2``, ... in
    the order given) and is loaded on its own. A scan name found twice within
    one run (e.g. the same directory passed twice) raises ``ValueError``.
    """
    runs: Dict[str, List[ScanTrajectories]] = {}
    sorted_roots = list(dict.fromkeys(Path(d).resolve() for d in directories))
    for directory in directories:
        root = Path(directory).resolve()
        label = f"run_{sorted_roots.index(root) + 1}"
        for scan in load_scans(directory):
            run = runs.setdefault(f"{label}/fold_{scan.fold}", [])
            if any(other.scan == scan.scan for other in run):
                raise ValueError(f"scan {scan.scan!r} appears twice in {directory}, fold {scan.fold}")
            run.append(scan)
    return runs


def sweep(runs: Dict[str, List[ScanTrajectories]], m_grid: Sequence[int]) -> Dict[str, object]:
    """Per-run metrics for every usable ``M`` and their mean/std across runs."""
    available = min(s.n_samples for scans in runs.values() for s in scans)
    m_used = [m for m in m_grid if m <= available]
    per_run = [{"run": name, "m": m, "n_scans": len(scans), **readout_metrics(scans, m)}
               for name, scans in runs.items() for m in m_used]
    summary = {}
    for m in m_used:
        rows = [r for r in per_run if r["m"] == m]
        summary[str(m)] = {
            k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
            for k in METRICS
        }
    return {"m_grid": m_used, "n_runs": len(runs), "summary": summary, "per_run": per_run}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Directories written by sample_trajectories.py.")
    p.add_argument("--m", type=int, nargs="+", default=list(M_GRID),
                   help="Ensemble sizes; values above the cached M are skipped.")
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = sweep(group_runs(args.trajectories), sorted(set(args.m)))
    skipped = sorted(set(args.m) - set(result["m_grid"]))
    if skipped:
        print(f"skipped M={skipped}: more than the cached trajectories per scan")
    print(f"{result['n_runs']} runs (mean ± std across runs)")
    print(f"{'Predictor':<28}{'FC Corr':>14}{'T.Corr':>14}{'MSE':>14}")
    for m in result["m_grid"]:
        s = result["summary"][str(m)]
        name = "Sampled trajectory (M=1)" if m == 1 else f"Ensemble mean M={m}"
        cells = "".join(f"{s[k]['mean']:>8.3f}±{s[k]['std']:.3f}" for k in METRICS)
        print(f"{name:<28}{cells}")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
