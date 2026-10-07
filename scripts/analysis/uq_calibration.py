#!/usr/bin/env python
"""Native ensemble uncertainty: error ranking, calibration error and coverage.

Corresponds to the "Raw ensemble" and "+ scalar recalibration" rows of the
uncertainty table (Table 2) and to "Output spread and scalar recalibration"
in the uncertainty appendix. For every held-out scan the prediction centre is
the pointwise mean of the first ``M`` cached trajectories and the raw
uncertainty is their Bessel-corrected standard deviation, per TR and
component. At nominal level ``q`` the interval is
``mean +/- z_{(1+q)/2} * alpha * std``.

A run is one trained model: one fold of one training seed. ``--val-dir`` and
``--test-dir`` take the same number of cache directories; the directories at
the same position hold the validation and test caches of one training seed,
and every fold found in such a pair is one run. One scalar ``alpha`` per run
is fitted on the validation scans and applied to the test scans; validation
and test subjects of a run must be disjoint.

Reported per run (pooled over all test TRs and components) and as mean and
standard deviation across runs (folds x seeds), for the raw and the
recalibrated spread:

* Spearman correlation between absolute residual and uncertainty,
* calibration error: mean absolute gap between empirical and nominal
  coverage of central Gaussian intervals at levels 0.05, 0.10, ..., 0.95,
* coverage of the nominal 95% interval.

Examples
--------
    # caches written by sample_trajectories.py with --split val / --split test
    python scripts/analysis/uq_calibration.py \\
        --val-dir outputs/trajectories_val --test-dir outputs/trajectories \\
        --n-samples 50 --output outputs/analysis/uq_calibration.json

    # three training seeds: one validation and one test directory per seed
    python scripts/analysis/uq_calibration.py \\
        --val-dir val_seed1 val_seed2 val_seed3 \\
        --test-dir test_seed1 test_seed2 test_seed3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import norm

from boldflow.analysis import ScanTrajectories, load_scans
from boldflow.uncertainty import (ScalarRecalibration, expected_calibration_error,
                                  spearman_residual_std)
from boldflow.utils import save_json

EPS = 1e-8
# 20 bins -> interior levels 0.05, 0.10, ..., 0.95 in expected_calibration_error.
CALIBRATION_BINS = 20
METRIC_KEYS = ("spearman", "calibration_error", "coverage")

CachePair = Tuple[Sequence[ScanTrajectories], Sequence[ScanTrajectories]]


def pool_scans(items: Sequence[ScanTrajectories], m: Optional[int] = None,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``(target, ensemble mean, ensemble std)`` over scans, each ``(N, R)``."""
    for item in items:
        if m is not None and item.n_samples < m:
            raise ValueError(f"{item.scan}: {item.n_samples} cached trajectories < M={m}")
    return (np.concatenate([i.target for i in items]).astype(np.float64),
            np.concatenate([i.ensemble_mean(m) for i in items]).astype(np.float64),
            np.concatenate([i.ensemble_std(m) for i in items]).astype(np.float64))


def interval_coverage(target: np.ndarray, mean: np.ndarray, std: np.ndarray,
                      level: float = 0.95) -> float:
    """Fraction of targets inside ``mean +/- z_{(1+level)/2} * std``."""
    z = norm.ppf(0.5 + level / 2.0)
    return float(np.mean(np.abs(np.asarray(target) - mean) <= z * np.clip(std, EPS, None)))


def uq_metrics(target: np.ndarray, mean: np.ndarray, std: np.ndarray,
               level: float = 0.95) -> Dict[str, float]:
    """Error ranking, calibration error and coverage for one uncertainty readout."""
    return {
        "spearman": spearman_residual_std(target, mean, std),
        "calibration_error": expected_calibration_error(target, mean, std,
                                                        n_bins=CALIBRATION_BINS),
        "coverage": interval_coverage(target, mean, std, level),
    }


def evaluate_run(val_items: Sequence[ScanTrajectories],
                 test_items: Sequence[ScanTrajectories],
                 m: Optional[int] = None, level: float = 0.95) -> Dict[str, Any]:
    """Fit ``alpha`` on validation scans and score raw / recalibrated test spread."""
    shared = sorted({i.subject for i in val_items} & {i.subject for i in test_items})
    if shared:
        raise ValueError(f"validation and test caches share subject(s) {shared}")
    val_target, val_mean, val_std = pool_scans(val_items, m)
    recalibration = ScalarRecalibration().fit(val_target - val_mean, val_std)
    target, mean, std = pool_scans(test_items, m)
    return {
        "alpha": recalibration.alpha,
        "n_val_scans": len(val_items), "n_test_scans": len(test_items),
        "n_test_points": int(target.size),
        "raw": uq_metrics(target, mean, std, level),
        "recalibrated": uq_metrics(target, mean, recalibration(std), level),
    }


def by_fold(items: Sequence[ScanTrajectories]) -> Dict[int, List[ScanTrajectories]]:
    grouped: Dict[int, List[ScanTrajectories]] = {}
    for item in items:
        grouped.setdefault(item.fold, []).append(item)
    return grouped


def run(pairs: Sequence[CachePair], m: Optional[int] = None,
        level: float = 0.95) -> Dict[str, Any]:
    """Per-run results plus mean and sample standard deviation across runs.

    ``pairs`` holds one ``(validation scans, test scans)`` tuple per training
    seed; every fold of a pair is one run, keyed ``"<pair>/fold_<k>"`` with the
    1-indexed pair position.
    """
    runs: Dict[str, Dict[str, Any]] = {}
    for position, (val_items, test_items) in enumerate(pairs, start=1):
        val, test = by_fold(val_items), by_fold(test_items)
        missing = sorted(set(test) - set(val))
        if missing:
            raise ValueError(f"pair {position}: no validation cache for fold(s) {missing}")
        for fold in sorted(test):
            runs[f"{position}/fold_{fold}"] = evaluate_run(val[fold], test[fold], m, level)

    def across(values: List[float]) -> Dict[str, float]:
        return {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    summary: Dict[str, Any] = {"alpha": across([r["alpha"] for r in runs.values()])}
    for readout in ("raw", "recalibrated"):
        summary[readout] = {key: across([r[readout][key] for r in runs.values()])
                            for key in METRIC_KEYS}
    return {"n_samples": m, "level": level, "n_runs": len(runs),
            "summary": summary, "runs": runs}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--val-dir", type=str, nargs="+", required=True,
                   help="Validation trajectory cache, one directory per training "
                        "seed; used only to fit alpha.")
    p.add_argument("--test-dir", type=str, nargs="+", required=True,
                   help="Test trajectory cache, one directory per training seed, "
                        "in the same order as --val-dir.")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Use the first M cached trajectories of every scan.")
    p.add_argument("--level", type=float, default=0.95, help="Nominal coverage level.")
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.val_dir) != len(args.test_dir):
        raise SystemExit("--val-dir and --test-dir need the same number of directories")
    pairs = [(load_scans(v), load_scans(t)) for v, t in zip(args.val_dir, args.test_dir)]
    result = run(pairs, m=args.n_samples, level=args.level)
    s = result["summary"]
    print(f"M={args.n_samples}, {result['n_runs']} run(s); "
          f"alpha = {s['alpha']['mean']:.3f} +/- {s['alpha']['std']:.3f}")
    print(f"{'readout':<14}{'Spearman':>18}{'Cal. Err.':>18}{f'Cov@{args.level:.0%}':>18}")
    for readout in ("raw", "recalibrated"):
        cells = "".join(f"{s[readout][k]['mean']:>10.3f} +/- {s[readout][k]['std']:.3f}"
                        for k in METRIC_KEYS)
        print(f"{readout:<14}{cells}")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
