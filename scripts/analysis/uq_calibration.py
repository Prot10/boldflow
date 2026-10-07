#!/usr/bin/env python
"""Native ensemble uncertainty: error ranking, calibration error and coverage.

Corresponds to the "Raw ensemble" and "+ scalar recalibration" rows of the
uncertainty table (Table 2) and to "Output spread and scalar recalibration"
in the uncertainty appendix. For every held-out scan the prediction centre is
the pointwise mean of the first ``M`` cached trajectories and the raw
uncertainty is their Bessel-corrected standard deviation, per TR and
component. One scalar ``alpha`` per fold is fitted on the validation cache
and applied to the test cache. At nominal level ``q`` the interval is
``mean +/- z_{(1+q)/2} * alpha * std``.

Reported per fold (pooled over all test TRs and components) and as mean and
standard deviation across folds, for the raw and the recalibrated spread:

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
from boldflow.uncertainty import expected_calibration_error, spearman_residual_std
from boldflow.utils import save_json

EPS = 1e-8
# 20 bins -> interior levels 0.05, 0.10, ..., 0.95 in expected_calibration_error.
CALIBRATION_BINS = 20


def pool_scans(items: Sequence[ScanTrajectories], m: Optional[int] = None,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``(target, ensemble mean, ensemble std)`` over scans, each ``(N, R)``."""
    for item in items:
        if m is not None and item.n_samples < m:
            raise ValueError(f"{item.scan}: {item.n_samples} cached trajectories < M={m}")
    return (np.concatenate([i.target for i in items]).astype(np.float64),
            np.concatenate([i.ensemble_mean(m) for i in items]).astype(np.float64),
            np.concatenate([i.ensemble_std(m) for i in items]).astype(np.float64))


def fit_scalar_alpha(target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Gaussian maximum-likelihood scale ``alpha = sqrt(mean((y - mu)^2 / s^2))``.

    This is the single multiplier that makes the standardised validation
    residuals have unit second moment, so that ``alpha * s`` is a calibrated
    Gaussian standard deviation.
    """
    ratio = (np.asarray(target) - np.asarray(mean)) / np.clip(std, EPS, None)
    return float(np.sqrt(np.mean(ratio ** 2)))


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
        "mean_std": float(np.mean(std)),
    }


def evaluate_fold(val_items: Sequence[ScanTrajectories],
                  test_items: Sequence[ScanTrajectories],
                  m: Optional[int] = None, level: float = 0.95) -> Dict[str, Any]:
    """Fit ``alpha`` on validation scans and score raw / recalibrated test spread."""
    alpha = fit_scalar_alpha(*pool_scans(val_items, m))
    target, mean, std = pool_scans(test_items, m)
    return {
        "alpha": alpha,
        "n_val_scans": len(val_items), "n_test_scans": len(test_items),
        "n_test_points": int(target.size),
        "raw": uq_metrics(target, mean, std, level),
        "recalibrated": uq_metrics(target, mean, alpha * std, level),
    }


def by_fold(items: Sequence[ScanTrajectories]) -> Dict[int, List[ScanTrajectories]]:
    grouped: Dict[int, List[ScanTrajectories]] = {}
    for item in items:
        grouped.setdefault(item.fold, []).append(item)
    return grouped


def run(val_items: Sequence[ScanTrajectories], test_items: Sequence[ScanTrajectories],
        m: Optional[int] = None, level: float = 0.95) -> Dict[str, Any]:
    """Per-fold results plus mean and (sample) standard deviation across folds."""
    val, test = by_fold(val_items), by_fold(test_items)
    missing = sorted(set(test) - set(val))
    if missing:
        raise ValueError(f"no validation cache for fold(s) {missing}")
    folds = {k: evaluate_fold(val[k], test[k], m, level) for k in sorted(test)}

    def across(values: List[float]) -> Dict[str, float]:
        return {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    summary: Dict[str, Any] = {"alpha": across([f["alpha"] for f in folds.values()])}
    for readout in ("raw", "recalibrated"):
        summary[readout] = {
            key: across([f[readout][key] for f in folds.values()])
            for key in ("spearman", "calibration_error", "coverage", "mean_std")
        }
    return {"n_samples": m, "level": level, "n_folds": len(folds),
            "summary": summary, "folds": {str(k): v for k, v in folds.items()}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--val-dir", type=str, nargs="+", required=True,
                   help="Validation trajectory cache(s); used only to fit alpha.")
    p.add_argument("--test-dir", type=str, nargs="+", required=True,
                   help="Test trajectory cache(s).")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Use the first M cached trajectories of every scan.")
    p.add_argument("--level", type=float, default=0.95, help="Nominal coverage level.")
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(load_scans(args.val_dir), load_scans(args.test_dir),
                 m=args.n_samples, level=args.level)
    s = result["summary"]
    print(f"M={args.n_samples}, {result['n_folds']} fold(s); "
          f"alpha = {s['alpha']['mean']:.3f} +/- {s['alpha']['std']:.3f}")
    print(f"{'readout':<14}{'Spearman':>18}{'Cal. Err.':>18}{f'Cov@{args.level:.0%}':>18}")
    for readout in ("raw", "recalibrated"):
        cells = "".join(f"{s[readout][k]['mean']:>10.3f} +/- {s[readout][k]['std']:.3f}"
                        for k in ("spearman", "calibration_error", "coverage"))
        print(f"{readout:<14}{cells}")
    if args.output:
        save_json(result, args.output)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
