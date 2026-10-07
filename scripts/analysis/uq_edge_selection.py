#!/usr/bin/env python
"""Edge-error selection: does ensemble spread rank held-out errors?

Corresponds to "Edge-error selection" in the uncertainty appendix. For every
held-out scan, with the first ``M`` cached trajectories:

* FC-edge level: FC is computed for each trajectory; the edge estimate is the
  mean and the edge uncertainty the Bessel-corrected standard deviation of
  the ``M`` edge values; the edge error is ``|estimate - measured FC edge|``.
* component level: per-component time-average of the pointwise ensemble
  spread against the time-average of ``|ensemble mean - target|``.
* time-point level: the same two quantities averaged over components.

At each level the script reports the within-scan Spearman correlation between
uncertainty and error and the selective error reduction
``1 - risk(least-uncertain fraction) / risk(random fraction)`` (retention
fraction one half by default; the random-subset risk is its expectation, the
mean error over all items). Scan values are averaged within subject (Fisher-z
for correlations); the reported estimate is the mean over subjects with a
percentile bootstrap over subjects, a one-sided subject sign-flip test against
zero, Holm correction across the edge and time-point correlations, and the
sign of the estimate in every fold.

Examples
--------
    python scripts/analysis/uq_edge_selection.py \\
        --input-dir outputs/trajectories --n-samples 50 \\
        --output outputs/analysis/uq_edge_selection.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import spearmanr

from boldflow.analysis import (ScanTrajectories, fc_components, fc_matrix, load_scans,
                               subject_bootstrap, subject_means, upper_triangle)
from boldflow.utils import save_json

LEVELS = ("edge", "component", "timepoint")
HOLM_FAMILY = ("edge_spearman", "timepoint_spearman")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation; NaN for fewer than four items or a constant input."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(spearmanr(x, y)[0])


def selection_gain(uncertainty: np.ndarray, error: np.ndarray, fraction: float = 0.5) -> float:
    """Relative error reduction from keeping the least-uncertain ``fraction``.

    ``1 - mean(error of kept items) / mean(error of all items)``; the
    denominator is the expected risk of a uniformly random subset of any size.
    """
    n_keep = max(1, min(error.size, int(round(fraction * error.size))))
    kept = np.argsort(uncertainty, kind="stable")[:n_keep]
    return float(1.0 - error[kept].mean() / max(error.mean(), 1e-12))


def edge_statistics(item: ScanTrajectories, m: Optional[int] = None,
                    components: Optional[Sequence[int]] = None,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """FC-edge uncertainty and absolute error for one scan, each ``(E,)``."""
    edges = np.stack([upper_triangle(fc_matrix(traj.astype(np.float64), components))
                      for traj in item.samples[:m]])
    target = upper_triangle(fc_matrix(item.target.astype(np.float64), components))
    return edges.std(axis=0, ddof=1), np.abs(edges.mean(axis=0) - target)


def scan_row(item: ScanTrajectories, m: Optional[int] = None,
             components: Optional[Sequence[int]] = None, fraction: float = 0.5,
             ) -> Dict[str, Any]:
    """Ranking and selection statistics of one scan at the three levels."""
    spread = item.ensemble_std(m).astype(np.float64)
    abs_error = np.abs(item.ensemble_mean(m).astype(np.float64) - item.target)
    pairs = {"edge": edge_statistics(item, m, components),
             "component": (spread.mean(axis=0), abs_error.mean(axis=0)),
             "timepoint": (spread.mean(axis=1), abs_error.mean(axis=1))}
    row: Dict[str, Any] = {"scan": item.scan, "subject": item.subject, "fold": item.fold}
    for level, (uncertainty, error) in pairs.items():
        row[f"{level}_spearman"] = safe_spearman(uncertainty, error)
        row[f"{level}_selection_gain"] = selection_gain(uncertainty, error, fraction)
    return row


def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values."""
    adjusted, running = {}, 0.0
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def subject_test(values: Sequence[float], subjects: Sequence[str], *, fisher: bool,
                 n_boot: int = 10000, n_perm: int = 50000, seed: int = 0) -> Dict[str, Any]:
    """Subject-level mean, bootstrap interval and one-sided sign-flip p-value.

    With ``fisher=True`` the values are correlations: they are averaged and
    tested in Fisher-z space and the estimate and interval are transformed back.
    """
    values = np.asarray(values, dtype=np.float64)
    z = np.arctanh(np.clip(values, -0.999999, 0.999999)) if fisher else values
    boot = subject_bootstrap(z, subjects, n_boot=n_boot, seed=seed)
    per_subject = np.array(list(subject_means(z, subjects).values()))
    signs = np.random.default_rng(seed + 1).choice([-1.0, 1.0], size=(n_perm, per_subject.size))
    null = (signs * per_subject).mean(axis=1)
    back = np.tanh if fisher else (lambda v: v)
    return {"estimate": float(back(boot["mean"])),
            "ci_low": float(back(boot["ci_low"])), "ci_high": float(back(boot["ci_high"])),
            "p_greater_zero": float((1 + np.count_nonzero(null >= boot["mean"])) / (n_perm + 1)),
            "n_subjects": boot["n_subjects"],
            "n_scans": int(np.isfinite(values).sum())}


def fold_estimates(rows: Sequence[Dict[str, Any]], key: str, fisher: bool) -> Dict[str, float]:
    """Mean over subjects within each fold (Fisher-z for correlations)."""
    out = {}
    for fold in sorted({r["fold"] for r in rows}):
        sub = [r for r in rows if r["fold"] == fold]
        values = np.array([r[key] for r in sub], dtype=np.float64)
        z = np.arctanh(np.clip(values, -0.999999, 0.999999)) if fisher else values
        mean = float(np.mean(list(subject_means(z, [r["subject"] for r in sub]).values())))
        out[str(fold)] = float(np.tanh(mean)) if fisher else mean
    return out


def summarize(rows: Sequence[Dict[str, Any]], *, n_boot: int = 10000, n_perm: int = 50000,
              seed: int = 0) -> Dict[str, Any]:
    """Subject-level inference for every endpoint, Holm correction and fold signs."""
    subjects = [r["subject"] for r in rows]
    endpoints: Dict[str, Any] = {}
    keys = [f"{lv}_{kind}" for kind in ("spearman", "selection_gain") for lv in LEVELS]
    for index, key in enumerate(keys):
        fisher = key.endswith("_spearman")
        entry = subject_test([r[key] for r in rows], subjects, fisher=fisher,
                             n_boot=n_boot, n_perm=n_perm, seed=seed + 100 * index)
        entry["fold_estimates"] = fold_estimates(rows, key, fisher)
        entry["positive_folds"] = int(sum(v > 0 for v in entry["fold_estimates"].values()))
        entry["n_folds"] = len(entry["fold_estimates"])
        endpoints[key] = entry
    for key, p in holm_adjust({k: endpoints[k]["p_greater_zero"] for k in HOLM_FAMILY}).items():
        endpoints[key]["p_holm"] = p
    return endpoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-dir", type=str, nargs="+", required=True,
                   help="Test trajectory cache(s) (all folds).")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Use the first M cached trajectories of every scan.")
    p.add_argument("--fraction", type=float, default=0.5, help="Retention fraction.")
    p.add_argument("--fc-components", choices=["all", "cortical"], default="all",
                   help="FC edges among all components or within the cortical-network mask.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.input_dir)
    n_rois = items[0].target.shape[1]
    components = fc_components(n_rois) if args.fc_components == "cortical" else None
    rows: List[Dict[str, Any]] = [scan_row(i, args.n_samples, components, args.fraction)
                                  for i in items]
    endpoints = summarize(rows, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)

    print(f"{len(rows)} scans, M={args.n_samples}, FC components: {args.fc_components}, "
          f"retention fraction {args.fraction}")
    for key, e in endpoints.items():
        holm = f", Holm p={e['p_holm']:.4g}" if "p_holm" in e else ""
        print(f"{key:<26} {e['estimate']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]  "
              f"p={e['p_greater_zero']:.4g}{holm}  "
              f"positive folds {e['positive_folds']}/{e['n_folds']}")
    if args.output:
        save_json({"n_samples": args.n_samples, "fraction": args.fraction,
                   "fc_components": args.fc_components, "holm_family": list(HOLM_FAMILY),
                   "endpoints": endpoints, "scans": rows}, args.output)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
