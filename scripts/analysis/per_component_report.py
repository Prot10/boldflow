#!/usr/bin/env python
"""Per-component temporal correlation and the thalamus contrast (paper Table 13).

For every DiFuMo-64 component the script computes the temporal correlation
(Pearson r between predicted and measured time courses within a held-out
scan, averaged over the scans of a fold and then over folds), using one
sampled trajectory per scan. With a second cache (``--compare-dir``) it also
reports the difference ``delta = model - comparison`` and
the number of components with ``delta > 0`` among the cortical and the
deep-gray/cerebellar components (Thalamus, Putamen, Caudate, Cerebellum
Crus II, Cerebellum I-V); non-neural components are listed but not counted.

It also reports the lag-1 autocorrelation of the measured BOLD (averaged over
a subject's scans, then over subjects) for the thalamus against the mean of
the cortical components.

Reads the caches written by ``sample_trajectories.py``. The comparison cache
is not produced by this repository: store the comparison model's predictions
for the same held-out scans in the same npz format (``samples`` of shape
``(1, L, R)``, ``target``, ``scan``, ``subject``, ``fold``, ``tr``).

Examples
--------
    python scripts/analysis/per_component_report.py \\
        --trajectories outputs/trajectories \\
        --compare-dir outputs/trajectories_baseline \\
        --output outputs/analysis/per_component.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import ScanTrajectories, load_scans
from boldflow.difumo import DIFUMO_64_LABELS, non_neural_indices
from boldflow.utils import save_json

DEEP_GRAY_CEREBELLAR = ("Thalamus", "Putamen", "Caudate",
                        "Cerebellum Crus II", "Cerebellum I-V")
GROUPS = ("cortical", "deep_gray_cerebellar", "non_neural")


def component_groups(labels: Sequence[str] = DIFUMO_64_LABELS) -> List[str]:
    """Anatomical group of every component: cortical, deep-gray/cerebellar, non-neural."""
    non_neural = non_neural_indices(len(labels))
    return ["non_neural" if i in non_neural
            else "deep_gray_cerebellar" if label in DEEP_GRAY_CEREBELLAR
            else "cortical" for i, label in enumerate(labels)]


def columnwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of two ``(L, R)`` arrays."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    return np.where(den > 0, (a * b).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)


def component_tcorr(items: Sequence[ScanTrajectories]) -> np.ndarray:
    """Per-component T.Corr: mean over scans within fold, then mean over folds."""
    by_fold: Dict[int, List[np.ndarray]] = {}
    for item in items:
        by_fold.setdefault(item.fold, []).append(
            columnwise_pearson(item.samples[0], item.target))
    return np.mean([np.nanmean(scans, axis=0) for scans in by_fold.values()], axis=0)


def lag1_autocorrelation(series: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of every column of a ``(L, R)`` series."""
    x = series - series.mean(axis=0)
    return (x[:-1] * x[1:]).sum(axis=0) / np.maximum((x ** 2).sum(axis=0), 1e-12)


def measured_lag1(items: Sequence[ScanTrajectories]) -> np.ndarray:
    """Per-component lag-1 autocorrelation of measured BOLD (scans -> subject -> mean)."""
    by_subject: Dict[str, List[np.ndarray]] = {}
    for item in items:
        by_subject.setdefault(item.subject, []).append(lag1_autocorrelation(item.target))
    return np.mean([np.mean(scans, axis=0) for scans in by_subject.values()], axis=0)


def win_counts(delta: np.ndarray, groups: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Components with ``delta > 0`` and the mean ``delta`` per anatomical group."""
    groups = np.asarray(groups)
    return {g: {"won": int((delta[groups == g] > 0).sum()), "n": int((groups == g).sum()),
                "mean_delta": float(delta[groups == g].mean())} for g in GROUPS}


def summarize(items: Sequence[ScanTrajectories],
              compare: Optional[Sequence[ScanTrajectories]] = None, *,
              labels: Sequence[str] = DIFUMO_64_LABELS) -> Dict[str, Any]:
    n_rois = items[0].target.shape[1]
    if n_rois != len(labels):
        raise ValueError(f"cache has {n_rois} components but {len(labels)} labels were given")
    groups = np.asarray(component_groups(labels))
    tcorr, lag1 = component_tcorr(items), measured_lag1(items)
    other = component_tcorr(compare) if compare is not None else None

    rows = []
    for i, label in enumerate(labels):
        row = {"index": i, "component": label, "group": groups[i],
               "tcorr": float(tcorr[i]), "measured_lag1": float(lag1[i])}
        if other is not None:
            row.update(tcorr_compare=float(other[i]), delta=float(tcorr[i] - other[i]))
        rows.append(row)
    result: Dict[str, Any] = {"components": rows, "n_scans": len(items)}
    result["lag1"] = {"cortical_mean": float(lag1[groups == "cortical"].mean())}
    if "Thalamus" in labels:
        thalamus = list(labels).index("Thalamus")
        result["lag1"]["thalamus"] = float(lag1[thalamus])
        result["thalamus"] = rows[thalamus]
    if other is not None:
        result["wins"] = win_counts(tcorr - other, groups)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Cache directories of the model (sample_trajectories.py).")
    p.add_argument("--compare-dir", type=str, nargs="+", default=None,
                   help="Cache directories of a second model to compare against.")
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectories)
    compare = load_scans(args.compare_dir) if args.compare_dir else None
    result = summarize(items, compare)

    for row in sorted(result["components"], key=lambda r: -r["tcorr"]):
        line = f"{row['component'][:48]:48s} {row['group']:21s} {row['tcorr']:6.3f}"
        if compare is not None:
            line += f" {row['tcorr_compare']:6.3f} {row['delta']:+6.3f}"
        print(line)
    for group, w in result.get("wins", {}).items():
        print(f"{group}: delta > 0 in {w['won']}/{w['n']} components "
              f"(mean delta {w['mean_delta']:+.3f})")
    lag1 = result["lag1"]
    print(f"measured lag-1 autocorrelation: thalamus {lag1.get('thalamus', float('nan')):.3f} "
          f"vs cortical mean {lag1['cortical_mean']:.3f}")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
