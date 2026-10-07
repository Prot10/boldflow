#!/usr/bin/env python
"""Regional and temporal associations of the ensemble spread.

Corresponds to "Regional and temporal associations" in the uncertainty
appendix. Works on the DiFuMo-64 test cache. The three non-neural components
are dropped; the remaining 61 are split into 5 deep-gray/cerebellar
components (Thalamus, Putamen, Caudate, Cerebellum Crus II, Cerebellum I-V)
and 56 cortical components.

Per scan and component the script computes the time-averaged ensemble spread
(Bessel-corrected std over the first ``M`` trajectories) and three properties
of the measured BOLD: temporal SD, roughness (one minus the lag-1 autocorrelation) and the
fraction of Welch power in 0.08-0.15 Hz. Scans are averaged within subject and
subjects with equal weight. Reported:

* mean spread in deep-gray/cerebellar versus cortical components, with a
  subject bootstrap interval and a component-label permutation p-value;
* Spearman correlation across the 61 components between spread and roughness
  and between spread and 0.08-0.15 Hz power: subject bootstrap interval,
  one-sided component-label permutation p-value, Holm correction across the
  two, and the rank-partial correlation controlling for target SD.

Examples
--------
    python scripts/analysis/uq_structure.py \\
        --input-dir outputs/trajectories --n-samples 50 \\
        --output outputs/analysis/uq_structure.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import rankdata

from boldflow.analysis import ScanTrajectories, load_scans
from boldflow.difumo import DIFUMO_64_LABELS, non_neural_indices
from boldflow.utils import save_json

DEEP_GRAY_CEREBELLAR = ("Thalamus", "Putamen", "Caudate", "Cerebellum Crus II",
                        "Cerebellum I-V")
BAND_HZ = (0.08, 0.15)
PRIMARY = ("spread_vs_roughness", "spread_vs_band_power")


def component_groups(n_rois: int = 64) -> Dict[str, np.ndarray]:
    """Index arrays of the cortical, deep-gray/cerebellar and non-neural components."""
    if n_rois != len(DIFUMO_64_LABELS):
        raise ValueError("the anatomical grouping is defined for DiFuMo-64 only")
    non_neural = sorted(non_neural_indices(n_rois))
    deep = [DIFUMO_64_LABELS.index(name) for name in DEEP_GRAY_CEREBELLAR]
    cortical = [i for i in range(n_rois) if i not in non_neural and i not in deep]
    return {"cortical": np.array(cortical), "deep_gray_cerebellar": np.array(sorted(deep)),
            "non_neural": np.array(non_neural)}


def target_dynamics(target: np.ndarray, tr: float) -> Dict[str, np.ndarray]:
    """Per-component temporal SD, roughness and band-power fraction of a ``(L, R)`` series."""
    x = np.asarray(target, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    lag1 = (x[:-1] * x[1:]).sum(axis=0) / np.maximum((x * x).sum(axis=0), 1e-12)
    segment = min(128, x.shape[0])
    freqs, power = welch(x, fs=1.0 / tr, nperseg=segment, noverlap=segment // 2,
                         detrend="constant", scaling="density", axis=0)
    band = (freqs >= BAND_HZ[0]) & (freqs < BAND_HZ[1])
    if band.sum() < 2:
        raise ValueError("scan too short to resolve the 0.08-0.15 Hz band")
    fraction = trapezoid(power[band], freqs[band], axis=0) / np.maximum(
        trapezoid(power, freqs, axis=0), 1e-12)
    return {"temporal_sd": x.std(axis=0, ddof=1), "roughness": 1.0 - lag1,
            "band_power": fraction}


def scan_component_stats(item: ScanTrajectories, m: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Per-component spread, absolute error and target dynamics of one scan."""
    spread = item.ensemble_std(m).astype(np.float64).mean(axis=0)
    return {"spread": spread, **target_dynamics(item.target, item.tr)}


def subject_matrix(values: np.ndarray, subjects: Sequence[str]) -> np.ndarray:
    """Average ``(n_scans, R)`` rows within subject -> ``(n_subjects, R)``."""
    subjects = np.asarray(subjects)
    return np.stack([values[subjects == s].mean(axis=0) for s in sorted(set(subjects))])


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-10, (a * b).sum(axis=-1) / den, np.nan)


def spearman_rows(x: np.ndarray, y: np.ndarray, control: Optional[np.ndarray] = None,
                  ) -> np.ndarray:
    """Spearman correlation of each row of ``x`` (``(n, R)`` or ``(R,)``) with ``y``.

    With ``control`` this is the rank-partial correlation: ranks of ``x`` and
    ``y`` are residualised on the ranks of ``control`` before correlating.
    """
    rx = rankdata(np.atleast_2d(x), axis=1)
    ry = rankdata(np.atleast_2d(y), axis=1)
    if control is not None:
        rc = rankdata(control)
        rc = rc - rc.mean()
        residual = lambda r: (r - r.mean(axis=1, keepdims=True)
                              - np.outer((r - r.mean(axis=1, keepdims=True)) @ rc, rc) / (rc @ rc))
        rx, ry = residual(rx), residual(ry)
    return _row_corr(rx, ry)


def association(spread: np.ndarray, covariate: np.ndarray, *,
                control: Optional[np.ndarray] = None, n_boot: int = 10000,
                n_perm: int = 20000, seed: int = 0) -> Dict[str, Any]:
    """Across-component Spearman between mean spread and a component covariate.

    ``spread`` is ``(n_subjects, R)``. The interval resamples subjects; the
    one-sided p-value (alternative: positive association) permutes the
    component labels of the covariate.
    """
    rng = np.random.default_rng(seed)
    n = spread.shape[0]
    estimate = float(spearman_rows(spread.mean(axis=0), covariate, control)[0])
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = spearman_rows(spread[draws].mean(axis=1), covariate, control)
    low, high = np.nanquantile(boot, [0.025, 0.975])
    out = {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
           "n_subjects": int(n), "n_components": int(covariate.size)}
    if control is None:
        permuted = rng.permuted(np.tile(covariate, (n_perm, 1)), axis=1)
        null = _row_corr(rankdata(permuted, axis=1),
                         rankdata(spread.mean(axis=0))[None, :])
        out["permutation_p"] = float((1 + np.count_nonzero(null >= estimate)) / (n_perm + 1))
    return out


def group_contrast(spread: np.ndarray, in_group: np.ndarray, *, n_boot: int = 10000,
                   n_perm: int = 20000, seed: int = 0) -> Dict[str, Any]:
    """Mean spread inside minus outside a component group (``in_group`` boolean ``(R,)``)."""
    rng = np.random.default_rng(seed)
    n = spread.shape[0]
    per_subject = spread[:, in_group].mean(axis=1) - spread[:, ~in_group].mean(axis=1)
    boot = per_subject[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    component_mean = spread.mean(axis=0)
    masks = rng.permuted(np.tile(in_group, (n_perm, 1)), axis=1)
    k = in_group.sum()
    inside = (masks * component_mean).sum(axis=1) / k
    outside = (~masks * component_mean).sum(axis=1) / (in_group.size - k)
    estimate = float(per_subject.mean())
    return {"group_mean": float(spread[:, in_group].mean()),
            "other_mean": float(spread[:, ~in_group].mean()),
            "difference": estimate, "ci_low": float(low), "ci_high": float(high),
            "label_permutation_p": float((1 + np.count_nonzero(inside - outside >= estimate))
                                         / (n_perm + 1)),
            "n_group": int(k), "n_other": int(in_group.size - k), "n_subjects": int(n)}


def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values."""
    adjusted, running = {}, 0.0
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def analyse(items: Sequence[ScanTrajectories], m: Optional[int] = None, *,
            n_boot: int = 10000, n_perm: int = 20000, seed: int = 0) -> Dict[str, Any]:
    """Full structure analysis on the 61 neural DiFuMo-64 components."""
    groups = component_groups(items[0].target.shape[1])
    neural = np.sort(np.concatenate([groups["cortical"], groups["deep_gray_cerebellar"]]))
    in_deep = np.isin(neural, groups["deep_gray_cerebellar"])
    stats = [scan_component_stats(item, m) for item in items]
    subjects = [item.subject for item in items]
    per_subject = {key: subject_matrix(np.stack([s[key] for s in stats]), subjects)[:, neural]
                   for key in stats[0]}
    spread = per_subject["spread"]
    cov = {key: value.mean(axis=0) for key, value in per_subject.items() if key != "spread"}
    kw = dict(n_boot=n_boot, n_perm=n_perm)

    primary = {"spread_vs_roughness": association(spread, cov["roughness"], seed=seed, **kw),
               "spread_vs_band_power": association(spread, cov["band_power"], seed=seed + 1, **kw)}
    holm = holm_adjust({k: v["permutation_p"] for k, v in primary.items()})
    for offset, (name, key) in enumerate(zip(PRIMARY, ("roughness", "band_power"))):
        primary[name]["holm_p"] = holm[name]
        primary[name]["partial_target_sd"] = association(
            spread, cov[key], control=cov["temporal_sd"], seed=seed + 10 + offset, **kw)
    return {
        "n_scans": len(items), "n_subjects": int(spread.shape[0]), "n_samples": m,
        "n_components": {"cortical": int((~in_deep).sum()),
                         "deep_gray_cerebellar": int(in_deep.sum()),
                         "non_neural_excluded": int(groups["non_neural"].size)},
        "group_contrast": group_contrast(spread, in_deep, seed=seed + 2, **kw),
        "primary": primary,
        "components": {"index": neural.tolist(),
                       "label": [DIFUMO_64_LABELS[i] for i in neural],
                       "deep_gray_cerebellar": in_deep.tolist(),
                       "spread": spread.mean(axis=0).tolist(),
                       **{key: value.tolist() for key, value in cov.items()}},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-dir", type=str, nargs="+", required=True,
                   help="Test trajectory cache(s) (all folds).")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Use the first M cached trajectories of every scan.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = analyse(load_scans(args.input_dir), args.n_samples, n_boot=args.n_boot,
                     n_perm=args.n_perm, seed=args.seed)
    g = result["group_contrast"]
    print(f"{result['n_scans']} scans, {result['n_subjects']} subjects, M={args.n_samples}")
    print(f"spread deep-gray/cerebellar ({g['n_group']}) {g['group_mean']:.3f} vs cortical "
          f"({g['n_other']}) {g['other_mean']:.3f}; difference {g['difference']:+.3f} "
          f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}], label permutation p="
          f"{g['label_permutation_p']:.4g}")
    for name in PRIMARY:
        e, partial = result["primary"][name], result["primary"][name]["partial_target_sd"]
        print(f"{name:<22} rho={e['estimate']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] "
              f"permutation p={e['permutation_p']:.4g} (Holm {e['holm_p']:.4g}); "
              f"partial | target SD {partial['estimate']:+.3f} "
              f"[{partial['ci_low']:+.3f}, {partial['ci_high']:+.3f}]")
    if args.output:
        save_json(result, args.output)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
