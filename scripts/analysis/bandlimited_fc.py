#!/usr/bin/env python
"""Filtering control for FC Corr (paper Appendix D, "Filtering control").

The measured targets are low-pass filtered at 0.15 Hz during preprocessing,
the generated series are not. This script applies the same filter family
(5th-order Butterworth, zero-phase) to the cached generated trajectories and
recomputes FC Corr against the measured FC on the cortical component mask for
two readouts:

* single draw   - FC Corr of each sampled trajectory, averaged over draws;
* ensemble mean - FC Corr of the pointwise mean of the trajectories.

It reports the filtered-minus-unfiltered change per readout (scan values
averaged within subject, percentile bootstrap over subjects) and how much
generated power the filter removes: the Welch power fraction above the cutoff
and the realised drop in temporal variance.
``--bandpass LOW HIGH`` swaps the low-pass for a band-pass, which additionally
removes slow content.

Examples
--------
    python scripts/analysis/bandlimited_fc.py outputs/trajectories \\
        --output outputs/analysis/bandlimited_fc.json

    python scripts/analysis/bandlimited_fc.py outputs/trajectories --bandpass 0.01 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.signal import butter, filtfilt, welch

from boldflow.analysis import (ScanTrajectories, fc_components, fc_matrix, fc_similarity,
                               load_scans, subject_bootstrap)
from boldflow.utils import save_json

CUTOFF_HZ = 0.15
FILTER_ORDER = 5
READOUTS = ("single_draw", "ensemble_mean")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def zero_phase_filter(series: np.ndarray, tr: float, *, high: float = CUTOFF_HZ,
                      low: Optional[float] = None, order: int = FILTER_ORDER) -> np.ndarray:
    """Butterworth low-pass (or band-pass if ``low``) along the time axis of ``(..., L, R)``."""
    nyquist = 0.5 / tr
    if low is None:
        b, a = butter(N=order, Wn=high / nyquist, btype="low")
    else:
        b, a = butter(N=order, Wn=[low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(series, dtype=np.float64), axis=-2)


def power_fraction_above(series: np.ndarray, tr: float, cutoff: float = CUTOFF_HZ) -> float:
    """Fraction of Welch power at or above ``cutoff`` (PSD averaged over series)."""
    x = np.moveaxis(np.asarray(series, dtype=np.float64), -2, -1)
    nperseg = min(128, x.shape[-1])
    freqs, power = welch(x, fs=1.0 / tr, nperseg=nperseg, noverlap=nperseg // 2,
                         detrend="constant", scaling="density", axis=-1)
    power = power.reshape(-1, power.shape[-1]).mean(axis=0)
    keep = freqs >= cutoff
    return float(_trapezoid(power[keep], freqs[keep]) / _trapezoid(power, freqs))


def _single_draw_fc(samples: np.ndarray, fc_target: np.ndarray, comps: np.ndarray) -> float:
    return float(np.nanmean([fc_similarity(fc_matrix(s, comps), fc_target) for s in samples]))


def scan_filtering_control(scan: ScanTrajectories, *, high: float = CUTOFF_HZ,
                           low: Optional[float] = None,
                           n_draws: Optional[int] = None) -> Dict[str, float]:
    """FC Corr of one scan before/after filtering, for both readouts."""
    samples = np.asarray(scan.samples[:n_draws], dtype=np.float64)
    target = np.asarray(scan.target, dtype=np.float64)
    comps = fc_components(target.shape[1])
    filtered = zero_phase_filter(samples, scan.tr, high=high, low=low)
    fc_target = fc_matrix(target, comps)

    row: Dict[str, float] = {}
    for name, gen in (("raw", samples), ("filtered", filtered)):
        row[f"single_draw_{name}"] = _single_draw_fc(gen, fc_target, comps)
        row[f"ensemble_mean_{name}"] = fc_similarity(fc_matrix(gen.mean(axis=0), comps), fc_target)
    for readout in READOUTS:
        row[f"{readout}_delta"] = row[f"{readout}_filtered"] - row[f"{readout}_raw"]
    row["power_above_cutoff"] = power_fraction_above(samples, scan.tr, high)
    row["power_above_cutoff_measured"] = power_fraction_above(target, scan.tr, high)
    row["variance_removed"] = float(1.0 - filtered.var(axis=1).sum() / samples.var(axis=1).sum())
    return row


def filtering_control(scans: Sequence[ScanTrajectories], *, high: float = CUTOFF_HZ,
                      low: Optional[float] = None, n_draws: Optional[int] = None,
                      n_boot: int = 10000, seed: int = 0) -> Dict[str, object]:
    """Per-scan rows and subject-bootstrap summaries of every column."""
    rows = [{"scan": s.scan, "subject": s.subject, "fold": s.fold,
             **scan_filtering_control(s, high=high, low=low, n_draws=n_draws)} for s in scans]
    subjects = [r["subject"] for r in rows]
    keys = [k for k in rows[0] if k not in ("scan", "subject", "fold")]
    summary = {k: subject_bootstrap([r[k] for r in rows], subjects, n_boot=n_boot, seed=seed)
               for k in keys}
    return {"filter": {"type": "lowpass" if low is None else "bandpass", "low_hz": low,
                       "high_hz": high, "order": FILTER_ORDER, "zero_phase": True},
            "n_scans": len(rows), "summary": summary, "per_scan": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cache_dirs", nargs="+",
                   help="Directories written by sample_trajectories.py.")
    p.add_argument("--cutoff", type=float, default=CUTOFF_HZ, help="Low-pass cutoff in Hz.")
    p.add_argument("--bandpass", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                   help="Use a band-pass between LOW and HIGH Hz instead.")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Use only the first M cached trajectories per scan.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return p.parse_args()


def _cell(s: Dict[str, float], signed: bool = False) -> str:
    fmt = "+.3f" if signed else ".3f"
    return f"{s['mean']:{fmt}} [{s['ci_low']:{fmt}}, {s['ci_high']:{fmt}}]"


def main() -> None:
    args = parse_args()
    low, high = args.bandpass if args.bandpass else (None, args.cutoff)
    result = filtering_control(load_scans(args.cache_dirs), high=high, low=low,
                               n_draws=args.n_samples, n_boot=args.n_boot, seed=args.seed)
    s = result["summary"]
    band = f"low-pass {high} Hz" if low is None else f"band-pass {low}-{high} Hz"
    print(f"{band}; {result['n_scans']} scans, {s['single_draw_raw']['n_subjects']} subjects; "
          "subject-bootstrap 95% intervals")
    print(f"{'Readout':<16}{'unfiltered':>12}{'filtered':>12}{'change':>34}")
    for r in READOUTS:
        print(f"{r:<16}{s[r + '_raw']['mean']:>12.3f}{s[r + '_filtered']['mean']:>12.3f}"
              f"{_cell(s[r + '_delta'], True):>34}")
    print(f"generated power above {high} Hz: {100 * s['power_above_cutoff']['mean']:.1f}% "
          f"(measured {100 * s['power_above_cutoff_measured']['mean']:.1f}%); "
          f"generated variance removed by the filter: {100 * s['variance_removed']['mean']:.1f}%")
    if args.output:
        save_json(result, args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
