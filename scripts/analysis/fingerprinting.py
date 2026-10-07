#!/usr/bin/env python
"""Subject identification from predicted connectomes (paper Table 12).

Each held-out scan is split in time into two non-overlapping halves.
``--gap-seconds`` of frames centred on the midpoint are left out of both halves
(default 40 s, which covers the 32 s EEG context plus the overlap of the
predicted blocks), so that no EEG sample contributes to predictions in both
halves; the same frames are left out of the measured series. For a subject
with two scans the first halves of both scans are concatenated, and likewise
the second halves. One FC matrix is computed per subject and half.

Every subject's first-half FC is matched against the pool of all subjects'
second-half FC by the Pearson correlation of the upper-triangular entries; the
identification is correct when the best match is the same subject. The
direction is then reversed and the two accuracies are averaged. Chance is
``1 / N`` for a pool of ``N`` subjects. The analysis is run on predicted
halves (predicted-to-predicted, one sampled trajectory per scan) and on
measured halves (measured-to-measured).

Reads the caches written by ``sample_trajectories.py``.

Examples
--------
    python scripts/analysis/fingerprinting.py \\
        --trajectories outputs/trajectories \\
        --output outputs/analysis/fingerprinting.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import (ScanTrajectories, fc_components, fc_matrix, load_scans,
                               upper_triangle)
from boldflow.utils import save_json


def split_halves(series: np.ndarray, gap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """First and second half (in time) of a ``(L, R)`` series.

    ``gap`` frames centred on the midpoint are dropped, split evenly between
    the two halves.
    """
    half = series.shape[0] // 2
    before = gap // 2
    return series[:half - before], series[half + gap - before:]


def subject_halves(series_by_scan: Sequence[np.ndarray],
                   gap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Two non-overlapping segments for one subject.

    Every scan is split in two; the first halves and the second halves are
    concatenated across the subject's scans.
    """
    firsts, seconds = zip(*(split_halves(s, gap) for s in series_by_scan))
    return np.concatenate(firsts, axis=0), np.concatenate(seconds, axis=0)


def identification_accuracy(first: np.ndarray, second: np.ndarray) -> Dict[str, Any]:
    """Top-1 identification between two ``(N, E)`` sets of FC edge vectors.

    Row ``i`` of both arrays belongs to subject ``i``. Similarity is the
    Pearson correlation between edge vectors; the reported accuracy is the
    mean of the two query directions.
    """
    n = first.shape[0]
    similarity = np.corrcoef(first, second)[:n, n:]
    forward = float((similarity.argmax(axis=1) == np.arange(n)).mean())
    backward = float((similarity.argmax(axis=0) == np.arange(n)).mean())
    accuracy = 0.5 * (forward + backward)
    return {"accuracy": accuracy, "first_to_second": forward, "second_to_first": backward,
            "n_subjects": int(n), "chance": 1.0 / n, "chance_ratio": accuracy * n}


def fingerprint(series_by_subject: Dict[str, List[np.ndarray]], *,
                components: Optional[Sequence[int]] = None,
                gap: int = 0) -> Dict[str, Any]:
    """Identification accuracy for a ``{subject: [series (L, R), ...]}`` mapping."""
    first, second = [], []
    for subject in sorted(series_by_subject):
        a, b = subject_halves(series_by_subject[subject], gap)
        first.append(upper_triangle(fc_matrix(a, components)))
        second.append(upper_triangle(fc_matrix(b, components)))
    first, second = np.stack(first), np.stack(second)
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("non-finite FC entries: a half has a constant component")
    return identification_accuracy(first, second)


def summarize(items: Sequence[ScanTrajectories], *,
              components: Optional[Sequence[int]] = None,
              gap: int = 0) -> Dict[str, Any]:
    """Predicted-to-predicted and measured-to-measured identification."""
    predicted: Dict[str, List[np.ndarray]] = {}
    measured: Dict[str, List[np.ndarray]] = {}
    for item in sorted(items, key=lambda it: it.scan):
        predicted.setdefault(item.subject, []).append(item.samples[0])
        measured.setdefault(item.subject, []).append(item.target)
    kwargs = dict(components=components, gap=gap)
    return {"predicted": fingerprint(predicted, **kwargs),
            "measured": fingerprint(measured, **kwargs)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Cache directories written by sample_trajectories.py.")
    p.add_argument("--components", choices=["all", "cortical"], default="all",
                   help="Component set of the FC matrices.")
    p.add_argument("--gap-seconds", type=float, default=40.0,
                   help="Time dropped at the midpoint so the halves share no EEG input.")
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectories)
    n_rois = items[0].target.shape[1]
    components = fc_components(n_rois) if args.components == "cortical" else None
    gap = int(round(args.gap_seconds / items[0].tr))
    result = summarize(items, components=components, gap=gap)
    result.update(components=args.components,
                  gap_seconds=args.gap_seconds, gap_frames=gap)
    for source in ("measured", "predicted"):
        r = result[source]
        print(f"{source:9s} FC halves: top-1 {100 * r['accuracy']:.1f}%  "
              f"(x{r['chance_ratio']:.1f} chance, N={r['n_subjects']})")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
