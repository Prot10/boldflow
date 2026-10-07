#!/usr/bin/env python
"""Within-scan temporal specificity of dynamic FC (Appendix D).

Static FC is unchanged by a common reordering of time points, so temporal
alignment is tested on dynamic FC: FC matrices of sliding windows (120 s long,
30 s apart). For each scan, the dynamic-FC similarity is the correlation
between the generated and the measured FC of a window, averaged over windows.

* aligned vs displaced: the generated FC of window ``w`` is compared with the
  measured FC of the same window, and with the measured FC of the window that
  starts 180 s later in the same scan (wrapping over the valid window starts,
  so every measured window stays contiguous). Subject identity is held fixed.
* correct vs different subject: the generated dynamic FC is compared with the
  measured dynamic FC of the same scan, and with that of every scan of a
  different held-out subject in the same fold (windows matched by index,
  truncated to the shorter scan), averaged over those scans.

Both contrasts are reported on raw FC and on population-residual FC (Fisher-z
FC minus the fold's training-population static template, built from the
measured FC of the cached scans whose subject is not held out in that fold).
Scores are computed per sampled trajectory and averaged over the first
``--n-trajectories`` trajectories. Subjects are the bootstrap unit.

Examples
--------
    python scripts/analysis/dynamic_fc_alignment.py \\
        --trajectory-dir outputs/trajectories \\
        --output outputs/analysis/dynamic_fc_alignment.json

    # 60-s windows as a sensitivity analysis
    python scripts/analysis/dynamic_fc_alignment.py ... --window-seconds 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import (ScanTrajectories, fc_components, fc_matrix, fc_similarity,
                               fisher_z, load_scans, nanmean,
                               population_templates, subject_bootstrap)
from boldflow.utils import save_json

SCORES = ("aligned", "displaced", "aligned_minus_displaced",
          "wrong_subject", "matched_minus_wrong")
FIELDS = SCORES + tuple(f"residual_{s}" for s in SCORES)


def window_starts(n_time: int, window: int, step: int) -> np.ndarray:
    """First frame of every full window of ``window`` frames, ``step`` apart."""
    return np.arange(0, n_time - window + 1, step, dtype=int)


def displaced_starts(starts: np.ndarray, n_time: int, window: int, shift: int) -> np.ndarray:
    """Window starts moved by ``shift`` frames, wrapping over the valid starts."""
    n_valid = n_time - window + 1
    if not 0 < shift < n_valid:
        raise ValueError(f"shift must be in [1, {n_valid - 1}] frames, got {shift}")
    return (np.asarray(starts) + shift) % n_valid


def dynamic_fc(series: np.ndarray, starts: Sequence[int], window: int,
               components: Optional[Sequence[int]] = None) -> np.ndarray:
    """``(W, C, C)`` FC matrices of the windows of a ``(L, R)`` series."""
    return np.stack([fc_matrix(series[s:s + window], components) for s in starts])


def dynamic_similarity(pred: np.ndarray, target: np.ndarray,
                       template_z: Optional[np.ndarray] = None) -> float:
    """Mean over windows of the FC similarity between two dynamic-FC stacks.

    Windows are matched by index and truncated to the shorter stack; fewer
    than two common windows give NaN. With ``template_z`` the Fisher-z
    template is subtracted from both sides first.
    """
    n = min(len(pred), len(target))
    if n < 2:
        return float("nan")
    if template_z is not None:
        pred, target = fisher_z(pred) - template_z, fisher_z(target) - template_z
    return nanmean([fc_similarity(pred[w], target[w]) for w in range(n)])


def dynamic_scores(
    items: Sequence[ScanTrajectories],
    *,
    window_seconds: float = 120.0,
    step_seconds: float = 30.0,
    shift_seconds: float = 180.0,
    n_trajectories: int = 1,
    templates: Optional[Dict[int, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """Scan-level dynamic-FC similarities and contrasts.

    ``templates`` maps fold to a Fisher-z template; ``None`` builds them from
    the cached scans and ``{}`` skips the residual scores.
    """
    templates = population_templates(items) if templates is None else templates
    comps = fc_components(items[0].target.shape[1])
    dyn = []  # per scan: generated (n, W, C, C), measured aligned / displaced (W, C, C)
    for item in items:
        window, step, shift = (int(round(s / item.tr))
                               for s in (window_seconds, step_seconds, shift_seconds))
        n_time = item.target.shape[0]
        starts = window_starts(n_time, window, step)
        if len(starts) < 2 or shift >= n_time - window + 1:
            dyn.append(None)  # scan too short for this window / displacement
            continue
        dyn.append((
            [dynamic_fc(traj, starts, window, comps) for traj in item.samples[:n_trajectories]],
            dynamic_fc(item.target, starts, window, comps),
            dynamic_fc(item.target, displaced_starts(starts, n_time, window, shift),
                       window, comps),
        ))

    rows = []
    for i, item in enumerate(items):
        row = {"scan": item.scan, "subject": item.subject, "fold": item.fold}
        if dyn[i] is None:
            rows.append({**row, **{f: float("nan") for f in FIELDS}})
            continue
        preds, aligned_fc, displaced_fc = dyn[i]
        others = [dyn[j][1] for j, other in enumerate(items) if dyn[j] is not None
                  and other.fold == item.fold and other.subject != item.subject]
        modes = [("", None)]
        if item.fold in templates:
            modes.append(("residual_", templates[item.fold]))
        for prefix, template in modes:
            def score(target_fc: np.ndarray) -> float:
                return nanmean([dynamic_similarity(p, target_fc, template) for p in preds])
            aligned, displaced = score(aligned_fc), score(displaced_fc)
            wrong = nanmean([score(o) for o in others])
            row.update({
                prefix + "aligned": aligned, prefix + "displaced": displaced,
                prefix + "aligned_minus_displaced": aligned - displaced,
                prefix + "wrong_subject": wrong,
                prefix + "matched_minus_wrong": aligned - wrong,
            })
        rows.append(row)
    return rows


def summarize(rows: Sequence[Dict[str, Any]], n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Subject-bootstrap mean and 95% interval of every scan-level score."""
    subjects = [r["subject"] for r in rows]
    return {f: subject_bootstrap([r[f] for r in rows], subjects, n_boot=n_boot, seed=seed)
            for f in FIELDS if any(np.isfinite(r.get(f, np.nan)) for r in rows)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory-dir", type=str, nargs="+", required=True)
    p.add_argument("--window-seconds", type=float, default=120.0)
    p.add_argument("--step-seconds", type=float, default=30.0)
    p.add_argument("--shift-seconds", type=float, default=180.0,
                   help="Within-scan displacement of the measured windows.")
    p.add_argument("--n-trajectories", type=int, default=1,
                   help="Sampled trajectories whose scores are averaged per scan.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectory_dir)
    rows = dynamic_scores(
        items, window_seconds=args.window_seconds, step_seconds=args.step_seconds,
        shift_seconds=args.shift_seconds, n_trajectories=args.n_trajectories,
    )
    summary = summarize(rows, args.n_boot, args.seed)
    n_used = sum(np.isfinite(r["aligned"]) for r in rows)
    result = {
        "window_seconds": args.window_seconds, "step_seconds": args.step_seconds,
        "shift_seconds": args.shift_seconds, "n_trajectories": args.n_trajectories,
        "n_scans": len(rows), "n_scans_used": int(n_used),
        "summary": summary, "scans": rows,
    }
    print(f"Dynamic FC, {args.window_seconds:g}-s windows every {args.step_seconds:g} s, "
          f"{args.shift_seconds:g}-s displacement ({n_used}/{len(rows)} scans long enough)")
    for f, s in summary.items():
        print(f"  {f:<34s} {s['mean']:+.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
              f"  (n={s['n_subjects']} subjects)")
    save_json(result, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
