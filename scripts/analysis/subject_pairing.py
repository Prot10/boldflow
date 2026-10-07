#!/usr/bin/env python
"""Subject pairing beyond population structure (Appendix D).

For a held-out target scan ``i`` the *matched* score compares the FC of the
trajectory generated from the EEG of scan ``i`` with the measured FC of scan
``i``. The *wrong-subject* score compares the measured FC of scan ``i`` with
the FC of the trajectory generated from the EEG of every scan of a different
held-out subject in the same fold (same checkpoint), and averages.

Both scores are reported on raw FC and on *population-residual* FC: Fisher-z
FC minus the fold's population template. The template of fold ``k`` is the
Fisher-z mean of the measured FC of the cached scans of all subjects not held
out in fold ``k``; it requires the caches of every fold.

FC is computed within scan on the cortical component mask. Scores are computed
per sampled trajectory and averaged over the first ``--n-trajectories``
trajectories. Subjects are the bootstrap unit.

Examples
--------
    python scripts/analysis/subject_pairing.py \\
        --trajectories outputs/trajectories \\
        --output outputs/analysis/subject_pairing.json

    # interaction with the generator retrained on constant (zero) EEG
    python scripts/analysis/subject_pairing.py \\
        --trajectories outputs/trajectories \\
        --constant-input-dir outputs/trajectories_constant_input \\
        --output outputs/analysis/subject_pairing.json
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

SCORES = ("matched", "wrong_subject", "matched_minus_wrong")
FIELDS = SCORES + tuple(f"residual_{s}" for s in SCORES)


def pairing_scores(
    items: Sequence[ScanTrajectories],
    templates: Optional[Dict[int, np.ndarray]] = None,
    n_trajectories: int = 1,
) -> List[Dict[str, Any]]:
    """Scan-level matched / wrong-subject FC similarity, raw and residual.

    Each row belongs to a target scan: its measured FC is compared with the FC
    generated from its own EEG (matched) and from the EEG of the other
    subjects' scans in the same fold (wrong subject).
    """
    templates = population_templates(items) if templates is None else templates
    comps = fc_components(items[0].target.shape[1])
    rows = []
    for fold in sorted({it.fold for it in items}):
        scans = [it for it in items if it.fold == fold]
        template = templates[fold]
        fold_rows = [{"scan": it.scan, "subject": it.subject, "fold": fold} for it in scans]
        for prefix, transform in (("", lambda fc: fc),
                                  ("residual_", lambda fc: fisher_z(fc) - template)):
            targets = [transform(fc_matrix(it.target, comps)) for it in scans]
            generated = [[transform(fc_matrix(traj, comps))
                          for traj in it.samples[:n_trajectories]] for it in scans]
            for i, (item, row) in enumerate(zip(scans, fold_rows)):
                # sim[j]: FC generated from scan j against the measured FC of scan i
                sim = [nanmean([fc_similarity(g, targets[i]) for g in gen]) for gen in generated]
                wrong = nanmean([s for s, other in zip(sim, scans)
                                 if other.subject != item.subject])
                row[prefix + "matched"] = sim[i]
                row[prefix + "wrong_subject"] = wrong
                row[prefix + "matched_minus_wrong"] = sim[i] - wrong
        rows.extend(fold_rows)
    return rows


def summarize(rows: Sequence[Dict[str, Any]], n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Subject-bootstrap summary of every score plus the per-fold contrasts."""
    subjects = [r["subject"] for r in rows]
    out: Dict[str, Any] = {
        f: subject_bootstrap([r[f] for r in rows], subjects, n_boot=n_boot, seed=seed)
        for f in FIELDS
    }
    out["per_fold"] = [
        {"fold": fold,
         **{f: nanmean([r[f] for r in rows if r["fold"] == fold])
            for f in ("matched_minus_wrong", "residual_matched_minus_wrong")}}
        for fold in sorted({r["fold"] for r in rows})
    ]
    for f in ("matched_minus_wrong", "residual_matched_minus_wrong"):
        out[f"n_folds_positive_{f}"] = int(sum(p[f] > 0 for p in out["per_fold"]))
    return out


def interaction(
    model_rows: Sequence[Dict[str, Any]],
    constant_rows: Sequence[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired model-minus-constant-input difference of the matched-minus-wrong scores."""
    constant = {r["scan"]: r for r in constant_rows}
    if set(constant) != {r["scan"] for r in model_rows}:
        raise ValueError("model and constant-input caches do not cover the same scans")
    subjects = [r["subject"] for r in model_rows]
    return {
        f: subject_bootstrap([r[f] - constant[r["scan"]][f] for r in model_rows],
                             subjects, n_boot=n_boot, seed=seed)
        for f in ("matched_minus_wrong", "residual_matched_minus_wrong")
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Cached trajectories of the model (all folds).")
    p.add_argument("--constant-input-dir", type=str, nargs="+", default=None,
                   help="Cached trajectories of the constant-input generator.")
    p.add_argument("--n-trajectories", type=int, default=1,
                   help="Sampled trajectories whose scores are averaged per scan.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def _print(label: str, stats: Dict[str, Any], fields: Sequence[str]) -> None:
    print(label)
    for f in fields:
        s = stats[f]
        print(f"  {f:<30s} {s['mean']:+.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectories)
    templates = population_templates(items)
    rows = pairing_scores(items, templates, args.n_trajectories)
    result: Dict[str, Any] = {
        "n_scans": len(rows), "n_subjects": len({r["subject"] for r in rows}),
        "n_trajectories": args.n_trajectories,
        "model": summarize(rows, args.n_boot, args.seed), "scans": {"model": rows},
    }
    _print(f"Model ({len(rows)} scans, {result['n_subjects']} subjects)",
           result["model"], FIELDS)
    for f in ("matched_minus_wrong", "residual_matched_minus_wrong"):
        n_pos = result["model"][f"n_folds_positive_{f}"]
        print(f"  {f}: positive in {n_pos}/{len(result['model']['per_fold'])} folds")

    if args.constant_input_dir:
        const_rows = pairing_scores(load_scans(args.constant_input_dir), templates,
                                    args.n_trajectories)
        result["constant_input"] = summarize(const_rows, args.n_boot, args.seed)
        result["model_minus_constant_input"] = interaction(rows, const_rows,
                                                           args.n_boot, args.seed)
        result["scans"]["constant_input"] = const_rows
        _print("Constant-input generator", result["constant_input"], FIELDS)
        _print("Model minus constant-input generator", result["model_minus_constant_input"],
               list(result["model_minus_constant_input"]))

    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
