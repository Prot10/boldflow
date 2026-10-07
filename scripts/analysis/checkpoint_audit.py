#!/usr/bin/env python
"""Audit validation-based checkpoint selection (paper Appendix A).

Reads the ``results.json`` written by ``scripts/train.py`` and reports, per
fold, the epoch whose checkpoint was selected (best validation Pearson r) out
of the epoch cap, the last epoch that was trained, and the change in
validation MSE between the selected epoch and the last trained epoch. A
positive change means that training past the selected epoch worsened the
validation error. The mean change is taken over the folds that trained past
their selected epoch.

Examples
--------
    python scripts/analysis/checkpoint_audit.py \\
        --results outputs/boldflow_neurobolt/results.json \\
        --config configs/neurobolt.yaml \\
        --output outputs/analysis/checkpoint_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.utils import load_yaml_config, save_json


def fold_records(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a single-seed or multi-seed ``results.json`` into per-fold records."""
    runs = results.get("per_seed", [results])
    return [dict(fold, seed=run.get("seed")) for run in runs
            for fold in run.get("fold_results", [])]


def audit_fold(fold: Dict[str, Any]) -> Dict[str, Any]:
    """Selected epoch, last trained epoch and late validation-MSE change for one fold."""
    history = {int(h["epoch"]): h for h in fold["history"]}
    selected, last = int(fold["best_epoch"]), max(history)
    selected_mse, last_mse = history[selected]["val_loss"], history[last]["val_loss"]
    return {
        "fold": int(fold["fold_idx"]), "seed": fold.get("seed"),
        "selected_epoch": selected, "last_epoch": last,
        "selected_val_mse": float(selected_mse), "last_val_mse": float(last_mse),
        "late_val_mse_change": float(last_mse - selected_mse),
        "trained_past_selection": last > selected,
    }


def audit(results: Dict[str, Any], epoch_cap: Optional[int] = None) -> Dict[str, Any]:
    """Per-fold audit and its summary over folds."""
    folds = [audit_fold(f) for f in fold_records(results)]
    if not folds:
        raise ValueError("no fold_results with a training history in the results file")
    late = [f["late_val_mse_change"] for f in folds if f["trained_past_selection"]]
    return {
        "epoch_cap": epoch_cap,
        "selected_epochs": [f["selected_epoch"] for f in folds],
        "last_epochs": [f["last_epoch"] for f in folds],
        "n_folds": len(folds),
        "n_folds_trained_past_selection": len(late),
        "mean_late_val_mse_change": float(np.mean(late)) if late else None,
        "n_folds_late_val_mse_worse": int(sum(change > 0 for change in late)),
        "folds": folds,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=str, required=True,
                   help="results.json written by scripts/train.py.")
    p.add_argument("--config", type=str, default=None,
                   help="Training config; provides the epoch cap (training.epochs).")
    p.add_argument("--epoch-cap", type=int, default=None,
                   help="Epoch cap, if it was overridden on the training command line.")
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.results, "r") as f:
        results = json.load(f)
    cap = args.epoch_cap
    if cap is None and args.config:
        cap = int(load_yaml_config(args.config)["training"]["epochs"])
    summary = audit(results, cap)

    cap_text = f" of the {cap}-epoch cap" if cap else ""
    print("selected epochs: " + "/".join(str(e) for e in summary["selected_epochs"]) + cap_text)
    for f in summary["folds"]:
        seed = f" seed {f['seed']}" if len(results.get("per_seed", [])) > 1 else ""
        print(f"  fold {f['fold']}{seed}: selected epoch {f['selected_epoch']}, "
              f"last epoch {f['last_epoch']}, val MSE {f['selected_val_mse']:.4f} -> "
              f"{f['last_val_mse']:.4f} ({f['late_val_mse_change']:+.4f})")
    if summary["mean_late_val_mse_change"] is not None:
        print(f"later training changed validation MSE by "
              f"{summary['mean_late_val_mse_change']:+.4f} on average over "
              f"{summary['n_folds_trained_past_selection']} folds "
              f"(worse in {summary['n_folds_late_val_mse_worse']})")
    if args.output:
        save_json(summary, args.output)
        print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
