#!/usr/bin/env python
"""Plot bar charts from one or more BOLDFlow ``results.json`` files.

Useful for the paper's ablation, context-length, and parcellation plots.
Each figure is a grouped bar chart of mean +/- std for one metric across runs.

Usage
-----
    python scripts/make_figures.py --metric pearson_r \\
        --results outputs/run1/results.json outputs/run2/results.json \\
        --labels REVE-only +MSS \\
        --output figures/ablation.pdf

    # context-length sweep
    python scripts/make_figures.py --metric pearson_r \\
        --results outputs/ctx16/results.json outputs/ctx24/results.json outputs/ctx32/results.json \\
        --labels 16s 24s 32s \\
        --output figures/context.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_metric(path: Path, metric: str) -> Tuple[float, float]:
    """Read ``mean_test_<metric>`` and a std (per-fold or per-seed)."""
    data = json.loads(path.read_text())
    mean_key = f"mean_test_{metric}"
    if mean_key not in data:
        raise KeyError(f"{path}: missing {mean_key} (have {list(data)[:8]}...)")
    mean = float(data[mean_key])
    # Prefer the per-seed std if a multi-seed run; else the per-fold std.
    for k in (f"std_test_{metric}_across_seeds", f"std_test_{metric}"):
        if k in data:
            return mean, float(data[k])
    return mean, 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bar chart from BOLDFlow results.json files.")
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="One or more results.json paths.")
    p.add_argument("--labels", type=str, nargs="+", default=None,
                   help="Labels for the bars (default: file names).")
    p.add_argument("--metric", type=str, default="pearson_r",
                   choices=["mse", "mae", "r2", "pearson_r", "spearman_r", "fc_correlation"])
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output", type=str, default="figure.pdf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    paths = [Path(p) for p in args.results]
    labels = args.labels or [p.parent.name for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f"--labels has {len(labels)} entries, --results has {len(paths)}")

    means: List[float] = []
    stds: List[float] = []
    for p in paths:
        m, s = load_metric(p, args.metric)
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(labels)), 4))
    xs = list(range(len(labels)))
    ax.bar(xs, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title or args.metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, m, s in zip(labels, means, stds, strict=True):
        print(f"  {label:>16s}: {m:.4f} +/- {s:.4f}")


if __name__ == "__main__":
    main()
