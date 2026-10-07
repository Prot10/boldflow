#!/usr/bin/env python
"""Community-structure preservation of predicted FC (paper Table 11).

For every held-out scan the predicted and the measured FC matrices are
thresholded at the same edge density (the strongest 5% of absolute
correlations), Louvain community detection (resolution ``gamma = 1``) is run on
each weighted graph, and the adjusted Rand index (ARI) between the two
partitions is computed. The predicted FC is that of one sampled trajectory
(the first cached one). The table entry is the mean ARI over held-out scans.

Reads the caches written by ``sample_trajectories.py``. Requires ``networkx``
for the Louvain step.

Examples
--------
    python scripts/analysis/community_structure.py \\
        --trajectories outputs/trajectories \\
        --output outputs/analysis/community_structure.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import ScanTrajectories, fc_components, fc_matrix, load_scans
from boldflow.utils import save_json


def strongest_edges(fc: np.ndarray, density: float) -> np.ndarray:
    """Boolean adjacency keeping the ``density`` fraction of largest ``|FC|`` edges."""
    rows, cols = np.triu_indices(fc.shape[0], k=1)
    weights = np.abs(fc[rows, cols])
    k = max(1, int(round(density * weights.size)))
    keep = np.argsort(-weights, kind="stable")[:k]
    adjacency = np.zeros(fc.shape, dtype=bool)
    adjacency[rows[keep], cols[keep]] = adjacency[cols[keep], rows[keep]] = True
    return adjacency


def louvain_labels(fc: np.ndarray, *, density: float = 0.05, gamma: float = 1.0,
                   seed: int = 0) -> np.ndarray:
    """Community label per component from Louvain on the thresholded ``|FC|`` graph.

    Components left without an edge by the threshold form singleton communities.
    """
    import networkx as nx

    adjacency = strongest_edges(fc, density)
    graph = nx.Graph()
    graph.add_nodes_from(range(fc.shape[0]))
    graph.add_weighted_edges_from(
        (int(i), int(j), float(abs(fc[i, j]))) for i, j in zip(*np.nonzero(np.triu(adjacency)))
    )
    labels = np.zeros(fc.shape[0], dtype=int)
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=gamma, seed=seed)
    for label, members in enumerate(communities):
        labels[list(members)] = label
    return labels


def adjusted_rand_index(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Adjusted Rand index between two partitions of the same items (Hubert & Arabie)."""
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(counts: np.ndarray) -> float:
        return float((counts * (counts - 1) / 2).sum())

    same_both, same_a, same_b = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = same_a * same_b / pairs(np.array([len(a)]))
    maximum = 0.5 * (same_a + same_b)
    if maximum == expected:          # both partitions trivial: identical by convention
        return 1.0
    return (same_both - expected) / (maximum - expected)


def community_ari(predicted: np.ndarray, measured: np.ndarray, *, density: float = 0.05,
                  gamma: float = 1.0, seed: int = 0,
                  components: Optional[Sequence[int]] = None) -> float:
    """ARI between Louvain partitions of the FC of two ``(L, R)`` series."""
    fc_pred, fc_true = fc_matrix(predicted, components), fc_matrix(measured, components)
    if not (np.isfinite(fc_pred).all() and np.isfinite(fc_true).all()):
        return float("nan")
    kwargs = dict(density=density, gamma=gamma, seed=seed)
    return adjusted_rand_index(louvain_labels(fc_true, **kwargs),
                               louvain_labels(fc_pred, **kwargs))


def summarize(items: Sequence[ScanTrajectories], *,
              components: Optional[Sequence[int]] = None, **kwargs) -> Dict[str, Any]:
    """Per-scan ARI and its mean / spread over held-out scans."""
    per_scan = {
        item.scan: community_ari(item.samples[0], item.target,
                                 components=components, **kwargs)
        for item in items
    }
    values = np.array([v for v in per_scan.values() if np.isfinite(v)])
    return {
        "ari_mean": float(values.mean()),
        "ari_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ari_sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "n_scans": int(len(values)),
        "per_scan": per_scan,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Cache directories written by sample_trajectories.py.")
    p.add_argument("--components", choices=["all", "cortical"], default="all",
                   help="Component set of the FC matrices.")
    p.add_argument("--density", type=float, default=0.05, help="Retained edge density.")
    p.add_argument("--gamma", type=float, default=1.0, help="Louvain resolution.")
    p.add_argument("--seed", type=int, default=0, help="Louvain seed.")
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectories)
    n_rois = items[0].target.shape[1]
    components = fc_components(n_rois) if args.components == "cortical" else None
    result = summarize(items, components=components,
                       density=args.density, gamma=args.gamma, seed=args.seed)
    result.update(density=args.density, gamma=args.gamma, components=args.components)
    print(f"ARI (predicted vs measured communities): {result['ari_mean']:.3f} "
          f"(std {result['ari_std']:.3f}, sem {result['ari_sem']:.3f}, "
          f"{result['n_scans']} scans)")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
