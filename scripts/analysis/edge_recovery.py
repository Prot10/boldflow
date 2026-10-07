#!/usr/bin/env python
"""Group-level edge recovery and per-network FC similarity (paper Figure 1).

Per-scan FC matrices on all components (predicted FC from one sampled
trajectory per scan) are transformed entrywise to Fisher-z, averaged over the
held-out scans and transformed back. On these group-average matrices the
script reports

* how many of the ``K`` strongest measured edges (largest ``|FC|``; ``K = 100``
  is about 5% of the 2016 DiFuMo-64 edges) also lie in the prediction's top
  ``K``;
* per network, the Pearson correlation between measured and predicted FC over
  the edges with at least one endpoint in the network. A network with fewer
  than two components is skipped.

Network labels are read from the atlas metadata (``--network-labels``, a CSV
with one row per component and a ``Yeo_networks7`` column, such as the DiFuMo
``labels_64_dictionary.csv``); without it the per-network part is skipped.

Reads the caches written by ``sample_trajectories.py``.

Examples
--------
    python scripts/analysis/edge_recovery.py \\
        --trajectories outputs/trajectories \\
        --network-labels labels_64_dictionary.csv \\
        --output outputs/analysis/edge_recovery.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from boldflow.analysis import (ScanTrajectories, fc_matrix, fisher_mean, load_scans,
                               upper_triangle)
from boldflow.difumo import cortical_network_indices
from boldflow.utils import save_json

NO_NETWORK = "No network found"


def group_average_fc(items: Sequence[ScanTrajectories]) -> Tuple[np.ndarray, np.ndarray]:
    """Fisher-z group-average ``(predicted, measured)`` FC over scans, all components."""
    predicted = fisher_mean([fc_matrix(item.samples[0]) for item in items])
    measured = fisher_mean([fc_matrix(item.target) for item in items])
    return predicted, measured


def top_edges(fc: np.ndarray, k: int) -> np.ndarray:
    """Upper-triangle indices of the ``k`` edges with the largest ``|FC|``."""
    return np.argsort(-np.abs(upper_triangle(fc)), kind="stable")[:k]


def edge_recovery(fc_true: np.ndarray, fc_pred: np.ndarray, k: int = 100) -> Dict[str, Any]:
    """Number of the top-``k`` measured edges that are in the prediction's top ``k``."""
    recovered = np.intersect1d(top_edges(fc_true, k), top_edges(fc_pred, k)).size
    n_edges = upper_triangle(fc_true).size
    return {"k": int(k), "recovered": int(recovered), "fraction": recovered / k,
            "edge_density": k / n_edges, "n_edges": int(n_edges)}


def network_similarity(fc_true: np.ndarray, fc_pred: np.ndarray,
                       labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per-network Pearson r over edges with at least one endpoint in the network.

    Networks with fewer than two components are skipped.
    """
    labels = np.asarray(labels)
    rows, cols = np.triu_indices(len(labels), k=1)
    true_edges, pred_edges = upper_triangle(fc_true), upper_triangle(fc_pred)
    out: Dict[str, Dict[str, Any]] = {}
    for network in sorted(set(labels) - {NO_NETWORK}):
        member = labels == network
        incident = member[rows] | member[cols]
        if member.sum() < 2 or true_edges[incident].std() < 1e-12 \
                or pred_edges[incident].std() < 1e-12:
            continue
        out[network] = {
            "r": float(np.corrcoef(true_edges[incident], pred_edges[incident])[0, 1]),
            "n_components": int(member.sum()), "n_edges": int(incident.sum()),
        }
    return out


def load_network_labels(path: str, n_rois: int, column: str = "Yeo_networks7") -> List[str]:
    """Per-component network labels from the atlas metadata CSV (component order)."""
    import pandas as pd

    table = pd.read_csv(path)
    if column not in table.columns or len(table) != n_rois:
        raise ValueError(f"{path}: expected {n_rois} rows and a {column!r} column")
    labels = table[column].astype(str).tolist()
    expected = cortical_network_indices(n_rois)
    assigned = [i for i, label in enumerate(labels) if label != NO_NETWORK]
    if expected is not None and assigned != expected:
        print("WARNING: components with a network label differ from the package's "
              "cortical evaluation mask; check the component order of the CSV.")
    return labels


def summarize(items: Sequence[ScanTrajectories], *, k: int = 100,
              labels: Sequence[str] | None = None) -> Dict[str, Any]:
    fc_pred, fc_true = group_average_fc(items)
    result: Dict[str, Any] = {
        "n_scans": len(items), "n_components": int(fc_true.shape[0]),
        "edge_recovery": edge_recovery(fc_true, fc_pred, k),
    }
    if labels is not None:
        result["network_similarity"] = network_similarity(fc_true, fc_pred, labels)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectories", type=str, nargs="+", required=True,
                   help="Cache directories written by sample_trajectories.py.")
    p.add_argument("--top-k", type=int, default=100, help="Number of strongest edges (K).")
    p.add_argument("--network-labels", type=str, default=None,
                   help="Atlas metadata CSV with a Yeo_networks7 column.")
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_scans(args.trajectories)
    n_rois = items[0].target.shape[1]
    labels = load_network_labels(args.network_labels, n_rois) if args.network_labels else None
    result = summarize(items, k=args.top_k, labels=labels)

    rec = result["edge_recovery"]
    print(f"group-average FC over {result['n_scans']} scans, "
          f"{result['n_components']} components")
    print(f"  top-{rec['k']} measured edges recovered: {rec['recovered']}/{rec['k']} "
          f"(edge density {100 * rec['edge_density']:.1f}%)")
    for network, row in result.get("network_similarity", {}).items():
        print(f"  {network:14s} r = {row['r']:.3f}  "
              f"({row['n_components']} components, {row['n_edges']} edges)")
    if labels is None:
        print("  per-network similarity skipped (no --network-labels)")
    if args.output:
        save_json(result, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
