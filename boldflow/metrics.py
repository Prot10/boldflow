"""Evaluation metrics: MSE, MAE, R2, Pearson r (T.Corr), Spearman, FC Corr."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error."""
    return float(((pred - target) ** 2).mean())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error."""
    return float((pred - target).abs().mean())


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination."""
    p, t = _to_numpy(pred), _to_numpy(target)
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Per-ROI Pearson r averaged across ROIs (T.Corr in the paper).

    Inputs ``(n_samples, n_rois)``. NaNs (zero-variance ROIs) become 0.
    """
    p, t = _to_numpy(pred), _to_numpy(target)
    if p.ndim == 1:
        p, t = p[:, None], t[:, None]
    p = p - p.mean(axis=0, keepdims=True)
    t = t - t.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0)) + 1e-12
    r = num / den
    return float(np.nan_to_num(r).mean())


def spearman_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Per-ROI Spearman r averaged across ROIs."""
    from scipy.stats import spearmanr
    p, t = _to_numpy(pred), _to_numpy(target)
    if p.ndim == 1:
        p, t = p[:, None], t[:, None]
    rs = []
    for r in range(p.shape[1]):
        if np.std(t[:, r]) < 1e-8 or np.std(p[:, r]) < 1e-8:
            continue
        rho, _ = spearmanr(p[:, r], t[:, r])
        if not np.isnan(rho):
            rs.append(rho)
    return float(np.mean(rs)) if rs else 0.0


def _upper_tri(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def fc_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson r between predicted and ground-truth FC upper triangles.

    The functional connectivity matrix is the ROI x ROI Pearson correlation
    of the time courses; the metric ignores the diagonal.
    """
    p, t = _to_numpy(pred), _to_numpy(target)
    if p.ndim == 1 or p.shape[1] < 2:
        return 0.0
    fc_p = np.corrcoef(p, rowvar=False)
    fc_t = np.corrcoef(t, rowvar=False)
    a, b = _upper_tri(fc_p), _upper_tri(fc_t)
    if np.isnan(a).any() or np.isnan(b).any():
        mask = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[mask], b[mask]
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute every metric used by :mod:`boldflow.training`."""
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "r2": r2_score(pred, target),
        "pearson_r": pearson_r(pred, target),
        "spearman_r": spearman_r(pred, target),
        "fc_correlation": fc_correlation(pred, target),
    }
