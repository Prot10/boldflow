"""Scalar recalibration and scoring of the native ensemble spread.

The ensemble is built from sampled trajectories: the prediction centre is the
pointwise mean of ``M`` trajectories and the raw uncertainty their
Bessel-corrected standard deviation, per TR and component
(:class:`boldflow.analysis.ScanTrajectories`). This module provides

* :class:`ScalarRecalibration`: one multiplier ``alpha`` for the raw spread,
  fitted on validation residuals;
* the calibration error and the residual-ranking score of an uncertainty
  readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ScalarRecalibration:
    """Single global multiplier ``alpha`` for the ensemble spread.

    Gaussian maximum-likelihood scale on validation residuals:
    ``alpha = sqrt(mean(r^2 / s^2))``, the multiplier that gives the
    standardised residuals unit second moment. Points whose spread does not
    exceed ``eps`` carry no scale information and are left out of the fit.
    Positive scaling changes coverage but not the ranking of predictions by
    uncertainty.
    """

    alpha: float = 1.0

    def fit(self, residuals: np.ndarray, raw_std: np.ndarray, eps: float = 1e-8) -> "ScalarRecalibration":
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        raw_std = np.asarray(raw_std, dtype=np.float64).reshape(-1)
        mask = raw_std > eps
        if not mask.any():
            raise ValueError("cannot fit alpha: the ensemble spread is zero everywhere")
        self.alpha = float(np.sqrt(np.mean((residuals[mask] / raw_std[mask]) ** 2)))
        return self

    def __call__(self, raw_std: np.ndarray | torch.Tensor) -> np.ndarray:
        std = raw_std.detach().cpu().numpy() if isinstance(raw_std, torch.Tensor) else np.asarray(raw_std)
        return self.alpha * std


def expected_calibration_error(
    targets: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calibration error of Gaussian prediction intervals.

    Mean absolute gap between empirical and nominal coverage of the central
    intervals ``mu +/- z_{(1+p)/2} * sigma`` at the nominal levels
    ``p = 1/n_bins, ..., (n_bins - 1)/n_bins``.
    """
    from scipy.stats import norm

    residuals = np.abs(np.asarray(targets) - np.asarray(means)).reshape(-1)
    stds = np.maximum(np.asarray(stds).reshape(-1), 1e-8)
    edges = np.linspace(0, 1, n_bins + 1)[1:-1]
    error = 0.0
    for p in edges:
        z = norm.ppf(0.5 + p / 2.0)
        empirical = (residuals <= z * stds).mean()
        error += abs(empirical - p)
    return float(error / len(edges))


def spearman_residual_std(
    targets: np.ndarray, means: np.ndarray, stds: np.ndarray,
) -> float:
    """Spearman correlation between absolute residual and uncertainty."""
    from scipy.stats import spearmanr
    res = np.abs(np.asarray(targets) - np.asarray(means)).reshape(-1)
    stds = np.asarray(stds).reshape(-1)
    rho, _ = spearmanr(res, stds)
    return float(0.0 if np.isnan(rho) else rho)
