"""Native ensemble UQ + post-hoc recalibration.

Pipeline:
    1. Run ``M`` flow trajectories from samples of the distributional prior.
       Use ``samples.mean(0)`` as the prediction centre, ``samples.std(0)`` as
       the raw uncertainty.
    2. Fit ``ScalarRecalibration`` on a held-out validation split: one
       multiplier ``alpha`` applied to the raw ensemble spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from boldflow.model import BoldFlow


@torch.no_grad()
def native_ensemble(
    model: BoldFlow,
    eeg: torch.Tensor,
    *,
    n_samples: int = 50,
    inference_sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensemble forward pass returning ``(mean, std)`` over members."""
    samples = model.sample_ensemble(eeg, n_samples=n_samples, inference_sigma=inference_sigma)
    return samples.mean(dim=0), samples.std(dim=0)


@dataclass
class ScalarRecalibration:
    """Fit a single global multiplier ``alpha`` for the ensemble spread.

    Gaussian maximum-likelihood scale on held-out residuals:
    ``alpha = sqrt(mean(r^2 / sigma^2))``. Positive scaling changes coverage
    but not the ranking of predictions by uncertainty.
    """

    alpha: float = 1.0

    def fit(self, residuals: np.ndarray, raw_std: np.ndarray, eps: float = 1e-8) -> "ScalarRecalibration":
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        raw_std = np.asarray(raw_std, dtype=np.float64).reshape(-1)
        mask = raw_std > eps
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
    """Reliability-diagram calibration error for Gaussian predictives.

    Mean absolute deviation between empirical and expected coverage of
    intervals ``mu +/- z(p) * sigma`` over a uniform bin grid.
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
    """Spearman correlation between |residual| and predicted std (rank quality)."""
    from scipy.stats import spearmanr
    res = np.abs(np.asarray(targets) - np.asarray(means)).reshape(-1)
    stds = np.asarray(stds).reshape(-1)
    rho, _ = spearmanr(res, stds)
    return float(0.0 if np.isnan(rho) else rho)
