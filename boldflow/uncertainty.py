"""Native ensemble UQ + post-hoc recalibration.

Pipeline:
    1. Run ``M`` flow trajectories from samples of the distributional prior.
       Use ``samples.mean(0)`` as the point estimate, ``samples.std(0)`` as
       the raw uncertainty.
    2. Fit ``ScalarRecalibration`` on a held-out validation split so that
       ``alpha * std`` matches expected residual magnitude.
    3. Fit ``SplitConformal`` for distribution-free coverage guarantees.
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
    """Fit a global multiplier ``alpha`` so ``alpha * raw_std`` matches |residual|.

    Closed-form least-squares (Kuleshov et al., 2018, Eq. 4).
    """

    alpha: float = 1.0

    def fit(self, residuals: np.ndarray, raw_std: np.ndarray, eps: float = 1e-8) -> "ScalarRecalibration":
        residuals = np.asarray(residuals).reshape(-1)
        raw_std = np.asarray(raw_std).reshape(-1)
        mask = raw_std > eps
        residuals, raw_std = np.abs(residuals[mask]), raw_std[mask]
        self.alpha = float((residuals * raw_std).sum() / (raw_std ** 2).sum() + eps)
        return self

    def __call__(self, raw_std: np.ndarray | torch.Tensor) -> np.ndarray:
        std = raw_std.detach().cpu().numpy() if isinstance(raw_std, torch.Tensor) else np.asarray(raw_std)
        return self.alpha * std


@dataclass
class SplitConformal:
    """Distribution-free prediction intervals via split conformal regression.

    Calibration: fit the quantile ``q`` of ``|y - mu| / max(std, eps)``.
    Inference:   interval ``[mu - q*std, mu + q*std]`` covers the truth at
    least ``1 - alpha`` of the time for exchangeable data.
    """

    alpha: float = 0.05
    q: float = 1.0

    def fit(
        self,
        targets: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        eps: float = 1e-6,
    ) -> "SplitConformal":
        residuals = np.abs(np.asarray(targets) - np.asarray(means))
        normalised = residuals / np.maximum(np.asarray(stds), eps)
        n = normalised.size
        # (1-alpha)(1+1/n) finite-sample correction.
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self.q = float(np.quantile(normalised.reshape(-1), level))
        return self

    def interval(
        self, means: np.ndarray, stds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        means, stds = np.asarray(means), np.asarray(stds)
        half = self.q * stds
        return means - half, means + half

    def coverage(
        self, targets: np.ndarray, means: np.ndarray, stds: np.ndarray,
    ) -> float:
        lower, upper = self.interval(means, stds)
        targets = np.asarray(targets)
        return float(((targets >= lower) & (targets <= upper)).mean())


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


def ause(
    targets: np.ndarray, means: np.ndarray, stds: np.ndarray, n_bins: int = 100,
) -> float:
    """Area Under the Sparsification Error curve (lower is better).

    Sort by predicted uncertainty (descending), drop the most uncertain
    fraction f, compute residual MSE; subtract the same MSE under the oracle
    (sort by true residual). Average the gap over f.
    """
    res = ((np.asarray(targets) - np.asarray(means)) ** 2).reshape(-1)
    stds = np.asarray(stds).reshape(-1)
    n = len(res)
    if n == 0:
        return 0.0
    idx_unc = np.argsort(-stds)
    idx_oracle = np.argsort(-res)
    fractions = np.linspace(0, 1, n_bins, endpoint=False)
    err_unc, err_oracle = [], []
    for f in fractions:
        keep = int(n * (1 - f))
        if keep == 0:
            err_unc.append(0.0)
            err_oracle.append(0.0)
            continue
        err_unc.append(res[idx_unc[-keep:]].mean())
        err_oracle.append(res[idx_oracle[-keep:]].mean())
    err_unc, err_oracle = np.asarray(err_unc), np.asarray(err_oracle)
    return float((err_unc - err_oracle).mean())
