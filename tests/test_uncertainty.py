"""Tests for the post-hoc UQ pipeline."""
from __future__ import annotations

import numpy as np

from boldflow.uncertainty import (
    ScalarRecalibration, SplitConformal, ause,
    expected_calibration_error, spearman_residual_std,
)


def test_scalar_recalibration_recovers_known_alpha():
    rng = np.random.RandomState(0)
    raw = np.abs(rng.randn(1000))
    true_alpha = 2.5
    residuals = true_alpha * raw * np.sign(rng.randn(1000))
    recal = ScalarRecalibration().fit(residuals, raw)
    assert abs(recal.alpha - true_alpha) < 0.1


def test_split_conformal_coverage_matches_target():
    rng = np.random.RandomState(0)
    means = np.zeros(2000)
    stds = np.ones(2000)
    targets = rng.randn(2000)
    conf = SplitConformal(alpha=0.1).fit(targets[:1000], means[:1000], stds[:1000])
    coverage = conf.coverage(targets[1000:], means[1000:], stds[1000:])
    assert 0.85 < coverage < 0.95   # 90% nominal +/- random fluctuation


def test_calibration_metrics_are_low_for_well_calibrated_predictives():
    rng = np.random.RandomState(0)
    means = np.zeros(2000)
    stds = np.ones(2000)
    targets = rng.randn(2000)
    ece = expected_calibration_error(targets, means, stds, n_bins=10)
    assert ece < 0.05


def test_spearman_residual_std_is_high_when_std_tracks_residual():
    rng = np.random.RandomState(0)
    residual = np.abs(rng.randn(500))
    std = residual + 0.1 * rng.randn(500)        # std tracks residual + noise
    means = np.zeros(500)
    targets = residual * np.sign(rng.randn(500))
    rho = spearman_residual_std(targets, means, std)
    assert rho > 0.7


def test_ause_is_zero_for_oracle():
    rng = np.random.RandomState(0)
    targets = rng.randn(500)
    means = np.zeros(500)
    # Use the actual residual as the predicted std -> AUSE should be ~0.
    auc = ause(targets, means, np.abs(targets - means))
    assert abs(auc) < 0.01
