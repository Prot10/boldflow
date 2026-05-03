"""Numerical correctness of the evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch

from boldflow.metrics import (all_metrics, fc_correlation, mae, mse,
                              pearson_r, r2_score, spearman_r)


def test_mse_zero_on_identity():
    x = torch.randn(10, 4)
    assert mse(x, x) == 0.0
    assert mae(x, x) == 0.0


def test_pearson_perfect_on_identity():
    x = torch.randn(20, 4)
    assert abs(pearson_r(x, x) - 1.0) < 1e-5


def test_pearson_negative_on_negation():
    x = torch.randn(20, 4)
    assert abs(pearson_r(x, -x) + 1.0) < 1e-5


def test_pearson_against_numpy():
    rng = np.random.RandomState(0)
    x = rng.randn(50, 3)
    y = x + 0.1 * rng.randn(50, 3)
    rs = []
    for r in range(3):
        rs.append(np.corrcoef(x[:, r], y[:, r])[0, 1])
    expected = float(np.mean(rs))
    got = pearson_r(torch.tensor(x), torch.tensor(y))
    assert abs(got - expected) < 1e-5


def test_r2_perfect_on_identity():
    x = torch.randn(20, 4)
    assert abs(r2_score(x, x) - 1.0) < 1e-5


def test_fc_correlation_is_high_for_consistent_predictions():
    rng = np.random.RandomState(42)
    target = rng.randn(40, 6)
    pred = target + 0.05 * rng.randn(40, 6)
    fc = fc_correlation(torch.tensor(pred), torch.tensor(target))
    assert fc > 0.9


def test_all_metrics_reports_each_key():
    pred = torch.randn(20, 4)
    target = torch.randn(20, 4)
    out = all_metrics(pred, target)
    for k in ("mse", "mae", "r2", "pearson_r", "spearman_r", "fc_correlation"):
        assert k in out
        assert isinstance(out[k], float)


def test_spearman_high_on_monotone_transform():
    """Spearman r should be near 1 for a monotone transform plus small noise."""
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 50).unsqueeze(1)
    y = (x ** 3) + 0.01 * torch.randn_like(x)
    # Cubic monotone with small additive noise: r should be very close to 1
    # but small ranking ties from noise can push us a bit below 0.99.
    assert spearman_r(x, y) > 0.95
