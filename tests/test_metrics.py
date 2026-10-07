"""Numerical correctness of the evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch

from boldflow.metrics import all_metrics, fc_correlation, mse, pearson_r


def test_mse_zero_on_identity():
    x = torch.randn(10, 4)
    assert mse(x, x) == 0.0


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


def test_fc_correlation_is_high_for_consistent_predictions():
    rng = np.random.RandomState(42)
    target = rng.randn(40, 6)
    pred = target + 0.05 * rng.randn(40, 6)
    fc = fc_correlation(torch.tensor(pred), torch.tensor(target))
    assert fc > 0.9


def test_fc_correlation_component_mask():
    """Components outside the mask must not influence FC Corr."""
    rng = np.random.RandomState(0)
    target = rng.randn(60, 6)
    pred = target + 0.05 * rng.randn(60, 6)
    pred[:, 5] = rng.randn(60)                # corrupt an excluded component
    keep = [0, 1, 2, 3, 4]
    masked = fc_correlation(torch.tensor(pred), torch.tensor(target), keep)
    assert masked > 0.9
    assert masked > fc_correlation(torch.tensor(pred), torch.tensor(target))


def test_cortical_mask_sizes():
    from boldflow.difumo import cortical_network_indices

    assert len(cortical_network_indices(64)) == 55
    assert len(cortical_network_indices(256)) == 193
    assert len(cortical_network_indices(512)) == 388
    assert cortical_network_indices(61) is None
    excluded = set(range(64)) - set(cortical_network_indices(64))
    assert excluded == {1, 8, 10, 14, 20, 21, 24, 46, 63}


def test_all_metrics_reports_each_key():
    pred = torch.randn(20, 4)
    target = torch.randn(20, 4)
    out = all_metrics(pred, target)
    for k in ("mse", "pearson_r", "fc_correlation"):
        assert k in out
        assert isinstance(out[k], float)
