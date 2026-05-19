"""Smoke tests for the BoldFlow model.

These tests use small dummy tensors and short flows so they run on CPU in
a few seconds. They check tensor shapes, gradient flow, and that ensemble
sampling produces non-zero variance.
"""
from __future__ import annotations

import torch
import pytest

from boldflow import BoldFlow


@pytest.fixture
def tiny_model() -> BoldFlow:
    """A trimmed-down BoldFlow that fits in CPU memory and is fast on CI.

    `input_length=1600` (8 s @ 200 Hz) is the smallest size that admits the
    default MSS scales `(100, 200, 400, 800)`; reducing further would require
    re-tuning the spectral scales as well.
    """
    return BoldFlow(
        n_channels=26,
        input_length=1600,
        n_rois=8,
        n_out_timesteps=1,        # seq2one: keeps the shape assertions simple
        embed_dim=64,
        velocity_layers=2,
        n_inference_steps=5,
    )


@pytest.fixture
def tiny_seq2seq_model() -> BoldFlow:
    """Tiny seq2seq variant (T_out=3): flow_dim = n_rois * T_out = 24."""
    return BoldFlow(
        n_channels=26, input_length=1600, n_rois=8, n_out_timesteps=3,
        embed_dim=64, velocity_layers=2, n_inference_steps=5,
    )


def test_full_size_instantiation():
    """The headline configuration should instantiate within reasonable budget."""
    model = BoldFlow()
    n_params = model.num_parameters()
    assert n_params > 80_000_000, f"too few params: {n_params}"
    assert n_params < 120_000_000, f"too many params: {n_params}"


def test_forward_inference_shape(tiny_model):
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    tiny_model.eval()
    with torch.no_grad():
        pred = tiny_model(eeg)
    assert pred.shape == (2, 8)
    assert torch.isfinite(pred).all()


def test_training_loss_is_scalar_and_backprops(tiny_model):
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    fmri = torch.randn(2, 8)
    tiny_model.train()
    loss = tiny_model(eeg, fmri_target=fmri)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads), "no parameters received gradient"


def test_two_term_loss_prior_term_active(tiny_model):
    """The training loss is L = L_CFM + lambda*L_prior (paper Eq. 4-5).
    Zeroing lambda (prior_loss_weight) must change the loss value, confirming
    the beta-NLL prior term is actually contributing."""
    torch.manual_seed(0)
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    fmri = torch.randn(2, 8)
    tiny_model.train()

    torch.manual_seed(123)
    full = tiny_model(eeg, fmri_target=fmri).item()

    saved = tiny_model.prior_loss_weight
    tiny_model.prior_loss_weight = 0.0
    torch.manual_seed(123)
    no_prior = tiny_model(eeg, fmri_target=fmri).item()
    tiny_model.prior_loss_weight = saved

    assert abs(full - no_prior) > 1e-6, (
        f"prior term inactive: full={full:.6f}, no_prior={no_prior:.6f}"
    )


def test_seq2seq_flow_dim_and_inference_shape(tiny_seq2seq_model):
    """A seq2seq model flows in T_out*R space and returns (B, T_out*R)."""
    m = tiny_seq2seq_model
    assert m.flow_dim == 8 * 3
    assert m.n_out_timesteps == 3
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    m.eval()
    with torch.no_grad():
        pred = m(eeg)
    assert pred.shape == (2, 24)              # (B, T_out * R)
    assert torch.isfinite(pred).all()


def test_seq2seq_training_loss_backprops(tiny_seq2seq_model):
    """Seq2seq training takes a flattened (B, T_out*R) target block."""
    m = tiny_seq2seq_model
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    fmri = torch.randn(2, 24)                 # (B, T_out * R)
    m.train()
    loss = m(eeg, fmri_target=fmri)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())


def test_overlap_average_recovers_interior_trajectory():
    """_overlap_average must reconstruct the per-TR trajectory exactly when
    every block is a clean slice of one ground-truth trajectory."""
    import numpy as np

    from boldflow.training import _overlap_average

    t_out, r, n_tr = 3, 5, 10
    traj = np.random.RandomState(0).randn(n_tr, r).astype(np.float32)
    n_win = n_tr - t_out + 1
    blocks = np.stack([traj[i:i + t_out] for i in range(n_win)])  # (n_win, T, R)
    agg_p, agg_t = _overlap_average(blocks, blocks, t_out)
    interior = traj[t_out - 1:n_win]          # TRs covered by all T_out windows
    assert agg_p.shape == interior.shape
    assert np.allclose(agg_p, interior, atol=1e-5)
    assert np.allclose(agg_t, interior, atol=1e-5)


def test_sample_ensemble_has_variance(tiny_model):
    eeg = torch.randn(1, 26, 1600).clamp(-15, 15)
    tiny_model.eval()
    samples = tiny_model.sample_ensemble(eeg, n_samples=8)
    assert samples.shape == (8, 1, 8)
    # The ensemble should not be deterministic (non-zero std).
    assert samples.std(dim=0).mean() > 0


def test_prior_sigma_stats_returns_floats(tiny_model):
    eeg = torch.randn(1, 26, 1600).clamp(-15, 15)
    stats = tiny_model.prior_sigma_stats(eeg)
    for k in ("mean", "min", "max"):
        assert k in stats
        assert isinstance(stats[k], float)
        assert stats[k] >= tiny_model.distributional_prior_head.sigma_floor - 1e-6


def test_sleep_montage_30_channels():
    """BoldFlow with n_channels=30 must use SLEEP_CHANNEL_ORDER and accept 30-channel EEG."""
    model = BoldFlow(
        n_channels=30, input_length=1600, n_rois=8, n_out_timesteps=1,
        embed_dim=64, velocity_layers=2, n_inference_steps=4,
    )
    eeg = torch.randn(2, 30, 1600).clamp(-15, 15)
    model.eval()
    with torch.no_grad():
        pred = model(eeg)
    assert pred.shape == (2, 8)
    assert torch.isfinite(pred).all()


def test_point_prior_ablation_trains_and_anneals():
    """BoldFlowPointPrior must train and respect set_epoch sigma annealing."""
    from boldflow.ablations import BoldFlowPointPrior

    m = BoldFlowPointPrior(
        n_channels=26, input_length=1600, n_rois=8,
        embed_dim=64, velocity_layers=2, n_inference_steps=4,
        sigma_anneal_start=0.5, sigma_anneal_end=0.1, sigma_anneal_epochs=4,
    )
    assert m.current_sigma == 0.5
    m.set_epoch(2)
    assert 0.1 < m.current_sigma < 0.5    # mid-anneal
    m.set_epoch(10)
    assert abs(m.current_sigma - 0.1) < 1e-6   # past end of schedule

    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    fmri = torch.randn(2, 8)
    m.train()
    loss = m(eeg, fmri_target=fmri)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)
    m.eval()
    with torch.no_grad():
        pred = m(eeg)
    assert pred.shape == (2, 8)
