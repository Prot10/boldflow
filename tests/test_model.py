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
    """The main-comparison configuration should instantiate within reasonable budget."""
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


def test_overlap_average_recovers_trajectory():
    """_overlap_average must reconstruct the per-TR trajectory exactly when
    every block is a clean slice of one ground-truth trajectory."""
    import numpy as np

    from boldflow.training import _overlap_average

    t_out, r, n_tr = 3, 5, 10
    traj = np.random.RandomState(0).randn(n_tr, r).astype(np.float32)
    n_win = n_tr - t_out + 1
    blocks = np.stack([traj[i:i + t_out] for i in range(n_win)])  # (n_win, T, R)
    # Every TR, averaged over the K_t <= T_out blocks covering it.
    agg_p, agg_t = _overlap_average(blocks, blocks, t_out)
    assert agg_p.shape == traj.shape
    assert np.allclose(agg_p, traj, atol=1e-5)
    assert np.allclose(agg_t, traj, atol=1e-5)


def test_inference_readouts(tiny_model):
    """Default inference draws a source per input; sample=False is deterministic."""
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    tiny_model.eval()
    with torch.no_grad():
        a, b = tiny_model(eeg), tiny_model(eeg)
        c, d = tiny_model(eeg, sample=False), tiny_model(eeg, sample=False)
    assert not torch.allclose(a, b)
    assert torch.allclose(c, d)


def test_evaluate_scan_level_protocol(tiny_seq2seq_model):
    """Aggregated evaluation returns one all-TR trajectory per scan."""
    from torch.utils.data import DataLoader, TensorDataset

    from boldflow.training import evaluate

    m = tiny_seq2seq_model
    sizes = [("scan_a", 5), ("scan_b", 4)]
    n = sum(s for _, s in sizes)
    loader = DataLoader(TensorDataset(torch.randn(n, 26, 1600).clamp(-15, 15),
                                      torch.randn(n, 24)), batch_size=4)
    out = evaluate(m, loader, "cpu", scan_sizes=sizes, aggregate=True)
    assert out["scan_lengths"] == [5 + 2, 4 + 2]      # n_anchors + T_out - 1
    assert out["predictions"].shape == (13, 8)
    assert "fc_correlation" in out["metrics"]


def test_sample_ensemble_has_variance(tiny_model):
    eeg = torch.randn(1, 26, 1600).clamp(-15, 15)
    tiny_model.eval()
    samples = tiny_model.sample_ensemble(eeg, n_samples=8)
    assert samples.shape == (8, 1, 8)
    # The ensemble should not be deterministic (non-zero std).
    assert samples.std(dim=0).mean() > 0


def test_load_model_selects_the_variant_and_loads_strictly(tmp_path):
    """load_model builds the class named by the config and rejects a mismatched checkpoint."""
    from boldflow.ablations import BoldFlowPointPrior
    from boldflow.model import load_model

    base = dict(n_channels=26, input_length=1600, n_rois=8, embed_dim=64,
                velocity_layers=2, n_inference_steps=4)
    full_cfg = {"model": dict(base, n_out_timesteps=1), "data": {}}
    point_cfg = {"model": dict(base, n_out_timesteps=1, variant="point_prior"), "data": {}}

    torch.manual_seed(0)
    full = BoldFlow(**base, n_out_timesteps=1)
    point = BoldFlowPointPrior(**base)
    full_path, point_path = tmp_path / "full.pt", tmp_path / "point.pt"
    torch.save({"model_state_dict": full.state_dict()}, full_path)
    torch.save({"model_state_dict": point.state_dict()}, point_path)

    loaded = load_model(full_cfg, full_path)
    assert isinstance(loaded, BoldFlow) and not loaded.training
    assert all(torch.equal(v, full.state_dict()[k]) for k, v in loaded.state_dict().items())
    loaded = load_model(point_cfg, point_path)
    assert isinstance(loaded, BoldFlowPointPrior) and not loaded.training
    assert all(torch.equal(v, point.state_dict()[k]) for k, v in loaded.state_dict().items())

    with pytest.raises(RuntimeError):
        load_model(full_cfg, point_path)
    with pytest.raises(RuntimeError):
        load_model(point_cfg, full_path)
    with pytest.raises(RuntimeError):
        BoldFlow.from_pretrained(point_path, **base, n_out_timesteps=1)


def test_warmup_starts_at_the_first_optimizer_step():
    """The scheduler sets the warmup rate at construction and reaches min_lr at the end."""
    from boldflow.schedulers import CosineAnnealingWarmup, get_param_groups

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    groups = get_param_groups(model, 1e-4, 0.01)
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
    assert sum(len(g["params"]) for g in groups) == 4
    optimizer = torch.optim.AdamW(groups)
    scheduler = CosineAnnealingWarmup(optimizer, total_steps=10, warmup_steps=4)
    rates = []
    for _ in range(10):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:4] == pytest.approx([0.25e-4, 0.5e-4, 0.75e-4, 1e-4])
    assert rates[-1] == pytest.approx(1e-6)
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))


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


def test_ot_pair_is_a_valid_coupling():
    """OT pairing returns rows of the inputs and never increases transport cost."""
    from boldflow.ablations import ot_pair

    torch.manual_seed(0)
    x0, x1 = torch.randn(16, 5), torch.randn(16, 5)
    a, b = ot_pair(x0, x1)
    assert a.shape == x0.shape and b.shape == x1.shape
    assert all((x1 == row).all(dim=1).any() for row in b)
    assert (a - b).pow(2).sum(1).mean() <= (x0 - x1).pow(2).sum(1).mean() + 1e-6


def test_reduced_montage_model():
    """A reduced montage is selected by passing its channel names as channel_order."""
    from boldflow.encoders import REDUCED_6_CHANNEL_ORDER, STANDARD_19_CHANNEL_ORDER

    assert set(REDUCED_6_CHANNEL_ORDER) < set(STANDARD_19_CHANNEL_ORDER)
    for order in (STANDARD_19_CHANNEL_ORDER, REDUCED_6_CHANNEL_ORDER):
        model = BoldFlow(
            n_channels=len(order), input_length=1600, n_rois=8, n_out_timesteps=1,
            embed_dim=64, velocity_layers=2, n_inference_steps=4, channel_order=order,
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.randn(2, len(order), 1600).clamp(-15, 15))
        assert pred.shape == (2, 8)


def test_spectral_branch_ablation():
    """use_spectral_encoder=False drops the spectral stream and still runs."""
    model = BoldFlow(n_channels=26, input_length=1600, n_rois=8, n_out_timesteps=1,
                     embed_dim=64, velocity_layers=2, n_inference_steps=4,
                     use_spectral_encoder=False)
    assert model.spectral_encoder is None
    eeg = torch.randn(2, 26, 1600).clamp(-15, 15)
    model.train()
    loss = model(eeg, fmri_target=torch.randn(2, 8))
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        assert model(eeg).shape == (2, 8)
