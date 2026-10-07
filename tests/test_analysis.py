"""Tests for the shared analysis helpers (synthetic data, CPU only)."""
from __future__ import annotations

import numpy as np
import torch

from boldflow import BoldFlow
from boldflow.analysis import (ScanTrajectories, effective_rank, fc_matrix,
                               fc_similarity, load_scans, sample_scan_trajectories,
                               save_scan, subject_bootstrap, subject_of)


def _tiny(t_out: int = 3) -> BoldFlow:
    return BoldFlow(n_channels=26, input_length=1600, n_rois=8, n_out_timesteps=t_out,
                    embed_dim=64, velocity_layers=2, n_inference_steps=4)


def test_sample_scan_trajectories_shapes_and_diversity():
    torch.manual_seed(0)
    model, n, t_out, r = _tiny(), 6, 3, 8
    eeg = torch.randn(n, 26, 1600).clamp(-15, 15)
    blocks = np.random.RandomState(0).randn(n, t_out, r).astype(np.float32)
    samples, target = sample_scan_trajectories(model, eeg, blocks, n_samples=4,
                                               device="cpu", batch_size=4)
    assert samples.shape == (4, n + t_out - 1, r)
    assert target.shape == (n + t_out - 1, r)
    assert samples.std(axis=0).mean() > 0
    det, _ = sample_scan_trajectories(model, eeg, blocks, n_samples=4, device="cpu",
                                      deterministic=True)
    assert det.shape[0] == 1


def test_cache_roundtrip(tmp_path):
    item = ScanTrajectories(scan="sub01-scan01", subject="sub01", fold=2,
                            target=np.zeros((5, 3), np.float32),
                            samples=np.ones((2, 5, 3), np.float32))
    save_scan(tmp_path / "fold_2", item)
    (loaded,) = load_scans(tmp_path)
    assert (loaded.scan, loaded.subject, loaded.fold) == ("sub01-scan01", "sub01", 2)
    assert loaded.samples.shape == (2, 5, 3)
    assert np.allclose(loaded.ensemble_std(), 0.0)


def test_fc_and_resampling_helpers():
    rng = np.random.RandomState(1)
    x = rng.randn(200, 6)
    assert abs(fc_similarity(fc_matrix(x), fc_matrix(x)) - 1.0) < 1e-9
    assert fc_matrix(x, [0, 2, 4]).shape == (3, 3)
    assert abs(effective_rank(np.eye(5)) - 5.0) < 1e-9
    assert subject_of("sub07-scan02") == "sub07"
    assert subject_of("sub-03_task-rest_run-1", "sleep") == "sub-03"
    out = subject_bootstrap([1.0, 3.0, 2.0], ["a", "a", "b"], n_boot=500)
    assert out["n_subjects"] == 2 and abs(out["mean"] - 2.0) < 1e-9
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
