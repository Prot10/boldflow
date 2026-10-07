"""Tests for the trajectory analysis scripts (synthetic data, CPU only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from boldflow.analysis import ScanTrajectories, load_scans, save_scan
from boldflow.difumo import non_neural_indices

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
TR = 2.1


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


averaging = _load("trajectory_averaging")
audit = _load("trajectory_audit")
bandlimited = _load("bandlimited_fc")
rank = _load("effective_rank")


def _scan(name, samples, target, fold=1):
    return ScanTrajectories(scan=name, subject=name.split("-")[0], fold=fold,
                            target=target.astype(np.float32),
                            samples=samples.astype(np.float32), tr=TR)


def _smooth(rng, length, n_latent=4, period=40.0):
    """Slow latent signals well below 0.15 Hz."""
    t = np.arange(length)[:, None] * TR
    freq = rng.uniform(0.01, 0.04, size=n_latent)
    return np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi, size=n_latent))


# --- trajectory averaging ---------------------------------------------------

def test_ensemble_mean_shrinks_error_variance_by_m(tmp_path):
    rng = np.random.default_rng(0)
    m, length, r, sigma = 50, 400, 8, 0.5
    for run, fold in (("seed0", 1), ("seed0", 2), ("seed1", 1)):
        signal = _smooth(rng, length) @ rng.normal(size=(4, r))
        samples = signal + sigma * rng.normal(size=(m, length, r))
        save_scan(tmp_path / run / f"fold_{fold}",
                  _scan(f"sub{fold:02d}-scan01", samples, signal, fold))
    runs = averaging.group_runs([tmp_path / "seed0", tmp_path / "seed1"])
    assert sorted(runs) == ["run_1/fold_1", "run_1/fold_2", "run_2/fold_1"]
    # directories sharing a basename are separate runs; a repeated directory is rejected
    for run in ("a", "b"):
        save_scan(tmp_path / run / "trajectories" / "fold_1",
                  _scan("sub01-scan01", np.zeros((1, 10, r)), np.zeros((10, r))))
    assert len(averaging.group_runs([tmp_path / "a" / "trajectories",
                                     tmp_path / "b" / "trajectories"])) == 2
    with pytest.raises(ValueError):
        averaging.group_runs([tmp_path / "seed0", tmp_path / "seed0"])
    save_scan(tmp_path / "a" / "trajectories" / "copy" / "fold_1",
              _scan("sub01-scan01", np.zeros((1, 10, r)), np.zeros((10, r))))
    with pytest.raises(ValueError):
        load_scans(tmp_path / "a")
    with pytest.raises(ValueError):          # one call, one run: no repeated scan name
        load_scans([tmp_path / "seed0", tmp_path / "seed1"])
    result = averaging.sweep(runs, averaging.M_GRID)
    assert result["m_grid"] == [1, 5, 10, 25, 50]          # 200 exceeds the cache
    for m_used in result["m_grid"]:
        mse = result["summary"][str(m_used)]["mse"]["mean"]
        assert mse == pytest.approx(sigma ** 2 / m_used, rel=0.1)
    s = result["summary"]
    assert s["50"]["t_corr"]["mean"] > s["1"]["t_corr"]["mean"]
    assert s["50"]["fc_corr"]["mean"] > 0.99
    # sample standard deviation across runs; 0.0 for a single run
    one = averaging.sweep({"run": runs[sorted(runs)[0]]}, [1])
    assert one["summary"]["1"]["mse"]["std"] == 0.0
    spread = [r["mse"] for r in result["per_run"] if r["m"] == 1]
    assert result["summary"]["1"]["mse"]["std"] == pytest.approx(np.std(spread, ddof=1))


# --- trajectory audit -------------------------------------------------------

def test_audit_white_noise_reference_values():
    rng = np.random.default_rng(1)
    m, length, r = 20, 400, 64        # many components: per-TR variances are stable
    samples = rng.normal(size=(m, length, r))
    target = rng.normal(size=(length, r))
    d = audit.scan_diagnostics(samples, target, tr=TR)
    assert d["terminal_rms"] == pytest.approx(1.0, rel=0.03)
    assert d["pairwise_rms_over_target_sd"] == pytest.approx(np.sqrt(2.0), rel=0.05)
    pair = audit.scan_diagnostics(samples[:2], target, tr=TR)   # two samples: their RMS difference
    assert pair["pairwise_rms_over_target_sd"] == pytest.approx(
        np.sqrt(((samples[0] - samples[1]) ** 2).mean()) / target.std(ddof=1))
    cov = audit.conditional_covariance(samples)
    assert np.trace(cov) / r == pytest.approx(samples.var(axis=0, ddof=1).mean())
    assert d["conditional_rank"] == pytest.approx(r, rel=0.02)
    assert d["conditional_rank_fraction"] == pytest.approx(1.0, rel=0.02)
    assert d["temporal_variance_ratio"] == pytest.approx(1.0, rel=0.15)
    assert d["component_variance_ratio"] == pytest.approx(1.0, rel=0.15)
    assert d["fc_rank_ratio"] == pytest.approx(1.0, rel=0.05)
    assert abs(d["lag1_generated"]) < 0.02 and abs(d["lag1_measured"]) < 0.1
    assert d["inband_tv_distance"] < 0.15
    # white noise: about half of the power lies above half the Nyquist frequency
    freqs, psd = audit.mean_psd(samples, TR)
    assert audit.band_fraction(freqs, psd, freqs[-1] / 2) == pytest.approx(0.5, abs=0.03)


def test_audit_detects_collapse_and_scale():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(300, 5))
    collapsed = np.repeat(target[None] * 0.5, 10, axis=0)
    d = audit.scan_diagnostics(collapsed, target, tr=TR)
    assert d["pairwise_rms_over_target_sd"] == pytest.approx(0.0, abs=1e-9)
    assert d["conditional_rank"] == 0.0
    assert d["temporal_variance_ratio"] == pytest.approx(0.25, rel=1e-6)
    assert d["component_variance_ratio"] == pytest.approx(0.25, rel=1e-6)
    assert d["inband_tv_distance"] == pytest.approx(0.0, abs=1e-9)
    assert d["lag1_generated"] == pytest.approx(d["lag1_measured"], abs=1e-9)


def test_rank_and_spectrum_building_blocks():
    assert audit.entropy_rank(np.eye(7)) == pytest.approx(7.0)
    assert audit.entropy_rank(np.ones((7, 7))) == pytest.approx(1.0)
    assert audit.total_variation([1, 0, 0, 0], [0, 0, 0, 1]) == 1.0
    ar = np.zeros((4000, 3))
    noise = np.random.default_rng(3).normal(size=ar.shape)
    for t in range(1, len(ar)):
        ar[t] = 0.8 * ar[t - 1] + noise[t]
    assert audit.lag1_autocorrelation(ar) == pytest.approx(0.8, abs=0.03)


def test_source_stats_and_ratio():
    floor = 0.05
    sigma = np.full((20, 4, 3), 0.4)
    stats = audit.source_stats_for_scan(sigma, floor)
    assert stats["source_rms_block"] == pytest.approx(0.4)
    assert stats["floor_fraction"] == 0.0
    interior = audit.source_stats_for_scan(np.full((2000, 4, 3), 0.4), floor)
    assert interior["source_rms_trajectory"] == pytest.approx(0.4 / 2.0, rel=0.01)
    seq2one = audit.source_stats_for_scan(np.full((20, 1, 3), 0.4), floor)
    assert seq2one["source_rms_trajectory"] == pytest.approx(seq2one["source_rms_block"])
    assert audit.source_stats_for_scan(np.full((5, 2, 3), floor), floor)["floor_fraction"] == 1.0

    rng = np.random.default_rng(4)
    scans = [_scan(f"sub{i:02d}-scan01", 0.2 * rng.normal(size=(30, 200, 4)),
                   rng.normal(size=(200, 4))) for i in range(6)]
    source = {"scans": {s.scan: {"source_rms_trajectory": 0.4, "source_rms_block": 0.8,
                                 "floor_fraction": 0.0} for s in scans}}
    result = audit.audit(scans, source, n_boot=200)
    ratio = result["summary"]["terminal_source_ratio"]
    assert ratio["mean"] == pytest.approx(0.5, rel=0.03)
    assert ratio["ci_low"] <= ratio["mean"] <= ratio["ci_high"]
    assert result["summary"]["source_floor_fraction"]["mean"] == 0.0
    assert result["n_subjects"] == 6
    block = audit.audit(scans, source, n_boot=200, source_level="block")
    assert block["summary"]["terminal_source_ratio"]["mean"] == pytest.approx(0.25, rel=0.03)
    assert "terminal_source_ratio" not in audit.audit(scans, None, n_boot=200)["summary"]


def test_geometric_subject_summary():
    out = audit.summarize([0.5, 2.0, 4.0, 1.0], ["a", "b", "c", "c"], geometric=True, n_boot=200)
    assert out["mean"] == pytest.approx((0.5 * 2.0 * 2.5) ** (1 / 3))
    assert out["n_subjects"] == 3


# --- filtering control ------------------------------------------------------

def test_lowpass_keeps_slow_and_removes_fast():
    t = np.arange(400) * TR
    slow, fast = np.sin(2 * np.pi * 0.03 * t), np.sin(2 * np.pi * 0.21 * t)
    out = bandlimited.zero_phase_filter((slow + fast)[:, None], TR)[:, 0]
    assert np.abs(out - slow)[50:-50].max() < 0.02
    assert bandlimited.power_fraction_above((slow + fast)[:, None], TR) == pytest.approx(0.5, abs=0.05)
    band = bandlimited.zero_phase_filter((slow + 3.0)[:, None], TR, low=0.01)[:, 0]
    assert abs(band[50:-50].mean()) < 0.05                 # band-pass removes the offset


def test_filtering_control_recovers_fc_under_fast_noise(tmp_path):
    rng = np.random.default_rng(5)
    length, r, m = 400, 64, 6
    t = np.arange(length)[:, None] * TR
    for i in range(4):
        target = _smooth(rng, length) @ rng.normal(size=(4, r))
        # structured contamination above the cutoff, shared across components
        fast = np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6, size=(m, 1, 1))) * rng.normal(size=r)
        save_scan(tmp_path / "fold_1", _scan(f"sub{i:02d}-scan01", target + 2.0 * fast, target))
    result = bandlimited.filtering_control(load_scans(tmp_path), n_boot=200)
    s = result["summary"]
    assert s["single_draw_filtered"]["mean"] > 0.99
    assert s["single_draw_delta"]["ci_low"] > 0.0
    assert s["ensemble_mean_filtered"]["mean"] > 0.99
    assert 0.2 < s["power_above_cutoff"]["mean"] < 1.0
    assert s["power_above_cutoff_measured"]["mean"] < 0.01
    assert "variance_removed" not in s
    assert result["filter"]["type"] == "lowpass"


def test_bandpass_filters_generated_and_measured():
    rng = np.random.default_rng(7)
    length, r = 400, 64
    t = np.arange(length)[:, None] * TR
    target = _smooth(rng, length) @ rng.normal(size=(4, r))
    # slow drift below the band, with its own spatial pattern, in the measured series only
    drift = 5.0 * np.sin(2 * np.pi * 0.002 * t) * rng.normal(size=r)
    scan = _scan("sub01-scan01", target[None], target + drift)
    low = bandlimited.scan_filtering_control(scan)
    assert low["single_draw_raw"] < 0.9
    assert low["single_draw_filtered"] == pytest.approx(low["single_draw_raw"], abs=0.02)
    band = bandlimited.scan_filtering_control(scan, low=0.01)
    assert band["single_draw_filtered"] > 0.97          # drift removed from the measured FC
    assert band["single_draw_raw"] == pytest.approx(low["single_draw_raw"])


# --- measured effective rank ------------------------------------------------

def test_measured_effective_rank_known_cases():
    rng = np.random.default_rng(6)
    white = rng.normal(size=(20000, 8))
    assert rank.measured_effective_rank(white)["effective_rank"] == pytest.approx(8.0, rel=0.01)
    shared = np.repeat(rng.normal(size=(500, 1)), 8, axis=1) * np.arange(1, 9)
    assert rank.measured_effective_rank(shared)["effective_rank"] == pytest.approx(1.0)

    wide = rng.normal(size=(300, 64))
    kept = rank.measured_effective_rank(wide, exclude_non_neural=True)
    assert kept["n_components"] == 64 - len(non_neural_indices(64))
    assert "rank_fraction" not in kept
    with pytest.raises(ValueError):                    # no non-neural list for 61 components
        rank.measured_effective_rank(wide[:, :61], exclude_non_neural=True)
    scans = [_scan(f"sub{i:02d}-scan0{j}", np.zeros((1, 300, 64)), rng.normal(size=(300, 64)))
             for i in range(5) for j in (1, 2)]
    result = rank.summarize(scans + scans[:2], n_boot=200)       # duplicates are ignored
    assert result["n_scans"] == 10
    summary = result["summary"]["effective_rank"]
    assert summary["n_subjects"] == 5
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"] < 64
