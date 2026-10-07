"""Tests for the uncertainty analysis scripts (synthetic data, CPU only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from boldflow import BoldFlow
from boldflow.ablations import BoldFlowPointPrior
from boldflow.analysis import ScanTrajectories
from boldflow.difumo import DIFUMO_64_LABELS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibration = _load("uq_calibration")
edges = _load("uq_edge_selection")
structure = _load("uq_structure")
spread = _load("source_spread")


def _gaussian_scan(rng, name, fold, *, narrow=1.0, m=50, length=400, r=6):
    """Target and members share the predictive N(centre, s^2); members are shrunk by ``narrow``."""
    centre = rng.normal(size=(length, r))
    s = rng.uniform(0.5, 2.0, size=(length, r))
    target = centre + s * rng.normal(size=(length, r))
    samples = centre + (s / narrow) * rng.normal(size=(m, length, r))
    return ScanTrajectories(scan=name, subject=name.split("-")[0], fold=fold,
                            target=target, samples=samples)


def test_calibrated_ensemble_has_nominal_coverage_and_unit_alpha():
    rng = np.random.default_rng(0)
    val = [_gaussian_scan(rng, f"sub0{i}-scan01", 1) for i in range(2)]
    test = [_gaussian_scan(rng, f"sub1{i}-scan01", 1) for i in range(2)]
    fold = calibration.run([(val, test)], m=50)["runs"]["1/fold_1"]
    assert abs(fold["alpha"] - 1.0) < 0.05
    for readout in ("raw", "recalibrated"):
        assert abs(fold[readout]["coverage"] - 0.95) < 0.02
        assert fold[readout]["calibration_error"] < 0.02
    assert fold["raw"]["spearman"] > 0.2


def test_narrow_ensemble_recovers_scale_and_keeps_ranking():
    rng, k = np.random.default_rng(1), 4.0
    val = [_gaussian_scan(rng, "sub01-scan01", 1, narrow=k),
           _gaussian_scan(rng, "sub02-scan01", 2, narrow=k)]
    test = [_gaussian_scan(rng, "sub03-scan01", 1, narrow=k),
            _gaussian_scan(rng, "sub04-scan01", 2, narrow=k)]
    result = calibration.run([(val, test)], m=50)
    assert result["n_runs"] == 2
    for fold in result["runs"].values():
        assert abs(fold["alpha"] - k) < 0.25
        assert fold["raw"]["coverage"] < 0.5
        assert abs(fold["recalibrated"]["coverage"] - 0.95) < 0.02
        assert fold["recalibrated"]["calibration_error"] < fold["raw"]["calibration_error"]
        assert abs(fold["raw"]["spearman"] - fold["recalibrated"]["spearman"]) < 1e-12
    assert abs(result["summary"]["recalibrated"]["coverage"]["mean"] - 0.95) < 0.02
    assert set(result["runs"]["1/fold_1"]["raw"]) == {"spearman", "calibration_error", "coverage"}


def test_runs_are_folds_times_cache_pairs_and_splits_must_be_disjoint():
    rng = np.random.default_rng(6)

    def pair(narrow):
        val = [_gaussian_scan(rng, f"sub0{f}-scan01", f, narrow=narrow, length=200) for f in (1, 2)]
        test = [_gaussian_scan(rng, f"sub1{f}-scan01", f, narrow=narrow, length=200) for f in (1, 2)]
        return val, test

    pairs = [pair(2.0), pair(4.0)]              # two training seeds, two folds each
    result = calibration.run(pairs, m=50)
    assert result["n_runs"] == 4
    assert sorted(result["runs"]) == ["1/fold_1", "1/fold_2", "2/fold_1", "2/fold_2"]
    alphas = [result["runs"][k]["alpha"] for k in sorted(result["runs"])]
    assert all(abs(a - 2.0) < 0.2 for a in alphas[:2]) and all(abs(a - 4.0) < 0.4 for a in alphas[2:])
    assert abs(result["summary"]["alpha"]["std"] - np.std(alphas, ddof=1)) < 1e-12

    val, test = pairs[0]
    try:
        calibration.run([(val, test + val[:1])], m=50)
    except ValueError as err:
        assert "share subject" in str(err)
    else:
        raise AssertionError("overlapping validation and test subjects must be rejected")


def test_calibration_helpers_known_values():
    target, mean = np.array([[0.0], [0.0], [0.0], [0.0]]), np.array([[1.0], [1.0], [3.0], [3.0]])
    std = np.ones((4, 1))
    assert calibration.interval_coverage(target, mean, std, level=0.95) == 0.5


def test_selection_gain_and_holm():
    error = np.arange(1.0, 9.0)
    assert abs(edges.selection_gain(error, error, 0.5) - (1 - 2.5 / 4.5)) < 1e-12
    assert abs(edges.selection_gain(-error, error, 0.5) - (1 - 6.5 / 4.5)) < 1e-12
    assert abs(edges.selection_gain(np.ones(8), np.ones(8), 0.5)) < 1e-12
    adjusted = edges.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def _edge_scan(rng, name, fold, m=30, length=300, r=8):
    """Trajectories with component-specific noise: noisier components have less certain edges."""
    target = rng.normal(size=(length, 2)) @ rng.normal(size=(2, r)) + 0.5 * rng.normal(size=(length, r))
    noise = np.linspace(0.05, 3.0, r)
    samples = target + noise * rng.normal(size=(m, length, r))
    return ScanTrajectories(scan=name, subject=name.split("-")[0], fold=fold,
                            target=target, samples=samples)


def test_edge_selection_detects_informative_spread():
    rng = np.random.default_rng(2)
    items = [_edge_scan(rng, f"sub{i:02d}-scan01", 1 + i % 2) for i in range(8)]
    uncertainty, error = edges.edge_statistics(items[0], 30)
    assert uncertainty.shape == error.shape == (28,)
    assert edges.edge_statistics(items[0], 30, [0, 2, 4])[0].shape == (3,)
    rows = [edges.scan_row(item, 30) for item in items]
    out = edges.summarize(rows, n_boot=500, n_perm=2000, seed=0)
    for key in ("edge_spearman", "component_spearman", "edge_selection_gain"):
        e = out[key]
        assert e["estimate"] > 0 and e["ci_low"] <= e["estimate"] <= e["ci_high"]
        assert e["positive_folds"] == e["n_folds"] == 2 and e["n_subjects"] == 8
    assert out["edge_spearman"]["p_greater_zero"] < 0.05
    assert out["edge_spearman"]["p_holm"] >= out["edge_spearman"]["p_greater_zero"]
    assert set(out) == {"edge_spearman", "component_spearman", "timepoint_spearman",
                        "edge_selection_gain"}
    assert all("p_holm" in out[key] for key in edges.HOLM_FAMILY)
    assert "p_holm" not in out["edge_selection_gain"]
    assert out["component_spearman"]["p_holm"] >= out["component_spearman"]["p_greater_zero"]


def test_subject_test_null_is_not_significant():
    rng = np.random.default_rng(3)
    values = rng.normal(size=40) * 0.1
    values -= values.mean()
    out = edges.subject_test(values, [f"s{i}" for i in range(40)], fisher=False,
                             n_boot=500, n_perm=2000)
    assert abs(out["estimate"]) < 1e-9 and out["p_greater_zero"] > 0.3


def test_component_groups_and_target_dynamics():
    groups = structure.component_groups(64)
    assert (len(groups["cortical"]), len(groups["deep_gray_cerebellar"]),
            len(groups["non_neural"])) == (56, 5, 3)
    assert {DIFUMO_64_LABELS[i] for i in groups["deep_gray_cerebellar"]} == set(
        structure.DEEP_GRAY_CEREBELLAR)
    t = np.arange(600) * 2.1
    series = np.stack([np.sin(2 * np.pi * 0.02 * t), np.sin(2 * np.pi * 0.11 * t),
                       np.random.default_rng(0).normal(size=600)], axis=1)
    dyn = structure.target_dynamics(series, tr=2.1)
    assert dyn["band_power"][1] > 0.9 and dyn["band_power"][0] < 0.05
    assert dyn["roughness"][0] < dyn["roughness"][1] < dyn["roughness"][2]
    assert abs(dyn["roughness"][2] - 1.0) < 0.15
    assert abs(dyn["temporal_sd"][0] - np.sqrt(0.5)) < 0.02


def test_rank_partial_spearman():
    rng = np.random.default_rng(4)
    control = rng.normal(size=200)
    x, y = control + 0.1 * rng.normal(size=200), control + 0.1 * rng.normal(size=200)
    assert structure.spearman_rows(x, y)[0] > 0.9
    assert abs(structure.spearman_rows(x, y, control)[0]) < 0.2
    assert abs(structure.spearman_rows(x, x)[0] - 1.0) < 1e-12
    assert abs(structure.spearman_rows(x, -x)[0] + 1.0) < 1e-12


def _structure_scan(rng, name, length=300, m=12):
    """AR(1) targets whose roughness grows with the component index; spread tracks roughness."""
    phi = np.linspace(0.95, 0.0, 64)
    target = np.zeros((length, 64))
    for t in range(1, length):
        target[t] = phi * target[t - 1] + rng.normal(size=64)
    target /= target.std(axis=0)
    scale = 0.2 + (1.0 - phi) * rng.uniform(0.9, 1.1, size=64)
    for label in structure.DEEP_GRAY_CEREBELLAR:
        scale[DIFUMO_64_LABELS.index(label)] += 3.0
    samples = target + scale * rng.normal(size=(m, length, 64))
    return ScanTrajectories(scan=name, subject=name.split("-")[0], fold=1,
                            target=target, samples=samples)


def test_structure_analysis_recovers_planted_associations():
    rng = np.random.default_rng(5)
    items = [_structure_scan(rng, f"sub{i:02d}-scan{j:02d}") for i in range(5) for j in range(2)]
    out = structure.analyse(items, 12, n_boot=300, n_perm=2000, seed=0)
    assert out["n_subjects"] == 5 and out["n_scans"] == 10
    assert out["n_components"] == {"cortical": 56, "deep_gray_cerebellar": 5,
                                   "non_neural_excluded": 3}
    g = out["group_contrast"]
    assert g["difference"] > 2.0 and g["ci_low"] > 0 and g["label_permutation_p"] < 0.01
    assert abs(g["group_mean"] - g["other_mean"] - g["difference"]) < 1e-9
    rough = out["primary"]["spread_vs_roughness"]
    assert rough["estimate"] > 0.5 and rough["permutation_p"] < 0.01
    assert rough["holm_p"] >= rough["permutation_p"]
    assert rough["ci_low"] <= rough["estimate"] <= rough["ci_high"]
    assert np.isfinite(rough["partial_target_sd"]["estimate"])
    assert len(out["components"]["spread"]) == 61


def test_source_spread_learned_and_fixed_scale():
    torch.manual_seed(0)
    eeg = torch.randn(5, 26, 1600).clamp(-15, 15)
    model = BoldFlow(n_channels=26, input_length=1600, n_rois=8, n_out_timesteps=3,
                     embed_dim=64, velocity_layers=2, n_inference_steps=4).eval()
    out = spread.source_spread(model, eeg, torch.randn(5, 3, 8), n_sources=6, batch_size=4)
    with torch.no_grad():
        sigma = model.distributional_prior_head(model.encode_eeg(eeg))[1]
    assert out["n_windows"] == 5 and out["std_output"] > 0
    assert 0.5 * sigma.mean().item() < out["std_source"] < 1.5 * sigma.mean().item()
    assert abs(out["spread_preserved"] - out["std_output"] / out["std_source"]) < 1e-9
    assert -1.0 <= out["t_corr"] <= 1.0

    fixed = BoldFlowPointPrior(n_channels=26, input_length=1600, n_rois=8, embed_dim=64,
                               velocity_layers=2, n_inference_steps=4).eval()
    target = torch.randn(5, 8)
    small = spread.source_spread(fixed, eeg, target, n_sources=200, source_sigma=0.1, batch_size=5)
    large = spread.source_spread(fixed, eeg, target, n_sources=200, source_sigma=1.0, batch_size=5)
    assert abs(small["std_source"] - 0.1) < 0.01 and abs(large["std_source"] - 1.0) < 0.1
    default = spread.source_spread(fixed, eeg, target, n_sources=200, batch_size=5)
    assert abs(default["std_source"] - fixed.sigma_anneal_end) < 0.01
    try:
        spread.source_parameters(model, torch.zeros(1, 64), source_sigma=0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("learned source must reject a fixed sigma")
