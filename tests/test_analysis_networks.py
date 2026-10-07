"""Tests for the network-level analysis scripts (synthetic data, CPU only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from boldflow import BoldFlow
from boldflow.analysis import ScanTrajectories, fc_matrix

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block_series(rng, n_time=400, n_blocks=3, block_size=4, noise=0.5):
    """Series whose covariance has ``n_blocks`` independent blocks of components."""
    latent = rng.randn(n_time, n_blocks)
    return np.repeat(latent, block_size, axis=1) + noise * rng.randn(n_time, n_blocks * block_size)


def _item(scan, subject, target, samples, fold=1):
    return ScanTrajectories(scan=scan, subject=subject, fold=fold,
                            target=target.astype(np.float32),
                            samples=samples.astype(np.float32))


def test_adjusted_rand_index_known_values():
    cs = _load("community_structure")
    assert cs.adjusted_rand_index([0, 0, 1, 1], [5, 5, 3, 3]) == pytest.approx(1.0)
    assert cs.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    # Hubert & Arabie style worked example: contingency [[1,1,0],[1,2,1],[0,0,4]]
    a = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b = [0, 1, 0, 1, 1, 2, 2, 2, 2, 2]
    same_both, same_a, same_b, total = 1 + 6, 1 + 6 + 6, 1 + 3 + 10, 45
    expected = same_a * same_b / total
    assert cs.adjusted_rand_index(a, b) == pytest.approx(
        (same_both - expected) / (0.5 * (same_a + same_b) - expected))


def test_louvain_recovers_blocks_and_identical_fc_gives_ari_one():
    pytest.importorskip("networkx")
    cs = _load("community_structure")
    rng = np.random.RandomState(0)
    series = _block_series(rng)
    density = 18 / 66                       # exactly the within-block edges
    assert cs.strongest_edges(fc_matrix(series), density).sum() == 2 * 18
    labels = cs.louvain_labels(fc_matrix(series), density=density)
    assert cs.adjusted_rand_index(labels, np.repeat(np.arange(3), 4)) == pytest.approx(1.0)

    same = _item("sub01-scan01", "sub01", series, series[None])
    shuffled = _item("sub02-scan01", "sub02", series, series[None][:, :, rng.permutation(12)])
    assert cs.summarize([same], density=density)["ari_mean"] == pytest.approx(1.0)
    assert cs.summarize([shuffled], density=density)["ari_mean"] < 0.5


def test_fingerprinting_identifies_subject_specific_structure():
    fp = _load("fingerprinting")
    rng = np.random.RandomState(1)
    items = []
    for s in range(6):
        mixing = rng.randn(4, 10)           # subject-specific covariance
        target = rng.randn(300, 4) @ mixing + 0.1 * rng.randn(300, 10)
        noise = rng.randn(1, 300, 10)       # prediction without subject structure
        items.append(_item(f"sub{s:02d}-scan01", f"sub{s:02d}", target, noise))
    result = fp.summarize(items)
    assert result["measured"]["accuracy"] == pytest.approx(1.0)
    assert result["measured"]["chance"] == pytest.approx(1 / 6)
    assert result["measured"]["chance_ratio"] == pytest.approx(6.0)
    assert result["predicted"]["accuracy"] < 1.0

    # identical edge vectors on both sides are matched perfectly in both directions
    vectors = rng.randn(5, 30)
    exact = fp.identification_accuracy(vectors, vectors.copy())
    assert exact["first_to_second"] == exact["second_to_first"] == 1.0
    # swapping two subjects in the pool breaks exactly those two identifications
    assert fp.identification_accuracy(vectors, vectors[[1, 0, 2, 3, 4]])["accuracy"] \
        == pytest.approx(0.6)


def test_fingerprinting_halves_and_gap():
    fp = _load("fingerprinting")
    a, b = np.arange(8.0)[:, None] * np.ones((1, 2)), 100 + np.arange(4.0)[:, None] * np.ones((1, 2))
    first, second = fp.subject_halves([a, b])
    assert first[:, 0].tolist() == [0, 1, 2, 3, 100, 101]
    assert second[:, 0].tolist() == [4, 5, 6, 7, 102, 103]
    first, second = fp.split_halves(a, gap=2)      # drops frames 3 and 4
    assert first[:, 0].tolist() == [0, 1, 2]
    assert second[:, 0].tolist() == [5, 6, 7]


def test_edge_recovery_and_network_similarity(tmp_path):
    er = _load("edge_recovery")
    rng = np.random.RandomState(2)
    series = _block_series(rng, n_blocks=3, block_size=4, noise=0.3)
    fc = fc_matrix(series)
    assert er.edge_recovery(fc, fc, k=18)["recovered"] == 18
    assert er.edge_recovery(fc, fc, k=18)["edge_density"] == pytest.approx(18 / 66)
    # a prediction with a different block layout shares only part of the strong edges:
    # blocks {0-3},{4-7},{8-11} vs {0-2},{3-5},{6-8},{9-11} share 3 + 1 + 1 + 3 pairs,
    # and the six remaining top-18 predicted edges (0, 4..9) cross the measured blocks.
    other = np.kron(np.eye(4), np.full((3, 3), -0.8))
    other[0, 4:10] = other[4:10, 0] = 0.5
    assert er.edge_recovery(fc, other, k=18)["recovered"] == 8

    labels = ["A"] * 4 + ["B"] * 4 + [er.NO_NETWORK] * 4
    sim = er.network_similarity(fc, fc, labels)
    assert set(sim) == {"A", "B"}
    assert sim["A"]["r"] == pytest.approx(1.0)
    assert sim["A"]["n_edges"] == 6 + 4 * 8     # within A + A to the other 8 components

    items = [_item(f"sub0{i}-scan01", f"sub0{i}", series, series[None]) for i in range(2)]
    result = er.summarize(items, k=18, labels=labels)
    assert result["edge_recovery"]["recovered"] == 18
    assert "group_fc_similarity" not in result
    assert "network_similarity" not in er.summarize(items, k=18)

    csv = tmp_path / "labels.csv"
    csv.write_text("Component,Yeo_networks7\n" + "\n".join(
        f"{i + 1},{label}" for i, label in enumerate(labels)) + "\n")
    assert er.load_network_labels(str(csv), 12) == labels
    with pytest.raises(ValueError):
        er.load_network_labels(str(csv), 64)


def test_per_component_report():
    pc = _load("per_component_report")
    groups = pc.component_groups()
    assert [groups.count(g) for g in pc.GROUPS] == [56, 5, 3]

    # AR(1) process: lag-1 autocorrelation equals the AR coefficient
    rng = np.random.RandomState(3)
    x = np.zeros((20000, 1))
    for t in range(1, len(x)):
        x[t] = 0.8 * x[t - 1] + rng.randn()
    assert pc.lag1_autocorrelation(x)[0] == pytest.approx(0.8, abs=0.02)

    target = rng.randn(200, 64)
    exact = [_item("sub01-scan01", "sub01", target, target[None], fold=1),
             _item("sub02-scan01", "sub02", -target, -target[None], fold=2)]
    half = [_item(it.scan, it.subject, it.target,
                  it.samples + np.where(np.arange(64) % 2 == 0, 1.0, 0.0) * rng.randn(1, 200, 64),
                  fold=it.fold) for it in exact]
    assert np.allclose(pc.component_tcorr(exact), 1.0, atol=1e-5)
    result = pc.summarize(exact, half)
    deltas = np.array([row["delta"] for row in result["components"]])
    assert (deltas[::2] > 0.1).all() and np.allclose(deltas[1::2], 0.0, atol=1e-5)
    wins = result["wins"]
    assert sum(w["n"] for w in wins.values()) == 64
    assert sum(w["won"] for w in wins.values()) == int((deltas > 0).sum())
    assert result["thalamus"]["component"] == "Thalamus"
    assert set(result["lag1"]) == {"cortical_mean", "thalamus"}


def test_checkpoint_audit():
    ca = _load("checkpoint_audit")

    def fold(idx, best, val_losses):
        return {"fold_idx": idx, "best_epoch": best,
                "history": [{"epoch": e + 1, "val_loss": v} for e, v in enumerate(val_losses)]}

    single = {"seed": 1, "fold_results": [fold(1, 2, [0.5, 0.3, 0.4, 0.45]),
                                          fold(2, 3, [0.5, 0.4, 0.35])]}
    summary = ca.audit(single, epoch_cap=30)
    assert summary["selected_epochs"] == [2, 3] and summary["last_epochs"] == [4, 3]
    assert summary["n_folds_trained_past_selection"] == 1
    assert summary["mean_late_val_mse_change"] == pytest.approx(0.15)
    assert summary["folds"][1]["late_val_mse_change"] == pytest.approx(0.0)
    multi = {"per_seed": [single, dict(single, seed=2)]}
    assert ca.audit(multi)["n_folds"] == 4


def _tiny() -> BoldFlow:
    return BoldFlow(n_channels=26, input_length=1600, n_rois=8, n_out_timesteps=3,
                    embed_dim=64, velocity_layers=2, n_inference_steps=4)


def _windows(seed, n=14):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 26, 1600, generator=g).clamp(-15, 15),
            torch.randn(n, 3, 8, generator=g))


@pytest.mark.parametrize("scope", ["decoder", "decoder_prior"])
def test_adaptation_only_changes_the_selected_scope(scope):
    cp = _load("cross_session_personalization")
    torch.manual_seed(0)
    model = _tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    eeg, fmri = _windows(0)
    info = cp.adapt(model, eeg, fmri, lr=1e-2, scope=scope, device="cpu", epochs=2,
                    batch_size=4, tail_fraction=0.3, gap=1)
    assert 1 <= info["best_epoch"] <= 2
    assert info["n_adapt_windows"] == 10 and info["n_tail_windows"] == 3

    def changed(prefix):
        return any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()
                   if k.startswith(prefix))

    assert not changed("encoder.") and not changed("spectral_encoder.")
    assert changed("velocity_net.")
    assert changed("distributional_prior_head.") == (scope == "decoder_prior")
    assert not model.training


def test_tail_split_and_cross_fitting():
    cp = _load("cross_session_personalization")
    adapt_idx, tail_idx = cp.tail_split(100)
    assert adapt_idx.tolist() == list(range(85)) and tail_idx.tolist() == list(range(95, 100))
    with pytest.raises(ValueError):
        cp.tail_split(20)

    # config "a" is best for s1 and s2, config "b" only for s3: leaving s3 out picks "a",
    # leaving s1 out picks whatever is best on average over s2 and s3.
    gains = {("a", "s1"): 0.3, ("a", "s2"): 0.2, ("a", "s3"): -0.5,
             ("b", "s1"): 0.0, ("b", "s2"): 0.1, ("b", "s3"): 0.4}
    assert cp.select_config(gains, "s3") == "a"
    assert cp.select_config(gains, "s1") == "b"

    rows = []
    for subject in ("s1", "s2", "s3"):
        for direction in range(2):
            zero = {"fc_correlation": 0.5, "pearson_r": 0.3, "mse": 0.25}
            adapted = {c: {"fc_correlation": 0.5 + gains[(c, subject)],
                           "pearson_r": 0.3 + gains[(c, subject)], "mse": 0.2}
                       for c in ("a", "b")}
            rows.append({"subject": subject, "adapt_scan": direction, "test_scan": 1 - direction,
                         "zero_shot": zero, "adapted": adapted})
    rows = cp.cross_fit(rows, "pearson_r")
    assert [r["selected_config"] for r in rows] == ["b", "b", "b", "b", "a", "a"]
    summary = cp.summarize(rows, n_boot=200)
    assert summary["n_subjects"] == 3 and summary["n_directions"] == 6
    assert summary["fc_correlation"]["gain"]["mean"] == pytest.approx((0.0 + 0.1 - 0.5) / 3)
    assert summary["fc_correlation"]["n_directions_improved"] == 2
    assert summary["mse"]["n_directions_improved"] == 6


def test_personalize_subject_runs_both_directions():
    cp = _load("cross_session_personalization")
    torch.manual_seed(0)
    model = _tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    scans = {"sub01-scan01": _windows(1), "sub01-scan02": _windows(2)}
    rows = cp.personalize_subject(model, "sub01", scans, [(1e-3, "decoder")], device="cpu",
                                  epochs=1, batch_size=4, tail_fraction=0.3, gap=1)
    assert [(r["adapt_scan"], r["test_scan"]) for r in rows] == [
        ("sub01-scan01", "sub01-scan02"), ("sub01-scan02", "sub01-scan01")]
    for row in rows:
        assert set(cp.METRICS) <= set(row["adapted"]["lr0.001_decoder"])
        assert all(np.isfinite(row["zero_shot"][m]) for m in cp.METRICS)
    # the base checkpoint is restored, and evaluation is reproducible for a fixed seed
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    again = cp.scan_metrics(model, *scans["sub01-scan01"], device="cpu")
    assert again == rows[1]["zero_shot"]
