"""Tests for the input-specificity analysis scripts (synthetic data, CPU only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from boldflow.analysis import ScanTrajectories, load_scans, save_scan

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
R = 8


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _series(rng, mixing, n_time=400):
    return (rng.standard_normal((n_time, R)) @ mixing).astype(np.float32)


def _cohort(mode: str, n_folds=3, per_fold=3, n_samples=3, seed=0):
    """Scans with shared + subject-specific covariance.

    ``specific``: trajectories share the scan's own mixing (new noise);
    ``population``: trajectories only carry the shared mixing;
    ``copy``: trajectories equal the measured series.
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal((R, R))
    items = []
    for fold in range(1, n_folds + 1):
        for s in range(per_fold):
            name = f"sub{fold}{s}"
            own = shared + 0.8 * rng.standard_normal((R, R))
            target = _series(rng, own)
            if mode == "copy":
                samples = np.repeat(target[None], n_samples, axis=0)
            else:
                mix = own if mode == "specific" else shared
                samples = np.stack([_series(rng, mix) for _ in range(n_samples)])
            items.append(ScanTrajectories(scan=f"{name}-scan01", subject=name, fold=fold,
                                          target=target, samples=samples))
    return items


def test_subject_pairing_known_answers():
    sp = _load("subject_pairing")
    items = _cohort("copy")
    templates = sp.population_templates(items)
    # template of a fold only uses scans of the other folds
    comps = np.arange(R)
    other = [np.corrcoef(it.target, rowvar=False) for it in items if it.fold != 1]
    expected = np.mean([np.arctanh(np.clip(fc, -0.999999, 0.999999)) for fc in other], axis=0)
    off = ~np.eye(len(comps), dtype=bool)
    assert np.allclose(templates[1][off], expected[off], atol=1e-6)

    rows = sp.pairing_scores(items, templates)
    assert len(rows) == len(items)
    for row in rows:  # generated == measured: matched similarity is exactly 1
        assert row["matched"] == pytest.approx(1.0, abs=1e-6)
        assert row["residual_matched"] == pytest.approx(1.0, abs=1e-6)
        assert row["residual_matched_minus_wrong"] > 0
    summary = sp.summarize(rows, n_boot=200)
    assert summary["residual_matched_minus_wrong"]["ci_low"] > 0
    assert summary["n_folds_positive_residual_matched_minus_wrong"] == 3
    assert summary["matched"]["n_subjects"] == 9


def test_subject_pairing_population_only_and_interaction(tmp_path):
    sp = _load("subject_pairing")
    model, constant = _cohort("specific"), _cohort("population")
    for it in model:
        save_scan(tmp_path / "model" / f"fold_{it.fold}", it)
    model = load_scans(tmp_path / "model")
    templates = sp.population_templates(model)
    rows = sp.pairing_scores(model, templates, n_trajectories=2)
    const_rows = sp.pairing_scores(constant, templates, n_trajectories=2)
    spec = np.mean([r["residual_matched_minus_wrong"] for r in rows])
    pop = np.mean([r["residual_matched_minus_wrong"] for r in const_rows])
    assert spec > 0.3 and abs(pop) < 0.15
    inter = sp.interaction(rows, const_rows, n_boot=200)
    assert inter["matched_minus_wrong"]["ci_low"] > 0
    assert inter["residual_matched_minus_wrong"]["ci_low"] > 0
    with pytest.raises(ValueError):
        sp.interaction(rows, const_rows[:-1])


def test_constant_input_fc():
    ci = _load("constant_input_fc")
    copies, population = _cohort("copy"), _cohort("population")
    rows = ci.fc_corr_scores(copies)
    assert all(r["sampled"] == pytest.approx(1.0, abs=1e-6) for r in rows)
    assert all(r["ensemble_mean"] == pytest.approx(1.0, abs=1e-6) for r in rows)
    assert rows[0]["ensemble_size"] == 3
    assert ci.fc_corr_scores(copies, ensemble_size=2)[0]["ensemble_size"] == 2

    const_rows = ci.fc_corr_scores(population)
    out = ci.compare(rows, const_rows, n_boot=200)
    for field in ci.READOUTS:
        assert out[field]["model"]["fold_mean"] == pytest.approx(1.0, abs=1e-6)
        expected = 1.0 - np.mean([r[field] for r in const_rows])
        assert out[field]["difference"]["mean"] == pytest.approx(expected, abs=1e-6)
        assert out[field]["difference"]["ci_low"] > 0
    # identical caches: the paired difference is exactly zero
    same = ci.compare(rows, rows, n_boot=50)
    assert same["sampled"]["difference"]["mean"] == pytest.approx(0.0, abs=1e-9)


def test_dynamic_window_bookkeeping():
    dyn = _load("dynamic_fc_alignment")
    starts = dyn.window_starts(n_time=100, window=40, step=20)
    assert starts.tolist() == [0, 20, 40, 60]
    assert dyn.displaced_starts(starts, 100, 40, 30).tolist() == [30, 50, 9, 29]
    with pytest.raises(ValueError):
        dyn.displaced_starts(starts, 100, 40, 61)
    series = np.random.default_rng(0).standard_normal((100, R))
    fc = dyn.dynamic_fc(series, starts, 40)
    assert fc.shape == (4, R, R)
    assert np.allclose(fc[1], np.corrcoef(series[20:60], rowvar=False))
    assert dyn.dynamic_similarity(fc, fc) == pytest.approx(1.0)
    assert np.isnan(dyn.dynamic_similarity(fc[:1], fc))


def _nonstationary(rng, n_segments=8, seg=60):
    """Series whose covariance changes every ``seg`` frames."""
    return np.concatenate([_series(rng, rng.standard_normal((R, R)), seg)
                           for _ in range(n_segments)])


def test_dynamic_alignment_contrasts():
    dyn = _load("dynamic_fc_alignment")
    rng = np.random.default_rng(1)
    items = []
    for fold in (1, 2):
        for s in range(3):
            target = _nonstationary(rng)
            items.append(ScanTrajectories(
                scan=f"sub{fold}{s}-scan01", subject=f"sub{fold}{s}", fold=fold,
                target=target, samples=target[None].copy(), tr=2.0))
    # 120-s windows = 60 frames, 30-s step = 15 frames, 180-s shift = 90 frames
    rows = dyn.dynamic_scores(items, templates={})
    for row in rows:  # generated == measured: aligned similarity is exactly 1
        assert row["aligned"] == pytest.approx(1.0, abs=1e-6)
        assert row["aligned_minus_displaced"] > 0.3
        assert row["matched_minus_wrong"] > 0.3
        assert "residual_aligned" not in row
    summary = dyn.summarize(rows, n_boot=200)
    assert summary["aligned_minus_displaced"]["ci_low"] > 0
    assert summary["matched_minus_wrong"]["n_subjects"] == 6

    rows = dyn.dynamic_scores(items)  # residual scores with cache-derived templates
    assert all(r["residual_aligned"] == pytest.approx(1.0, abs=1e-6) for r in rows)
    assert all(r["residual_aligned_minus_displaced"] > 0 for r in rows)


def test_dynamic_alignment_stationary_and_short_scans():
    dyn = _load("dynamic_fc_alignment")
    rng = np.random.default_rng(2)
    items = _cohort("specific", n_folds=2, n_samples=1, seed=3)
    for it in items:
        it.tr = 2.0
    rows = dyn.dynamic_scores(items, templates={})
    # stationary covariance: no window is special, so displacement does not matter
    assert abs(np.mean([r["aligned_minus_displaced"] for r in rows])) < 0.05
    assert np.mean([r["matched_minus_wrong"] for r in rows]) > 0.1

    short = ScanTrajectories(scan="sub99-scan01", subject="sub99", fold=1,
                             target=_series(rng, np.eye(R), 100),
                             samples=_series(rng, np.eye(R), 100)[None], tr=2.0)
    rows = dyn.dynamic_scores(items + [short], templates={})
    assert np.isnan(rows[-1]["aligned"])
    assert dyn.summarize(rows, n_boot=50)["aligned"]["n_subjects"] == len(items)
