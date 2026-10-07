#!/usr/bin/env python
"""Held-out trajectory audit: diversity and calibration diagnostics (paper Table 9).

Reads cached sampled trajectories (``sample_trajectories.py --n-samples 200``)
and computes one value per scan for every diagnostic; individual generated
trajectories are evaluated, not their mean. Scan values are averaged within
subject and summarised across subjects with a percentile bootstrap. Ratios
(reference value one) use the geometric mean over subjects, all other
diagnostics the arithmetic mean.

Per-scan definitions, for samples ``y (M, L, R)`` and measured BOLD ``x (L, R)``:

* terminal/source RMS spread - RMS over (t, r) of the across-sample SD of ``y``
  divided by the RMS source SD (needs ``--source-stats``);
* prior scales at floor      - fraction of source sigmas at the numerical floor;
* pairwise RMS / target SD   - RMS difference between two samples over SD of ``x``;
* conditional effective rank - effective rank of the covariance of ``y`` around
  the per-TR ensemble mean, pooled over samples and TRs (also divided by R);
* temporal variance          - per-component variance over time, sample / target,
  geometric mean over components;
* across-component variance  - per-TR variance over components, sample / target,
  geometric mean over TRs;
* FC effective rank          - effective rank of the within-scan FC matrix,
  mean over sampled trajectories / target;
* spectral TV distance       - total-variation distance between the generated
  and measured Welch power profiles over four bands below 0.15 Hz;
* lag-1 autocorrelation and fraction of Welch power above 0.15 Hz, generated
  and measured.

The first two rows need the model's source scale, which the caches do not
store: run ``--collect-source-stats`` once per fold and pass the JSON back.

Examples
--------
    # optional: source scale of each held-out scan (appends to the JSON)
    python scripts/analysis/trajectory_audit.py --collect-source-stats \\
        --config configs/neurobolt.yaml \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt --fold 1 \\
        --source-stats outputs/analysis/source_stats.json

    # the audit itself
    python scripts/analysis/trajectory_audit.py outputs/trajectories_m200 \\
        --source-stats outputs/analysis/source_stats.json \\
        --output outputs/analysis/trajectory_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.signal import welch

from boldflow.analysis import (ScanTrajectories, effective_rank, fc_matrix, load_scans,
                               subject_bootstrap, subject_means)
from boldflow.utils import save_json

EPS = 1e-10
CUTOFF_HZ = 0.15
BAND_EDGES_HZ = (0.0, 0.01, 0.05, 0.10, CUTOFF_HZ)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# (key, geometric mean over subjects?)
SUMMARIES = (
    ("terminal_source_ratio", True), ("source_floor_fraction", False),
    ("pairwise_rms_over_target_sd", True), ("conditional_rank", False),
    ("conditional_rank_fraction", False), ("temporal_variance_ratio", True),
    ("component_variance_ratio", True), ("fc_rank_ratio", True),
    ("inband_tv_distance", False), ("lag1_generated", False), ("lag1_measured", False),
    ("power_above_cutoff_generated", False), ("power_above_cutoff_measured", False),
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def entropy_rank(covariance: np.ndarray) -> float:
    """Effective rank ``exp(H(p))`` with ``p`` the normalised eigenvalue spectrum."""
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(covariance)), 0.0, None)
    if eig.sum() <= EPS:
        return 0.0
    p = eig / eig.sum()
    p = p[p > EPS]
    return float(np.exp(-(p * np.log(p)).sum()))


RANKS = {"entropy": entropy_rank, "participation": effective_rank}


def geometric_mean_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Geometric mean of elementwise ratios (entries with a zero side are skipped)."""
    ok = (numerator > EPS) & (denominator > EPS)
    return float(np.exp(np.log(numerator[ok] / denominator[ok]).mean())) if ok.any() else float("nan")


def lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 autocorrelation along time of ``(..., L, R)``, averaged over series."""
    c = series - series.mean(axis=-2, keepdims=True)
    den = (c * c).sum(axis=-2)
    num = (c[..., :-1, :] * c[..., 1:, :]).sum(axis=-2)
    return float((num[den > EPS] / den[den > EPS]).mean())


def mean_psd(series: np.ndarray, tr: float, nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD along time of ``(..., L, R)``, averaged over samples and components."""
    x = np.moveaxis(series, -2, -1)
    seg = min(nperseg, x.shape[-1])
    freqs, power = welch(x, fs=1.0 / tr, nperseg=seg, noverlap=seg // 2,
                         detrend="constant", scaling="density", axis=-1)
    return freqs, power.reshape(-1, power.shape[-1]).mean(axis=0)


def band_fraction(freqs: np.ndarray, power: np.ndarray, low: float, high: float = np.inf) -> float:
    """Share of total power on the frequency bins with ``low <= f < high``."""
    keep = (freqs >= low) & (freqs < high)
    if keep.sum() < 2:
        return 0.0
    return float(_trapezoid(power[keep], freqs[keep]) / _trapezoid(power, freqs))


def inband_profile(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Power shares of the bands below the cutoff, renormalised to sum to one."""
    shares = np.array([band_fraction(freqs, power, lo, hi)
                       for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])])
    return shares / shares.sum()


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def conditional_covariance(samples: np.ndarray) -> np.ndarray:
    """Covariance over components of samples around the per-TR ensemble mean."""
    resid = (samples - samples.mean(axis=0, keepdims=True)).reshape(-1, samples.shape[-1])
    return resid.T @ resid / max(1, resid.shape[0] - 1)


# ---------------------------------------------------------------------------
# Per-scan diagnostics
# ---------------------------------------------------------------------------

def scan_diagnostics(samples: np.ndarray, target: np.ndarray, *, tr: float = 2.1,
                     fc_members: int = 50, rank: str = "entropy") -> Dict[str, float]:
    """All cache-only diagnostics of one scan (``samples (M, L, R)``, ``target (L, R)``)."""
    y = np.asarray(samples, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64)
    rank_fn = RANKS[rank]
    freqs, psd_y = mean_psd(y, tr)
    _, psd_x = mean_psd(x, tr)
    cond_rank = rank_fn(conditional_covariance(y))
    fc_rank_y = float(np.mean([rank_fn(fc_matrix(s)) for s in y[:fc_members]]))
    fc_rank_x = rank_fn(fc_matrix(x))
    return {
        "terminal_rms": float(np.sqrt(y.var(axis=0, ddof=1).mean())),
        "pairwise_rms_over_target_sd": float(np.sqrt(2.0 * y.var(axis=0).mean()) / x.std(ddof=1)),
        "conditional_rank": cond_rank,
        "conditional_rank_fraction": cond_rank / x.shape[1],
        "temporal_variance_ratio": geometric_mean_ratio(
            y.var(axis=1, ddof=1).mean(axis=0), x.var(axis=0, ddof=1)),
        "component_variance_ratio": geometric_mean_ratio(
            y.var(axis=2, ddof=1).mean(axis=0), x.var(axis=1, ddof=1)),
        "fc_rank_generated": fc_rank_y,
        "fc_rank_measured": fc_rank_x,
        "fc_rank_ratio": fc_rank_y / max(fc_rank_x, EPS),
        "inband_tv_distance": total_variation(inband_profile(freqs, psd_y),
                                              inband_profile(freqs, psd_x)),
        "lag1_generated": lag1_autocorrelation(y),
        "lag1_measured": lag1_autocorrelation(x),
        "power_above_cutoff_generated": band_fraction(freqs, psd_y, CUTOFF_HZ),
        "power_above_cutoff_measured": band_fraction(freqs, psd_x, CUTOFF_HZ),
    }


def summarize(values: Sequence[float], subjects: Sequence[str], *, geometric: bool = False,
              n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Subject-level mean (arithmetic or geometric) with a bootstrap interval."""
    means = subject_means(values, subjects)
    v = np.array(list(means.values()))
    out = subject_bootstrap(np.log(v) if geometric else v, list(means), n_boot=n_boot, seed=seed)
    if geometric:
        out.update({k: float(np.exp(out[k])) for k in ("mean", "ci_low", "ci_high")})
    return out


def audit(scans: Sequence[ScanTrajectories], source_stats: Optional[Dict[str, Any]] = None, *,
          n_samples: Optional[int] = None, fc_members: int = 50, rank: str = "entropy",
          source_level: str = "trajectory", n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Per-scan diagnostics and their subject-bootstrap summaries."""
    per_scan = (source_stats or {}).get("scans", {})
    rows = []
    for scan in scans:
        row = {"scan": scan.scan, "subject": scan.subject, "fold": scan.fold,
               **scan_diagnostics(scan.samples[:n_samples], scan.target, tr=scan.tr,
                                  fc_members=fc_members, rank=rank)}
        if scan.scan in per_scan:
            src = per_scan[scan.scan]
            row["source_rms"] = src[f"source_rms_{source_level}"]
            row["terminal_source_ratio"] = row["terminal_rms"] / row["source_rms"]
            row["source_floor_fraction"] = src["floor_fraction"]
        rows.append(row)
    subjects = [r["subject"] for r in rows]
    summary = {}
    for i, (key, geometric) in enumerate(SUMMARIES):
        values = [r.get(key, np.nan) for r in rows]
        if np.isfinite(values).any():
            summary[key] = summarize(values, subjects, geometric=geometric,
                                     n_boot=n_boot, seed=seed + i)
    return {"n_scans": len(rows), "n_subjects": len(set(subjects)),
            "n_components": int(scans[0].target.shape[1]),
            "n_samples": int(min(s.samples[:n_samples].shape[0] for s in scans)),
            "rank_definition": rank, "source_level": source_level,
            "summary": summary, "per_scan": rows}


# ---------------------------------------------------------------------------
# Source statistics (needs the model)
# ---------------------------------------------------------------------------

def source_stats_for_scan(sigma_blocks: np.ndarray, sigma_floor: float) -> Dict[str, float]:
    """Source scale of one scan from the per-anchor sigmas ``(N, T_out, R)``.

    ``source_rms_block`` is the RMS sigma of the predicted blocks.
    ``source_rms_trajectory`` is the RMS per-TR SD of the source after the same
    overlap average the trajectories go through (independent draws per anchor:
    a TR covered by ``K`` blocks has variance ``sum sigma^2 / K^2``); it is
    the like-for-like denominator for the terminal spread and equals the block
    value when ``T_out = 1``.
    """
    sigma = np.asarray(sigma_blocks, dtype=np.float64)
    n, t_out, r = sigma.shape
    acc, cnt = np.zeros((n + t_out - 1, r)), np.zeros(n + t_out - 1)
    for t in range(t_out):
        acc[t:t + n] += sigma[:, t] ** 2
        cnt[t:t + n] += 1
    return {
        "source_rms_block": float(np.sqrt((sigma ** 2).mean())),
        "source_rms_trajectory": float(np.sqrt((acc / cnt[:, None] ** 2).mean())),
        "floor_fraction": float((sigma <= sigma_floor * 1.01 + 1e-6).mean()),
    }


def collect_source_stats(args: argparse.Namespace) -> None:
    """Record the source scale of every held-out scan of one fold."""
    import torch

    from boldflow.analysis import model_kwargs, scan_load_kwargs
    from boldflow.data import load_scan
    from boldflow.model import BoldFlow
    from boldflow.splits import SubjectLevelCVSplitter
    from boldflow.utils import ENV_DATA_ROOT, autodetect_device, load_yaml_config, resolve_path

    cfg = load_yaml_config(args.config)
    data_root = resolve_path(args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"))
    if not data_root or data_root.startswith("/path/to/"):
        raise SystemExit(f"data_root not set. Pass --data-root or set ${ENV_DATA_ROOT}.")
    device = autodetect_device(args.device or cfg.get("device", "cuda"))
    fold = SubjectLevelCVSplitter(
        data_root=data_root, k_folds=int(cfg.get("k_folds", 5)),
        seed=int(cfg.get("seed", 12345)), dataset=cfg["data"]["dataset"],
        n_rois=int(cfg["data"]["n_rois"]),
    ).get_fold(args.fold)
    model = BoldFlow.from_pretrained(args.checkpoint, device=device, **model_kwargs(cfg))
    model.eval()
    floor = float(model.distributional_prior_head.sigma_floor)
    t_out = int(getattr(model, "n_out_timesteps", 1))

    path = Path(args.source_stats)
    stats = json.loads(path.read_text()) if path.exists() else {"scans": {}}
    stats["sigma_floor"] = floor
    for scan in (fold.test_scans if args.split == "test" else fold.val_scans):
        eeg, _, _ = load_scan(data_root, scan, **scan_load_kwargs(cfg))
        if not eeg:
            continue
        eeg_t, sigmas = torch.from_numpy(np.stack(eeg)).float(), []
        with torch.no_grad():
            for start in range(0, eeg_t.shape[0], args.batch_size):
                z = model.encode_eeg(eeg_t[start:start + args.batch_size].to(device))
                sigmas.append(model.distributional_prior_head(z)[1].float().cpu().numpy())
        sigma = np.concatenate(sigmas).reshape(eeg_t.shape[0], t_out, -1)
        stats["scans"][scan] = {"fold": args.fold, **source_stats_for_scan(sigma, floor)}
        print(f"{scan}: {stats['scans'][scan]}")
    save_json(stats, path)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cache_dirs", nargs="*",
                   help="Directories written by sample_trajectories.py.")
    p.add_argument("--source-stats", type=str, default=None,
                   help="JSON written by --collect-source-stats (enables the source rows).")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Use only the first M cached trajectories per scan.")
    p.add_argument("--fc-members", type=int, default=50,
                   help="Sampled trajectories used for the FC effective rank.")
    p.add_argument("--rank", choices=sorted(RANKS), default="entropy",
                   help="Effective-rank definition (entropy or participation ratio).")
    p.add_argument("--source-level", choices=["trajectory", "block"], default="trajectory",
                   help="Source scale used as denominator of the terminal/source ratio.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    g = p.add_argument_group("source statistics (runs the encoder, not the flow)")
    g.add_argument("--collect-source-stats", action="store_true")
    g.add_argument("--config", type=str, default=None)
    g.add_argument("--checkpoint", type=str, default=None)
    g.add_argument("--fold", type=int, default=1, help="1-indexed fold number.")
    g.add_argument("--split", choices=["test", "val"], default="test")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--data-root", type=str, default=None)
    g.add_argument("--device", type=str, default=None)
    return p.parse_args()


def print_table(result: Dict[str, Any]) -> None:
    s, r = result["summary"], result["n_components"]

    def ci(key: str) -> str:
        return (f"{s[key]['mean']:.3f} [{s[key]['ci_low']:.3f}, {s[key]['ci_high']:.3f}]"
                if key in s else "n/a (needs --source-stats)")

    def pair(a: str, b: str, pct: bool = False) -> str:
        fmt = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
        return f"{fmt(s[a]['mean'])} / {fmt(s[b]['mean'])}"

    floor = (f"{100 * s['source_floor_fraction']['mean']:.1f}%"
             if "source_floor_fraction" in s else "n/a (needs --source-stats)")
    table = [
        ("Terminal/source RMS spread", ci("terminal_source_ratio")),
        ("Prior scales at numerical floor", floor),
        ("Pairwise sample RMS / target SD", ci("pairwise_rms_over_target_sd")),
        ("Conditional-covariance effective rank", f"{s['conditional_rank']['mean']:.1f}/{r}"),
        ("Conditional effective rank / dimension", ci("conditional_rank_fraction")),
        ("Sample/target temporal variance", ci("temporal_variance_ratio")),
        ("Sample/target across-component variance", ci("component_variance_ratio")),
        ("Sample/target FC effective rank", ci("fc_rank_ratio")),
        ("In-band spectral total-variation distance", ci("inband_tv_distance")),
        ("Lag-1 autocorrelation: generated / measured", pair("lag1_generated", "lag1_measured")),
        (f"Power above {CUTOFF_HZ} Hz: generated / measured",
         pair("power_above_cutoff_generated", "power_above_cutoff_measured", pct=True)),
    ]
    print(f"{result['n_scans']} scans, {result['n_subjects']} subjects, "
          f"M={result['n_samples']}; brackets: subject-bootstrap 95% intervals")
    for label, value in table:
        print(f"{label:<46}{value}")


def main() -> None:
    args = parse_args()
    if args.collect_source_stats:
        if not (args.config and args.checkpoint and args.source_stats):
            raise SystemExit("--collect-source-stats needs --config, --checkpoint, --source-stats")
        collect_source_stats(args)
        return
    if not args.cache_dirs:
        raise SystemExit("pass at least one trajectory cache directory")
    source = json.loads(Path(args.source_stats).read_text()) if args.source_stats else None
    result = audit(load_scans(args.cache_dirs), source, n_samples=args.n_samples,
                   fc_members=args.fc_members, rank=args.rank,
                   source_level=args.source_level, n_boot=args.n_boot, seed=args.seed)
    print_table(result)
    if args.output:
        save_json(result, args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
