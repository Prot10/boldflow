"""Shared helpers for the analysis scripts in ``scripts/analysis``.

The analyses work on cached per-scan trajectories. A *sampled trajectory* is
built by drawing one independent source per anchor, integrating the flow, and
overlap-averaging the predicted blocks into a per-TR series (paper Eq. 8).
``sample_scan_trajectories`` repeats this ``M`` times for one scan; the
scripts derive their statistics (FC, ensemble mean, spread, ...) from the
resulting ``(M, L, R)`` array.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from boldflow.difumo import cortical_network_indices
from boldflow.flow import euler_integrate
from boldflow.training import _overlap_average

_SUBJECT_PATTERNS = {
    "neurobolt": re.compile(r"^(sub\d+)-"),
    "sleep": re.compile(r"^(sub-\d+)_"),
}


def subject_of(scan_name: str, dataset: str = "neurobolt") -> str:
    """Subject identifier encoded in a scan name."""
    match = _SUBJECT_PATTERNS[dataset].match(scan_name)
    if match is None:
        raise ValueError(f"cannot parse subject from scan name {scan_name!r}")
    return match.group(1)


def scan_load_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for :func:`boldflow.data.load_scan` from a YAML config."""
    data, model = cfg["data"], cfg["model"]
    return dict(
        dataset=data["dataset"],
        n_rois=int(data["n_rois"]),
        apply_eeg_filter=bool(data.get("apply_eeg_filter", True)),
        apply_fmri_filter=bool(data.get("apply_fmri_filter", True)),
        normalize_eeg=bool(data.get("normalize_eeg", True)),
        tr=float(data.get("tr", 2.1)),
        tmin=float(data.get("tmin", -32.0)),
        tmax=float(data.get("tmax", 0.0)),
        crop=int(model["input_length"]),
        n_out_timesteps=int(model.get("n_out_timesteps", 4)),
        channels=data.get("channels"),
        zero_eeg=bool(data.get("zero_eeg", False)),
    )


def model_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Constructor arguments for :class:`boldflow.model.BoldFlow` from a config."""
    m = cfg["model"]
    kwargs = dict(
        n_channels=int(m["n_channels"]),
        input_length=int(m["input_length"]),
        n_rois=int(m["n_rois"]),
        n_out_timesteps=int(m.get("n_out_timesteps", 4)),
        embed_dim=int(m["embed_dim"]),
        velocity_layers=int(m["velocity_layers"]),
        n_inference_steps=int(m["n_inference_steps"]),
    )
    for key in ("prior_beta", "prior_loss_weight", "prior_sigma_floor", "prior_init_sigma"):
        if key in m:
            kwargs[key] = float(m[key])
    if "use_spectral_encoder" in m:
        kwargs["use_spectral_encoder"] = bool(m["use_spectral_encoder"])
    channels = cfg["data"].get("channels")
    if channels is not None:
        # Reduced montage: the encoder's coordinate encoding follows the data order.
        kwargs["channel_order"] = tuple(str(ch).upper() for ch in channels)
    return kwargs


# ---------------------------------------------------------------------------
# Trajectory sampling and caching
# ---------------------------------------------------------------------------

@dataclass
class ScanTrajectories:
    """Sampled trajectories and measured BOLD for one held-out scan."""

    scan: str
    subject: str
    fold: int
    target: np.ndarray        # (L, R) measured, preprocessed BOLD
    samples: np.ndarray       # (M, L, R) independently sampled trajectories
    tr: float = 2.1

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    def ensemble_mean(self, m: Optional[int] = None) -> np.ndarray:
        """Pointwise mean of the first ``m`` trajectories (all if ``None``)."""
        return self.samples[:m].mean(axis=0)

    def ensemble_std(self, m: Optional[int] = None) -> np.ndarray:
        """Bessel-corrected pointwise standard deviation across trajectories."""
        return self.samples[:m].std(axis=0, ddof=1)


@torch.no_grad()
def sample_scan_trajectories(
    model,
    eeg: torch.Tensor,
    fmri_blocks: np.ndarray,
    *,
    n_samples: int,
    device: str,
    batch_size: int = 64,
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n_samples`` independent trajectories for one scan.

    ``eeg`` is ``(N, C, T)`` in anchor order and ``fmri_blocks`` the matching
    ``(N, T_out, R)`` targets (``(N, R)`` for seq2one). Each trajectory uses a
    fresh source draw for every anchor; the encoder is evaluated once per
    window. Returns ``(samples (M, L, R), target (L, R))``.

    ``deterministic=True`` integrates from the source mean instead and returns
    a single trajectory; it is the only readout of a model without a learned
    source scale (no ``distributional_prior_head``).
    """
    model.eval()
    has_learned_source = hasattr(model, "distributional_prior_head")
    if not has_learned_source and not deterministic:
        raise ValueError("the model has no learned source scale: only the "
                         "deterministic readout is available")
    t_out = int(getattr(model, "n_out_timesteps", 1))
    n = eeg.shape[0]
    tgt = np.asarray(fmri_blocks, dtype=np.float32).reshape(n, t_out, -1)
    r = tgt.shape[-1]
    m_total = 1 if deterministic else n_samples
    blocks = np.zeros((m_total, n, t_out, r), dtype=np.float32)

    for start in range(0, n, batch_size):
        batch = eeg[start:start + batch_size].to(device)
        if not has_learned_source:
            blocks[0, start:start + batch.shape[0]] = (
                model(batch).float().cpu().numpy().reshape(-1, t_out, r))
            continue
        z = model.encode_eeg(batch)
        mu, sigma = model.distributional_prior_head(z)
        for m in range(m_total):
            x0 = mu if deterministic else mu + sigma * torch.randn_like(mu)
            out = euler_integrate(model.velocity_net, x0, z, model.n_inference_steps)
            blocks[m, start:start + batch.shape[0]] = (
                out.float().cpu().numpy().reshape(-1, t_out, r)
            )

    samples, target = [], None
    for m in range(m_total):
        traj, target = _overlap_average(blocks[m], tgt, t_out)
        samples.append(traj)
    return np.stack(samples), target


def save_scan(directory: str | Path, item: ScanTrajectories) -> Path:
    """Write one scan's trajectories to ``<directory>/<scan>.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item.scan}.npz"
    np.savez_compressed(
        path, samples=item.samples.astype(np.float32),
        target=item.target.astype(np.float32),
        scan=item.scan, subject=item.subject, fold=item.fold, tr=item.tr,
    )
    return path


def load_scans(directories: str | Path | Iterable[str | Path]) -> List[ScanTrajectories]:
    """Load every cached scan under one or more directories (searched recursively).

    One call loads the caches of one run per fold: a scan name found twice
    raises ``ValueError``. Scripts that combine several runs call this once
    per directory.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    items: List[ScanTrajectories] = []
    seen: Dict[str, Path] = {}
    for directory in directories:
        for path in sorted(Path(directory).rglob("*.npz")):
            with np.load(path, allow_pickle=False) as f:
                item = ScanTrajectories(
                    scan=str(f["scan"]), subject=str(f["subject"]),
                    fold=int(f["fold"]), target=f["target"],
                    samples=f["samples"], tr=float(f["tr"]),
                )
            if item.scan in seen:
                raise ValueError(
                    f"scan {item.scan!r} is cached twice: {seen[item.scan]} and {path}")
            seen[item.scan] = path
            items.append(item)
    if not items:
        raise FileNotFoundError(f"no cached trajectories (*.npz) under {directories}")
    return items


# ---------------------------------------------------------------------------
# Functional connectivity
# ---------------------------------------------------------------------------

def fc_components(n_rois: int) -> np.ndarray:
    """Component set used for FC (cortical-network mask; all if unknown)."""
    idx = cortical_network_indices(n_rois)
    return np.arange(n_rois) if idx is None else np.asarray(idx)


def fc_matrix(series: np.ndarray, components: Optional[Sequence[int]] = None) -> np.ndarray:
    """Pearson FC matrix of a ``(L, R)`` series, optionally on a component subset."""
    if components is not None:
        series = series[:, np.asarray(components)]
    return np.corrcoef(series, rowvar=False)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Off-diagonal upper-triangular entries of a symmetric matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def fc_similarity(fc_a: np.ndarray, fc_b: np.ndarray) -> float:
    """Pearson correlation between the upper triangles of two FC matrices."""
    a, b = upper_triangle(fc_a), upper_triangle(fc_b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fisher_z(fc: np.ndarray) -> np.ndarray:
    """Entrywise Fisher transform of an FC matrix (clipped away from +/-1)."""
    return np.arctanh(np.clip(fc, -0.999999, 0.999999))


def fisher_mean(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Group-average FC: entrywise Fisher-z mean, transformed back."""
    out = np.tanh(np.mean([fisher_z(m) for m in matrices], axis=0))
    np.fill_diagonal(out, 1.0)
    return out


def population_templates(items: Sequence["ScanTrajectories"]) -> Dict[int, np.ndarray]:
    """Fisher-z population FC template of every fold.

    The template of fold ``k`` is the Fisher-z mean of the measured FC
    (cortical mask) of the cached scans of all subjects not held out in fold
    ``k``; it requires the held-out caches of every fold.
    """
    comps = fc_components(items[0].target.shape[1])
    target_fc = [fc_matrix(it.target, comps) for it in items]
    templates = {}
    for fold in sorted({it.fold for it in items}):
        held_out = {it.subject for it in items if it.fold == fold}
        train = [fc for fc, it in zip(target_fc, items) if it.subject not in held_out]
        if not train:
            raise ValueError(f"fold {fold}: no cached scan of a subject outside the fold; "
                             "the template requires the caches of the other folds")
        templates[fold] = fisher_z(fisher_mean(train))
    return templates


def nanmean(values: Sequence[float]) -> float:
    """Mean of the finite entries (NaN if there are none)."""
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def effective_rank(covariance: np.ndarray) -> float:
    """Participation ratio ``(sum l)^2 / sum l^2`` of a covariance's eigenvalues."""
    eig = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = eig.sum()
    return float(total ** 2 / (eig ** 2).sum()) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def subject_means(values: Sequence[float], subjects: Sequence[str]) -> Dict[str, float]:
    """Average scan-level values within subject."""
    grouped: Dict[str, List[float]] = {}
    for value, subject in zip(values, subjects):
        if np.isfinite(value):
            grouped.setdefault(subject, []).append(float(value))
    return {s: float(np.mean(v)) for s, v in grouped.items()}


def subject_bootstrap(
    values: Sequence[float],
    subjects: Sequence[str],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, float]:
    """Mean of subject-level values with a percentile bootstrap over subjects.

    Scan-level ``values`` are first averaged within subject, so subjects (not
    scans or time points) are the resampling unit.
    """
    per_subject = np.array(list(subject_means(values, subjects).values()))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_subject), size=(n_boot, len(per_subject)))
    boot = per_subject[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(per_subject.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n_subjects": int(len(per_subject))}
