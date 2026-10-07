#!/usr/bin/env python
"""Cross-session personalization on subjects with two scans (paper Appendix F).

For every subject with two scans, and for both adaptation/test directions:

1. start from the checkpoint of the outer fold that held the subject out, so
   neither scan entered its training;
2. freeze the EEG feature path (REVE encoder, spectral encoder and their
   fusion) and adapt the velocity decoder, optionally with the prior head, on
   one scan with the training objective. The last part of the adaptation scan,
   separated from the adaptation windows by a gap of frames, is held out and
   controls early stopping;
3. evaluate sampled-trajectory FC Corr, T.Corr and MSE on the other scan,
   before (zero-shot) and after adaptation, with the same sampling seed.

Learning rate and adaptation scope are selected by leave-one-subject-out
cross-fitting: the configuration used for a subject is the one with the best
mean gain over the *other* subjects. Gains are summarized over directions
with a subject-clustered bootstrap interval.

Examples
--------
    python scripts/analysis/cross_session_personalization.py \\
        --config configs/neurobolt.yaml \\
        --checkpoint-dir outputs/boldflow_neurobolt \\
        --output outputs/analysis/cross_session_personalization.json
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from boldflow.analysis import (fc_components, fc_matrix, fc_similarity, model_kwargs,
                               sample_scan_trajectories, scan_load_kwargs, subject_bootstrap)
from boldflow.data import load_scan
from boldflow.metrics import pearson_r
from boldflow.model import BoldFlow
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.training import _train_step
from boldflow.utils import (ENV_DATA_ROOT, autodetect_device, load_yaml_config,
                            resolve_path, save_json, setup_logging)

# Modules adapted under each scope; every other module (the EEG feature path)
# stays frozen and in evaluation mode.
SCOPES: Dict[str, Tuple[str, ...]] = {
    "decoder": ("velocity_net",),
    "decoder_prior": ("velocity_net", "distributional_prior_head"),
}
METRICS = ("fc_correlation", "pearson_r", "mse")
_HIGHER_IS_BETTER = {"fc_correlation": 1.0, "pearson_r": 1.0, "mse": -1.0}

Windows = Tuple[torch.Tensor, torch.Tensor]   # EEG (N, C, T), fMRI blocks (N, T_out, R)


def config_name(lr: float, scope: str) -> str:
    return f"lr{lr:g}_{scope}"


def tail_split(n_windows: int, *, tail_fraction: float = 0.15,
               gap: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptation / early-stopping window indices of one scan (anchor order).

    The first ``1 - tail_fraction`` of the windows are used for adaptation,
    the next ``gap`` windows are dropped, the rest is the held-out tail.
    """
    n_adapt = min(max(1, int(round((1.0 - tail_fraction) * n_windows))), n_windows - 1)
    if n_adapt + gap >= n_windows:
        raise ValueError(f"scan with {n_windows} windows is too short for a "
                         f"{tail_fraction:g} tail after a {gap}-frame gap")
    return np.arange(n_adapt), np.arange(n_adapt + gap, n_windows)


def set_adaptation_scope(model: BoldFlow, scope: str) -> List[torch.nn.Parameter]:
    """Freeze the model except the modules of ``scope``; return the adapted parameters."""
    for param in model.parameters():
        param.requires_grad_(False)
    adapted = [p for name in SCOPES[scope] for p in getattr(model, name).parameters()]
    for param in adapted:
        param.requires_grad_(True)
    return adapted


def _adaptation_mode(model: BoldFlow, scope: str) -> None:
    """Training mode for the adapted modules only (frozen ones keep dropout off)."""
    model.train()                       # forward() returns the loss in training mode
    for name, module in model.named_children():
        if name not in SCOPES[scope]:
            module.eval()


@torch.no_grad()
def deterministic_pearson(model: BoldFlow, eeg: torch.Tensor, target: torch.Tensor,
                          device: str, batch_size: int = 64) -> float:
    """Pearson r between the deterministic readout and the flattened targets."""
    model.eval()
    pred = torch.cat([model(eeg[i:i + batch_size].to(device), sample=False).float().cpu()
                      for i in range(0, len(eeg), batch_size)])
    a, b = pred.flatten().numpy(), target.flatten().numpy()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def adapt(model: BoldFlow, eeg: torch.Tensor, fmri: torch.Tensor, *, lr: float, scope: str,
          device: str, epochs: int = 30, patience: int = 8, weight_decay: float = 0.01,
          max_grad_norm: float = 2.0, batch_size: int = 32, tail_fraction: float = 0.15,
          gap: int = 10, seed: int = 0) -> Dict[str, Any]:
    """Adapt ``model`` in place on one scan; keep the epoch with the best tail score."""
    params = set_adaptation_scope(model, scope)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    adapt_idx, tail_idx = tail_split(len(eeg), tail_fraction=tail_fraction, gap=gap)
    target = fmri.reshape(len(fmri), -1)

    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        _adaptation_mode(model, scope)
        order = np.random.RandomState(seed + epoch).permutation(adapt_idx)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            _train_step(model, eeg[idx].to(device), target[idx].to(device),
                        optimizer, None, max_grad_norm)
        score = deterministic_pearson(model, eeg[tail_idx], target[tail_idx], device)
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: copy.deepcopy(getattr(model, name).state_dict())
                          for name in SCOPES[scope]}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        for name, state in best_state.items():
            getattr(model, name).load_state_dict(state)
    model.eval()
    return {"best_epoch": best_epoch, "tail_pearson_r": float(best_score),
            "n_adapt_windows": int(len(adapt_idx)), "n_tail_windows": int(len(tail_idx))}


def scan_metrics(model: BoldFlow, eeg: torch.Tensor, fmri: torch.Tensor, *, device: str,
                 n_samples: int = 1, seed: int = 0, batch_size: int = 64) -> Dict[str, float]:
    """FC Corr, T.Corr and MSE of sampled trajectories on one scan (mean over samples)."""
    torch.manual_seed(seed)
    samples, target = sample_scan_trajectories(
        model, eeg, fmri.numpy(), n_samples=n_samples, device=device, batch_size=batch_size)
    components = fc_components(target.shape[1])
    fc_true = fc_matrix(target, components)
    return {
        "fc_correlation": float(np.mean(
            [fc_similarity(fc_matrix(s, components), fc_true) for s in samples])),
        "pearson_r": float(np.mean([pearson_r(s, target) for s in samples])),
        "mse": float(np.mean((samples - target[None]) ** 2)),
    }


def personalize_subject(model: BoldFlow, subject: str, scans: Dict[str, Windows],
                        configs: Sequence[Tuple[float, str]], *, device: str,
                        n_samples: int = 1, seed: int = 0, **adapt_kwargs) -> List[Dict[str, Any]]:
    """Zero-shot and adapted test-scan metrics for both directions and every config."""
    base_state = copy.deepcopy(model.state_dict())
    evaluate = dict(device=device, n_samples=n_samples, seed=seed)
    rows = []
    for adapt_scan, test_scan in itertools.permutations(sorted(scans), 2):
        model.load_state_dict(base_state)
        row = {"subject": subject, "adapt_scan": adapt_scan, "test_scan": test_scan,
               "zero_shot": scan_metrics(model, *scans[test_scan], **evaluate), "adapted": {}}
        for lr, scope in configs:
            model.load_state_dict(base_state)
            info = adapt(model, *scans[adapt_scan], lr=lr, scope=scope, device=device,
                         seed=seed, **adapt_kwargs)
            row["adapted"][config_name(lr, scope)] = {
                **scan_metrics(model, *scans[test_scan], **evaluate), **info}
        rows.append(row)
    model.load_state_dict(base_state)
    return rows


def select_config(gains: Dict[Tuple[str, str], float], target_subject: str) -> str:
    """Config with the best mean gain over all subjects except ``target_subject``.

    ``gains`` maps ``(config, subject)`` to that subject's mean gain.
    """
    configs = sorted({config for config, _ in gains})
    means = {c: np.mean([g for (config, subject), g in gains.items()
                         if config == c and subject != target_subject]) for c in configs}
    return max(configs, key=lambda c: means[c])


def cross_fit(rows: List[Dict[str, Any]], metric: str = "pearson_r") -> List[Dict[str, Any]]:
    """Attach the leave-one-subject-out selected config and its gains to every direction."""
    sign = _HIGHER_IS_BETTER[metric]
    per_direction: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        for config, adapted in row["adapted"].items():
            per_direction.setdefault((config, row["subject"]), []).append(
                sign * (adapted[metric] - row["zero_shot"][metric]))
    gains = {key: float(np.mean(values)) for key, values in per_direction.items()}
    for row in rows:
        chosen = select_config(gains, row["subject"])
        row["selected_config"] = chosen
        row["gain"] = {m: row["adapted"][chosen][m] - row["zero_shot"][m] for m in METRICS}
    return rows


def summarize(rows: Sequence[Dict[str, Any]], *, n_boot: int = 10000,
              seed: int = 12345) -> Dict[str, Any]:
    """Mean metrics over directions and subject-clustered bootstrap of the gains."""
    subjects = [row["subject"] for row in rows]
    summary: Dict[str, Any] = {"n_directions": len(rows), "n_subjects": len(set(subjects))}
    for metric in METRICS:
        gains = np.array([row["gain"][metric] for row in rows])
        summary[metric] = {
            "zero_shot": float(np.mean([row["zero_shot"][metric] for row in rows])),
            "adapted": float(np.mean(
                [row["adapted"][row["selected_config"]][metric] for row in rows])),
            "gain": subject_bootstrap(gains, subjects, n_boot=n_boot, seed=seed),
            "n_directions_improved": int((_HIGHER_IS_BETTER[metric] * gains > 0).sum()),
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Directory with fold_<k>/best.pt for every outer fold.")
    p.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 5e-5])
    p.add_argument("--scopes", type=str, nargs="+", choices=sorted(SCOPES),
                   default=sorted(SCOPES))
    p.add_argument("--selection-metric", choices=METRICS, default="pearson_r",
                   help="Gain maximised by the leave-one-subject-out selection.")
    p.add_argument("--epochs", type=int, default=30, help="Maximum adaptation epochs.")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--tail-fraction", type=float, default=0.15,
                   help="Fraction of the adaptation scan held out for early stopping.")
    p.add_argument("--gap", type=int, default=10,
                   help="Frames dropped between adaptation windows and the tail.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=1,
                   help="Sampled trajectories per evaluation (metrics are averaged).")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None,
                   help="Sampling / shuffling seed (default: the config seed).")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    cfg = load_yaml_config(args.config)
    data_root = resolve_path(args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"))
    if not data_root or data_root.startswith("/path/to/"):
        raise SystemExit(f"data_root not set. Pass --data-root or set ${ENV_DATA_ROOT}.")
    cfg_seed = int(cfg.get("seed", 12345))
    seed = cfg_seed if args.seed is None else args.seed
    device = autodetect_device(args.device or cfg.get("device", "cuda"))
    splitter = SubjectLevelCVSplitter(
        data_root=data_root, k_folds=int(cfg.get("k_folds", 5)), seed=cfg_seed,
        dataset=cfg["data"]["dataset"], n_rois=int(cfg["data"]["n_rois"]),
    )
    configs = list(itertools.product(args.learning_rates, args.scopes))

    rows: List[Dict[str, Any]] = []
    for fold in splitter.get_folds():
        subjects = [s for s in fold.test_subjects if len(splitter.subjects[s]) == 2]
        if not subjects:
            continue
        model = BoldFlow.from_pretrained(
            Path(args.checkpoint_dir) / f"fold_{fold.fold_idx}" / "best.pt",
            device=device, **model_kwargs(cfg))
        for subject in subjects:
            scans = {}
            for scan in splitter.subjects[subject]:
                eeg, fmri, _ = load_scan(data_root, scan, **scan_load_kwargs(cfg))
                scans[scan] = (torch.from_numpy(np.stack(eeg)).float(),
                               torch.from_numpy(np.stack(fmri)).float())
            new = personalize_subject(
                model, subject, scans, configs, device=device, n_samples=args.n_samples,
                seed=seed, epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, tail_fraction=args.tail_fraction, gap=args.gap)
            for row in new:
                row["fold"] = fold.fold_idx
                print(f"fold {fold.fold_idx} {row['adapt_scan']} -> {row['test_scan']}: "
                      f"zero-shot FC Corr {row['zero_shot']['fc_correlation']:.3f}")
            rows.extend(new)
    if len({row["subject"] for row in rows}) < 2:
        raise SystemExit("cross-fitting needs at least two subjects with two scans")

    rows = cross_fit(rows, args.selection_metric)
    summary = summarize(rows, n_boot=args.n_boot)
    print(f"{summary['n_subjects']} subjects, {summary['n_directions']} directions")
    for metric in METRICS:
        s, g = summary[metric], summary[metric]["gain"]
        print(f"  {metric:15s} zero-shot {s['zero_shot']:.3f} -> adapted {s['adapted']:.3f}  "
              f"gain {g['mean']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]  "
              f"improved in {s['n_directions_improved']}/{summary['n_directions']} directions")
    if args.output:
        save_json({"summary": summary, "directions": rows,
                   "grid": [config_name(lr, scope) for lr, scope in configs],
                   "selection_metric": args.selection_metric, "seed": seed}, args.output)
        print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
