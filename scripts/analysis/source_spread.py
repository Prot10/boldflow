#!/usr/bin/env python
"""Source-spread preservation through the ODE flow.

Corresponds to Table 10 ("Source-spread preservation through the ODE flow").
Unlike the other analysis scripts this one runs the model. For up to 512
validation windows of one fold it draws ``M`` sources per window, integrates
each through the flow, and reports

* ``std(tau=0)``: Bessel-corrected std across the ``M`` sources, averaged
  over output coordinates and windows,
* ``std(tau=1)``: the same for the ``M`` flow outputs,
* spread preserved ``= std(tau=1) / std(tau=0)``,
* T. Corr: per-coordinate Pearson correlation between one sampled prediction
  and the target across the windows, averaged over coordinates and over the
  ``M`` draws.

The learned-source model (``BoldFlow``) samples ``mu + sigma(x) * eps``. For
the fixed-scale ablation (config with ``model.variant: point_prior``) the
source is ``mu + sigma_src * eps`` with every ``--source-sigma`` value applied
at inference only (default: the training value ``sigma_anneal_end``).

Examples
--------
    # learned source
    python scripts/analysis/source_spread.py \\
        --config configs/neurobolt.yaml \\
        --checkpoint outputs/boldflow_neurobolt/fold_1/best.pt --fold 1 \\
        --output outputs/analysis/source_spread_learned_fold1.json

    # fixed-scale ablation at its training scale and three larger scales
    python scripts/analysis/source_spread.py \\
        --config configs/ablation_point_prior.yaml \\
        --checkpoint outputs/boldflow_point_prior/fold_1/best.pt --fold 1 \\
        --source-sigma 0.1 0.3 0.5 1.0 \\
        --output outputs/analysis/source_spread_fixed_fold1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from boldflow.ablations import BoldFlowPointPrior
from boldflow.analysis import model_kwargs, scan_load_kwargs
from boldflow.data import load_scan
from boldflow.flow import euler_integrate
from boldflow.metrics import pearson_r
from boldflow.model import BoldFlow
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.utils import (ENV_DATA_ROOT, autodetect_device, load_yaml_config,
                            resolve_path, save_json, set_seed, setup_logging)


def is_point_prior(cfg: Dict[str, Any]) -> bool:
    return cfg["model"].get("variant", "default") == "point_prior"


def load_model(cfg: Dict[str, Any], checkpoint: str, device: str) -> torch.nn.Module:
    """Instantiate the model class named by the config and load a checkpoint."""
    if not is_point_prior(cfg):
        return BoldFlow.from_pretrained(checkpoint, device=device, **model_kwargs(cfg))
    m = cfg["model"]
    model = BoldFlowPointPrior(
        n_channels=int(m["n_channels"]), input_length=int(m["input_length"]),
        n_rois=int(m["n_rois"]), embed_dim=int(m["embed_dim"]),
        velocity_layers=int(m["velocity_layers"]),
        n_inference_steps=int(m["n_inference_steps"]),
        sigma_anneal_start=float(m.get("sigma_anneal_start", 0.5)),
        sigma_anneal_end=float(m.get("sigma_anneal_end", 0.1)),
        sigma_anneal_epochs=int(m.get("sigma_anneal_epochs", 10)),
    )
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.to(device).eval()


def source_parameters(model: torch.nn.Module, z: torch.Tensor,
                      source_sigma: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and scale of the Gaussian source for embeddings ``z``.

    Learned source: the prior head's ``(mu, sigma)``. Fixed-scale ablation:
    its deterministic ``mu`` with the constant ``source_sigma``.
    """
    if isinstance(model, BoldFlowPointPrior):
        mu = model.detached_prior_net(z)
        scale = model.sigma_anneal_end if source_sigma is None else source_sigma
        return mu, torch.full_like(mu, float(scale))
    if source_sigma is not None:
        raise ValueError("--source-sigma applies to the fixed-scale ablation only")
    return model.distributional_prior_head(z)


@torch.no_grad()
def source_spread(model: torch.nn.Module, eeg: torch.Tensor, target: torch.Tensor, *,
                  n_sources: int = 20, source_sigma: Optional[float] = None,
                  batch_size: int = 32, device: str = "cpu") -> Dict[str, float]:
    """Spread of ``n_sources`` draws per window before and after the flow.

    ``eeg`` is ``(N, C, T)`` and ``target`` ``(N, D)`` with ``D`` the flow
    dimension (blocks flattened). Returns the mean coordinatewise std at
    ``tau=0`` and ``tau=1``, their ratio and the sampled-prediction T. Corr.
    """
    model.eval()
    std0, std1, outputs = [], [], []
    for start in range(0, eeg.shape[0], batch_size):
        z = model.encode_eeg(eeg[start:start + batch_size].to(device))
        mu, sigma = source_parameters(model, z, source_sigma)
        x0 = torch.stack([mu + sigma * torch.randn_like(mu) for _ in range(n_sources)])
        x1 = torch.stack([euler_integrate(model.velocity_net, x, z, model.n_inference_steps)
                          for x in x0])
        std0.append(x0.std(dim=0).float().cpu().numpy())   # torch.std is Bessel-corrected
        std1.append(x1.std(dim=0).float().cpu().numpy())
        outputs.append(x1.float().cpu().numpy())
    outputs = np.concatenate(outputs, axis=1)               # (M, N, D)
    flat_target = target.reshape(target.shape[0], -1).numpy()
    s0, s1 = float(np.concatenate(std0).mean()), float(np.concatenate(std1).mean())
    return {"std_source": s0, "std_output": s1, "spread_preserved": s1 / max(s0, 1e-12),
            "t_corr": float(np.mean([pearson_r(o, flat_target) for o in outputs])),
            "n_windows": int(eeg.shape[0]), "n_sources": int(n_sources)}


def collect_windows(data_root: str, scans: Sequence[str], cfg: Dict[str, Any],
                    max_windows: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """First ``max_windows`` windows of the given scans, in scan and anchor order."""
    eeg: List[np.ndarray] = []
    fmri: List[np.ndarray] = []
    for scan in scans:
        if len(eeg) >= max_windows:
            break
        scan_eeg, scan_fmri, _ = load_scan(data_root, scan, **scan_load_kwargs(cfg))
        eeg.extend(scan_eeg)
        fmri.extend(scan_fmri)
    if not eeg:
        raise SystemExit("no validation windows found")
    return (torch.from_numpy(np.stack(eeg[:max_windows])).float(),
            torch.from_numpy(np.stack(fmri[:max_windows])).float())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fold", type=int, default=1, help="1-indexed fold number.")
    p.add_argument("--n-windows", type=int, default=512, help="Validation windows used.")
    p.add_argument("--n-sources", type=int, default=20, help="Source draws per window (M).")
    p.add_argument("--source-sigma", type=float, nargs="+", default=None,
                   help="Fixed-scale ablation only: source scales applied at inference.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=None,
                   help="Sampling seed (default: the config seed).")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    cfg = load_yaml_config(args.config)
    data_root = resolve_path(args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"))
    if not data_root or data_root.startswith("/path/to/"):
        raise SystemExit(f"data_root not set. Pass --data-root or set ${ENV_DATA_ROOT}.")
    seed = int(cfg.get("seed", 12345))
    device = autodetect_device(args.device or cfg.get("device", "cuda"))
    if args.source_sigma and not is_point_prior(cfg):
        raise SystemExit("--source-sigma applies to the fixed-scale ablation only")

    fold = SubjectLevelCVSplitter(
        data_root=data_root, k_folds=int(cfg.get("k_folds", 5)), seed=seed,
        dataset=cfg["data"]["dataset"], n_rois=int(cfg["data"]["n_rois"]),
    ).get_fold(args.fold)
    model = load_model(cfg, args.checkpoint, device)
    eeg, target = collect_windows(data_root, fold.val_scans, cfg, args.n_windows)

    rows: List[Dict[str, Any]] = []
    print(f"{'source':<22}{'std(tau=0)':>12}{'std(tau=1)':>12}{'preserved':>12}{'T. Corr':>10}")
    for sigma in (args.source_sigma or [None]):
        set_seed(seed if args.seed is None else args.seed)
        row = source_spread(model, eeg, target, n_sources=args.n_sources, source_sigma=sigma,
                            batch_size=args.batch_size, device=device)
        if is_point_prior(cfg):
            sigma = model.sigma_anneal_end if sigma is None else sigma
        row["source"] = "learned sigma" if sigma is None else f"fixed sigma={sigma:g}"
        rows.append(row)
        print(f"{row['source']:<22}{row['std_source']:>12.3f}{row['std_output']:>12.3f}"
              f"{row['spread_preserved']:>12.3f}{row['t_corr']:>10.3f}")
    if args.output:
        save_json({"fold": args.fold, "config": Path(args.config).name, "rows": rows},
                  args.output)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
