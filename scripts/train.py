#!/usr/bin/env python
"""Train BOLDFlow with K-fold CV (optionally averaged over multiple seeds).

Examples
--------
    # Reproduce the NeuroBOLT row of Table 1 (5-fold, 30 epochs/fold)
    python scripts/train.py --config configs/neurobolt.yaml

    # Single fold, 2 epochs (sanity check)
    python scripts/train.py --config configs/neurobolt.yaml --folds 1 --epochs 2

    # Match the paper's evaluation protocol (3 seeds per fold, 15 runs total)
    python scripts/train.py --config configs/neurobolt.yaml --seeds 12345 22345 32345

    # Path overrides
    BOLDFLOW_DATA_ROOT=/scratch/neurobolt python scripts/train.py --config configs/neurobolt.yaml
    python scripts/train.py --config configs/neurobolt.yaml --data-root /scratch/neurobolt
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boldflow.data import create_cv_dataloaders
from boldflow.splits import SubjectLevelCVSplitter
from boldflow.training import aggregate, run_cv
from boldflow.utils import (ENV_CHECKPOINTS_DIR, ENV_DATA_ROOT, ENV_OUTPUT_DIR,
                            autodetect_device, load_yaml_config, resolve_path,
                            save_json, set_seed, setup_logging)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BOLDFlow with K-fold CV.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--variant", type=str, default=None,
                   choices=("default", "point_prior"),
                   help="Which model class to train. Defaults to BoldFlow; "
                        "use point_prior for the Table 2 detached-prior ablation.")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Override data root (env: {ENV_DATA_ROOT}).")
    p.add_argument("--output-dir", type=str, default=None,
                   help=f"Override output dir (env: {ENV_OUTPUT_DIR}).")
    p.add_argument("--checkpoints-dir", type=str, default=None,
                   help=f"Where to look for the pretrained encoder "
                        f"(env: {ENV_CHECKPOINTS_DIR}).")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Subset of folds to run (1-indexed).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds to repeat the CV with (paper uses 3 seeds = 15 runs).")
    p.add_argument("--device", type=str, default=None, help="`cuda` or `cpu`.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _resolve_pretrained(cfg_value, ckpt_dir):
    """Pretrained encoder lookup: CLI/config path first, else <ckpt_dir>/reve-base.safetensors."""
    if cfg_value and Path(cfg_value).exists():
        return cfg_value
    if ckpt_dir:
        candidate = Path(ckpt_dir) / "reve-base.safetensors"
        if candidate.exists():
            return str(candidate)
    return None


def _make_model_factory(cfg: Dict[str, Any], variant: str):
    """Return a zero-arg callable that instantiates the chosen model variant."""
    if variant == "point_prior":
        from boldflow.ablations import BoldFlowPointPrior
        m = cfg["model"]
        kwargs = dict(
            n_channels=int(m["n_channels"]),
            input_length=int(m["input_length"]),
            n_rois=int(m["n_rois"]),
            embed_dim=int(m["embed_dim"]),
            velocity_layers=int(m["velocity_layers"]),
            n_inference_steps=int(m["n_inference_steps"]),
            num_train_steps=int(m.get("num_train_steps", 10)),
            flow_weight=float(m.get("flow_weight", 1.0)),
            recon_weight=float(m.get("recon_weight", 1.0)),
            sigma_anneal_start=float(m.get("sigma_anneal_start", 0.5)),
            sigma_anneal_end=float(m.get("sigma_anneal_end", 0.1)),
            sigma_anneal_epochs=int(m.get("sigma_anneal_epochs", 10)),
        )
        return lambda: BoldFlowPointPrior(**kwargs)
    return None  # falls back to BoldFlow(**model_kwargs) inside run_cv


def _run_one_seed(cfg: Dict[str, Any], seed: int, output_dir: Path,
                  pretrained: str | None, device: str,
                  variant: str = "default") -> Dict[str, Any]:
    """Run a complete K-fold CV at one seed; return the aggregate dict."""
    set_seed(seed)

    splitter = SubjectLevelCVSplitter(
        data_root=cfg["data"]["data_root"],
        k_folds=int(cfg.get("k_folds", 5)),
        seed=seed,
        dataset=cfg["data"]["dataset"],
        n_rois=int(cfg["data"]["n_rois"]),
    )
    print(splitter.summary())

    loader_kwargs = dict(
        dataset=cfg["data"]["dataset"],
        n_rois=int(cfg["data"]["n_rois"]),
        target_roi=cfg["data"].get("target_roi"),
        multi_roi=bool(cfg["data"].get("multi_roi", True)),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        apply_eeg_filter=bool(cfg["data"].get("apply_eeg_filter", True)),
        apply_fmri_filter=bool(cfg["data"].get("apply_fmri_filter", True)),
        normalize_eeg=bool(cfg["data"].get("normalize_eeg", True)),
        eeg_lowpass=cfg["data"].get("eeg_lowpass"),
        exclude_non_neural=bool(cfg["data"].get("exclude_non_neural", False)),
        tr=float(cfg["data"].get("tr", 2.1)),
        tmin=float(cfg["data"].get("tmin", -32.0)),
        tmax=float(cfg["data"].get("tmax", 0.0)),
        crop=int(cfg["model"]["input_length"]),
    )

    def make_loaders(fold):
        return create_cv_dataloaders(cfg["data"]["data_root"], fold, **loader_kwargs)

    factory = _make_model_factory(cfg, variant)
    if factory is None:
        m = cfg["model"]
        model_kwargs = dict(
            n_channels=int(m["n_channels"]),
            input_length=int(m["input_length"]),
            n_rois=int(m["n_rois"]),
            embed_dim=int(m["embed_dim"]),
            velocity_layers=int(m["velocity_layers"]),
            n_inference_steps=int(m["n_inference_steps"]),
            num_train_steps=int(m.get("num_train_steps", 10)),
            flow_weight=float(m.get("flow_weight", 1.0)),
            recon_weight=float(m.get("recon_weight", 1.0)),
            prior_beta=float(m["prior_beta"]),
            prior_loss_weight=float(m["prior_loss_weight"]),
            prior_sigma_floor=float(m["prior_sigma_floor"]),
            prior_init_sigma=float(m["prior_init_sigma"]),
        )
    else:
        model_kwargs = None

    results = run_cv(
        splitter, make_loaders,
        model_kwargs=model_kwargs,
        model_factory=factory,
        pretrained_encoder=pretrained,
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        mixed_precision=bool(cfg["training"]["mixed_precision"]),
        device=device,
        output_dir=output_dir,
        folds=cfg.get("folds"),
        early_stopping_patience=cfg["training"].get("early_stopping_patience"),
    )
    summary = aggregate(results)
    summary["seed"] = seed
    return summary


def _aggregate_seeds(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the per-fold-mean metrics across seeds (mean +/- std across seeds)."""
    metric_keys = [k for k in seed_summaries[0] if k.startswith("mean_test_")]
    out: Dict[str, Any] = {"n_seeds": len(seed_summaries),
                           "seeds": [s["seed"] for s in seed_summaries]}
    for k in metric_keys:
        values = [s[k] for s in seed_summaries]
        # Replace ``mean_test_`` with the seed-aggregated mean and add std across seeds.
        bare = k.removeprefix("mean_test_")
        out[f"mean_test_{bare}"] = float(statistics.mean(values))
        out[f"std_test_{bare}_across_seeds"] = (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        )
    out["per_seed"] = seed_summaries
    return out


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_yaml_config(args.config)

    cfg["data"]["data_root"] = resolve_path(
        args.data_root, ENV_DATA_ROOT, cfg["data"].get("data_root"),
    ) or cfg["data"].get("data_root")
    cfg["output_dir"] = resolve_path(
        args.output_dir, ENV_OUTPUT_DIR, cfg.get("output_dir"),
    ) or cfg.get("output_dir", "./outputs/boldflow")
    ckpt_dir = resolve_path(
        args.checkpoints_dir, ENV_CHECKPOINTS_DIR, None, "./checkpoints",
    )

    if not cfg["data"]["data_root"] or cfg["data"]["data_root"].startswith("/path/to/"):
        raise SystemExit(
            f"data_root not set. Pass --data-root, set ${ENV_DATA_ROOT}, "
            f"or edit data.data_root in {args.config}."
        )

    # Sanity-check that ROI counts agree, otherwise the model and data shapes
    # will silently diverge inside training (cryptic shape error).
    if int(cfg["model"]["n_rois"]) != int(cfg["data"]["n_rois"]):
        raise SystemExit(
            f"config mismatch: model.n_rois={cfg['model']['n_rois']} but "
            f"data.n_rois={cfg['data']['n_rois']}. Both must agree."
        )

    if args.folds is not None:
        cfg["folds"] = args.folds
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    variant = args.variant or cfg.get("model", {}).get("variant", "default")
    device = autodetect_device(cfg.get("device", "cuda"))
    base_output = Path(cfg["output_dir"])
    base_output.mkdir(parents=True, exist_ok=True)

    pretrained = _resolve_pretrained(cfg.get("pretrained_encoder"), ckpt_dir)
    if pretrained is None:
        print("[train] WARNING: pretrained REVE weights not found, "
              "encoder will be trained from scratch.")
    else:
        print(f"[train] using pretrained encoder: {pretrained}")

    seeds = args.seeds if args.seeds else [int(cfg.get("seed", 12345))]
    seed_summaries: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = base_output if len(seeds) == 1 else base_output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("=" * 60)
        print(f"seed = {seed}  -> {seed_dir}")
        print("=" * 60)
        summary = _run_one_seed(cfg, seed, seed_dir, pretrained, device, variant=variant)
        save_json(summary, seed_dir / "results.json")
        seed_summaries.append(summary)

    if len(seed_summaries) == 1:
        final = seed_summaries[0]
    else:
        final = _aggregate_seeds(seed_summaries)
    final["config_name"] = cfg["name"]
    save_json(final, base_output / "results.json")

    print()
    print("=" * 60)
    print(f"Results ({cfg['name']})")
    print("=" * 60)
    for k, v in final.items():
        if (k.startswith("mean_test_") or k.startswith("std_test_")) and isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"  saved to: {base_output / 'results.json'}")


if __name__ == "__main__":
    main()
