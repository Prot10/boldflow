"""Training and evaluation loops for BOLDFlow.

Adds optimiser, AMP, cosine-warmup schedule, periodic validation, best-
checkpoint tracking, and a final test evaluation. The model itself does the
heavy lifting in :mod:`boldflow.model`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from boldflow.metrics import all_metrics
from boldflow.model import BoldFlow
from boldflow.schedulers import CosineAnnealingWarmup, get_param_groups

logger = logging.getLogger("boldflow.training")


@dataclass
class FoldResult:
    fold_idx: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_pearson_r: float = 0.0
    test_metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    training_time: float = 0.0
    history: list[Dict[str, float]] = field(default_factory=list)


def _train_step(
    model: BoldFlow,
    eeg: torch.Tensor,
    fmri: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional["torch.amp.GradScaler"],
    max_grad_norm: Optional[float],
    device_type: str = "cuda",
) -> float:
    """One forward / backward / optimiser step, with optional AMP."""
    optimizer.zero_grad()
    if scaler is not None:
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            loss = model(eeg, fmri_target=fmri)
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = model(eeg, fmri_target=fmri)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return float(loss.detach())


def _overlap_average(
    preds: np.ndarray, tgts: np.ndarray, t_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-average one scan's seq2seq blocks into a per-TR trajectory.

    ``preds``/``tgts`` are ``(n_windows, T_out, R)`` for a single scan, with
    windows in anchor (stride-1 TR) order. Window ``i`` offset ``t`` predicts
    within-scan TR ``i + t``; averaging the ``T_out`` estimates of each TR is
    the bagging-style aggregation of the paper (Eq. 9). Only *interior* TRs,
    covered by all ``T_out`` windows, are returned.
    """
    n, _, r = preds.shape
    n_tr = n + t_out - 1
    acc = np.zeros((n_tr, r), dtype=np.float64)
    cnt = np.zeros(n_tr, dtype=np.int64)
    tgt = np.zeros((n_tr, r), dtype=np.float64)
    for i in range(n):
        for t in range(t_out):
            j = i + t
            acc[j] += preds[i, t]
            tgt[j] = tgts[i, t]
            cnt[j] += 1
    interior = cnt == t_out
    return (acc[interior] / t_out).astype(np.float32), tgt[interior].astype(np.float32)


@torch.no_grad()
def evaluate(
    model: BoldFlow,
    loader: DataLoader,
    device: str,
    *,
    scan_sizes: Optional[List[Tuple[str, int]]] = None,
    aggregate: bool = False,
) -> Dict[str, Any]:
    """Run ``model`` over ``loader``; return ``{predictions, targets, metrics}``.

    For a seq2seq model (``model.n_out_timesteps > 1``) the raw predictions are
    ``(N, T_out*R)`` blocks. With ``aggregate=True`` and ``scan_sizes`` (the
    ordered ``[(scan, n_anchors), ...]`` list from :func:`create_cv_dataloaders`)
    the blocks are overlap-averaged per scan into the per-TR trajectory and
    metrics are computed on that (the headline protocol). Otherwise the blocks
    are flattened ``(N, T_out, R) -> (N*T_out, R)`` for a quick per-block metric
    (used for validation / model selection during training).
    """
    model.eval()
    preds, targets = [], []
    for eeg, fmri in loader:
        eeg = eeg.to(device, non_blocking=True)
        pred = model(eeg)
        preds.append(pred.cpu())
        targets.append(fmri.cpu())
    preds_t = torch.cat(preds, dim=0)
    targets_t = torch.cat(targets, dim=0)

    t_out = int(getattr(model, "n_out_timesteps", 1))
    if t_out <= 1:
        metrics = all_metrics(preds_t, targets_t)
        return {"predictions": preds_t, "targets": targets_t, "metrics": metrics}

    r = preds_t.shape[1] // t_out
    p3 = preds_t.numpy().reshape(-1, t_out, r)
    t3 = targets_t.numpy().reshape(-1, t_out, r)

    if not aggregate or not scan_sizes:
        # Quick per-block metric: every (T_out, R) block flattened onto the
        # sample axis. Used for validation / checkpoint selection.
        pm = torch.from_numpy(p3.reshape(-1, r))
        tm = torch.from_numpy(t3.reshape(-1, r))
        return {"predictions": preds_t, "targets": targets_t,
                "metrics": all_metrics(pm, tm)}

    # Headline protocol: overlap-average per scan into the per-TR trajectory.
    agg_p, agg_t, off = [], [], 0
    for _, n_win in scan_sizes:
        if n_win >= t_out:
            ap, at = _overlap_average(p3[off:off + n_win], t3[off:off + n_win], t_out)
            if len(ap):
                agg_p.append(ap)
                agg_t.append(at)
        off += n_win
    pred_traj = torch.from_numpy(np.concatenate(agg_p, axis=0))
    true_traj = torch.from_numpy(np.concatenate(agg_t, axis=0))
    return {"predictions": pred_traj, "targets": true_traj,
            "metrics": all_metrics(pred_traj, true_traj)}


def train_fold(
    model: BoldFlow,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    fold_idx: int,
    epochs: int,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3,
    layer_decay: float = 1.0,
    max_grad_norm: Optional[float] = 2.0,
    mixed_precision: bool = True,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    early_stopping_patience: Optional[int] = 10,
    test_scan_sizes: Optional[List[Tuple[str, int]]] = None,
) -> FoldResult:
    """Train one fold with cosine-warmup LR and periodic validation.

    Saves ``best.pt`` under ``output_dir/fold_<idx>/`` whenever validation
    Pearson r improves; returns a :class:`FoldResult` with test metrics from
    the best checkpoint. For a seq2seq model the final test metrics use the
    per-scan overlap-averaged trajectory (``test_scan_sizes`` from the loader
    metadata); validation uses the quick per-block metric.
    """
    model = model.to(device)
    param_groups = get_param_groups(model, lr, weight_decay, layer_decay)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = CosineAnnealingWarmup(optimizer, total_steps, warmup_steps)

    # AMP only on CUDA; torch.amp.GradScaler replaces the deprecated cuda.amp one.
    device_type = torch.device(device).type
    use_amp = mixed_precision and device_type == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device_type) if use_amp else None

    fold_dir: Optional[Path] = None
    if output_dir is not None:
        fold_dir = Path(output_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best.pt" if fold_dir is not None else None

    result = FoldResult(fold_idx=fold_idx)
    best_metric = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        # Optional epoch hook for models with annealing schedules (e.g. the
        # point-prior ablation's sigma annealing).
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch - 1)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for eeg, fmri in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            fmri = fmri.to(device, non_blocking=True)
            loss_val = _train_step(
                model, eeg, fmri, optimizer, scaler, max_grad_norm,
                device_type=device_type,
            )
            scheduler.step()
            epoch_loss += loss_val
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        val = evaluate(model, val_loader, device)
        val_pearson = val["metrics"]["pearson_r"]

        result.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["metrics"]["mse"],
            "val_pearson_r": val_pearson,
            "lr": optimizer.param_groups[0]["lr"],
        })
        logger.info(
            "fold %d / epoch %d: train_loss=%.4f val_pearson=%.4f val_mse=%.4f",
            fold_idx, epoch, train_loss, val_pearson, val["metrics"]["mse"],
        )

        if val_pearson > best_metric:
            best_metric = val_pearson
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            epochs_without_improvement = 0
            if best_path is not None:
                torch.save({"model_state_dict": best_state, "epoch": epoch,
                            "val_pearson_r": val_pearson}, best_path)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info("early stop fold %d at epoch %d (patience %d)",
                            fold_idx, epoch, early_stopping_patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, device,
                    scan_sizes=test_scan_sizes, aggregate=True)
    result.test_metrics = test["metrics"]
    result.train_loss = train_loss
    result.val_loss = val["metrics"]["mse"]
    result.val_pearson_r = best_metric
    result.training_time = time.time() - started
    return result


def run_cv(
    splitter,                       # SubjectLevelCVSplitter
    create_loaders_fn,              # callable: fold -> (train, val, test, meta)
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_factory: Optional[Any] = None,
    pretrained_encoder: Optional[str] = None,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3,
    max_grad_norm: float = 2.0,
    mixed_precision: bool = True,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    folds: Optional[list[int]] = None,
    early_stopping_patience: Optional[int] = 10,
) -> list[FoldResult]:
    """Run K-fold CV with a fresh model per fold.

    ``create_loaders_fn(fold)`` returns ``(train, val, test, metadata)``.
    By default a fresh :class:`BoldFlow` is built from ``model_kwargs``;
    pass ``model_factory`` (a zero-arg callable returning the model) to
    train an ablation variant such as :class:`BoldFlowPointPrior`.
    """
    model_kwargs = model_kwargs or {}
    folds_to_run = folds or [f.fold_idx for f in splitter.get_folds()]
    results: list[FoldResult] = []

    for fold_idx in folds_to_run:
        fold = splitter.get_fold(fold_idx)
        train_loader, val_loader, test_loader, meta = create_loaders_fn(fold)

        model = model_factory() if model_factory is not None else BoldFlow(**model_kwargs)
        if pretrained_encoder is not None and hasattr(model, "load_pretrained_encoder"):
            model.load_pretrained_encoder(pretrained_encoder)

        results.append(train_fold(
            model, train_loader, val_loader, test_loader,
            fold_idx=fold_idx, epochs=epochs, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision, device=device, output_dir=output_dir,
            early_stopping_patience=early_stopping_patience,
            test_scan_sizes=(meta or {}).get("test_scan_sizes"),
        ))
    return results


def aggregate(results: list[FoldResult]) -> Dict[str, Any]:
    """Mean / std of test metrics across folds, JSON-friendly."""
    import numpy as np
    if not results:
        return {}
    keys = list(results[0].test_metrics.keys())
    agg = {}
    for k in keys:
        values = np.array([r.test_metrics[k] for r in results])
        agg[f"mean_test_{k}"] = float(values.mean())
        agg[f"std_test_{k}"] = float(values.std())
    agg["n_folds"] = len(results)
    agg["fold_results"] = [asdict(r) for r in results]
    return agg
