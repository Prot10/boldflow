"""Cosine-with-warmup LR + layer-wise LR decay parameter groups.

The paper uses cosine-with-warmup and ``layer_decay=1.0`` (no decay) on top
of the released REVE weights. Lower values are exposed for users who want
to reproduce a BEiT/MAE-style fine-tuning ablation.
"""
from __future__ import annotations

import math
from typing import List

import torch


class CosineAnnealingWarmup:
    """Cosine LR schedule preceded by linear warmup, parameterised in steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = self._lr_for(base_lr)

    def _lr_for(self, base_lr: float) -> float:
        if self.step_count < self.warmup_steps:
            return base_lr * self.step_count / max(1, self.warmup_steps)
        progress = (self.step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


def get_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    layer_decay: float = 1.0,
) -> List[dict]:
    """Optimizer parameter groups with optional layer-wise LR decay.

    Encoder layer at depth ``i`` gets LR ``base_lr * layer_decay^(n_layers - i)``.
    Biases and norm parameters go into a no-weight-decay group.
    """
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "transformer"):
        decay = [p for n, p in model.named_parameters() if p.requires_grad and not _no_decay(n)]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and _no_decay(n)]
        return [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]

    n_layers = len(model.encoder.transformer.layers)
    groups: dict[tuple[int, bool], dict] = {}

    def add(param: torch.nn.Parameter, depth: int, no_decay: bool) -> None:
        scale = layer_decay ** (n_layers - depth) if layer_decay < 1.0 else 1.0
        key = (depth, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [], "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = _no_decay(name)
        if name.startswith("encoder.transformer.layers."):
            depth = int(name.split(".")[3])
        else:
            depth = n_layers
        add(param, depth, no_decay)

    return list(groups.values())


def _no_decay(name: str) -> bool:
    return name.endswith(".bias") or "norm" in name.lower() or "ln" in name.split(".")
