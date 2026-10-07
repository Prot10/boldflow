"""Cosine learning-rate schedule with linear warmup, and optimizer parameter groups."""
from __future__ import annotations

import math
from typing import List

import torch


class CosineAnnealingWarmup:
    """Cosine LR schedule preceded by linear warmup, parameterised in steps.

    The learning rate of optimizer step ``k`` (1-indexed) is
    ``base_lr * k / warmup_steps`` during warmup and follows a cosine from
    ``base_lr`` to ``min_lr`` afterwards. The rate of the first step is set at
    construction; call :meth:`step` after every optimizer step.
    """

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
        self.step()

    def step(self) -> None:
        """Set the learning rate of the next optimizer step."""
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
) -> List[dict]:
    """Optimizer parameter groups at one learning rate.

    Biases and normalisation parameters go into a group without weight decay.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return [
        {"params": [p for n, p in named if not _no_decay(n)],
         "lr": base_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in named if _no_decay(n)],
         "lr": base_lr, "weight_decay": 0.0},
    ]


def _no_decay(name: str) -> bool:
    return name.endswith(".bias") or "norm" in name.lower() or "ln" in name.split(".")
