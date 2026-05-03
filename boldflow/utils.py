"""Logging, seeding, config IO, device detection, env-var path resolution."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# Environment variables recognised by the scripts.
ENV_DATA_ROOT = "BOLDFLOW_DATA_ROOT"
ENV_CHECKPOINTS_DIR = "BOLDFLOW_CHECKPOINTS_DIR"
ENV_OUTPUT_DIR = "BOLDFLOW_OUTPUT_DIR"


def setup_logging(level: str = "INFO") -> None:
    """Idempotent root-logger setup."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def autodetect_device(prefer: str = "cuda") -> str:
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_path(
    cli_value: Optional[str],
    env_var: str,
    config_value: Optional[str] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Path resolution order: CLI -> env var -> config -> default.

    Returns None if all sources are empty (or only contain placeholder strings
    like ``/path/to/...`` from the example configs).
    """
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    if config_value and not _is_placeholder(config_value):
        return config_value
    return default


def _is_placeholder(value: str) -> bool:
    return value.startswith("/path/to/") or value == "CHANGE_ME"
