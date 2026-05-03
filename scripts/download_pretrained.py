#!/usr/bin/env python
"""Download REVE-base pretrained weights from HuggingFace into a checkpoints dir.

The default destination is ``./checkpoints/reve-base.safetensors`` (which the
training scripts pick up automatically). Override with ``--output`` or with
the ``BOLDFLOW_CHECKPOINTS_DIR`` environment variable.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REVE_REPO = "brain-bzh/reve-base"
REVE_FILE = "model.safetensors"


def parse_args() -> argparse.Namespace:
    default_dir = os.environ.get("BOLDFLOW_CHECKPOINTS_DIR", "./checkpoints")
    p = argparse.ArgumentParser(description="Download REVE-base pretrained weights.")
    p.add_argument("--output", type=str, default=f"{default_dir}/reve-base.safetensors")
    p.add_argument("--repo", type=str, default=REVE_REPO,
                   help="HuggingFace repo id holding the weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    print(f"Downloading {args.repo}/{REVE_FILE} -> {output} ...")
    cached = Path(hf_hub_download(repo_id=args.repo, filename=REVE_FILE))
    if cached != output:
        # hf_hub_download returns a path inside the HF cache; copy (don't move)
        # so the cache stays consistent for future calls.
        shutil.copy(cached, output)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved {size_mb:.1f} MB to {output}")


if __name__ == "__main__":
    main()
