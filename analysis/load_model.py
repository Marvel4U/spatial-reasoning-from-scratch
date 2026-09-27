"""Load LocalGridViT from harness checkpoint (no compile)."""
from __future__ import annotations

from pathlib import Path

import torch

from plain_gpt_module.checkpoint import load_checkpoint, read_checkpoint


def _model_config_from_ckpt(ckpt: dict) -> dict:
    return ckpt.get("model_config") or ckpt.get("config") or {}
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT


def load_vit_checkpoint(
    path: str | Path,
    *,
    device: torch.device | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[LocalGridViT, LocalGridViTConfig, dict]:
    path = Path(path)
    ckpt = read_checkpoint(path, map_location=map_location)
    cfg_dict = _model_config_from_ckpt(ckpt)
    cfg = LocalGridViTConfig(**cfg_dict)
    model = LocalGridViT(cfg)
    load_checkpoint(path, model, map_location=map_location, verbose=False)
    model.eval()
    if device is not None:
        model.to(device)
    return model, cfg, ckpt
