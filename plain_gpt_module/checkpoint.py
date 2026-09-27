"""Save/load training checkpoints. Project-specific fields go in extra={}."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .build import unwrap_compiled

SCHEMA_VERSION = 1


def _model_config_dict(model, model_config) -> dict | None:
    if model_config is not None:
        if is_dataclass(model_config):
            return asdict(model_config)
        if isinstance(model_config, dict):
            return model_config
        raise TypeError("model_config must be a dataclass or dict")
    core = unwrap_compiled(model)
    if hasattr(core, "config") and is_dataclass(core.config):
        return asdict(core.config)
    return None


def save_checkpoint(
    path,
    model,
    *,
    optimizer=None,
    training_stats=None,
    model_config=None,
    extra=None,
    verbose=True,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = unwrap_compiled(model)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": core.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "training": training_stats,
        "extra": dict(extra or {}),
    }
    cfg = _model_config_dict(model, model_config)
    if cfg is not None:
        payload["model_config"] = cfg
    torch.save(payload, path)
    if verbose:
        print(f"Saved checkpoint to {path}")
        if training_stats:
            t = training_stats
            print(
                f"  steps={t.get('steps')}, train={t.get('train_loss'):.4f}, "
                f"val={t.get('val_loss'):.4f}, time={t.get('training_time_s', 0):.1f}s"
            )


def load_checkpoint(
    path,
    model,
    *,
    optimizer=None,
    map_location=None,
    strict=True,
    verbose=True,
    load_state_dict_fn=None,
) -> dict:
    ckpt = read_checkpoint(path, map_location=map_location)
    if ckpt.get("schema_version") != SCHEMA_VERSION and verbose:
        print(f"  warning: checkpoint schema_version={ckpt.get('schema_version')} "
              f"!= current {SCHEMA_VERSION}")
    state_dict = ckpt["model_state_dict"]
    if load_state_dict_fn is not None:
        load_state_dict_fn(model, state_dict)
    else:
        core = unwrap_compiled(model)
        core.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if verbose:
        training = ckpt.get("training")
        if training:
            print(f"Loaded checkpoint from {path}")
            print(
                f"  steps={training.get('steps')}, train={training.get('train_loss'):.4f}, "
                f"val={training.get('val_loss'):.4f}, time={training.get('training_time_s', 0):.1f}s"
            )
        else:
            print(f"Loaded checkpoint from {path} (no training stats)")
    return ckpt


def read_checkpoint(path, map_location=None) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
