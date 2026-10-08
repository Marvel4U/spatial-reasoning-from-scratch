"""Save/load training checkpoints. Project-specific fields go in extra={}."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .build import unwrap_compiled
from .patch_embed import expand_proj_weight

SCHEMA_VERSION = 1

_PATCH_EMBED_WEIGHT_KEY = "patch_embed.proj.weight"


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


def _infer_in_chans_from_patch_weight(weight: torch.Tensor, patch_size: int) -> int:
    pp = patch_size * patch_size
    if weight.ndim != 2 or weight.shape[1] % pp != 0:
        raise ValueError(f"patch_embed.proj.weight shape {tuple(weight.shape)} invalid for patch_size={patch_size}")
    return weight.shape[1] // pp


def adapt_state_dict_patch_embed_in_chans(model, state_dict: dict, *, verbose: bool = True) -> dict:
    """Copy ``state_dict`` and widen ``patch_embed.proj.weight`` if the model has more input channels.

    C5 M1: load a 16-channel checkpoint into a 17-channel model (scratchpad plane zero-init).
    """
    out = dict(state_dict)
    w = out.get(_PATCH_EMBED_WEIGHT_KEY)
    if w is None:
        return out
    core = unwrap_compiled(model)
    pe = core.patch_embed
    old_in = _infer_in_chans_from_patch_weight(w, pe.patch_size)
    new_in = pe.in_chans
    if old_in == new_in:
        return out
    if old_in > new_in:
        raise ValueError(
            f"checkpoint patch_embed has in_chans={old_in}, model expects {new_in} (cannot shrink)"
        )
    out[_PATCH_EMBED_WEIGHT_KEY] = expand_proj_weight(
        w, old_in_chans=old_in, new_in_chans=new_in, patch_size=pe.patch_size,
    )
    if verbose:
        print(f"  checkpoint: expanded patch_embed {old_in} → {new_in} input channels (new plane zero-init)")
    return out


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
        state_dict = adapt_state_dict_patch_embed_in_chans(model, state_dict, verbose=verbose)
        model_sd = core.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        for k, v in state_dict.items():
            if k not in model_sd:
                continue
            if model_sd[k].shape != v.shape:
                skipped_shape.append(k)
                continue
            filtered[k] = v
        if skipped_shape and verbose:
            print(f"  checkpoint: skipped {len(skipped_shape)} keys (shape mismatch), e.g. {skipped_shape[:3]}")
        missing = [k for k in model_sd if k not in filtered]
        if missing and verbose and strict:
            print(f"  checkpoint: {len(missing)} model keys not in ckpt (random init), e.g. {missing[:5]}")
        core.load_state_dict(filtered, strict=False)
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
