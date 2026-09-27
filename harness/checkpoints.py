"""Checkpoint metadata helpers."""

def checkpoint_model_config(ckpt: dict) -> dict:
    return ckpt.get("model_config") or ckpt.get("config") or {}


def checkpoint_extra(ckpt: dict) -> dict:
    extra = dict(ckpt.get("extra") or {})
    legacy_cfg = ckpt.get("config") or {}
    extra.setdefault("code_version", legacy_cfg.get("code_version"))
    extra.setdefault("dtype", legacy_cfg.get("dtype"))
    return extra
