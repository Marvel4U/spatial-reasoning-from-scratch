"""Semantic parameter groups (ANALYSIS_SUITE_SPEC §4)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from plain_gpt_module.local_config import LocalGridViTConfig


@dataclass(frozen=True)
class GroupSpec:
    key: str
    param: nn.Parameter
    channel: int | None  # patch_embed slice index


def view_group_tensor(t: torch.Tensor, channel: int | None, cfg: LocalGridViTConfig) -> torch.Tensor:
    f = t.detach().float()
    if channel is None:
        return f
    pp = cfg.patch_size ** 2
    return f.view(cfg.n_embd, cfg.in_chans, pp)[:, channel]


def build_semantic_groups(
    model: nn.Module,
    cfg: LocalGridViTConfig,
    channel_names: list[str],
) -> list[GroupSpec]:
    if len(channel_names) != cfg.in_chans:
        raise ValueError(f"channel_names len {len(channel_names)} != in_chans {cfg.in_chans}")
    out: list[GroupSpec] = []
    for name, p in model.named_parameters():
        if p.ndim < 2 or not p.requires_grad:
            continue
        if name == "patch_embed.proj.weight":
            for c, cn in enumerate(channel_names):
                out.append(GroupSpec(f"patch_embed/{cn}", p, c))
        elif name.endswith(".weight"):
            out.append(GroupSpec(name.removesuffix(".weight"), p, None))
    return out


def patch_embed_decomposition(cfg: LocalGridViTConfig, weight: torch.Tensor, channel_names: list[str]) -> dict[str, float]:
    w = weight.detach().float().view(cfg.n_embd, cfg.in_chans, cfg.patch_size ** 2)
    row: dict[str, float] = {}
    for c, cn in enumerate(channel_names):
        cols = w[:, c]
        shared = cols.mean(dim=1, keepdim=True)
        resid = cols - shared
        row[f"patch_shared/{cn}"] = shared.norm().item()
        row[f"patch_resid/{cn}"] = resid.norm(dim=0).mean().item()
    return row
