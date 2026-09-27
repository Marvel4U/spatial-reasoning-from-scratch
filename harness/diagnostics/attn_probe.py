"""Attention probe helpers (entropy, sidewalk mass, self-attn ablation hooks)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from plain_gpt_module.local_config import LocalGridViTConfig


def surface_onehot_slice(in_chans: int) -> slice:
    if in_chans == 16:
        return slice(7, 11)
    if in_chans == 13:
        return slice(4, 8)
    raise ValueError(f"surface_onehot_slice: unsupported in_chans={in_chans}")


def sidewalk_patch_mask(img: torch.Tensor, cfg: LocalGridViTConfig) -> torch.Tensor:
    """(B, C, H, W) -> (B, num_patches) bool at patch centres."""
    g, ps = cfg.patch_grid, cfg.patch_size
    surf = img[:, surface_onehot_slice(cfg.in_chans)].argmax(dim=1)
    pr = torch.arange(g, device=img.device) * ps + ps // 2
    pc = torch.arange(g, device=img.device) * ps + ps // 2
    gy, gx = torch.meshgrid(pr, pc, indexing="ij")
    return surf[:, gy, gx].reshape(img.shape[0], -1) == 2


def full_seq_sidewalk_mask(
    img: torch.Tensor, cfg: LocalGridViTConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Key mask (B, T) bool sidewalk; sw_frac (B,) patch-only fraction."""
    sw = sidewalk_patch_mask(img, cfg)
    n_pre = cfg.n_cond_token_slots
    t = n_pre + cfg.num_patches
    full = torch.zeros(sw.shape[0], t, dtype=torch.bool, device=img.device)
    full[:, n_pre:] = sw
    return full, sw.float().mean(dim=1)


def attention_matrix(x: torch.Tensor, attn_module) -> torch.Tensor:
    """(B,T,C) -> (B, nh, T, T) softmax weights (matches probe_metrics)."""
    b, t, c = x.shape
    nh = attn_module.n_head
    qkv = attn_module.c_attn(x)
    q, k, _v = qkv.split(c, dim=2)
    q = q.view(b, t, nh, c // nh).transpose(1, 2)
    k = k.view(b, t, nh, c // nh).transpose(1, 2)
    return F.softmax(q @ k.transpose(-2, -1) / math.sqrt(c // nh), dim=-1)


def sidewalk_attention_scalars(
    att: torch.Tensor,
    sw_full: torch.Tensor,
    sw_frac: torch.Tensor,
) -> dict[str, float]:
    """att (B,nh,T,T); sw_full (B,T) keys; mean mass and ratio vs uniform."""
    b, nh, t, _ = att.shape
    sw_k = sw_full.unsqueeze(1).unsqueeze(2).expand(b, nh, t, t)
    mass = (att * sw_k.float()).sum(-1)
    ent = -(att * att.clamp_min(1e-12).log()).sum(-1).mean(dim=(0, 2))
    return {
        "mass_sidewalk": mass.mean().item(),
        "mass_vs_uniform": (mass.mean(dim=(1, 2)) / sw_frac.clamp(min=1e-6)).mean().item(),
        "entropy_mean": ent.mean().item(),
    }


def layer_probe_scalars(
    att: torch.Tensor,
    sw_full: torch.Tensor,
    sw_frac: torch.Tensor,
    layer: int,
    *,
    prefix: str = "",
) -> dict[str, float]:
    s = sidewalk_attention_scalars(att, sw_full, sw_frac)
    ent = -(att * att.clamp_min(1e-12).log()).sum(-1).mean(dim=(0, 2))
    out: dict[str, float] = {
        f"{prefix}attn_sw_mass/{layer}": s["mass_sidewalk"],
        f"{prefix}attn_sw_ratio/{layer}": s["mass_vs_uniform"],
        f"{prefix}attn_entropy_mean/{layer}": s["entropy_mean"],
    }
    for h in range(ent.shape[0]):
        out[f"{prefix}attn_entropy/{layer}/{h}"] = ent[h].item()
    return out
