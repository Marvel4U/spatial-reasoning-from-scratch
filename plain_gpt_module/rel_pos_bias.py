"""2-D relative position bucket indices for Swin-style attention bias (C3 E5b)."""
from __future__ import annotations

import torch


def build_rel_pos_bucket_index(
    *,
    patch_grid: int,
    n_prefix: int,
    max_offset: int,
) -> torch.Tensor:
    """Return (T, T) long indices: patch pairs → clipped (Δrow, Δcol) bucket; prefix → last bucket."""
    if patch_grid < 1:
        raise ValueError(f"patch_grid must be >= 1, got {patch_grid}")
    if n_prefix < 0:
        raise ValueError(f"n_prefix must be >= 0, got {n_prefix}")
    if max_offset < 0:
        raise ValueError(f"max_offset must be >= 0, got {max_offset}")
    g = patch_grid
    k = max_offset
    n_spatial = (2 * k + 1) ** 2
    prefix_bucket = n_spatial
    n_patch = g * g
    t_len = n_prefix + n_patch
    idx = torch.full((t_len, t_len), prefix_bucket, dtype=torch.long)
    if n_patch == 0:
        return idx
    pr = torch.arange(n_patch, dtype=torch.long) // g
    pc = torch.arange(n_patch, dtype=torch.long) % g
    dr = (pr.view(n_patch, 1) - pr.view(1, n_patch)).clamp(-k, k) + k
    dc = (pc.view(n_patch, 1) - pc.view(1, n_patch)).clamp(-k, k) + k
    patch_buckets = dr * (2 * k + 1) + dc
    p0 = n_prefix
    idx[p0:, p0:] = patch_buckets
    return idx
