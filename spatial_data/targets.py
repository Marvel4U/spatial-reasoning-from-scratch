"""Grid targets for c0/c1 (Phase B)."""
from __future__ import annotations

import torch

from .c1_tasks import C1TaskSpec, Downsample

FOREGROUND_CLASS = 1


def c0_target_grid(
    grid_size: int,
    img_size: int,
    marker_row: int,
    marker_col: int,
    *,
    foreground_class: int = FOREGROUND_CLASS,
) -> torch.Tensor:
    """Single foreground cell at marker, downsampled to grid_size (256 px → 64 default)."""
    tgt = torch.zeros(grid_size, grid_size, dtype=torch.long)
    gr = marker_row * grid_size // img_size
    gc = marker_col * grid_size // img_size
    tgt[gr, gc] = foreground_class
    return tgt


def c1_positive_pixels(plane: torch.Tensor, task: C1TaskSpec) -> torch.Tensor:
    """Class plane (H, W) or (B, H, W) uint8 → bool same shape."""
    if task.key == "noise_ge_65":
        return plane >= 2
    return plane == task.class_id


def c1_mask_grid_from_plane(
    plane: torch.Tensor,
    task: C1TaskSpec,
    *,
    grid_size: int = 64,
    img_size: int = 256,
    foreground_class: int = FOREGROUND_CLASS,
) -> torch.Tensor:
    """One label plane (H, W) → (grid_size, grid_size) int targets."""
    pos = c1_positive_pixels(plane, task)
    return c1_mask_grid_from_pos(pos, task.downsample, grid_size=grid_size, img_size=img_size, foreground_class=foreground_class)


def c1_mask_grid_from_pos(
    pos: torch.Tensor,
    rule: Downsample,
    *,
    grid_size: int = 64,
    img_size: int = 256,
    foreground_class: int = FOREGROUND_CLASS,
) -> torch.Tensor:
    """Bool (H, W) positive pixels → binary grid (grid_size, grid_size)."""
    if pos.shape[0] != img_size or pos.shape[1] != img_size:
        raise ValueError(f"pos shape {tuple(pos.shape)} != ({img_size}, {img_size})")
    s = img_size // grid_size
    if s * grid_size != img_size:
        raise ValueError(f"img_size {img_size} not divisible by grid_size {grid_size}")
    blk = pos.reshape(grid_size, s, grid_size, s).permute(0, 2, 1, 3).reshape(grid_size, grid_size, s * s)
    cnt = blk.sum(dim=-1)
    if rule == "any":
        cell_pos = cnt > 0
    elif rule == "majority":
        cell_pos = cnt >= (s * s + 1) // 2
    elif rule == "centre":
        centre = pos[s // 2 :: s, s // 2 :: s]
        cell_pos = centre[:grid_size, :grid_size]
    else:
        raise ValueError(f"unknown downsample rule: {rule!r}")
    tgt = torch.zeros(grid_size, grid_size, dtype=torch.long)
    tgt[cell_pos] = foreground_class
    return tgt


def c1_surface_pure_cell_mask(
    surface_plane: torch.Tensor,
    *,
    grid_size: int = 64,
    img_size: int = 256,
) -> torch.Tensor:
    """(H, W) uint8 → (G, G) bool: all s×s pixels in cell share one surface class."""
    s = img_size // grid_size
    blk = surface_plane.reshape(grid_size, s, grid_size, s).permute(0, 2, 1, 3).reshape(grid_size, grid_size, s * s)
    return blk.max(dim=-1).values == blk.min(dim=-1).values
