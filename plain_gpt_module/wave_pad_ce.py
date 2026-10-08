"""Scratchpad-aware CE weights for C5 wave tasks (docs/memo/CHANGE_WEIGHTED_LOSS_2026-10-02.md)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_TARGET = -100
HALO_GRP = 4


def scratchpad_cells(img: torch.Tensor, grid: int) -> torch.Tensor | None:
    """(B, g, g) bool pad from channel 16; None if no scratchpad channel."""
    if img.shape[1] <= 16:
        return None
    s = img.shape[-1] // grid
    if s < 1:
        return None
    return F.max_pool2d(img[:, 16:17].float(), s, stride=s)[:, 0] > 0


def _dilate_cells(change: torch.Tensor, *, pad_px: int, img_side: int) -> torch.Tensor:
    """Expand (B,g,g) bool change by ``pad_px`` at full resolution, then pool back to cells."""
    if pad_px <= 0:
        return change
    b, g, _ = change.shape
    s = img_side // g
    x = change.float().unsqueeze(1)
    x = x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)
    k = 2 * pad_px + 1
    x = F.max_pool2d(x, k, stride=1, padding=pad_px)
    x = F.max_pool2d(x, s, stride=s)
    return x[:, 0] > 0


def change_group_cell_weights(
    img: torch.Tensor,
    targets: torch.Tensor,
    *,
    foreground_class: int,
    change_pad_px: int = 0,
) -> torch.Tensor | None:
    """Per-cell CE weights: each present stay/change/halo group contributes equally per sample."""
    pad = scratchpad_cells(img, targets.shape[1])
    if pad is None:
        return None
    fg = foreground_class
    valid = targets != IGNORE_TARGET
    tgt_fg = targets == fg
    core_change = (tgt_fg != pad) & valid
    grp = torch.where(valid, core_change.long() * 2 + tgt_fg.long(), torch.zeros_like(targets))
    n_grp = 5 if change_pad_px > 0 else 4
    if change_pad_px > 0:
        focus = _dilate_cells(core_change, pad_px=change_pad_px, img_side=img.shape[-1]) & valid
        halo = focus & ~core_change
        grp = torch.where(halo, torch.full_like(grp, HALO_GRP), grp)
    oh = F.one_hot(grp.clamp(0, n_grp - 1), n_grp).float() * valid.unsqueeze(-1).float()
    counts = oh.sum(dim=(1, 2))
    present = (counts > 0).sum(dim=1).clamp(min=1).float()
    scale = 1.0 / present
    cnt_at = torch.gather(counts, 1, grp.reshape(targets.shape[0], -1).clamp(0, n_grp - 1)).view_as(targets)
    w = torch.where(valid, scale.view(-1, 1, 1) / cnt_at.clamp(min=1.0), torch.zeros_like(cnt_at))
    return w


def change_beta_multiplier(
    img: torch.Tensor,
    targets: torch.Tensor,
    *,
    foreground_class: int,
    beta: float,
) -> torch.Tensor | None:
    """(B, g, g) float multiplier 1 + beta on cells where target != pad_in."""
    if beta <= 0:
        return None
    pad = scratchpad_cells(img, targets.shape[1])
    if pad is None:
        return None
    fg = foreground_class
    valid = targets != IGNORE_TARGET
    change = ((targets == fg) != pad) & valid
    return 1.0 + beta * change.float()
