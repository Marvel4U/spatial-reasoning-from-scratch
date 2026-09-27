"""GPU c3 targets from v4 planes (noise7, surface, estab)."""
from __future__ import annotations

import torch

from .c3_tasks import (
    C3TaskSpec,
    MIN_SW_PX,
    REL_MEDIAN_STRICT,
    REL_QUIETEST_TERTILE,
    REL_WITHIN_IDS,
    WITHIN_M_BY_REL_ID,
)


def _atom_pix_mask(layers_u8: torch.Tensor, atom_id: int) -> torch.Tensor:
    layer_ix = atom_id // 4
    cls = atom_id % 4
    if layer_ix == 0:
        raise ValueError("c3 v4 targets do not support noise-layer atoms in conjunctions")
    plane = {1: 1, 2: 2}[layer_ix]
    return layers_u8[:, plane] == cls


def _cells_from_pix(pix: torch.Tensor, *, grid_size: int = 64, foreground_class: int = 1) -> torch.Tensor:
    b, h, w = pix.shape
    g = grid_size
    s = h // g
    blk = pix.reshape(b, g, s, g, s).permute(0, 1, 3, 2, 4).reshape(b, g, g, s * s)
    tgt = torch.zeros(b, g, g, dtype=torch.long, device=pix.device)
    tgt[blk.sum(-1) >= (s * s + 1) // 2] = foreground_class
    return tgt


def _median_strict_pix(n7: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    b, h, w = n7.shape
    pix = torch.zeros(b, h, w, dtype=torch.bool, device=n7.device)
    for bi in range(b):
        v = n7[bi][sw[bi]]
        if v.numel() < MIN_SW_PX:
            continue
        med = v.median().to(dtype=n7.dtype)
        pix[bi] = sw[bi] & (n7[bi] < med)
    return pix


def _tertile_pix(n7: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    b, h, w = n7.shape
    pix = torch.zeros(b, h, w, dtype=torch.bool, device=n7.device)
    for bi in range(b):
        mask = sw[bi]
        v = n7[bi][mask]
        if v.numel() < MIN_SW_PX:
            continue
        order = torch.argsort(v, stable=True)
        k = max(1, v.numel() // 3)
        pick = order[:k]
        flat = torch.zeros(mask.sum(), dtype=torch.bool, device=n7.device)
        flat[pick] = True
        row = pix[bi]
        row[mask] = flat
    return pix


def _within_radius_pix(
    marker_col: torch.Tensor,
    marker_row: torch.Tensor,
    radius: float,
    *,
    h: int,
    w: int,
) -> torch.Tensor:
    # (B,) single marker or (B,K) several; -1 marks an unused slot. A pixel is inside if it
    # lies within `radius` of ANY marker (union of discs; the loader keeps them >= 2R apart).
    mc = marker_col.view(marker_col.shape[0], -1).float()
    mr = marker_row.view(marker_row.shape[0], -1).float()
    b, k = mc.shape
    dev = mc.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij",
    )
    dx = xx.view(1, 1, h, w) - mc.view(b, k, 1, 1)
    dy = yy.view(1, 1, h, w) - mr.view(b, k, 1, 1)
    inside = (dx * dx + dy * dy <= radius * radius) & (mc >= 0).view(b, k, 1, 1)
    return inside.any(dim=1)


def c3_targets_from_planes(
    layers_u8: torch.Tensor,
    relative_cond_ids: torch.Tensor,
    *,
    grid_size: int = 64,
    foreground_class: int = 1,
    marker_col: torch.Tensor | None = None,
    marker_row: torch.Tensor | None = None,
    radius_m: int | None = None,
) -> torch.Tensor:
    """(B,3,H,W) with [noise7,surface,estab] + cond op id -> (B,G,G) long."""
    n7 = layers_u8[:, 0]
    surf = layers_u8[:, 1]
    sw = surf == 2
    b, _, h, w = layers_u8.shape
    pix = torch.zeros(b, h, w, dtype=torch.bool, device=layers_u8.device)
    for bi in range(b):
        rid = int(relative_cond_ids[bi].item())
        if rid in REL_WITHIN_IDS:
            if marker_col is None or marker_row is None:
                raise ValueError("within disc requires marker_col, marker_row")
            rm = radius_m if radius_m is not None else WITHIN_M_BY_REL_ID[rid]
            pix[bi] = _within_radius_pix(
                marker_col[bi : bi + 1], marker_row[bi : bi + 1], float(rm), h=h, w=w,
            )[0]
        elif rid == REL_MEDIAN_STRICT:
            pix[bi] = _median_strict_pix(n7[bi : bi + 1], sw[bi : bi + 1])[0]
        elif rid == REL_QUIETEST_TERTILE:
            pix[bi] = _tertile_pix(n7[bi : bi + 1], sw[bi : bi + 1])[0]
        else:
            raise ValueError(f"unknown relative cond id {rid}")
    return _cells_from_pix(pix, grid_size=grid_size, foreground_class=foreground_class)


def c3_targets_for_spec(
    layers_u8: torch.Tensor,
    spec: C3TaskSpec,
    *,
    grid_size: int = 64,
    marker_col: torch.Tensor | None = None,
    marker_row: torch.Tensor | None = None,
) -> torch.Tensor:
    if spec.absolute_atom_id is not None:
        pix = _atom_pix_mask(layers_u8, spec.absolute_atom_id)
        return _cells_from_pix(pix, grid_size=grid_size)
    if spec.conjunction_atom_id is not None:
        if marker_col is None or marker_row is None or spec.radius_m is None:
            raise ValueError("conjunctive marker task requires marker_col, marker_row, radius_m")
        h, w = layers_u8.shape[-2], layers_u8.shape[-1]
        disc = _within_radius_pix(marker_col, marker_row, float(spec.radius_m), h=h, w=w)
        pix = disc & _atom_pix_mask(layers_u8, spec.conjunction_atom_id)
        return _cells_from_pix(pix, grid_size=grid_size)
    rel = torch.full((layers_u8.shape[0],), spec.relative_cond_id, dtype=torch.long, device=layers_u8.device)
    return c3_targets_from_planes(
        layers_u8,
        rel,
        grid_size=grid_size,
        marker_col=marker_col,
        marker_row=marker_row,
        radius_m=spec.radius_m,
    )
