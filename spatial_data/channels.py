"""Worldsnap v2 label npz → model input tensor (Phase B)."""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

Encoding = Literal["scalar", "onehot", "onehot_v4"]
LAYER_KEYS = ("noise_256", "surface_256", "estab_256")
NUM_CLASSES_PER_LAYER = 4
NUM_NOISE7_CLASSES = 7


def _layer_arrays(npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(npz[k]) for k in LAYER_KEYS)


def scalar_layers_from_arrays(planes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor:
    out = np.stack(planes, axis=0).astype(np.float32)
    out /= NUM_CLASSES_PER_LAYER - 1
    return torch.from_numpy(out)


def scalar_layers_from_npz(npz) -> torch.Tensor:
    """Class indices 0..3 per layer → (3, H, W) float in [0, 1]."""
    return scalar_layers_from_arrays(_layer_arrays(npz))


def onehot_layers_from_arrays(planes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor:
    chunks = []
    for arr in planes:
        t = torch.from_numpy(arr.astype(np.int64))
        chunks.append(F.one_hot(t, NUM_CLASSES_PER_LAYER).permute(2, 0, 1).float())
    return torch.cat(chunks, dim=0)


def onehot_layers_from_npz(npz) -> torch.Tensor:
    """One-hot per layer → (12, H, W) float {0, 1}."""
    return onehot_layers_from_arrays(_layer_arrays(npz))


MARKER_RADIUS_PX = 0  # 0 = single pixel (c0 default). Set >0 to draw a disc; see c3 within_20m finding.


def marker_plane(height: int, width: int, row: int, col: int, radius: int | None = None) -> torch.Tensor:
    """Marker plane: a single pixel (radius 0) or a filled disc of ``radius`` px.

    Why a disc exists: a single marker pixel is one column out of 256 in the patch
    projection and is trained on 1/256 of the samples (the c0 starvation). For
    within_20m at P = 16 the model collapsed to all-background (val IoU 0.0). A small
    disc lights several columns per patch and survives the compression.
    """
    m = torch.zeros(1, height, width)
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError(f"marker ({row}, {col}) outside ({height}, {width})")
    r = MARKER_RADIUS_PX if radius is None else radius
    if r <= 0:
        m[0, row, col] = 1.0
        return m
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    m[0] = (((yy - row) ** 2 + (xx - col) ** 2) <= r * r).float()
    return m


def build_input_from_layers(
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    encoding: Encoding = "scalar",
    marker_row: int,
    marker_col: int,
) -> torch.Tensor:
    if encoding == "scalar":
        layers = scalar_layers_from_arrays(planes)
    elif encoding == "onehot":
        layers = onehot_layers_from_arrays(planes)
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")
    _, h, w = layers.shape
    return torch.cat([layers, marker_plane(h, w, marker_row, marker_col)], dim=0)


def build_input_from_npz(
    npz,
    *,
    encoding: Encoding = "scalar",
    marker_row: int,
    marker_col: int,
) -> torch.Tensor:
    """Full c0 input: layers + marker → (Cin, H, W)."""
    return build_input_from_layers(
        _layer_arrays(npz),
        encoding=encoding,
        marker_row=marker_row,
        marker_col=marker_col,
    )


def onehot_v4_layers_from_arrays(
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    """noise7 (7) + surface (4) + estab (4) -> (15, H, W)."""
    n7, surf, estab = planes
    chunks = [
        F.one_hot(torch.from_numpy(n7.astype(np.int64)), NUM_NOISE7_CLASSES).permute(2, 0, 1).float(),
        F.one_hot(torch.from_numpy(surf.astype(np.int64)), NUM_CLASSES_PER_LAYER).permute(2, 0, 1).float(),
        F.one_hot(torch.from_numpy(estab.astype(np.int64)), NUM_CLASSES_PER_LAYER).permute(2, 0, 1).float(),
    ]
    return torch.cat(chunks, dim=0)


def in_chans_for_encoding(encoding: Encoding) -> int:
    if encoding == "scalar":
        base = 3
    elif encoding == "onehot_v4":
        base = NUM_NOISE7_CLASSES + 2 * NUM_CLASSES_PER_LAYER
    else:
        base = 12
    return base + 1  # marker slot (zero for c1/c2/c3 grid tasks)
