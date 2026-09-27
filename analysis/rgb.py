"""RGB composite from v2 label planes (matches worldsnap crop renders)."""
from __future__ import annotations

import numpy as np

GREY = (0, 85, 170, 255)
LAYER_KEYS = ("noise_256", "surface_256", "estab_256")


def rgb_from_label_planes(
    noise: np.ndarray,
    surface: np.ndarray,
    estab: np.ndarray,
) -> np.ndarray:
    """uint8 H×W×3, R=noise G=surface B=estab."""
    planes = (noise, surface, estab)
    chans = [np.take(GREY, p.astype(np.int64)) for p in planes]
    return np.dstack(chans).astype(np.uint8)
