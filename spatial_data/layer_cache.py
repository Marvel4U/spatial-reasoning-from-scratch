"""In-RAM cache of v2 label npz layer rasters (shared across c0 items on the same crop)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .channels import LAYER_KEYS

LayerPlanes = tuple[np.ndarray, np.ndarray, np.ndarray]


class LayerCache:
    def __init__(self) -> None:
        self._store: dict[Path, LayerPlanes] = {}

    def __len__(self) -> int:
        return len(self._store)

    def get(self, path: Path) -> LayerPlanes:
        key = path.resolve()
        hit = self._store.get(key)
        if hit is not None:
            return hit
        with np.load(key) as npz:
            planes = tuple(np.asarray(npz[k]) for k in LAYER_KEYS)
        self._store[key] = planes
        return planes

    def warm(self, paths: set[Path] | list[Path], *, log=None) -> None:
        unique = {p.resolve() for p in paths}
        missing = [p for p in unique if p not in self._store]
        if not missing:
            if log:
                log(f"  layer_cache: {len(unique)} npz already warm")
            return
        t0 = time.perf_counter()
        bytes_est = 0
        for p in missing:
            planes = self.get(p)
            bytes_est += sum(a.nbytes for a in planes)
        if log:
            log(
                f"  layer_cache: loaded {len(missing)} npz "
                f"({bytes_est / 1e6:.0f} MB layers, {time.perf_counter() - t0:.1f}s)"
            )
