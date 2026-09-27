"""c4 route tasks: segment (straight line) and detour (path pixels from store)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .c2_tasks import fg_ce_weight

STATS_FILENAME = "c4_task_stats.json"
GRID = 64
IMG = 256
FOREGROUND = 1

# cond_emb index = relative_cond_id + 1 (same convention as c3).
REL_SEGMENT = 18
REL_DETOUR = 19
REL_BUILDING = 20  # dense helper: mark building cells, markers ignored (C4_PLAN §10 addendum)

TASK_KEYS: tuple[str, ...] = ("segment", "detour")
HELPER_KEYS: tuple[str, ...] = ("building",)
C4_MIX_L8B: tuple[str, ...] = ("within_20m", "segment", "detour")

HUMAN_NAMES = {
    "segment": "straight segment between two markers",
    "detour": "shortest path around buildings between two markers",
    "building": "building cells (crop-wide, markers ignored)",
}

DETOUR_STRATA: tuple[tuple[float, float], ...] = ((1.0, 1.05), (1.05, 1.3), (1.3, 99.0))
STRATUM_LABELS = ("ratio_1.00_1.05", "ratio_1.05_1.30", "ratio_gt_1.30")
M_PER_CELL = 256.0 / GRID


def stratum_from_detour_ratio(r: float) -> int:
    for i, (lo, hi) in enumerate(DETOUR_STRATA):
        if lo <= r < hi:
            return i
    return len(DETOUR_STRATA) - 1


class C4TargetKind(str, Enum):
    SEGMENT = "segment"
    DETOUR_PATH = "detour_path"
    BUILDING = "building"


@dataclass(frozen=True)
class C4TaskSpec:
    key: str
    relative_cond_id: int
    name: str
    target_kind: C4TargetKind

    @property
    def show_marker_plane(self) -> bool:
        return True

    def padded_cond_ids(self) -> tuple[int, int]:
        return (-1, self.relative_cond_id)


def n_cond_emb_for_c4() -> int:
    return max(REL_SEGMENT, REL_DETOUR, REL_BUILDING) + 2


def get_c4_task(key: str) -> C4TaskSpec:
    if key == "segment":
        return C4TaskSpec(key=key, relative_cond_id=REL_SEGMENT, name=HUMAN_NAMES[key],
                          target_kind=C4TargetKind.SEGMENT)
    if key == "detour":
        return C4TaskSpec(key=key, relative_cond_id=REL_DETOUR, name=HUMAN_NAMES[key],
                          target_kind=C4TargetKind.DETOUR_PATH)
    if key == "building":
        return C4TaskSpec(key=key, relative_cond_id=REL_BUILDING, name=HUMAN_NAMES[key],
                          target_kind=C4TargetKind.BUILDING)
    raise ValueError(f"unknown c4 task {key!r}")


def c4_task_specs_for_keys(keys: list[str]) -> list[C4TaskSpec]:
    return [get_c4_task(k) for k in keys]


def rasterise_segment_numpy(a: np.ndarray, b: np.ndarray, *, crop_px: int = IMG) -> np.ndarray:
    """Unique (x, y) pixels along segment a->b (same sampling as c4_routes.rasterise)."""
    import math
    poly = np.stack([a, b], axis=0)
    out = []
    for p, q in zip(poly[:-1], poly[1:]):
        L = max(1, int(math.ceil(float(np.linalg.norm(q - p)) * 2)))
        t = np.linspace(0.0, 1.0, L + 1)[:, None]
        out.append(p[None] + t * (q - p)[None])
    pts = np.floor(np.vstack(out)).astype(np.int64)
    pts = np.clip(pts, 0, crop_px - 1)
    _, first = np.unique(pts, axis=0, return_index=True)
    return pts[np.sort(first)]


def path_pix_to_grid_numpy(path_xy: np.ndarray, *, grid: int = GRID, img: int = IMG) -> np.ndarray:
    """(n, 2) int x,y -> (grid, grid) bool, any pixel in 4×4 cell."""
    g, s = grid, img // grid
    cells = np.zeros((g, g), dtype=bool)
    if len(path_xy) == 0:
        return cells
    xs, ys = path_xy[:, 0], path_xy[:, 1]
    gc = (xs // s).astype(np.int64)
    gr = (ys // s).astype(np.int64)
    ok = (gc >= 0) & (gc < g) & (gr >= 0) & (gr < g)
    cells[gr[ok], gc[ok]] = True
    return cells


def fg_ce_weights_for_key(key: str, stats: dict, *, alpha: float = 0.5) -> tuple[float, ...]:
    return (fg_ce_weight(key, stats, alpha=alpha),)


def compute_task_stats(routes_dir: Path) -> dict:
    """Positive cell share per task from train split route store."""
    path = routes_dir / "c4_routes_train.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    markers = z["markers"]
    path_pix = z["path_pix"]
    n_pix = z["n_pix"]
    n = markers.shape[0]
    seg_cells, det_cells = [], []
    for i in range(n):
        a, b = markers[i, 0], markers[i, 1]
        seg = rasterise_segment_numpy(a, b)
        seg_cells.append(path_pix_to_grid_numpy(seg))
        det_cells.append(path_pix_to_grid_numpy(path_pix[i, : int(n_pix[i])]))
    seg_stack = np.stack(seg_cells)
    det_stack = np.stack(det_cells)
    tasks = {}
    for key, stack in (("segment", seg_stack), ("detour", det_stack)):
        per = stack.reshape(n, -1).mean(1)
        tasks[key] = {
            "f": float(stack.mean()),
            "empty_share": float((per == 0).mean()),
            "p10": float(np.percentile(per, 10)),
            "p50": float(np.percentile(per, 50)),
            "p90": float(np.percentile(per, 90)),
        }
    district = Path(routes_dir).parent / "district_labels_1m.npz"
    if district.is_file():
        surf = np.load(district)["surface"]
        h, w = (surf.shape[0] // 4) * 4, (surf.shape[1] // 4) * 4
        blk = (surf[:h, :w] == 3).reshape(h // 4, 4, w // 4, 4).sum(axis=(1, 3))
        cells = blk >= 8
        tasks["building"] = {"f": float(cells.mean()), "empty_share": 0.0, "p10": float(cells.mean()),
                             "p50": float(cells.mean()), "p90": float(cells.mean()),
                             "rule": "surface == 3 majority (>= 8 of 16 px) per cell, district-wide"}
    return {
        "routes_dir": str(routes_dir),
        "n_routes": n,
        "rule": "c4 route cells; any pixel in 4×4 block",
        "tasks": tasks,
    }


def load_task_stats(routes_dir: Path, *, rebuild: bool = False) -> dict:
    path = Path(routes_dir) / STATS_FILENAME
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    stats = compute_task_stats(routes_dir)
    path.write_text(json.dumps(stats, indent=1))
    return stats
