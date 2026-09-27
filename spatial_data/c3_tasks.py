"""c3 relative tasks (C3_PLAN §4): quieter sidewalk on v4 noise7 planes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .c2_tasks import atom_id, fg_ce_weight

C3_LAYER_KEYS = ("noise7", "surface", "estab")
NUM_NOISE7 = 7
STATS_FILENAME = "c3_task_stats.json"
MIN_SW_PX = 50
GRID = 64
MAJ = 8

# Second cond-token slot: relative op (not a layer atom). cond_emb index = id + 1.
REL_MEDIAN_STRICT = 12
REL_QUIETEST_TERTILE = 13
REL_WITHIN_20M = 14
REL_WITHIN_50M = 15
REL_WITHIN_100M = 16
REL_WITHIN_75M = 17
WITHIN_RADIUS_M = 20  # legacy alias

WITHIN_RADII_M: tuple[int, ...] = (20, 50, 75, 100)
WITHIN_M_BY_REL_ID: dict[int, int] = {
    REL_WITHIN_20M: 20,
    REL_WITHIN_50M: 50,
    REL_WITHIN_75M: 75,
    REL_WITHIN_100M: 100,
}
WITHIN_REL_ID_BY_M: dict[int, int] = {m: rid for rid, m in WITHIN_M_BY_REL_ID.items()}
REL_WITHIN_IDS = frozenset(WITHIN_M_BY_REL_ID.keys())

LOCAL_CEILING = {
    "median_strict": 0.762,
    "quietest_tertile": 0.588,
}

RELATIVE_TASK_KEYS: tuple[str, ...] = ("median_strict", "quietest_tertile")


def within_task_key(radius_m: int) -> str:
    return f"within_{radius_m}m"


MARKER_TASK_KEYS: tuple[str, ...] = tuple(within_task_key(r) for r in WITHIN_RADII_M)
M1_TASK_KEY = "food_drink_within_100m"
M1_TASK_KEYS: tuple[str, ...] = (M1_TASK_KEY,)
DENSE_ESTAB_TASK_KEYS: tuple[str, ...] = ("food_drink",)
TASK_KEYS: tuple[str, ...] = RELATIVE_TASK_KEYS + MARKER_TASK_KEYS + DENSE_ESTAB_TASK_KEYS + M1_TASK_KEYS
ESTAB_FOOD_DRINK_ATOM = atom_id("estab", 3)
N5_MIX_TASK_KEYS: tuple[str, ...] = ("food_drink", "within_100m", M1_TASK_KEY)

HUMAN_NAMES = {
    "median_strict": "quieter sidewalk (strictly below crop median)",
    "quietest_tertile": "quietest third of sidewalk by rank",
    **{within_task_key(r): f"within {r} m of the marker" for r in WITHIN_RADII_M},
    "food_drink": "food & drink establishments (crop-wide)",
    M1_TASK_KEY: "food & drink within 100 m of the marker",
}


def n_cond_emb_for_c3() -> int:
    """Pad slot 0 + cond atom ids through max REL_WITHIN_* id."""
    return max(WITHIN_M_BY_REL_ID.keys()) + 2


def within_train_markers(radius_m: int) -> tuple[int, float]:
    """(K, min marker center distance in px) for train-only multi-marker (C3_PLAN N4).

    Val is always one fixed t0 marker. K>1 is train-only union-of-discs surrogate.
    At 75/100 m a single disc already covers much of the 256 m crop — use K=1 there.
    """
    if radius_m <= 20:
        return 4, 40.0
    if radius_m <= 50:
        return 2, float(min(2 * radius_m, 200))
    return 1, 0.0


@dataclass(frozen=True)
class C3TaskSpec:
    key: str
    relative_cond_id: int
    name: str
    radius_m: int | None = None
    conjunction_atom_id: int | None = None  # m1: layer atom in slot 0 (e.g. estab food & drink)
    absolute_atom_id: int | None = None  # dense c1-style layer mask; slot 1 = -1 (always true)

    @property
    def uses_marker(self) -> bool:
        return self.radius_m is not None

    @property
    def show_marker_plane(self) -> bool:
        """Draw t0 marker on input (N5 mix: even dense estab task keeps marker visible)."""
        return self.uses_marker or self.absolute_atom_id is not None

    def padded_cond_ids(self) -> tuple[int, int]:
        if self.absolute_atom_id is not None:
            return (self.absolute_atom_id, -1)
        if self.uses_marker and self.conjunction_atom_id is not None:
            return (self.conjunction_atom_id, self.relative_cond_id)
        if self.uses_marker:
            return (-1, self.relative_cond_id)
        return (atom_id("surface", 2), self.relative_cond_id)


def get_c3_task(key: str) -> C3TaskSpec:
    if key == "median_strict":
        return C3TaskSpec(key=key, relative_cond_id=REL_MEDIAN_STRICT, name=HUMAN_NAMES[key])
    if key == "quietest_tertile":
        return C3TaskSpec(key=key, relative_cond_id=REL_QUIETEST_TERTILE, name=HUMAN_NAMES[key])
    if key.startswith("within_") and key.endswith("m"):
        r = int(key[len("within_") : -1])
        if r not in WITHIN_REL_ID_BY_M:
            raise ValueError(f"unsupported within radius {r!r}")
        rid = WITHIN_REL_ID_BY_M[r]
        return C3TaskSpec(key=key, relative_cond_id=rid, name=HUMAN_NAMES[key], radius_m=r)
    if key == "food_drink":
        return C3TaskSpec(
            key=key,
            relative_cond_id=REL_MEDIAN_STRICT,
            name=HUMAN_NAMES[key],
            absolute_atom_id=ESTAB_FOOD_DRINK_ATOM,
        )
    if key == M1_TASK_KEY:
        return C3TaskSpec(
            key=key,
            relative_cond_id=REL_WITHIN_100M,
            name=HUMAN_NAMES[key],
            radius_m=100,
            conjunction_atom_id=ESTAB_FOOD_DRINK_ATOM,
        )
    raise ValueError(f"unknown c3 task {key!r}")


def c3_task_specs_for_keys(keys: list[str]) -> list[C3TaskSpec]:
    return [get_c3_task(k) for k in keys]


def _rank_tertile(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    n = len(v)
    k = max(1, n // 3)
    low = np.zeros(n, dtype=bool)
    low[order[:k]] = True
    return low


def pixel_target_numpy(
    noise7: np.ndarray, surface: np.ndarray, rule: str,
) -> np.ndarray | None:
    """Full-crop bool pixel mask, or None if degenerate (stats skip only)."""
    sw = surface == 2
    if sw.sum() < MIN_SW_PX:
        return None
    v = noise7
    vv = v[sw]
    if rule == "median_strict":
        med = int(np.median(vv))
        pix = sw & (v < med)
    elif rule == "quietest_tertile":
        low = np.zeros(sw.shape, dtype=bool)
        low[sw] = _rank_tertile(vv)
        pix = low
    else:
        raise ValueError(rule)
    if not pix.any():
        return None
    return pix


def cell_target_numpy(noise7: np.ndarray, surface: np.ndarray, rule: str) -> np.ndarray:
    """(64, 64) bool cells; all-false grid when crop is empty/degenerate."""
    pix = pixel_target_numpy(noise7, surface, rule)
    if pix is None:
        return np.zeros((GRID, GRID), dtype=bool)
    return _cells_from_pix_numpy(pix)


def cell_target_within_numpy(
    marker_col: int, marker_row: int, *, radius_m: int = WITHIN_RADIUS_M, h: int = 256, w: int = 256,
) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - marker_col) ** 2 + (yy - marker_row) ** 2)
    return _cells_from_pix_numpy(dist <= radius_m)


def cell_target_food_drink_within_100m_numpy(
    estab: np.ndarray,
    marker_col: int,
    marker_row: int,
    *,
    radius_m: int = 100,
    h: int = 256,
    w: int = 256,
) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - marker_col) ** 2 + (yy - marker_row) ** 2)
    pix = (dist <= radius_m) & (estab == 3)
    return _cells_from_pix_numpy(pix)


def _cells_from_pix_numpy(pix: np.ndarray) -> np.ndarray:
    blk = pix.reshape(GRID, 4, GRID, 4).transpose(0, 2, 1, 3).reshape(GRID, GRID, 16)
    return blk.sum(-1) >= MAJ


def _load_train_planes(crops_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((crops_dir / "train").glob("*_labels.npz"))
    if not files:
        raise FileNotFoundError(f"no train crops under {crops_dir}")
    n7 = np.stack([np.load(f)["noise7_256"] for f in files])
    surf = np.stack([np.load(f)["surface_256"] for f in files])
    return n7, surf


def compute_task_stats(crops_dir: Path, district_dir: Path | None = None) -> dict:
    n7, surf = _load_train_planes(crops_dir)
    n = n7.shape[0]
    tasks: dict[str, dict] = {}
    for key in RELATIVE_TASK_KEYS:
        cells = np.stack([cell_target_numpy(n7[i], surf[i], key) for i in range(n)])
        per = cells.reshape(n, -1).mean(1)
        tasks[key] = {
            "f": float(cells.mean()),
            "empty_share": float((per == 0).mean()),
            "p10": float(np.percentile(per, 10)),
            "p50": float(np.percentile(per, 50)),
            "p90": float(np.percentile(per, 90)),
        }
    if district_dir is not None:
        from .c3_markers import marker_table_for_split
        cols, rows = marker_table_for_split(crops_dir, district_dir, "train")
        estab = np.stack([np.load(f)["estab_256"] for f in sorted((crops_dir / "train").glob("*_labels.npz"))])
        for r in WITHIN_RADII_M:
            key = within_task_key(r)
            cells = np.stack([
                cell_target_within_numpy(int(cols[i]), int(rows[i]), radius_m=r) for i in range(n)
            ])
            per = cells.reshape(n, -1).mean(1)
            tasks[key] = {
                "f": float(cells.mean()),
                "empty_share": float((per == 0).mean()),
                "p10": float(np.percentile(per, 10)),
                "p50": float(np.percentile(per, 50)),
                "p90": float(np.percentile(per, 90)),
            }
        key = M1_TASK_KEY
        cells = np.stack([
            cell_target_food_drink_within_100m_numpy(estab[i], int(cols[i]), int(rows[i])) for i in range(n)
        ])
        per = cells.reshape(n, -1).mean(1)
        tasks[key] = {
            "f": float(cells.mean()),
            "empty_share": float((per == 0).mean()),
            "p10": float(np.percentile(per, 10)),
            "p50": float(np.percentile(per, 50)),
            "p90": float(np.percentile(per, 90)),
        }
        fd = np.stack([_cells_from_pix_numpy(estab[i] == 3) for i in range(n)])
        per_fd = fd.reshape(n, -1).mean(1)
        tasks["food_drink"] = {
            "f": float(fd.mean()),
            "empty_share": float((per_fd == 0).mean()),
            "p10": float(np.percentile(per_fd, 10)),
            "p50": float(np.percentile(per_fd, 50)),
            "p90": float(np.percentile(per_fd, 90)),
        }
    return {
        "cropset_dir": str(crops_dir),
        "n_crops": n,
        "rule": "c3 tasks; 4x4 majority",
        "tasks": tasks,
    }


def load_task_stats(crops_dir: Path, *, district_dir: Path | None = None, rebuild: bool = False) -> dict:
    path = Path(crops_dir) / STATS_FILENAME
    need_keys = set(MARKER_TASK_KEYS) | set(M1_TASK_KEYS) | set(DENSE_ESTAB_TASK_KEYS)
    if path.exists() and not rebuild:
        doc = json.loads(path.read_text())
        have = set(doc.get("tasks", {}))
        if need_keys <= have:
            return doc
        rebuild = True
    ddir = district_dir or crops_dir.parent.parent
    stats = compute_task_stats(Path(crops_dir), ddir)
    path.write_text(json.dumps(stats, indent=1))
    return stats


def positive_share(key: str, stats: dict) -> float:
    return float(stats["tasks"][key]["f"])


def fg_ce_weights_for_key(key: str, stats: dict, *, alpha: float = 0.5) -> tuple[float, ...]:
    return (fg_ce_weight(key, stats, alpha=alpha),)
