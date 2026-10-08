"""c4 route tasks: segment (straight line) and detour (path pixels from store)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .c2_tasks import fg_ce_weight


class _NpzOnce:
    """np.load(npz) whose members are read once: NpzFile decompresses the whole member on every z[key], so the
    per-sample loops below re-read a whole array per sample (hours on the 20k 512 m wave store)."""

    def __init__(self, path, rows: dict | None = None):
        self._z = np.load(path); self._path = path; self._cache = {}; self._rows = rows or {}
        self.files = self._z.files

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = (_npz_head_rows(self._path, key, self._rows[key]) if key in self._rows else self._z[key])
        return self._cache[key]


def _npz_head_rows(path, key: str, m: int) -> np.ndarray:
    """The first m rows of a member of an npz without inflating the rest (the 512 m wave field is 2.6 GB)."""
    import zipfile
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as fp:
        version = np.lib.format.read_magic(fp)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(fp)
        if fortran:
            raise ValueError(f"{path}:{key} is Fortran-ordered")
        m = min(m, shape[0]); row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        return np.frombuffer(fp.read(m * row), dtype).reshape((m,) + tuple(shape[1:]))

STATS_FILENAME = "c4_task_stats.json"
GRID = 64
IMG = 256
FOREGROUND = 1
IGNORE_TARGET = -100  # cell value that cross_entropy skips (its default ignore_index): no mask supervision
TOL_M = 4.0  # eval tolerance for thin lines, in metres (one cell at grid 64)

# cond_emb index = relative_cond_id + 1 (same convention as c3).
REL_SEGMENT = 18
REL_DETOUR = 19
REL_BUILDING = 20  # dense helper: mark building cells, markers ignored (C4_PLAN §10 addendum)
REL_RAYS = 21  # C5 rung 1: rays from each marker to the building corners it sees within 80 m
REL_LENGTH = 22  # C5 rungs 2-3: length of what is drawn on the scratchpad (one token for both rungs)
REL_TIGHTEN = 23  # C5 rung 4: pull the loose path on the scratchpad tight around the buildings
REL_TRAJ = 24  # C5 rung 1 (30 Sep): key trajectories within 100 m walking from each marker
REL_TRAJ_ONE = 25  # the same from one marker only (marker 0; the other is not drawn)
REL_TRAJ_ONE_R40 = 26  # 40 m horizon, one marker, round 1 (1 Oct 2026)
REL_TRAJ_STEP2_R40, REL_TRAJ_STEP3_R40, REL_TRAJ_STEP4_R40 = 27, 28, 29  # rounds 2-4: previous rounds on the scratchpad
# C5 R5 (1 Oct 2026): the wave search in four step types (C5_PLAN §7)
REL_WAVE = 30  # ONE task id for every pass of the wave search: the state token (slot 0) is the only difference
REL_STATE_SEARCH, REL_STATE_BACK, REL_STATE_DRAW = 34, 35, 36  # the state token IN, condition slot 0
WAVE_HORIZON_M = 40.0  # one front every 40 m of walking
LENGTH_BIN_M = 10.0
N_LENGTH_BINS = 37  # 0-9 m ... 350-359 m, 36 = 360 m or more
TOK_SEARCH, TOK_BACK, TOK_DRAW, TOK_DONE = 37, 38, 39, 40  # state tokens OUT, after the length bins
N_TOKEN_CLASSES = 41
PATH_WIDTH_CELLS = 3  # the backward path is drawn 3 cells wide (thick = path, thin = front)


def length_bin(length_m):
    """Length in metres -> token class (works on floats, numpy arrays and torch tensors)."""
    import numpy as _np
    try:
        import torch as _t
        if isinstance(length_m, _t.Tensor):
            return (length_m / LENGTH_BIN_M).floor().clamp(0, N_LENGTH_BINS - 1).long()
    except ImportError:
        pass
    return _np.clip(_np.floor(_np.asarray(length_m) / LENGTH_BIN_M), 0, N_LENGTH_BINS - 1).astype(_np.int64)


TASK_KEYS: tuple[str, ...] = ("segment", "detour")
HELPER_KEYS: tuple[str, ...] = ("building",)
C4_MIX_L8B: tuple[str, ...] = ("within_20m", "segment", "detour")

HUMAN_NAMES = {
    "segment": "straight segment between two markers",
    "detour": "shortest path around buildings between two markers",
    "building": "building cells (crop-wide, markers ignored)",
    "rays": "rays from each marker to the building corners it can see and grazes (within 80 m)",
    "length_line": "length of the straight line on the scratchpad (token, 10 m bins)",
    "length_path": "length of the polyline on the scratchpad (token, 10 m bins)",
    "tighten": "the taut version of the loose path on the scratchpad",
    "trajectories": "key trajectories within 100 m walking from each marker",
    "trajectories_one": "key trajectories within 100 m walking from the one marker",
    "trajectories_one_r40": "key trajectories within 40 m walking from the one marker (round 1)",
    "trajectories_step2_r40": "round 2 of the 40 m key trajectories, given round 1 on the scratchpad",
    "trajectories_step3_r40": "round 3 of the 40 m key trajectories, given rounds 1-2 on the scratchpad",
    "trajectories_step4_r40": "round 4 of the 40 m key trajectories, given rounds 1-3 on the scratchpad",
    "wave_first": "wave search, first pass: start on the pad -> start + first front (40 m)",
    "wave_step": "wave search, a middle pass: fronts so far -> fronts plus the next one",
    "wave_reach": "wave search, the pass whose new front reaches the goal -> token BACK",
    "wave_back": "wave search backward: the outermost front consumed, the thick path extended one front inward",
    "wave_origin": "wave search backward, the pass whose path reaches the start -> token DRAW",
    "wave_draw": "wave search, the final pass: start disc + thick path -> the path, token DONE",
    "wave_fwd": "wave search forward, any pass (k uniform over the route's passes; token BACK on the last)",
    "wave_bwd": "wave search backward, any pass (j uniform over the fronts; token DRAW on the last)",
}

DETOUR_STRATA: tuple[tuple[float, float], ...] = ((1.0, 1.05), (1.05, 1.3), (1.3, 99.0))
STRATUM_LABELS = ("ratio_1.00_1.05", "ratio_1.05_1.30", "ratio_gt_1.30")
M_PER_CELL = 256.0 / GRID


def m_per_cell(grid: int) -> float:
    return IMG / grid


def stratum_from_detour_ratio(r: float) -> int:
    for i, (lo, hi) in enumerate(DETOUR_STRATA):
        if lo <= r < hi:
            return i
    return len(DETOUR_STRATA) - 1


class C4TargetKind(str, Enum):
    SEGMENT = "segment"
    DETOUR_PATH = "detour_path"
    BUILDING = "building"
    RAYS = "rays"
    LENGTH_LINE = "length_line"
    LENGTH_PATH = "length_path"
    TIGHTEN = "tighten"
    TRAJ = "trajectories"
    TRAJ_ONE = "trajectories_one"
    TRAJ_ONE_R40 = "trajectories_one_r40"
    TRAJ_STEP2_R40 = "trajectories_step2_r40"
    TRAJ_STEP3_R40 = "trajectories_step3_r40"
    TRAJ_STEP4_R40 = "trajectories_step4_r40"
    WAVE_FIRST = "wave_first"
    WAVE_STEP = "wave_step"
    WAVE_REACH = "wave_reach"
    WAVE_BACK = "wave_back"
    WAVE_ORIGIN = "wave_origin"
    WAVE_DRAW = "wave_draw"
    WAVE_FWD = "wave_fwd"
    WAVE_BWD = "wave_bwd"


#: trajectory kinds -> (file suffix of c5_traj{suffix}_{split}.npz, marker shown or None for both, round):
#: the target is that round; rounds before it are drawn on the scratchpad (round 1: nothing)
TRAJ_KIND_INFO = {
    C4TargetKind.TRAJ: ("", None, 1), C4TargetKind.TRAJ_ONE: ("", 0, 1),
    C4TargetKind.TRAJ_ONE_R40: ("_r40", 0, 1), C4TargetKind.TRAJ_STEP2_R40: ("_r40", 0, 2),
    C4TargetKind.TRAJ_STEP3_R40: ("_r40", 0, 3), C4TargetKind.TRAJ_STEP4_R40: ("_r40", 0, 4),
}
#: wave keys -> (task cond id, kind, state cond id in)
WAVE_KEYS = {"wave_first": (REL_WAVE, C4TargetKind.WAVE_FIRST, REL_STATE_SEARCH),
             "wave_step": (REL_WAVE, C4TargetKind.WAVE_STEP, REL_STATE_SEARCH),
             "wave_reach": (REL_WAVE, C4TargetKind.WAVE_REACH, REL_STATE_SEARCH),
             "wave_back": (REL_WAVE, C4TargetKind.WAVE_BACK, REL_STATE_BACK),
             "wave_origin": (REL_WAVE, C4TargetKind.WAVE_ORIGIN, REL_STATE_BACK),
             "wave_draw": (REL_WAVE, C4TargetKind.WAVE_DRAW, REL_STATE_DRAW),
             "wave_fwd": (REL_WAVE, C4TargetKind.WAVE_FWD, REL_STATE_SEARCH),
             "wave_bwd": (REL_WAVE, C4TargetKind.WAVE_BWD, REL_STATE_BACK)}
WAVE_KINDS = (C4TargetKind.WAVE_FIRST, C4TargetKind.WAVE_STEP, C4TargetKind.WAVE_REACH, C4TargetKind.WAVE_BACK,
              C4TargetKind.WAVE_ORIGIN, C4TargetKind.WAVE_DRAW, C4TargetKind.WAVE_FWD, C4TargetKind.WAVE_BWD)
# Full search schedule for rollout / test viz (independent of training task mix).
WAVE_ROLLOUT_TASK_KEYS = ("wave_first", "wave_step", "wave_reach", "wave_back", "wave_origin", "wave_draw")  # the fixed step keys; wave_fwd/wave_bwd are training samplers
TRAJ_KEY_REL = {"trajectories": (REL_TRAJ, C4TargetKind.TRAJ), "trajectories_one": (REL_TRAJ_ONE, C4TargetKind.TRAJ_ONE),
                "trajectories_one_r40": (REL_TRAJ_ONE_R40, C4TargetKind.TRAJ_ONE_R40),
                "trajectories_step2_r40": (REL_TRAJ_STEP2_R40, C4TargetKind.TRAJ_STEP2_R40),
                "trajectories_step3_r40": (REL_TRAJ_STEP3_R40, C4TargetKind.TRAJ_STEP3_R40),
                "trajectories_step4_r40": (REL_TRAJ_STEP4_R40, C4TargetKind.TRAJ_STEP4_R40)}


@dataclass(frozen=True)
class C4TaskSpec:
    key: str
    relative_cond_id: int
    name: str
    target_kind: C4TargetKind
    state_cond_id: int = -1   # C5 R5: the state token in (condition slot 0); -1 = slot unused

    @property
    def show_marker_plane(self) -> bool:
        return True

    def padded_cond_ids(self) -> tuple[int, int]:
        return (self.state_cond_id, self.relative_cond_id)


def n_cond_emb_for_c4() -> int:
    return max(REL_SEGMENT, REL_DETOUR, REL_BUILDING, REL_RAYS, REL_LENGTH, REL_TIGHTEN, REL_TRAJ, REL_TRAJ_ONE, REL_TRAJ_STEP4_R40, REL_STATE_DRAW) + 2


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
    if key == "rays":
        return C4TaskSpec(key=key, relative_cond_id=REL_RAYS, name=HUMAN_NAMES[key],
                          target_kind=C4TargetKind.RAYS)
    if key in ("length_line", "length_path"):
        # both rungs share one condition token: "measure what is on the scratchpad"
        return C4TaskSpec(key=key, relative_cond_id=REL_LENGTH, name=HUMAN_NAMES[key],
                          target_kind=C4TargetKind.LENGTH_LINE if key == "length_line" else C4TargetKind.LENGTH_PATH)
    if key == "tighten":
        return C4TaskSpec(key=key, relative_cond_id=REL_TIGHTEN, name=HUMAN_NAMES[key], target_kind=C4TargetKind.TIGHTEN)
    if key in TRAJ_KEY_REL:
        rel, kind = TRAJ_KEY_REL[key]
        return C4TaskSpec(key=key, relative_cond_id=rel, name=HUMAN_NAMES[key], target_kind=kind)
    if key in WAVE_KEYS:
        rel, kind, state = WAVE_KEYS[key]
        return C4TaskSpec(key=key, relative_cond_id=rel, name=HUMAN_NAMES[key], target_kind=kind, state_cond_id=state)
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


def stats_filename(grid: int = GRID) -> str:
    """Grid 64 keeps the old name; other output grids get their own cache file."""
    return STATS_FILENAME if grid == GRID else f"c4_task_stats_g{grid}.json"


def compute_task_stats(routes_dir: Path, grid: int = GRID) -> dict:
    """Positive cell share per task from train split route store, on a grid x grid output."""
    path = routes_dir / "c4_routes_train.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    markers = z["markers"]
    path_pix = z["path_pix"]
    n_pix = z["n_pix"]
    sj = Path(routes_dir) / "store.json"
    crop = int(json.loads(sj.read_text())["crop_px"]) if sj.is_file() else IMG
    n = markers.shape[0]
    seg_cells, det_cells = [], []
    for i in range(n):
        a, b = markers[i, 0], markers[i, 1]
        seg = rasterise_segment_numpy(a, b, crop_px=crop)
        seg_cells.append(path_pix_to_grid_numpy(seg, grid=grid, img=crop))
        det_cells.append(path_pix_to_grid_numpy(path_pix[i, : int(n_pix[i])], grid=grid, img=crop))
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
    # length tasks have no mask target (token only); the entries only keep the weight table complete
    tasks["length_line"] = dict(tasks["segment"], rule="no mask target (token only); placeholder copied from segment")
    tasks["length_path"] = dict(tasks["detour"], rule="no mask target (token only); placeholder copied from detour")
    tasks["tighten"] = dict(tasks["detour"], rule="the stored tight path (share taken from detour)")
    fans = Path(routes_dir) / "c5_fans_train.npz"
    if fans.is_file():
        fz = _NpzOnce(fans); m = min(2000, n); cells = []
        for i in range(m):
            gm = np.zeros((grid, grid), dtype=bool)
            for k in range(2):
                for q in fz["fan_xy"][i, k, : int(fz["fan_n"][i, k])]:
                    gm |= path_pix_to_grid_numpy(rasterise_segment_numpy(markers[i, k], q, crop_px=crop), grid=grid, img=crop)
            cells.append(gm)
        st = np.stack(cells); per = st.reshape(m, -1).mean(1)
        tasks["rays"] = {"f": float(st.mean()), "empty_share": float((per == 0).mean()), "p10": float(np.percentile(per, 10)),
                         "p50": float(np.percentile(per, 50)), "p90": float(np.percentile(per, 90)),
                         "rule": "grazing rays marker -> visible corners within 80 m, any pixel per cell; first 2000 train samples"}
    for sfx in ("", "_r40"):
        traj = Path(routes_dir) / f"c5_traj{sfx}_train.npz"
        if not traj.is_file():
            continue
        tz = _NpzOnce(traj); m = min(2000, min(n, len(tz["offset"]) - 1))
        jobs = [("trajectories_one" + sfx, "pix", "offset", True, f"key trajectories{sfx} of marker 0")]
        if tz["n_traj"][:, 1].any():
            jobs.append(("trajectories" + sfx, "pix", "offset", False, f"key trajectories{sfx} of both markers"))
        jobs += [(f"trajectories_step{r}{sfx}", f"pix{r}", f"offset{r}", False, f"round {r} of the key trajectories{sfx}")
                 for r in (2, 3, 4) if f"pix{r}" in tz.files]
        for key, pk, ok, only0, rule in jobs:
            cells = []
            for i in range(m):
                sl = slice(tz[ok][i], tz[ok][i + 1]); px = tz[pk][sl].astype(np.int64)
                if only0:
                    px = px[tz["pix_tag"][sl] < 32]
                cells.append(path_pix_to_grid_numpy(px, grid=grid, img=crop))
            st = np.stack(cells); per = st.reshape(m, -1).mean(1)
            tasks[key] = {"f": float(st.mean()), "empty_share": float((per == 0).mean()), "p10": float(np.percentile(per, 10)),
                          "p50": float(np.percentile(per, 50)), "p90": float(np.percentile(per, 90)),
                          "rule": rule + ", any pixel per cell; first 2000 train samples"}
    wave = Path(routes_dir) / "c5_wave_train.npz"
    if wave.is_file():
        from .c5_wave import cell_minmax_numpy, fronts_upto_numpy, start_disc_numpy, UNREACHED
        wz = _NpzOnce(wave, rows={"dist": 2000}); m = min(2000, min(n, len(wz["goal_m"])))
        cells_f, cells_p = [], []
        for i in range(m):
            dmin, dmax = cell_minmax_numpy(wz["dist"][i], grid)
            gm = wz["goal_m"][i]; n_goal = int(np.floor(gm / WAVE_HORIZON_M)) + 1 if np.isfinite(gm) else 1
            cells_f.append(fronts_upto_numpy(dmin, dmax, n_goal, WAVE_HORIZON_M) | start_disc_numpy(markers[i, 0], grid, crop))
            from scipy import ndimage
            cells_p.append(ndimage.binary_dilation(path_pix_to_grid_numpy(wz["path"][wz["offset"][i]:wz["offset"][i + 1]].astype(np.int64), grid=grid, img=crop), structure=np.ones((3, 3), bool)))
        for keys, cells, rule in ((("wave_first", "wave_step", "wave_reach", "wave_fwd"), cells_f, "start disc + fronts up to the goal pass"),
                                  (("wave_back", "wave_origin", "wave_draw", "wave_bwd"), cells_p, "the way back goal -> start, 3 cells wide")):
            st = np.stack(cells); per = st.reshape(m, -1).mean(1)
            for key in keys:
                tasks[key] = {"f": float(st.mean()), "empty_share": float((per == 0).mean()), "p10": float(np.percentile(per, 10)),
                              "p50": float(np.percentile(per, 50)), "p90": float(np.percentile(per, 90)),
                              "rule": rule + "; first 2000 train samples"}
    district = Path(routes_dir).parent / "district_labels_1m.npz"
    if district.is_file():
        surf = np.load(district)["surface"]
        s = crop // grid
        h, w = (surf.shape[0] // s) * s, (surf.shape[1] // s) * s
        blk = (surf[:h, :w] == 3).reshape(h // s, s, w // s, s).sum(axis=(1, 3))
        cells = blk >= (s * s) // 2
        tasks["building"] = {"f": float(cells.mean()), "empty_share": 0.0, "p10": float(cells.mean()),
                             "p50": float(cells.mean()), "p90": float(cells.mean()),
                             "rule": "surface == 3 majority (>= 8 of 16 px) per cell, district-wide"}
    return {
        "routes_dir": str(routes_dir),
        "n_routes": n,
        "grid": grid,
        "crop_px": crop,
        "rule": f"c4 route cells; any pixel in a {crop // grid}x{crop // grid} px block",
        "tasks": tasks,
    }


def load_task_stats(routes_dir: Path, *, rebuild: bool = False, grid: int = GRID) -> dict:
    path = Path(routes_dir) / stats_filename(grid)
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    stats = compute_task_stats(routes_dir, grid=grid)
    path.write_text(json.dumps(stats, indent=1))
    return stats
