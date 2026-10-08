"""Track A shared contract (4 Oct 2026): route images + teacher polylines for a VLM, and the polyline scorer
(docs/plans/TRACK_A_QWEN_PLAN_2026-10-04.md §1).

Both tracks have to read the same routes and be scored by the same rule, so this file is the only place where a
route becomes a picture and a polyline becomes "solved" or a failure reason.

Coordinates. Window metres: continuous, origin at the window's top-left corner, x right, y down; district pixel
(origin + i) covers [i, i+1) m, so the markers of the store (x.5) sit at cell centres. Image pixels: window metres
times px_per_m, same origin and axes (the 1 m raster is upscaled nearest-neighbour, so image pixel j covers
[j/ppm, (j+1)/ppm) m). At 2 px/m a cell centre lands on an integer image coordinate.

The teacher. The route's teacher is the way back of the c5 wave (spatial_data/c5_wave.py): goal -> start along the
clearance-priced Dijkstra tree, buildings the only obstacle, the walking length goal_m in metres. We reverse it
(start -> goal) and simplify it with Douglas-Peucker. A simplified polyline can cut a corner through a building,
so the simplification uses the scorer's own building test: a span whose chord touches a building is split at its
farthest point like a span beyond the tolerance (only that span, not the whole route; halving the tolerance for the
whole route, the first version, gave 30+ points on the ~2 % of routes with one tight corner). c5_wave stores only the part of the way back inside the window; for the ~3-10 % of routes whose way back
leaves the window (the wave runs 100 m beyond it), we recompute the full path with the same code, so those teacher
polylines have points outside the image (flag `leaves_window`).

The rule (mirrors the success rule of the wave search, C5 report §4.3, for polylines): start within 6 m of the start
marker, end within 6 m of the goal marker, no segment crosses a building pixel (each segment sampled at <= 0.5 m on
the 1 m raster), length <= 1.10 x the teacher's walking length. Beyond the window the district map counts (as for the
wave, up to 100 m out); beyond that and beyond the district everything counts as building.

Usage (project venv, repo root):
  python -m spatial_data.track_a_export export --store data/amsterdam/de_pijp/crops/v4/c4_100k --split val \
      --out data/track_a/de_pijp_val_plain
  python -m spatial_data.track_a_export baseline --routes data/track_a/de_pijp_val_plain/routes.jsonl \
      --kind straight --out /tmp/straight.jsonl
  python -m spatial_data.track_a_export score --routes data/track_a/de_pijp_val_plain/routes.jsonl \
      --pred /tmp/straight.jsonl --out /tmp/straight_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# --- the rule ---------------------------------------------------------------------------------------------------
ENDPOINT_TOL_M = 6.0        # start / end of the polyline within this many metres of the markers
MAX_LENGTH_RATIO = 1.10     # polyline length <= this x the teacher's walking length
SAMPLE_STEP_M = 0.5         # a segment is tested for buildings at points at most this far apart
MASK_MARGIN_M = 100         # the building mask reaches this far beyond the window (= c5_wave.MARGIN_M)
BUILDING_CLASS = 3
HORIZON_M = 40.0            # one front of the wave search; n_goal = floor(goal_m / 40) + 1
BANDS = (("2-3", 0, 3), ("4-5", 4, 5), ("6+", 6, 10 ** 9))   # n_goal bands of the report (n_goal 1 counts as 2-3)
REASONS = ("unparseable", "wrong endpoints", "crosses a building", "too long")   # priority order

# --- rendering ----------------------------------------------------------------------------------------------------
MARKER_RADIUS_M = 4.0
COLOURS = {
    "plain": {"ground": "#F5F3EE", "building": "#343A46"},
    # surface classes of legend.json: 0 none (courtyards, green, water, ... everything not paved), 1 roadway
    # (incl. cycle path and parking), 2 sidewalk, 3 building
    "layers": {"none": "#E4E0D2", "roadway": "#B7BEC8", "sidewalk": "#E9D6A9", "building": "#343A46"},
    "start": "#1FA34A", "goal": "#D7263D", "outline": "#FFFFFF",
}


def _hex(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def class_lut(style: str) -> np.ndarray:
    """(4, 3) uint8 colour per surface class for a style."""
    if style == "plain":
        g, b = _hex(COLOURS["plain"]["ground"]), _hex(COLOURS["plain"]["building"])
        return np.array([g, g, g, b], np.uint8)
    if style == "layers":
        c = COLOURS["layers"]
        return np.array([_hex(c["none"]), _hex(c["roadway"]), _hex(c["sidewalk"]), _hex(c["building"])], np.uint8)
    raise ValueError(f"unknown style {style!r}")


# --- coordinates ----------------------------------------------------------------------------------------------------
def window_m_to_image_px(xy, px_per_m: float) -> np.ndarray:
    return np.asarray(xy, np.float64) * px_per_m


def image_px_to_window_m(xy, px_per_m: float) -> np.ndarray:
    return np.asarray(xy, np.float64) / px_per_m


def _num(v: float):
    """JSON number: an int when the value is integral (teacher points at 2 px/m are), else 3 decimals."""
    r = round(float(v), 3)
    return int(r) if r == int(r) else r


def _pts_json(p: np.ndarray) -> list:
    return [[_num(x), _num(y)] for x, y in p]


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# --- the building test (shared by export and score) -----------------------------------------------------------
def segments_hit_building(pts_m: np.ndarray, mask: np.ndarray, margin: int = MASK_MARGIN_M) -> np.ndarray:
    """(k-1,) bool per segment of a polyline in window metres: True if a sample point (<= SAMPLE_STEP_M apart,
    endpoints included) lies on a building pixel or off the mask. `mask` is window + margin on every side."""
    pts_m = np.asarray(pts_m, np.float64)
    a, b = pts_m[:-1], pts_m[1:]
    n = np.maximum(np.ceil(np.hypot(*(b - a).T) / SAMPLE_STEP_M).astype(np.int64), 1) + 1   # samples per segment
    seg = np.repeat(np.arange(len(a)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    s = a[seg] + t[:, None] * (b - a)[seg]
    ix = np.floor(s[:, 0]).astype(np.int64) + margin; iy = np.floor(s[:, 1]).astype(np.int64) + margin
    H, W = mask.shape
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    hit = ~inside
    hit[inside] = mask[iy[inside], ix[inside]]
    return np.bincount(seg, weights=hit, minlength=len(a)) > 0


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, np.float64)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


@lru_cache(maxsize=8)
def _district_buildings(district_path: str) -> np.ndarray:
    return np.load(district_path)["surface"] == BUILDING_CLASS


def cut_building_mask(bld: np.ndarray, origin, crop: int, margin: int = MASK_MARGIN_M) -> np.ndarray:
    """Window + margin of the district building mask; outside the district counts as building."""
    dx, dy = int(origin[0]), int(origin[1])
    out = np.ones((crop + 2 * margin, crop + 2 * margin), bool)
    x0, y0 = dx - margin, dy - margin
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + out.shape[1], bld.shape[1]), min(y0 + out.shape[0], bld.shape[0])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = bld[sy0:sy1, sx0:sx1]
    return out


def load_building_mask(record: dict) -> np.ndarray:
    """The building mask of a route (window + MASK_MARGIN_M each side, True = building), re-cut from the district
    raster named in the record, so the scorer never trusts the image."""
    return cut_building_mask(_district_buildings(str(resolve(record["district"]))), record["origin"],
                             int(record["window_m"]))


# --- Douglas-Peucker ------------------------------------------------------------------------------------------------
def douglas_peucker(pts: np.ndarray, tol: float, must_split=None) -> np.ndarray:
    """Indices of the kept points (always the first and the last), in order. `must_split(a, b)` -> True forces the
    span between points a and b to be split even if it is within the tolerance (used for "the chord touches a
    building"); spans of two neighbouring points are never split, so it ends."""
    pts = np.asarray(pts, np.float64); n = len(pts)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]; d = b - a; L = math.hypot(*d)
        q = pts[i + 1:j] - a
        dist = np.abs(d[0] * q[:, 1] - d[1] * q[:, 0]) / L if L > 0 else np.hypot(q[:, 0], q[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol or (must_split is not None and must_split(a, b)):
            m = i + 1 + k; keep[m] = True
            stack += [(i, m), (m, j)]
    return np.nonzero(keep)[0]


def simplify_route(dense_m: np.ndarray, mask: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """DP at `tol` whose chords may not touch a building. -> (points, points added for buildings compared with
    plain DP at `tol`). The dense path itself is free (checked by the caller), so a valid result always exists."""
    hits = lambda a, b: bool(segments_hit_building(np.stack([a, b]), mask)[0])
    k = douglas_peucker(dense_m, tol, must_split=hits)
    return dense_m[k], int(len(k) - len(douglas_peucker(dense_m, tol)))


# --- the teacher path -----------------------------------------------------------------------------------------------
def full_wave_path(bld: np.ndarray, origin, crop: int, markers: np.ndarray):
    """The way back goal -> start of c5_wave.wave_for_sample, NOT clipped to the window. -> (path (k,2) int, goal_m).
    Same mask cut, same graph, same Dijkstra, so the same tree as the stored wave."""
    from scipy.sparse.csgraph import dijkstra
    from spatial_data import c5_trajectories as T
    from spatial_data.c5_wave import MARGIN_M
    dx, dy = int(origin[0]), int(origin[1]); M = int(MARGIN_M)
    x0, y0 = max(dx - M, 0), max(dy - M, 0)
    x1, y1 = min(dx + crop + M, bld.shape[1]), min(dy + crop + M, bld.shape[0])
    field = T._Field(bld[y0:y1, x0:x1])
    off = np.array([dx - x0, dy - y0])
    start, goal = markers[0].astype(np.float64) + off, markers[1].astype(np.float64) + off
    src = field.idx[int(start[1]), int(start[0])]; g = field.idx[int(goal[1]), int(goal[0])]
    D, pred = dijkstra(field.G, indices=src, return_predecessors=True)
    length = T._walk_length(pred, np.isfinite(D), field.fy, field.fx, src)
    nodes = [g]
    while nodes[-1] != src:
        nodes.append(int(pred[nodes[-1]]))
    return np.stack([field.fx[nodes] - off[0], field.fy[nodes] - off[1]], 1), float(length[g])


def teacher_dense_m(path_px: np.ndarray, markers: np.ndarray, bld, origin, crop: int, goal_m: float):
    """Dense teacher path start -> goal in window metres (cell centres) and whether it leaves the window."""
    p = np.asarray(path_px, np.int64)
    gap = len(p) < 2 or (np.abs(np.diff(p, axis=0)).max(axis=1) > 1).any()
    if gap:   # c5_wave clipped the way back at the window edge: recompute it in full
        p, g_m = full_wave_path(bld, origin, crop, markers)
        assert abs(g_m - goal_m) < 1e-3, (g_m, goal_m)
    p = p[::-1]
    assert (p[0] == np.floor(markers[0])).all() and (p[-1] == np.floor(markers[1])).all(), "teacher path misses a marker"
    p = p.astype(np.float64) + 0.5
    p[0], p[-1] = markers[0], markers[1]     # the markers themselves (cell centres in our stores, x.5)
    return p, bool(gap)


# --- rendering ------------------------------------------------------------------------------------------------------
def render_route(surface_win: np.ndarray, markers_m: np.ndarray, px_per_m: int, style: str):
    """PIL image of one window: surface classes coloured, nearest-neighbour upscaled, start (green) and goal (red)
    discs with a thin white outline."""
    from PIL import Image, ImageDraw
    rgb = class_lut(style)[surface_win]
    rgb = np.repeat(np.repeat(rgb, px_per_m, 0), px_per_m, 1)
    im = Image.fromarray(rgb, "RGB"); dr = ImageDraw.Draw(im)
    r = MARKER_RADIUS_M * px_per_m
    for xy, col in ((markers_m[0], COLOURS["start"]), (markers_m[1], COLOURS["goal"])):
        cx, cy = window_m_to_image_px(xy, px_per_m)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=COLOURS["outline"], width=1)
    return im


# --- export -----------------------------------------------------------------------------------------------------------
_W: dict = {}


def _init(store: str, split: str, district: str, crop: int, ppm: int, style: str, tol: float, out: str):
    z = np.load(Path(store) / f"c4_routes_{split}.npz")
    w = np.load(Path(store) / f"c5_wave_{split}.npz")
    sf = np.load(district)["surface"]
    _W.update(origin=z["origin"], markers=z["markers"], goal_m=w["goal_m"], path=w["path"], offset=w["offset"],
              surface=sf, bld=sf == BUILDING_CLASS, crop=crop, ppm=ppm, style=style, tol=tol, out=Path(out),
              store=store, split=split, district=district)


def export_one(i: int) -> dict | None:
    from spatial_data.wave_search_schedule import n_goal_from_m
    W = _W; crop, ppm = W["crop"], W["ppm"]
    goal_m = float(W["goal_m"][i])
    if not math.isfinite(goal_m):
        return None
    origin, mk = W["origin"][i], W["markers"][i].astype(np.float64)
    mask = cut_building_mask(W["bld"], origin, crop)
    dense, leaves = teacher_dense_m(W["path"][W["offset"][i]:W["offset"][i + 1]], mk, W["bld"], origin, crop, goal_m)
    assert not segments_hit_building(dense, mask).any(), f"route {i}: dense teacher path touches a building"
    poly, n_added = simplify_route(dense, mask, W["tol"])
    assert not segments_hit_building(poly, mask).any(), f"route {i}: simplified teacher touches a building"
    dx, dy = int(origin[0]), int(origin[1])
    img = render_route(W["surface"][dy:dy + crop, dx:dx + crop], mk, ppm, W["style"])
    name = f"images/route_{i:06d}.png"; img.save(W["out"] / name)
    return dict(route=int(i), image=name, image_px=[crop * ppm, crop * ppm], window_m=crop, px_per_m=ppm,
                start=_pts_json(window_m_to_image_px(mk[:1], ppm))[0], goal=_pts_json(window_m_to_image_px(mk[1:], ppm))[0],
                teacher_m=round(goal_m, 3), n_goal=n_goal_from_m(goal_m, HORIZON_M),
                teacher=_pts_json(window_m_to_image_px(poly, ppm)), n_points=int(len(poly)),
                straight_m=round(float(np.hypot(*(mk[1] - mk[0]))), 3), leaves_window=leaves,
                n_points_added_for_buildings=n_added, origin=[dx, dy], store=W["store"], split=W["split"], district=W["district"])


def _rel(p: Path) -> str:
    """Repo-relative path for the records (so they work on any machine with the repo); not .resolve(), because
    data/ is a symlink to another disk."""
    try:
        return str(p.absolute().relative_to(ROOT.absolute()))
    except ValueError:
        return str(p)


def export(store: str | Path, split: str, out: str | Path, *, limit: int | None = None, px_per_m: int = 2,
           style: str = "plain", simplify_m: float = 2.0, workers: int = 3) -> dict:
    from spatial_data.dataset_c4_gpu import store_crop_px
    store_p = resolve(store); out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True)
    crop = store_crop_px(store_p); district = store_p.parent / "district_labels_1m.npz"
    n_store = int(np.load(store_p / f"c4_routes_{split}.npz")["origin"].shape[0])
    n_wave = int(np.load(store_p / f"c5_wave_{split}.npz")["goal_m"].shape[0])
    assert n_wave == n_store, f"c5_wave_{split}.npz has {n_wave} routes, the store {n_store}"
    N = n_store if limit is None else min(limit, n_store)
    t0 = time.perf_counter()
    args = (_rel(store_p), split, _rel(district), crop, px_per_m, style, simplify_m, str(out))
    # the worker paths are repo-relative strings in the records; the workers open them from the repo root
    init_args = (str(store_p), split, str(district)) + args[3:]
    with Pool(workers, initializer=_init, initargs=init_args) as pool:
        recs = pool.map(export_one, range(N), chunksize=50)
    for r in recs:
        if r is not None:
            r["store"], r["district"] = args[0], args[2]
    kept = [r for r in recs if r is not None]
    with open(out / "routes.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    npts = np.array([r["n_points"] for r in kept])
    meta = dict(
        args=dict(store=args[0], split=split, limit=limit, px_per_m=px_per_m, style=style, simplify_m=simplify_m),
        district=args[2], window_m=crop, image_px=crop * px_per_m, n_store_routes=n_store, n_considered=N,
        n_exported=len(kept), n_skipped_unreachable=N - len(kept),
        n_leaves_window=int(sum(r["leaves_window"] for r in kept)),
        n_routes_split_for_buildings=int(sum(r["n_points_added_for_buildings"] > 0 for r in kept)),
        n_points_added_for_buildings=int(sum(r["n_points_added_for_buildings"] for r in kept)),
        n_points=dict(median=float(np.median(npts)), p90=float(np.percentile(npts, 90)), max=int(npts.max()),
                      mean=round(float(npts.mean()), 2)),
        n_goal_counts={b: int(sum(lo <= r["n_goal"] <= hi for r in kept)) for b, lo, hi in BANDS},
        seconds=round(time.perf_counter() - t0, 1),
        colours=dict(COLOURS[style], start=COLOURS["start"], goal=COLOURS["goal"], marker_outline=COLOURS["outline"]),
        legend={"0": "none (not paved: courtyards, green, water, ...)", "1": "roadway", "2": "sidewalk",
                "3": "building"} if style == "layers" else {"ground": "everything that is not a building",
                                                             "building": "building"},
        marker_radius_m=MARKER_RADIUS_M, start_marker="green disc", goal_marker="red disc",
        coordinates="image pixels, origin top-left corner, x right, y down; window metres = image px / px_per_m",
        rule=dict(endpoint_tol_m=ENDPOINT_TOL_M, max_length_ratio=MAX_LENGTH_RATIO, sample_step_m=SAMPLE_STEP_M,
                  mask_margin_m=MASK_MARGIN_M, building_class=BUILDING_CLASS),
    )
    (out / "export.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_exported", "n_skipped_unreachable", "n_leaves_window",
                                           "n_routes_split_for_buildings", "n_points_added_for_buildings", "n_points", "n_goal_counts", "seconds")}))
    return meta


# --- scoring ------------------------------------------------------------------------------------------------------------
def load_routes(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _parse_points(points):
    """(k, 2) float array or None: at least two points, each two finite numbers."""
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return None
    try:
        p = np.array([[float(v) for v in q] for q in points], np.float64)
    except (TypeError, ValueError):
        return None
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        return None
    return p


def score_polyline(points, record: dict, building_mask: np.ndarray) -> dict:
    """One prediction (image pixels) against one route. -> {solved, reason (None if solved), start_err_m, goal_err_m,
    crosses, length_m, length_ratio}; the reason is the first failing check in REASONS order, the other fields are
    filled as far as the polyline allows."""
    p = _parse_points(points)
    if p is None:
        return dict(solved=False, reason="unparseable", start_err_m=None, goal_err_m=None, crosses=None,
                    length_m=None, length_ratio=None)
    ppm = float(record["px_per_m"])
    pm = image_px_to_window_m(p, ppm)
    s_err = float(np.hypot(*(pm[0] - image_px_to_window_m(record["start"], ppm))))
    g_err = float(np.hypot(*(pm[-1] - image_px_to_window_m(record["goal"], ppm))))
    crosses = bool(segments_hit_building(pm, building_mask).any())
    L = polyline_length(pm); ratio = L / float(record["teacher_m"])
    if s_err > ENDPOINT_TOL_M or g_err > ENDPOINT_TOL_M:
        reason = "wrong endpoints"
    elif crosses:
        reason = "crosses a building"
    elif ratio > MAX_LENGTH_RATIO:
        reason = "too long"
    else:
        reason = None
    return dict(solved=reason is None, reason=reason, start_err_m=round(s_err, 3), goal_err_m=round(g_err, 3),
                crosses=crosses, length_m=round(L, 3), length_ratio=round(ratio, 4))


def load_predictions(path: str | Path) -> dict[int, object]:
    """route -> points; a line that is not valid JSON or has no route id is ignored (its route stays missing)."""
    out = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                out[int(d["route"])] = d.get("points")
            except (ValueError, KeyError, TypeError):
                continue
    return out


def _band(n_goal: int) -> str:
    return next(b for b, lo, hi in BANDS if lo <= n_goal <= hi)


def score(routes: list[dict], preds: dict[int, object]) -> dict:
    per = []
    for r in routes:
        res = score_polyline(preds.get(r["route"]), r, load_building_mask(r))
        per.append(dict(route=r["route"], n_goal=r["n_goal"], **res))
    n = len(per)
    share = lambda k, m: round(k / m, 4) if m else None
    bands = {}
    for b, lo, hi in BANDS:
        sel = [x for x in per if lo <= x["n_goal"] <= hi]
        bands[b] = dict(n=len(sel), solved=sum(x["solved"] for x in sel), solved_share=share(sum(x["solved"] for x in sel), len(sel)))
    counts = {k: sum(x["reason"] == k for x in per) for k in REASONS}
    solved = sum(x["solved"] for x in per)
    ratios = [x["length_ratio"] for x in per if x["solved"]]
    return dict(n=n, solved=solved, solved_share=share(solved, n), by_n_goal=bands,
                failure_share={k: share(v, n) for k, v in counts.items()}, failure_counts=counts,
                mean_length_ratio_solved=round(float(np.mean(ratios)), 4) if ratios else None,
                rule=dict(endpoint_tol_m=ENDPOINT_TOL_M, max_length_ratio=MAX_LENGTH_RATIO, sample_step_m=SAMPLE_STEP_M),
                routes=per)


def summary_line(rep: dict) -> str:
    b = " / ".join(f"{k} {v['solved_share'] if v['solved_share'] is not None else '-'} (n={v['n']})" for k, v in rep["by_n_goal"].items())
    f = ", ".join(f"{k} {v}" for k, v in rep["failure_share"].items())
    return f"n={rep['n']} solved {rep['solved_share']} | by n_goal {b} | failures {f}"


# --- baselines ------------------------------------------------------------------------------------------------------------
def baseline_teacher(record: dict) -> list:
    return record["teacher"]


def baseline_straight(record: dict) -> list:
    return [record["start"], record["goal"]]


BASELINES = {"teacher": baseline_teacher, "straight": baseline_straight}


def write_baseline(routes: list[dict], kind: str, out: str | Path) -> None:
    fn = BASELINES[kind]
    with open(out, "w") as f:
        for r in routes:
            f.write(json.dumps({"route": r["route"], "points": fn(r)}) + "\n")


# --- CLI ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--store", required=True); e.add_argument("--split", required=True)
    e.add_argument("--limit", type=int, default=None); e.add_argument("--px-per-m", type=int, default=2)
    e.add_argument("--style", choices=("plain", "layers"), default="plain")
    e.add_argument("--simplify-m", type=float, default=2.0); e.add_argument("--workers", type=int, default=3)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--routes", required=True); s.add_argument("--pred", required=True); s.add_argument("--out", default=None)
    b = sub.add_parser("baseline")
    b.add_argument("--routes", required=True); b.add_argument("--kind", choices=tuple(BASELINES), required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.store, a.split, a.out, limit=a.limit, px_per_m=a.px_per_m, style=a.style, simplify_m=a.simplify_m,
               workers=a.workers)
    elif a.cmd == "score":
        rep = score(load_routes(a.routes), load_predictions(a.pred))
        print(summary_line(rep))
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1))
    else:
        write_baseline(load_routes(a.routes), a.kind, a.out)


if __name__ == "__main__":
    main()
