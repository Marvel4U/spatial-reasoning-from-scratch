"""C5 R5 data (1 Oct 2026): the wave series per route for the four step tasks of the wave search
(docs/plans/C5_PLAN.md §7, docs/memo/WAVE_SEARCH_2026-10-01.md).

Per store sample: ONE clearance-priced wave from marker 0 (the start) over the district map padded by one horizon
around the window (spatial_data/c5_trajectories._Field: buildings are the only obstacle, walking metres along the
tree), clipped to the window. Stored:
  dist    (N, S, S) uint8  walking metres from the start / 2 m at 2 m resolution (min over the 2x2 px of a cell),
                           255 = not reached; S = window / 2. uint8 tops out at 508 m: harmless on 256 m windows
                           (fronts are needed up to the goal's, and De Pijp's goals walk at most 448 m), wrong on
                           512 m windows (goals walk up to ~1.1 km, 15 % beyond 508 m): windows above 256 px store
                           uint16 (65535 = not reached), see dist_dtype().
  goal_m  (N,) float32     walking metres from the start to the goal (marker 1), inf if unreachable
  path    (M, 2) int16     pixels of the way back goal -> start along the wave's parent links (x, y), ragged
  offset  (N+1,) int64     sample i owns path[offset[i]:offset[i+1]]
Everything else (bands, the two value maps, passes, tokens, classes) is derived per batch on the GPU.
The horizon is not stored: the field is in metres, the loader picks the horizon (40 m by default).

Usage (project venv, repo root):
  python -m spatial_data.c5_wave --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val --limit 200 --gallery 12
  python -m spatial_data.c5_wave --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val,test --workers 4
  python -m spatial_data.c5_wave --store ... --splits train --limit 40000 --workers 4   # first 40k routes
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from spatial_data import c5_trajectories as T

MARGIN_M = 100.0        # the wave runs this far beyond the window (>= any horizon we use)
BUILDING_CLASS = 3
UNREACHED = 255         # uint8 stores (windows up to 256 px)
UNREACHED16 = 65535     # uint16 stores (larger windows); the GPU loader views it as int16, where it reads -1


def dist_dtype(crop: int):
    """uint8 keeps the 256 m stores byte-identical to what was built before; a 512 m window has walking distances
    beyond 508 m (2 m units), which uint8 would clip, losing every front past the 12th (at 480 m)."""
    return np.uint8 if crop <= 256 else np.uint16


def unreached_mask(dist2: np.ndarray) -> np.ndarray:
    """The 'not reached' cells of a stored field of any of the three dtypes (uint8, uint16, the loader's int16)."""
    if dist2.dtype == np.uint8:
        return dist2 == UNREACHED
    if dist2.dtype == np.uint16:
        return dist2 == UNREACHED16
    return dist2 < 0


def wave_for_sample(bld: np.ndarray, origin, crop: int, markers: np.ndarray):
    """-> dist2 (crop/2, crop/2) uint8 / uint16 (dist_dtype) in 2 m units, goal_m float, path (k, 2) int16 window
    pixels goal -> start"""
    dx, dy = int(origin[0]), int(origin[1]); M = int(MARGIN_M)
    x0, y0 = max(dx - M, 0), max(dy - M, 0)
    x1, y1 = min(dx + crop + M, bld.shape[1]), min(dy + crop + M, bld.shape[0])
    mask = bld[y0:y1, x0:x1]
    off = np.array([dx - x0, dy - y0], dtype=np.float64)
    start, goal = markers[0].astype(np.float64) + off, markers[1].astype(np.float64) + off
    field = T._Field(mask)
    src = field.idx[int(start[1]), int(start[0])]
    dist = np.full(mask.shape, np.inf)
    path = np.zeros((0, 2), np.int16); goal_m = np.inf
    if src >= 0:
        D, pred = dijkstra(field.G, indices=src, return_predecessors=True)
        length = T._walk_length(pred, np.isfinite(D), field.fy, field.fx, src)
        dist[field.fy, field.fx] = length
        g = field.idx[int(goal[1]), int(goal[0])]
        if g >= 0 and np.isfinite(length[g]):
            goal_m = float(length[g])
            nodes = [g]
            while nodes[-1] != src:
                nodes.append(int(pred[nodes[-1]]))
            path = np.stack([field.fx[nodes] - int(off[0]), field.fy[nodes] - int(off[1])], 1).astype(np.int16)
    win = dist[int(off[1]):int(off[1]) + crop, int(off[0]):int(off[0]) + crop]
    s = crop // 2
    cell = win.reshape(s, 2, s, 2).min(axis=(1, 3))
    dt = dist_dtype(crop); top = UNREACHED if dt == np.uint8 else UNREACHED16
    dist2 = np.where(np.isfinite(cell), np.clip(np.round(cell / 2.0), 0, top - 1), top).astype(dt)
    inside = (path[:, 0] >= 0) & (path[:, 0] < crop) & (path[:, 1] >= 0) & (path[:, 1] < crop)
    return dist2, goal_m, path[inside]


def cell_minmax_numpy(dist2: np.ndarray, grid: int):
    """2 m field (S,S) uint8/uint16 -> (dmin, dmax) (grid,grid) metres per output cell; a 2 m sub-cell holds its
    minimum, its maximum is at most 2.9 m more; inf where unreached."""
    d = dist2.astype(np.float64) * 2.0; d[unreached_mask(dist2)] = np.inf
    s = dist2.shape[0] // grid
    blk = d.reshape(grid, s, grid, s)
    return blk.min(axis=(1, 3)), blk.max(axis=(1, 3)) + 2.9


def fronts_upto_numpy(dmin, dmax, k: int, horizon: float) -> np.ndarray:
    """(grid,grid) bool: the rings at j*horizon for j = 1..k; a cell is on ring j if the line crosses it."""
    out = np.zeros(dmin.shape, bool)
    for j in range(1, k + 1):
        out |= (dmin < j * horizon) & (dmax >= j * horizon)
    return out


def start_disc_numpy(start_xy, grid: int, crop: int) -> np.ndarray:
    """(grid,grid) bool: the 3x3 cells around the start's cell (the start seeds the scratchpad)."""
    s = crop // grid; cx, cy = int(start_xy[0] // s), int(start_xy[1] // s)
    out = np.zeros((grid, grid), bool)
    out[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
    return out


_W = {}


def _init(district_path, routes_path, crop):
    _W["bld"] = np.load(district_path)["surface"] == BUILDING_CLASS
    z = np.load(routes_path); _W["origin"] = z["origin"]; _W["markers"] = z["markers"]; _W["crop"] = crop


def _work(rng_ix):
    lo, hi = rng_ix
    return lo, [wave_for_sample(_W["bld"], _W["origin"][i], _W["crop"], _W["markers"][i]) for i in range(lo, hi)]


def build(store: Path, split: str, limit: int | None, gallery: int, workers: int):
    from spatial_data.dataset_c4_gpu import store_crop_px
    crop = store_crop_px(store)
    routes = store / f"c4_routes_{split}.npz"; district = store.parent / "district_labels_1m.npz"
    N = int(np.load(routes)["origin"].shape[0]); N = N if limit is None else min(limit, N)
    chunks = [(a, min(a + 250, N)) for a in range(0, N, 250)]
    res = [None] * N; t0 = time.perf_counter(); done = 0
    with Pool(workers, initializer=_init, initargs=(str(district), str(routes), crop)) as pool:
        for lo, out in pool.imap_unordered(_work, chunks):
            res[lo:lo + len(out)] = out; done += len(out)
            if done % 5000 < 250:
                print(f"  {split}: {done}/{N}  {1000 * (time.perf_counter() - t0) / done:.1f} ms/sample (all workers)", flush=True)
    dist = np.stack([r[0] for r in res]); goal_m = np.array([r[1] for r in res], np.float32)
    counts = np.array([len(r[2]) for r in res], np.int64)
    out = dict(dist=dist, goal_m=goal_m, path=np.concatenate([r[2] for r in res]), offset=np.concatenate([[0], np.cumsum(counts)]))
    suffix = "" if limit is None else f"_first{N}"
    np.savez_compressed(store / f"c5_wave_{split}{suffix}.npz", **out)
    reached = np.isfinite(goal_m)
    stats = dict(split=split, n=N, seconds=round(time.perf_counter() - t0, 1), workers=workers, margin_m=MARGIN_M,
                 lam=T.LAM, cap_m=T.CAP, share_goal_unreachable=float((~reached).mean()),
                 goal_m_median=float(np.median(goal_m[reached])), goal_m_p90=float(np.percentile(goal_m[reached], 90)),
                 goal_m_max=float(goal_m[reached].max()),
                 passes_at_40m_median=float(np.median(np.floor(goal_m[reached] / 40) + 1)),
                 passes_at_40m_max=float((np.floor(goal_m[reached] / 40) + 1).max()),
                 share_cells_reached=float((~unreached_mask(dist)).mean()), path_px_mean=float(counts.mean()),
                 crop_px=crop, dist_dtype=str(dist.dtype),
                 # a store whose dtype is too small for its distances would show cells at the top value
                 share_cells_at_top=float((dist == np.iinfo(dist.dtype).max - 1).mean()))
    (store / f"c5_wave_{split}{suffix}_stats.json").write_text(json.dumps(stats, indent=1))
    print(split, stats)
    if gallery:
        draw_gallery(store, split, suffix, min(gallery, N))


def draw_gallery(store: Path, split: str, suffix: str, n: int, horizon: float = 40.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spatial_data.dataset_c4_gpu import store_crop_px
    C = store_crop_px(store)
    bld = np.load(store.parent / "district_labels_1m.npz")["surface"] == BUILDING_CLASS
    z = np.load(store / f"c4_routes_{split}.npz"); w = np.load(store / f"c5_wave_{split}{suffix}.npz")
    cols = 6; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows))
    for k, ax in enumerate(np.ravel(axes)):
        if k >= n: ax.axis("off"); continue
        dx, dy = z["origin"][k]
        d = w["dist"][k].astype(float) * 2.0; d[unreached_mask(w["dist"][k])] = np.nan
        ax.imshow(bld[dy:dy + C, dx:dx + C], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        ax.imshow(np.repeat(np.repeat(np.floor(d / horizon), 2, 0), 2, 1), cmap="viridis", alpha=0.6, interpolation="nearest")
        p = w["path"][w["offset"][k]:w["offset"][k + 1]]
        ax.plot(p[:, 0], p[:, 1], "-", color="white", lw=1.5)
        ax.plot(*z["markers"][k, 0], "o", color="gold", ms=5, mec="k"); ax.plot(*z["markers"][k, 1], "*", color="tab:green", ms=9, mec="k")
        ax.set_title(f"#{k}: goal at {w['goal_m'][k]:.0f} m, {int(np.floor(w['goal_m'][k] / horizon)) + 1} passes", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C5 R5 data: walking-distance wave from the start (bands of {horizon:.0f} m), the way back (white), goal (star)", fontsize=11)
    fig.tight_layout(); out = store / f"c5_wave_{split}{suffix}_gallery.png"; fig.savefig(out, dpi=100); print("gallery ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--splits", default="val")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--gallery", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for split in a.splits.split(","):
        build(Path(a.store), split, a.limit, a.gallery, a.workers)


if __name__ == "__main__":
    main()
