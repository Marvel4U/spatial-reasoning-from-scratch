"""c4 sample store: two markers per 256 m window, the straight segment (c4a) and the
Euclidean shortest path around buildings (c4b), computed on a visibility graph.

Why a visibility graph and not A* on the pixel grid: on an 8-connected grid every monotone
staircase between two points has the same length, so "the shortest path" is one of thousands
and the target would be arbitrary. The Euclidean shortest path around polygons is the taut
string: straight where it can be, bending only at building corners, unique in general.

Obstacles are the building pixels of the district surface plane (class 3), polygonised per
window and lightly simplified. Nodes of the visibility graph are the corners of the polygons
pushed outwards by NODE_PUSH_PX; two nodes see each other if the segment between them does not
enter any polygon shrunk by LOS_SHRINK_PX (so a path may run along a wall but not through it).
The search is a lazy A*: neighbours are computed only for expanded nodes, candidates are the
nodes inside the ellipse |pa| + |pb| <= ELLIPSE * |ab|.

Output (npz, per split): window origins, markers, segment and path polylines, rasterised
path pixels, lengths, detour ratio, plus rejection statistics in a JSON sidecar.

Usage (project venv):
  python -m spatial_data.c4_routes --n-train 20000 --n-val 500 --n-test 500 --out data/.../c4_routes
  python -m spatial_data.c4_routes --n-train 200 --n-val 50 --overlays 8   # smoke + pictures
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import shapely
from rasterio import features
from scipy import ndimage
from shapely.geometry import LineString, Polygon, shape
from shapely.strtree import STRtree

CROP_PX = 256
BUILDING_CLASS = 3
EDGE_MARGIN_PX = 8          # markers at least this far from the window edge
NODE_PUSH_PX = 1.2          # corners pushed outwards so the path clears the (simplified) wall
LOS_SHRINK_PX = 0.3         # polygons shrunk for the line-of-sight test
SIMPLIFY_PX = 1.0           # strip the 1 m staircase from rasterised outlines
MAX_EXPANSIONS = 600        # A* gives up (sample rejected) beyond this
ELLIPSE = 2.2               # candidate nodes: |pa| + |pb| <= ELLIPSE * |ab|
D_MIN_PX, D_MAX_PX = 40.0, 250.0
MAX_PATH_PIX = 640          # padded pixel list length (path <= ELLIPSE * 250 m)
MAX_VERTS = 64
STRATA = ((1.0, 1.05), (1.05, 1.3), (1.3, 99.0))   # detour-ratio strata, equal shares


def load_district(crops_dir: Path):
    z = np.load(crops_dir / "district_labels_1m.npz")
    building = z["surface"] == BUILDING_CLASS
    split = json.loads((crops_dir / "split.json").read_text())
    return building, split


def block_rects_px(split: dict, which: str) -> list[tuple[int, int, int, int]]:
    bw, bh = split["block_m"]
    rects = []
    for r, c in split[f"{which}_blocks"]:
        x0, y0 = c * bw, r * bh
        rects.append((int(round(x0)), int(round(y0)), int(round(x0 + bw)), int(round(y0 + bh))))
    return rects


class WindowSampler:
    """Train: windows that touch no held-out block. Val/test: windows fully inside a block of that split."""

    def __init__(self, split: dict, h: int, w: int, rng: np.random.Generator):
        self.rng = rng
        self.h, self.w = h, w
        self.heldout = block_rects_px(split, "val") + block_rects_px(split, "test")
        self.val = block_rects_px(split, "val")
        self.test = block_rects_px(split, "test")

    def draw(self, which: str) -> tuple[int, int]:
        if which == "train":
            while True:
                dx = int(self.rng.integers(0, self.w - CROP_PX + 1))
                dy = int(self.rng.integers(0, self.h - CROP_PX + 1))
                if not any(dx < x1 and dx + CROP_PX > x0 and dy < y1 and dy + CROP_PX > y0
                           for x0, y0, x1, y1 in self.heldout):
                    return dx, dy
        rects = self.val if which == "val" else self.test
        x0, y0, x1, y1 = rects[int(self.rng.integers(0, len(rects)))]
        dx = int(self.rng.integers(x0, x1 - CROP_PX + 1))
        dy = int(self.rng.integers(y0, y1 - CROP_PX + 1))
        return dx, dy


def window_obstacles(mask: np.ndarray):
    """Polygonise the building mask of one window. Returns (los_polys, nodes) with nodes (n, 2) as (x, y)."""
    exact, polys = [], []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, connectivity=4):
        if val != 1:
            continue
        e = shape(geom)
        if e.is_empty or e.area < 2.0:
            continue
        exact.append(e)
        polys.append(e.simplify(SIMPLIFY_PX, preserve_topology=True))
    if not polys:
        return [], np.zeros((0, 2), dtype=np.float64)
    # line of sight is tested against the exact raster outline (shrunk a little so a path may run
    # along a wall); the simplified outline only supplies the corner nodes
    los = [p.buffer(-LOS_SHRINK_PX) for p in exact]
    los = [p for p in los if not p.is_empty]
    pushed = [p.buffer(NODE_PUSH_PX, join_style="mitre", mitre_limit=3.0) for p in polys]
    pts = []
    for p in pushed:
        geoms = p.geoms if hasattr(p, "geoms") else [p]
        for g in geoms:
            for ring in [g.exterior, *g.interiors]:
                # all corners of the pushed, simplified outline. A convex-only filter was tried and
                # dropped corners the optimal path needs (24 Sep: paths 1.4-2.4x too long on 58 % of
                # samples against a dense grid search); the raster line-of-sight test is cheap enough.
                pts.extend(tuple(v) for v in np.asarray(ring.coords[:-1]))
    nodes = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))
    # drop nodes that fall outside the window or inside another building
    if len(nodes):
        inside = (nodes[:, 0] >= 0.5) & (nodes[:, 0] <= CROP_PX - 0.5) & (nodes[:, 1] >= 0.5) & (nodes[:, 1] <= CROP_PX - 0.5)
        nodes = nodes[inside]
        free = ~mask[np.floor(nodes[:, 1]).astype(int), np.floor(nodes[:, 0]).astype(int)]
        nodes = nodes[free]
    return los, nodes


def convex_corners(ring: np.ndarray, *, is_hole: bool) -> list:
    """Vertices where the obstacle is convex (a shortest path only bends there).

    Shapely rings have no fixed orientation, so the sign of the ring's signed area says which
    way it turns; a corner is convex for the solid if its turn has the same sign as the ring's
    overall turn (exterior) or the opposite sign (hole: the free space is inside the ring)."""
    if len(ring) < 3:
        return []
    x, y = ring[:, 0], ring[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    prev = np.roll(ring, 1, axis=0); nxt = np.roll(ring, -1, axis=0)
    cross = (ring[:, 0] - prev[:, 0]) * (nxt[:, 1] - ring[:, 1]) - (ring[:, 1] - prev[:, 1]) * (nxt[:, 0] - ring[:, 0])
    convex = (cross * area2 > 0) if not is_hole else (cross * area2 < 0)
    return [tuple(v) for v in ring[convex]]


def same_component(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Cheap reachability: both endpoints in one 4-connected component of the free pixels."""
    lab, _ = ndimage.label(~mask)
    return lab[int(a[1]), int(a[0])] == lab[int(b[1]), int(b[0])] != 0


def visible_from(p: np.ndarray, targets: np.ndarray, mask: np.ndarray, _unused=None) -> np.ndarray:
    """Boolean mask over targets: the segment p->target crosses no building pixel.

    Tested on the raster itself (0.5 px sampling), which is what the model sees and what the
    rasterised target is checked against; polygons only supply the corner nodes."""
    if len(targets) == 0:
        return np.zeros(0, dtype=bool)
    d = np.linalg.norm(targets - p, axis=1)
    L = int(np.ceil(d.max() * 2)) + 2
    t = np.linspace(0.0, 1.0, L)[None, :, None]
    pts = p[None, None, :] + t * (targets - p)[:, None, :]
    xi = np.clip(np.floor(pts[..., 0]).astype(np.int64), 0, CROP_PX - 1)
    yi = np.clip(np.floor(pts[..., 1]).astype(np.int64), 0, CROP_PX - 1)
    return ~mask[yi, xi].any(axis=1)


def shortest_path(a: np.ndarray, b: np.ndarray, nodes: np.ndarray, mask: np.ndarray, _unused=None):
    """Lazy A* on the visibility graph. Returns polyline (k, 2) or None."""
    ab = float(np.linalg.norm(b - a))
    if len(nodes):
        d = np.linalg.norm(nodes - a, axis=1) + np.linalg.norm(nodes - b, axis=1)
        cand = nodes[d <= ELLIPSE * ab]
    else:
        cand = nodes
    pts = np.vstack([a[None], b[None], cand])          # 0 = start, 1 = goal
    n = len(pts)
    if visible_from(a, b[None], mask)[0]:
        return pts[:2].copy()
    g = np.full(n, np.inf); g[0] = 0.0
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)
    h = np.linalg.norm(pts - b, axis=1)
    heap = [(h[0], 0)]; expansions = 0
    while heap:
        f, i = heapq.heappop(heap)
        if closed[i]:
            continue
        closed[i] = True; expansions += 1
        if expansions > MAX_EXPANSIONS:
            return None
        if i == 1:
            path = [1]
            while path[-1] != 0:
                path.append(int(parent[path[-1]]))
            return pts[path[::-1]].copy()
        vis = visible_from(pts[i], pts, mask)
        vis[i] = False
        vis &= ~closed
        idx = np.nonzero(vis)[0]
        if len(idx) == 0:
            continue
        dist = np.linalg.norm(pts[idx] - pts[i], axis=1)
        newg = g[i] + dist
        better = newg < g[idx]
        for j, gj in zip(idx[better], newg[better]):
            g[j] = gj; parent[j] = i
            heapq.heappush(heap, (gj + h[j], int(j)))
    return None


def rasterise(poly: np.ndarray) -> np.ndarray:
    """Dense 1-px samples along the polyline -> unique integer pixels (x, y) in order."""
    out = []
    for p, q in zip(poly[:-1], poly[1:]):
        L = max(1, int(math.ceil(np.linalg.norm(q - p) * 2)))
        t = np.linspace(0.0, 1.0, L + 1)[:, None]
        out.append(p[None] + t * (q - p)[None])
    pts = np.floor(np.vstack(out)).astype(np.int64)
    pts = np.clip(pts, 0, CROP_PX - 1)
    _, first = np.unique(pts, axis=0, return_index=True)
    return pts[np.sort(first)]


def stratum_of(r: float) -> int:
    for i, (lo, hi) in enumerate(STRATA):
        if lo <= r < hi:
            return i
    return len(STRATA) - 1


def build_split(which: str, n: int, building: np.ndarray, sampler: WindowSampler, rng: np.random.Generator,
                overlays: int, out_dir: Path):
    H, W = building.shape
    recs = []; quota = [n // len(STRATA)] * len(STRATA); quota[0] += n - sum(quota)
    counts = [0] * len(STRATA)
    stats = dict(windows=0, no_free_pixels=0, blocked=0, gave_up=0, path_hits_building=0, stratum_full=0, too_long=0)
    t0 = time.perf_counter(); cache = {}
    while sum(counts) < n:
        dx, dy = sampler.draw(which); stats["windows"] += 1
        mask = building[dy:dy + CROP_PX, dx:dx + CROP_PX]
        key = (dx, dy)
        if key not in cache:
            los, nodes = window_obstacles(mask)
            tree = None
            lab, _ = ndimage.label(~mask)
            cache = {key: (los, nodes, tree, lab)}     # one window at a time
        los, nodes, tree, lab = cache[key]
        # endpoints at least 2 px from any building: the simplified outlines move the wall by up to
        # SIMPLIFY_PX, and a marker inside a simplified polygon sees nothing
        free_mask = ndimage.binary_erosion(~mask, iterations=2)
        free = np.argwhere(free_mask[EDGE_MARGIN_PX:-EDGE_MARGIN_PX, EDGE_MARGIN_PX:-EDGE_MARGIN_PX]) + EDGE_MARGIN_PX
        if len(free) < 2:
            stats["no_free_pixels"] += 1; continue
        # a few endpoint pairs per window
        for _ in range(4):
            ia, ib = rng.integers(0, len(free), 2)
            a = free[ia][::-1].astype(np.float64) + 0.5; b = free[ib][::-1].astype(np.float64) + 0.5   # (x, y)
            ab = float(np.linalg.norm(b - a))
            if not (D_MIN_PX <= ab <= D_MAX_PX):
                continue
            if lab[int(a[1]), int(a[0])] != lab[int(b[1]), int(b[0])]:
                stats["blocked"] += 1; continue
            poly = shortest_path(a, b, nodes, mask)
            if poly is None:
                stats["gave_up"] += 1; continue
            seg = np.diff(poly, axis=0); plen = float(np.sqrt((seg ** 2).sum(1)).sum())
            ratio = plen / ab
            s = stratum_of(ratio)
            if counts[s] >= quota[s]:
                stats["stratum_full"] += 1; continue
            pix = rasterise(poly)
            if len(pix) > MAX_PATH_PIX or len(poly) > MAX_VERTS:
                stats["too_long"] += 1; continue
            if mask[pix[:, 1], pix[:, 0]].any():
                stats["path_hits_building"] += 1; continue
            counts[s] += 1
            recs.append(dict(dx=dx, dy=dy, a=a, b=b, poly=poly, pix=pix, ab=ab, plen=plen, ratio=ratio, stratum=s))
            if len(recs) % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  {which}: {len(recs)}/{n} samples, {el:.0f}s, {1000 * el / len(recs):.0f} ms/sample, strata {counts}, stats {stats}", flush=True)
    N = len(recs)
    out = dict(
        origin=np.array([[r["dx"], r["dy"]] for r in recs], dtype=np.int32),
        markers=np.array([[r["a"], r["b"]] for r in recs], dtype=np.float32),          # (N, 2, 2) x,y in window px
        poly=np.full((N, MAX_VERTS, 2), -1, dtype=np.float32),
        n_verts=np.array([len(r["poly"]) for r in recs], dtype=np.int16),
        path_pix=np.full((N, MAX_PATH_PIX, 2), -1, dtype=np.int16),                    # (x, y)
        n_pix=np.array([len(r["pix"]) for r in recs], dtype=np.int16),
        straight_px=np.array([r["ab"] for r in recs], dtype=np.float32),
        path_px=np.array([r["plen"] for r in recs], dtype=np.float32),
        detour_ratio=np.array([r["ratio"] for r in recs], dtype=np.float32),
        stratum=np.array([r["stratum"] for r in recs], dtype=np.int8),
    )
    for i, r in enumerate(recs):
        out["poly"][i, :len(r["poly"])] = r["poly"]; out["path_pix"][i, :len(r["pix"])] = r["pix"]
    np.savez_compressed(out_dir / f"c4_routes_{which}.npz", **out)
    stats.update(n=N, seconds=round(time.perf_counter() - t0, 1), strata=counts,
                 mean_detour=float(np.mean(out["detour_ratio"])), mean_path_m=float(np.mean(out["path_px"])))
    (out_dir / f"c4_routes_{which}_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"{which}: {N} samples -> {out_dir / f'c4_routes_{which}.npz'}  {stats}")
    if overlays:
        by_s = [[r for r in recs if r["stratum"] == k] for k in range(len(STRATA))]
        pick = [r for k in range(overlays) for r in by_s[k % len(STRATA)][k // len(STRATA):k // len(STRATA) + 1]]
        draw_overlays(pick[:overlays], building, out_dir / f"c4_overlay_{which}.png")


def draw_overlays(recs, building, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(recs); fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.6))
    for ax, r in zip(axes.ravel(), recs):
        m = building[r["dy"]:r["dy"] + CROP_PX, r["dx"]:r["dx"] + CROP_PX]
        ax.imshow(m, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        ax.plot([r["a"][0], r["b"][0]], [r["a"][1], r["b"][1]], "c--", lw=1, label="segment")
        ax.plot(r["poly"][:, 0], r["poly"][:, 1], "r-", lw=1.4, label="path")
        ax.plot(*r["a"], "go", ms=5); ax.plot(*r["b"], "bo", ms=5)
        ax.set_title(f"{r['ab']:.0f} m straight, x{r['ratio']:.2f}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=110); print("overlay ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", default="data/amsterdam/de_pijp/crops/v4")
    ap.add_argument("--out", default="data/amsterdam/de_pijp/crops/v4/c4")
    ap.add_argument("--n-train", type=int, default=200); ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=0); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--overlays", type=int, default=8)
    args = ap.parse_args()
    crops_dir = Path(args.crops_dir); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    building, split = load_district(crops_dir)
    rng = np.random.default_rng(args.seed)
    sampler = WindowSampler(split, *building.shape, rng)
    print(f"district {building.shape}, building share {building.mean():.3f}, heldout rects {sampler.heldout}")
    for which, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        if n > 0:
            build_split(which, n, building, sampler, rng, args.overlays, out_dir)


if __name__ == "__main__":
    main()
