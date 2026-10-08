"""C5 rung 1 targets: for every sample of a c4 route store and each of its two markers, the building
corners the marker can see within a radius and whose ray grazes the building ("the fan"). The ray task
draws the straight segments from the marker to these corners.

Grazing: the ray continued 1.5, 3 and 4.5 px past the corner stays on free ground, i.e. the building's
two edges at the corner lie on the same side of the ray. Only such corners can be a turning point of a
shortest path from the marker; at the others the ray runs into the building right behind the corner.
The filter keeps ~40 % of the visible corners (val: 15.8 -> 6.3 rays per marker, p90 28 -> 12; 29 Sep 2026).

Corners and visibility are exactly those of the c4 route builder (`spatial_data/c4_routes.py`):
corners of the simplified building outlines pushed 1.2 px outwards, line of sight sampled on the
building raster every 0.5 px. So a ray in this task is an edge the path search could take.

Output next to the store: `c5_fans_{split}.npz` with
  fan_xy  (N, 2, K, 2) float32  corner coordinates (x, y) in window pixels, -1 padded
  fan_n   (N, 2)       int16    number of corners per marker (<= K)
  fan_all (N, 2)       int16    number of grazing visible corners before the K cap (for the record)
  fan_vis (N, 2)       int16    number of visible corners before the grazing filter (for the record)
Rays are kept nearest-first when a marker sees more than K corners.

Usage (project venv, repo root):
  python -m spatial_data.c5_fans --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val --limit 200 --gallery 12
  python -m spatial_data.c5_fans --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits train,val,test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spatial_data import c4_routes as R

RADIUS_PX = 80.0     # fan radius (1 px = 1 m); a task parameter, see C5_PLAN §4.1
K_MAX = 48           # rays kept per marker, nearest first
BEYOND_PX = (1.5, 3.0, 4.5)   # points past the corner that must be free for a grazing ray


def grazing(p: np.ndarray, corners: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean over corners: the ray p -> corner continued BEYOND_PX past the corner stays off buildings
    (points leaving the window count as free)."""
    u = corners - p
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    ok = np.ones(len(corners), dtype=bool)
    for d in BEYOND_PX:
        q = corners + d * u
        inside = (q[:, 0] >= 0) & (q[:, 0] < R.CROP_PX) & (q[:, 1] >= 0) & (q[:, 1] < R.CROP_PX)
        xi = np.clip(np.floor(q[:, 0]).astype(np.int64), 0, R.CROP_PX - 1)
        yi = np.clip(np.floor(q[:, 1]).astype(np.int64), 0, R.CROP_PX - 1)
        ok &= ~inside | ~mask[yi, xi]
    return ok


def fans_for_window(mask: np.ndarray, markers: np.ndarray, radius: float = RADIUS_PX, k_max: int = K_MAX):
    _, nodes = R.window_obstacles(mask)
    out = np.full((2, k_max, 2), -1.0, dtype=np.float32); n = np.zeros(2, np.int16); n_all = np.zeros(2, np.int16)
    n_vis = np.zeros(2, np.int16)
    if len(nodes) == 0:
        return out, n, n_all, n_vis
    for m in range(2):
        p = markers[m].astype(np.float64)
        d = np.linalg.norm(nodes - p, axis=1)
        near = nodes[d <= radius]; dn = d[d <= radius]
        if len(near) == 0:
            continue
        vis = R.visible_from(p, near, mask)
        n_vis[m] = int(vis.sum())
        vis &= grazing(p, near, mask)
        cand = near[vis]; dc = dn[vis]
        n_all[m] = len(cand)
        order = np.argsort(dc)[:k_max]
        out[m, :len(order)] = cand[order]; n[m] = len(order)
    return out, n, n_all, n_vis


def build(store: Path, split: str, limit: int | None, gallery: int):
    zf = np.load(store / f"c4_routes_{split}.npz")
    z = {k: zf[k] for k in zf.files}      # decompress once: indexing an NpzFile re-reads the whole array every time
    building = np.load(store.parent / "district_labels_1m.npz")["surface"] == R.BUILDING_CLASS
    N = z["origin"].shape[0] if limit is None else min(limit, z["origin"].shape[0])
    fan_xy = np.full((N, 2, K_MAX, 2), -1.0, np.float32); fan_n = np.zeros((N, 2), np.int16); fan_all = np.zeros((N, 2), np.int16)
    fan_vis = np.zeros((N, 2), np.int16)
    t0 = time.perf_counter()
    for i in range(N):
        dx, dy = z["origin"][i]; mask = building[dy:dy + R.CROP_PX, dx:dx + R.CROP_PX]
        fan_xy[i], fan_n[i], fan_all[i], fan_vis[i] = fans_for_window(mask, z["markers"][i])
        if (i + 1) % 2000 == 0:
            el = time.perf_counter() - t0
            print(f"  {split}: {i + 1}/{N}  {1000 * el / (i + 1):.0f} ms/sample", flush=True)
    stats = dict(split=split, n=int(N), radius_px=RADIUS_PX, k_max=K_MAX, seconds=round(time.perf_counter() - t0, 1),
                 mean_rays=float(fan_n.mean()), median_rays=float(np.median(fan_n)), p90_rays=float(np.percentile(fan_n, 90)),
                 share_capped=float((fan_all > K_MAX).mean()), share_empty=float((fan_n == 0).mean()),
                 rule="grazing rays only", mean_visible_before_grazing=float(fan_vis.mean()))
    suffix = "" if limit is None else f"_first{N}"
    np.savez_compressed(store / f"c5_fans_{split}{suffix}.npz", fan_xy=fan_xy, fan_n=fan_n, fan_all=fan_all, fan_vis=fan_vis)
    (store / f"c5_fans_{split}{suffix}_stats.json").write_text(json.dumps(stats, indent=1))
    print(split, stats)
    if gallery:
        draw_gallery(z, building, fan_xy, fan_n, min(gallery, N), store / f"c5_fans_{split}{suffix}_gallery.png")


def draw_gallery(z, building, fan_xy, fan_n, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 6; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.9 * rows))
    for k, ax in enumerate(axes.ravel()):
        if k >= n: ax.axis("off"); continue
        dx, dy = z["origin"][k]; ax.imshow(building[dy:dy + 256, dx:dx + 256], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        for m, col in ((0, "tab:green"), (1, "tab:blue")):
            p = z["markers"][k, m]
            for q in fan_xy[k, m, :fan_n[k, m]]:
                ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=col, lw=0.7, alpha=0.8)
            ax.plot(*p, "o", color=col, ms=5, mec="k")
        ax.set_title(f"#{k}: {fan_n[k, 0]} + {fan_n[k, 1]} rays", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C5 rung 1 targets: grazing rays from each marker to the building corners it sees within {RADIUS_PX:.0f} m", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=100); print("gallery ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--splits", default="val")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--gallery", type=int, default=0)
    a = ap.parse_args()
    for split in a.splits.split(","):
        build(Path(a.store), split, a.limit, a.gallery)


if __name__ == "__main__":
    main()
