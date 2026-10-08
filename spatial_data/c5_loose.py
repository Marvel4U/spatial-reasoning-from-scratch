"""C5 rung 4 inputs: loose versions of the stored shortest paths, for the "tighten" task.

For each store sample, 1-3 extra waypoints are inserted into the stored optimal polyline. A waypoint sits
on a random piece of the path, pushed sideways by 4-25 px. It is kept only if
  - it lies on free ground and both new pieces have line of sight (the raster test of c4_routes), and
  - the loose path goes around the buildings the same way as the optimal path: the closed loop formed by
    the loose path and the optimal path reversed encloses no building pixel.
The second condition makes the tightened loose path equal to the stored optimal path, so the task target is
the stored path itself and no new target has to be computed.

Output next to the store: `c5_loose_{split}.npz` with
  loose_poly (N, V, 2) float32 (-1 padded), loose_n (N,) int16, loose_px (N,) float32 (length),
  n_way (N,) int8 (waypoints actually inserted; 0 = generation failed, the loose path equals the optimal one)

Usage (project venv, repo root):
  python -m spatial_data.c5_loose --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val --limit 200 --gallery 12
  python -m spatial_data.c5_loose --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val,test,train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from spatial_data import c4_routes as R

OFFSET_PX = (4.0, 25.0)
MAX_TRIES = 30


def _free(mask, p):
    x, y = int(np.floor(p[0])), int(np.floor(p[1]))
    return 0 <= x < R.CROP_PX and 0 <= y < R.CROP_PX and not mask[y, x]


def _encloses_building(loop: np.ndarray, mask: np.ndarray) -> bool:
    """True if the closed polygon `loop` (k, 2) contains any building pixel centre."""
    x0, y0 = np.floor(loop.min(0)).astype(int); x1, y1 = np.ceil(loop.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, R.CROP_PX - 1), min(y1, R.CROP_PX - 1)
    sub = mask[y0:y1 + 1, x0:x1 + 1]
    if not sub.any():
        return False
    ys, xs = np.nonzero(sub)
    pts = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], axis=1)
    return bool(MplPath(loop).contains_points(pts).any())


def loosen(poly: np.ndarray, mask: np.ndarray, rng: np.random.Generator, n_way: int):
    """Insert up to n_way waypoints into poly (k, 2). Returns (loose (k', 2), inserted)."""
    cur = poly.copy(); inserted = 0
    for _ in range(n_way):
        for _try in range(MAX_TRIES):
            i = int(rng.integers(0, len(cur) - 1)); a, b = cur[i], cur[i + 1]
            d = b - a; L = float(np.linalg.norm(d))
            if L < 8.0:
                continue
            t = rng.uniform(0.2, 0.8); n = np.array([-d[1], d[0]]) / L
            w = a + t * d + n * rng.choice([-1.0, 1.0]) * rng.uniform(*OFFSET_PX)
            if not _free(mask, w):
                continue
            if not (R.visible_from(a, w[None], mask)[0] and R.visible_from(w, b[None], mask)[0]):
                continue
            cand = np.vstack([cur[:i + 1], w[None], cur[i + 1:]])
            loop = np.vstack([cand, poly[::-1]])
            if _encloses_building(loop, mask):
                continue
            cur = cand; inserted += 1
            break
    return cur, inserted


def build(store: Path, split: str, limit: int | None, gallery: int, seed: int = 20260929):
    zf = np.load(store / f"c4_routes_{split}.npz")
    z = {k: zf[k] for k in zf.files}      # decompress once: indexing an NpzFile re-reads the whole array every time
    building = np.load(store.parent / "district_labels_1m.npz")["surface"] == R.BUILDING_CLASS
    N = z["origin"].shape[0] if limit is None else min(limit, z["origin"].shape[0])
    V = z["poly"].shape[1] + 3
    rng = np.random.default_rng([seed, {"train": 0, "val": 1, "test": 2}.get(split, 3)])
    loose = np.full((N, V, 2), -1.0, np.float32); ln = np.zeros(N, np.int16); lpx = np.zeros(N, np.float32); nw = np.zeros(N, np.int8)
    t0 = time.perf_counter()
    for i in range(N):
        dx, dy = z["origin"][i]; mask = building[dy:dy + R.CROP_PX, dx:dx + R.CROP_PX]
        poly = z["poly"][i, :z["n_verts"][i]].astype(np.float64)
        cur, ins = loosen(poly, mask, rng, int(rng.integers(1, 4)))
        loose[i, :len(cur)] = cur; ln[i] = len(cur); nw[i] = ins
        lpx[i] = float(np.linalg.norm(np.diff(cur, axis=0), axis=1).sum())
        if (i + 1) % 5000 == 0:
            print(f"  {split}: {i + 1}/{N}  {1000 * (time.perf_counter() - t0) / (i + 1):.1f} ms/sample", flush=True)
    ratio = lpx / np.maximum(z["path_px"][:N], 1e-6)
    stats = dict(split=split, n=int(N), seconds=round(time.perf_counter() - t0, 1),
                 share_failed=float((nw == 0).mean()), mean_waypoints=float(nw.mean()),
                 looseness_median=float(np.median(ratio)), looseness_p90=float(np.percentile(ratio, 90)),
                 looseness_max=float(ratio.max()))
    suffix = "" if limit is None else f"_first{N}"
    np.savez_compressed(store / f"c5_loose_{split}{suffix}.npz", loose_poly=loose, loose_n=ln, loose_px=lpx, n_way=nw)
    (store / f"c5_loose_{split}{suffix}_stats.json").write_text(json.dumps(stats, indent=1))
    print(split, stats)
    if gallery:
        draw_gallery(z, building, loose, ln, min(gallery, N), store / f"c5_loose_{split}{suffix}_gallery.png")


def draw_gallery(z, building, loose, ln, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 6; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.9 * rows))
    for k, ax in enumerate(axes.ravel()):
        if k >= n: ax.axis("off"); continue
        dx, dy = z["origin"][k]; ax.imshow(building[dy:dy + 256, dx:dx + 256], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        opt = z["poly"][k, :z["n_verts"][k]]; lo = loose[k, :ln[k]]
        ax.plot(lo[:, 0], lo[:, 1], "-", color="tab:orange", lw=1.6, label="loose (input)")
        ax.plot(opt[:, 0], opt[:, 1], "-", color="tab:blue", lw=1.0, label="tight (target)")
        ax.plot(*opt[0], "o", color="k", ms=4); ax.plot(*opt[-1], "o", color="k", ms=4)
        ax.set_title(f"#{k}: +{ln[k] - len(opt)} waypoints", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("C5 rung 4: loose paths (input, orange) and the tight path they must become (target, blue)", fontsize=11)
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
