"""C5 rung 1 targets, second definition (30 Sep / 1 Oct 2026): key trajectories from a marker, in two rounds.

Round 1, from the marker:
  1. clearance = distance of every free pixel to the nearest building pixel (buildings are the only obstacle)
  2. price per metre walked: 1 + LAM * (1 - min(clearance, CAP) / CAP) ** 2  (5 at a wall, 1 from 8 m on)
  3. one cheapest-path wave from the marker over the free pixels, 8 neighbours (straight 1 m, diagonal sqrt 2 m,
     each step priced by the mean of its two pixels); every pixel keeps the pixel it was reached from (a tree)
  4. walking length = real metres along each pixel's cheapest path; only pixels with length <= the horizon are kept
  5. repeat until every kept pixel is covered: among the uncovered pixels whose walking length is within
     FRONT_SHARE of the longest, take the one farthest from every trajectory drawn so far (on an empty field this
     gives the four directions: first north by pixel order, then south, west, east); trace its path back; cover a
     band of radius min(TUBE_A * clearance + TUBE_B, BAND_CAP_SHARE * horizon) around the whole path (straight-line
     distance to the nearest path pixel, clearance taken there); keep the path as a key trajectory only if its own
     part (from where it leaves covered ground) is >= L_MIN_SHARE * horizon
Rounds 2-4 (Marvin, 1 Oct: "branching out"), each from every end of the previous round in order: the same wave
and rule from the end pixel, with everything covered so far still covered and all trajectories so far counting as
"drawn" (so new trajectories only lead into new ground and the ends do not duplicate each other). Each round is
capped at K_MAX trajectories (ends are taken in order). The rounds are stored apart.
Only marker 0 is used (Marvin, 1 Oct: the second marker is not needed for this rung; --markers 2 keeps both).
The wave runs on the district map with a margin of one horizon around the window (Marvin, 1 Oct: a marker near the
window edge must still explore towards and beyond the edge); the stored pixels are clipped to the window, ends are
kept in window coordinates even when they lie outside it.

Output next to the store: `c5_traj{suffix}_{split}.npz` (suffix "" at 100 m, "_r40" at 40 m, ...)
  pix     (M, 2) int16   x, y of every round-1 trajectory pixel (per marker duplicates removed)
  pix_tag (M,)   int8    marker * 32 + trajectory index (picking order, < 32)
  offset  (N+1,) int64   sample i owns pix[offset[i]:offset[i+1]]
  n_traj  (N, 2) int8    round-1 trajectories per marker
  ends    (N, 2, 32, 2) int16  end pixel of each round-1 trajectory (-1 padded)
  pix2, tag2, offset2, n_traj2 (N,), ends2 (N, 32, 2): the same for round 2 (marker 0; tag2 = index of the
        previous-round end it starts from); pix3 ... ends3 and pix4 ... ends4 for rounds 3 and 4
Usage (project venv, repo root):
  python -m spatial_data.c5_trajectories --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val --limit 200 --gallery 12 --radius 40
  python -m spatial_data.c5_trajectories --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits val,test,train --workers 4 --radius 40
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

RADIUS_M, LAM, CAP = 100.0, 4.0, 8.0
TUBE_A, TUBE_B = 1.5, 3.0
L_MIN_SHARE = 0.15                # minimum own branch length as a share of the horizon: 15 m at 100 m
BAND_CAP_SHARE = 0.75             # the band never exceeds this share of the horizon (open ground: four directions)
FRONT_SHARE = 0.05                # "as far as the farthest": within this share of the horizon of the longest length
L_MIN_M = L_MIN_SHARE * RADIUS_M
K_MAX = 32
N_ROUNDS = 4
BUILDING_CLASS = 3
_OFFS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def set_radius(radius_m: float) -> None:
    """--radius: horizon in metres; the minimum branch length follows (L_MIN_SHARE of it)."""
    global RADIUS_M, L_MIN_M
    RADIUS_M = float(radius_m); L_MIN_M = L_MIN_SHARE * RADIUS_M


def file_suffix(radius_m: float | None = None) -> str:
    r = RADIUS_M if radius_m is None else radius_m
    return "" if int(round(r)) == 100 else f"_r{int(round(r))}"


def _graph(free: np.ndarray, cost: np.ndarray):
    H, W = free.shape
    idx = -np.ones((H, W), np.int64); fy, fx = np.nonzero(free); idx[fy, fx] = np.arange(len(fy))
    rows, cols, w = [], [], []
    for dy, dx in _OFFS:
        ys, xs = fy + dy, fx + dx
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ok[ok] &= free[ys[ok], xs[ok]]
        rows.append(idx[fy[ok], fx[ok]]); cols.append(idx[ys[ok], xs[ok]])
        w.append(np.hypot(dy, dx) * 0.5 * (cost[fy[ok], fx[ok]] + cost[ys[ok], xs[ok]]))
    G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(fy),) * 2)
    return G, idx, fy, fx


def _walk_length(pred: np.ndarray, reached: np.ndarray, fy: np.ndarray, fx: np.ndarray, src: int) -> np.ndarray:
    """Real metres from the source along each pixel's tree path (pointer doubling, no Python loop over pixels)."""
    n = len(pred)
    par = np.where(reached & (pred >= 0), pred, np.arange(n))
    par[src] = src
    step = np.where(par != np.arange(n), np.hypot(fy - fy[par], fx - fx[par]), 0.0)
    L, P = step.copy(), par.copy()
    # pointer doubling until every pixel points at its root. It used to stop after 12 rounds (paths up to 4096 steps),
    # enough under the cost limit of the key trajectories; the uncapped wave of c5_wave.py on a 512 m window plus
    # margin (712 px, a maze) can have longer tree paths, which then read as unreached. Extra rounds change nothing
    # for pixels that had already arrived (they add L[root] = 0).
    while True:
        L = L + L[P] * (P != np.arange(n)); P = P[P]
        if np.array_equal(P, P[P]):
            break
    # rounded to 1 um: equal lengths (sums of 1 and sqrt 2) must tie exactly whatever the summation order
    return np.where(reached & (P == src), np.round(L, 6), np.inf)


class _Field:
    """One window: free-pixel graph, clearance, and the shared state of the rounds (covered, distance to drawn)."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask; self.free = ~mask
        self.clr = ndimage.distance_transform_edt(self.free)
        cost = 1.0 + LAM * (1.0 - np.minimum(self.clr, CAP) / CAP) ** 2
        self.G, self.idx, self.fy, self.fx = _graph(self.free, cost)
        self.covered = np.zeros(len(self.fy), bool)          # free-node mask
        self.dist_drawn = np.full(mask.shape, np.inf)         # distance of every pixel to the nearest drawn pixel

    def trajectories_from(self, xy) -> list[np.ndarray]:
        """Key trajectories from pixel xy (x, y) with the current coverage; updates the shared state."""
        H, W = self.mask.shape
        src = self.idx[int(xy[1]), int(xy[0])]
        if src < 0:
            return []
        D, pred = dijkstra(self.G, indices=src, return_predecessors=True, limit=RADIUS_M * (1 + LAM))
        length = _walk_length(pred, np.isfinite(D), self.fy, self.fx, src)
        keep = length <= RADIUS_M
        covered = self.covered | ~keep                       # this wave's view; the shared state gets the bands
        trajs = []
        while len(trajs) < K_MAX:
            open_len = np.where(~covered, length, -1.0)
            lmax = open_len.max()
            if lmax < 0:
                break
            front = np.nonzero(open_len >= lmax - FRONT_SHARE * RADIUS_M)[0]
            t = int(front[np.argmax(self.dist_drawn[self.fy[front], self.fx[front]])])   # farthest from what is drawn
            path = [t]
            while path[-1] != src:
                path.append(int(pred[path[-1]]))
            path = path[::-1]
            own = [n for n in path if not covered[n]]
            own_len = length[t] - (length[own[0]] if own else length[t])
            pm = np.zeros((H, W), bool); pm[self.fy[path], self.fx[path]] = True
            dist, (iy, ix) = ndimage.distance_transform_edt(~pm, return_indices=True)
            band = dist <= np.minimum(TUBE_A * self.clr[iy, ix] + TUBE_B, BAND_CAP_SHARE * RADIUS_M)
            covered |= band[self.fy, self.fx]; covered[t] = True
            self.covered |= band[self.fy, self.fx]; self.covered[t] = True
            if own_len >= L_MIN_M:
                trajs.append(np.stack([self.fx[path], self.fy[path]], 1).astype(np.int16))
                self.dist_drawn = np.minimum(self.dist_drawn, dist)
        return trajs


def trajectories_for_window(mask: np.ndarray, markers: np.ndarray, n_markers: int = 1, rounds: int = N_ROUNDS):
    """-> [round-1 trajectories per marker, round-2 trajectories per round-1 end of marker 0, round 3 per round-2
    end, ...]: a list of `rounds` lists of lists."""
    field = _Field(mask)
    r1 = [field.trajectories_from(markers[m]) for m in range(n_markers)]
    r1 += [[] for _ in range(2 - n_markers)]
    out = [r1]
    prev_ends = [tr[-1] for tr in r1[0]]
    for _ in range(1, rounds):
        rr, total = [], 0
        for e in prev_ends:
            got = field.trajectories_from(e)[: max(0, K_MAX - total)]
            rr.append(got); total += len(got)
        out.append(rr)
        prev_ends = [tr[-1] for trajs in rr for tr in trajs]
    return out


def _pack_round(groups, tag_of):
    """groups: per group a list of (trajectory (k,2) in window coords, inside (k,) bool); only inside pixels stored."""
    pix, tag, seen = [], [], set()
    for g, trajs in enumerate(groups):
        for k, (tr, inside) in enumerate(trajs):
            for x, y in tr[inside]:
                if (x, y) not in seen:
                    seen.add((x, y)); pix.append((x, y)); tag.append(tag_of(g, k))
    return np.asarray(pix, np.int16).reshape(-1, 2), np.asarray(tag, np.int8)


def _pack(rounds):
    """-> dict: pix, pix_tag, n_traj (2,), ends (2,K,2) for round 1; pix{r}, tag{r}, n_traj{r}, ends{r} for r >= 2"""
    r1 = rounds[0]
    pix, tag = _pack_round(r1, lambda m, k: m * 32 + k)
    n_traj = np.array([len(t) for t in r1], np.int8); ends = np.full((2, K_MAX, 2), -1, np.int16)
    for m, trajs in enumerate(r1):
        for k, (tr, _) in enumerate(trajs):
            ends[m, k] = tr[-1]                    # window coordinates; may lie outside the window
    out = dict(pix=pix, pix_tag=tag, n_traj=n_traj, ends=ends)
    for r, rr in enumerate(rounds[1:], start=2):
        px, tg = _pack_round(rr, lambda e, k: e)
        flat = [tr for trajs in rr for tr, _ in trajs]
        en = np.full((K_MAX, 2), -1, np.int16)
        for k, tr in enumerate(flat):
            en[k] = tr[-1]
        out[f"pix{r}"] = px; out[f"tag{r}"] = tg; out[f"n_traj{r}"] = np.int8(len(flat)); out[f"ends{r}"] = en
    return out


_W = {}


def _init(district_path: str, routes_path: str, crop: int, n_markers: int, radius: float):
    set_radius(radius)
    _W["bld"] = np.load(district_path)["surface"] == BUILDING_CLASS
    z = np.load(routes_path); _W["origin"] = z["origin"]; _W["markers"] = z["markers"]; _W["crop"] = crop
    _W["n_markers"] = n_markers


def trajectories_for_sample(bld: np.ndarray, origin, crop: int, markers: np.ndarray, n_markers: int = 1,
                            rounds: int = N_ROUNDS):
    """Rounds for one store sample: the wave runs on the district map padded by one horizon around the window;
    the returned trajectories are in window coordinates and clipped to the window (empty pieces dropped)."""
    dx, dy = int(origin[0]), int(origin[1]); M = int(np.ceil(RADIUS_M))
    x0, y0 = max(dx - M, 0), max(dy - M, 0)
    x1, y1 = min(dx + crop + M, bld.shape[1]), min(dy + crop + M, bld.shape[0])
    mask = bld[y0:y1, x0:x1]
    mk = np.asarray(markers, dtype=np.float64) + np.array([dx - x0, dy - y0], dtype=np.float64)
    rounds_out = trajectories_for_window(mask, mk, n_markers, rounds)
    off = np.array([dx - x0, dy - y0], dtype=np.int16)

    def clip(tr):
        q = tr - off
        inside = (q[:, 0] >= 0) & (q[:, 0] < crop) & (q[:, 1] >= 0) & (q[:, 1] < crop)
        return q, inside

    out = []
    for groups in rounds_out:
        out.append([[clip(tr) for tr in trajs] for trajs in groups])
    return out


def _work(rng_ix):
    lo, hi = rng_ix; out = []; C = _W["crop"]
    for i in range(lo, hi):
        out.append(_pack(trajectories_for_sample(_W["bld"], _W["origin"][i], C, _W["markers"][i], _W["n_markers"])))
    return lo, out


def build(store: Path, split: str, limit: int | None, gallery: int, workers: int, n_markers: int):
    from spatial_data.dataset_c4_gpu import store_crop_px
    crop = store_crop_px(store)
    routes = store / f"c4_routes_{split}.npz"; district = store.parent / "district_labels_1m.npz"
    N = int(np.load(routes)["origin"].shape[0]); N = N if limit is None else min(limit, N)
    chunks = [(a, min(a + 250, N)) for a in range(0, N, 250)]
    res = [None] * N; t0 = time.perf_counter(); done = 0
    with Pool(workers, initializer=_init, initargs=(str(district), str(routes), crop, n_markers, RADIUS_M)) as pool:
        for lo, out in pool.imap_unordered(_work, chunks):
            res[lo:lo + len(out)] = out; done += len(out)
            if done % 5000 < 250:
                print(f"  {split}: {done}/{N}  {1000 * (time.perf_counter() - t0) / done:.1f} ms/sample (all workers)", flush=True)
    c1 = np.array([len(r["pix"]) for r in res], np.int64)
    out = dict(pix=np.concatenate([r["pix"] for r in res]), pix_tag=np.concatenate([r["pix_tag"] for r in res]),
               offset=np.concatenate([[0], np.cumsum(c1)]), n_traj=np.stack([r["n_traj"] for r in res]), ends=np.stack([r["ends"] for r in res]))
    for r in range(2, N_ROUNDS + 1):
        cr = np.array([len(x[f"pix{r}"]) for x in res], np.int64)
        out[f"pix{r}"] = np.concatenate([x[f"pix{r}"] for x in res]); out[f"tag{r}"] = np.concatenate([x[f"tag{r}"] for x in res])
        out[f"offset{r}"] = np.concatenate([[0], np.cumsum(cr)]); out[f"n_traj{r}"] = np.array([x[f"n_traj{r}"] for x in res], np.int8)
        out[f"ends{r}"] = np.stack([x[f"ends{r}"] for x in res])
    suffix = "" if limit is None else f"_first{N}"; rs = file_suffix()
    np.savez_compressed(store / f"c5_traj{rs}_{split}{suffix}.npz", **out)
    nt = out["n_traj"][:, :n_markers]
    stats = dict(split=split, n=N, seconds=round(time.perf_counter() - t0, 1), workers=workers, markers=n_markers,
                 radius_m=RADIUS_M, lam=LAM, cap_m=CAP, tube_a=TUBE_A, tube_b_m=TUBE_B, l_min_m=L_MIN_M, l_min_share=L_MIN_SHARE,
                 band_cap_share=BAND_CAP_SHARE, front_share=FRONT_SHARE,
                 round1_mean_traj_per_marker=float(nt.mean()), round1_p90=float(np.percentile(nt, 90)),
                 round1_share_markers_without=float((nt == 0).mean()), round1_mean_pixels=float(c1.mean()),
                 **{f"round{r}_{k}": v for r in range(2, N_ROUNDS + 1) for k, v in (
                     ("mean_traj", float(out[f"n_traj{r}"].mean())), ("p90", float(np.percentile(out[f"n_traj{r}"], 90))),
                     ("share_without", float((out[f"n_traj{r}"] == 0).mean())),
                     ("mean_pixels", float((out[f"offset{r}"][1:] - out[f"offset{r}"][:-1]).mean())))})
    (store / f"c5_traj{rs}_{split}{suffix}_stats.json").write_text(json.dumps(stats, indent=1))
    print(split, stats)
    if gallery:
        draw_gallery(store, split, suffix, min(gallery, N))


def draw_gallery(store: Path, split: str, suffix: str, n: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spatial_data.dataset_c4_gpu import store_crop_px
    C = store_crop_px(store)
    bld = np.load(store.parent / "district_labels_1m.npz")["surface"] == BUILDING_CLASS
    z = np.load(store / f"c4_routes_{split}.npz"); t = np.load(store / f"c5_traj{file_suffix()}_{split}{suffix}.npz")
    cols = 6; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows))
    for k, ax in enumerate(np.ravel(axes)):
        if k >= n: ax.axis("off"); continue
        dx, dy = z["origin"][k]; ax.imshow(bld[dy:dy + C, dx:dx + C], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        for r, col in ((4, "tab:purple"), (3, "tab:red"), (2, "tab:orange")):
            if f"pix{r}" in t:
                p2 = t[f"pix{r}"][t[f"offset{r}"][k]:t[f"offset{r}"][k + 1]]
                ax.scatter(p2[:, 0], p2[:, 1], s=0.6, c=col)
        p = t["pix"][t["offset"][k]:t["offset"][k + 1]]; g = t["pix_tag"][t["offset"][k]:t["offset"][k + 1]] // 32
        for m, col in ((0, "tab:blue"), (1, "tab:green")):
            q = p[g == m]; ax.scatter(q[:, 0], q[:, 1], s=0.6, c=col)
            if t["n_traj"][k, m]:
                ax.plot(*z["markers"][k, m], "o", color=col, ms=5, mec="k")
        e = t["ends"][k, 0, : t["n_traj"][k, 0]]; e = e[(e >= 0).all(1) & (e < C).all(1)]
        ax.plot(e[:, 0], e[:, 1], "s", color="tab:blue", ms=3, mec="k", mew=0.5)
        counts = " / ".join(str(int(t[f"n_traj{r}"][k])) for r in range(2, N_ROUNDS + 1) if f"n_traj{r}" in t)
        ax.set_title(f"#{k}: rounds {t['n_traj'][k, 0]} / {counts}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C5 rung 1 targets: key trajectories within {RADIUS_M:.0f} m walking; round 1 from the marker (blue, ends = squares), "
                 "then from each end: round 2 orange, 3 red, 4 purple", fontsize=11)
    fig.tight_layout(); out = store / f"c5_traj{file_suffix()}_{split}{suffix}_gallery.png"; fig.savefig(out, dpi=100); print("gallery ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--splits", default="val")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--gallery", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--radius", type=float, default=RADIUS_M, help="horizon in metres (40, 100)")
    ap.add_argument("--markers", type=int, default=1, choices=(1, 2), help="markers used per sample (round 2 always from marker 0)")
    a = ap.parse_args()
    set_radius(a.radius)
    for split in a.splits.split(","):
        build(Path(a.store), split, a.limit, a.gallery, a.workers, a.markers)


if __name__ == "__main__":
    main()
