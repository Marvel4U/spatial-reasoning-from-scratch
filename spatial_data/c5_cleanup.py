"""C5 cleanup pass (5 Oct 2026): train the final pass to turn the network's own messy drawing into one sharp line.

Why: free-running, the wave search ends with a drawing that usually connects start and goal (the success rule of
harness/diagnostics/wave_rollout_eval.py), but with far more lit cells than the route: halos, side branches, fronts
left over. The draw pass was trained on the teacher's clean input (start disc + the teacher's path) or, with
`c5_wave_own_pads`, on the network's own input but still with the teacher's path as target, which asks it to jump to a
path its own drawing may not contain. The cleanup target is instead the network's OWN best route: the shortest chain
through the lit cells of the drawing that enters the draw pass, drawn 3 cells wide like the teacher's path. The pass
then only has to keep the cells of that chain and drop the rest.

Target rule per route (`cleanup_target`), on the drawing that enters the draw pass (own-pad slot 2 n_goal - 1, the
output of the origin pass; see spatial_data/c5_own_pads.py):
  if the drawing is a success by the rollout eval's rule (connected start -> goal through lit cells with the 3 x 3
  cells around each marker counted as lit, no cell of the chain entirely building, chain length <= 1.10 x the
  teacher's walking length): the target is that shortest 8-neighbour chain, widened to PATH_WIDTH_CELLS (3 x 3
  max-pool of the chain cells, as _wave_path_cells widens the teacher's path); flag 1
  otherwise: the target is the teacher's path (the usual draw target, 3 cells wide), so the pass learns to repair a
  failed drawing; flag 0
  routes without a stored drawing (unreachable, or beyond the own-pad store): flag -1, the loader keeps the usual
  sample.
The graph logic (`shortest_lit_chain`) is the one route_score uses; route_score now imports it from here.

Store (`c5_cleanup_{split}[_tag].npz` next to the route store, uncompressed):
  pad_in   (n, g, g/8) uint8  packed drawing that enters the draw pass (a copy of the own-pad slot, so the loader
                              needs only this file: ~0.2 GB for 100k routes at grid 128, 0.16 GB for 20k at grid 256)
  target   (n, g, g/8) uint8  packed cleanup target
  flag     (n,) int8          1 own chain, 0 teacher's path, -1 no drawing
  reason   (n,) int8          why the teacher's path: 0 none (own chain), 1 not connected, 2 a building cell on the
                              chain, 3 chain too long (2 wins over 3), -1 no drawing
  ratio    (n,) float32       chain length / teacher's walking length (nan when not connected)
  lit_in, lit_out (n,) int32  lit cells of the drawing in and of the target
  n_goal   (n,) int16         as the loader computes it (0 when no drawing), checked against the route store on load
  meta     0-d str            json: own-pad store and its run, grid, crop, split
plus `..._stats.json` (shares own chain / teacher, mean lit cells in vs out = the excess the pass must remove, time).

Loader (`config.c5_wave_cleanup`, harness/data.py): for wave_draw samples in training, with probability p the pad in
is the stored drawing and the target the stored cleanup target (`gather_cleanup`, called in _wave_post). Val loaders
never read it.

Usage (from the repo root): python -m spatial_data.c5_cleanup STORE_DIR --own c5_own_pads_train_r2.npz [--tag r2]
       [--split train] [--workers 3] [--limit N]"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SUCCESS_LENGTH_RATIO = 1.10     # the rollout eval's success rule: chain length <= 1.10 x the teacher's walking length
CLEAN_EXCESS_RATIO = 1.25       # eval: a final drawing is "clean" when its lit cells are at most 1.25 x its own 3-wide chain
PATH_WIDTH = 3                  # = spatial_data.c4_tasks.PATH_WIDTH_CELLS (checked in the tests; not imported to keep this torch-free)

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- the chain (shared with the rollout eval)

def shortest_lit_chain(mask: np.ndarray, start_c, goal_c, cell_m: float):
    """Shortest 8-neighbour chain start -> goal through the lit cells of mask (g,g) bool; the 3 x 3 cells around each
    marker count as lit. start_c, goal_c = (col, row). Diagonal steps cost sqrt(2) cells. Returns (ys, xs, length_m)
    of the chain's cells (goal first), or None when start and goal are not connected. This is route_score's graph,
    moved here so the eval and the cleanup targets use the same chain."""
    g = mask.shape[0]
    m = mask.copy()
    for cx, cy in (start_c, goal_c):                       # the marker cells themselves count as lit (3x3)
        m[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
    idx = -np.ones((g, g), np.int64); ys, xs = np.nonzero(m); idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in _OFFS:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < g) & (x2 >= 0) & (x2 < g)
        ok[ok] &= m[y2[ok], x2[ok]]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]]); w.append(np.full(int(ok.sum()), np.hypot(dy, dx) * cell_m))
    G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys),) * 2)
    s, t = idx[start_c[1], start_c[0]], idx[goal_c[1], goal_c[0]]
    D, pred = dijkstra(G, indices=s, return_predecessors=True)
    if not np.isfinite(D[t]):
        return None
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(int(pred[nodes[-1]]))
    return ys[nodes], xs[nodes], float(D[t])


def chain_success(length_ratio: float, building_share: float) -> bool:
    """The rollout eval's success rule on a connected chain."""
    return bool(length_ratio <= SUCCESS_LENGTH_RATIO and building_share == 0.0)


def widen(cells: np.ndarray, width: int = PATH_WIDTH) -> np.ndarray:
    """(g,g) bool -> widened by a width x width max-pool (zero border), as _wave_path_cells widens the teacher's path."""
    if width <= 1:
        return cells.copy()
    return ndimage.binary_dilation(cells, np.ones((width, width), bool))


def chain_mask(g: int, ys: np.ndarray, xs: np.ndarray, width: int = PATH_WIDTH) -> np.ndarray:
    m = np.zeros((g, g), bool); m[ys, xs] = True
    return widen(m, width)


def cleanliness(mask: np.ndarray, start_c, goal_c, cell_m: float) -> dict:
    """How much of a final drawing is its own route. chain = the 3-wide shortest chain through the drawing;
    excess_ratio = lit cells / chain cells (nan when not connected); clean = excess_ratio <= 1.25 (0 when not
    connected); iou_own_chain = IoU of the drawing against that chain (nan when not connected)."""
    lit = float(mask.sum())
    ch = shortest_lit_chain(mask, start_c, goal_c, cell_m)
    if ch is None:
        return dict(chain_cells=float("nan"), excess_ratio=float("nan"), clean=0.0, iou_own_chain=float("nan"))
    cm = chain_mask(mask.shape[0], ch[0], ch[1])
    n_chain = float(cm.sum())
    union = float((cm | mask).sum())
    ratio = lit / max(n_chain, 1.0)
    return dict(chain_cells=n_chain, excess_ratio=ratio, clean=float(ratio <= CLEAN_EXCESS_RATIO),
                iou_own_chain=float((cm & mask).sum()) / union if union > 0 else 1.0)


# ---------------------------------------------------------------- the target of one route

def teacher_path_cells(path_px: np.ndarray, g: int, crop: int, width: int = PATH_WIDTH) -> np.ndarray:
    """The teacher's draw target as _wave_path_cells builds it: path pixels (x, y) -> cells, widened."""
    s = crop // g
    m = np.zeros((g, g), bool)
    if len(path_px):
        px = np.asarray(path_px, np.int64)
        m[np.clip(px[:, 1] // s, 0, g - 1), np.clip(px[:, 0] // s, 0, g - 1)] = True
    return widen(m, width)


def cleanup_target(pad_in: np.ndarray, start_c, goal_c, blocked: np.ndarray, goal_m: float, cell_m: float,
                   teacher: np.ndarray) -> dict:
    """The cleanup target of one route (module docstring). pad_in (g,g) bool = the drawing entering the draw pass,
    blocked (g,g) bool = cells entirely building, teacher (g,g) bool = the teacher's 3-wide path."""
    ch = shortest_lit_chain(pad_in, start_c, goal_c, cell_m)
    if ch is None:
        return dict(target=teacher, flag=0, reason=1, ratio=float("nan"))
    ys, xs, length = ch
    ratio = float(length / max(goal_m, 1e-6))
    bshare = float(blocked[ys, xs].mean())
    if chain_success(ratio, bshare):
        return dict(target=chain_mask(pad_in.shape[0], ys, xs), flag=1, reason=0, ratio=ratio)
    return dict(target=teacher, flag=0, reason=2 if bshare > 0 else 3, ratio=ratio)


# ---------------------------------------------------------------- store: build, save, load, gather

def npz_member_memmap(path: Path, key: str) -> np.ndarray:
    """A read-only memmap of one array of an uncompressed .npz (np.savez), so a 2-3 GB own-pad store is read
    route by route instead of loaded whole; falls back to loading for compressed members."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return np.load(path)[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = npf.read_magic(f)
        shape, fortran, dtype = (npf.read_array_header_1_0(f) if version == (1, 0) else npf.read_array_header_2_0(f))
        data_off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=data_off, order="F" if fortran else "C")


_W: dict = {}                    # per worker process: the inputs (set by _init_worker; fork shares them read-only)


def _init_worker(own_path: str, inputs: dict) -> None:
    _W.clear(); _W.update(inputs)
    _W["pads"] = npz_member_memmap(Path(own_path), "pads")


def _work(chunk: tuple[int, int]) -> dict:
    lo, hi = chunk
    W = _W; g, crop = W["grid"], W["crop"]; s = crop // g; cell_m = float(s)
    n = hi - lo; gb = g // 8
    out = dict(lo=lo, pad_in=np.zeros((n, g, gb), np.uint8), target=np.zeros((n, g, gb), np.uint8),
               flag=np.full(n, -1, np.int8), reason=np.full(n, -1, np.int8), ratio=np.full(n, np.nan, np.float32),
               lit_in=np.zeros(n, np.int32), lit_out=np.zeros(n, np.int32))
    off, ng = W["own_offset"], W["own_n_goal"]
    for i, r in enumerate(range(lo, hi)):
        cnt = int(off[r + 1] - off[r]) if r < len(ng) else 0
        if cnt == 0 or not np.isfinite(W["goal_m"][r]):
            continue
        slot = 2 * int(ng[r]) - 1                                  # the origin pass's output = the draw pass's input
        if slot < 0 or slot >= cnt:
            continue
        packed = np.asarray(W["pads"][off[r] + slot])
        pad = np.unpackbits(packed, axis=-1).astype(bool)
        mk = W["markers"][r]
        start_c = (int(mk[0][0]) // s, int(mk[0][1]) // s); goal_c = (int(mk[1][0]) // s, int(mk[1][1]) // s)
        ox, oy = int(W["origin"][r][0]), int(W["origin"][r][1])
        blocked = W["bld"][oy:oy + crop, ox:ox + crop].reshape(g, s, g, s).all(axis=(1, 3))
        teacher = teacher_path_cells(W["path"][W["path_off"][r]:W["path_off"][r + 1]], g, crop)
        res = cleanup_target(pad, start_c, goal_c, blocked, float(W["goal_m"][r]), cell_m, teacher)
        out["pad_in"][i] = packed; out["target"][i] = np.packbits(res["target"], axis=-1)
        out["flag"][i] = res["flag"]; out["reason"][i] = res["reason"]; out["ratio"][i] = res["ratio"]
        out["lit_in"][i] = int(pad.sum()); out["lit_out"][i] = int(res["target"].sum())
    return out


def save_cleanup(path: Path, *, pad_in, target, flag, reason, ratio, lit_in, lit_out, n_goal, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, pad_in=np.asarray(pad_in, np.uint8), target=np.asarray(target, np.uint8), flag=np.asarray(flag, np.int8),
             reason=np.asarray(reason, np.int8), ratio=np.asarray(ratio, np.float32), lit_in=np.asarray(lit_in, np.int32),
             lit_out=np.asarray(lit_out, np.int32), n_goal=np.asarray(n_goal, np.int16), meta=np.array(json.dumps(meta)))
    tmp.replace(path)                                             # never leave a half-written store under the real name


def build_cleanup(store: Path, own_name: str, *, split: str = "train", workers: int = 3, limit: int | None = None,
                  chunk: int = 1000, log=print) -> dict:
    """Compute the cleanup store's arrays for the routes of `store` from the own-pad store `own_name` next to it."""
    import multiprocessing as mp
    import time
    t0 = time.perf_counter()
    store = Path(store)
    own_path = store / own_name
    with np.load(own_path) as z:
        own_offset = z["offset"].astype(np.int64); own_n_goal = z["n_goal"].astype(np.int64)
        own_meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    pads_shape = npz_member_memmap(own_path, "pads").shape
    grid = int(pads_shape[1])
    sj = store / "store.json"
    crop = int(json.loads(sj.read_text())["crop_px"]) if sj.is_file() else 256
    if crop % grid:
        raise ValueError(f"grid {grid} does not divide the window {crop} px")
    rz = np.load(store / f"c4_routes_{split}.npz")
    wz = np.load(store / f"c5_wave_{split}.npz")
    n = len(own_n_goal) if limit is None else min(int(limit), len(own_n_goal))
    inputs = dict(grid=grid, crop=crop, own_offset=own_offset, own_n_goal=own_n_goal,
                  markers=rz["markers"][:n], origin=rz["origin"][:n], goal_m=wz["goal_m"][:n].astype(np.float64),
                  path_off=wz["offset"][:n + 1].astype(np.int64), path=None,
                  bld=np.load(store.parent / "district_labels_1m.npz")["surface"] == 3)
    inputs["path"] = wz["path"][: int(inputs["path_off"][-1])]
    # the loader's n_goal (float32 goal metres), to check that store and route store agree on which slot is the draw input
    from .c4_tasks import WAVE_HORIZON_M
    gm32 = wz["goal_m"][:n].astype(np.float32)
    fin = np.isfinite(gm32)
    want = np.where(fin, np.floor(np.where(fin, gm32, 0) / np.float32(WAVE_HORIZON_M)).astype(np.int64) + 1, 1)
    has = (own_offset[1:n + 1] - own_offset[:n]) > 0
    if (has & (own_n_goal[:n] != want)).any():
        raise ValueError(f"{own_path}: n_goal differs from the route store for {int((has & (own_n_goal[:n] != want)).sum())} routes")
    t_load = time.perf_counter() - t0
    chunks = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    gb = grid // 8
    res = dict(pad_in=np.zeros((n, grid, gb), np.uint8), target=np.zeros((n, grid, gb), np.uint8),
               flag=np.full(n, -1, np.int8), reason=np.full(n, -1, np.int8), ratio=np.full(n, np.nan, np.float32),
               lit_in=np.zeros(n, np.int32), lit_out=np.zeros(n, np.int32))
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(own_path), inputs)) as pool:
        for part in pool.imap_unordered(_work, chunks):
            lo = part["lo"]; hi = lo + len(part["flag"])
            for k in res:
                res[k][lo:hi] = part[k]
            done += hi - lo
            if log is not None and (done // 10000) != ((done - (hi - lo)) // 10000):
                log(f"  {done}/{n} routes, {time.perf_counter() - t0:.0f} s")
    res["n_goal"] = np.where(res["flag"] >= 0, own_n_goal[:n], 0)
    res["meta"] = dict(own_pads=own_name, own_pads_run=own_meta.get("run"), own_pads_step=own_meta.get("checkpoint_step"),
                       store=str(store.resolve()), split=split, grid=grid, crop=crop, routes=n, rule=dict(
                           success_length_ratio=SUCCESS_LENGTH_RATIO, path_width=PATH_WIDTH, input_slot="2 n_goal - 1"))
    res["seconds"] = time.perf_counter() - t0; res["seconds_load"] = t_load
    return res


def cleanup_stats(res: dict) -> dict:
    """Shares own chain / teacher and the excess: mean lit cells of the drawing in vs the target."""
    flag, reason = res["flag"], res["reason"]
    lit_in, lit_out = res["lit_in"].astype(np.float64), res["lit_out"].astype(np.float64)
    has = flag >= 0; own = flag == 1; tea = flag == 0
    nh = max(int(has.sum()), 1)

    def mean(x, m):
        return round(float(x[m].mean()), 2) if m.any() else None

    def excess(m):
        return dict(lit_in=mean(lit_in, m), lit_out=mean(lit_out, m),
                    ratio_of_means=round(float(lit_in[m].mean() / max(lit_out[m].mean(), 1e-9)), 3) if m.any() else None,
                    median_ratio=round(float(np.median(lit_in[m] / np.maximum(lit_out[m], 1))), 3) if m.any() else None)

    ng = res["n_goal"]
    bands = (("n_goal 2-3", 2, 3), ("n_goal 4-5", 4, 5), ("n_goal 6-8", 6, 8), ("n_goal 9-12", 9, 12), ("n_goal 13+", 13, 999))
    return dict(routes=int(len(flag)), with_drawing=int(has.sum()),
                share_own_chain=round(float(own.sum()) / nh, 4), share_teacher=round(float(tea.sum()) / nh, 4),
                teacher_reason_share=dict(not_connected=round(float((reason == 1).sum()) / nh, 4),
                                          building_on_chain=round(float((reason == 2).sum()) / nh, 4),
                                          too_long=round(float((reason == 3).sum()) / nh, 4)),
                own_chain_length_ratio_mean=mean(res["ratio"].astype(np.float64), own),
                excess_all=excess(has), excess_own_chain=excess(own), excess_teacher=excess(tea),
                by_n_goal={name: dict(n=int((has & (ng >= lo) & (ng <= hi)).sum()),
                                      share_own_chain=round(float((own & (ng >= lo) & (ng <= hi)).sum()) / max(int((has & (ng >= lo) & (ng <= hi)).sum()), 1), 4),
                                      **excess(own & (ng >= lo) & (ng <= hi)))
                           for name, lo, hi in bands})


def load_cleanup(path: Path, *, p: float, device, own_pads_name: str | None = None, grid: int | None = None,
                 goal_m=None) -> dict:
    """The cleanup store for the train loader: packed pads in and targets on the CPU (pinned when on cuda), flags on
    the device. Checks: the own-pad store it was built from (own_pads_name), the grid, and n_goal against the
    loader's goal distances (so slot 2 n_goal - 1 meant the same thing when the store was built)."""
    import torch
    z = np.load(Path(path))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if own_pads_name is not None and Path(str(meta.get("own_pads", ""))).name != Path(own_pads_name).name:
        raise ValueError(f"{path} was built from {meta.get('own_pads')}, the config says {own_pads_name}")
    pad_in = torch.from_numpy(np.ascontiguousarray(z["pad_in"])); target = torch.from_numpy(np.ascontiguousarray(z["target"]))
    if grid is not None and int(pad_in.shape[1]) != int(grid):
        raise ValueError(f"{path}: grid {pad_in.shape[1]}, the run's output grid is {grid}")
    if torch.device(device).type == "cuda":
        pad_in = pad_in.pin_memory(); target = target.pin_memory()
    flag = torch.from_numpy(z["flag"].astype(np.int64))
    n = int(flag.shape[0])
    if goal_m is not None:
        from .c4_tasks import WAVE_HORIZON_M
        m = min(n, int(goal_m.shape[0]))
        gm = goal_m[:m].detach().cpu()
        want = torch.where(torch.isfinite(gm), (gm / WAVE_HORIZON_M).floor().long() + 1, torch.ones_like(gm, dtype=torch.long))
        ng = torch.from_numpy(z["n_goal"].astype(np.int64))[:m]
        bad = (flag[:m] >= 0) & (ng != want)
        if bool(bad.any()):
            raise ValueError(f"{path}: n_goal of {int(bad.sum())} routes differs from the route store (another store?)")
    return dict(pad_in=pad_in, target=target, flag=flag.to(device), n_routes=n, p=float(p), path=str(path), meta=meta,
                share_own_chain=float((flag == 1).sum()) / max(int((flag >= 0).sum()), 1))


def gather_cleanup(cl: dict, route_ix, is_draw, gen):
    """Per sample: cleanup sample (probability cl['p'], only draw samples of routes with a stored drawing)? Returns
    use (B,) bool, pad_in (B,g,g) bool, target (B,g,g) bool on route_ix's device (rows where use is False are
    empty). Consumes B uniforms from gen whenever p > 0, so the random stream does not depend on the store."""
    import torch
    from .c5_own_pads import unpack_pads
    b = route_ix.shape[0]; dev = route_ix.device
    g = cl["pad_in"].shape[1]
    pin = torch.zeros(b, g, g, dtype=torch.bool, device=dev); tgt = torch.zeros_like(pin)
    p = float(cl["p"])
    if p <= 0:
        return torch.zeros(b, dtype=torch.bool, device=dev), pin, tgt
    coin = torch.rand(b, device=dev, generator=gen) < p
    n = int(cl["n_routes"])
    in_store = route_ix < n
    r = route_ix.clamp(max=n - 1)
    use = coin & in_store & is_draw & (cl["flag"][r] >= 0)
    if not bool(use.any()):
        return use, pin, tgt
    idx = r[use].cpu()
    pin[use] = unpack_pads(cl["pad_in"][idx].to(dev, non_blocking=True))
    tgt[use] = unpack_pads(cl["target"][idx].to(dev, non_blocking=True))
    return use, pin, tgt


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("store", type=Path, help="route store dir, e.g. data/amsterdam/de_pijp/crops/v4/c4_100k")
    ap.add_argument("--own", required=True, help="own-pad store in that dir, e.g. c5_own_pads_train_r2.npz")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--tag", default=None, help="file suffix: c5_cleanup_{split}_{tag}.npz (default: none)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="first N routes only")
    args = ap.parse_args(argv)
    res = build_cleanup(args.store, args.own, split=args.split, workers=args.workers, limit=args.limit)
    out = args.store / (f"c5_cleanup_{args.split}" + (f"_{args.tag}" if args.tag else "") + ".npz")
    save_cleanup(out, **{k: res[k] for k in ("pad_in", "target", "flag", "reason", "ratio", "lit_in", "lit_out", "n_goal", "meta")})
    stats = {**res["meta"], **cleanup_stats(res), "seconds": round(res["seconds"], 1), "seconds_load": round(res["seconds_load"], 1),
             "workers": args.workers, "file_mb": round(out.stat().st_size / 2 ** 20, 1)}
    (out.parent / (out.stem + "_stats.json")).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
