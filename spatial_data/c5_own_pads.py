"""C5 R5 own pads (3 Oct 2026): the network's own scratchpads, to train on instead of only the teacher's.

Why: in a rollout the network reads its own last output. In training it has only seen the teacher's clean pads
(plus synthetic pad noise, `_wave_pad_noise`). The errors it makes itself are the most natural noise there is, so we
run a trained network over the train routes under the teacher's pass schedule, keep every output pad, and let the
train loader feed those pads in place of the teacher's (config key `c5_wave_own_pads`). The target stays the
teacher's clean next pad: the network learns to repair its own drift.

Rollout ("scheduled" mode of harness/diagnostics/wave_rollout_eval.py): the pass order, the task key and the
state token come from the teacher's schedule (`wave_pass_schedule(n_goal)`); the pad starts as the start disc and
from then on is the network's own output of the previous pass. Batched over routes with the same n_goal, so every
route of a batch runs the same schedule.

Store layout (`c5_own_pads_{split}.npz` next to the route store, uncompressed so it loads fast):
  pads     (P, g, g/8) uint8   np.packbits(pad, axis=-1) of every stored output pad
  offset   (n+1,) int64        route r owns pads[offset[r]:offset[r+1]]; 0 passes for unreachable / unrun routes
  n_goal   (n,) int16          the route's n_goal as the loader computes it (0 when not run)
  slot_key (P,) int8           index into TASK_KEYS of the pass that wrote the slot
  slot_k_in, slot_j (P,) int16 the pass identity (k_in of a forward pass, j of a backward pass; -1 otherwise)
  iou      (P,) float16        IoU of the stored pad against the teacher's output of the same pass
  meta     0-d str             json: run, checkpoint, step, split, grid
Slots are in schedule order: slot s of a route is the output of the pass at position s, where the position of a
pass is `pass_position` below (forward k_in -> k_in, backward j -> 2 n_goal - j, draw -> 2 n_goal), so a route
has 2 n_goal + 1 slots. (For n_goal = 1 the teacher's schedule runs wave_first and wave_reach, both with k_in = 0;
the reach output overwrites slot 0, because that is the pad the backward part reads. The 100k store has no such
route; the rule is there so the layout never breaks.)

Which own pad replaces which input (`own_pad_source_slot`): the input of a pass is the output of the pass before
it in the schedule, i.e. slot position - 1:
  forward, k_in = k     -> the output of the forward pass with k_in = k - 1 (slot k - 1); k = 0 is the start disc
                           and is never replaced
  backward, front j     -> the output of the backward pass j + 1 (slot 2 n_goal - j - 1); for j = n_goal that is
                           the reach pass (slot n_goal - 1)
  draw                  -> the output of the origin pass, j = 1 (slot 2 n_goal - 1)

Usage (from harness/): PYTHONPATH=.:.. ../.venv/bin/python -m spatial_data.c5_own_pads RUN_ID [--limit 40000]
       [--batch 128] [--split train] [--store DIR] [--checkpoint PATH] [--out NAME] [--teacher]
--teacher stores the teacher's own outputs instead of the network's (a check of the mapping: fed back, they
reproduce the teacher's inputs exactly).

Other window sizes (4 Oct): the store's window (store.json crop_px) sets the grid. A 512 m store gets 256 x 256 cells
of 2 m (packed 256 x 32 bytes per pass, 4x a 256 m pass); a checkpoint trained at another window is rebuilt for the
store's window with its cell size kept, as harness/diagnostics/wave_rollout_eval.py does (--grid forces a size)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

TASK_KEYS = ("wave_first", "wave_step", "wave_reach", "wave_back", "wave_origin", "wave_draw")

# WAVE_KINDS indices as _wave_post numbers them: 0 first 1 step 2 reach 3 back 4 origin 5 draw 6 fwd 7 bwd
_FWD_CODES = (0, 1, 2, 6)
_BWD_CODES = (3, 4, 7)
_DRAW_CODE = 5


def pass_position(task_key: str, k_in: int | None, j: int | None, n_goal: int) -> int:
    """Slot of a schedule pass (0-based): forward k_in -> k_in, backward j -> 2 n_goal - j, draw -> 2 n_goal."""
    if task_key in ("wave_first", "wave_step", "wave_reach"):
        return int(k_in if k_in is not None else (0 if task_key == "wave_first" else n_goal - 1))
    if task_key in ("wave_back", "wave_origin"):
        return 2 * n_goal - int(j if j is not None else 1)
    if task_key == "wave_draw":
        return 2 * n_goal
    raise ValueError(f"not a wave pass: {task_key!r}")


def own_pad_source_slot(kind_t: torch.Tensor, k_in: torch.Tensor, j: torch.Tensor, n_goal: torch.Tensor) -> torch.Tensor:
    """(B,) slot of the pass whose OUTPUT is the input of each sample's pass, -1 = none (not a wave pass, or the
    first forward pass, whose input is the start disc). kind_t = WAVE_KINDS index (-1 for other tasks); k_in, j,
    n_goal as _wave_post computes them. The input of a pass is the output of the pass before it: position - 1."""
    fwd = torch.zeros_like(kind_t, dtype=torch.bool)
    for c in _FWD_CODES:
        fwd |= kind_t == c
    bwd = torch.zeros_like(fwd)
    for c in _BWD_CODES:
        bwd |= kind_t == c
    draw = kind_t == _DRAW_CODE
    none = torch.full_like(n_goal, -1)
    slot = torch.where(fwd, k_in - 1, none)                       # k_in = 0 gives -1: the start disc stays
    slot = torch.where(bwd, 2 * n_goal - j - 1, slot)             # j = n_goal: 2 n_goal - n_goal - 1 = the reach pass
    slot = torch.where(draw, 2 * n_goal - 1, slot)                # the origin pass (j = 1)
    return slot


# ---------------------------------------------------------------- store: write, load, gather

def pack_pads(pads: torch.Tensor) -> torch.Tensor:
    """(N,g,g) bool -> (N,g,g/8) uint8, the same bit order as np.packbits(axis=-1) (first cell = high bit)."""
    n, g, _ = pads.shape
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int32, device=pads.device)
    return (pads.view(n, g, g // 8, 8).to(torch.int32) * w).sum(-1).to(torch.uint8)


def unpack_pads(packed: torch.Tensor) -> torch.Tensor:
    """(N,g,g/8) uint8 -> (N,g,g) bool, inverse of pack_pads / np.packbits(axis=-1)."""
    n, g, gb = packed.shape
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> sh) & 1).view(n, g, gb * 8).bool()


def save_own_pads(path: Path, *, pads: np.ndarray, offset: np.ndarray, n_goal: np.ndarray, slot_key: np.ndarray,
                  slot_k_in: np.ndarray, slot_j: np.ndarray, iou: np.ndarray, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, pads=pads.astype(np.uint8), offset=offset.astype(np.int64), n_goal=n_goal.astype(np.int16),
             slot_key=slot_key.astype(np.int8), slot_k_in=slot_k_in.astype(np.int16), slot_j=slot_j.astype(np.int16),
             iou=iou.astype(np.float16), meta=np.array(json.dumps(meta)))
    tmp.replace(path)                                             # never leave a half-written store under the real name


def load_own_pads(path: Path, *, p: float, device: torch.device, goal_m: torch.Tensor | None = None) -> dict:
    """The store for the train loader: packed pads on the CPU (pinned when on cuda, ~0.85 GB for 40k routes), the
    offsets and the route count on the device. goal_m (the loader's per-route goal distance) is used to check that
    store and loader agree on n_goal, so the slot arithmetic means the same thing on both sides."""
    z = np.load(Path(path))
    pads = torch.from_numpy(np.ascontiguousarray(z["pads"]))
    if device.type == "cuda":
        pads = pads.pin_memory()
    offset = torch.from_numpy(z["offset"].astype(np.int64))
    n = int(offset.shape[0] - 1)
    ng = torch.from_numpy(z["n_goal"].astype(np.int64))
    if goal_m is not None:
        from .c4_tasks import WAVE_HORIZON_M
        m = min(n, int(goal_m.shape[0]))
        gm = goal_m[:m].detach().cpu()
        want = torch.where(torch.isfinite(gm), (gm / WAVE_HORIZON_M).floor().long() + 1, torch.ones_like(gm, dtype=torch.long))
        has = (offset[1:m + 1] - offset[:m]) > 0
        bad = has & (ng[:m] != want)
        if bool(bad.any()):
            raise ValueError(f"{path}: n_goal of {int(bad.sum())} routes differs from the route store (another store?)")
    return dict(pads=pads, offset=offset.to(device), n_routes=n, p=float(p), path=str(path),
                meta=json.loads(str(z["meta"])) if "meta" in z.files else {})


def own_pads_start_step(own_cfg: dict, t_train: int) -> int:
    """First training step with own pads: own_cfg['start_step'], else round(own_cfg['start_frac'] * t_train), else 0."""
    if own_cfg.get("start_step") is not None:
        return int(own_cfg["start_step"])
    return int(round(float(own_cfg.get("start_frac", 0.0)) * int(t_train)))


def set_own_pads_step(own: dict | None, step: int) -> None:
    """Called by the train loop before every batch: own pads on from own['start_step'] (hard switch, as planned)."""
    if own is not None:
        own["active"] = step >= int(own.get("start_step", 0))


def gather_own_pads(own: dict, route_ix: torch.Tensor, slot: torch.Tensor, gen: torch.Generator):
    """Per sample: use the stored own pad (probability own['p'], only where the route and slot exist)? Returns
    use (B,) bool and pads (B,g,g) bool on route_ix's device (rows where use is False are empty). The draw always
    consumes B uniforms from gen when p > 0, so the loader's random stream does not depend on the store's content."""
    b = route_ix.shape[0]; dev = route_ix.device
    pads_cpu = own["pads"]; g = pads_cpu.shape[1]
    out = torch.zeros(b, g, g, dtype=torch.bool, device=dev)
    p = float(own["p"]) if own.get("active", True) else 0.0     # off before start_step (set per step by the train loop)
    if p <= 0:
        return torch.zeros(b, dtype=torch.bool, device=dev), out
    coin = torch.rand(b, device=dev, generator=gen) < p
    off = own["offset"]; n = int(own["n_routes"])
    in_store = route_ix < n
    r = route_ix.clamp(max=n - 1)
    count = off[r + 1] - off[r]
    use = coin & in_store & (slot >= 0) & (slot < count)
    if not bool(use.any()):
        return use, out
    idx = (off[r] + slot)[use].cpu()
    rows = pads_cpu[idx].to(dev, non_blocking=True)
    out[use] = unpack_pads(rows)
    return use, out


# ---------------------------------------------------------------- generation (needs the harness on sys.path)

def _sub_route(route: dict, ix: torch.Tensor, dev: torch.device) -> dict:
    """The wave fields of the routes ix (CPU route index) on the device, re-indexed 0..B-1, ragged path included."""
    off = route["wave_off"]; lo = off[ix]; cnt = off[ix + 1] - lo
    pos = torch.cat([torch.arange(int(a), int(a) + int(c)) for a, c in zip(lo.tolist(), cnt.tolist())]) if int(cnt.sum()) else torch.zeros(0, dtype=torch.long)
    return dict(origin=route["origin"][ix].to(dev), markers=route["markers"][ix].to(dev),
                wave_dist=route["wave_dist"][ix].to(dev), wave_goal_m=route["wave_goal_m"][ix].to(dev),
                wave_path=route["wave_path"][pos].to(dev), wave_off=torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)]).to(dev),
                crop_px=route.get("crop_px"))


def loader_n_goal(goal_m: torch.Tensor) -> torch.Tensor:
    """n_goal exactly as _wave_post computes it (float32 goal metres, 1 for unreachable)."""
    from .c4_tasks import WAVE_HORIZON_M
    return torch.where(torch.isfinite(goal_m), (goal_m / WAVE_HORIZON_M).floor().long() + 1, torch.ones_like(goal_m, dtype=torch.long))


@torch.no_grad()
def rollout_own_pads(route: dict, district: torch.Tensor, *, encoding, grid: int, batch: int, step_fn=None,
                     limit: int | None = None, log=print) -> dict:
    """Run every reachable route of `route` (a CPU route index from load_c4_route_index) through the teacher's
    schedule. step_fn(img (B,17,S,S), cond (B,2)) -> (B,g,g) bool is the network; None stores the teacher's outputs.
    Returns the arrays of the store (see the module docstring) plus timing."""
    import time
    from .c4_tasks import WAVE_KEYS, get_c4_task
    from .dataset_c1_gpu import build_image_from_crops
    from .dataset_c4_gpu import CROP_PX, _cut_windows, _marker_cols_rows, _wave_post, _wave_start_disc
    from .wave_search_schedule import wave_pass_schedule

    dev = district.device
    n = int(route["origin"].shape[0]) if limit is None else min(int(limit), int(route["origin"].shape[0]))
    crop = int(route.get("crop_px", CROP_PX)); s = crop // grid
    gm = route["wave_goal_m"][:n]
    ng = loader_n_goal(gm)
    ok = torch.isfinite(gm)
    n_slot = torch.where(ok, 2 * ng + 1, torch.zeros_like(ng))
    offset = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(n_slot, 0)]).numpy()
    P = int(offset[-1])
    pads = np.zeros((P, grid, grid // 8), np.uint8)
    slot_key = np.full(P, -1, np.int8); slot_k_in = np.full(P, -1, np.int16); slot_j = np.full(P, -1, np.int16)
    iou = np.zeros(P, np.float32)
    cond_of = {k: torch.tensor(get_c4_task(k).padded_cond_ids(), dtype=torch.long, device=dev) for k in TASK_KEYS}
    t0 = time.perf_counter(); n_pass = 0; n_done = 0; next_log = 5000
    for ngv in sorted(set(ng[ok].tolist())):
        members = torch.nonzero(ok & (ng == ngv), as_tuple=False).view(-1)
        sched = wave_pass_schedule(int(ngv))
        for c0 in range(0, members.numel(), batch):
            ix = members[c0:c0 + batch]; b = ix.numel()
            sub = _sub_route(route, ix, dev); lix = torch.arange(b, device=dev)
            layers = _cut_windows(district, sub["origin"], crop)
            mc, mr = _marker_cols_rows(sub["markers"], crop)
            img = build_image_from_crops(layers, encoding, marker_col=mc, marker_row=mr)
            del layers
            pad = _wave_start_disc(sub["markers"], grid, crop)
            zero_tgt = torch.zeros(b, grid, grid, dtype=torch.long, device=dev)
            base = torch.from_numpy(offset[ix.numpy()]).long()
            for wp in sched:
                kind = WAVE_KEYS[wp.task_key][1]
                x, tgt, _tok, _ = _wave_post(img, zero_tgt, [kind] * b, sub, lix, grid, crop, None,
                                             force_k_in=wp.k_in, force_j=wp.j_back)
                teacher = tgt > 0
                if step_fn is None:
                    out = teacher
                else:
                    x[:, 16] = pad.float().repeat_interleave(s, 1).repeat_interleave(s, 2)   # the network's own pad in
                    out = step_fn(x, cond_of[wp.task_key].expand(b, -1))
                del x, tgt
                inter = (out & teacher).flatten(1).sum(1).float(); union = (out | teacher).flatten(1).sum(1).float()
                pi = pass_position(wp.task_key, wp.k_in, wp.j_back, int(ngv))
                rows = (base + pi).numpy()
                pads[rows] = pack_pads(out).cpu().numpy()
                slot_key[rows] = TASK_KEYS.index(wp.task_key)
                slot_k_in[rows] = -1 if wp.task_key not in ("wave_first", "wave_step", "wave_reach") else (wp.k_in if wp.k_in is not None else (0 if wp.task_key == "wave_first" else int(ngv) - 1))
                slot_j[rows] = -1 if wp.task_key not in ("wave_back", "wave_origin") else (wp.j_back if wp.j_back is not None else 1)
                iou[rows] = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).cpu().numpy()
                pad = out
                n_pass += b
            n_done += b
            if log is not None and n_done >= next_log:
                log(f"  {n_done}/{int(ok.sum())} routes, {n_pass} passes, {time.perf_counter() - t0:.0f} s")
                next_log += 5000
    return dict(pads=pads, offset=offset, n_goal=torch.where(ok, ng, torch.zeros_like(ng)).numpy(), slot_key=slot_key,
                slot_k_in=slot_k_in, slot_j=slot_j, iou=iou, n_routes=n, n_run=int(ok.sum()), n_pass=n_pass,
                seconds=time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    import time
    harness_dir = Path(__file__).resolve().parents[1] / "harness"
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    import config
    import data
    import grid_vit_adapter
    from checkpoints import checkpoint_model_config
    from diagnostics import wave_search_rollout as ws
    from plain_gpt_module.checkpoint import read_checkpoint
    from plain_gpt_module.local_config import LocalGridViTConfig
    from plain_gpt_module.local_grid_vit import LocalGridViT
    from spatial_data.dataset_c4_gpu import load_c4_route_index, load_district_planes_v4

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: runs/checkpoints/RUN_ID.pt")
    ap.add_argument("--store", default=None, help="route store dir (default: the run's, c4_100k)")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=40000, help="first N routes of the split")
    ap.add_argument("--batch", type=int, default=128, help="routes per forward")
    ap.add_argument("--out", default=None, help="file name in the store dir (default c5_own_pads_{split}.npz)")
    ap.add_argument("--teacher", action="store_true", help="store the teacher's outputs (mapping check), no network")
    ap.add_argument("--grid", type=int, default=None, help="output cells per side (default: the run's cell size on the store's window)")
    args = ap.parse_args(argv)

    # the run's config, as wave_rollout_eval.py sets it up
    overrides = ws._overrides_for_run(args.run_id)
    meta_path = Path(config.diagnostics_dir) / args.run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta:
        ws._apply_meta_config(meta)
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.sync_in_chans_from_encoding(); data.setup_device()
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)        # as the harness does; the default draws 1 px markers
    device = data.device; device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    store = Path(args.store) if args.store else Path(config.c4_routes_dir())
    # the store's window decides the grid; the run's cell size is kept (256 m / 128 cells -> 512 m / 256 cells)
    from spatial_data.dataset_c4_gpu import store_crop_px
    trained_img, trained_grid = int(config.img_size), int(config.grid_out_size)
    crop = store_crop_px(store)
    grid = args.grid or trained_grid * crop // trained_img
    t_load = time.perf_counter()
    district = load_district_planes_v4(store.parent, device)
    route = load_c4_route_index(store, args.split, torch.device("cpu"), max_items=args.limit)   # CPU: moved per batch
    if "wave_dist" not in route:
        raise SystemExit(f"no c5_wave_{args.split}.npz covering {args.limit} routes in {store}")
    t_load = time.perf_counter() - t_load

    step_fn = None; ckpt = None; step = None
    if not args.teacher:
        ckpt = ws._resolve_checkpoint(args.run_id, args.checkpoint)
        doc = read_checkpoint(ckpt, map_location="cpu")
        cfg = checkpoint_model_config(doc) or dict(meta.get("model") or {})
        if (int(cfg.get("img_size", crop)), int(cfg.get("grid_out_size", grid))) != (crop, grid):
            print(f"note: checkpoint trained at {cfg.get('img_size')} px / {cfg.get('grid_out_size')} cells, run at {crop} px / {grid} cells")
        cfg = dict(cfg, img_size=crop, grid_out_size=grid)   # no learned weight depends on the window (wave_rollout_eval.py)
        model = LocalGridViT(LocalGridViTConfig(**cfg)).to(device)
        grid_vit_adapter.load_checkpoint(ckpt, model=model)
        model.train(False)
        step = grid_vit_adapter.checkpoint_saved_step(ckpt)
        fg = model.config.foreground_class
        task_id0 = torch.zeros(args.batch, dtype=torch.long, device=device)   # unused with cond tokens; the forward wants one

        def step_fn(x, cond):
            with torch.autocast(device_type=device_type, dtype=data.dtype):
                logits, _ = ws.model_forward(model, x, None, task_id0[:x.shape[0]], cond)
            return ws._prob_fg(logits.float(), fg) >= 0.5           # the same 0.5 read-out as the rollout eval

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res = rollout_own_pads(route, district, encoding=config.encoding_mode, grid=grid, batch=args.batch, step_fn=step_fn,
                           limit=args.limit)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device_type == "cuda" else 0.0
    out = store / (args.out or f"c5_own_pads_{args.split}.npz")
    info = dict(run=args.run_id, checkpoint=str(ckpt) if ckpt else None, checkpoint_step=step, teacher=bool(args.teacher),
                store=str(store), split=args.split, grid=grid, batch=args.batch)
    t_save = time.perf_counter()
    save_own_pads(out, pads=res["pads"], offset=res["offset"], n_goal=res["n_goal"], slot_key=res["slot_key"],
                  slot_k_in=res["slot_k_in"], slot_j=res["slot_j"], iou=res["iou"], meta=info)
    t_save = time.perf_counter() - t_save
    iou = res["iou"]; key = res["slot_key"]
    stats = dict(**info, routes=res["n_routes"], routes_run=res["n_run"], passes=res["n_pass"], slots=int(len(iou)),
                 seconds_rollout=round(res["seconds"], 1), seconds_load=round(t_load, 1), seconds_save=round(t_save, 1),
                 passes_per_s=round(res["n_pass"] / max(res["seconds"], 1e-9), 1), peak_gpu_gb=round(peak, 2),
                 file_mb=round(out.stat().st_size / 2 ** 20, 1),
                 iou_vs_teacher={k: round(float(iou[key == i].mean()), 4) for i, k in enumerate(TASK_KEYS) if (key == i).any()},
                 iou_vs_teacher_all=round(float(iou.mean()), 4) if len(iou) else None)
    (out.parent / (out.stem + "_stats.json")).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
