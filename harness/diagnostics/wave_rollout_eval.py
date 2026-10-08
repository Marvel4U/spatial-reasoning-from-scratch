#!/usr/bin/env python3
"""Rollout eval over many val routes, numbers only (2 Oct 2026). The network runs on its own scratchpad.

Two modes:
  scheduled  the pass order and the state token come from the teacher's schedule (as in wave_search_rollout.py);
             the pad is the network's own output from --start on. Measures how the pad drifts when the state is right.
  free       the network's own output token picks the next state (SEARCH -> BACK -> DRAW -> DONE); it stops on DONE,
             on a token that is no state, or after 2 * n_goal + 4 passes. Nothing from the teacher after the start.
Per route the final pad is scored as a route: connected start -> goal through lit cells, its length against the
teacher's walking length, share of its cells that are entirely building. success = connected, length <= 1.10 x the
teacher's, no building cell. Free mode also scores how clean the final drawing is (5 Oct, for the cleanup pass,
spatial_data/c5_cleanup.py): lit cells, cells of the 3-wide shortest chain through it, excess_ratio = lit / chain
cells, clean = excess_ratio <= 1.25 (unconnected routes count as not clean; excess_ratio and the IoU against the
own chain are averaged over connected routes only), solved_and_clean.

Usage (from harness/): PYTHONPATH=. ../.venv/bin/python diagnostics/wave_rollout_eval.py RUN_ID [--n 200]
       [--mode both|scheduled|free] [--start 1|first_back] [--checkpoint PATH]
       [--crops DIR --store NAME --split val|test] [--img-size PX --grid CELLS]
Writes runs/diagnostics/RUN_ID/wave_rollout_eval.json.

Other window sizes (4 Oct): when the store's window (store.json crop_px) differs from the size the checkpoint was
trained at, the model is rebuilt for the store's window with the same cell size (a 256 m network with 128 cells of
2 m becomes a 512 m network with 256 cells of 2 m) and the trained weights are loaded unchanged. This works because
no learned weight depends on the number of patches: the sin-cos positions and the relative-bias bucket table are
rebuilt from the patch grid, the bias itself is learned per clipped offset (+-7 patches), and the grid head maps one
patch token to its own 8 x 8 cells. What the network has never seen are absolute positions beyond its trained patch
grid (sin-cos rows 16..31) and attention over 4x as many tokens. --img-size / --grid force other values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import config
import data
import grid_vit_adapter
from checkpoints import checkpoint_model_config
from diagnostics import wave_search_rollout as ws
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from spatial_data.c4_tasks import TOK_BACK, TOK_DONE, TOK_DRAW, TOK_SEARCH, WAVE_HORIZON_M
from spatial_data.c5_cleanup import chain_success, cleanliness, shortest_lit_chain
from spatial_data.dataset_c4_gpu import _wave_start_disc
from spatial_data.wave_search_schedule import n_goal_from_m, wave_pass_schedule

K3 = np.ones((3, 3), bool)
STATE_KEY = {TOK_SEARCH: "wave_first", TOK_BACK: "wave_back", TOK_DRAW: "wave_draw"}   # a task key carrying that state token


def route_score(mask: np.ndarray, start_c, goal_c, blocked: np.ndarray, goal_m: float, cell_m: float) -> dict:
    """mask (g,g) bool = the final pad. Shortest way start -> goal through lit cells (8 neighbours); the graph is
    spatial_data.c5_cleanup.shortest_lit_chain, shared with the cleanup targets."""
    chain = shortest_lit_chain(mask, start_c, goal_c, cell_m)
    if chain is None:
        return dict(connected=0.0, length_ratio=float("nan"), building_share=float("nan"), success=0.0, lit_cells=float(mask.sum()))
    ys, xs, length = chain
    bshare = float(blocked[ys, xs].mean())
    ratio = float(length / max(goal_m, 1e-6))
    return dict(connected=1.0, length_ratio=ratio, building_share=bshare, success=float(chain_success(ratio, bshare)),
                lit_cells=float(mask.sum()))


def iou_4m(teacher: np.ndarray, pred: np.ndarray, tol: int) -> float:
    """IoU where a cell counts as matched if the other mask has a lit cell within tol cells."""
    td, pd_ = ndimage.binary_dilation(teacher, K3, iterations=tol), ndimage.binary_dilation(pred, K3, iterations=tol)
    tp_p, tp_t = (pred & td).sum(), (teacher & pd_).sum()
    denom = pred.sum() + teacher.sum() - 0.5 * (tp_p + tp_t)
    return float(0.5 * (tp_p + tp_t) / denom) if denom > 0 else 1.0


def eval_geometry(trained_img: int, trained_grid: int, store_crop: int, img_size: int | None = None,
                  grid: int | None = None) -> tuple[int, int]:
    """(img_size, grid) the model is built with: the store's window, and the trained cell size kept on it
    (256 px / 128 cells on a 512 px store -> 512 / 256), unless --img-size / --grid say otherwise."""
    img = int(img_size or store_crop)
    g = int(grid or trained_grid * img // trained_img)
    if img % g != 0:
        raise ValueError(f"grid {g} does not divide the window {img} px")
    return img, g


@torch.no_grad()
def step(model, img, pad, task_id, cond, crop, grid, fg, device_type):
    x = img.clone(); x[16] = ws._pad_grid_to_img(pad, crop, grid)
    with torch.autocast(device_type=device_type, dtype=data.dtype):
        logits, _ = ws.model_forward(model, x.unsqueeze(0), None, task_id.unsqueeze(0), cond.unsqueeze(0))
        tl = ws.token_logits(model, x.unsqueeze(0), task_id.unsqueeze(0), cond.unsqueeze(0))
    pred = ws._prob_fg(logits[0], fg) >= 0.5
    return pred, (int(tl.argmax(dim=-1).item()) if tl is not None else -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", default="both", choices=("both", "scheduled", "free"))
    ap.add_argument("--start", default="1", help="1 = from the start disc; first_back = teacher through the reach pass")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--crops", default=None, help="crops dir of another district (default: the run's)")
    ap.add_argument("--store", default=None, help="route store under that crops dir, e.g. c4_256_val (default: the run's)")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--tag", default=None, help="suffix of the output file (default: from crops/store/split when given)")
    ap.add_argument("--img-size", type=int, default=None, help="build the model for this window in px (default: the store's crop_px)")
    ap.add_argument("--grid", type=int, default=None, help="output cells per side (default: keep the trained cell size)")
    args = ap.parse_args()
    overrides = ws._overrides_for_run(args.run_id)
    meta_path = Path(config.diagnostics_dir) / args.run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta:
        ws._apply_meta_config(meta)
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    if args.crops:                                            # another district: crops dir + route store subdir
        config.c3_crops_dir = lambda: Path(args.crops)
        config.c4_routes_dir = lambda: Path(args.crops) / (args.store or "c4_256_val")
    # The window of the store decides the model's input size; the trained cell size is kept unless --grid says
    # otherwise. The loader below reads config.grid_out_size, so this has to happen before it is built.
    from spatial_data.dataset_c4_gpu import store_crop_px
    trained_img, trained_grid = int(config.img_size), int(config.grid_out_size)
    eval_img, eval_grid = eval_geometry(trained_img, trained_grid, store_crop_px(Path(config.c4_routes_dir())),
                                        args.img_size, args.grid)
    config.img_size, config.grid_out_size = eval_img, eval_grid
    config.sync_in_chans_from_encoding(); data.setup_device()
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)        # as the harness does; the default draws 1 px markers
    device = data.device; device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    grid = config.grid_out_size
    if args.split != "val":
        _orig_load = ws.load_c4_route_index
        ws.load_c4_route_index = lambda d, split, dev, max_items=None: _orig_load(d, args.split, dev, max_items=max_items)
    loader, val_routes = ws.build_wave_rollout_loader()
    ckpt = ws._resolve_checkpoint(args.run_id, args.checkpoint)
    doc = read_checkpoint(ckpt, map_location="cpu")
    cfg = checkpoint_model_config(doc) or dict(meta.get("model") or {})
    trained_img, trained_grid = int(cfg.get("img_size", trained_img)), int(cfg.get("grid_out_size", trained_grid))
    if (eval_img, eval_grid) != (trained_img, trained_grid):
        print(f"note: checkpoint trained at {trained_img} px / {trained_grid} cells, evaluated at {eval_img} px / {eval_grid} cells")
    cfg = dict(cfg, img_size=eval_img, grid_out_size=eval_grid)
    model = LocalGridViT(LocalGridViTConfig(**cfg)).to(device)
    grid_vit_adapter.load_checkpoint(ckpt, model=model)
    model.train(False)
    fg = model.config.foreground_class
    crop = int(val_routes.get("crop_px", config.img_size)); s = crop // grid; cell_m = float(s); tol = max(1, int(round(4.0 / cell_m)))
    store = Path(config.c4_routes_dir())
    goal_all = np.load(store / f"c5_wave_{args.split}.npz")["goal_m"]
    rz = np.load(store / f"c4_routes_{args.split}.npz"); origin = rz["origin"]; markers = rz["markers"]
    bld = np.load(store.parent / "district_labels_1m.npz")["surface"] == 3
    start_mode = args.start

    sched_rows, free_rows, per_pass = [], [], {}
    n_done = 0
    for rix in range(min(args.n, loader.n_routes)):
        goal_m = float(goal_all[rix])
        if not np.isfinite(goal_m):
            continue
        n_done += 1
        n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M); schedule = wave_pass_schedule(n_goal)
        ox, oy = int(origin[rix][0]), int(origin[rix][1])
        blocked = bld[oy:oy + crop, ox:ox + crop].reshape(grid, s, grid, s).all(axis=(1, 3))
        start_c = (int(markers[rix][0][0]) // s, int(markers[rix][0][1]) // s); goal_c = (int(markers[rix][1][0]) // s, int(markers[rix][1][1]) // s)
        first_model_pass = n_goal + 1 if start_mode == "first_back" else int(start_mode)
        teacher_final = None; teacher_pad_at_start = None

        if args.mode in ("both", "scheduled"):
            pad = _wave_start_disc(loader.route["markers"][rix:rix + 1], grid, crop)[0]
            ious, tok_ok, after_origin = [], [], None
            for pi, wp in enumerate(schedule, start=1):
                img_t, tgt_t, task_id, cond, tok_gt = loader.fetch_wave_pass(rix, wp.task_key, force_k_in=wp.k_in, force_j=wp.j_back)
                teacher = ws._teacher_mask_to_grid(tgt_t, grid, crop)
                if pi < first_model_pass:
                    pad = teacher
                    if pi == first_model_pass - 1:
                        teacher_pad_at_start = teacher
                    continue
                pad, tok = step(model, img_t, pad, task_id, cond, crop, grid, fg, device_type)
                t_np, p_np = teacher.cpu().numpy(), pad.cpu().numpy()
                iou = ws._mask_iou(t_np, p_np); ious.append(iou); tok_ok.append(float(tok == tok_gt))
                a = per_pass.setdefault(wp.task_key, [0, 0.0, 0.0, 0.0]); a[0] += 1; a[1] += iou; a[2] += iou_4m(t_np, p_np, tol); a[3] += float(tok == tok_gt)
                if wp.task_key == "wave_origin":
                    after_origin = p_np
                if wp.task_key == "wave_draw":
                    teacher_final = t_np
            row = dict(route=rix, n_goal=n_goal, goal_m=goal_m, passes=len(ious), iou_mean=float(np.mean(ious)), iou_last=float(ious[-1]),
                       tok_acc=float(np.mean(tok_ok)), tok_all=float(all(tok_ok)))
            row.update({"final_" + k: v for k, v in route_score(pad.cpu().numpy(), start_c, goal_c, blocked, goal_m, cell_m).items()})
            if after_origin is not None:
                row.update({"origin_" + k: v for k, v in route_score(after_origin, start_c, goal_c, blocked, goal_m, cell_m).items()})
            if teacher_final is not None:
                row["final_iou4m_vs_teacher_path"] = iou_4m(teacher_final, pad.cpu().numpy(), tol)
            sched_rows.append(row)

        if args.mode in ("both", "free"):
            base = {}
            for tok_state, key in STATE_KEY.items():
                img_t, tgt_t, task_id, cond, _ = loader.fetch_wave_pass(rix, key)
                base[tok_state] = (img_t, task_id, cond)
                if key == "wave_draw":
                    teacher_final = ws._teacher_mask_to_grid(tgt_t, grid, crop).cpu().numpy()
            if start_mode == "first_back":
                wp = schedule[n_goal - 1]                   # the reach pass; its target is the pad the backward part starts from
                _, tgt_t, _, _, _ = loader.fetch_wave_pass(rix, wp.task_key, force_k_in=wp.k_in, force_j=wp.j_back)
                pad = ws._teacher_mask_to_grid(tgt_t, grid, crop); state = TOK_BACK; budget = n_goal + 4
            else:
                pad = _wave_start_disc(loader.route["markers"][rix:rix + 1], grid, crop)[0]; state = TOK_SEARCH; budget = 2 * n_goal + 4
            toks, stop = [], "budget"
            for _ in range(budget):
                img_t, task_id, cond = base[state]
                pad, tok = step(model, img_t, pad, task_id, cond, crop, grid, fg, device_type)
                toks.append(tok)
                if tok == TOK_DONE:
                    stop = "done"; break
                if tok not in STATE_KEY:
                    stop = "bad_token"; break
                state = tok
            want = [{"SEARCH": TOK_SEARCH, "BACK": TOK_BACK, "DRAW": TOK_DRAW, "DONE": TOK_DONE}[wp.tok_out] for wp in schedule]
            if start_mode == "first_back":
                want = want[n_goal:]
            row = dict(route=rix, n_goal=n_goal, goal_m=goal_m, passes=len(toks), passes_teacher=len(want), finished=float(stop == "done"),
                       stop=stop, tokens_as_teacher=float(toks == want))
            p_np = pad.cpu().numpy()
            row.update({"final_" + k: v for k, v in route_score(p_np, start_c, goal_c, blocked, goal_m, cell_m).items()})
            row["final_iou4m_vs_teacher_path"] = iou_4m(teacher_final, p_np, tol)
            row["success_and_finished"] = float(row["final_success"] == 1.0 and stop == "done")
            cl = cleanliness(p_np, start_c, goal_c, cell_m)        # how much of the final drawing is its own route
            row.update({"final_" + k: v for k, v in cl.items()})
            row["solved_and_clean"] = float(row["final_success"] == 1.0 and cl["clean"] == 1.0)
            free_rows.append(row)

    def agg(rows, keys):
        out = {"n": len(rows)}
        for k in keys:
            v = np.array([r[k] for r in rows if k in r], float)
            out[k] = float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan")
        return out

    # the first three bands are those of the 256 m reports; the finer long bands matter on 512 m windows (up to ~28 fronts)
    bands = (("n_goal 2-3", 2, 3), ("n_goal 4-5", 4, 5), ("n_goal 6+", 6, 999), ("n_goal 6-8", 6, 8), ("n_goal 9+", 9, 999),
             ("n_goal 9-12", 9, 12), ("n_goal 13+", 13, 999))

    def by_len(rows, keys):
        return {name: agg([r for r in rows if lo <= r["n_goal"] <= hi], keys) for name, lo, hi in bands}

    out = dict(run=args.run_id, checkpoint=str(ckpt), routes=n_done, start=start_mode, cell_m=cell_m,
               crops=str(config.c3_crops_dir()), store=str(store), split=args.split,
               img_size=eval_img, grid=eval_grid, trained_img_size=trained_img, trained_grid=trained_grid)
    if sched_rows:
        keys = ["iou_mean", "iou_last", "tok_acc", "tok_all", "final_connected", "final_success", "final_length_ratio", "final_building_share",
                "final_iou4m_vs_teacher_path", "origin_connected", "origin_success", "final_lit_cells"]
        out["scheduled"] = dict(all=agg(sched_rows, keys), **by_len(sched_rows, keys),
                                per_pass={k: dict(n=a[0], iou=a[1] / a[0], iou4m=a[2] / a[0], tok_acc=a[3] / a[0]) for k, a in per_pass.items()})
    if free_rows:
        keys = ["finished", "tokens_as_teacher", "passes", "passes_teacher", "final_connected", "final_success", "success_and_finished",
                "final_length_ratio", "final_building_share", "final_iou4m_vs_teacher_path", "final_lit_cells",
                "final_chain_cells", "final_excess_ratio", "final_clean", "solved_and_clean", "final_iou_own_chain"]
        stops = {}
        for r in free_rows:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        out["free"] = dict(all=agg(free_rows, keys), stops=stops, **by_len(free_rows, keys))
        out["free_per_route"] = [dict(route=r["route"], n_goal=r["n_goal"], solved=r["final_success"], stop=r["stop"],
                                     clean=r["final_clean"], excess_ratio=r["final_excess_ratio"]) for r in free_rows]
    odir = Path(config.diagnostics_dir) / args.run_id; odir.mkdir(parents=True, exist_ok=True)
    if args.tag is not None:
        tag = args.tag
    elif args.crops or args.split != "val":
        tag = "_" + "_".join(x for x in ((Path(args.crops).parts[-3] if args.crops else ""), args.store or "", args.split) if x)
    else:
        tag = ""
    name = f"wave_rollout_eval{tag}.json" if start_mode == "1" else f"wave_rollout_eval_{start_mode}{tag}.json"
    (odir / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("wrote", odir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
