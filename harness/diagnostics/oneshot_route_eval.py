#!/usr/bin/env python3
"""One-shot path finding scored by the wave search's route rule (5 Oct 2026), so the two methods can be compared
number for number. One forward pass per route: the map and the two markers in, a path mask out (task `detour`,
C4 report §7, C5_PLAN §5-6 R0), thresholded at 0.5 exactly as the search thresholds its scratchpad. The mask is then
scored with `route_score` from wave_rollout_eval.py, imported, not copied: connected start -> goal through lit cells
(8 neighbours, the 3 x 3 cells around each marker count as lit), no cell of the connecting path entirely building,
length <= 1.10 x the teacher's walking length `goal_m` from the store's c5_wave_{split}.npz. Same routes, same
length bands (fronts of 40 m), same rule as the search's free-running success.

Caveat on the length rule: the one-shot model was trained on the geometric shortest path (visibility graph, the
taut string around buildings), while `goal_m` is the walking length of the clearance-priced path the search was
taught, which is longer wherever the taut path hugs a wall. The 1.10 x goal_m rule is therefore slightly lenient for
the one-shot model. The output also carries the stricter reading against the geometric length (`path_px` of the route
store, 1 px = 1 m) as `geom_*`, for reference; the headline number is the goal_m one.

Other caveats: a model with 64 output cells (4 m) is scored on 4 m cells, where "entirely building" needs all 16
pixels of a cell to be building, so it is more lenient than at 2 m cells; compare 2 m with 2 m. On the CPU the model
runs in float32, while training and the search evals ran in bf16 on the GPU; a few routes near the 0.5 threshold can
flip.

Usage (from harness/): PYTHONPATH=. python diagnostics/oneshot_route_eval.py RUN_ID [--n 500] [--task detour]
       [--crops DIR --store NAME --split val|test] [--checkpoint PATH] [--batch 16] [--threads 3] [--tag SUFFIX]
       [--img-size PX --grid CELLS] [--save-pads FILE.npz]
--save-pads keeps the thresholded mask of every route (for harness/diagnostics/drawing_to_polyline.py).
Writes runs/diagnostics/RUN_ID/oneshot_route_eval{tag}.json; tag as in wave_rollout_eval.py (empty for the run's own
store and split, else _{district}_{store}_{split}). Runs on the CPU when CUDA_VISIBLE_DEVICES is empty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
import data
import grid_vit_adapter
from checkpoints import checkpoint_model_config
from diagnostics import wave_search_rollout as ws
from diagnostics.wave_rollout_eval import eval_geometry, iou_4m, route_score, save_pads
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from spatial_data.c4_tasks import WAVE_HORIZON_M, get_c4_task
from spatial_data.dataset_c4_gpu import C4GpuValLoader, load_c4_route_index, load_district_planes_v4, store_crop_px
from spatial_data.wave_search_schedule import n_goal_from_m

# the bands of wave_rollout_eval.py: the first three are those of the 256 m reports, the finer long ones matter at 512 m
BANDS = (("n_goal 2-3", 2, 3), ("n_goal 4-5", 4, 5), ("n_goal 6+", 6, 999), ("n_goal 6-8", 6, 8), ("n_goal 9+", 9, 999),
         ("n_goal 9-12", 9, 12), ("n_goal 13+", 13, 999))


def configure_for_run(run_id: str, crops: str | None, store: str | None, img_size: int | None = None,
                      grid: int | None = None, threads: int | None = None) -> dict:
    """Set the global config the way the run was trained (meta.json config, then the ledger overrides), point it at
    another route store if asked, and fix the model's window and grid for that store. Returns what the caller needs."""
    meta_path = Path(config.diagnostics_dir) / run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta:
        ws._apply_meta_config(meta)
    for k, v in ws._overrides_for_run(run_id).items():
        if hasattr(config, k):
            setattr(config, k, v)
    if crops:                                                 # another district: crops dir + route store subdir
        config.c3_crops_dir = lambda: Path(crops)
        config.c4_routes_dir = lambda: Path(crops) / (store or "c4_256_val")
    elif store:                                               # another store of the run's own district
        config.c4_routes_subdir = store
    trained_img, trained_grid = int(config.img_size), int(config.grid_out_size)
    eval_img, eval_grid = eval_geometry(trained_img, trained_grid, store_crop_px(Path(config.c4_routes_dir())), img_size, grid)
    config.img_size, config.grid_out_size = eval_img, eval_grid
    if threads:                                               # CPU threads (only used when there is no GPU)
        config.cpu_num_threads = int(threads)
    config.sync_in_chans_from_encoding(); data.setup_device()
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)        # as the harness does; the default draws 1 px markers
    return dict(meta=meta, eval_img=eval_img, eval_grid=eval_grid)


def load_model(run_id: str, checkpoint: Path | None, meta: dict, eval_img: int, eval_grid: int):
    """The model from the checkpoint's own config (incl. its condition-table size n_cond_emb), rebuilt for the
    evaluation window as wave_rollout_eval.py does. Returns (model, checkpoint path, trained (img, grid))."""
    ckpt = ws._resolve_checkpoint(run_id, checkpoint)
    doc = read_checkpoint(ckpt, map_location="cpu")
    cfg = checkpoint_model_config(doc) or dict(meta.get("model") or {})
    trained = (int(cfg.get("img_size", eval_img)), int(cfg.get("grid_out_size", eval_grid)))
    if (eval_img, eval_grid) != trained:
        print(f"note: {run_id} trained at {trained[0]} px / {trained[1]} cells, evaluated at {eval_img} px / {eval_grid} cells")
    model = LocalGridViT(LocalGridViTConfig(**dict(cfg, img_size=eval_img, grid_out_size=eval_grid))).to(data.device)
    grid_vit_adapter.load_checkpoint(ckpt, model=model)
    model.train(False)
    return model, ckpt, trained


def build_task_loader(task: str, split: str, batch: int):
    """The val loader of the one-shot task on the configured store, as the harness builds it for c4_eval_task_key
    detour (data.py): C4GpuValLoader over the route store, one task, no c3 mix."""
    crops_dir, routes_dir = Path(config.c3_crops_dir()), Path(config.c4_routes_dir())
    district = load_district_planes_v4(crops_dir, data.device)
    routes = load_c4_route_index(routes_dir, split, data.device, max_items=None)
    return C4GpuValLoader(get_c4_task(task), config.encoding_mode, config.grid_out_size, batch,
                          district=district, route_index=routes), routes


@torch.no_grad()
def oneshot_masks(model, loader, route_ixs: list[int], batch: int) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass per route. Returns (pred (N,g,g) bool at p >= 0.5, teacher target (N,g,g) bool)."""
    fg = model.config.foreground_class
    on_gpu = str(data.device).startswith("cuda")
    preds, tgts = [], []
    for lo in range(0, len(route_ixs), batch):
        ix = torch.tensor(route_ixs[lo:lo + batch], dtype=torch.long, device=data.device)
        img, tgt, task_index, cond = loader._build(ix)
        if on_gpu:
            with torch.autocast(device_type="cuda", dtype=data.dtype):
                logits, _ = ws.model_forward(model, img, None, task_index, cond)
        else:                                                  # float32 on the CPU (autocast there would want bf16)
            logits, _ = ws.model_forward(model, img, None, task_index, cond)
        preds.append((ws._prob_fg(logits.float(), fg) >= 0.5).cpu().numpy())
        tgts.append((tgt == fg).cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def route_geometry(store: Path, split: str, crop: int, grid: int):
    """What route_score needs, read as wave_rollout_eval.py reads it: goal_m, markers, origins, the building plane,
    and the geometric shortest-path length (path_px, 1 px = 1 m) of the one-shot teacher."""
    goal_all = np.load(store / f"c5_wave_{split}.npz")["goal_m"]
    rz = np.load(store / f"c4_routes_{split}.npz")
    bld = np.load(store.parent / "district_labels_1m.npz")["surface"] == 3
    return goal_all, rz["origin"], rz["markers"], rz["path_px"], bld


def score_route(pred: np.ndarray, rix: int, geo, crop: int, grid: int) -> dict:
    goal_all, origin, markers, path_px, bld = geo
    s = crop // grid; cell_m = float(s)
    ox, oy = int(origin[rix][0]), int(origin[rix][1])
    blocked = bld[oy:oy + crop, ox:ox + crop].reshape(grid, s, grid, s).all(axis=(1, 3))
    start_c = (int(markers[rix][0][0]) // s, int(markers[rix][0][1]) // s)
    goal_c = (int(markers[rix][1][0]) // s, int(markers[rix][1][1]) // s)
    goal_m = float(goal_all[rix])
    out = route_score(pred, start_c, goal_c, blocked, goal_m, cell_m)
    geom = route_score(pred, start_c, goal_c, blocked, float(path_px[rix]), cell_m)
    out.update(geom_length_ratio=geom["length_ratio"], geom_success=geom["success"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id"); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--task", default="detour", help="the one-shot task key (default detour)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--crops", default=None, help="crops dir of another district (default: the run's)")
    ap.add_argument("--store", default=None, help="route store under that crops dir, e.g. c4_256_val (default: the run's)")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=3, help="CPU threads when running without a GPU")
    ap.add_argument("--tag", default=None, help="suffix of the output file (default: from crops/store/split when given)")
    ap.add_argument("--img-size", type=int, default=None, help="build the model for this window in px (default: the store's crop_px)")
    ap.add_argument("--grid", type=int, default=None, help="output cells per side (default: keep the trained cell size)")
    ap.add_argument("--save-pads", type=Path, default=None, help="npz with the thresholded mask per route")
    args = ap.parse_args()

    setup = configure_for_run(args.run_id, args.crops, args.store, args.img_size, args.grid, args.threads)
    eval_img, eval_grid = setup["eval_img"], setup["eval_grid"]
    model, ckpt, (trained_img, trained_grid) = load_model(args.run_id, args.checkpoint, setup["meta"], eval_img, eval_grid)
    loader, routes = build_task_loader(args.task, args.split, args.batch)
    crop = int(routes.get("crop_px", config.img_size)); grid = eval_grid; cell_m = float(crop // grid)
    tol = max(1, int(round(4.0 / cell_m)))
    store = Path(config.c4_routes_dir())
    geo = route_geometry(store, args.split, crop, grid)
    goal_all = geo[0]

    # the routes the search eval scores: the first n with a finite walking length
    todo = [r for r in range(min(args.n, loader.n_routes)) if np.isfinite(goal_all[r])]
    pred, tgt = oneshot_masks(model, loader, todo, args.batch)
    rows = []
    for i, rix in enumerate(todo):
        goal_m = float(goal_all[rix]); n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M)
        sc = score_route(pred[i], rix, geo, crop, grid)
        rows.append(dict(route=rix, n_goal=n_goal, goal_m=goal_m, geom_m=float(geo[3][rix]),
                         **{"final_" + k: v for k, v in sc.items()},
                         final_iou4m_vs_teacher_path=iou_4m(tgt[i], pred[i], tol)))

    keys = ["final_connected", "final_success", "final_length_ratio", "final_building_share", "final_lit_cells",
            "final_geom_success", "final_geom_length_ratio", "final_iou4m_vs_teacher_path"]

    def agg(rs):
        out = {"n": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            out[k] = float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan")
        return out

    out = dict(run=args.run_id, checkpoint=str(ckpt), task=args.task, routes=len(rows), cell_m=cell_m,
               crops=str(config.c3_crops_dir()), store=str(store), split=args.split, device=str(data.device),
               img_size=eval_img, grid=eval_grid, trained_img_size=trained_img, trained_grid=trained_grid,
               marker_radius_px=int(config.marker_radius_px), encoding_mode=config.encoding_mode,
               rule="route_score of wave_rollout_eval.py on the one-shot mask at p >= 0.5; length vs goal_m (clearance-priced "
                    "walking length, lenient for a model trained on the geometric path); geom_* = the same against path_px",
               oneshot=dict(all=agg(rows), **{name: agg([r for r in rows if lo <= r["n_goal"] <= hi]) for name, lo, hi in BANDS}),
               oneshot_per_route=[dict(route=r["route"], n_goal=r["n_goal"], solved=r["final_success"], connected=r["final_connected"],
                                       length_ratio=r["final_length_ratio"], lit_cells=r["final_lit_cells"],
                                       geom_solved=r["final_geom_success"]) for r in rows])
    odir = Path(config.diagnostics_dir) / args.run_id; odir.mkdir(parents=True, exist_ok=True)
    if args.tag is not None:
        tag = args.tag
    elif args.crops or args.store or args.split != "val":
        tag = "_" + "_".join(x for x in ((Path(args.crops).parts[-3] if args.crops else ""), args.store or "", args.split) if x)
    else:
        tag = ""
    path = odir / f"oneshot_route_eval{tag}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if args.save_pads is not None:
        save_pads(args.save_pads, list(zip(todo, pred)), kind="oneshot", run=args.run_id, checkpoint=str(ckpt),
                  store=str(store), split=args.split, crop=crop, grid=grid, origin=geo[1], markers=geo[2],
                  eval_json=str(path))
    print(json.dumps({k: v for k, v in out.items() if k != "oneshot_per_route"}, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
