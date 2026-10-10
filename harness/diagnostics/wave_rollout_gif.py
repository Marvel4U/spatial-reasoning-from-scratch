#!/usr/bin/env python3
"""Animation of one free-running wave search (2 Oct 2026): buildings and the two markers in one colour, the network's
scratchpad after every pass in another. The network starts from the start disc; its own output is the next scratchpad
and its own token picks the next state, until DONE.

Usage (from harness/): PYTHONPATH=. ../.venv/bin/python diagnostics/wave_rollout_gif.py RUN_ID --route 363 [--scale 3]
       [--crops DIR --store c4_256_val --split val|test]   # another district, as in wave_rollout_eval.py
Writes runs/diagnostics/RUN_ID/wave_search{tag}_route{route}.gif (+ _final.png); tag is empty for the run's own store,
else _{district}_{store}_{split} (e.g. _maze_v1_c4_256_val_val).

--cleanup-run RUN_ID (10 Oct): after the search ends, ONE extra draw pass of another network (a cleanup-trained one,
spatial_data/c5_cleanup.py) on the final drawing: state DRAW (task key wave_draw), the same input image with the final
drawing as the pad in channel 16, as harness/diagnostics/wave_rollout_eval.py `step` does. The pass is appended as a
last frame "cleanup pass (separate network)"; the script prints lit cells and route_score (connected start -> goal,
length / teacher, building share, success) before and after it, and the file name gets _cleanup_{that run}.
Both networks must read the same input encoding (the cleanup runs differ from their search runs only in training
data and loss weights). --cleanup-input draw_in applies the pass instead to the drawing that entered the search's own
last DRAW pass (the input the cleanup network was trained on: the origin-pass output), i.e. it replaces that DRAW
pass; the frame title then says so. --out writes the GIF to that path instead. --save-frames NPZ keeps the pad of every
frame (packed bits, with the frame titles and the state tokens), e.g. for diagnostics/concept_three_methods_figure.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
import data
import grid_vit_adapter
from checkpoints import checkpoint_model_config
from diagnostics import wave_search_rollout as ws
from diagnostics.wave_rollout_eval import STATE_KEY, route_score, step
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from spatial_data.c4_tasks import TOK_BACK, TOK_DONE, TOK_DRAW, TOK_SEARCH, WAVE_HORIZON_M
from spatial_data.dataset_c4_gpu import _wave_start_disc
from spatial_data.wave_search_schedule import n_goal_from_m

NAMES = {TOK_SEARCH: "SEARCH", TOK_BACK: "BACK", TOK_DRAW: "DRAW", TOK_DONE: "DONE"}
BG, MAP, PAD = (247, 245, 240), (52, 58, 70), (232, 106, 23)        # ground, buildings + markers, scratchpad


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id"); ap.add_argument("--route", type=int, default=363); ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--ms", type=int, default=700)
    ap.add_argument("--crops", default=None, help="crops dir of another district (default: the run's)")
    ap.add_argument("--store", default=None, help="route store under that crops dir, e.g. c4_256_val (default: the run's)")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--tag", default=None, help="suffix of the output file (default: from crops/store/split when given)")
    ap.add_argument("--cleanup-run", default=None, help="run id of a network whose draw pass is applied once to the final drawing")
    ap.add_argument("--cleanup-input", default="final", choices=("final", "draw_in"),
                    help="final: the search's final drawing; draw_in: the drawing that entered its last DRAW pass")
    ap.add_argument("--save-frames", type=Path, default=None, help="npz with the pad of every frame")
    ap.add_argument("--out", type=Path, default=None, help="GIF path (default: runs/diagnostics/RUN_ID/...)")
    args = ap.parse_args()
    meta_path = Path(config.diagnostics_dir) / args.run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta:
        ws._apply_meta_config(meta)
    for k, v in ws._overrides_for_run(args.run_id).items():
        if hasattr(config, k):
            setattr(config, k, v)
    if args.crops:                                            # another district: crops dir + route store subdir (as wave_rollout_eval.py)
        config.c3_crops_dir = lambda: Path(args.crops)
        config.c4_routes_dir = lambda: Path(args.crops) / (args.store or "c4_256_val")
    config.sync_in_chans_from_encoding(); data.setup_device()
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)
    device_type = "cuda" if str(data.device).startswith("cuda") else "cpu"
    grid = config.grid_out_size
    if args.split != "val":
        _orig_load = ws.load_c4_route_index
        ws.load_c4_route_index = lambda d, split, dev, max_items=None: _orig_load(d, args.split, dev, max_items=max_items)
    loader, val_routes = ws.build_wave_rollout_loader()
    ckpt = ws._resolve_checkpoint(args.run_id, None)
    doc = read_checkpoint(ckpt, map_location="cpu")
    model = LocalGridViT(LocalGridViTConfig(**(checkpoint_model_config(doc) or dict(meta.get("model") or {})))).to(data.device)
    grid_vit_adapter.load_checkpoint(ckpt, model=model); model.train(False)
    fg = model.config.foreground_class
    crop = int(val_routes.get("crop_px", config.img_size)); s = crop // grid
    store = Path(config.c4_routes_dir()); rix = args.route
    goal_m = float(np.load(store / f"c5_wave_{args.split}.npz")["goal_m"][rix]); n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M)
    rz = np.load(store / f"c4_routes_{args.split}.npz"); ox, oy = int(rz["origin"][rix][0]), int(rz["origin"][rix][1]); mk = rz["markers"][rix]
    bld = (np.load(store.parent / "district_labels_1m.npz")["surface"] == 3)[oy:oy + crop, ox:ox + crop]

    base = {tok: loader.fetch_wave_pass(rix, key)[:4] for tok, key in STATE_KEY.items()}
    pad = _wave_start_disc(loader.route["markers"][rix:rix + 1], grid, crop)[0]
    state = TOK_SEARCH
    frames = [("the map: buildings, start and goal", None), ("start: the start disc on the scratchpad", pad.cpu().numpy())]
    toks = []; draw_in = None
    for i in range(2 * n_goal + 4):
        img_t, _, task_id, cond = base[state]
        if state == TOK_DRAW:
            draw_in = pad
        pad, tok = step(model, img_t, pad, task_id, cond, crop, grid, fg, device_type)
        toks.append(tok)
        frames.append((f"pass {i + 1}: state {NAMES[state]}  ->  says {NAMES.get(tok, '?')}", pad.cpu().numpy()))
        if tok == TOK_DONE or tok not in STATE_KEY:
            break
        state = tok

    cleanup = None
    if args.cleanup_run:                                      # one draw pass of another network on the final drawing
        cdoc_ckpt = ws._resolve_checkpoint(args.cleanup_run, None)
        cdoc = read_checkpoint(cdoc_ckpt, map_location="cpu")
        ccfg = checkpoint_model_config(cdoc)
        if (int(ccfg.get("img_size", crop)), int(ccfg.get("grid_out_size", grid))) != (crop, grid):
            raise SystemExit(f"cleanup network built for {ccfg.get('img_size')} px / {ccfg.get('grid_out_size')} cells, "
                             f"the search runs at {crop} px / {grid} cells")
        cmodel = LocalGridViT(LocalGridViTConfig(**ccfg)).to(data.device)
        grid_vit_adapter.load_checkpoint(cdoc_ckpt, model=cmodel); cmodel.train(False)
        img_t, _, task_id, cond = base[TOK_DRAW]
        if args.cleanup_input == "draw_in" and draw_in is None:
            raise SystemExit("the search never ran a DRAW pass; --cleanup-input draw_in has no input")
        before = pad if args.cleanup_input == "final" else draw_in
        after, ctok = step(cmodel, img_t, before, task_id, cond, crop, grid, cmodel.config.foreground_class, device_type)
        blocked = bld.reshape(grid, s, grid, s).all(axis=(1, 3))
        sc, gc = (int(mk[0][0]) // s, int(mk[0][1]) // s), (int(mk[1][0]) // s, int(mk[1][1]) // s)
        cleanup = {k: route_score(m.cpu().numpy(), sc, gc, blocked, goal_m, float(s))
                   for k, m in (("before", before), ("after", after))}
        cleanup["token"] = NAMES.get(ctok, ctok)
        cleanup["input"] = args.cleanup_input
        frames.append(("cleanup pass (separate network)" if args.cleanup_input == "final" else
                       "cleanup pass (separate network) in place of the last DRAW pass", after.cpu().numpy()))

    S = args.scale; W = crop * S; top = 46
    yy, xx = np.mgrid[0:crop, 0:crop]
    marker = np.zeros((crop, crop), bool)
    for x, y in mk:
        marker |= (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= 4 ** 2
    f_big, f_small = font(20), font(15)
    images = []
    for title, p in frames:
        rgb = np.empty((crop, crop, 3), np.uint8); rgb[:] = BG
        rgb[bld] = MAP
        if p is not None:
            rgb[np.kron(p, np.ones((s, s), bool))] = PAD
        rgb[marker] = MAP
        im = Image.new("RGB", (W, W + top), (255, 255, 255))
        im.paste(Image.fromarray(rgb).resize((W, W), Image.NEAREST), (0, top))
        d = ImageDraw.Draw(im)
        d.text((10, 6), title, fill=(20, 20, 20), font=f_big)
        d.text((10, 28), f"route {rix} · {goal_m:.0f} m walking · window {crop} m · the network's own scratchpad in orange", fill=(110, 110, 110), font=f_small)
        for (x, y), name in zip(mk, ("start", "goal")):
            d.text((int(x) * S + 16, int(y) * S + top - 9), name, fill=MAP, font=f_small)
        images.append(im)
    odir = Path(config.diagnostics_dir) / args.run_id
    if args.tag is not None:
        tag = args.tag
    elif args.crops or args.split != "val":
        tag = "_" + "_".join(x for x in ((Path(args.crops).parts[-3] if args.crops else ""), args.store or "", args.split) if x)
    else:
        tag = ""
    if args.cleanup_run:
        tag += ("_cleanup_" if args.cleanup_input == "final" else "_cleanupdrawin_") + args.cleanup_run
    out = args.out or odir / f"wave_search{tag}_route{rix:03d}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [1500, 1000] + [args.ms] * (len(images) - 3) + [4000]
    if cleanup is not None:
        durations[-2] = 2500                                  # hold the search's final drawing before the cleanup frame
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
    images[-1].save(odir / f"wave_search{tag}_route{rix:03d}_final.png")
    print("tokens:", [NAMES.get(t, t) for t in toks], "| passes", len(toks), "| n_goal", n_goal)
    if cleanup is not None:
        print("cleanup pass:", json.dumps(cleanup))
    if args.save_frames is not None:
        pads_np = np.stack([np.zeros((grid, grid), bool) if p is None else p.astype(bool) for _, p in frames])
        args.save_frames.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_frames, pad=np.packbits(pads_np, axis=-1), grid=grid, crop=crop, route=rix,
                            titles=np.array([t for t, _ in frames]), tokens=np.array(toks, np.int64), run=args.run_id)
        print("wrote", args.save_frames)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
