#!/usr/bin/env python3
"""Showcase pictures: one-shot path finding against the free-running wave search on the same routes (5 Oct 2026).

One PNG per route with four panels side by side, in the style of wave_rollout_gif.py (ground #F5F3EE, buildings and
markers #343A46, drawing #E86A17):
  1. the map with the two markers
  2. the teacher's path: the search's final target (wave_draw), the clearance-priced walking path whose length
     goal_m is what the success rule measures against
  3. the one-shot model's output: one forward pass, p >= 0.5 (as diagnostics/oneshot_route_eval.py)
  4. the search's final drawing, free-running from the start disc until DONE (the stepping of wave_rollout_eval.py /
     wave_rollout_gif.py; nothing from the teacher after the start)
Panels 3 and 4 are titled solved / not solved by route_score of wave_rollout_eval.py and their lit-cell count. Plus one
contact sheet of all routes.

Optional fifth panel (8 Oct): `--polyline-pred FILE --polyline-export DIR [--polyline-label NAME]` adds a polyline
prediction in the Track A format (image pixels, e.g. a Qwen LoRA's pred_val_v1_n500.jsonl) for the routes of that
export's store and split that have a prediction, titled by the polyline rule (spatial_data/track_a_export.py score).
Those routes are written as ..._route{NNN}_polyline.png (the four-panel picture is written as well).

Each model is run with its own training config (meta.json + ledger overrides, marker radius, encoding, window and
grid), so a one-shot model with 64 cells (4 m) can sit next to a search with 128 cells (2 m); the global config is
restored between the two.

Usage (from harness/): PYTHONPATH=. python diagnostics/showcase_compare.py --oneshot RUN_ID [--oneshot-512 RUN_ID]
       [--only all|256|512] [--search RUN_ID] [--route CROPS STORE SPLIT ROUTE ...] [--out DIR]
Without --route the built-in showcase set below is drawn (256 m and 512 m routes, each with its search checkpoint);
512 m routes are skipped when no --oneshot-512 is given. --search replaces the search checkpoint of every route.
Writes runs/diagnostics/ONESHOT_RUN/showcase_vs_search/{district}_{store}_{split}_route{NNN}.png and contact_sheet.png."""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import config
import data
from diagnostics import oneshot_route_eval as ore
from diagnostics import wave_search_rollout as ws
from diagnostics.wave_rollout_eval import STATE_KEY, route_score, step
from diagnostics.wave_rollout_gif import BG, MAP, PAD, font
from spatial_data.c4_tasks import TOK_DONE, TOK_SEARCH, WAVE_HORIZON_M
from spatial_data.dataset_c4_gpu import _wave_start_disc
from spatial_data.wave_search_schedule import n_goal_from_m

_DATA = Path(config.REPO_ROOT) / "data"
DP = str(_DATA / "amsterdam/de_pijp/crops/v4")
ZC = str(_DATA / "amsterdam/zuid_centrum/crops/v4ns")
MZ = str(_DATA / "synthetic/maze_v2_houses/crops/v4ns")
SEARCH = {
    "256": "c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k",
    "512map": "c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512",
    "512maze": "c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512",
}
# (crops, store, split, route, search key); all val
SHOWCASE = [
    (DP, "c4_100k", "val", 333, "256"),
    (ZC, "c4_256_val", "val", 342, "256"), (ZC, "c4_256_val", "val", 442, "256"),
    (MZ, "c4_256_val", "val", 195, "256"), (MZ, "c4_256_val", "val", 271, "256"),
    (ZC, "c4_512_20k", "val", 176, "512map"), (ZC, "c4_512_20k", "val", 54, "512map"), (ZC, "c4_512_20k", "val", 392, "512map"),
    (MZ, "c4_512_val", "val", 8, "512maze"), (MZ, "c4_512_val", "val", 102, "512maze"),
]
PLACE = {"de_pijp": "De Pijp", "zuid_centrum": "Zuid + Centrum", "maze_v2_houses": "house maze"}

# the config as imported, restored before each model is set up (configure_for_run changes module globals)
_CFG0 = {k: v for k, v in vars(config).items() if not k.startswith("__") and not isinstance(v, types.ModuleType)}
_LOAD0 = ws.load_c4_route_index


def _reset_config() -> None:
    for k, v in _CFG0.items():
        setattr(config, k, v)
    ws.load_c4_route_index = _LOAD0


def _route_inputs(store: Path, split: str, rix: int, crop: int, s: int):
    """Building pixels of the window, markers, start / goal cells, walking length: as wave_rollout_eval.py reads them."""
    goal_m = float(np.load(store / f"c5_wave_{split}.npz")["goal_m"][rix])
    rz = np.load(store / f"c4_routes_{split}.npz")
    ox, oy = int(rz["origin"][rix][0]), int(rz["origin"][rix][1]); mk = rz["markers"][rix]
    bld = (np.load(store.parent / "district_labels_1m.npz")["surface"] == 3)[oy:oy + crop, ox:ox + crop]
    return goal_m, mk, bld


def _score(mask: np.ndarray, mk, bld: np.ndarray, goal_m: float, crop: int) -> dict:
    g = mask.shape[0]; s = crop // g
    blocked = bld.reshape(g, s, g, s).all(axis=(1, 3))
    start_c = (int(mk[0][0]) // s, int(mk[0][1]) // s); goal_c = (int(mk[1][0]) // s, int(mk[1][1]) // s)
    return route_score(mask, start_c, goal_c, blocked, goal_m, float(s))


def run_search(run_id: str, crops: str, store: str, split: str, routes: list[int], threads: int,
               bf16_cpu: bool = True) -> dict:
    """Free-running search per route, as wave_rollout_eval.py --mode free: start disc, state SEARCH, the network's
    own token picks the next state, stop on DONE / a non-state token / 2 n + 4 passes."""
    _reset_config()
    st = ore.configure_for_run(run_id, crops, store, threads=threads)
    if split != "val":
        ws.load_c4_route_index = lambda d, sp, dev, max_items=None: _LOAD0(d, split, dev, max_items=max_items)
    loader, val_routes = ws.build_wave_rollout_loader()
    model, _, _ = ore.load_model(run_id, None, st["meta"], st["eval_img"], st["eval_grid"])
    if bf16_cpu and not str(data.device).startswith("cuda"):
        data.dtype = torch.bfloat16       # step() autocasts to data.dtype: bf16 on the CPU too, as the GPU evals ran
    fg = model.config.foreground_class
    device_type = "cuda" if str(data.device).startswith("cuda") else "cpu"
    crop = int(val_routes.get("crop_px", config.img_size)); grid = config.grid_out_size
    out = {}
    for rix in routes:
        goal_m, mk, bld = _route_inputs(Path(config.c4_routes_dir()), split, rix, crop, crop // grid)
        n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M)
        base, teacher = {}, None
        for tok_state, key in STATE_KEY.items():
            img_t, tgt_t, task_id, cond, _ = loader.fetch_wave_pass(rix, key)
            base[tok_state] = (img_t, task_id, cond)
            if key == "wave_draw":
                teacher = ws._teacher_mask_to_grid(tgt_t, grid, crop).cpu().numpy()
        pad = _wave_start_disc(loader.route["markers"][rix:rix + 1], grid, crop)[0]; state = TOK_SEARCH
        passes, stop = 0, "budget"
        for _ in range(2 * n_goal + 4):
            img_t, task_id, cond = base[state]
            pad, tok = step(model, img_t, pad, task_id, cond, crop, grid, fg, device_type)
            passes += 1
            if tok == TOK_DONE:
                stop = "done"; break
            if tok not in STATE_KEY:
                stop = "bad_token"; break
            state = tok
        p = pad.cpu().numpy()
        out[rix] = dict(mask=p, teacher=teacher, score=_score(p, mk, bld, goal_m, crop), passes=passes, stop=stop,
                        goal_m=goal_m, n_goal=n_goal, markers=mk, bld=bld, crop=crop)
    return out


def run_oneshot(run_id: str, crops: str, store: str, split: str, routes: list[int], threads: int) -> dict:
    _reset_config()
    st = ore.configure_for_run(run_id, crops, store, threads=threads)
    model, _, _ = ore.load_model(run_id, None, st["meta"], st["eval_img"], st["eval_grid"])
    loader, rt = ore.build_task_loader("detour", split, 8)
    crop = int(rt.get("crop_px", config.img_size))
    pred, _ = ore.oneshot_masks(model, loader, routes, 8)
    out = {}
    for i, rix in enumerate(routes):
        goal_m, mk, bld = _route_inputs(Path(config.c4_routes_dir()), split, rix, crop, crop // pred.shape[-1])
        out[rix] = dict(mask=pred[i], score=_score(pred[i], mk, bld, goal_m, crop))
    return out


def _panel(bld: np.ndarray, mk, mask: np.ndarray | None, scale: int) -> Image.Image:
    crop = bld.shape[0]
    rgb = np.empty((crop, crop, 3), np.uint8); rgb[:] = BG
    rgb[bld] = MAP
    if mask is not None:
        s = crop // mask.shape[0]
        rgb[np.kron(mask, np.ones((s, s), bool)).astype(bool)] = PAD
    yy, xx = np.mgrid[0:crop, 0:crop]
    r = 4 if crop <= 256 else 6
    for x, y in mk:
        rgb[(xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= r ** 2] = MAP
    return Image.fromarray(rgb).resize((crop * scale, crop * scale), Image.NEAREST)


def _verdict(sc: dict) -> str:
    return "solved" if sc["success"] == 1.0 else ("not solved (no connection)" if sc["connected"] == 0.0 else
                                                   f"not solved (length x{sc['length_ratio']:.2f})" if sc["length_ratio"] > 1.10 else
                                                   "not solved (crosses a building)")


def load_polyline_preds(pred_path: Path, export_dir: Path) -> dict:
    """route -> (record, points) for a Track A prediction file, with the polyline-rule verdict of each route."""
    import sys
    sys.path.insert(0, str(config.REPO_ROOT))
    from spatial_data import track_a_export as TA
    recs = {r["route"]: r for r in TA.load_routes(export_dir / "routes.jsonl")}
    preds = TA.load_predictions(pred_path)
    out = {}
    for rix, pts in preds.items():
        if rix in recs:
            res = TA.score_polyline(pts, recs[rix], TA.load_building_mask(recs[rix]))
            out[rix] = dict(record=recs[rix], points=pts, score=res)
    return out


def _poly_verdict(res: dict) -> str:
    if res["solved"]:
        return f"solved (length x{res['length_ratio']:.2f})"
    return f"not solved ({res['reason']}" + (f", x{res['length_ratio']:.2f})" if res["reason"] == "too long" else ")")


def route_figure(key: tuple, srch: dict, one: dict, oneshot_id: str, search_id: str, poly: dict | None = None,
                 poly_label: str = "") -> Image.Image:
    crops, store, split, rix = key
    crop = srch["crop"]; scale = max(1, 512 // crop); P = crop * scale
    gap, top, head = 14, 58, 52
    n_pan = 5 if poly is not None else 4
    W = n_pan * P + (n_pan + 1) * gap; H = head + top + P + gap
    im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    f_big, f_mid, f_small = font(22), font(17), font(14)
    place = PLACE.get(Path(crops).parts[-3], Path(crops).parts[-3])
    d.text((gap, 8), f"{place} · {store} {split} · route {rix} · {srch['goal_m']:.0f} m walking · {srch['n_goal']} fronts · window {crop} m",
           fill=(20, 20, 20), font=f_big)
    d.text((gap, 33), f"one-shot {oneshot_id}   |   search {search_id}", fill=(110, 110, 110), font=f_small)
    g1 = one["mask"].shape[0]; g2 = srch["mask"].shape[0]
    panels = [
        ("the map: buildings, start and goal", "", None),
        ("the teacher's path", f"walking path, {srch['goal_m']:.0f} m (the length rule's reference)", srch["teacher"]),
        (f"one-shot (1 pass): {_verdict(one['score'])}", f"{one['score']['lit_cells']:.0f} lit cells of {crop // g1} m", one["mask"]),
        (f"search ({srch['passes']} passes, {srch['stop']}): {_verdict(srch['score'])}",
         f"{srch['score']['lit_cells']:.0f} lit cells of {crop // g2} m", srch["mask"]),
    ]
    if poly is not None:
        n_pts = len(poly["points"]) if isinstance(poly["points"], list) else 0
        panels.append((f"{poly_label}: {_poly_verdict(poly['score'])}", f"polyline of {n_pts} points, polyline rule", None))
    for i, (t1, t2, m) in enumerate(panels):
        x = gap + i * (P + gap)
        d.text((x, head + 4), t1, fill=(20, 20, 20), font=f_mid)
        d.text((x, head + 28), t2, fill=(110, 110, 110), font=f_small)
        im.paste(_panel(srch["bld"], srch["markers"], m, scale), (x, head + top))
        if poly is not None and i == 4 and isinstance(poly["points"], list) and len(poly["points"]) >= 2:
            ppm = float(poly["record"]["px_per_m"])            # image px -> window metres -> panel px
            xy = [(x + float(px) / ppm * scale, head + top + float(py) / ppm * scale) for px, py in poly["points"]]
            d.line(xy, fill=PAD, width=4, joint="curve")
            for q in xy:
                d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=PAD)
        for (mx, my), name in zip(srch["markers"], ("start", "goal")):
            d.text((x + int(mx) * scale + 10, head + top + int(my) * scale - 8), name, fill=MAP, font=f_small)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oneshot", required=True, help="one-shot run id for the 256 m routes")
    ap.add_argument("--oneshot-512", default=None, help="one-shot run id for the 512 m routes (skipped if not given)")
    ap.add_argument("--search", default=None, help="search run id for every route (default: per route, see SEARCH)")
    ap.add_argument("--only", default="all", choices=("all", "256", "512"))
    ap.add_argument("--route", nargs=4, action="append", metavar=("CROPS", "STORE", "SPLIT", "ROUTE"),
                    help="a route instead of the built-in set (repeatable); the search defaults to the 256 m one")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--fp32-cpu", action="store_true", help="run the search in float32 on the CPU (default: bf16 autocast, as on the GPU)")
    ap.add_argument("--polyline-pred", type=Path, default=None, help="Track A prediction jsonl for a fifth panel")
    ap.add_argument("--polyline-export", type=Path, default=None, help="the Track A export dir of those predictions")
    ap.add_argument("--polyline-label", default="Qwen", help="title of the fifth panel")
    args = ap.parse_args()
    poly, poly_key = {}, None
    if args.polyline_pred is not None:
        poly = load_polyline_preds(args.polyline_pred, args.polyline_export)
        meta = json.loads((args.polyline_export / "export.json").read_text())["args"]
        poly_key = (Path(meta["store"]).parts[-4], Path(meta["store"]).name, meta["split"])   # district, store, split

    if args.route:
        todo = [(c, s, sp, int(r), "256") for c, s, sp, r in args.route]
    else:
        todo = [t for t in SHOWCASE if args.only == "all" or (args.only == "256") == (t[4] == "256")]
    groups: dict[tuple, list[int]] = {}
    for crops, store, split, rix, sk in todo:
        one_id = args.oneshot if sk == "256" else args.oneshot_512
        if one_id is None:
            print(f"skip {store} route {rix}: no --oneshot-512 given"); continue
        groups.setdefault((crops, store, split, args.search or SEARCH[sk], one_id), []).append(rix)

    odir = args.out or Path(config.diagnostics_dir) / args.oneshot / "showcase_vs_search"
    odir.mkdir(parents=True, exist_ok=True)
    figs = []
    for (crops, store, split, search_id, one_id), routes in groups.items():
        srch = run_search(search_id, crops, store, split, routes, args.threads, bf16_cpu=not args.fp32_cpu)
        one = run_oneshot(one_id, crops, store, split, routes, args.threads)
        for rix in routes:
            im = route_figure((crops, store, split, rix), srch[rix], one[rix], one_id, search_id)
            path = odir / f"{Path(crops).parts[-3]}_{store}_{split}_route{rix:03d}.png"
            im.save(path); figs.append(im)
            if poly_key == (Path(crops).parts[-3], store, split) and rix in poly:
                imp = route_figure((crops, store, split, rix), srch[rix], one[rix], one_id, search_id, poly[rix], args.polyline_label)
                pp = odir / f"{Path(crops).parts[-3]}_{store}_{split}_route{rix:03d}_polyline.png"
                imp.save(pp)
                print(f"{pp.name}: {args.polyline_label} {_poly_verdict(poly[rix]['score'])}")
            print(f"{path.name}: one-shot {_verdict(one[rix]['score'])} ({one[rix]['score']['lit_cells']:.0f} cells) | "
                  f"search {_verdict(srch[rix]['score'])} ({srch[rix]['score']['lit_cells']:.0f} cells, {srch[rix]['passes']} passes)")
    _reset_config()
    if figs:                                                   # contact sheet: every route row, scaled to one width
        w = 1400
        rows = [f.resize((w, int(f.height * w / f.width)), Image.LANCZOS) for f in figs]
        sheet = Image.new("RGB", (w, sum(r.height for r in rows)), (255, 255, 255))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height
        sheet.save(odir / "contact_sheet.png")
        print("wrote", odir / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
