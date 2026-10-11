#!/usr/bin/env python3
"""Two numbers of COMPARISON_TABLE_2026-10-08.md §4, recomputed from the saved final drawings (11 Oct 2026, CPU, no model).

(a) Building contact of the chain: the share of chain cells (the shortest lit chain start -> goal, the cells
    drawing_to_polyline.py turns into the raw polyline) that contain at least one building pixel of the 1 m raster;
    pooled over all chain cells of the connected routes, and as the mean of the per-route shares.
(b) Conversion sensitivity: a 2 m cell centre sits on a pixel corner and the scorer's floor() puts it in the cell's
    lower-right pixel. Every raw chain point whose scored pixel is building is moved to the centre of the first free
    pixel (row-major) of its own cell, if the cell has one, and the raw chain is re-scored by the polyline rule.
    One fixed choice of pixel, not an optimum over choices. Reported: raw-chain solved before / after, and how many
    raw "crosses a building" failures become solved.

Reads runs/diagnostics/<run>/pads_<kind><tag>.npz and data/track_a/<set>_val_plain/routes.jsonl (500 val routes).
Usage (repo root): PYTHONPATH=.:harness python harness/diagnostics/chain_building_share.py OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "harness"); sys.path.insert(0, ".")
from spatial_data import track_a_export as TA
from spatial_data.c5_cleanup import shortest_lit_chain
from diagnostics.drawing_to_polyline import load_pads

RUNS = [("c5_os_bld_det_L4_64k_g128_store100k", "oneshot"),
        ("c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k", "wave"),
        ("c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g128_store100k", "wave")]
SETS = {"de_pijp": ("data/track_a/de_pijp_val_plain", ""),
        "zuid_centrum": ("data/track_a/zuid_centrum_val_plain", "_zuid_centrum_c4_256_val_val"),
        "maze": ("data/track_a/maze_v2_houses_val_plain", "_maze_v2_houses_c4_256_val_val")}
M = TA.MASK_MARGIN_M
out = {}
for sname, (exp, tag) in SETS.items():
    routes = TA.load_routes(Path(exp) / "routes.jsonl")[:500]
    for run, kind in RUNS:
        P = load_pads(Path("runs/diagnostics") / run / f"pads_{kind}{tag}.npz")
        cm = float(P["meta"]["cell_m"]); by = {int(r): i for i, r in enumerate(P["route"])}
        n_cells = n_touch = 0; per_route = []
        raw, moved = {}, {}
        for rec in routes:
            i = by[rec["route"]]; mk = P["markers"][i]; ppm = float(rec["px_per_m"])
            s = (int(mk[0][0] // cm), int(mk[0][1] // cm)); g = (int(mk[1][0] // cm), int(mk[1][1] // cm))
            ch = shortest_lit_chain(P["pad"][i], s, g, cm)
            if ch is None:
                raw[rec["route"]] = moved[rec["route"]] = None; continue
            ys, xs, _ = ch; ys, xs = ys[::-1], xs[::-1]
            bm = TA.load_building_mask(rec); k = int(round(cm))
            pts, pts2, t = [], [], 0
            for y, x in zip(ys, xs):
                blk = bm[k * y + M:k * y + M + k, k * x + M:k * x + M + k]
                hit = bool(blk.any()); t += hit
                cx, cy = (x + 0.5) * cm, (y + 0.5) * cm
                pts.append((cx, cy))
                if bm[int(np.floor(cy)) + M, int(np.floor(cx)) + M] and not blk.all():
                    fy, fx = np.argwhere(~blk)[0]
                    pts2.append((k * x + fx + 0.5, k * y + fy + 0.5))
                else:
                    pts2.append((cx, cy))
            n_cells += len(ys); n_touch += t; per_route.append(t / len(ys))
            raw[rec["route"]] = TA._pts_json(TA.window_m_to_image_px(np.array(pts), ppm))
            moved[rec["route"]] = TA._pts_json(TA.window_m_to_image_px(np.array(pts2), ppm))
        r0 = TA.score(routes, raw); r1 = TA.score(routes, moved)
        rescued = sum(1 for a, b in zip(r0["routes"], r1["routes"]) if a["reason"] == "crosses a building" and b["solved"])
        out[f"{sname}/{run}"] = dict(chain_cells=n_cells, touching_building=n_touch,
                                     share_pooled=round(n_touch / n_cells, 4) if n_cells else None,
                                     share_mean_per_route=round(float(np.mean(per_route)), 4) if per_route else None,
                                     raw_solved=r0["solved"], raw_crosses=r0["failure_counts"]["crosses a building"],
                                     moved_solved=r1["solved"], rescued_from_crosses=rescued,
                                     raw_solved_share=r0["solved_share"], moved_solved_share=r1["solved_share"])
        print(sname, run[:45], out[f"{sname}/{run}"], flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else "chain_building.json").write_text(json.dumps(out, indent=1))
