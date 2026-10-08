#!/usr/bin/env python3
"""Score the cell drawings of the wave search and the one-shot model with the Track A polyline rule (8 Oct 2026).

Why: Qwen is scored by `spatial_data/track_a_export.py score` (endpoints within 6 m of the markers, no segment
crosses a building pixel, length <= 1.10 x the teacher's walking length), the grid models by the cell rule of
`route_score` (wave_rollout_eval.py: connected through lit cells, no chain cell ENTIRELY building, chain length
<= 1.10 x the same walking length). For one table under one rule, each final drawing is turned into a polyline and
handed to the same scorer as Qwen's JSON.

Drawing -> polyline (`drawing_to_polyline`): the shortest 8-neighbour chain start -> goal through the lit cells of the
final drawing (`spatial_data.c5_cleanup.shortest_lit_chain`, the chain the cell rule scores; the 3 x 3 cells around
each marker count as lit, as there), its cell centres in window metres, times the export's px_per_m = image pixels.
No snapping to the markers, no repair: a drawing whose start and goal do not connect is `points: null` and scores as
"unparseable". `simplify` then applies the export's own building-aware Douglas-Peucker (`simplify_route`, the
tolerance of the export, 2 m) as the teacher polylines were simplified: chords that touch a building are split, so it
never adds a crossing and can only lower the length. It can remove one: where the chain's cell centres touch a
building pixel (see the artifact below) but a chord between two other chain points does not. Both versions are
written and scored; the simplified one is `points` (headline), the raw chain `points_raw`.

Known artifact: with 2 m cells a cell centre sits on a pixel corner, and the scorer's floor() puts it in the cell's
lower-right pixel; a chain cell that is part building can therefore count as a crossing even where a line could have
passed through its free pixels.

Usage (from harness/):
  PYTHONPATH=. python diagnostics/drawing_to_polyline.py RUN_ID --kind wave|oneshot
      --export ../data/track_a/de_pijp_val_plain [--crops DIR --store NAME] [--split val] [--n 500] [--pads FILE] [--rerun]
The drawings come from the eval scripts themselves (wave_rollout_eval.py --mode free / oneshot_route_eval.py, run as
subprocesses with --save-pads; they set the config, the marker radius and the store exactly as for the cell-rule
numbers, and write their own cell-rule JSON with the tag `{tag}_pl` next to the existing ones). An existing pads file
is reused unless --rerun. Writes <export>/preds/<run_id>_<kind>.jsonl and <export>/preds/<run_id>_<kind>_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_data import track_a_export as TA          # noqa: E402
from spatial_data.c5_cleanup import shortest_lit_chain  # noqa: E402

DIAG = ROOT / "runs" / "diagnostics"
HARNESS = ROOT / "harness"


def drawing_to_polyline(mask: np.ndarray, start_c, goal_c, *, cell_m: float = 2.0, px_per_m: float = 2.0):
    """mask (g,g) bool final drawing, start_c / goal_c = (col, row) cells of the markers. -> [[x_px, y_px], ...]
    start -> goal through the cell centres of the shortest lit chain, or None when start and goal do not connect."""
    chain = shortest_lit_chain(np.asarray(mask, bool), start_c, goal_c, cell_m)
    if chain is None:
        return None
    ys, xs, _ = chain                                         # goal first
    pts_m = np.stack([(xs[::-1] + 0.5) * cell_m, (ys[::-1] + 0.5) * cell_m], 1)
    return TA._pts_json(TA.window_m_to_image_px(pts_m, px_per_m))


def simplify(points_px, record: dict, tol_m: float, building_mask: np.ndarray | None = None):
    """The export's building-aware Douglas-Peucker on a polyline in image pixels (None stays None)."""
    if points_px is None or len(points_px) <= 2:
        return points_px
    ppm = float(record["px_per_m"])
    mask = TA.load_building_mask(record) if building_mask is None else building_mask
    poly, _ = TA.simplify_route(TA.image_px_to_window_m(points_px, ppm), mask, tol_m)
    return TA._pts_json(TA.window_m_to_image_px(poly, ppm))


def eval_tag(crops: str | None, store: str | None, split: str) -> str:
    """The output tag of the eval scripts: empty for the run's own store and split, else _{district}_{store}_{split}."""
    if crops or split != "val":
        return "_" + "_".join(x for x in ((Path(crops).parts[-3] if crops else ""), store or "", split) if x)
    return ""


def make_pads(run_id: str, kind: str, pads: Path, n: int, crops, store, split: str, tag: str) -> None:
    script = {"wave": "diagnostics/wave_rollout_eval.py", "oneshot": "diagnostics/oneshot_route_eval.py"}[kind]
    cmd = [sys.executable, script, run_id, "--n", str(n), "--split", split, "--save-pads", str(pads), "--tag", tag + "_pl"]
    if kind == "wave":
        cmd += ["--mode", "free"]
    if crops:
        cmd += ["--crops", str(crops), "--store", store or "c4_256_val"]
    env = dict(os.environ, PYTHONPATH=str(HARNESS))
    print("running", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HARNESS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0 or not pads.is_file():
        print(r.stdout[-3000:])
        raise SystemExit(f"{script} failed ({r.returncode})")


def load_pads(path: Path) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    g = int(meta["grid"])
    pad = np.unpackbits(z["pad"], axis=-1)[..., :g].astype(bool)
    return dict(meta=meta, route=z["route"], pad=pad, origin=z["origin"], markers=z["markers"])


def check_alignment(routes: list[dict], P: dict) -> dict:
    """The export's records and the drawings must describe the same routes: same route index -> same window origin
    and same markers (store metres x px_per_m = the record's image pixels), and the same store."""
    by_route = {int(r): i for i, r in enumerate(P["route"])}
    bad, missing = [], []
    store_ok = str(P["meta"]["store"]).rstrip("/").endswith(routes[0]["store"].rstrip("/"))
    for rec in routes:
        i = by_route.get(rec["route"])
        if i is None:
            missing.append(rec["route"]); continue
        ppm = float(rec["px_per_m"])
        mk = np.asarray(P["markers"][i], np.float64)
        ok = (list(map(int, P["origin"][i])) == list(map(int, rec["origin"]))
              and np.allclose(mk[0] * ppm, rec["start"]) and np.allclose(mk[1] * ppm, rec["goal"]))
        if not ok:
            bad.append(rec["route"])
    return dict(store_matches=bool(store_ok), n_records=len(routes), n_drawings=int(len(P["route"])),
                n_misaligned=len(bad), misaligned=bad[:20], n_records_without_drawing=len(missing),
                records_without_drawing=missing[:20])


def cell_rule_per_route(eval_json: Path) -> dict[int, float]:
    d = json.loads(eval_json.read_text())
    rows = d.get("free_per_route") or d.get("oneshot_per_route") or []
    return {int(r["route"]): float(r["solved"]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--kind", required=True, choices=("wave", "oneshot"))
    ap.add_argument("--export", required=True, help="Track A export dir with routes.jsonl and export.json")
    ap.add_argument("--crops", default=None); ap.add_argument("--store", default=None)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pads", type=Path, default=None, help="pads npz (default runs/diagnostics/RUN_ID/pads_{kind}{tag}.npz)")
    ap.add_argument("--rerun", action="store_true", help="recompute the drawings even if the pads file exists")
    a = ap.parse_args()

    export = Path(os.path.abspath(a.export)) if Path(a.export).exists() else TA.resolve(a.export)
    routes = TA.load_routes(export / "routes.jsonl")[: a.n]
    tol = float(json.loads((export / "export.json").read_text())["args"]["simplify_m"])
    tag = eval_tag(a.crops, a.store, a.split)
    pads_path = a.pads or DIAG / a.run_id / f"pads_{a.kind}{tag}.npz"
    if a.rerun or not pads_path.is_file():
        make_pads(a.run_id, a.kind, pads_path, a.n, a.crops, a.store, a.split, tag)
    P = load_pads(pads_path)
    align = check_alignment(routes, P)
    print("alignment", json.dumps(align))
    if align["n_misaligned"] or not align["store_matches"]:
        raise SystemExit("drawings and export do not describe the same routes")

    cell_m = float(P["meta"]["cell_m"])
    by_route = {int(r): i for i, r in enumerate(P["route"])}
    out_dir = export / "preds"; out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / f"{a.run_id}_{a.kind}.jsonl"
    preds, preds_raw, npts, npts_raw = {}, {}, [], []
    with open(pred_path, "w") as f:
        for rec in routes:
            i = by_route.get(rec["route"])
            raw = simp = None
            if i is not None:
                mk = P["markers"][i]
                start_c = (int(mk[0][0] // cell_m), int(mk[0][1] // cell_m))
                goal_c = (int(mk[1][0] // cell_m), int(mk[1][1] // cell_m))
                raw = drawing_to_polyline(P["pad"][i], start_c, goal_c, cell_m=cell_m, px_per_m=float(rec["px_per_m"]))
                simp = simplify(raw, rec, tol)
            preds[rec["route"]], preds_raw[rec["route"]] = simp, raw
            if raw is not None:
                npts.append(len(simp)); npts_raw.append(len(raw))
            f.write(json.dumps(dict(route=rec["route"], points=simp, points_raw=raw, connected=raw is not None,
                                    lit_cells=int(P["pad"][i].sum()) if i is not None else None)) + "\n")
    rep = TA.score(routes, preds)
    rep_raw = TA.score(routes, preds_raw)
    print("simplified:", TA.summary_line(rep))
    print("raw chain: ", TA.summary_line(rep_raw))

    # the cell rule on exactly these drawings (the eval JSON written together with the pads), route by route
    ej = Path(P["meta"]["eval_json"])
    cell = cell_rule_per_route(ej) if ej.is_file() else {}
    cross = {}
    for x in rep["routes"]:
        c = cell.get(x["route"])
        if c is None:
            continue
        key = f"cell_{'solved' if c else 'failed'}__polyline_{'solved' if x['solved'] else x['reason']}"
        cross[key] = cross.get(key, 0) + 1
    summary = dict(
        run=a.run_id, kind=a.kind, export=TA._rel(export), pads=str(pads_path), pads_meta=P["meta"], alignment=align,
        n=len(routes), n_connected=int(sum(v is not None for v in preds_raw.values())),
        n_not_connected=int(sum(v is None for v in preds_raw.values())),
        points_simplified=dict(median=float(np.median(npts)), mean=round(float(np.mean(npts)), 2)) if npts else None,
        points_raw=dict(median=float(np.median(npts_raw)), mean=round(float(np.mean(npts_raw)), 2)) if npts_raw else None,
        simplify_m=tol,
        cell_rule=dict(eval_json=str(ej), n=len(cell),
                       solved_share=round(float(np.mean(list(cell.values()))), 4) if cell else None),
        cell_vs_polyline=dict(sorted(cross.items())),
        polyline_simplified={k: v for k, v in rep.items() if k != "routes"},
        polyline_raw={k: v for k, v in rep_raw.items() if k != "routes"},
        routes=[dict(route=x["route"], n_goal=x["n_goal"], cell_solved=cell.get(x["route"]), solved=x["solved"],
                     reason=x["reason"], solved_raw=y["solved"], reason_raw=y["reason"], length_ratio=x["length_ratio"],
                     length_ratio_raw=y["length_ratio"])
                for x, y in zip(rep["routes"], rep_raw["routes"])],
    )
    rp = out_dir / f"{a.run_id}_{a.kind}_report.json"
    rp.write_text(json.dumps(summary, indent=1))
    print("cell rule", summary["cell_rule"]["solved_share"], "| not connected", summary["n_not_connected"], "|",
          json.dumps(summary["cell_vs_polyline"]))
    print("wrote", pred_path, rp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
