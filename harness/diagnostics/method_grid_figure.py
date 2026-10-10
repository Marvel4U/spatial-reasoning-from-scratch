#!/usr/bin/env python3
"""The same De Pijp val routes solved by five methods, side by side (9 Oct 2026, revised 10 Oct; README figure).

Columns = routes, rows = methods, in this order: Qwen3-VL-4B zero-shot (run two-pass: a marker-probe prompt locates
the two discs, then the route prompt is anchored on those coordinates); Qwen3-VL-2B + LoRA on 19.3k routes; the
one-shot network; the wave search; the wave search + cleanup pass. (10 Oct: the straight-line row was dropped from the
picture; the straight line is still loaded, as the auto rule uses it.)
Every verdict is the polyline rule of `spatial_data/track_a_export.py score` (the one-rule comparison,
docs/reports/COMPARISON_TABLE_2026-10-08.md), recomputed here from the prediction files:

  straight      data/track_a/de_pijp_val_plain/baselines/straight.jsonl   (route choice only, not drawn)
  Qwen 4B       runs/qwen_vanilla/pred_4b_two-pass_n200.jsonl            (first 200 val routes only)
  Qwen LoRA     runs/qwen_sft/lora20000_v1/pred_val_v1_n500.jsonl
  grid models   data/track_a/de_pijp_val_plain/preds/<run>_{oneshot,wave}.jsonl (the converted polylines, verdicts)
                runs/diagnostics/<run>/pads_{oneshot,wave}.npz           (the final cell drawings, what is drawn)

Each panel: the map (1 m building raster re-cut from the district, as the scorer does; ground, buildings, start and
goal discs with labels) and the method's output in orange: a polyline 3 px wide with small vertices, or the final cell
drawing as filled 2 m cells (for the wave search the last frame of the free-running search). Colours as
wave_rollout_gif.py / showcase_compare.py (ground #F5F3EE, buildings and markers #343A46, drawing #E86A17).

Subtitles: the verdict ("solved · length 0.96 × teacher" or the reason for the failure) and, for the grid methods,
the lit cells of the final drawing. The length reference is the teacher's clearance-priced walking path (it keeps
away from walls), so a legal line closer to the buildings can come out shorter than 1.0 x; the cap is 1.10 x.

Route choice: `--routes A B C`, or `--auto`, which picks from the first 200 val routes (so the 4B row exists), one
column per group, in this order:
(1) both solved (Qwen LoRA and the wave search): the longest such route, by fronts (n_goal), then walking length
    (teacher_m);
(2) Qwen LoRA solved, wave search not; (3) wave search solved, Qwen LoRA not. Within (2) and (3) it prefers routes
    with 4+ fronts where the straight line fails (a visible detour), for (3) a Qwen failure that crosses a building,
    for (2) a search failure by a gap (not connected); ties broken by the largest detour (teacher_m / straight_m),
    routes from distinct windows.
On the 10 Oct prediction files this gives routes 188 for (2) and 192 for (3), the same as the 9 Oct figure.
Column headers name the group from the two verdicts (both solved / only Qwen solved / only the search solved /
both failed).

Usage (from harness/, CPU only, a few seconds):
  PYTHONPATH=. python diagnostics/method_grid_figure.py --auto [--out PNG ...]
  PYTHONPATH=. python diagnostics/method_grid_figure.py --routes 12 34 56 [--out PNG ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_data import track_a_export as TA  # noqa: E402

BG, MAP, PAD = (247, 245, 240), (52, 58, 70), (232, 106, 23)   # = wave_rollout_gif.py (not imported: it loads torch)

EXPORT = ROOT / "data/track_a/de_pijp_val_plain"
ONESHOT = "c5_os_bld_det_L4_64k_g128_store100k"
WAVE = "c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k"
CLEANUP = "c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g128_store100k"

# (key, panel title, kind, prediction file, pads file)
STRAIGHT = ("straight", "Straight line between the markers", "poly", EXPORT / "baselines/straight.jsonl", None)
METHODS = [
    ("qwen4b", "Qwen3-VL-4B, zero-shot", "poly", ROOT / "runs/qwen_vanilla/pred_4b_two-pass_n200.jsonl", None),
    ("lora", "Qwen3-VL-2B + LoRA, 19.3k routes", "poly", ROOT / "runs/qwen_sft/lora20000_v1/pred_val_v1_n500.jsonl", None),
    ("oneshot", "One-shot network (1 pass)", "grid", EXPORT / f"preds/{ONESHOT}_oneshot.jsonl",
     ROOT / f"runs/diagnostics/{ONESHOT}/pads_oneshot.npz"),
    ("wave", "Wave search", "grid", EXPORT / f"preds/{WAVE}_wave.jsonl", ROOT / f"runs/diagnostics/{WAVE}/pads_wave.npz"),
    ("cleanup", "Wave search + cleanup pass", "grid", EXPORT / f"preds/{CLEANUP}_wave.jsonl",
     ROOT / f"runs/diagnostics/{CLEANUP}/pads_wave.npz"),
]


def font(size: int, bold: bool = False):
    names = ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else \
            ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_pads(path: Path) -> dict[int, np.ndarray]:
    z = np.load(path)
    g = int(json.loads(str(z["meta"]))["grid"])
    pad = np.unpackbits(z["pad"], axis=-1)[..., :g].astype(bool)
    return {int(r): pad[i] for i, r in enumerate(z["route"])}


def verdict_text(res: dict) -> str:
    if res["solved"]:
        return f"solved · length {res['length_ratio']:.2f} × teacher"
    r = res["reason"]
    if r == "unparseable":
        return "not solved: start and goal not connected"
    if r == "too long":
        return f"not solved: too long ({res['length_ratio']:.2f} × teacher)"
    return f"not solved: {r}"


def group_name(q: dict | None, w: dict | None) -> str:
    """Column header from the Qwen LoRA and wave search verdicts."""
    qs, wsv = bool(q and q["solved"]), bool(w and w["solved"])
    return {(True, True): "both solved", (True, False): "only Qwen solved",
            (False, True): "only the search solved", (False, False): "both failed"}[(qs, wsv)]


def load_all(n: int = 500):
    """routes (first n of the export), per method: route -> points, route -> verdict, and the pads of grid methods."""
    routes = TA.load_routes(EXPORT / "routes.jsonl")[:n]
    masks = {r["route"]: TA.load_building_mask(r) for r in routes}
    preds, verd, pads = {}, {}, {}
    for key, _, kind, pf, padf in [STRAIGHT] + METHODS:
        p = TA.load_predictions(pf)
        preds[key] = p
        verd[key] = {r["route"]: TA.score_polyline(p.get(r["route"]), r, masks[r["route"]])
                     for r in routes if r["route"] in p}
        if padf is not None:
            pads[key] = load_pads(padf)
    return routes, masks, preds, verd, pads


def auto_pick(routes: list[dict], verd: dict, first: int = 200) -> list[tuple[int, str]]:
    pool = [r for r in routes[:first] if all(r["route"] in verd[k] for k, *_ in METHODS)]
    both = [r for r in pool if verd["lora"][r["route"]]["solved"] and verd["wave"][r["route"]]["solved"]]
    both.sort(key=lambda r: (r["n_goal"], r["teacher_m"]), reverse=True)
    print(f"[auto] both solved: {len(both)} candidates; top 5:",
          ", ".join(f"{r['route']} (n_goal {r['n_goal']}, {r['teacher_m']:.0f} m)" for r in both[:5]))
    if not both:
        raise SystemExit("no route for group 'both solved'")
    picked, windows = [(both[0]["route"], "both solved")], {tuple(both[0]["origin"])}
    groups = [
        ("Qwen LoRA solved, wave search not", lambda q, w: q["solved"] and not w["solved"],
         lambda q, w: w["reason"] == "unparseable"),
        ("wave search solved, Qwen LoRA not", lambda q, w: w["solved"] and not q["solved"],
         lambda q, w: q["reason"] == "crosses a building"),
    ]
    for name, cond, pref in groups:
        cands = []
        for r in pool:
            q, w = verd["lora"][r["route"]], verd["wave"][r["route"]]
            if not cond(q, w):
                continue
            s = verd["straight"][r["route"]]
            key = (r["n_goal"] >= 4, not s["solved"], bool(pref(q, w)), tuple(r["origin"]) not in windows,
                   r["teacher_m"] / r["straight_m"])
            cands.append((key, r))
        cands.sort(key=lambda t: t[0], reverse=True)
        print(f"[auto] {name}: {len(cands)} candidates; top 5:",
              ", ".join(f"{r['route']} (n_goal {r['n_goal']}, detour x{r['teacher_m'] / r['straight_m']:.2f})"
                        for _, r in cands[:5]))
        if not cands:
            raise SystemExit(f"no route for group '{name}'")
        r = cands[0][1]
        picked.append((r["route"], name)); windows.add(tuple(r["origin"]))
    return picked


def draw_panel(rec: dict, bmask: np.ndarray, kind: str, points, pad: np.ndarray | None, P: int, ss: int = 3) -> Image.Image:
    """One map panel of P x P px (drawn at ss x and downsampled for smooth lines and discs)."""
    win = int(rec["window_m"]); m = TA.MASK_MARGIN_M
    bld = bmask[m:m + win, m:m + win]
    S = P * ss; k = S / win                                   # super-sampled px per metre
    rgb = np.empty((win, win, 3), np.uint8); rgb[:] = BG; rgb[bld] = MAP
    im = Image.fromarray(rgb).resize((S, S), Image.NEAREST)
    if kind == "grid" and pad is not None:
        cell = Image.fromarray((pad * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        im.paste(Image.new("RGB", (S, S), PAD), (0, 0), cell)
    d = ImageDraw.Draw(im)
    ppm = float(rec["px_per_m"])
    rad = TA.MARKER_RADIUS_M * k
    for xy in (rec["start"], rec["goal"]):
        cx, cy = xy[0] / ppm * k, xy[1] / ppm * k
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=MAP)
    if kind == "poly" and isinstance(points, list) and len(points) >= 2:
        xy = [(float(px) / ppm * k, float(py) / ppm * k) for px, py in points]
        d.line(xy, fill=PAD, width=3 * ss, joint="curve")
        v = 2.6 * ss
        for q in xy:
            d.ellipse([q[0] - v, q[1] - v, q[0] + v, q[1] + v], fill=PAD)
    im = im.resize((P, P), Image.LANCZOS)
    d = ImageDraw.Draw(im); f = font(15, bold=True)
    for xy, name in ((rec["start"], "start"), (rec["goal"], "goal")):
        cx, cy = xy[0] / ppm * P / win, xy[1] / ppm * P / win
        w = d.textlength(name, font=f)
        tx = cx + 10 if cx + 10 + w < P - 2 else cx - 10 - w
        ty = min(max(cy - 9, 2), P - 20)
        d.text((tx, ty), name, fill=MAP, font=f, stroke_width=3, stroke_fill=(255, 255, 255))
    return im


def figure(picks: list[int], routes, masks, preds, verd, pads, P: int = 440) -> Image.Image:
    recs = {r["route"]: r for r in routes}
    mx, gap, head, ttl, rgap = 20, 20, 64, 50, 18
    W = 2 * mx + len(picks) * P + (len(picks) - 1) * gap
    f_note = font(15)
    note = wrap(NOTE, f_note, W - 2 * mx)
    H = head + len(METHODS) * (ttl + P + rgap) + 8 + 22 * len(note) + 10
    im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    f_head, f_sub, f_ttl, f_small = font(22, bold=True), font(15), font(17, bold=True), font(14)
    for c, rix in enumerate(picks):
        rec = recs[rix]; x = mx + c * (P + gap)
        d.text((x, 10), group_name(verd["lora"].get(rix), verd["wave"].get(rix)), fill=(20, 20, 20), font=f_head)
        d.text((x, 38), f"De Pijp val route {rix} · {rec['teacher_m']:.0f} m walking · {rec['n_goal']} fronts",
               fill=(90, 90, 90), font=f_sub)
        for r, (key, title, kind, *_rest) in enumerate(METHODS):
            y = head + r * (ttl + P + rgap)
            res = verd[key].get(rix)
            pts = preds[key].get(rix)
            pad = pads[key].get(rix) if kind == "grid" else None
            d.text((x, y), title, fill=(20, 20, 20), font=f_ttl)
            if res is None:
                sub, col = "no prediction for this route", (110, 110, 110)
            else:
                sub = verdict_text(res)
                if kind == "grid" and pad is not None:
                    sub += f" · {int(pad.sum())} lit cells"
                col = (30, 120, 60) if res["solved"] else (170, 40, 30)
            d.text((x, y + 23), sub, fill=col, font=f_small)
            im.paste(draw_panel(rec, masks[rix], kind, pts, pad, P), (x, y + ttl))
            d.rectangle([x - 1, y + ttl - 1, x + P, y + ttl + P], outline=(210, 210, 210))
    y = head + len(METHODS) * (ttl + P + rgap) + 4
    for line in note:
        d.text((mx, y), line, fill=(90, 90, 90), font=f_note); y += 22
    return im


NOTE = ("The length reference is the teacher's clearance-priced walking path, which keeps away from walls; a legal line "
        "closer to the buildings can be shorter than 1.0×. The cap is 1.10×.")


def wrap(text: str, f, width: int) -> list[str]:
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if cur and d.textlength(t, font=f) > width:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + [cur] if cur else lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--routes", type=int, nargs="+", help="De Pijp val route ids, one column each")
    g.add_argument("--auto", action="store_true", help="pick three routes (both solved / only Qwen / only the search) from the first 200 by the rule in the docstring")
    ap.add_argument("--out", type=Path, nargs="+",
                    default=[ROOT / "docs/memo/wave_search_2026-10-05/method_comparison_3_routes.png"])
    ap.add_argument("--panel", type=int, default=440, help="panel size in px (3 columns at 440 -> 1400 px wide)")
    a = ap.parse_args()

    routes, masks, preds, verd, pads = load_all()
    for key, *_ in [STRAIGHT] + METHODS:                      # the shares must match the comparison table
        v = verd[key]; n = len(v)
        print(f"{key:9s} n={n:3d} solved {sum(x['solved'] for x in v.values()) / n:.3f}")
    picks = [r for r, _ in auto_pick(routes, verd)] if a.auto else a.routes
    for rix in picks:
        print(f"route {rix}: " + " | ".join(
            f"{key} {verd[key][rix]['reason'] or 'solved'}" if rix in verd[key] else f"{key} -" for key, *_ in METHODS))
    im = figure(picks, routes, masks, preds, verd, pads, a.panel)
    for out in a.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, optimize=True)
        print("wrote", out, im.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
