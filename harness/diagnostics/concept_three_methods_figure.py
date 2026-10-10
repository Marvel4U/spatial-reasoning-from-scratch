#!/usr/bin/env python3
"""Concept figure: the three methods of the comparison, each as a chain from input to scored output (10 Oct 2026;
README figure, no title inside the image, the README caption carries it).

Three lanes, one per method, each read left to right, with real pictures of De Pijp val route 192 (350 m walking,
9 fronts; the "only the search solved" column of method_comparison_3_routes.png):
  1 One-shot network   the map with the two markers -> transformer, 4 blocks, one pass -> its drawing -> scored as
                       lit cells (the drawing: runs/diagnostics/<ONESHOT>/pads_oneshot.npz)
  2 Wave search        map + scratchpad -> the same transformer, one pass -> next scratchpad + state token; output i
                       = input i+1 for 10-30 passes -> final drawing -> scored as lit cells. The two scratchpads are
                       passes 4 and 5 of the free-running search, the final drawing its last frame (= pads_wave.npz,
                       checked on load). Frames from
                         PYTHONPATH=. python diagnostics/wave_rollout_gif.py <WAVE> --route 192 \\
                             --save-frames ../runs/diagnostics/<WAVE>/wave_search_route192_frames.npz
  3 Qwen3-VL-2B + LoRA the RGB picture the VLM sees (data/track_a/de_pijp_val_plain/images/route_000192.png) ->
                       vision encoder -> language model -> its answer (JSON points on Qwen's 0-1000 grid; the
                       prediction file holds them converted to pixels, shown here converted back and abbreviated) ->
                       the polyline on the map -> scored as a polyline (runs/qwen_sft/lora20000_v1/pred_val_v1_n500.jsonl)
A bar across the bottom states the one rule all three are scored by (spatial_data/track_a_export.py score_polyline);
the grid methods' drawings are converted to the shortest chain through their lit cells first.

Colours as wave_rollout_gif.py: ground #F5F3EE, buildings and markers #343A46, drawing #E86A17, grey boxes.

Usage (from harness/, CPU only): PYTHONPATH=. python diagnostics/concept_three_methods_figure.py [--out PNG ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_data import track_a_export as TA  # noqa: E402

BG, MAP, PAD = (247, 245, 240), (52, 58, 70), (232, 106, 23)
C_BG, C_MAP, C_PAD = "#F5F3EE", "#343A46", "#E86A17"
BOX, BOX_EDGE, TXT, SUB = "#E6E6E3", "#9A9A96", "#1E1E1E", "#5A5A5A"

ROUTE = 192
EXPORT = ROOT / "data/track_a/de_pijp_val_plain"
ONESHOT = "c5_os_bld_det_L4_64k_g128_store100k"
WAVE = "c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k"
LORA = ROOT / "runs/qwen_sft/lora20000_v1/pred_val_v1_n500.jsonl"
PASS_IN, PASS_OUT = 4, 5                                     # the scratchpads shown in the wave lane


def unpack(z, grid: int) -> np.ndarray:
    return np.unpackbits(z["pad"], axis=-1)[..., :grid].astype(bool)


def thumb(rec: dict, bmask: np.ndarray, pad: np.ndarray | None = None, points=None, P: int = 480) -> np.ndarray:
    """The map (1 m building raster of the window), optional cell drawing or polyline, markers as dark discs."""
    win = int(rec["window_m"]); m = TA.MASK_MARGIN_M
    bld = bmask[m:m + win, m:m + win]; k = P / win
    rgb = np.empty((win, win, 3), np.uint8); rgb[:] = BG; rgb[bld] = MAP
    im = Image.fromarray(rgb).resize((P, P), Image.NEAREST)
    if pad is not None:
        cell = Image.fromarray((pad * 255).astype(np.uint8)).resize((P, P), Image.NEAREST)
        im.paste(Image.new("RGB", (P, P), PAD), (0, 0), cell)
    d = ImageDraw.Draw(im); ppm = float(rec["px_per_m"])
    if points:
        xy = [(float(x) / ppm * k, float(y) / ppm * k) for x, y in points]
        d.line(xy, fill=PAD, width=max(3, P // 50), joint="curve")
    r = 1.6 * TA.MARKER_RADIUS_M * k                         # a little larger than the scorer's disc, to read in a thumbnail
    for x, y in (rec["start"], rec["goal"]):
        cx, cy = x / ppm * k, y / ppm * k
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=MAP, outline=(255, 255, 255), width=max(1, P // 160))
    return np.asarray(im)


def load_inputs():
    rec = next(r for r in TA.load_routes(EXPORT / "routes.jsonl") if r["route"] == ROUTE)
    bmask = TA.load_building_mask(rec)
    zo = np.load(ROOT / f"runs/diagnostics/{ONESHOT}/pads_oneshot.npz")
    g = int(json.loads(str(zo["meta"]))["grid"])
    one = unpack(zo, g)[list(zo["route"]).index(ROUTE)]
    zf = np.load(ROOT / f"runs/diagnostics/{WAVE}/wave_search_route{ROUTE}_frames.npz")
    frames = unpack(zf, int(zf["grid"])); titles = [str(t) for t in zf["titles"]]
    zw = np.load(ROOT / f"runs/diagnostics/{WAVE}/pads_wave.npz")
    final = unpack(zw, int(json.loads(str(zw["meta"]))["grid"]))[list(zw["route"]).index(ROUTE)]
    if not (frames[-1] == final).all():
        raise SystemExit("the saved frames do not end in the drawing of pads_wave.npz; re-run wave_rollout_gif.py --save-frames")
    i_in = titles.index(next(t for t in titles if t.startswith(f"pass {PASS_IN}:")))
    i_out = titles.index(next(t for t in titles if t.startswith(f"pass {PASS_OUT}:")))
    n_pass = sum(t.startswith("pass ") for t in titles)
    pts = TA.load_predictions(LORA)[ROUTE]
    w = int(rec["image_px"][0])
    q1000 = [[round(x / w * 1000), round(y / w * 1000)] for x, y in pts]
    rgb = np.asarray(Image.open(EXPORT / rec["image"]).convert("RGB"))
    return rec, bmask, one, frames[i_in], frames[i_out], final, n_pass, pts, q1000, rgb, titles[i_out]


def box(ax, x, y, w, h, text, fs=13, fc=BOX, ec=BOX_EDGE, color=TXT, weight="normal", r=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.25)


def img(ax, x, y, s, a, caption=None):
    ax.imshow(a, extent=(x, x + s, y + s, y), interpolation="lanczos", zorder=2)
    ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec=BOX_EDGE, lw=1.2, zorder=3))
    if caption:
        ax.text(x + s / 2, y + s + 6, caption, ha="center", va="top", fontsize=13.5, color=SUB)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=C_MAP,
                                 shrinkA=0, shrinkB=0, **kw))


def figure(out: list[Path]):
    rec, bmask, one, pad_in, pad_out, final, n_pass, pts, q1000, rgb, out_title = load_inputs()
    W, H = 1600, 765
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor("white")
    S = 170                                                    # thumbnail size in px
    LX = 18                                                    # lane label x
    X0 = 300                                                   # first thumbnail x

    def lane_label(y, title, sub):
        ax.text(LX, y, title, fontsize=16, weight="bold", color=TXT, va="top")
        ax.text(LX, y + 30, sub, fontsize=13.5, color=SUB, va="top", linespacing=1.3)

    def score_box(x, y, text, note):
        box(ax, x, y, 190, 62, text, fs=13.5, fc="#FBE3D3", ec=C_PAD, weight="bold")
        ax.text(x + 95, y + 70, note, ha="center", va="top", fontsize=13.5, color=SUB, style="italic")

    # ---------------- lane 1: one-shot network
    y = 18; cy = y + S / 2
    lane_label(y + 10, "One-shot network", "trained from scratch\nto draw the route\nin one pass")
    img(ax, X0, y, S, thumb(rec, bmask), "map + the two markers")
    arrow(ax, X0 + S + 8, cy, X0 + S + 52, cy)
    box(ax, X0 + S + 58, cy - 45, 190, 90, "transformer,\n4 blocks,\none pass")
    arrow(ax, X0 + S + 254, cy, X0 + S + 298, cy)
    x_out = X0 + S + 304
    img(ax, x_out, y, S, thumb(rec, bmask, pad=one), f"its drawing ({int(one.sum())} lit cells)")
    arrow(ax, x_out + S + 8, cy, 1340, cy)
    score_box(1346, cy - 31, "scored as\nlit cells", "cells → shortest chain\nthrough lit cells")

    # ---------------- lane 2: wave search
    y = 295; cy = y + S / 2
    lane_label(y + 10, "Wave search", "trained from scratch\nto run one pass\nof a search")
    img(ax, X0, y, S, thumb(rec, bmask, pad=pad_in), f"map + scratchpad (pass {PASS_IN})")
    arrow(ax, X0 + S + 8, cy, X0 + S + 52, cy)
    box(ax, X0 + S + 58, cy - 45, 190, 90, "the same\ntransformer,\none pass")
    arrow(ax, X0 + S + 254, cy, X0 + S + 298, cy)
    img(ax, x_out, y, S, thumb(rec, bmask, pad=pad_out), "next scratchpad")
    # token chip: the four state tokens, the one this pass says highlighted
    said = out_title.split("says")[-1].strip()
    tx = x_out + S + 14; ty = y + 8
    ax.text(tx, ty - 2, "+ a state token", fontsize=13.5, color=SUB, va="bottom")
    for j, t in enumerate(("SEARCH", "BACK", "DRAW", "DONE")):
        on = t == said
        box(ax, tx, ty + 6 + j * 32, 92, 26, t, fs=11.5, fc=C_PAD if on else "white", ec=C_PAD if on else BOX_EDGE,
            color="white" if on else TXT, weight="bold" if on else "normal", r=6)
    # loop back: output i = input i + 1
    ax.add_patch(FancyArrowPatch((x_out + S / 2, y - 4), (X0 + S / 2, y - 4), connectionstyle="arc3,rad=0.2",
                                 arrowstyle="-|>", mutation_scale=18, lw=1.8, color=C_PAD))
    ax.text((x_out + X0 + S) / 2, y - 56, f"output i = input i+1, 10–30 passes (here {n_pass})",
            ha="center", va="bottom", fontsize=13, color=C_PAD, weight="bold",
            bbox=dict(fc="white", ec="none", pad=1))
    xf = tx + 124
    yd = ty + 6 + 3 * 32 + 13                                  # from the DONE chip: the search ends
    arrow(ax, tx + 96, yd, xf - 6, yd)
    img(ax, xf, y, S, thumb(rec, bmask, pad=final), f"final drawing ({int(final.sum())} lit cells)")
    arrow(ax, xf + S + 8, cy, 1340, cy)
    score_box(1346, cy - 31, "scored as\nlit cells", "cells → shortest chain\nthrough lit cells")

    # ---------------- lane 3: Qwen3-VL-2B + LoRA
    y = 505; cy = y + S / 2
    lane_label(y + 10, "Qwen3-VL-2B + LoRA", "pretrained VLM,\nfine-tuned on\n19.3k routes")
    img(ax, X0, y, S, rgb, "the picture it sees")
    arrow(ax, X0 + S + 8, cy, X0 + S + 30, cy)
    box(ax, X0 + S + 34, cy - 50, 108, 100, "vision\nencoder")
    arrow(ax, X0 + S + 146, cy, X0 + S + 168, cy)
    box(ax, X0 + S + 172, cy - 50, 108, 100, "language\nmodel")
    arrow(ax, X0 + S + 284, cy, X0 + S + 306, cy)
    shown = ",".join(f"[{a},{b}]" for a, b in q1000[:3])
    ans = f"[{shown},\n ... {len(q1000) - 4} more ...,\n [{q1000[-1][0]},{q1000[-1][1]}]]"
    xa = X0 + S + 312
    box(ax, xa, cy - 50, 250, 100, ans, fs=12, fc="white", ec=BOX_EDGE, r=6)
    ax.text(xa + 125, cy + 56, "its answer: JSON points", ha="center", va="top", fontsize=13.5, color=SUB)
    arrow(ax, xa + 256, cy, xa + 284, cy)
    xp = xa + 290
    img(ax, xp, y, S, thumb(rec, bmask, points=pts), "the polyline on the map")
    arrow(ax, xp + S + 8, cy, 1340, cy)
    score_box(1346, cy - 31, "scored as\na polyline", "0–1000 grid → pixels")

    # ---------------- the one rule
    yb = 712
    ax.add_patch(FancyBboxPatch((18, yb), W - 36, 40, boxstyle="round,pad=0,rounding_size=8", fc=C_MAP, ec="none"))
    ax.text(W / 2, yb + 20, "one rule for all: start and goal within 6 m of the markers, no building crossed, "
            "length at most 1.10 × the walking path", ha="center", va="center", fontsize=15, color="white",
            weight="bold")
    for o in out:
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=100, facecolor="white")
        print("wrote", o, Image.open(o).size)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, nargs="+", default=[ROOT / "docs/memo/wave_search_2026-10-05/concept_three_methods.png"])
    figure(ap.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
