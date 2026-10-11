#!/usr/bin/env python3
"""Headline figure for the public README (11 Oct 2026): 1600 x 900 px (16:9, GitHub social preview), no title inside.

Left 55 %: four frames of the free-running wave search on De Pijp val route 192 (passes 1, 5, 9 and the final pass,
18), drawn as in the GIFs, each labelled "pass n · TOKEN" with the state token that pass said (the final one DONE).
Frames from the saved-frames file that concept_three_methods_figure.py reads (written by
  PYTHONPATH=. python diagnostics/wave_rollout_gif.py <WAVE> --route 192 \
      --save-frames ../runs/diagnostics/<WAVE>/wave_search_route192_frames.npz),
checked to end in the drawing of pads_wave.npz.
Right 45 %: routes solved on De Pijp val (500 held-out routes) under the one polyline rule, from
docs/reports/COMPARISON_TABLE_2026-10-08.md (constants below). Footer: the rule and the 6+-fronts numbers.

Default output docs/memo/wave_search_2026-10-05/headline_result.png (public copy: docs/figures/headline_result.png).

Usage (from harness/, CPU only): PYTHONPATH=. python diagnostics/headline_figure.py [--out PNG ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from diagnostics.concept_three_methods_figure import (BOX_EDGE, C_MAP, C_PAD, EXPORT, ROOT, ROUTE, SUB, TA,  # noqa: E402
                                                      TXT, WAVE, thumb, unpack)

FRAMES = (1, 5, 9)                                             # + the last pass
# De Pijp val, n = 500, polyline rule (COMPARISON_TABLE_2026-10-08.md): (label, all routes %, 6+ fronts %, highlight)
BARS = (("learned search (9 M params, 10–30 passes)", 76.4, 62.8, True),
        ("Qwen3-VL-2B + LoRA, 19.3k routes", 49.6, 28.3, False),
        ("same transformer, one pass", 37.4, 15.4, False),
        ("straight line", 27.6, 2.4, False))


def load_frames():
    rec = next(r for r in TA.load_routes(EXPORT / "routes.jsonl") if r["route"] == ROUTE)
    bmask = TA.load_building_mask(rec)
    zf = np.load(ROOT / f"runs/diagnostics/{WAVE}/wave_search_route{ROUTE}_frames.npz")
    frames = unpack(zf, int(zf["grid"])); titles = [str(t) for t in zf["titles"]]
    zw = np.load(ROOT / f"runs/diagnostics/{WAVE}/pads_wave.npz")
    final = unpack(zw, int(json.loads(str(zw["meta"]))["grid"]))[list(zw["route"]).index(ROUTE)]
    if not (frames[-1] == final).all():
        raise SystemExit("the saved frames do not end in the drawing of pads_wave.npz; re-run wave_rollout_gif.py --save-frames")
    passes = {}                                                # pass number -> (frame index, token said)
    for i, t in enumerate(titles):
        if t.startswith("pass "):
            head, rest = t.split(":", 1)
            passes[int(head.split()[1])] = (i, rest.split("->")[1].strip().split()[-1])
    last = max(passes)
    if passes[last][0] != len(titles) - 1 or passes[last][1] != "DONE":
        raise SystemExit("the last saved frame is not the last pass, or it did not say DONE")
    return rec, bmask, frames, passes, [p for p in FRAMES if p < last] + [last]


def figure(out: list[Path]):
    rec, bmask, frames, passes, shown = load_frames()
    W, H = 1600, 900
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------------- left 55 %: the search, 2 x 2 frames
    S, gx, gy = 340, 36, 54
    x0 = (0.55 * W - (2 * S + gx)) / 2; y0 = 22
    for k, p in enumerate(shown):
        i, said = passes[p]
        x = x0 + (k % 2) * (S + gx); y = y0 + (k // 2) * (S + gy)
        ax.imshow(thumb(rec, bmask, pad=frames[i], P=2 * S), extent=(x, x + S, y + S, y), interpolation="lanczos",
                  zorder=2)
        ax.add_patch(plt.Rectangle((x, y), S, S, fill=False, ec=BOX_EDGE, lw=1.2, zorder=3))
        done = said == "DONE"
        ax.text(x + S / 2, y + S + 9, f"pass {p} · {said}", ha="center", va="top", fontsize=21,
                color=C_PAD if done else TXT, weight="bold")
        if k % 2:
            ax.annotate("", xy=(x - 6, y + S / 2), xytext=(x - gx + 6, y + S / 2),
                        arrowprops=dict(arrowstyle="-|>", color=C_MAP, lw=2, mutation_scale=18))

    # ---------------- right 45 %: routes solved
    xl = 0.55 * W + 6; xmax = W - 30
    full = (xmax - 110 - xl) / 100                             # px per percentage point; room for the value text
    ax.text(xl, 40, "routes solved", fontsize=19, color=SUB, va="top")
    yb, step, bh = 118, 172, 58
    for j, (lab, v, _, hi) in enumerate(BARS):
        y = yb + j * step
        ax.text(xl, y, lab, fontsize=19.5, color=TXT, va="bottom", weight="bold" if hi else "normal")
        ax.add_patch(plt.Rectangle((xl, y + 10), v * full, bh, fc=C_PAD if hi else C_MAP, ec="none"))
        ax.text(xl + v * full + 12, y + 10 + bh / 2, f"{v:.1f} %", fontsize=26 if hi else 23, va="center",
                color=C_PAD if hi else TXT, weight="bold")
    ax.add_line(plt.Line2D((xl, xl), (yb - 4, yb + 3 * step + bh + 16), color=BOX_EDGE, lw=1.2))

    # ---------------- footer
    six = " / ".join(f"{round(b[2])} %" for b in BARS)
    ax.add_line(plt.Line2D((30, W - 30), (H - 86, H - 86), color=BOX_EDGE, lw=1))
    ax.text(W / 2, H - 60, "500 held-out routes · one rule for all methods (endpoints, no building crossed, "
            "length ≤ 1.10 × the walking path)", ha="center",
            va="center", fontsize=18, color=SUB)
    ax.text(W / 2, H - 22, f"routes of 6+ fronts: {six}", ha="center", va="center", fontsize=18, color=SUB)
    for o in out:
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=100, facecolor="white")
        print("wrote", o, Image.open(o).size, "passes", shown)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, nargs="+",
                    default=[ROOT / "docs/memo/wave_search_2026-10-05/headline_result.png"])
    figure(ap.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
