"""The spatial split: an n x n grid of equal blocks over a district, some held out.

Kept out of ``crops.py`` only so both modules stay short. ``crops.py`` owns the
*policy* (the grid size and which blocks are val / test); this module owns the
geometry, the crop-origin sampling, the split.json record and the split figure.
Blocks are indexed (row, col) with row 0 = north, col 0 = west, and all offsets
here are metres east / south of the district NW corner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rig
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Blocks:
    """``n`` x ``n`` grid; the ``val`` and ``test`` (row, col) blocks are held out."""

    n: int
    val: list
    test: list

    def split_of(self, row: int, col: int) -> str:
        if (row, col) in self.val:
            return "val"
        return "test" if (row, col) in self.test else "train"

    def rect(self, row: int, col: int, w_m: int, h_m: int) -> tuple:
        """(x0, y0, x1, y1) of one block, in metres from the district NW corner."""
        bw, bh = w_m / self.n, h_m / self.n
        return (col * bw, row * bh, (col + 1) * bw, (row + 1) * bh)

    def sample_origins(self, rng: np.random.Generator, split: str, w_m: int, h_m: int,
                       crop_m: int, n: int) -> list:
        """``n`` distinct crop origins (dx, dy), snapped to the 1 m grid.

        val / test: the crop lies FULLY inside one block of its own split (block
        uniform, origin uniform over that block's allowed range), so val and test
        sit on disjoint ground. train: origin uniform over the district, rejecting
        every crop that touches any held-out block (it may straddle train blocks).
        """
        blocks = {"val": self.val, "test": self.test}.get(split)
        held = [self.rect(r, c, w_m, h_m) for r, c in self.val + self.test]
        seen: set[tuple[int, int]] = set()
        while len(seen) < n:
            if blocks is None:
                dx = int(rng.integers(0, w_m - crop_m + 1))
                dy = int(rng.integers(0, h_m - crop_m + 1))
                if any(dx < x1 and dx + crop_m > x0 and dy < y1 and dy + crop_m > y0
                       for x0, y0, x1, y1 in held):
                    continue
            else:
                x0, y0, x1, y1 = self.rect(*blocks[int(rng.integers(len(blocks)))], w_m, h_m)
                dx = int(rng.integers(int(np.ceil(x0)), int(x1) - crop_m + 1))
                dy = int(rng.integers(int(np.ceil(y0)), int(y1) - crop_m + 1))
            seen.add((dx, dy))
        return sorted(seen)

    def records(self, ox: float, oy_top: float, w_m: int, h_m: int) -> list:
        """One dict per block: grid position, split, NW-corner offsets and RD bbox."""
        out = []
        for row in range(self.n):
            for col in range(self.n):
                x0, y0, x1, y1 = self.rect(row, col, w_m, h_m)
                out.append({"row": row, "col": col, "split": self.split_of(row, col),
                            "offset_m_from_district_nw": [round(v, 3) for v in (x0, y0, x1, y1)],
                            "bbox_rd": [round(ox + x0, 3), round(oy_top - y1, 3),
                                        round(ox + x1, 3), round(oy_top - y0, 3)]})
        return out


def figure(png: Path, rgb: np.ndarray, w_m: int, h_m: int, recs: list, centres: dict,
           colours: dict, title: str) -> None:
    """District composite + block grid + a light scatter of crop centres, by split."""
    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=130)
    ax.imshow(rgb, extent=(0, w_m, h_m, 0), interpolation="nearest")
    for b in recs:
        x0, y0, x1, y1 = b["offset_m_from_district_nw"]
        held = b["split"] != "train"
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                                   ec=colours[b["split"]] if held else "white",
                                   lw=3.0 if held else 0.8, ls="-" if held else ":"))
        ax.text(x0 + 8, y0 + 30, f"({b['row']},{b['col']}) {b['split']}", fontsize=8,
                color=colours[b["split"]] if held else "white",
                bbox={"fc": "black", "ec": "none", "alpha": 0.5, "pad": 1.5})
    for split, pts in centres.items():
        p = np.asarray(pts, dtype=float)
        ax.scatter(p[:, 0], p[:, 1], s=7, c=colours[split], alpha=0.5, linewidths=0.2,
                   edgecolors="black", label=f"{split} n={len(pts)}")
    ax.legend(loc="lower right", fontsize=9, markerscale=5, framealpha=0.85)
    ax.set_xlabel("m east of district NW corner")
    ax.set_ylabel("m south of district NW corner")
    ax.set_title(title, fontsize=10)
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
