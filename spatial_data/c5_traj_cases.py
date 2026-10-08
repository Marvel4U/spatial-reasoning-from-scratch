"""Three made-up windows for checking any key-trajectory rule by eye (Marvin, 1 Oct 2026):
  (i)   an empty field, marker in the middle
  (ii)  a regular grid of square blocks (side BLOCK_M) with streets STREET_M wide, marker at a crossing and,
        as a second window, in the middle of a street
  (iii) a diagonal cross of two streets, marker at their crossing
Draws round 1 (blue) and round 2 (orange) for the given horizon.
Usage: python -m spatial_data.c5_traj_cases --radius 40 --out /tmp/traj_cases_r40.png"""
from __future__ import annotations

import argparse

import numpy as np

from spatial_data import c5_trajectories as T

SIZE = 256
BLOCK_M, STREET_M = 20, 5
CROSS_STREET_M = 8


def empty_field():
    return np.zeros((SIZE, SIZE), bool), (SIZE // 2 + 0.5, SIZE // 2 + 0.5)


def block_grid(marker_at="crossing"):
    m = np.ones((SIZE, SIZE), bool); period = BLOCK_M + STREET_M
    # streets are the strips [k*period, k*period + STREET_M); a crossing lies at the middle of two such strips
    for k in range(0, SIZE, period):
        m[k:k + STREET_M, :] = False; m[:, k:k + STREET_M] = False
    kc = (SIZE // 2 // period) * period + STREET_M / 2          # centre of the street strip nearest the middle
    if marker_at == "crossing":
        return m, (kc, kc)
    return m, (kc + period / 2, kc)                            # mid-street, between two crossings


def diagonal_cross():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]; c = SIZE / 2
    d1 = np.abs((xx - c) - (yy - c)) / np.sqrt(2); d2 = np.abs((xx - c) + (yy - c)) / np.sqrt(2)
    m = ~((d1 <= CROSS_STREET_M / 2) | (d2 <= CROSS_STREET_M / 2))
    return m, (c + 0.5, c + 0.5)


CASES = [("(i) empty field", empty_field), ("(ii) block grid, marker at a crossing", lambda: block_grid("crossing")),
         ("(ii) block grid, marker mid-street", lambda: block_grid("street")), ("(iii) diagonal cross", diagonal_cross)]


def run_cases():
    out = []
    for name, fn in CASES:
        mask, xy = fn()
        rounds = T.trajectories_for_window(mask, np.array([xy, xy]), n_markers=1)
        out.append((name, mask, xy, rounds[0][0], rounds[1:]))
    return out


def draw(path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = run_cases()
    fig, axes = plt.subplots(1, len(res), figsize=(4.4 * len(res), 4.7))
    cols = ["tab:orange", "tab:red", "tab:purple", "tab:brown"]
    for ax, (name, mask, xy, r1, later) in zip(axes, res):
        ax.imshow(mask, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        for rr, col in reversed(list(zip(later, cols))):
            for trajs in rr:
                for tr in trajs:
                    ax.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=1.6)
        for tr in r1:
            ax.plot(tr[:, 0], tr[:, 1], "-", color="tab:blue", lw=2.2); ax.plot(*tr[-1], "s", color="tab:blue", ms=4, mec="k")
        ax.plot(*xy, "o", color="red", ms=7, mec="k")
        ax.add_patch(plt.Circle(xy, T.RADIUS_M, fill=False, ls=":", color="0.4"))
        counts = " / ".join(str(sum(len(t) for t in rr)) for rr in later)
        ax.set_title(f"{name}\nrounds: {len(r1)} / {counts}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Key trajectories, horizon {T.RADIUS_M:.0f} m (dotted circle): round 1 from the marker (blue, ends = squares), "
                 f"then from each end: round 2 orange, 3 red, 4 purple", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=100); print("figure ->", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--radius", type=float, default=40.0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); T.set_radius(a.radius); draw(a.out)


if __name__ == "__main__":
    main()
