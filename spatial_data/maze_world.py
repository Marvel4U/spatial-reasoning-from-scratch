"""Synthetic maze district (3 Oct 2026): a made-up district whose only obstacles are maze walls, written in the same
files and arrays as a real worldsnap store, so the route builder, the wave builder, the loaders and the rollout eval
run on it unchanged. Purpose: test the wave-search network zero-shot on ground it has never seen anything like.

What the c4/c5 path reads from a crops dir, and therefore what this writes:
  district_labels_1m.npz  planes noise7, surface, estab (dataset_c4_gpu.load_district_planes_v4) and surface alone
                          (c4_routes, c5_wave, the rollout eval and gif: building = class 3). The 4-level noise plane is
                          written too, for the RGB loader. Walls are class 3; corridors are GROUND_CLASS (0, "none" in
                          the legend: 43 % of Zuid-Centrum is class 0 and walkable); noise, noise7 and estab are all 0.
  split.json              block_m, val_blocks, test_blocks (c4_routes.WindowSampler); the other keys as worldsnap
                          writes them, plus the maze style of every block.
Written for people only: legend.json, district_labels_transforms.json, maze.json (parameters and checks),
maze_overview.png. Not written: the 0.25 m planes, index.jsonl and the train/val/test crop dirs; only the c0-c3
loaders read crop files, nothing on the c4/c5 path does.

The maze. One rectilinear lattice over the whole district: along each axis a wall strip, a corridor, a wall strip, ...
with every corridor width drawn from --cell-m (the cell size: a cell is the open square between four wall strips) and
every wall thickness from --wall-m, so widths vary from street to street as in a city. A cell graph on that lattice
(neighbouring cells are joined where their wall segment is open) is built per block of the n x n split grid, in one of
three styles:
  perfect  a random spanning tree (Kruskal): exactly one way between any two cells, many dead ends
  braided  the same tree, then dead ends opened into a neighbour with probability --braid: loops, few dead ends
  rooms    recursive division into rooms of --room-cells cells a side; each dividing wall gets one door (one cell
           segment wide), long walls sometimes a second one; rooms are open inside
Neighbouring blocks are joined by a few open segments on their shared border, so the whole district is one connected
walkable component. A lattice crossing (the square where four wall segments meet) stays building unless all four of
its segments are open; otherwise rooms would be full of free-standing pillars.

Blocks get the styles in a diagonal pattern; val and test each hold one block of every style (3 + 3 of 16 blocks at
n = 4), so the eval sees all three. Deterministic from --seed.

Usage (project venv, repo root):
  python -m spatial_data.maze_world --out data/synthetic/maze_v1/crops/v4ns
  python -m spatial_data.c4_routes --crops-dir data/synthetic/maze_v1/crops/v4ns --out data/synthetic/maze_v1/crops/v4ns/c4_256_val \
      --n-train 0 --n-val 500 --n-test 500 --crop-px 256 --d-min 40 --d-max 250 --max-expansions 600
  python -m spatial_data.c5_wave --store data/synthetic/maze_v1/crops/v4ns/c4_256_val --splits val,test --gallery 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

BUILDING_CLASS = 3
GROUND_CLASS = 0
STYLES = ("perfect", "braided", "rooms")


# ---------------------------------------------------------------- lattice

def lattice(length: int, cell: tuple[int, int], wall: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Corridor intervals (n, 2) [a, b) in metres along one axis of `length` m: wall, corridor, wall, ..., corridor,
    wall. Widths are integers drawn uniformly from the ranges; the remainder is spread one metre at a time over
    elements that are not yet at their maximum (or taken from ones above their minimum), so every width stays in its
    range and the axis is filled exactly."""
    pitch = (cell[0] + cell[1]) / 2 + (wall[0] + wall[1]) / 2
    n = max(1, int(round((length - (wall[0] + wall[1]) / 2) / pitch)))
    w = rng.integers(cell[0], cell[1] + 1, n)
    t = rng.integers(wall[0], wall[1] + 1, n + 1)
    lo = np.concatenate([np.full(n, cell[0]), np.full(n + 1, wall[0])])
    hi = np.concatenate([np.full(n, cell[1]), np.full(n + 1, wall[1])])
    v = np.concatenate([w, t])
    while v.sum() != length:
        grow = v.sum() < length
        ok = np.nonzero(v < hi if grow else v > lo)[0]
        if len(ok) == 0:
            raise ValueError(f"cannot fill {length} m with cells {cell} and walls {wall}")
        v[ok[rng.integers(len(ok))]] += 1 if grow else -1
    w, t = v[:n], v[n:]
    a = np.cumsum(np.concatenate([[t[0]], (w + t[1:])[:-1]]))
    return np.stack([a, a + w], 1)


class Maze:
    """Cell graph on the lattice: east[r, c] = cell (r, c) open to (r, c + 1); south[r, c] = open to (r + 1, c)."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs, self.ys = xs, ys
        self.ny, self.nx = len(ys), len(xs)
        self.east = np.zeros((self.ny, self.nx - 1), bool)
        self.south = np.zeros((self.ny - 1, self.nx), bool)

    def degree(self, r: int, c: int) -> int:
        d = 0
        d += c > 0 and self.east[r, c - 1]
        d += c < self.nx - 1 and self.east[r, c]
        d += r > 0 and self.south[r - 1, c]
        d += r < self.ny - 1 and self.south[r, c]
        return int(d)

    def edges(self, r0, r1, c0, c1):
        """All edges between cells of the rectangle rows r0..r1-1, cols c0..c1-1: ('e'|'s', r, c)."""
        out = [("e", r, c) for r in range(r0, r1) for c in range(c0, c1 - 1)]
        out += [("s", r, c) for r in range(r0, r1 - 1) for c in range(c0, c1)]
        return out

    def set(self, e, val: bool = True):
        (self.east if e[0] == "e" else self.south)[e[1], e[2]] = val

    def render(self, h: int, w: int) -> np.ndarray:
        """(h, w) bool, True = wall."""
        free = np.zeros((h, w), bool)
        xs, ys = self.xs, self.ys
        for r in range(self.ny):
            for c in range(self.nx):
                free[ys[r, 0]:ys[r, 1], xs[c, 0]:xs[c, 1]] = True
        for r, c in zip(*np.nonzero(self.east)):        # the wall strip between column c and c + 1, row r's span
            free[ys[r, 0]:ys[r, 1], xs[c, 1]:xs[c + 1, 0]] = True
        for r, c in zip(*np.nonzero(self.south)):
            free[ys[r, 1]:ys[r + 1, 0], xs[c, 0]:xs[c, 1]] = True
        # crossings: open only when all four segments around them are open
        cross = self.east[:-1, :] & self.east[1:, :] & self.south[:, :-1] & self.south[:, 1:]
        for r, c in zip(*np.nonzero(cross)):
            free[ys[r, 1]:ys[r + 1, 0], xs[c, 1]:xs[c + 1, 0]] = True
        return ~free


# ---------------------------------------------------------------- styles (each works on one block's cell rectangle)

def perfect(m: Maze, r0, r1, c0, c1, rng):
    """Kruskal: shuffle the edges, open one whenever it joins two trees."""
    parent = {(r, c): (r, c) for r in range(r0, r1) for c in range(c0, c1)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    edges = m.edges(r0, r1, c0, c1)
    for i in rng.permutation(len(edges)):
        e = edges[i]; a = (e[1], e[2]); b = (e[1], e[2] + 1) if e[0] == "e" else (e[1] + 1, e[2])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; m.set(e)


def braided(m: Maze, r0, r1, c0, c1, rng, braid: float):
    """The perfect maze, then every dead end opened into a neighbour with probability `braid` (a neighbour that is a
    dead end itself first, so one opening removes two)."""
    perfect(m, r0, r1, c0, c1, rng)
    cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    for i in rng.permutation(len(cells)):
        r, c = cells[i]
        if m.degree(r, c) != 1 or rng.random() >= braid:
            continue
        cand = []
        for e, (rr, cc) in ((("e", r, c - 1), (r, c - 1)), (("e", r, c), (r, c + 1)),
                            (("s", r - 1, c), (r - 1, c)), (("s", r, c), (r + 1, c))):
            if r0 <= rr < r1 and c0 <= cc < c1 and not (m.east if e[0] == "e" else m.south)[e[1], e[2]]:
                cand.append((m.degree(rr, cc) == 1, e))
        if cand:
            best = [e for dead, e in cand if dead] or [e for _, e in cand]
            m.set(best[rng.integers(len(best))])


def rooms(m: Maze, r0, r1, c0, c1, rng, room_cells: tuple[int, int], second_door: float):
    """Recursive division: the block starts as one open room; a room larger than k cells a side (k drawn per room
    from room_cells) is cut across its longer side by a wall with one door, and both halves recurse."""
    for e in m.edges(r0, r1, c0, c1):
        m.set(e)

    def divide(a0, a1, b0, b1):                      # rows a0..a1-1, cols b0..b1-1
        h, w = a1 - a0, b1 - b0
        k = int(rng.integers(room_cells[0], room_cells[1] + 1))
        if h <= k and w <= k:
            return
        vertical = w > h if w != h else bool(rng.random() < 0.5)
        if vertical:                                 # a wall between columns c and c + 1, rows a0..a1-1
            c = int(rng.integers(b0, b1 - 1))
            seg = [("e", r, c) for r in range(a0, a1)]
            halves = ((a0, a1, b0, c + 1), (a0, a1, c + 1, b1))
        else:
            r = int(rng.integers(a0, a1 - 1))
            seg = [("s", r, cc) for cc in range(b0, b1)]
            halves = ((a0, r + 1, b0, b1), (r + 1, a1, b0, b1))
        for e in seg:
            m.set(e, False)
        doors = 1 + (len(seg) > 4 and rng.random() < second_door)
        for i in rng.choice(len(seg), size=min(doors, len(seg)), replace=False):
            m.set(seg[i])
        for hv in halves:
            divide(*hv)
    divide(r0, r1, c0, c1)


# ---------------------------------------------------------------- district

def block_cells(centres: np.ndarray, n: int, size: int) -> list[tuple[int, int]]:
    """Index range [lo, hi) of the cells whose centre lies in each of the n blocks along one axis."""
    b = np.minimum((centres // (size / n)).astype(int), n - 1)
    return [(int(np.searchsorted(b, k, "left")), int(np.searchsorted(b, k, "right"))) for k in range(n)]


def assign_blocks(n: int, styles: tuple[str, ...], rng: np.random.Generator):
    """Styles on diagonals ((row + col) mod 3); one val and one test block per style, drawn from the seed."""
    style = {(r, c): styles[(r + c) % len(styles)] for r in range(n) for c in range(n)}
    val, test = [], []
    for s in styles:
        cand = [b for b in sorted(style) if style[b] == s]
        if len(cand) < 2:
            raise ValueError(f"style {s} needs at least two blocks for val and test (n = {n})")
        i, j = rng.choice(len(cand), 2, replace=False)
        val.append(list(cand[i])); test.append(list(cand[j]))
    return style, sorted(val), sorted(test)


def build_maze(size_m: tuple[int, int], n_blocks: int, cell_m, wall_m, seed: int, *, braid: float = 0.6,
               room_cells=(2, 4), second_door: float = 0.5, links: int = 2, styles=STYLES):
    """-> wall (H, W) bool, Maze, style per block, val blocks, test blocks."""
    W, H = size_m
    rng = np.random.default_rng(seed)
    xs = lattice(W, cell_m, wall_m, rng); ys = lattice(H, cell_m, wall_m, rng)
    m = Maze(xs, ys)
    style, val, test = assign_blocks(n_blocks, tuple(styles), rng)
    bx = block_cells(xs.mean(1), n_blocks, W); by = block_cells(ys.mean(1), n_blocks, H)
    for (br, bc), s in sorted(style.items()):
        (r0, r1), (c0, c1) = by[br], bx[bc]
        if s == "perfect":
            perfect(m, r0, r1, c0, c1, rng)
        elif s == "braided":
            braided(m, r0, r1, c0, c1, rng, braid)
        elif s == "rooms":
            rooms(m, r0, r1, c0, c1, rng, room_cells, second_door)
        else:
            raise ValueError(s)
    # join neighbouring blocks through a few segments of their shared border
    for br in range(n_blocks):
        for bc in range(n_blocks):
            (r0, r1), (c0, c1) = by[br], bx[bc]
            if bc + 1 < n_blocks:                        # east border: edges east[r, c1 - 1]
                for r in rng.choice(np.arange(r0, r1), min(links, r1 - r0), replace=False):
                    m.east[r, c1 - 1] = True
            if br + 1 < n_blocks:
                for c in rng.choice(np.arange(c0, c1), min(links, c1 - c0), replace=False):
                    m.south[r1 - 1, c] = True
    return m.render(H, W), m, style, val, test


def check(wall: np.ndarray, m: Maze, cell_m, wall_m) -> dict:
    """The maze properties the eval relies on: one walkable component, widths in range."""
    lab, ncomp = ndimage.label(~wall)
    cw = np.concatenate([np.diff(m.xs, axis=1).ravel(), np.diff(m.ys, axis=1).ravel()])
    ww = np.concatenate([m.xs[1:, 0] - m.xs[:-1, 1], m.ys[1:, 0] - m.ys[:-1, 1],
                         [m.xs[0, 0], m.ys[0, 0], wall.shape[1] - m.xs[-1, 1], wall.shape[0] - m.ys[-1, 1]]])
    return dict(walkable_components=int(ncomp), wall_share=float(wall.mean()),
                corridor_m_min=int(cw.min()), corridor_m_max=int(cw.max()), wall_m_min=int(ww.min()), wall_m_max=int(ww.max()),
                corridors_in_range=bool(cw.min() >= cell_m[0] and cw.max() <= cell_m[1]),
                walls_in_range=bool(ww.min() >= wall_m[0] and ww.max() <= wall_m[1]),
                cells=[m.ny, m.nx], open_edges=int(m.east.sum() + m.south.sum()))


def write_store(out: Path, size_m=(2000, 2000), n_blocks: int = 4, cell_m=(8, 12), wall_m=(2, 6), seed: int = 0,
                *, crop_m: int = 256, overview: bool = True, ground_class: int = GROUND_CLASS, **kw) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    W, H = size_m
    wall, m, style, val, test = build_maze(size_m, n_blocks, cell_m, wall_m, seed, **kw)
    surface = np.where(wall, BUILDING_CLASS, ground_class).astype(np.uint8)
    zero = np.zeros_like(surface)
    np.savez_compressed(out / "district_labels_1m.npz", noise=zero, surface=surface, estab=zero, noise7=zero)
    bw, bh = W / n_blocks, H / n_blocks
    blocks = []
    for r in range(n_blocks):
        for c in range(n_blocks):
            sp = "val" if [r, c] in val else "test" if [r, c] in test else "train"
            blocks.append(dict(row=r, col=c, split=sp, style=style[(r, c)],
                               offset_m_from_district_nw=[c * bw, r * bh, (c + 1) * bw, (r + 1) * bh]))
    split = dict(crs=None, origin_nw_rd=[0.0, 0.0], district_m=[W, H], n_blocks=n_blocks, block_m=[bw, bh], crop_m=crop_m,
                 val_blocks=val, test_blocks=test,
                 rule="A val/test crop lies fully inside one block of its own split, so val and test ground is disjoint; "
                      "a train crop intersects no held-out block.",
                 blocks=blocks)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "district_labels_transforms.json").write_text(json.dumps(dict(
        crs=None, origin_nw_rd=[0.0, 0.0], width_m=W, height_m=H,
        frames={"1m": dict(transform=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0], res_m=1.0, width_px=W, height_px=H)},
        note="synthetic maze, no real-world position"), indent=1))
    legend = dict(crop_m=crop_m, frames={"256": 1.0}, planes=["noise", "surface", "estab", "noise7"],
                  n_classes=dict(noise=4, surface=4, estab=4, noise7=7),
                  classes=dict(surface={"0": "none (maze corridor)", "1": "roadway (unused)", "2": "sidewalk (unused)",
                                        "3": "building (maze wall)"},
                               noise={"0": "<55 dB"}, estab={"0": "none"}, noise7={"0": "<50 dB (no polygon)"}),
                  plane_notes=dict(surface="maze walls = 3, corridors = 0", noise="all 0", estab="all 0", noise7="all 0"))
    (out / "legend.json").write_text(json.dumps(legend, indent=1))
    stats = check(wall, m, cell_m, wall_m)
    params = dict(size_m=list(size_m), n_blocks=n_blocks, cell_m=list(cell_m), wall_m=list(wall_m), seed=seed,
                  ground_class=ground_class, building_class=BUILDING_CLASS,
                  **{k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()})
    doc = dict(params=params, checks=stats, styles={f"{r},{c}": s for (r, c), s in sorted(style.items())},
               val_blocks=val, test_blocks=test)
    (out / "maze.json").write_text(json.dumps(doc, indent=1))
    if overview:
        draw_overview(out / "maze_overview.png", wall, blocks, bw, bh)
    print(f"maze district {W} x {H} m -> {out}  {stats}")
    return doc


def draw_overview(png: Path, wall: np.ndarray, blocks: list, bw: float, bh: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = wall.shape
    fig, ax = plt.subplots(figsize=(10, 10 * H / W), dpi=150)
    ax.imshow(wall, cmap="Greys", vmin=0, vmax=1.3, interpolation="nearest", extent=(0, W, H, 0))
    col = dict(val="tab:blue", test="tab:red", train="white")
    for b in blocks:
        x0, y0, x1, y1 = b["offset_m_from_district_nw"]
        held = b["split"] != "train"
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=col[b["split"]] if held else "grey",
                                   lw=2.5 if held else 0.8))
        ax.text(x0 + 8, y0 + 30, f"({b['row']},{b['col']}) {b['style']} {b['split'] if held else ''}", fontsize=7,
                color="black", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    ax.set_title("synthetic maze district: walls (dark), blocks with style; val blue, test red", fontsize=10)
    ax.set_xlabel("m east"); ax.set_ylabel("m south")
    fig.savefig(png, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("overview ->", png)


def _pair(s: str, cast=int):
    a, b = (cast(v) for v in s.split(","))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/maze_v1/crops/v4ns")
    ap.add_argument("--size-m", default="2000,2000", help="W,H in metres (1 px = 1 m)")
    ap.add_argument("--n-blocks", type=int, default=4, help="n x n split blocks, one style each")
    ap.add_argument("--cell-m", default="8,12", help="cell size = corridor width range lo,hi in m")
    ap.add_argument("--wall-m", default="2,6", help="wall thickness range lo,hi in m")
    ap.add_argument("--braid", type=float, default=0.6, help="braided: share of dead ends opened")
    ap.add_argument("--room-cells", default="2,4", help="rooms: room side in cells lo,hi")
    ap.add_argument("--second-door", type=float, default=0.5, help="rooms: chance of a second door in walls > 4 cells")
    ap.add_argument("--links", type=int, default=2, help="open segments on every border between two blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ground-class", type=int, default=GROUND_CLASS, choices=(0, 1, 2),
                    help="surface class of the corridors: 0 none, 1 roadway, 2 sidewalk (the geometry does not change)")
    a = ap.parse_args()
    write_store(Path(a.out), _pair(a.size_m), a.n_blocks, _pair(a.cell_m), _pair(a.wall_m), a.seed,
                ground_class=a.ground_class, braid=a.braid, room_cells=_pair(a.room_cells), second_door=a.second_door,
                links=a.links)


if __name__ == "__main__":
    main()
