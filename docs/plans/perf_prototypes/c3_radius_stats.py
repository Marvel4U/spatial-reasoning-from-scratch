"""Which radius makes 'X within R m of the marker' a usable task? Measured on v4 train crops with the same
markers the within_20m run used (t0_point_v0, first marker per crop)."""
import sys, glob, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3] if "perf_prototypes" in __file__ else Path.home() / "Github/spatial_reasoning_LLM_artifact"
ddir = REPO / "data/amsterdam/de_pijp"
sys.path.insert(0, str(REPO))
from spatial_data.c3_markers import marker_table_for_split

files = sorted(glob.glob(str(ddir / "crops/v4/train/*_labels.npz")))
cols, rows = marker_table_for_split(ddir / "crops/v4", ddir, "train")
E = np.stack([np.load(f)["estab_256"] for f in files]); S = np.stack([np.load(f)["surface_256"] for f in files])
n = len(files); assert len(cols) == n
yy, xx = np.mgrid[0:256, 0:256]
CONDS = {"food & drink (estab=3)": E == 3, "any establishment (estab>0)": E > 0, "sidewalk (surface=2)": S == 2}
print(f"{n} train crops; per radius: share of samples whose target is NON-EMPTY, mean positive share of the 64x64 grid (majority rule), and the disc alone")
print(f"{'radius':>7s} {'disc cells%':>11s} | " + " | ".join(f"{k:>34s}" for k in CONDS))
for R in (20, 30, 50, 75, 100, 150):
    d2 = (xx[None] - cols[:, None, None]) ** 2 + (yy[None] - rows[:, None, None]) ** 2
    disc = d2 <= R * R                                                      # (n,256,256)
    def cells(mask):
        return mask.reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16).sum(-1) >= 8
    dc = cells(disc)
    out = []
    for k, m in CONDS.items():
        c = cells(disc & m); ne = c.reshape(n, -1).any(1)
        out.append(f"non-empty {100 * ne.mean():5.1f}%  pos {100 * c.mean():5.2f}%  ")
    print(f"{R:5d} m {100 * dc.mean():10.2f}% | " + " | ".join(f"{o:>34s}" for o in out))
