"""Target statistics for c2 candidates: conjunctions (layer A = class a) AND (layer B = class b) on the 64x64 grid.
Rule measured: pixel-level AND, then majority over the 4x4 cell. Also: cell-level AND of the two majority masks."""
import glob, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3] if "perf_prototypes" in __file__ else Path.home() / "Github/spatial_reasoning_LLM_artifact"
cropset = sys.argv[1] if len(sys.argv) > 1 else "v3b"
files = sorted(glob.glob(str(REPO / f"data/amsterdam/de_pijp/crops/{cropset}/train/*_labels.npz")))
L = {k: np.stack([np.load(f)[f"{k}_256"] for f in files]) for k in ("noise", "surface", "estab")}
n = len(files)
NAMES = {"noise": ["<55dB", "55-65", "65-75", ">=75"], "surface": ["none", "roadway", "sidewalk", "building"],
         "estab": ["no-estab", "other", "shop", "food&drink"]}


def cells(mask):  # (n,256,256) bool -> per-cell count of true pixels (n,64,64)
    return mask.reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16).sum(-1)


maj = {(k, c): cells(L[k] == c) >= 8 for k in L for c in range(4)}
print(f"cropset {cropset}, {n} train crops. f = share of positive cells; empty = crops with no positive cell;")
print("ratio = f(A and B) / min(f(A), f(B)): near 1 means the conjunction adds nothing to its rarer part; per-crop f p10/p50/p90 shows sample dependence")
for a, b in (("noise", "surface"), ("noise", "estab"), ("surface", "estab")):
    print(f"\n== {a} x {b}")
    print(f"{'conjunction':34s} {'f %':>6s} {'empty%':>7s} {'ratio':>6s} {'w=(1-f)/f':>10s} {'per-crop f% p10/p50/p90':>26s} {'cellAND!=pixAND %':>18s}")
    for ca in range(4):
        for cb in range(4):
            pix = cells((L[a] == ca) & (L[b] == cb)) >= 8
            f = pix.mean()
            if f == 0:
                print(f"{NAMES[a][ca] + ' & ' + NAMES[b][cb]:34s} {0:6.2f}  (never)"); continue
            per = pix.reshape(n, -1).mean(1)
            empty = (per == 0).mean()
            ratio = f / min(maj[(a, ca)].mean(), maj[(b, cb)].mean())
            cell_and = maj[(a, ca)] & maj[(b, cb)]
            diff = (cell_and != pix).mean() / max(f, 1e-9)
            q = np.percentile(per, [10, 50, 90]) * 100
            print(f"{NAMES[a][ca] + ' & ' + NAMES[b][cb]:34s} {100 * f:6.2f} {100 * empty:7.1f} {ratio:6.2f} {(1 - f) / f:10.1f} "
                  f"{q[0]:8.1f}/{q[1]:5.1f}/{q[2]:5.1f} {100 * diff:18.1f}")
