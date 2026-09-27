import numpy as np, json, glob
from pathlib import Path
root = Path.home() / "Github/spatial_reasoning_LLM_artifact/data/amsterdam/de_pijp/crops/v2"
leg = json.load(open(root / "legend.json")); print("classes:", json.dumps(leg["classes"])[:400])
files = sorted(glob.glob(str(root / "train/*_labels.npz")))[:400]
L = {"noise": [], "surface": [], "estab": []}
for f in files:
    z = np.load(f)
    for k in L: L[k].append(z[f"{k}_256"])
L = {k: np.stack(v) for k, v in L.items()}                       # (n,256,256)
n = len(files); print("crops:", n)
print(f"{'task':22s} {'px%':>6s} {'any%':>6s} {'maj%':>6s} {'ctr%':>6s} {'empty crops% (any)':>18s} {'any/maj':>8s}")
for k, arr in L.items():
    blk = arr.reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16)
    for c in range(4):
        m = blk == c
        cnt = m.sum(-1)
        any_ = cnt > 0; maj = cnt >= 8; ctr = arr[:, 2::4, 2::4] == c
        empty = (any_.reshape(n, -1).sum(1) == 0).mean()
        print(f"{k+'=='+str(c):22s} {100*(arr==c).mean():6.2f} {100*any_.mean():6.2f} {100*maj.mean():6.2f} {100*ctr.mean():6.2f} {100*empty:18.1f} {any_.mean()/max(maj.mean(),1e-9):8.2f}")
# noise >= 65 dB (classes 2,3)
arr = L["noise"]; blk = (arr >= 2).reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16).sum(-1)
print(f"{'noise>=2 (>=65dB)':22s} {100*(arr>=2).mean():6.2f} {100*(blk>0).mean():6.2f} {100*(blk>=8).mean():6.2f} {'':6s} {100*((blk>0).reshape(n,-1).sum(1)==0).mean():18.1f}")
# mixed cells: share of grid cells containing more than one surface class
s = L["surface"].reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16)
mixed = (s.max(-1) != s.min(-1)).mean(); print(f"surface: {100*mixed:.1f}% of 4x4 cells contain more than one class")
e = L["estab"].reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16)
print(f"estab: cells with any establishment {100*(e.max(-1)>0).mean():.2f}% ; median per crop {np.median((e.max(-1)>0).reshape(n,-1).sum(1)):.0f} cells")
