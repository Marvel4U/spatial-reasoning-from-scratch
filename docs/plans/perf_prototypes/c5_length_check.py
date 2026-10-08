"""Visual check of the C5 length tasks: scratchpad input, mask target (copy), token target."""
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spatial_data.c4_tasks import get_c4_task, LENGTH_BIN_M
from spatial_data.dataset_c4_gpu import load_district_planes_v4, load_c4_route_index, C4GpuValLoader
dev = torch.device("cuda"); crops = REPO / "data/amsterdam/de_pijp/crops/v4"
district = load_district_planes_v4(crops, dev); val = load_c4_route_index(crops / "c4_100k", "val", dev)
fig, axes = plt.subplots(2, 6, figsize=(16, 5.8))
for r, key in enumerate(("length_line", "length_path")):
    vl = C4GpuValLoader(get_c4_task(key), "onehot_v4_sp", 64, 16, district=district, route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta(); tok = meta["tok_tgt"].cpu().numpy()
    for c in range(6):
        ax = axes[r, c]; bld = img[c, 10].cpu().numpy(); sp = img[c, 16].cpu().numpy()
        ax.imshow(bld, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        ax.imshow(np.ma.masked_where(sp == 0, sp), cmap="winter", interpolation="nearest")
        ax.set_title(f"{key}: token {tok[c]} = {tok[c] * LENGTH_BIN_M:.0f}–{tok[c] * LENGTH_BIN_M + 9:.0f} m", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("C5 rungs 2–3: scratchpad input (blue) over the buildings, markers hidden, and the length token target", fontsize=11)
fig.tight_layout(); fig.savefig("/tmp/c5_length_check.png", dpi=90); print("saved")
