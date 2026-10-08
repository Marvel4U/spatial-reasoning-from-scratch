"""End-to-end check of the C5 rays task through the real c4 loaders (validation store), with a picture."""
import sys, os, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spatial_data import channels as ch
from spatial_data.c4_tasks import c4_task_specs_for_keys, get_c4_task
from spatial_data.dataset_c4_gpu import (load_district_planes_v4, load_c4_route_index, C4GpuValLoader,
                                         C4MultiTaskGpuTrainLoader)
dev = torch.device("cuda"); ch.MARKER_RADIUS_PX = 3
crops = REPO / "data/amsterdam/de_pijp/crops/v4"; store = crops / "c4_100k"
district = load_district_planes_v4(crops, dev)
val = load_c4_route_index(store, "val", dev)
print("val routes", val["origin"].shape[0], "fans loaded:", "fan_xy" in val, tuple(val["fan_xy"].shape))
vl = C4GpuValLoader(get_c4_task("rays"), "onehot_v4_sp", 64, 16, district=district, route_index=val)
img, tgt, tix, cond, meta = vl.next_batch_with_meta()
print("img", tuple(img.shape), "scratch channel sum", float(img[:, 16].sum()), "marker channel sum", float(img[:, 15].sum()),
      "| ray cells per sample", tgt.flatten(1).sum(1).tolist()[:8], "| cond", cond[0].tolist())
# mixed train loader: rays + segment, timing
tl = C4MultiTaskGpuTrainLoader(["segment", "rays"], "onehot_v4_sp", 64, 128, 1, district=district, route_index=val)
for _ in range(3): tl.next_batch()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): im, tg, ti, cd = tl.next_batch()
torch.cuda.synchronize(); print(f"mixed segment+rays loader, batch 128: {(time.perf_counter() - t) / 20 * 1000:.1f} ms; tasks {torch.bincount(ti).tolist()}")
# picture: planes, rays target
fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for k in range(4):
    bld = img[k, 10].cpu().numpy(); mk = img[k, 15].cpu().numpy()
    axes[0, k].imshow(bld, cmap="Greys", vmin=0, vmax=1.6); axes[0, k].imshow(np.ma.masked_where(mk == 0, mk), cmap="autumn")
    axes[0, k].set_title(f"buildings + markers #{k}"); axes[1, k].imshow(tgt[k].cpu().numpy(), cmap="Blues", vmin=0, vmax=1.2)
    axes[1, k].set_title(f"rays target, {int(tgt[k].sum())} cells")
    for a in axes[:, k]: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("/tmp/c5_rays_loader_check.png", dpi=90); print("saved")
