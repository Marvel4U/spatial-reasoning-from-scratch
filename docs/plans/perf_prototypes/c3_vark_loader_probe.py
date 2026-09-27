import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led["within_20m_rand_varK_relV2"]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data
data.setup_device(); data.load_data()
tl = data.train_loader
print("loader n_markers", tl.n_markers, "random", tl.n_markers_random, "min_dist_px", tl.marker_min_dist_px)
img, tgt, tix, cond = tl.next_batch()
mp = img[:, -1]
print("img shape", tuple(img.shape), "marker plane sum per sample:", mp.flatten(1).sum(1).tolist())
print("marker at corner (255,255) per sample:", mp[:, 255, 255].tolist(), " (0,0):", mp[:, 0, 0].tolist(), " row 255 sum:", mp[:, 255, :].sum(1).tolist())
print("target fg share per sample:", [round(x, 4) for x in tgt.float().flatten(1).mean(1).tolist()])
print("target unique:", tgt.unique().tolist())
# where are the marker pixels vs target discs? count target fg pixels within 20 px of each marker
