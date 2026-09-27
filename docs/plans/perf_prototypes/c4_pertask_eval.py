"""Per-task val metrics (every val route x every trained task) for c4 multi-task checkpoints."""
import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
RUN = sys.argv[1]
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.c4_eval_task_key = "all"; config.verbose = False
import data, grid_vit_adapter as A, eval_c4
data.setup_device(); data.load_data()
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt"); model.train(False)
r = eval_c4.eval_loader_c4(model, data.val_loader)
keys = r["c4_tasks"]
print(f"{RUN}: val routes x tasks = {r['n_grids']}")
for tk in keys:
    print(f"  {tk:9s} plain IoU {r[f'iou_{tk}']:.3f}  dilated {r[f'iou_dilated_{tk}']:.3f}  prec1 {r[f'precision_1cell_{tk}']:.3f}  rec1 {r[f'recall_1cell_{tk}']:.3f}")
