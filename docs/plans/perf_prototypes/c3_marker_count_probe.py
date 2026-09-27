"""Score E5b checkpoints on train-distribution batches with K markers vs the fixed 1-marker val set."""
import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
RUNS = ["c3_within_20m_8k_rand4","within_20m_rand4_relV1","within_20m_rand4_relV2",
        "within_50m_6k_rand2","within_50m_rand2_relV1","within_50m_rand2_relV2"]
import importlib
for run in RUNS:
    importlib.reload(config)
    for k, v in led[run]["overrides"].items():
        if hasattr(config, k): setattr(config, k, v)
    config.verbose = False
    for m in ("data","grid_vit_adapter","eval_spatial"):
        sys.modules.pop(m, None)
    import data, grid_vit_adapter as A, eval_spatial as E
    data.setup_device(); data.load_data()
    model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{run}.pt"); model.train(False)
    tl = data.train_loader
    out = {}
    with torch.no_grad():
        out["val_1marker"] = E.eval_loader_c3(model, data.val_loader, max_batches=20)
        for K in (1, 2, 4):
            tl.n_markers = K
            torch.manual_seed(0)
            out[f"train_K{K}"] = E.eval_loader_c3(model, tl, max_batches=20)
    print(f"\n## {run}  (trained with n_markers={config.c3_n_markers}, rel_pos_bias={getattr(config,'rel_pos_bias','none')})")
    for name, r in out.items():
        iou = r.get("IoU_fg", r.get("iou_fg")); keys = [k for k in r if "iou" in k.lower()][:3]
        print(f"  {name:14s} " + "  ".join(f"{k}={r[k]:.3f}" for k in keys if isinstance(r[k], float)))
