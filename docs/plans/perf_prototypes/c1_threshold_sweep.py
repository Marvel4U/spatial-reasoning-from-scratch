"""Read-only: sweep the decision threshold on P(fg) for a c1 checkpoint, per task, on ALL val crops.

Question: are the establishment tasks bad because the loss weight biases the model to over-mark
(then IoU recovers at a higher threshold), or because the model cannot tell the classes apart
(then IoU stays low at every threshold)?
Usage (from repo root):  .venv/bin/python <this file> <checkpoint name without .pt> [<more>...]
"""
import sys, glob
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from plain_gpt_module.local_grid_vit import LocalGridViT
from plain_gpt_module.local_config import LocalGridViTConfig
from spatial_data import c1_tasks as T
from spatial_data.dataset_c1_gpu import build_batch_from_crops

dev = "cuda"
import os
CROPSET = os.environ.get("CROPSET", "v2")
paths = sorted(glob.glob(str(REPO / f"data/amsterdam/de_pijp/crops/{CROPSET}/val/*_labels.npz")))
planes = []
for p in paths:
    z = np.load(p)
    planes.append(np.stack([z["noise_256"], z["surface_256"], z["estab_256"]], 0))
crops = torch.from_numpy(np.stack(planes).astype(np.uint8)).to(dev)          # (250, 3, 256, 256)
print(f"cropset {CROPSET} val crops: {tuple(crops.shape)}")
THR = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def task_keys(cfg):
    if cfg.c1_task_conditioning == "factorised":
        return T.task_keys_for_mode("factorised_twelve")
    return T.task_keys_for_mode("core_four")


@torch.no_grad()
def run(name):
    ck = torch.load(REPO / "runs/checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    fields = LocalGridViTConfig.__dataclass_fields__
    cfg = LocalGridViTConfig(**{k: v for k, v in ck["model_config"].items() if k in fields})
    model = LocalGridViT(cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
    missing = model.load_state_dict(sd, strict=False)
    model.train(False).to(dev)
    keys = task_keys(cfg)
    enc = (ck.get("extra") or {}).get("encoding_mode", "scalar")
    print(f"\n##### {name}  tasks={len(keys)} enc={enc} load={missing}")
    print(f"{'task':12s} {'w_fg':>6s} {'nonempty':>8s} | " + " ".join(f"thr{t:<5}" for t in THR) + " | best  P/R@0.5  P/R@best  emptyFP@0.5 -> @best")
    for ti, key in enumerate(keys):
        spec = T.get_c1_task(key, CROPSET)
        P, TG = [], []
        for i in range(0, crops.shape[0], 50):
            c = crops[i:i + 50]
            img, tgt = build_batch_from_crops(c, None, spec, encoding=enc, grid_size=cfg.grid_out_size, img_size=cfg.img_size)
            tid = torch.full((c.shape[0],), ti, device=dev, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img, task_id=tid)
            logits = out[0] if isinstance(out, tuple) else out
            P.append(torch.softmax(logits.float(), -1)[..., 1]); TG.append(tgt == 1)
        P = torch.cat(P); TG = torch.cat(TG)                                  # (250, 64, 64)
        nonempty = TG.flatten(1).any(1)
        res = []
        for t in THR:
            pr = P > t
            inter = (pr & TG).flatten(1).sum(1).float(); union = (pr | TG).flatten(1).sum(1).float()
            iou = (inter[nonempty] / union[nonempty].clamp_min(1)).mean().item() if nonempty.any() else float("nan")
            tp = (pr & TG).sum().item(); fp = (pr & ~TG).sum().item(); fn = (~pr & TG).sum().item()
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            efp = pr[~nonempty].flatten(1).any(1).float().mean().item() if (~nonempty).any() else float("nan")
            res.append((iou, prec, rec, efp))
        b = int(np.nanargmax([r[0] for r in res]))
        w = cfg.c1_per_task_fg_weights[ti] if cfg.c1_per_task_fg_weights else 1.0
        print(f"{key:12s} {w:6.1f} {int(nonempty.sum()):8d} | " + " ".join(f"{r[0]:8.3f}" for r in res)
              + f" | {THR[b]:<5} {res[0][1]:.2f}/{res[0][2]:.2f}  {res[b][1]:.2f}/{res[b][2]:.2f}   {res[0][3]:.2f} -> {res[b][3]:.2f}")


for n in sys.argv[1:]:
    run(n)
