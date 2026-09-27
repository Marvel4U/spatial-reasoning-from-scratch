"""Read-only: does ONE analytic rule fix the over-marking for every task?
Rule: training used positive weight w per task; weighted CE shifts the fg log-odds by log(w).
Undo it at decision time: mark a cell if  logit_fg - logit_bg - log(w) > 0   (same as P(fg) > w/(1+w)).
Compared with the plain rule (P > 0.5) and with the best threshold found by a sweep (oracle)."""
import sys, os, glob, math
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3] if "perf_prototypes" in __file__ else Path.home() / "Github/spatial_reasoning_LLM_artifact"
sys.path.insert(0, str(REPO))
from plain_gpt_module.local_grid_vit import LocalGridViT
from plain_gpt_module.local_config import LocalGridViTConfig
from spatial_data import c1_tasks as T
from spatial_data.dataset_c1_gpu import build_batch_from_crops

dev = "cuda"
THR = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]


def iou_at(P, TG, nonempty, thr):
    pr = P > thr
    inter = (pr & TG).flatten(1).sum(1).float(); union = (pr | TG).flatten(1).sum(1).float()
    tp = (pr & TG).sum().item(); fp = (pr & ~TG).sum().item(); fn = (~pr & TG).sum().item()
    efp = pr[~nonempty].flatten(1).any(1).float().mean().item() if (~nonempty).any() else float("nan")
    return (inter[nonempty] / union[nonempty].clamp_min(1)).mean().item(), tp / max(1, tp + fp), tp / max(1, tp + fn), efp


@torch.no_grad()
def run(name, cropset):
    paths = sorted(glob.glob(str(REPO / f"data/amsterdam/de_pijp/crops/{cropset}/val/*_labels.npz")))
    crops = torch.from_numpy(np.stack([np.stack([np.load(p)[k] for k in ("noise_256", "surface_256", "estab_256")], 0)
                                       for p in paths]).astype(np.uint8)).to(dev)
    ck = torch.load(REPO / "runs/checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    fields = LocalGridViTConfig.__dataclass_fields__
    cfg = LocalGridViTConfig(**{k: v for k, v in ck["model_config"].items() if k in fields})
    model = LocalGridViT(cfg)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}, strict=False)
    model.train(False).to(dev)
    keys = T.task_keys_for_mode("factorised_twelve" if cfg.c1_task_conditioning == "factorised" else "core_four")
    enc = (ck.get("extra") or {}).get("encoding_mode", "scalar")
    print(f"\n##### {name} on {cropset}")
    print(f"{'task':12s} {'w':>5s} {'rule thr':>8s} | {'IoU plain':>9s} {'IoU rule':>8s} {'IoU oracle':>10s} (thr) | P/R plain   P/R rule   | emptyFP plain -> rule")
    rows = []
    for ti, key in enumerate(keys):
        spec = T.get_c1_task(key, cropset)
        P, TG = [], []
        for i in range(0, crops.shape[0], 50):
            c = crops[i:i + 50]
            img, tgt = build_batch_from_crops(c, None, spec, encoding=enc, grid_size=cfg.grid_out_size, img_size=cfg.img_size)
            tid = torch.full((c.shape[0],), ti, device=dev, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img, task_id=tid)
            logits = out[0] if isinstance(out, tuple) else out
            P.append(torch.softmax(logits.float(), -1)[..., 1]); TG.append(tgt == 1)
        P, TG = torch.cat(P), torch.cat(TG); ne = TG.flatten(1).any(1)
        w = cfg.c1_per_task_fg_weights[ti]
        rule = w / (1 + w)
        plain, corr = iou_at(P, TG, ne, 0.5), iou_at(P, TG, ne, rule)
        sweep = [(iou_at(P, TG, ne, t)[0], t) for t in THR]; best = max(sweep)
        rows.append((key, plain[0], corr[0], best[0]))
        print(f"{key:12s} {w:5.1f} {rule:8.3f} | {plain[0]:9.3f} {corr[0]:8.3f} {best[0]:10.3f} ({best[1]:<5}) | {plain[1]:.2f}/{plain[2]:.2f}   {corr[1]:.2f}/{corr[2]:.2f}  | {plain[3]:.2f} -> {corr[3]:.2f}")
    trained = [r for r in rows if r[0] not in ("surface_0", "estab_2", "estab_0")]
    print(f"mean over trained, non-degenerate tasks: plain {np.mean([r[1] for r in trained]):.3f} | rule {np.mean([r[2] for r in trained]):.3f} | oracle {np.mean([r[3] for r in trained]):.3f}")


for arg in sys.argv[1:]:
    n, cs = arg.split("@")
    run(n, cs)
