"""Read-only baseline (C2_PLAN §7): how far does "just call c1 twice" get on a c2 conjunction?

Take a trained c1 multi-task checkpoint, predict the two single-condition masks of a c2
core conjunction, AND them at cell level, and score against the true c2 target (pixel-AND
then majority) on all 250 val crops. Two limits are printed next to it: the definitional
ceiling (AND of the two TRUE c1 majority masks, i.e. what a perfect c1 pair could reach)
and each part's own IoU, so a low score can be read as compounding vs. definition.

Usage:  python docs/plans/perf_prototypes/c2_composition_baseline.py c1_L5c_factorised_2k_v3b@v3b
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from plain_gpt_module.local_grid_vit import LocalGridViT  # noqa: E402
from plain_gpt_module.local_config import LocalGridViTConfig  # noqa: E402
from spatial_data import c1_tasks as T  # noqa: E402
from spatial_data.c2_tasks import ATOM_KEYS, C2_CORE_KEYS, atom_key, get_c2_task  # noqa: E402
from spatial_data.dataset_c1_gpu import build_image_from_crops  # noqa: E402
from spatial_data.dataset_c2_gpu import c2_targets_from_planes, load_crop_planes  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK = 50


def iou(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Per-sample IoU averaged over samples with a non-empty target (project convention)."""
    inter = (pred & tgt).flatten(1).sum(1).float()
    union = (pred | tgt).flatten(1).sum(1).float()
    ne = tgt.flatten(1).any(1)
    if not bool(ne.any()):
        return float("nan")
    return float((inter[ne] / union[ne].clamp_min(1)).mean().item())


def load_checkpoint(name: str):
    ck = torch.load(REPO / "runs/checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    fields = LocalGridViTConfig.__dataclass_fields__
    cfg = LocalGridViTConfig(**{k: v for k, v in ck["model_config"].items() if k in fields})
    model = LocalGridViT(cfg)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()},
                          strict=False)
    model.train(False).to(DEV)
    mode = "factorised_twelve" if cfg.c1_task_conditioning == "factorised" else "core_four"
    enc = (ck.get("extra") or {}).get("encoding_mode", "scalar")
    return model, cfg, T.task_keys_for_mode(mode), enc


@torch.no_grad()
def part_probabilities(model, cfg, crops, enc, task_ix):
    """P(fg) per cell for one c1 task over every val crop -> (N, G, G)."""
    out = []
    for i in range(0, crops.shape[0], CHUNK):
        c = crops[i:i + CHUNK]
        img = build_image_from_crops(c, enc)
        tid = torch.full((c.shape[0],), task_ix, device=DEV, dtype=torch.long)
        with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            res = model(img, task_id=tid)
        logits = res[0] if isinstance(res, tuple) else res
        out.append(torch.softmax(logits.float(), -1)[..., 1])
    return torch.cat(out)


def true_majority(crops: torch.Tensor, aid: int) -> torch.Tensor:
    cond = torch.tensor([[aid, -1]] * crops.shape[0], dtype=torch.long, device=crops.device)
    return c2_targets_from_planes(crops, cond) == 1


def run(name: str, cropset: str) -> None:
    model, cfg, c1_keys, enc = load_checkpoint(name)
    crops = load_crop_planes(REPO / f"data/amsterdam/de_pijp/crops/{cropset}", "val", DEV)
    weights = cfg.c1_per_task_fg_weights
    print(f"\n##### {name} on {cropset} (encoding={enc}, {crops.shape[0]} val crops)")
    print(f"c1 tasks in checkpoint: {c1_keys}")
    expressible = [k for k in C2_CORE_KEYS
                   if all(atom_key(a) in c1_keys for a in get_c2_task(k).atoms)]
    missing = [k for k in C2_CORE_KEYS if k not in expressible]
    print(f"expressible c2 core conjunctions: {expressible or 'none'}")
    if missing:
        print(f"not expressible (a part is not a task of this checkpoint): {missing}")
    if not expressible:
        return
    print(f"\n{'conjunction':22s} {'partA':>6s} {'partB':>6s} {'AND@0.5':>8s} {'AND@rule':>9s} "
          f"{'ceiling':>8s}")
    for key in expressible:
        spec = get_c2_task(key)
        cond = torch.tensor([spec.padded_cond_ids()] * crops.shape[0], dtype=torch.long, device=DEV)
        tgt = c2_targets_from_planes(crops, cond) == 1
        masks_plain, masks_rule, part_iou = [], [], []
        for a in spec.atoms:
            ti = c1_keys.index(atom_key(a))
            prob = part_probabilities(model, cfg, crops, enc, ti)
            w = weights[ti] if weights is not None else 1.0
            masks_plain.append(prob > 0.5)
            masks_rule.append(prob > w / (1.0 + w))
            part_iou.append(iou(prob > w / (1.0 + w), true_majority(crops, a)))
        ceiling = true_majority(crops, spec.atoms[0]) & true_majority(crops, spec.atoms[1])
        print(f"{key:22s} {part_iou[0]:6.3f} {part_iou[1]:6.3f} "
              f"{iou(masks_plain[0] & masks_plain[1], tgt):8.3f} "
              f"{iou(masks_rule[0] & masks_rule[1], tgt):9.3f} {iou(ceiling, tgt):8.3f}")
    print("partA/partB = each c1 mask vs its own true majority mask (rule cut-off); "
          "ceiling = AND of the two TRUE majority masks vs the c2 target (definition gap only).")


if __name__ == "__main__":
    args = sys.argv[1:] or ["c1_L5c_factorised_2k_v3b@v3b"]
    for arg in args:
        n, cs = arg.split("@")
        run(n, cs)
