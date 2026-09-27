"""One A–G viewer PNG per c2_full task using the L6c (or any c2) checkpoint."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

from analysis.c2_adapter import predict_c2_sample
from analysis.geometry import C0Geometry
from analysis.load_c2_sample import load_c2_worldsnap_sample
from analysis.load_model import load_vit_checkpoint
from analysis.render_c2 import render_c2_sample
from spatial_data.c2_tasks import C2_FULL_HELD_OUT, HUMAN_NAMES, load_task_stats, task_keys_for_set


def _safe_name(key: str) -> str:
    return re.sub(r"[^\w.+_-]+", "_", key)


def _pick_crop(task_key: str, *, split: str, cropset: str, grid: int, n_crops: int) -> int:
    """Val crop with rich fg (scan until a decent mask, else best seen)."""
    best_i, best_n = 0, 0
    for ci in range(n_crops):
        s = load_c2_worldsnap_sample(
            split=split, crop_index=ci, task_key=task_key, cropset=cropset, grid_size=grid,
        )
        n = int((s.target > 0).sum().item())
        if n > best_n:
            best_n, best_i = n, ci
        if n >= 80:
            return ci
    return best_i


def main() -> None:
    p = argparse.ArgumentParser(description="Generate viewer PNGs for each c2_full task")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=REPO / "runs/checkpoints/c2_L6c_c2_full_2k.pt",
    )
    p.add_argument("--out-dir", type=Path, default=REPO / "runs/diagnostics/l6c_task_gallery")
    p.add_argument("--cropset", default="v3b")
    p.add_argument("--split", default="val", choices=("val", "train", "test"))
    p.add_argument("--n-crops", type=int, default=250, help="crop indices 0..n-1 for auto-pick")
    p.add_argument("--crop", type=int, default=None, help="fixed crop index for every task")
    p.add_argument("--use-gpu", action="store_true", default=True)
    p.add_argument("--tasks", default="", help="comma-separated subset; default all c2_full")
    args = p.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_vit_checkpoint(args.ckpt, device=device)
    crops_dir = REPO / "data/amsterdam/de_pijp/crops" / args.cropset
    stats = load_task_stats(crops_dir)
    keys = task_keys_for_set("c2_full", stats)
    if args.tasks.strip():
        keys = [k.strip() for k in args.tasks.split(",") if k.strip()]
    key_to_ix = {k: i for i, k in enumerate(task_keys_for_set("c2_full", stats))}
    hold = set(C2_FULL_HELD_OUT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    geom = C0Geometry.from_config(cfg)
    ckpt_stem = args.ckpt.stem

    print(f"tasks={len(keys)} ckpt={args.ckpt.name} device={device} out={args.out_dir}")
    for key in keys:
        crop_i = args.crop if args.crop is not None else _pick_crop(
            key, split=args.split, cropset=args.cropset, grid=cfg.grid_out_size, n_crops=args.n_crops,
        )
        sample = load_c2_worldsnap_sample(
            split=args.split, crop_index=crop_i, task_key=key, cropset=args.cropset,
            grid_size=cfg.grid_out_size,
        )
        ti = key_to_ix[key]
        _, prob, err, metrics = predict_c2_sample(model, sample, cfg, device=device, task_index=ti)
        label = HUMAN_NAMES.get(key, key)
        tag = "HELD-OUT" if key in hold else "trained"
        fig = render_c2_sample(
            sample, geom, cfg=cfg, prob=prob, err_rgb=err, metrics=metrics,
            ckpt_label=ckpt_stem, title_suffix=f"{label}  [{tag}]  crop={crop_i}",
        )
        out = args.out_dir / f"{_safe_name(key)}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {out.name}  IoU={metrics['iou']:.3f}  crop={crop_i}  {tag}")
    print(f"done → {args.out_dir}")


if __name__ == "__main__":
    main()
