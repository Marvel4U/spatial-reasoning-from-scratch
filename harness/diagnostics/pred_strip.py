"""Tier 2(c): prediction PNG strip at CE eval (8 fixed probe samples)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
import data
from plain_gpt_module import unwrap_compiled
from spatial_batch import forward as model_forward

from .probe_batch import ProbeBatch

_c4_rgb_district: torch.Tensor | None = None


def _c4_rgb_window(
    flat_ix: int,
    device: torch.device,
    val_loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _c4_rgb_district
    from spatial_data.dataset_c4_gpu import _cut_windows, load_district_rgb_planes_v4

    if _c4_rgb_district is None or _c4_rgb_district.device != device:
        crops_dir = config.district_data_dir() / "crops" / config.cropset
        _c4_rgb_district = load_district_rgb_planes_v4(crops_dir, device)
    origin = val_loader.route["origin"][flat_ix : flat_ix + 1]
    win = _cut_windows(_c4_rgb_district, origin)[0].cpu().numpy()
    return win[0], win[1], win[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@torch.no_grad()
def save_pred_strip(
    model,
    probe: ProbeBatch,
    *,
    step: int,
    out_dir: Path,
    run_id: str,
    c4_val_loader=None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> Path:
    core = unwrap_compiled(model)
    cfg = core.config
    was_training = model.training
    model.eval()
    device = probe.img.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    strip_dir = Path(out_dir)
    strip_dir.mkdir(parents=True, exist_ok=True)
    path = strip_dir / f"pred_step{step:06d}{filename_suffix}.png"
    c4_vl = c4_val_loader if c4_val_loader is not None else data.val_loader

    rows = len(probe.pred_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 2.2 * rows), dpi=100)
    if rows == 1:
        axes = np.array([axes])
    with torch.autocast(device_type=device_type, dtype=data.dtype):
        logits, _ = model_forward(model, probe.img, None, probe.task_ids, probe.cond_ids)
    fg = cfg.foreground_class
    if cfg.num_classes > 1:
        prob = torch.sigmoid(logits[..., fg] - logits[..., 0])
    else:
        prob = F.softmax(logits, dim=-1)[..., 0]

    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from analysis.load_sample import load_synthetic_sample, load_worldsnap_sample
    from analysis.rgb import rgb_from_label_planes

    for row, pi in enumerate(probe.pred_indices):
        crop_ix = probe.item_indices[pi]
        probe_slot = pi
        if config.data_source == "synthetic":
            sample = load_synthetic_sample(split="val", item_index=crop_ix, seed=config.seed)
        elif config.task_rung == "c1":
            from analysis.load_c1_sample import load_c1_worldsnap_sample
            tkey = config.c1_task_key
            if config.c1_is_multi_task():
                keys = config.c1_task_keys_active()
                tkey = keys[probe_slot % len(keys)]
            sample = load_c1_worldsnap_sample(
                split="val", crop_index=crop_ix, task_key=tkey,
                encoding=config.encoding_mode, img_size=config.img_size,
                grid_size=config.grid_out_size,
            )
        elif config.task_rung == "c3":
            from analysis.load_c3_sample import load_c3_worldsnap_sample
            sample = load_c3_worldsnap_sample(
                split="val", crop_index=crop_ix, task_key=config.c3_eval_key(),
                encoding=config.encoding_mode, img_size=config.img_size,
                grid_size=config.grid_out_size, cropset=config.cropset,
            )
        elif config.task_rung == "c4":
            ax0, ax1, ax2 = axes[row]
            pi_row = probe.pred_indices[row]
            n4, surf, estab = _c4_rgb_window(probe.item_indices[pi_row], device, c4_vl)
            ax0.imshow(rgb_from_label_planes(n4, surf, estab), interpolation="nearest")
            mc0, mr0 = int(probe.marker_col[pi_row]), int(probe.marker_row[pi_row])
            ax0.plot(mc0, mr0, "yo", ms=6, mew=1.2)
            ax0.set_title(f"val[{crop_ix}] v4 RGB (noise/surf/estab)", fontsize=8)
            ax0.set_xticks([])
            ax0.set_yticks([])
            tgt = probe.tgt[pi_row].cpu().numpy()
            ax1.imshow((tgt > 0).astype(np.float32), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax1.set_title("target", fontsize=8)
            ax1.set_xticks([])
            ax1.set_yticks([])
            p = prob[pi_row].float().cpu().numpy()
            ax2.imshow(p, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax2.set_title("P(fg)", fontsize=8)
            ax2.set_xticks([])
            ax2.set_yticks([])
            continue
        else:
            sample = load_worldsnap_sample(
                split="val", item_index=crop_ix, encoding=config.encoding_mode,
                img_size=config.img_size, grid_size=config.grid_out_size,
            )
        ax0, ax1, ax2 = axes[row]
        if sample.label_planes is not None:
            ax0.imshow(rgb_from_label_planes(*sample.label_planes), interpolation="nearest")
        else:
            ax0.imshow(sample.img[:3].permute(1, 2, 0).numpy(), interpolation="nearest")
        show_marker = config.task_rung not in ("c1", "c2", "c3")
        if config.task_rung == "c3":
            from spatial_data.c3_tasks import get_c3_task
            show_marker = get_c3_task(config.c3_eval_key()).show_marker_plane
        if show_marker:
            mr, mc = sample.meta.marker_row, sample.meta.marker_col
            ax0.plot(mc, mr, "+", color="yellow", ms=8, mew=1.5)
        ax0.set_title(f"i={crop_ix}", fontsize=8)
        ax0.set_xticks([])
        ax0.set_yticks([])
        tgt = probe.tgt[pi].cpu().numpy()
        ax1.imshow((tgt > 0).astype(np.float32), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("target", fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        p = prob[pi].float().cpu().numpy()
        ax2.imshow(p, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title("P(fg)", fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.suptitle(f"{run_id} @ step {step}{title_suffix}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if was_training:
        model.train()
    return path
