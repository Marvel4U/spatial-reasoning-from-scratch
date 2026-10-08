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
_c4_rgb_district_key: tuple | None = None    # (crops dir, device): a queue may switch districts between runs


def _c4_route_ix(val_loader, index: int) -> int:
    """``item_indices`` are route rows; legacy probes stored flat val index (route × task)."""
    n_tasks = int(getattr(val_loader, "n_tasks", 0) or 0)
    n_routes = int(getattr(val_loader, "n_routes", 0) or 0)
    if n_tasks > 1 and n_routes and len(val_loader) == n_routes * n_tasks and index >= n_routes:
        return index // n_tasks
    return index


def _c4_rgb_window(
    route_ix: int,
    device: torch.device,
    val_loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _c4_rgb_district, _c4_rgb_district_key
    from spatial_data.dataset_c4_gpu import _cut_windows, load_district_rgb_planes_v4

    crops_dir = config.district_data_dir() / "crops" / config.cropset
    key = (str(crops_dir), str(device))
    if _c4_rgb_district is None or _c4_rgb_district_key != key:    # keyed on the district too: a queue of runs on
        _c4_rgb_district = load_district_rgb_planes_v4(crops_dir, device)   # different districts shares this process
        _c4_rgb_district_key = key
    ri = _c4_route_ix(val_loader, route_ix)
    origin = val_loader.route["origin"][ri : ri + 1]
    win = _cut_windows(_c4_rgb_district, origin, int(getattr(val_loader, "crop", 256)))[0].cpu().numpy()
    return win[0], win[1], win[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _length_bin_label(bin_idx: int) -> str:
    """Human-readable 10 m length bin (C5 token target)."""
    from spatial_data.c4_tasks import LENGTH_BIN_M, N_LENGTH_BINS

    if bin_idx < 0:
        return "—"
    lo = bin_idx * LENGTH_BIN_M
    if bin_idx >= N_LENGTH_BINS - 1:
        return f"bin {bin_idx} (≥{lo:.0f} m)"
    return f"bin {bin_idx} ({lo:.0f}–{lo + LENGTH_BIN_M:.0f} m)"


def _length_match_label(gt_bin: int, pred_bin: int) -> str:
    if gt_bin < 0 or pred_bin < 0:
        return ""
    if pred_bin == gt_bin:
        return "exact"
    if abs(pred_bin - gt_bin) <= 1:
        return "±1 bin"
    return "miss"


def _scratchpad_plane(im_row: torch.Tensor) -> np.ndarray:
    """Scratchpad channel (C5 length input), thickened for visibility."""
    if im_row.shape[0] <= 16:
        return np.zeros(im_row.shape[1:], dtype=np.float32)
    sp = im_row[16].float()
    k = 5
    thick = F.max_pool2d(sp[None, None], k, stride=1, padding=k // 2)[0, 0]
    return thick.cpu().numpy()


def _length_readout_axes(ax, *, gt_bin: int, pred_bin: int) -> None:
    gt_len = _length_bin_label(gt_bin)
    pred_len = _length_bin_label(pred_bin)
    flag = _length_match_label(gt_bin, pred_bin)
    ax.set_axis_off()
    ax.text(
        0.5, 0.55, f"GT length\n{gt_len}",
        transform=ax.transAxes, ha="center", va="center", fontsize=9,
    )
    ax.text(
        0.5, 0.2, f"Pred length\n{pred_len}\n({flag})",
        transform=ax.transAxes, ha="center", va="center", fontsize=9,
    )


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
        tok_pred = None
        if config.task_rung == "c4" and getattr(probe, "tok_tgt", None) is not None and bool((probe.tok_tgt >= 0).any()):
            from eval_c4 import token_logits
            tl = token_logits(model, probe.img, probe.task_ids, probe.cond_ids)
            tok_pred = tl.argmax(dim=-1).cpu().numpy() if tl is not None else None
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
            from spatial_data.c4_tasks import IGNORE_TARGET

            ax0, ax1, ax2 = axes[row]
            pi_row = probe.pred_indices[row]
            n4, surf, estab = _c4_rgb_window(probe.item_indices[pi_row], device, c4_vl)
            ax0.imshow(rgb_from_label_planes(n4, surf, estab), interpolation="nearest")
            im = probe.img[pi_row]
            thick = lambda a, k=5: F.max_pool2d(a.float()[None, None], k, stride=1, padding=k // 2)[0, 0].cpu().numpy() > 0
            ov = np.zeros(im.shape[1:] + (4,), dtype=np.float32)
            if im.shape[0] > 16:
                ov[thick(im[16] > 0)] = (0.15, 0.45, 1.0, 1.0)
            ov[thick(im[15] > 0, 9)] = (1.0, 0.9, 0.0, 1.0)
            ax0.imshow(ov, interpolation="nearest")
            keys = config.c4_keys_active()
            if hasattr(c4_vl, "spec") and not hasattr(c4_vl, "task_specs"):
                keys = [c4_vl.spec.key]
            tid = int(probe.task_ids[pi_row]) if probe.task_ids is not None else 0
            tname = keys[tid] if tid < len(keys) else f"task {tid}"
            ax0.set_title(f"route {crop_ix} · {tname}", fontsize=8)
            ax0.set_xticks([])
            ax0.set_yticks([])
            tgt_np = probe.tgt[pi_row].cpu().numpy()
            tt = int(probe.tok_tgt[pi_row]) if getattr(probe, "tok_tgt", None) is not None else -1
            length_row = tt >= 0 and tname.startswith("length")          # the length layout only for the length tasks
            state_names = {37: "SEARCH", 38: "BACK", 39: "DRAW", 40: "DONE"}   # C5 R5 state tokens (spatial_data.c4_tasks)
            state_row = tt in state_names
            if length_row:
                sp = _scratchpad_plane(im)
                ax1.imshow(sp, cmap="Blues", vmin=0, vmax=max(float(sp.max()), 1e-6), interpolation="nearest")
                ax1.set_title("scratchpad (length input)", fontsize=8)
            else:
                ax1.imshow((tgt_np > 0).astype(np.float32), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                ax1.set_title("mask target", fontsize=8)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if length_row and tok_pred is not None:
                _length_readout_axes(ax2, gt_bin=tt, pred_bin=int(tok_pred[pi_row]))
            else:
                p = prob[pi_row].float().cpu().numpy()
                ax2.imshow(p, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                if state_row:
                    tp = int(tok_pred[pi_row]) if tok_pred is not None else -1
                    ax2.set_title(f"P(fg); token out: {state_names[tt]}, pred {state_names.get(tp, tp)}"
                                  + (" (ok)" if tp == tt else " (miss)"), fontsize=8)
                else:
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

    legend = "  (input: markers yellow, scratchpad blue)" if config.task_rung == "c4" else ""
    fig.suptitle(f"{run_id} @ step {step}{title_suffix}{legend}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if was_training:
        model.train()
    return path
