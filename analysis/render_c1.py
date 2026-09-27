"""c1 mask task viewer — panels A–G (ANALYSIS_SUITE_SPEC §13, C1_PLAN §9)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from plain_gpt_module.local_config import LocalGridViTConfig

from .geometry import C0Geometry
from .load_c1_sample import C1LoadedSample
from .render import (
    _channel_mosaic,
    _draw_patch_grid,
    _imshow_grid,
    _outline_patch_block,
    _outline_subcell,
)
from .rgb import rgb_from_label_planes


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _grid_to_image_px(gr: int, gc: int, geom: C0Geometry) -> tuple[int, int]:
    h, g = geom.img_size, geom.grid_size
    return gr * h // g, gc * h // g


def _panel_a_c1(ax, sample: C1LoadedSample, geom: C0Geometry) -> None:
    planes = sample.label_planes
    if planes is None:
        raise ValueError("panel A needs label_planes")
    rgb = rgb_from_label_planes(*planes)
    ax.imshow(rgb, interpolation="nearest")
    _draw_patch_grid(ax, geom.img_size, geom.patch_size)
    ax.set_title("A: RGB + patch grid (marker off)", fontsize=9)
    ax.set_xlim(-0.5, geom.img_size - 0.5)
    ax.set_ylim(geom.img_size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_b_c1(ax, sample: C1LoadedSample) -> None:
    img = sample.img.numpy()
    n = img.shape[0]
    mosaic, h, w = _channel_mosaic(img)
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for k in range(1, n):
        ax.axvline(k * w - 0.5, color="white", lw=0.8)
    label = "noise | surface | estab | marker(0)" if n == 4 else " | ".join(f"ch{i}" for i in range(n))
    ax.set_title(f"B: {label}", fontsize=9)
    ax.set_xlim(-0.5, n * w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_c_c1(ax, sample: C1LoadedSample, geom: C0Geometry, gr: int, gc: int) -> None:
    mr, mc = _grid_to_image_px(gr, gc, geom)
    pr, pc = geom.patch_index(mr, mc)
    rs, cs = geom.patch_pixel_bounds(pr, pc)
    img = sample.img.numpy()
    n = img.shape[0]
    patches = [img[ch, rs, cs] for ch in range(n)]
    mosaic, _, _ = _channel_mosaic(np.stack(patches, axis=0))
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    p = geom.patch_size
    for k in range(1, n):
        ax.axvline(k * p - 0.5, color="white", lw=0.8)
    s = geom.subcells
    for k in range(1, s):
        ax.axhline(k * p / s - 0.5, color="white", lw=0.5, alpha=0.7)
        ax.axvline(k * p / s - 0.5, color="white", lw=0.5, alpha=0.7)
    sub_r, sub_c = geom.subcell_in_patch(gr, gc)
    _outline_subcell(ax, slice(0, p), slice(0, p), sub_r, sub_c, s, color="lime")
    ax.set_title(f"C: focal patch P={p} (centroid cell), s={s}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_d_c1(ax, target: torch.Tensor, geom: C0Geometry, gr: int, gc: int) -> None:
    t = _np(target)
    fg = (t > 0).astype(np.float32)
    _imshow_grid(ax, fg, title="D: target mask", cmap="viridis", vmin=0, vmax=1)
    br, bc = geom.patch_block(gr, gc)
    _outline_patch_block(ax, br, bc, geom.subcells, geom.grid_size)


def _panel_e_c1(ax, prob: torch.Tensor | None) -> None:
    if prob is None:
        ax.axis("off")
        ax.set_title("E: (no model)", fontsize=9)
        return
    _imshow_grid(ax, _np(prob), title="E: P(fg)", cmap="magma", vmin=0.0, vmax=1.0)


def _panel_error(ax, err_rgb: torch.Tensor | None) -> None:
    if err_rgb is None:
        ax.axis("off")
        ax.set_title("error map (needs model)", fontsize=9)
        return
    ax.imshow(_np(err_rgb), interpolation="nearest")
    ax.set_title("error: TP green, FP red, FN blue", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_f_c1(
    ax_t: plt.Axes,
    ax_p: plt.Axes,
    target: torch.Tensor,
    prob: torch.Tensor | None,
    geom: C0Geometry,
    gr: int,
    gc: int,
) -> None:
    br, bc = geom.patch_block(gr, gc)
    rs, cs = geom.neighborhood_blocks(br, bc, radius=1)
    t = _np(target)[rs, cs]
    fg = (t > 0).astype(np.float32)
    _imshow_grid(ax_t, fg, title="F: target zoom 3×3 patches", cmap="viridis", vmin=0, vmax=1)
    if prob is not None:
        p = _np(prob)[rs, cs]
        _imshow_grid(ax_p, p, title="F: P(fg) zoom", cmap="magma", vmin=0.0, vmax=1.0)
    else:
        ax_p.axis("off")
        ax_p.set_title("F: (no model)", fontsize=9)


def _panel_g_c1(
    ax,
    sample: C1LoadedSample,
    geom: C0Geometry,
    cfg: LocalGridViTConfig | None,
    metrics: dict | None,
    ckpt_label: str,
) -> None:
    ax.axis("off")
    meta = sample.meta
    lines = [
        f"split={meta.split} crop_index={meta.crop_index} task={meta.task_key}",
        f"crop_id={meta.crop_id} encoding={meta.encoding}",
    ]
    if cfg is not None:
        lines.append(
            f"P={cfg.patch_size} N={cfg.num_patches} s={geom.subcells} "
            f"grid={cfg.grid_out_size} ckpt={ckpt_label or '—'}"
        )
    if metrics:
        lines.extend([
            f"IoU={metrics['iou']:.4f} P/R={metrics['precision']:.3f}/{metrics['recall']:.3f}",
            f"n_tgt_fg={metrics['n_tgt_fg']} n_pred_fg={metrics['n_pred_fg']} "
            f"empty_tgt={metrics['empty_target']} empty_fp={metrics['empty_fp']}",
            f"P_max={metrics['prob_max']:.3f} thresh={metrics['threshold']}",
        ])
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")


def render_c1_sample(
    sample: C1LoadedSample,
    geom: C0Geometry,
    *,
    cfg: LocalGridViTConfig | None = None,
    prob: torch.Tensor | None = None,
    err_rgb: torch.Tensor | None = None,
    metrics: dict | None = None,
    ckpt_label: str = "",
    focal_gr: int | None = None,
    focal_gc: int | None = None,
) -> Figure:
    fg_class = cfg.foreground_class if cfg else 1
    if focal_gr is None or focal_gc is None:
        from .c1_adapter import focal_grid_cell
        focal_gr, focal_gc = focal_grid_cell(sample.target, fg_class, geom.grid_size)
    fig = plt.figure(figsize=(13, 11), dpi=110)
    gs = GridSpec(
        3, 3, figure=fig,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1.15, 1.0, 0.85],
        hspace=0.38, wspace=0.28,
    )
    _panel_a_c1(fig.add_subplot(gs[0, 0]), sample, geom)
    _panel_b_c1(fig.add_subplot(gs[1, 0]), sample)
    _panel_c_c1(fig.add_subplot(gs[2, 0]), sample, geom, focal_gr, focal_gc)
    _panel_d_c1(fig.add_subplot(gs[0, 1]), sample.target, geom, focal_gr, focal_gc)
    _panel_e_c1(fig.add_subplot(gs[0, 2]), prob)
    _panel_f_c1(
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        sample.target, prob, geom, focal_gr, focal_gc,
    )
    _panel_error(fig.add_subplot(gs[2, 1]), err_rgb)
    _panel_g_c1(fig.add_subplot(gs[2, 2]), sample, geom, cfg, metrics, ckpt_label)
    title = f"c1 task viewer — {sample.meta.task_key}"
    if ckpt_label:
        title += f" ({ckpt_label})"
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.04)
    return fig


def render_c1_targets(sample: C1LoadedSample, *, patch_size: int = 16) -> Figure:
    """Legacy compact 3-panel target preview (--no-model shortcut)."""
    geom = C0Geometry(
        patch_size=patch_size,
        img_size=sample.meta.img_size,
        grid_size=sample.meta.grid_size,
        n_patch=sample.meta.img_size // patch_size,
        subcells=sample.meta.grid_size // (sample.meta.img_size // patch_size),
    )
    return render_c1_sample(sample, geom, cfg=None, prob=None, err_rgb=None, metrics=None)


def render_c1_compare(
    sample: C1LoadedSample,
    geom: C0Geometry,
    rows: list[tuple[torch.Tensor, torch.Tensor, dict, str]],
) -> Figure:
    """Panel A once; one row per checkpoint: E, error map, F prob zoom."""
    from .c1_adapter import focal_grid_cell

    gr, gc = focal_grid_cell(sample.target, 1, geom.grid_size)
    br, bc = geom.patch_block(gr, gc)
    rs, cs = geom.neighborhood_blocks(br, bc, radius=1)
    fig = plt.figure(figsize=(12, 2.5 + 2.2 * len(rows)), dpi=110)
    gs = GridSpec(len(rows) + 1, 3, figure=fig, height_ratios=[1.4] + [1] * len(rows), hspace=0.4)
    _panel_a_c1(fig.add_subplot(gs[0, :]), sample, geom)
    for i, (prob, err, metrics, label) in enumerate(rows):
        ax_e = fig.add_subplot(gs[i + 1, 0])
        _panel_e_c1(ax_e, prob)
        ax_e.set_title(f"E ({label})", fontsize=8)
        ax_err = fig.add_subplot(gs[i + 1, 1])
        _panel_error(ax_err, err)
        ax_err.set_title(f"err ({label}) IoU={metrics['iou']:.3f}", fontsize=8)
        ax_f = fig.add_subplot(gs[i + 1, 2])
        _imshow_grid(ax_f, _np(prob)[rs, cs], title=f"F ({label})", cmap="magma", vmin=0, vmax=1.0)
    fig.suptitle(
        f"c1 compare @ {sample.meta.split} crop {sample.meta.crop_index} ({sample.meta.task_key})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
