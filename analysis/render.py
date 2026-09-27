"""Matplotlib layout for c0 task viewer (§13.1 panels A–G)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from plain_gpt_module.local_config import LocalGridViTConfig

from .geometry import C0Geometry
from .load_sample import LoadedSample
from .rgb import rgb_from_label_planes


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _draw_patch_grid(ax, img_size: int, patch_size: int, *, color="white", lw=0.6, alpha=0.85):
    for x in range(0, img_size + 1, patch_size):
        ax.axvline(x - 0.5, color=color, lw=lw, alpha=alpha)
    for y in range(0, img_size + 1, patch_size):
        ax.axhline(y - 0.5, color=color, lw=lw, alpha=alpha)


def _crosshair(ax, col: int, row: int, size: int = 12, color="yellow", lw=1.2):
    ax.plot([col - size, col + size], [row, row], color=color, lw=lw)
    ax.plot([col, col], [row - size, row + size], color=color, lw=lw)


def _imshow_grid(ax, arr, *, title: str, cmap="gray", vmin=None, vmax=None):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _outline_subcell(ax, sr: slice, sc: slice, sub_r: int, sub_c: int, s: int, *, color="lime", lw=1.5):
    r0 = sr.start + sub_r * (sr.stop - sr.start) // s
    c0 = sc.start + sub_c * (sc.stop - sc.start) // s
    cell_h = (sr.stop - sr.start) / s
    cell_w = (sc.stop - sc.start) / s
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), cell_w, cell_h, fill=False, ec=color, lw=lw))


def _outline_patch_block(ax, br: int, bc: int, s: int, G: int, *, color="cyan", lw=1.2):
    ax.add_patch(Rectangle((bc * s - 0.5, br * s - 0.5), s, s, fill=False, ec=color, lw=lw))


def _panel_a(ax, sample: LoadedSample, geom: C0Geometry):
    planes = sample.label_planes
    if planes is None:
        raise ValueError("panel A needs label_planes (worldsnap or synthetic)")
    rgb = rgb_from_label_planes(*planes)
    mr, mc = sample.meta.marker_row, sample.meta.marker_col
    ax.imshow(rgb, interpolation="nearest")
    _draw_patch_grid(ax, geom.img_size, geom.patch_size)
    _crosshair(ax, mc, mr)
    ax.set_title("A: input RGB + patch grid + marker", fontsize=9)
    ax.set_xlim(-0.5, geom.img_size - 0.5)
    ax.set_ylim(geom.img_size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _channel_mosaic(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Stack (C, H, W) planes horizontally; return mosaic, H, W."""
    h, w = img.shape[1], img.shape[2]
    mosaic = np.concatenate([img[ch] for ch in range(img.shape[0])], axis=1)
    return mosaic, h, w


def _panel_b(ax, sample: LoadedSample):
    img = sample.img.numpy()
    n = img.shape[0]
    mosaic, h, w = _channel_mosaic(img)
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for k in range(1, n):
        ax.axvline(k * w - 0.5, color="white", lw=0.8)
    label = "noise | surface | estab | marker" if n == 4 else " | ".join(f"ch{i}" for i in range(n))
    ax.set_title(f"B: {label}", fontsize=9)
    ax.set_xlim(-0.5, n * w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_c(ax, sample: LoadedSample, geom: C0Geometry):
    mr, mc = sample.meta.marker_row, sample.meta.marker_col
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
    gr, gc = geom.marker_grid_cell(mr, mc)
    sub_r, sub_c = geom.subcell_in_patch(gr, gc)
    _outline_subcell(ax, slice(0, p), slice(0, p), sub_r, sub_c, s)
    ax.set_title(f"C: marker patch P={p} (×{n} ch), s={s} subcells", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_d(ax, target: torch.Tensor, geom: C0Geometry, mr: int, mc: int):
    t = target.numpy()
    _imshow_grid(ax, t, title="D: target grid", cmap="viridis", vmin=0, vmax=max(1, t.max()))
    gr, gc = geom.marker_grid_cell(mr, mc)
    br, bc = geom.patch_block(gr, gc)
    _outline_patch_block(ax, br, bc, geom.subcells, geom.grid_size)


def _mark_argmax_pixel(ax, ar: int, ac: int, *, cmap_name: str = "magma"):
    # Overlay one cell; red unless colormap already uses strong red at high end.
    cmap = mpl.colormaps[cmap_name]
    high = cmap(1.0)[:3]
    use_green = high[0] > 0.75 and high[1] < 0.45
    color = "#00dd00" if use_green else "#ff0000"
    ax.add_patch(
        Rectangle((ac - 0.5, ar - 0.5), 1, 1, facecolor=color, edgecolor="none", zorder=5)
    )


def _panel_e(ax, prob: torch.Tensor, metrics: dict | None):
    _imshow_grid(ax, _np(prob), title="E: P(fg)", cmap="magma", vmin=0.0, vmax=1.0)
    if metrics:
        ar, ac = metrics["argmax_cell"]
        _mark_argmax_pixel(ax, ar, ac)


def _panel_f_pair(ax_d, ax_e, target: torch.Tensor, prob: torch.Tensor | None, geom: C0Geometry, mr: int, mc: int):
    gr, gc = geom.marker_grid_cell(mr, mc)
    br, bc = geom.patch_block(gr, gc)
    rs, cs = geom.neighborhood_blocks(br, bc, radius=1)
    t = _np(target)[rs, cs]
    _imshow_grid(ax_d, t, title="F: target zoom 3×3 patches", cmap="viridis", vmin=0, vmax=1)
    if prob is not None:
        p = _np(prob)[rs, cs]
        _imshow_grid(ax_e, p, title="F: P(fg) zoom", cmap="magma", vmin=0.0, vmax=1.0)


def _panel_g(ax, meta, geom: C0Geometry, cfg: LocalGridViTConfig | None, metrics: dict | None, ckpt_label: str):
    ax.axis("off")
    lines = [
        f"split={meta.split} item={meta.item_index} source={meta.source}",
        f"crop={meta.crop_id or '—'} id={meta.item_id or '—'}",
        f"marker (row,col)=({meta.marker_row},{meta.marker_col}) encoding={meta.encoding}",
    ]
    if cfg is not None:
        n_tok = cfg.num_patches
        lines.append(
            f"P={cfg.patch_size} N={n_tok} s={geom.subcells} "
            f"grid={cfg.grid_out_size} params_ckpt={ckpt_label or '—'}"
        )
    if metrics:
        sg = metrics.get("subcell_given_patch")
        sg_s = f"{sg}" if sg is not None else "n/a"
        lines.extend([
            f"n_fg={metrics['n_fg']} patch_hit={metrics['patch_hit']} "
            f"subcell|patch={sg_s} (chance={metrics['chance_subcell']:.3f})",
            f"argmax_dist={metrics['argmax_dist_cells']:.2f} cells "
            f"recall={metrics['true_cell_recall']} exact={metrics['global_argmax_exact']}",
        ])
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")


def render_sample(
    sample: LoadedSample,
    geom: C0Geometry,
    *,
    cfg: LocalGridViTConfig | None = None,
    prob: torch.Tensor | None = None,
    metrics: dict | None = None,
    ckpt_label: str = "",
) -> Figure:
    """Single-sample figure with panels A–G (E/F need prob when model provided)."""
    mr, mc = sample.meta.marker_row, sample.meta.marker_col
    fig = plt.figure(figsize=(13, 11), dpi=110)
    gs = GridSpec(
        3, 3, figure=fig,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1.15, 1.0, 0.85],
        hspace=0.38, wspace=0.28,
    )
    _panel_a(fig.add_subplot(gs[0, 0]), sample, geom)
    _panel_b(fig.add_subplot(gs[1, 0]), sample)
    _panel_c(fig.add_subplot(gs[2, 0]), sample, geom)
    _panel_d(fig.add_subplot(gs[0, 1]), sample.target, geom, mr, mc)
    ax_e = fig.add_subplot(gs[0, 2])
    if prob is not None:
        _panel_e(ax_e, prob, metrics)
    else:
        ax_e.axis("off")
        ax_e.set_title("E: (no model)", fontsize=9)
    _panel_f_pair(
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        sample.target, prob, geom, mr, mc,
    )
    _panel_g(fig.add_subplot(gs[2, 2]), sample.meta, geom, cfg, metrics, ckpt_label)
    fig.add_subplot(gs[2, 1]).axis("off")
    fig.suptitle("c0 task viewer (static)", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.04)
    return fig


def render_compare_row(
    fig: Figure,
    gs_row,
    sample: LoadedSample,
    geom: C0Geometry,
    prob: torch.Tensor,
    metrics: dict,
    ckpt_label: str,
):
    ax_c = fig.add_subplot(gs_row[0])
    _panel_c(ax_c, sample, geom)
    ax_c.set_title(f"C ({ckpt_label})", fontsize=8)
    mr, mc = sample.meta.marker_row, sample.meta.marker_col
    ax_e = fig.add_subplot(gs_row[1])
    _panel_e(ax_e, prob, metrics)
    ax_e.set_title(f"E ({ckpt_label})", fontsize=8)
    ax_f = fig.add_subplot(gs_row[2])
    gr, gc = geom.marker_grid_cell(mr, mc)
    br, bc = geom.patch_block(gr, gc)
    rs, cs = geom.neighborhood_blocks(br, bc, radius=1)
    _imshow_grid(ax_f, _np(prob)[rs, cs], title=f"F ({ckpt_label})", cmap="magma", vmin=0, vmax=1.0)


def render_compare(
    sample: LoadedSample,
    rows: list[tuple[C0Geometry, LocalGridViTConfig, torch.Tensor, dict, str]],
) -> Figure:
    """Panel A once; one row per checkpoint (C, E, F)."""
    geom0 = rows[0][0]
    fig = plt.figure(figsize=(12, 2.5 + 2.2 * len(rows)), dpi=110)
    gs = GridSpec(len(rows) + 1, 3, figure=fig, height_ratios=[1.4] + [1] * len(rows), hspace=0.4)
    ax_a = fig.add_subplot(gs[0, :])
    _panel_a(ax_a, sample, geom0)
    for i, (geom, cfg, prob, metrics, label) in enumerate(rows):
        render_compare_row(fig, gs[i + 1, :], sample, geom, prob, metrics, label)
    fig.suptitle(f"compare @ {sample.meta.split} item {sample.meta.item_index}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
