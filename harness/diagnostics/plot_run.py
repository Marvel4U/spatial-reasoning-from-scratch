#!/usr/bin/env python3
"""Regenerate ANALYSIS_SUITE_SPEC §9 plots from diagnostics JSONL."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt

import config
from grid_vit_adapter import GRAD_CLIP

from .paths import (
    compare_path,
    dashboard_path,
    diagnostics_root,
    resolve_eval,
    resolve_meta,
    resolve_tier1,
    run_dir,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _steps(rows: list[dict]) -> list[int]:
    return [int(r["step"]) for r in rows]


def _series(rows: list[dict], key: str) -> list[float | None]:
    out: list[float | None] = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if v is not None else None)
    return out


def _keys_with_prefix(rows: list[dict], prefix: str) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(k for k in r if k.startswith(prefix))
    return sorted(keys)


def _step_x(ax, *, log: bool = False) -> None:
    if log:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("step (symlog)")
    else:
        ax.set_xlabel("step")


def _is_c3_eval(eval_rows: list[dict], meta: dict | None) -> bool:
    if any(r.get("c3_task") for r in eval_rows):
        return True
    return (meta or {}).get("config", {}).get("task_rung") == "c3"


def _is_c4_eval(eval_rows: list[dict], meta: dict | None) -> bool:
    if any("c4_connectivity" in r for r in eval_rows):
        return True
    return (meta or {}).get("config", {}).get("task_rung") == "c4"


def _c4_run_short_label(run_id: str) -> str:
    if "det_only" in run_id:
        mix = "det"
    elif "bld_seg_det" in run_id:
        mix = "bld+seg+det"
    elif "bld_det" in run_id:
        mix = "bld+det"
    elif "bld_segdet" in run_id or "bld_segdet" in run_id.replace("_", ""):
        mix = "bld+seg+det"
    elif "segdet" in run_id:
        mix = "seg+det"
    else:
        mix = run_id.removeprefix("c4n_").split("_L")[0].replace("_", " ")
    depth = "L4" if "_L4_" in run_id else "L2" if "_L2_" in run_id else "?"
    store = "100k" if "store100k" in run_id or "c4_100k" in run_id else "20k"
    return f"{depth} {mix}@{store}"


def _plot_c4_eval(ax, eval_rows: list[dict], *, log_x_step: bool) -> None:
    es = _steps(eval_rows)
    si = 0
    for key, lbl, lw in (
        ("IoU_fg", "IoU plain", 2.0),
        ("c4_iou_dilated", "IoU dilated", 1.5),
        ("c4_connectivity", "connectivity", 1.2),
    ):
        if any(key in r for r in eval_rows):
            _plot_styled_series(ax, es, _series(eval_rows, key), lbl, si, lw=lw)
            si += 1
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("4: c4 val (detour pool)")
    _step_x(ax, log=log_x_step)


def _plot_c3_eval(ax, eval_rows: list[dict], *, log_x_step: bool) -> None:
    es = _steps(eval_rows)
    for key, lbl, lw in (
        ("IoU_fg", "IoU_fg", 2.0),
        ("IoU_fg_patch_local", "patch-local", 1.0),
        ("c3_geometry_oracle_iou", "oracle", 1.0),
    ):
        if any(key in r for r in eval_rows):
            ax.plot(es, _series(eval_rows, key), label=lbl, lw=lw)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("4: c3 val IoU (spatial eval)")
    _step_x(ax, log=log_x_step)


def _is_c1_eval(eval_rows: list[dict], meta: dict | None) -> bool:
    if any(r.get("c1_task") for r in eval_rows):
        return True
    cfg = (meta or {}).get("config") or {}
    return cfg.get("task_rung") == "c1"


def _c1_per_task_iou_keys(eval_rows: list[dict]) -> list[str]:
    skip = {
        "iou_held_out_correct",
        "iou_held_out_wrong_task",
        "iou_held_out_composition_gap",
    }
    keys: set[str] = set()
    for r in eval_rows:
        for k in r:
            if k.startswith("iou_") and k not in skip:
                keys.add(k)
    return sorted(keys, key=lambda x: x.removeprefix("iou_"))


def _plot_c0_decomposition(ax, eval_rows: list[dict], *, log_x_step: bool) -> None:
    es = _steps(eval_rows)
    for key, lbl in (
        ("patch_hit", "patch_hit"),
        ("argmax_hit", "argmax_hit"),
        ("subcell_hit_given_patch", "subcell|patch"),
        ("exact_cell", "exact_cell"),
        ("recall", "recall"),
    ):
        if any(key in r for r in eval_rows):
            ax.plot(es, _series(eval_rows, key), label=lbl, lw=1.2)
    s = eval_rows[-1].get("subcells_per_patch")
    if s:
        ax.axhline(1.0 / (s * s), color="k", ls=":", lw=1, alpha=0.5, label=f"chance 1/{s*s}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("4: c0 decomposition (val)")
    _step_x(ax, log=log_x_step)


def _plot_c1_task_iou(ax, eval_rows: list[dict], *, log_x_step: bool) -> None:
    es = _steps(eval_rows)
    task_keys = _c1_per_task_iou_keys(eval_rows)
    if task_keys:
        cmap = plt.get_cmap("tab20")
        for i, fk in enumerate(task_keys):
            lbl = fk.removeprefix("iou_")
            ax.plot(es, _series(eval_rows, fk), color=cmap(i % 20), label=lbl, lw=1.0, alpha=0.9)
        ax.set_title("4: IoU by task (val)")
    else:
        for key, lbl, lw in (
            ("IoU_fg", "IoU_fg", 2.0),
            ("IoU_pure_cells", "IoU_pure", 1.2),
            ("IoU_mixed_cells", "IoU_mixed", 1.2),
        ):
            if any(key in r for r in eval_rows):
                ax.plot(es, _series(eval_rows, key), label=lbl, lw=lw)
        ax.set_title("4: c1 val IoU (pooled; re-run harness for per-task series)")
    for hk, lbl, ls in (
        ("iou_held_out_correct", "held_out ✓", "-"),
        ("iou_held_out_wrong_task", "held_out wrong", "--"),
    ):
        if any(hk in r for r in eval_rows):
            ax.plot(es, _series(eval_rows, hk), color="k", ls=ls, lw=1.5, label=lbl)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5, loc="best", ncol=2)
    _step_x(ax, log=log_x_step)


def _balanced_ce_refs(meta: dict | None) -> tuple[float, float | None]:
    """Untrained ln2 and balanced-CE 'all background' reference (spec §9 plot 1)."""
    untrained = math.log(2)
    trivial = None
    if meta:
        cfg = meta.get("config") or {}
        g = int(cfg.get("grid_out_size", 64))
        if cfg.get("num_grid_classes", 2) >= 2:
            n_bg = g * g - 1
            # spec: ≈0.5·6.9 at default c0; same order as 0.5·ln(n_bg) for n_bg≈4095
            trivial = 0.5 * math.log(max(n_bg, 2))
    return untrained, trivial


def _enrich_c1_per_task_from_ckpts(run_id: str, meta: dict, eval_rows: list[dict]) -> list[dict]:
    """Fill iou_<task> on eval rows from checkpoint series when JSONL predates flatten export."""
    if _c1_per_task_iou_keys(eval_rows):
        return eval_rows
    ckpt_dir = Path(config.harness_checkpoint_dir)
    pairs: list[tuple[int, Path]] = []
    series = ckpt_dir / run_id
    if series.is_dir():
        for p in series.glob("step*.pt"):
            try:
                pairs.append((int(p.stem.replace("step", "")), p))
            except ValueError:
                continue
    main = ckpt_dir / f"{run_id}.pt"
    if main.is_file() and eval_rows:
        final_step = max(int(r["step"]) for r in eval_rows)
        pairs.append((final_step, main))
    if not pairs:
        return eval_rows
    pairs.sort(key=lambda x: x[0])
    import torch

    import data
    import eval_spatial
    import grid_vit_adapter

    saved_cfg: dict[str, object] = {}
    top = meta.get("config") or {}
    for k, v in top.items():
        if hasattr(config, k):
            saved_cfg[k] = getattr(config, k)
            if k in ("data_root", "harness_checkpoint_dir", "diagnostics_dir") and v is not None:
                v = Path(v)
            setattr(config, k, v)
    model_cfg = dict(meta.get("model") or {})
    c1_mode = (eval_rows[0].get("c1_task") if eval_rows else None) or model_cfg.get("c1_task_mode")
    if c1_mode in ("core_four", "factorised_twelve", "single"):
        config.c1_task_mode = c1_mode
    else:
        n_tasks = int(model_cfg.get("n_tasks", 0))
        if n_tasks == 12:
            config.c1_task_mode = "factorised_twelve"
        elif n_tasks == 4:
            config.c1_task_mode = "core_four"
    config.task_rung = "c1"
    config.sync_in_chans_from_encoding()
    data.setup_device()
    data.load_data()
    from checkpoints import checkpoint_model_config
    from plain_gpt_module.checkpoint import read_checkpoint
    from plain_gpt_module.local_config import LocalGridViTConfig
    from plain_gpt_module.local_grid_vit import LocalGridViT

    per_step: dict[int, dict] = {}
    try:
        for step, path in pairs:
            ckpt = read_checkpoint(path, map_location="cpu")
            cfg_dict = checkpoint_model_config(ckpt) or model_cfg
            model = LocalGridViT(LocalGridViTConfig(**cfg_dict))
            model.to(data.device)
            grid_vit_adapter.load_checkpoint(path, model=model)
            model.eval()
            val = eval_spatial.eval_loader(model, data.val_loader, max_batches=config.eval_iters)
            per_step[step] = eval_spatial.flatten_for_eval_jsonl(val)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for k, v in saved_cfg.items():
            setattr(config, k, v)
    if not per_step:
        return eval_rows
    ckpt_steps = sorted(per_step)
    enriched: list[dict] = []
    for r in eval_rows:
        row = dict(r)
        step = int(r["step"])
        nearest = max((s for s in ckpt_steps if s <= step), default=ckpt_steps[0])
        for k, v in per_step[nearest].items():
            if k.startswith("iou_") or k in ("iou_held_out_correct", "iou_held_out_wrong_task", "held_out_composition_gap"):
                row[k] = v
        enriched.append(row)
    return enriched


def plot_run_dashboard(
    run_id: str,
    *,
    diag_dir: Path | None = None,
    out_dir: Path | None = None,
    meta: dict | None = None,
    tier1: list[dict] | None = None,
    eval_rows: list[dict] | None = None,
    log_x_step: bool = False,
    enrich_c1_from_ckpts: bool = True,
) -> Path:
    diag_dir = diag_dir or diagnostics_root()
    out_dir = out_dir or (diag_dir / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_p = resolve_meta(run_id)
    if meta is None and meta_p is not None:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if tier1 is None:
        t1p = resolve_tier1(run_id)
        tier1 = load_jsonl(t1p) if t1p else []
    if eval_rows is None:
        evp = resolve_eval(run_id)
        eval_rows = load_jsonl(evp) if evp else []
    if not tier1 and not eval_rows:
        raise SystemExit(f"no diagnostics JSONL for run {run_id!r} under {diag_dir}")
    if enrich_c1_from_ckpts and eval_rows and meta and _is_c1_eval(eval_rows, meta):
        eval_rows = _enrich_c1_per_task_from_ckpts(run_id, meta, eval_rows)

    channels = (meta or {}).get("input_channels") or ["noise", "surface", "estab", "marker"]
    untrained_ce, trivial_ce = _balanced_ce_refs(meta)

    has_probe = bool(eval_rows) and any(
        k.startswith("attn_entropy/") or k.startswith("resid_rms/") for r in eval_rows for k in r
    )
    n_rows = 4 if has_probe else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"Run diagnostics: {run_id}", fontsize=12)

    # --- 1: loss + lr ---
    ax = axes[0, 0]
    ax2 = ax.twinx()
    if eval_rows:
        es = _steps(eval_rows)
        _plot_styled_series(ax, es, _series(eval_rows, "train_loss_eval"), "train CE eval", 0, lw=1.8)
        _plot_styled_series(ax, es, _series(eval_rows, "val_loss_eval"), "val CE eval", 1, lw=1.8)
    if tier1:
        ts = _steps(tier1)
        _plot_styled_series(ax, ts, _series(tier1, "loss"), "batch loss (tier1)", 2, lw=0.9, alpha=0.45)
        _plot_styled_series(ax2, ts, _series(tier1, "lr"), "lr", 3, lw=1.2, alpha=0.85)
    ax.axhline(untrained_ce, color="gray", ls=":", lw=1, label=f"ln2≈{untrained_ce:.2f}")
    if trivial_ce is not None:
        ax.axhline(trivial_ce, color="gray", ls="--", lw=1, alpha=0.7, label=f"trivial≈{trivial_ce:.2f}")
    ax.set_ylabel("loss")
    ax2.set_ylabel("lr")
    ax.set_title("1: CE eval + lr")
    ax.legend(loc="upper right", fontsize=7)
    ax2.legend(loc="center right", fontsize=7)
    ax.grid(True, alpha=0.3)
    _step_x(ax, log=log_x_step)

    # --- 4: c1 per-task IoU or c0 decomposition ---
    ax = axes[0, 1]
    if eval_rows:
        if _is_c1_eval(eval_rows, meta):
            _plot_c1_task_iou(ax, eval_rows, log_x_step=log_x_step)
        elif _is_c3_eval(eval_rows, meta):
            _plot_c3_eval(ax, eval_rows, log_x_step=log_x_step)
        elif _is_c4_eval(eval_rows, meta):
            _plot_c4_eval(ax, eval_rows, log_x_step=log_x_step)
        else:
            _plot_c0_decomposition(ax, eval_rows, log_x_step=log_x_step)
    else:
        ax.text(0.5, 0.5, "no eval.jsonl", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("4: task metrics (val)")
    ax.grid(True, alpha=0.3)

    # --- 2: update ratios ---
    ax = axes[1, 0]
    if tier1:
        ts = _steps(tier1)
        for si, key in enumerate(_keys_with_prefix(tier1, "update_ratio/")):
            short = key.replace("update_ratio/", "")
            _plot_styled_series(ax, ts, _series(tier1, key), short, si, lw=1.1, alpha=0.9)
        ax.axhline(-3.0, color="gray", ls="--", lw=1, label="ref −3")
        ax.set_ylabel("log10 update ratio")
        ax.legend(fontsize=5, ncol=2, loc="best")
        _step_x(ax, log=log_x_step)
    ax.set_title("2: update_ratio / group")
    ax.grid(True, alpha=0.3)

    # --- 5: grad norms ---
    ax = axes[1, 1]
    if tier1:
        ts = _steps(tier1)
        ax.plot(ts, _series(tier1, "grad_norm_total"), "k-", lw=2, label="total (pre-clip)")
        ax.axhline(GRAD_CLIP, color="red", ls="--", lw=1, label=f"clip={GRAD_CLIP}")
        gi = 0
        for key in _keys_with_prefix(tier1, "grad_norm/"):
            if key == "grad_norm_total":
                continue
            _plot_styled_series(
                ax, ts, _series(tier1, key), key.replace("grad_norm/", ""), gi, lw=1.0, alpha=0.85,
            )
            gi += 1
        ax.set_yscale("log")
        ax.legend(fontsize=5, ncol=2, loc="best")
        _step_x(ax, log=log_x_step)
    ax.set_title("5: grad norm / group")
    ax.grid(True, alpha=0.3)

    # --- 3: patch shared / resid ---
    ax = axes[2, 0]
    if tier1:
        ts = _steps(tier1)
        has_line = False
        pi = 0
        for cn in channels:
            sk, rk = f"patch_shared/{cn}", f"patch_resid/{cn}"
            if any(sk in r for r in tier1):
                _plot_styled_series(ax, ts, _series(tier1, sk), f"{cn} shared", pi, lw=1.1)
                pi += 1
                has_line = True
            if any(rk in r for r in tier1):
                _plot_styled_series(ax, ts, _series(tier1, rk), f"{cn} resid", pi, lw=1.0, alpha=0.85)
                pi += 1
                has_line = True
        if has_line:
            ax.legend(fontsize=6, ncol=2, loc="best")
        _step_x(ax, log=log_x_step)
    ax.set_title("3: patch_embed shared vs resid")
    ax.grid(True, alpha=0.3)

    # --- 6: probe attn entropy + resid rms ---
    if has_probe:
        ax = axes[3, 0]
        es = _steps(eval_rows)
        n_blocks = max(
            (int(k.split("/")[1]) for r in eval_rows for k in r if k.startswith("attn_entropy/")),
            default=-1,
        ) + 1
        cfg_m = (meta or {}).get("config") or {}
        ps = int(cfg_m.get("patch_size", 16))
        isize = int(cfg_m.get("img_size", 256))
        ln_t = math.log(max((isize // ps) ** 2, 2))
        for i in range(n_blocks):
            keys = [f"attn_entropy/{i}/{h}" for h in range(6) if any(f"attn_entropy/{i}/{h}" in r for r in eval_rows)]
            if not keys:
                continue
            mean_ent = []
            for r in eval_rows:
                vals = [r.get(k) for k in keys if r.get(k) is not None]
                mean_ent.append(sum(vals) / len(vals) if vals else None)
            _plot_styled_series(ax, es, mean_ent, f"blk{i} mean H", i, lw=1.2)
        ax.axhline(ln_t, color="gray", ls=":", lw=1, label=f"ln T≈{ln_t:.2f}")
        ax.set_ylabel("attn entropy")
        ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.92)
        ax.set_title("6: attn entropy (probe, mean/head)")
        ax.grid(True, alpha=0.3)
        _step_x(ax, log=log_x_step)

        ax = axes[3, 1]
        for i in range(n_blocks):
            rk = f"resid_rms/{i}"
            if any(rk in r for r in eval_rows):
                _plot_styled_series(ax, es, _series(eval_rows, rk), f"blk{i}", i, lw=1.2)
        ax.set_ylabel("resid RMS")
        ax.legend(fontsize=7, loc="best", framealpha=0.92)
        ax.set_title("6: block resid RMS (probe)")
        ax.grid(True, alpha=0.3)
        _step_x(ax, log=log_x_step)

    # --- 7: timing ---
    ax = axes[2, 1]
    ax2 = ax.twinx()
    if tier1:
        ts = _steps(tier1)
        _plot_styled_series(ax, ts, _series(tier1, "gpu_starvation"), "gpu_starvation", 0, lw=1.5)
        ax.axhline(0.1, color="gray", ls=":", label="10% ref")
        ax.axhline(0.5, color="red", ls=":", alpha=0.5, label="50% warn")
        _plot_styled_series(ax2, ts, _series(tier1, "samples_per_s"), "samples/s", 1, lw=1.2, alpha=0.9)
        ax.set_ylabel("t_data / (t_data+t_step)")
        ax2.set_ylabel("samples/s")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        _step_x(ax, log=log_x_step)
    ax.set_title("7: loader vs throughput")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = dashboard_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


_C4_COMPARE_METRICS = (
    ("val_loss_eval", "val CE"),
    ("train_loss_eval", "train CE"),
    ("IoU_fg", "IoU plain"),
    ("c4_iou_dilated", "IoU dilated"),
    ("c4_connectivity", "connectivity"),
    ("c4_length_ratio", "length ratio"),
)

# Distinct series styling when many lines share a panel (dashboard + c4 compares).
_DIAG_SERIES_COLORS = (
    list(plt.get_cmap("tab10").colors)
    + list(plt.get_cmap("Set2").colors)
    + list(plt.get_cmap("Dark2").colors)
)
_DIAG_SERIES_LINE_STYLES = ("-", "--", "-.", ":")
_DIAG_SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "8", "p", "<", ">")


def _diag_series_style(index: int) -> dict[str, object]:
    n_c = len(_DIAG_SERIES_COLORS)
    return {
        "color": _DIAG_SERIES_COLORS[index % n_c],
        "linestyle": _DIAG_SERIES_LINE_STYLES[(index // n_c) % len(_DIAG_SERIES_LINE_STYLES)],
        "marker": _DIAG_SERIES_MARKERS[index % len(_DIAG_SERIES_MARKERS)],
    }


def _diag_mark_every(n: int) -> list[int]:
    """First and last point — trace curves when color cycles repeat."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    return [0, n - 1]


def _plot_styled_series(
    ax,
    xs: list,
    ys: list,
    label: str,
    series_index: int,
    *,
    lw: float = 1.6,
    alpha: float = 1.0,
    markers: bool = True,
    **extra,
) -> None:
    st = _diag_series_style(series_index)
    kw: dict[str, object] = {
        "label": label,
        "lw": lw,
        "alpha": alpha,
        "color": st["color"],
        "linestyle": st["linestyle"],
        **extra,
    }
    if markers and xs:
        kw["marker"] = st["marker"]
        kw["markevery"] = _diag_mark_every(len(xs))
        kw["markersize"] = 5.5
        kw["markeredgewidth"] = 0.5
        kw["markerfacecolor"] = st["color"]
        kw["markeredgecolor"] = "white"
    ax.plot(xs, ys, **kw)


def _plot_c4_compare_axes(axes, run_series: list[tuple[str, list[dict]]], *, log_x_step: bool, suptitle: str) -> None:
    n_runs = len(run_series)
    legend_fs = 7 if n_runs <= 4 else 6
    legend_ncol = 1 if n_runs >= 4 else 1
    for ax, (key, title) in zip(axes.flat, _C4_COMPARE_METRICS):
        for i, (run_id, rows) in enumerate(run_series):
            if not rows:
                continue
            _plot_styled_series(
                ax,
                _steps(rows),
                _series(rows, key),
                _c4_run_short_label(run_id),
                i,
                lw=1.8,
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=legend_fs, loc="best", ncol=legend_ncol, framealpha=0.92)
        _step_x(ax, log=log_x_step)
        if key in ("IoU_fg", "c4_iou_dilated", "c4_connectivity"):
            ax.set_ylim(-0.05, 1.05)


def plot_compare_c4(
    run_a: str,
    run_b: str,
    *,
    diag_dir: Path | None = None,
    log_x_step: bool = False,
) -> Path:
    diag_dir = diag_dir or diagnostics_root()
    ea = load_jsonl(resolve_eval(run_a) or Path())
    eb = load_jsonl(resolve_eval(run_b) or Path())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _plot_c4_compare_axes(
        axes,
        [(run_a, ea), (run_b, eb)],
        log_x_step=log_x_step,
        suptitle=f"{_c4_run_short_label(run_a)} vs {_c4_run_short_label(run_b)}",
    )
    fig.suptitle(f"C4 compare: {_c4_run_short_label(run_a)} vs {_c4_run_short_label(run_b)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = compare_path(run_a, run_b)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare_c4_multi(
    run_ids: list[str],
    *,
    primary: str | None = None,
    diag_dir: Path | None = None,
    log_x_step: bool = False,
    out_path: Path | None = None,
) -> Path:
    if len(run_ids) < 2:
        raise ValueError("need at least two run ids")
    diag_dir = diag_dir or diagnostics_root()
    primary = primary or run_ids[0]
    series = [(rid, load_jsonl(resolve_eval(rid) or Path())) for rid in run_ids]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    labels = ", ".join(_c4_run_short_label(r) for r, _ in series)
    _plot_c4_compare_axes(axes, series, log_x_step=log_x_step, suptitle=labels)
    fig.suptitle(f"C4 compare: {labels}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_path or (run_dir(primary) / "compare_c4_multi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare(
    run_a: str,
    run_b: str,
    *,
    diag_dir: Path | None = None,
    out_dir: Path | None = None,
    log_x_step: bool = False,
) -> Path:
    diag_dir = diag_dir or diagnostics_root()
    out_dir = out_dir or (diag_dir / run_a)
    ea = load_jsonl(resolve_eval(run_a) or Path())
    eb = load_jsonl(resolve_eval(run_b) or Path())
    if _is_c4_eval(ea, None) or _is_c4_eval(eb, None):
        return plot_compare_c4(run_a, run_b, diag_dir=diag_dir, log_x_step=log_x_step)
    t1a = load_jsonl(resolve_tier1(run_a) or Path())
    t1b = load_jsonl(resolve_tier1(run_b) or Path())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Compare {run_a} vs {run_b}")

    ax = axes[0, 0]
    if ea:
        ax.plot(_steps(ea), _series(ea, "val_loss_eval"), label=f"{run_a} val")
    if eb:
        ax.plot(_steps(eb), _series(eb, "val_loss_eval"), label=f"{run_b} val")
    ax.set_title("val CE eval")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _step_x(ax, log=log_x_step)

    ax = axes[0, 1]
    for rows, name, c in ((ea, run_a, "C0"), (eb, run_b, "C1")):
        if rows:
            ax.plot(_steps(rows), _series(rows, "patch_hit"), color=c, label=f"{name} patch_hit")
            ax.plot(_steps(rows), _series(rows, "subcell_hit_given_patch"), color=c, ls="--",
                    label=f"{name} subcell|patch")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("c0 metrics")
    ax.grid(True, alpha=0.3)
    _step_x(ax, log=log_x_step)

    ax = axes[1, 0]
    for rows, name in ((t1a, run_a), (t1b, run_b)):
        if rows and any("update_ratio/patch_embed/marker" in r for r in rows):
            ax.plot(_steps(rows), _series(rows, "update_ratio/patch_embed/marker"), label=name)
    ax.axhline(-3, color="gray", ls="--")
    ax.set_title("update_ratio patch_embed/marker")
    ax.grid(True, alpha=0.3)
    _step_x(ax, log=log_x_step)

    ax = axes[1, 1]
    for rows, name in ((t1a, run_a), (t1b, run_b)):
        if rows:
            ax.plot(_steps(rows), _series(rows, "gpu_starvation"), label=name)
    ax.axhline(0.5, color="red", ls=":", alpha=0.5)
    ax.set_title("gpu starvation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _step_x(ax, log=log_x_step)

    fig.tight_layout()
    out = compare_path(run_a, run_b)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot diagnostics JSONL for a harness run")
    p.add_argument("run_id", help="run id (runs/diagnostics/{run_id}/)")
    p.add_argument("--compare", metavar="RUN_ID", default=None, help="overlay second run")
    p.add_argument(
        "--compare-multi",
        metavar="RUN_ID",
        nargs="+",
        default=None,
        help="overlay several runs (c4 metrics); writes compare_c4_multi.png under primary run_id",
    )
    p.add_argument("-o", "--out-dir", type=Path, default=None)
    p.add_argument("--diag-dir", type=Path, default=None)
    p.add_argument(
        "--log-x-step",
        action="store_true",
        help="symlog x-axis (for very long runs); default is linear step",
    )
    p.add_argument(
        "--no-enrich-c1",
        action="store_true",
        help="skip loading checkpoints to backfill per-task IoU on old c1 eval.jsonl",
    )
    args = p.parse_args(argv)
    diag_dir = args.diag_dir or diagnostics_root()
    out_dir = args.out_dir or (diag_dir / args.run_id)
    png = plot_run_dashboard(
        args.run_id,
        diag_dir=diag_dir,
        out_dir=out_dir,
        log_x_step=args.log_x_step,
        enrich_c1_from_ckpts=not args.no_enrich_c1,
    )
    print(f"saved {png}")
    if args.compare_multi:
        ids = [args.run_id, *args.compare_multi]
        cmp_png = plot_compare_c4_multi(ids, primary=args.run_id, diag_dir=diag_dir, log_x_step=args.log_x_step)
        print(f"saved {cmp_png}")
    elif args.compare:
        cmp_png = plot_compare(
            args.run_id, args.compare, diag_dir=diag_dir, out_dir=out_dir,
            log_x_step=args.log_x_step,
        )
        print(f"saved {cmp_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
