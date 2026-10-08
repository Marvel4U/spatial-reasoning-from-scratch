#!/usr/bin/env python3
"""Iterative wave-search rollout on one val route: model output becomes the next scratchpad."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_HARNESS = _REPO / "harness"
for p in (str(_REPO), str(_HARNESS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import data
import grid_vit_adapter
from checkpoints import checkpoint_model_config
from eval_c4 import token_logits
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from spatial_batch import forward as model_forward
from spatial_data.c4_tasks import (
    TOK_BACK,
    TOK_DONE,
    TOK_DRAW,
    TOK_SEARCH,
    WAVE_HORIZON_M,
    WAVE_ROLLOUT_TASK_KEYS,
)
from spatial_data.dataset_c4_gpu import (
    C4MultiTaskGpuValLoader,
    _wave_start_disc,
    load_c4_route_index,
    load_district_planes_v4,
)
from spatial_data.wave_search_schedule import WavePass, n_goal_from_m, wave_pass_schedule

_TOK_NAMES = {TOK_SEARCH: "SEARCH", TOK_BACK: "BACK", TOK_DRAW: "DRAW", TOK_DONE: "DONE"}


def wave_rollout_route_indices() -> list[int]:
    xs = getattr(config, "diagnostics_wave_rollout_route_indices", None)
    if xs:
        return [int(x) for x in xs]
    return [int(getattr(config, "diagnostics_wave_rollout_route_index", 0))]


def _wave_val_goal_m(route_ix: int) -> float:
    w = np.load(Path(config.c4_routes_dir()) / "c5_wave_val.npz")
    return float(w["goal_m"][route_ix])


def wave_rollout_start_pass_for_route(route_ix: int, start: int | str | None = None) -> int:
    """Resolve 1-based first model pass. ``first_back``: teacher through wave_reach, then peel fronts."""
    if start is None:
        start = getattr(config, "diagnostics_wave_rollout_start_pass", 1)
    if isinstance(start, str):
        goal_m = _wave_val_goal_m(route_ix)
        n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M)
        if start in ("first_back", "after_reach", "back"):
            return n_goal + 1
        if start in ("reach", "at_reach"):
            return max(1, n_goal)
        raise ValueError(f"unknown diagnostics_wave_rollout_start_pass {start!r}")
    return int(start)


def _overrides_for_run(run_id: str) -> dict:
    path = Path(config.RESULTS_PATH)
    if not path.is_file():
        return {}
    for r in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
        if r.get("id") == run_id:
            return dict(r.get("overrides") or {})
    return {}


def _apply_meta_config(meta: dict) -> dict:
    saved: dict = {}
    for k, v in (meta.get("config") or {}).items():
        if hasattr(config, k):
            saved[k] = getattr(config, k)
            if k in ("data_root", "harness_checkpoint_dir", "diagnostics_dir") and v is not None:
                v = Path(v)
            setattr(config, k, v)
    return saved


def _pad_grid_to_img(pad: torch.Tensor, crop: int, grid: int) -> torch.Tensor:
    """(grid, grid) bool/float -> (crop, crop) float for channel 16."""
    s = crop // grid
    return pad.float().repeat_interleave(s, 0).repeat_interleave(s, 1)


def _teacher_mask_to_grid(tgt: torch.Tensor, grid: int, crop: int) -> torch.Tensor:
    """Teacher mask at crop or grid resolution -> (grid, grid) bool."""
    m = (tgt > 0).float()
    if m.shape[-1] == grid and m.shape[-2] == grid:
        return m.bool()
    s = crop // grid
    return F.max_pool2d(m[None, None], s, stride=s)[0, 0] > 0


def _prob_fg(logits: torch.Tensor, fg: int) -> torch.Tensor:
    if logits.shape[-1] > 1:
        return torch.sigmoid(logits[..., fg] - logits[..., 0])
    return F.softmax(logits, dim=-1)[..., 0]


def _plot_c4_input(ax, img_row: torch.Tensor, route_ix: int, val_loader, device) -> None:
    """Same base map + yellow markers + blue scratchpad as pred_strip."""
    from analysis.rgb import rgb_from_label_planes
    from diagnostics.pred_strip import _c4_rgb_window

    n4, surf, estab = _c4_rgb_window(route_ix, device, val_loader)
    ax.imshow(rgb_from_label_planes(n4, surf, estab), interpolation="nearest")
    im = img_row.cpu()
    thick = lambda a, k=5: (
        F.max_pool2d(a.float()[None, None], k, stride=1, padding=k // 2)[0, 0].cpu().numpy() > 0
    )
    ov = np.zeros(im.shape[1:] + (4,), dtype=np.float32)
    if im.shape[0] > 16:
        ov[thick(im[16] > 0)] = (0.15, 0.45, 1.0, 1.0)
    ov[thick(im[15] > 0, 9)] = (1.0, 0.9, 0.0, 1.0)
    ax.imshow(ov, interpolation="nearest")


def _mask_iou(teacher: np.ndarray, pred: np.ndarray) -> float:
    inter = float((teacher & pred).sum())
    union = float((teacher | pred).sum()) + 1e-9
    return inter / union


def _resolve_checkpoint(run_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"missing checkpoint {explicit}")
        return explicit
    final = Path(config.harness_checkpoint_dir) / f"{run_id}.pt"
    if final.is_file():
        return final
    series = Path(config.harness_checkpoint_dir) / run_id
    if series.is_dir():
        steps = sorted(series.glob("step*.pt"), key=lambda p: int(p.stem.removeprefix("step")))
        if steps:
            return steps[-1]
    raise SystemExit(f"missing checkpoint for {run_id} (no {final} or step*.pt under {series})")


@torch.no_grad()
def run_rollout(
    model,
    loader: C4MultiTaskGpuValLoader,
    *,
    route_ix: int,
    grid: int,
    crop: int,
    start_pass: int = 1,
) -> list[dict]:
    goal_m = _wave_val_goal_m(route_ix)
    n_goal = n_goal_from_m(goal_m, WAVE_HORIZON_M)
    schedule = wave_pass_schedule(n_goal)
    start_pass = wave_rollout_start_pass_for_route(route_ix, start_pass)
    if start_pass < 1 or start_pass > len(schedule):
        raise ValueError(f"start_pass must be 1..{len(schedule)}, got {start_pass}")
    device = loader.device
    core = getattr(model, "_orig_mod", model)
    fg = core.config.foreground_class
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    markers = loader.route["markers"][route_ix : route_ix + 1]
    pad_pred = _wave_start_disc(markers, grid, crop)[0]
    rows: list[dict] = []

    for pass_i, wp in enumerate(schedule, start=1):
        img_t, tgt_t, task_id, cond, tok_gt = loader.fetch_wave_pass(
            route_ix, wp.task_key, force_k_in=wp.k_in, force_j=wp.j_back,
        )
        if pass_i < start_pass:
            pad_pred = _teacher_mask_to_grid(tgt_t, grid, crop)
            continue
        task_id = task_id.unsqueeze(0)
        cond = cond.unsqueeze(0)
        img_m = img_t.clone()
        img_m[16] = _pad_grid_to_img(pad_pred, crop, grid)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, _ = model_forward(model, img_m.unsqueeze(0), None, task_id, cond)
            tl = token_logits(model, img_m.unsqueeze(0), task_id, cond)
        prob = _prob_fg(logits[0], fg)
        pred = (prob >= 0.5)
        pad_pred = pred
        tok_pred = int(tl.argmax(dim=-1).item()) if tl is not None else -1
        rows.append(
            {
                "pass": wp,
                "route_ix": route_ix,
                "n_goal": n_goal,
                "goal_m": goal_m,
                "img_in": img_m,
                "teacher": tgt_t.cpu().numpy() > 0,
                "pred": pred.cpu().numpy(),
                "prob": prob.float().cpu().numpy(),
                "tok_gt": tok_gt,
                "tok_pred": tok_pred,
                "tok_gt_name": _TOK_NAMES.get(tok_gt, "?"),
                "tok_pred_name": _TOK_NAMES.get(tok_pred, "?"),
            }
        )
    return rows


def wave_rollout_summary(rows: list[dict]) -> dict[str, float | int]:
    ious: list[float] = []
    tok_ok = 0
    for row in rows:
        t, p = row["teacher"], row["pred"]
        inter = float((t & p).sum())
        union = float((t | p).sum()) + 1e-9
        ious.append(inter / union)
        tok_ok += int(row["tok_pred"] == row["tok_gt"])
    n = len(rows)
    return {
        "wave_rollout_passes": n,
        "wave_rollout_mask_iou_mean": sum(ious) / n if n else 0.0,
        "wave_rollout_tok_acc": tok_ok / n if n else 0.0,
    }


def build_wave_rollout_loader():
    """Val loader with all six wave passes (teacher), on the current c4 route store."""
    crops_dir = config.c3_crops_dir()
    routes_dir = config.c4_routes_dir()
    enc, g = config.encoding_mode, config.grid_out_size
    device = data.device
    district = load_district_planes_v4(crops_dir, device)
    val_routes = load_c4_route_index(routes_dir, "val", device, max_items=config.worldsnap_max_val_items)
    return C4MultiTaskGpuValLoader(
        list(WAVE_ROLLOUT_TASK_KEYS), enc, g, 1, district=district, route_index=val_routes,
    ), val_routes


def save_rollout_figure(
    rows: list[dict],
    *,
    run_id: str,
    route_ix: int,
    out: Path,
    val_loader,
    start_pass: int = 1,
) -> Path:
    device = val_loader.device
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n), dpi=100)
    if n == 1:
        axes = np.array([axes])
    for i, row in enumerate(rows):
        wp: WavePass = row["pass"]
        ax0, ax1, ax2 = axes[i]
        _plot_c4_input(ax0, row["img_in"], route_ix, val_loader, device)
        ax0.set_title(f"route {route_ix} · {wp.task_key}\n{wp.label}", fontsize=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax1.imshow(row["teacher"].astype(np.float32), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("teacher mask (this pass target)", fontsize=8)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.imshow(row["prob"], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        tok_ok = row["tok_pred"] == row["tok_gt"]
        iou = _mask_iou(row["teacher"], row["pred"])
        tok_line = (
            f"token: pred {row['tok_pred_name']}, gt {row['tok_gt_name']}"
            + (" (ok)" if tok_ok else " (miss)")
        )
        ax2.set_title(
            f"P(fg) per cell (model mask logits)\n{tok_line}\n"
            f"IoU@0.5 vs teacher {iou:.2f} → fed back as scratchpad next pass",
            fontsize=7,
        )
        ax2.set_xticks([]); ax2.set_yticks([])
    r0 = rows[0]
    start_note = f" · from pass {start_pass} (scratchpad seeded by teacher through pass {start_pass - 1})" if start_pass > 1 else ""
    fig.suptitle(
        f"{run_id} · iterative wave search · route {route_ix} · n_goal={r0['n_goal']} · "
        f"{r0['goal_m']:.0f} m walking{start_note}  (markers yellow, scratchpad blue)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C5 wave search iterative rollout on one val route")
    p.add_argument("run_id")
    p.add_argument("--route-index", type=int, default=None, help="val route index (default: all from config route_indices, else 0)")
    p.add_argument(
        "--start-pass",
        type=int,
        default=None,
        help="1-based first model pass; default from config (int or first_back)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to runs/checkpoints/<run_id>.pt, else latest step*.pt in that run's series folder",
    )
    args = p.parse_args(argv)
    overrides = _overrides_for_run(args.run_id)
    meta_path = Path(config.diagnostics_dir) / args.run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    saved_meta = _apply_meta_config(meta) if meta else {}
    saved_ov: dict = {}
    for k, v in overrides.items():
        if hasattr(config, k):
            saved_ov[k] = getattr(config, k)
            setattr(config, k, v)
    try:
        config.sync_in_chans_from_encoding()
        data.setup_device()
        device = data.device
        g = config.grid_out_size
        loader, val_routes = build_wave_rollout_loader()
        route_list = [args.route_index] if args.route_index is not None else wave_rollout_route_indices()
        start_cfg = getattr(config, "diagnostics_wave_rollout_start_pass", 1)
        ckpt = _resolve_checkpoint(args.run_id, args.checkpoint)
        print(f"checkpoint: {ckpt}")
        doc = read_checkpoint(ckpt, map_location="cpu")
        cfg = checkpoint_model_config(doc) or dict(meta.get("model") or {})
        model = LocalGridViT(LocalGridViTConfig(**cfg)).to(device)
        grid_vit_adapter.load_checkpoint(ckpt, model=model)
        model.eval()
        crop = int(val_routes.get("crop_px", config.img_size))
        rc = 0
        for route_ix in route_list:
            if route_ix < 0 or route_ix >= loader.n_routes:
                print(f"skip route-index {route_ix} (out of range 0..{loader.n_routes - 1})")
                rc = 1
                continue
            start_pass = args.start_pass if args.start_pass is not None else wave_rollout_start_pass_for_route(route_ix, start_cfg)
            rows = run_rollout(
                model, loader, route_ix=route_ix, grid=g, crop=crop, start_pass=start_pass,
            )
            if args.out is not None and len(route_list) == 1:
                path_out = args.out
            else:
                path_out = Path(config.diagnostics_dir) / args.run_id / f"wave_rollout_route{route_ix:03d}.png"
            path = save_rollout_figure(
                rows, run_id=args.run_id, route_ix=route_ix, out=path_out, val_loader=loader,
                start_pass=args.start_pass,
            )
            print(f"saved {path} ({len(rows)} passes)")
        return rc
    finally:
        for k, v in saved_ov.items():
            setattr(config, k, v)
        for k, v in saved_meta.items():
            setattr(config, k, v)


if __name__ == "__main__":
    raise SystemExit(main())
