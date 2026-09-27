"""c0 forward, metrics, and prediction maps for the viewer."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT

from .geometry import C0Geometry
from .load_sample import LoadedSample


@torch.no_grad()
def forward_c0(
    model: LocalGridViT,
    img: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    task_id: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    x = img.unsqueeze(0).to(device=device, dtype=dtype)
    tid = None
    if task_id is not None:
        tid = task_id.to(device=device, dtype=torch.long).reshape(-1)
        if tid.numel() != x.shape[0]:
            tid = tid[: x.shape[0]] if tid.numel() >= x.shape[0] else tid.expand(x.shape[0])
    elif getattr(model, "task_emb", None) is not None:
        tid = torch.zeros(x.shape[0], device=device, dtype=torch.long)
    logits, _ = model(x, None, task_id=tid)
    return logits[0]


def fg_probability(logits: torch.Tensor, cfg: LocalGridViTConfig) -> torch.Tensor:
    fg = cfg.foreground_class
    if cfg.num_classes > 1:
        score = logits[..., fg] - logits[..., 0]
        prob = torch.sigmoid(score)
    else:
        prob = F.softmax(logits, dim=-1)[..., 0]
    return prob


def c0_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    geom: C0Geometry,
    meta_row: int,
    meta_col: int,
    cfg: LocalGridViTConfig,
) -> dict:
    fg = cfg.foreground_class
    pred = logits.argmax(dim=-1)
    G = geom.grid_size
    s = geom.subcells
    prob = fg_probability(logits, cfg)
    if cfg.num_classes > 1:
        score = (logits[..., fg] - logits[..., 0]).reshape(-1)
    else:
        score = logits.reshape(-1)
    am = int(score.argmax().item())
    tidx = int(target.reshape(-1).argmax().item())
    tr, tc = tidx // G, tidx % G
    ar, ac = am // G, am % G
    pred_fg = pred == fg
    n_fg = int(pred_fg.sum().item())
    ph = (ar // s == tr // s) and (ac // s == tc // s)
    block_cells = s * s
    return {
        "n_fg": n_fg,
        "patch_hit": int(ph),
        "subcell_hit": int(am == tidx) if ph else 0,
        "subcell_given_patch": int(am == tidx) if ph else None,
        "global_argmax_exact": int(am == tidx),
        "argmax_dist_cells": float(math.hypot(ar - tr, ac - tc)),
        "true_cell_recall": int(pred[tr, tc].item() == fg),
        "full_patch_frac": float(n_fg == block_cells),
        "chance_subcell": 1.0 / block_cells,
        "prob_max": float(prob.max().item()),
        "marker_row": meta_row,
        "marker_col": meta_col,
        "target_cell": (tr, tc),
        "argmax_cell": (ar, ac),
    }


def predict_sample(
    model: LocalGridViT,
    sample: LoadedSample,
    cfg: LocalGridViTConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    geom = C0Geometry.from_config(cfg)
    logits = forward_c0(model, sample.img, device=device)
    prob = fg_probability(logits, cfg)
    metrics = c0_metrics(
        logits, sample.target, geom,
        sample.meta.marker_row, sample.meta.marker_col, cfg,
    )
    return logits.cpu(), prob.cpu(), metrics
