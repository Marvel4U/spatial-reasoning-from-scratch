"""c1 forward, mask metrics, and prediction maps for the viewer."""
from __future__ import annotations

import torch

from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT

from .c0_adapter import fg_probability, forward_c0
from .geometry import C0Geometry
from .load_c1_sample import C1LoadedSample


def _binary_masks(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: LocalGridViTConfig,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fg = cfg.foreground_class
    prob = fg_probability(logits, cfg)
    pred = logits.argmax(dim=-1)
    pred_fg = pred == fg
    tgt_fg = target == fg
    return pred_fg, tgt_fg, prob


def c1_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    geom: C0Geometry,
    cfg: LocalGridViTConfig,
    *,
    threshold: float = 0.5,
) -> dict:
    pred_fg, tgt_fg, prob = _binary_masks(logits, target, cfg, threshold=threshold)
    inter = (pred_fg & tgt_fg).sum().item()
    union = (pred_fg | tgt_fg).sum().item()
    iou = inter / union if union > 0 else 1.0
    tp = inter
    fp = (pred_fg & ~tgt_fg).sum().item()
    fn = (~pred_fg & tgt_fg).sum().item()
    prec = tp / (tp + fp) if tp + fp > 0 else 1.0
    rec = tp / (tp + fn) if tp + fn > 0 else 1.0
    n_pred = int(pred_fg.sum().item())
    n_tgt = int(tgt_fg.sum().item())
    empty_tgt = n_tgt == 0
    return {
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "n_pred_fg": n_pred,
        "n_tgt_fg": n_tgt,
        "empty_target": empty_tgt,
        "empty_fp": bool(empty_tgt and n_pred > 0),
        "prob_max": float(prob.max().item()),
        "threshold": threshold,
    }


def error_map_rgb(pred_fg: torch.Tensor, tgt_fg: torch.Tensor) -> torch.Tensor:
    """(G, G) bool masks → (G, G, 3) float in [0, 1]: TP green, FP red, FN blue."""
    g = pred_fg.shape[0]
    out = torch.zeros(g, g, 3)
    tp = pred_fg & tgt_fg
    fp = pred_fg & ~tgt_fg
    fn = ~pred_fg & tgt_fg
    out[tp, 1] = 1.0
    out[fp, 0] = 1.0
    out[fn, 2] = 1.0
    return out


def focal_grid_cell(target: torch.Tensor, fg_class: int, grid_size: int) -> tuple[int, int]:
    fg = target == fg_class
    nz = fg.nonzero(as_tuple=False)
    if nz.numel() == 0:
        c = grid_size // 2
        return c, c
    return int(nz[:, 0].float().mean().item()), int(nz[:, 1].float().mean().item())


@torch.no_grad()
def predict_c1_sample(
    model: LocalGridViT,
    sample: C1LoadedSample,
    cfg: LocalGridViTConfig,
    *,
    device: torch.device,
    task_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    geom = C0Geometry.from_config(cfg)
    tid = None
    if cfg.n_tasks > 0:
        from spatial_data.c1_tasks import CORE_FOUR_KEYS, task_index
        keys = list(CORE_FOUR_KEYS) if cfg.n_tasks == len(CORE_FOUR_KEYS) else [sample.meta.task_key]
        ti = task_id if task_id is not None else task_index(sample.meta.task_key, keys)
        tid = torch.tensor([ti], dtype=torch.long)
    logits = forward_c0(model, sample.img, device=device, task_id=tid)
    tgt = sample.target.to(device=logits.device)
    prob = fg_probability(logits, cfg)
    metrics = c1_metrics(logits, tgt, geom, cfg)
    pred_fg, tgt_fg, _ = _binary_masks(logits, tgt, cfg)
    err = error_map_rgb(pred_fg, tgt_fg)
    return logits.cpu(), prob.cpu(), err.cpu(), metrics
