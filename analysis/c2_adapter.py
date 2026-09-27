"""c2 forward and mask metrics for the viewer (cond_tokens + cond_ids)."""
from __future__ import annotations

import torch

from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT

from .c0_adapter import fg_probability
from .c1_adapter import c1_metrics, error_map_rgb, focal_grid_cell, _binary_masks
from .geometry import C0Geometry
from .load_c2_sample import C2LoadedSample


@torch.no_grad()
def predict_c2_sample(
    model: LocalGridViT,
    sample: C2LoadedSample,
    cfg: LocalGridViTConfig,
    *,
    device: torch.device,
    task_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    geom = C0Geometry.from_config(cfg)
    img = sample.img.unsqueeze(0).to(device)
    a, b = sample.cond_ids
    cond = torch.tensor([[a, b]], dtype=torch.long, device=device)
    tid = torch.tensor([task_index], dtype=torch.long, device=device)
    logits, _ = model(img, None, task_id=tid, cond_ids=cond)
    logits = logits.squeeze(0)
    tgt = sample.target.to(device=logits.device)
    metrics = c1_metrics(logits.unsqueeze(0), tgt.unsqueeze(0), geom, cfg)
    pred_fg, tgt_fg, prob = _binary_masks(logits.unsqueeze(0), tgt.unsqueeze(0), cfg)
    prob = prob.squeeze(0)
    err = error_map_rgb(pred_fg.squeeze(0), tgt_fg.squeeze(0))
    return logits.cpu(), prob.cpu(), err.cpu(), metrics
