"""Balanced CE weights stay on device (no .item() sync per forward)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT


def test_c0_weight_ratio_matches_grid_geometry():
    cfg = LocalGridViTConfig(balanced_ce=True, num_classes=2)
    model = LocalGridViT(cfg)
    b, g = 4, cfg.grid_out_size
    img = torch.randn(b, cfg.in_chans, cfg.img_size, cfg.img_size)
    tgt = torch.zeros(b, g, g, dtype=torch.long)
    for i in range(b):
        tgt[i, i % g, (i * 3) % g] = cfg.foreground_class
    logits, loss = model(img, tgt)
    assert loss.ndim == 0
    fg = cfg.foreground_class
    n_fg = (tgt == fg).sum().float()
    n_bg = (tgt != fg).sum().float()
    expected_ratio = (n_bg / n_fg).item()
    assert abs(expected_ratio - (g * g - 1)) < 1e-3
    flat_logits = logits.reshape(-1, cfg.num_classes)
    flat_tgt = tgt.reshape(-1)
    w_old = torch.tensor([1.0, expected_ratio], dtype=logits.dtype)
    loss_ref = torch.nn.functional.cross_entropy(flat_logits, flat_tgt, weight=w_old)
    assert torch.allclose(loss, loss_ref, rtol=0, atol=1e-5)


def test_balanced_ce_cuda_smoke():
    if not torch.cuda.is_available():
        return
    cfg = LocalGridViTConfig(balanced_ce=True)
    model = LocalGridViT(cfg).cuda()
    img = torch.randn(2, cfg.in_chans, cfg.img_size, cfg.img_size, device="cuda")
    tgt = torch.zeros(2, cfg.grid_out_size, cfg.grid_out_size, dtype=torch.long, device="cuda")
    tgt[0, 10, 10] = 1
    tgt[1, 20, 30] = 1
    _, loss = model(img, tgt)
    assert loss.device.type == "cuda"
    loss.backward()
