"""Relative position bias: additive mask and learned table."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plain_gpt_module.encdec_attention import BidirectionalSelfAttention, EncDecAttnConfig
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT


def test_zero_attn_bias_same_as_no_bias():
    cfg = EncDecAttnConfig(n_embd=48, n_head=4, dropout=0.0)
    attn = BidirectionalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(2, 8, 48)
    z = torch.zeros(1, cfg.n_head, 8, 8)
    with torch.no_grad():
        a = attn(x)
        b = attn(x, attn_bias_add=z)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_nonzero_attn_bias_changes_output():
    cfg = EncDecAttnConfig(n_embd=48, n_head=4, dropout=0.0)
    attn = BidirectionalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(2, 8, 48)
    bias = torch.zeros(1, cfg.n_head, 8, 8)
    bias[0, 0, 3, 4] = 4.0
    with torch.no_grad():
        a = attn(x)
        b = attn(x, attn_bias_add=bias)
    assert not torch.allclose(a, b, atol=1e-5)


def test_first_block_only_has_rel_embedding():
    cfg_off = LocalGridViTConfig(rel_pos_bias="none")
    cfg_on = LocalGridViTConfig(rel_pos_bias="first")
    off = LocalGridViT(cfg_off)
    on = LocalGridViT(cfg_on)
    assert off.encoder[0].rel_pos_bias is None
    assert on.encoder[0].rel_pos_bias is not None
    assert on.encoder[1].rel_pos_bias is None
    cfg_all = LocalGridViTConfig(rel_pos_bias="all", enc_n_layer=2)
    all_b = LocalGridViT(cfg_all)
    assert all_b.encoder[0].rel_pos_bias is not None
    assert all_b.encoder[1].rel_pos_bias is not None


def test_rel_pos_embedding_zero_then_nonzero():
    cfg = LocalGridViTConfig(
        in_chans=4,
        n_tasks=1,
        c1_task_conditioning="cond_tokens",
        n_cond_emb=13,
        rel_pos_bias="first",
        balanced_ce=False,
    )
    model = LocalGridViT(cfg)
    model.eval()
    block = model.encoder[0]
    assert block.rel_pos_bias is not None
    rel_idx = model.rel_pos_bucket_idx
    with torch.no_grad():
        assert block.rel_pos_bias(rel_idx).abs().max().item() == 0.0
        block.rel_pos_bias.weight[5, 1] = 3.0
        assert block.rel_pos_bias(rel_idx).abs().max().item() > 0.0
