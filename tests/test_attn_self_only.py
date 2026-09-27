"""Encoder self-attn-only mask: output differs from full attention."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plain_gpt_module.encdec_attention import BidirectionalSelfAttention, EncDecAttnConfig


def test_self_attn_only_changes_output():
    cfg = EncDecAttnConfig(n_embd=48, n_head=4, dropout=0.0)
    attn = BidirectionalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(2, 8, 48)
    full = attn(x)
    diag = attn(x, self_attn_only=True)
    assert full.shape == diag.shape
    assert not torch.allclose(full, diag, atol=1e-5)
