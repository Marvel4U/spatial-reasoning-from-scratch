"""Encoder-decoder attention primitives (Model B, Phase 1)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class EncDecAttnConfig:
    n_embd: int
    n_head: int
    dropout: float = 0.0


def _split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, C = x.shape
    return x.view(B, T, n_head, C // n_head).transpose(1, 2)


def _merge_heads(x: torch.Tensor, n_embd: int) -> torch.Tensor:
    B, _, T, _ = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, n_embd)


def _key_keep_mask(key_pad: torch.Tensor, n_head: int, q_len: int) -> torch.Tensor:
    """key_pad [B, T_k] True = padded key (ignore). SDPA bool mask: True = attend."""
    B, t_k = key_pad.shape
    keep = ~key_pad
    return keep.view(B, 1, 1, t_k).expand(B, n_head, q_len, t_k)


class BidirectionalSelfAttention(nn.Module):
    """Full self-attention (no causal mask). Optional key padding mask for padded batches."""

    def __init__(self, config: EncDecAttnConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        self_attn_only: bool = False,
        n_prefix_tokens: int = 0,
        patch_local_only: bool = False,
        attn_bias_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = _split_heads(q, self.n_head)
        k = _split_heads(k, self.n_head)
        v = _split_heads(v, self.n_head)
        bool_mask = None
        if self_attn_only:
            bool_mask = torch.eye(T, device=x.device, dtype=torch.bool).view(1, 1, T, T)
            bool_mask = bool_mask.expand(B, self.n_head, T, T)
        elif patch_local_only:
            if n_prefix_tokens <= 0 or n_prefix_tokens >= T:
                raise ValueError("patch_local_only requires 0 < n_prefix_tokens < T")
            idx = torch.arange(T, device=x.device)
            allow = (idx.view(T, 1) == idx.view(1, T)) | (idx.view(1, T) < n_prefix_tokens)
            bool_mask = allow.view(1, 1, T, T).expand(B, self.n_head, T, T)
        if key_padding_mask is not None:
            pad = _key_keep_mask(key_padding_mask, self.n_head, T)
            bool_mask = pad if bool_mask is None else bool_mask & pad
        attn_mask = None
        if attn_bias_add is not None or bool_mask is not None:
            attn_mask = torch.zeros(B, self.n_head, T, T, device=x.device, dtype=q.dtype)
            if attn_bias_add is not None:
                attn_mask = attn_mask + attn_bias_add.to(device=x.device, dtype=q.dtype)
            if bool_mask is not None:
                attn_mask = attn_mask.masked_fill(~bool_mask, torch.finfo(q.dtype).min)
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=drop,
        )
        y = _merge_heads(y, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class CrossAttention(nn.Module):
    """Decoder queries attend to encoder memory. Q from x; K/V from enc_out."""

    def __init__(self, config: EncDecAttnConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, C = x.shape
        q = _split_heads(self.c_q(x), self.n_head)
        k, v = self.c_kv(enc_out).split(self.n_embd, dim=2)
        k = _split_heads(k, self.n_head)
        v = _split_heads(v, self.n_head)
        attn_mask = None
        if enc_key_padding_mask is not None:
            attn_mask = _key_keep_mask(enc_key_padding_mask, self.n_head, T_q)
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=drop,
        )
        y = _merge_heads(y, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


def _smoke():
    cfg = EncDecAttnConfig(n_embd=384, n_head=6, dropout=0.0)
    enc_attn = BidirectionalSelfAttention(cfg)
    cross = CrossAttention(cfg)
    enc_attn.eval()
    cross.eval()
    B, t_src, t_tgt = 2, 10, 8
    src = torch.randn(B, t_src, cfg.n_embd)
    tgt = torch.randn(B, t_tgt, cfg.n_embd)
    src_pad = torch.zeros(B, t_src, dtype=torch.bool)
    src_pad[:, 7:] = True
    enc_out = enc_attn(src, key_padding_mask=src_pad)
    assert enc_out.shape == src.shape
    out = cross(tgt, enc_out, enc_key_padding_mask=src_pad)
    assert out.shape == tgt.shape
    pad_pos = src_pad[0, 7]
    src_no_pad = src.clone()
    src_no_pad[:, 7:, :] = 999.0
    enc_alt = enc_attn(src_no_pad, key_padding_mask=src_pad)
    assert torch.allclose(enc_out[0, 0], enc_alt[0, 0], atol=1e-5)
    print("encdec_attention smoke ok:", enc_out.shape, out.shape)


if __name__ == "__main__":
    _smoke()
