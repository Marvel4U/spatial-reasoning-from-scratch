"""Encoder-decoder Seq2Seq model (Model B, Phase 2)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .encdec_attention import BidirectionalSelfAttention, CrossAttention, EncDecAttnConfig
from .neural_network import CausalSelfAttention, MLP


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    n_embd: int = 384
    n_head: int = 6
    enc_n_layer: int = 6
    dec_n_layer: int = 6
    max_src_len: int = 128
    max_tgt_len: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True
    lm_head_bias: bool = False
    pad_id: int = 0


def _attn_cfg(cfg: Seq2SeqConfig) -> GPTConfigShim:
    return GPTConfigShim(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.max_tgt_len)


class GPTConfigShim:
    """Minimal config surface for CausalSelfAttention / MLP (Model A modules)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size


class EncoderBlock(nn.Module):
    def __init__(self, config: Seq2SeqConfig, *, rel_pos_n_buckets: int = 0):
        super().__init__()
        ac = EncDecAttnConfig(config.n_embd, config.n_head, config.dropout)
        mlp_cfg = _attn_cfg(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(ac)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(mlp_cfg)
        if rel_pos_n_buckets > 0:
            self.rel_pos_bias = nn.Embedding(rel_pos_n_buckets, config.n_head)
            nn.init.zeros_(self.rel_pos_bias.weight)
        else:
            self.rel_pos_bias = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        self_attn_only: bool = False,
        n_prefix_tokens: int = 0,
        patch_local_only: bool = False,
        rel_pos_bucket_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_bias = None
        if self.rel_pos_bias is not None:
            if rel_pos_bucket_idx is None:
                raise ValueError("rel_pos_bucket_idx required when block has rel_pos_bias")
            idx = rel_pos_bucket_idx.to(device=x.device, dtype=torch.long)
            attn_bias = self.rel_pos_bias(idx).permute(2, 0, 1).unsqueeze(0).to(dtype=x.dtype)
        x = x + self.attn(
            self.ln_1(x),
            key_padding_mask=key_padding_mask,
            self_attn_only=self_attn_only,
            n_prefix_tokens=n_prefix_tokens,
            patch_local_only=patch_local_only,
            attn_bias_add=attn_bias,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        ac = EncDecAttnConfig(config.n_embd, config.n_head, config.dropout)
        mlp_cfg = _attn_cfg(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(mlp_cfg)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(ac)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(mlp_cfg)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln_1(x), key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_key_padding_mask=enc_key_padding_mask)
        x = x + self.mlp(self.ln_3(x))
        return x


class Seq2Seq(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe_src = nn.Embedding(config.max_src_len, config.n_embd)
        self.wpe_tgt = nn.Embedding(config.max_tgt_len, config.n_embd)
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.enc_n_layer)])
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.dec_n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    @property
    def _residual_sublayers(self) -> int:
        c = self.config
        return c.enc_n_layer * 2 + c.dec_n_layer * 3

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self._residual_sublayers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, t_src = src_ids.shape
        assert t_src <= self.config.max_src_len
        pos = torch.arange(t_src, device=src_ids.device, dtype=torch.long)
        x = self.wte(src_ids) + self.wpe_src(pos)
        for block in self.encoder:
            x = block(x, key_padding_mask=src_key_padding_mask)
        return x

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, t_tgt = tgt_ids.shape
        assert t_tgt <= self.config.max_tgt_len
        pos = torch.arange(t_tgt, device=tgt_ids.device, dtype=torch.long)
        x = self.wte(tgt_ids) + self.wpe_tgt(pos)
        for block in self.decoder:
            x = block(x, memory, src_key_padding_mask, tgt_key_padding_mask)
        return self.ln_f(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory = self.encode(src_ids, src_key_padding_mask)
        x = self.decode(tgt_ids, memory, src_key_padding_mask, tgt_key_padding_mask)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str, verbose: bool = True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if verbose:
            nd = sum(p.numel() for p in decay_params)
            nn_ = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {nd:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {nn_:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


def _smoke():
    cfg = Seq2SeqConfig(vocab_size=4096, n_embd=384, n_head=6, enc_n_layer=6, dec_n_layer=6, dropout=0.0)
    model = Seq2Seq(cfg)
    model.eval()
    B, t_src, t_tgt = 2, 12, 10
    src = torch.randint(1, cfg.vocab_size, (B, t_src))
    tgt_full = torch.randint(1, cfg.vocab_size, (B, t_tgt))
    tgt_in = tgt_full[:, :-1]
    targets = tgt_full[:, 1:]
    src_pad = torch.zeros(B, t_src, dtype=torch.bool)
    src_pad[:, 9:] = True
    tgt_pad = torch.zeros(B, t_tgt - 1, dtype=torch.bool)
    tgt_pad[:, 7:] = True
    logits, loss = model(src, tgt_in, src_pad, tgt_pad, targets)
    assert logits.shape == (B, t_tgt - 1, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encdec_model smoke ok: logits={tuple(logits.shape)} loss={loss.item():.4f} params={n_params:,}")


if __name__ == "__main__":
    _smoke()
