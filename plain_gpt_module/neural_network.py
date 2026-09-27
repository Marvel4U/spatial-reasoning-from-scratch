from dataclasses import dataclass
import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head ==0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1,1, config.block_size, config.block_size))

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.size() # batch, sequence, embdedding dimensions n_embd
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn_dropout = self.dropout if self.training else 0.0
        attn_mask = None
        if key_padding_mask is not None:
            keep = ~key_padding_mask
            attn_mask = keep.view(B, 1, 1, T).expand(B, self.n_head, T, T)
        if False:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if attn_dropout > 0:
                att = F.dropout(att, p=attn_dropout, training=True)
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=True, dropout_p=attn_dropout,
            ) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def forward_cached(self, x, past_kv=None):
        """Inference-only: optional KV cache from prior tokens (B, nh, T_past, hs)."""
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        T_q, T_k = q.size(2), k.size(2)
        if past_len == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        elif T_q == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
        else:
            bias = torch.zeros(T_q, T_k, device=x.device, dtype=q.dtype)
            for i in range(T_q):
                bias[i, past_len + i + 1:] = float("-inf")
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.view(1, 1, T_q, T_k), dropout_p=0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), (k, v)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_cached(self, x, past_kv=None):
        attn_out, new_kv = self.attn.forward_cached(self.ln_1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv
    


@dataclass
class GPTConfig:
    block_size: int = 1024 #256
    vocab_size: int = 50304 #50257 #65
    n_layer: int = 12 #6
    n_head:  int = 12 #6
    n_embd:  int = 768 #384
    tie_embeddings: bool = True
    lm_head_bias: bool = False
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,  'NANOGPT_SCALE_INIT'): 
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}, block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, idx, past_key_values=None):
        """Autoregressive inference with per-layer KV cache. idx: (B, T_new)."""
        B, T = idx.size()
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        if past_len + T > self.config.block_size:
            raise ValueError(f"sequence {past_len + T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        new_past = []
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values else None
            x, layer_kv = block.forward_cached(x, layer_past)
            new_past.append(layer_kv)
        x = self.transformer.ln_f(x)
        return self.lm_head(x), tuple(new_past)

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
