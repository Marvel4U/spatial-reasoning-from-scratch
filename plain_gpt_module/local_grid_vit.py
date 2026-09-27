"""Local grid ViT: patch encoder + dense 64×64 class head (c0)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from plain_gpt_module.encdec_model import EncoderBlock, Seq2SeqConfig
from plain_gpt_module.grid_head import GridHead
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.patch_embed import PatchEmbed
from plain_gpt_module.pos_embed import sincos_2d
from plain_gpt_module.rel_pos_bias import build_rel_pos_bucket_index


class LocalGridViT(nn.Module):
    def __init__(self, config: LocalGridViTConfig):
        super().__init__()
        self.config = config
        g = config.patch_grid
        enc_cfg = Seq2SeqConfig(
            vocab_size=1,
            n_embd=config.n_embd,
            n_head=config.n_head,
            enc_n_layer=config.enc_n_layer,
            dec_n_layer=0,
            max_src_len=config.encoder_seq_len,
            max_tgt_len=1,
            dropout=config.dropout,
            tie_embeddings=False,
        )
        self.patch_embed = PatchEmbed(config.in_chans, config.patch_size, config.n_embd)
        pos = sincos_2d(g, config.n_embd, temperature=config.pos_temperature)
        self.register_buffer("pos_embed", pos, persistent=False)
        rel_idx = build_rel_pos_bucket_index(
            patch_grid=g,
            n_prefix=config.n_cond_token_slots,
            max_offset=config.rel_pos_max_offset,
        )
        self.register_buffer("rel_pos_bucket_idx", rel_idx, persistent=False)
        nb = config.rel_pos_n_buckets if config.rel_pos_bias != "none" else 0
        blocks: list[EncoderBlock] = []
        for layer_i in range(config.enc_n_layer):
            n_b = nb if (config.rel_pos_bias == "all" or (config.rel_pos_bias == "first" and layer_i == 0)) else 0
            blocks.append(EncoderBlock(enc_cfg, rel_pos_n_buckets=n_b))
        self.encoder = nn.ModuleList(blocks)
        self.ln_enc = nn.LayerNorm(config.n_embd)
        self.head = GridHead(config.n_embd, g, config.subcells, config.num_classes)
        cond = config.c1_task_conditioning
        if config.n_tasks > 0 and cond == "flat":
            self.task_emb = nn.Embedding(config.n_tasks, config.n_embd)
            self.layer_emb = None
            self.class_emb = None
            self.cond_emb = None
            self.cond_slot_pos = None
            self.c1_task_layer_ix = None
            self.c1_task_class_ix = None
        elif config.n_tasks > 0 and cond == "factorised":
            self.task_emb = None
            self.layer_emb = nn.Embedding(3, config.n_embd)
            self.class_emb = nn.Embedding(4, config.n_embd)
            self.cond_emb = None
            self.cond_slot_pos = None
            li = config.c1_task_layer_ids
            ci = config.c1_task_class_ids
            if len(li) != config.n_tasks or len(ci) != config.n_tasks:
                raise ValueError("c1_task_layer_ids/class_ids must len n_tasks")
            self.register_buffer("c1_task_layer_ix", torch.tensor(li, dtype=torch.long), persistent=False)
            self.register_buffer("c1_task_class_ix", torch.tensor(ci, dtype=torch.long), persistent=False)
        elif config.n_tasks > 0 and cond == "cond_tokens":
            self.task_emb = None
            self.layer_emb = None
            self.class_emb = None
            self.c1_task_layer_ix = None
            self.c1_task_class_ix = None
            self.cond_emb = nn.Embedding(config.n_cond_emb, config.n_embd)
            self.cond_slot_pos = nn.Embedding(config.n_cond_token_slots, config.n_embd)
        else:
            self.task_emb = None
            self.layer_emb = None
            self.class_emb = None
            self.c1_task_layer_ix = None
            self.c1_task_class_ix = None
            self.cond_emb = None
            self.cond_slot_pos = None
        if config.c1_per_task_fg_weights is not None:
            w = config.c1_per_task_fg_weights
            if len(w) != config.n_tasks and config.n_tasks > 0:
                raise ValueError(f"c1_per_task_fg_weights len {len(w)} != n_tasks {config.n_tasks}")
            self.register_buffer(
                "c1_fg_weight_table",
                torch.tensor(w, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.c1_fg_weight_table = None
        self.apply(self._init_weights)

    @property
    def _residual_sublayers(self) -> int:
        return self.config.enc_n_layer * 2

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "NANOGPT_SCALE_INIT", False):
                std *= (2 * self._residual_sublayers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_patches(
        self,
        img: torch.Tensor,
        task_id: torch.Tensor | None = None,
        *,
        cond_ids: torch.Tensor | None = None,
        encoder_self_attn_only: bool = False,
        encoder_patch_local_only: bool = False,
    ) -> torch.Tensor:
        """Encoder output for the grid head: (B, num_patches, n_embd) after ln_enc."""
        x = self._encode_full(
            img,
            task_id=task_id,
            cond_ids=cond_ids,
            encoder_self_attn_only=encoder_self_attn_only,
            encoder_patch_local_only=encoder_patch_local_only,
        )
        n_prefix = self.config.n_cond_token_slots
        if n_prefix:
            x = x[:, n_prefix:, :]
        return x

    def _encode_full(
        self,
        img: torch.Tensor,
        task_id: torch.Tensor | None = None,
        *,
        cond_ids: torch.Tensor | None = None,
        encoder_self_attn_only: bool = False,
        encoder_patch_local_only: bool = False,
    ) -> torch.Tensor:
        if encoder_self_attn_only and encoder_patch_local_only:
            raise ValueError("use one of encoder_self_attn_only or encoder_patch_local_only")
        patches = self.patch_embed(img)
        if self.config.use_sincos_pos:
            patches = patches + self.pos_embed.unsqueeze(0)
        if self.cond_emb is not None:
            if cond_ids is None:
                raise ValueError("cond_ids (B, 2) required when cond_tokens conditioning is enabled")
            if cond_ids.shape != (img.shape[0], self.config.n_cond_token_slots):
                raise ValueError(
                    f"cond_ids shape {tuple(cond_ids.shape)} != (B, {self.config.n_cond_token_slots})"
                )
            slots = torch.arange(self.config.n_cond_token_slots, device=img.device)
            prefix = self.cond_emb((cond_ids + 1).long()) + self.cond_slot_pos(slots).unsqueeze(0)
            x = torch.cat([prefix, patches], dim=1)
        else:
            x = patches
            if self.task_emb is not None:
                if task_id is None:
                    raise ValueError("task_id required when n_tasks > 0")
                x = x + self.task_emb(task_id.to(dtype=torch.long)).unsqueeze(1)
            elif self.layer_emb is not None:
                if task_id is None:
                    raise ValueError("task_id required when factorised task conditioning is enabled")
                tid = task_id.to(dtype=torch.long)
                te = self.layer_emb(self.c1_task_layer_ix[tid]) + self.class_emb(self.c1_task_class_ix[tid])
                x = x + te.unsqueeze(1)
        n_pre = self.config.n_cond_token_slots if self.cond_emb is not None else 0
        rel_idx = self.rel_pos_bucket_idx
        for block in self.encoder:
            x = block(
                x,
                self_attn_only=encoder_self_attn_only,
                n_prefix_tokens=n_pre,
                patch_local_only=encoder_patch_local_only,
                rel_pos_bucket_idx=rel_idx if block.rel_pos_bias is not None else None,
            )
        return self.ln_enc(x)

    def _soft_dice_fg_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fg = self.config.foreground_class
        probs = F.softmax(logits.float(), dim=-1)[..., fg]
        tgt = (targets == fg).float()
        inter = (probs * tgt).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        return (1.0 - dice).mean()

    def _grid_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task_id: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.config
        flat_logits = logits.reshape(-1, cfg.num_classes)
        flat_tgt = targets.reshape(-1)
        if cfg.grid_loss == "ce_dice":
            ce = F.cross_entropy(flat_logits, flat_tgt)
            return ce + cfg.ce_dice_weight * self._soft_dice_fg_loss(logits, targets)
        if self.c1_fg_weight_table is not None and task_id is not None:
            fg = cfg.foreground_class
            b, g, _ = targets.shape
            cells = g * g
            sample_ix = torch.arange(b, device=targets.device).repeat_interleave(cells)
            fg_w = self.c1_fg_weight_table[task_id.long()[sample_ix]]
            per_cell = torch.where(flat_tgt == fg, fg_w, torch.ones_like(fg_w))
            loss = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
            return (loss * per_cell).mean()
        weight = None
        if cfg.balanced_ce and cfg.num_classes == 2:
            fg = cfg.foreground_class
            n_fg = (targets == fg).sum().to(dtype=logits.dtype).clamp(min=1)
            n_bg = (targets != fg).sum().to(dtype=logits.dtype)
            ratio = n_bg / n_fg
            weight = torch.stack([torch.ones_like(ratio), ratio])
        return F.cross_entropy(flat_logits, flat_tgt, weight=weight)

    def forward(
        self,
        img: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        task_id: torch.Tensor | None = None,
        cond_ids: torch.Tensor | None = None,
        encoder_self_attn_only: bool = False,
        encoder_patch_local_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """img (B,C,H,W); task_id (B,) for loss weights; cond_ids (B,2) when cond_tokens."""
        logits = self.head(self.encode_patches(
            img,
            task_id,
            cond_ids=cond_ids,
            encoder_self_attn_only=encoder_self_attn_only,
            encoder_patch_local_only=encoder_patch_local_only,
        ))
        if targets is None:
            return logits, None
        if targets.shape != logits.shape[:3]:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} != logits spatial {tuple(logits.shape[:3])}"
            )
        if self.config.n_tasks > 0 and task_id is None:
            raise ValueError("task_id required in forward when n_tasks > 0")
        if self.cond_emb is not None and cond_ids is None:
            raise ValueError("cond_ids required in forward when cond_tokens conditioning is enabled")
        loss = self._grid_ce_loss(logits, targets, task_id)
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        verbose: bool = True,
    ) -> torch.optim.AdamW:
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
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )


def _smoke() -> None:
    cfg = LocalGridViTConfig()
    model = LocalGridViT(cfg)
    B = 2
    img = torch.randn(B, cfg.in_chans, cfg.img_size, cfg.img_size)
    tgt = torch.randint(0, cfg.num_classes, (B, cfg.grid_out_size, cfg.grid_out_size))
    logits, loss = model(img, tgt)
    assert logits.shape == (B, cfg.grid_out_size, cfg.grid_out_size, cfg.num_classes)
    assert loss is not None and loss.ndim == 0
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    opt = model.configure_optimizers(0.1, 1e-3, "cpu", verbose=False)
    losses = []
    model.train()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        img = torch.randn(B, cfg.in_chans, cfg.img_size, cfg.img_size)
        tgt = torch.randint(0, cfg.num_classes, (B, cfg.grid_out_size, cfg.grid_out_size))
        _, loss = model(img, tgt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    tok_cfg = LocalGridViTConfig(
        in_chans=13,
        n_tasks=4,
        c1_task_conditioning="cond_tokens",
        c1_per_task_fg_weights=(1.0, 2.0, 3.0, 4.0),
        balanced_ce=False,
    )
    tok = LocalGridViT(tok_cfg)
    cond = torch.tensor([[7, -1], [0, 6]], dtype=torch.long)
    tid = torch.tensor([0, 1], dtype=torch.long)
    img13 = torch.randn(B, 13, tok_cfg.img_size, tok_cfg.img_size)
    logits_t, loss_t = tok(img13, tgt, task_id=tid, cond_ids=cond)
    assert logits_t.shape == (B, tok_cfg.grid_out_size, tok_cfg.grid_out_size, tok_cfg.num_classes)
    full = tok._encode_full(
        torch.randn(B, 13, tok_cfg.img_size, tok_cfg.img_size), task_id=tid, cond_ids=cond,
    )
    assert full.shape == (B, tok_cfg.encoder_seq_len, tok_cfg.n_embd)
    print(
        f"local_grid_vit smoke ok: logits={tuple(logits.shape)} "
        f"loss={loss.item():.4f} params={n_params:,} "
        f"loss_20step {losses[0]:.3f}->{losses[-1]:.3f} "
        f"cond_tokens_seq={full.shape[1]}"
    )


if __name__ == "__main__":
    _smoke()
