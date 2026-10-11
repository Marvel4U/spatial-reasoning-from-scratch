"""Config for the local grid ViT (c0 defaults from docs/archive/LOCAL_EXPERIMENTS.md / IMPLEMENTATION_PLAN.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RelPosBiasMode = Literal["none", "first", "all"]


@dataclass
class LocalGridViTConfig:
    in_chans: int = 4  # scalar noise/surface/estab + marker
    img_size: int = 256
    patch_size: int = 16
    grid_out_size: int = 64
    num_classes: int = 2  # c0: background vs marker cell
    foreground_class: int = 1  # grid target id for the marker / mask foreground
    balanced_ce: bool = True  # c0 batch weights [1, n_bg/n_fg]; off when c1 fixed per-task weights used
    grid_loss: str = "ce"  # ce | ce_dice (L6d: plain CE + soft Dice on fg, no per-task f)
    ce_dice_weight: float = 1.0  # loss = CE + ce_dice_weight * (1 - soft Dice)
    n_tasks: int = 0  # >0 → task conditioning (flat / factorised / cond_tokens)
    c1_task_conditioning: str = "flat"  # flat | factorised | cond_tokens (C2_PLAN §4)
    c1_task_layer_ids: tuple[int, ...] = ()  # len n_tasks when factorised
    c1_task_class_ids: tuple[int, ...] = ()
    c1_per_task_fg_weights: tuple[float, ...] | None = None  # len n_tasks; None = plain CE
    n_embd: int = 384
    n_head: int = 6
    enc_n_layer: int = 2
    dropout: float = 0.0
    pos_temperature: float = 10000.0
    rel_pos_bias: RelPosBiasMode = "none"  # E5b: which encoder blocks get learned 2-D rel bias
    rel_pos_max_offset: int = 7  # K: clip Δrow/Δcol to ±K patches → (2K+1)² buckets + 1 prefix bucket
    use_sincos_pos: bool = True  # V3 ablation: patch tokens get fixed 2-D sin-cos (off when rel-only)
    # C5 M2: optional scalar token head (length bins, later phase tokens) from a readout prefix slot.
    token_head_enabled: bool = False
    n_token_classes: int = 0
    token_loss_weight: float = 1.0
    # C5 wave: upweight cells that differ from scratchpad ch16 (memo CHANGE_WEIGHTED_LOSS_2026-10-02).
    wave_change_group_ce: bool = False  # four-group equal CE share per sample
    wave_change_ce_beta: float = 0.0  # else multiply changing cells by (1 + beta) on top of fg weights
    wave_change_pad_px: int = 0  # with group_ce: fifth halo group = core change dilated by this many px

    def __post_init__(self) -> None:
        if self.in_chans < 1:
            raise ValueError(f"in_chans must be >= 1, got {self.in_chans}")
        if self.img_size < 1 or self.patch_size < 1:
            raise ValueError("img_size and patch_size must be >= 1")
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size must be divisible by patch_size "
                f"({self.img_size} % {self.patch_size} != 0)"
            )
        g = self.patch_grid
        if self.grid_out_size % g != 0:
            raise ValueError(
                f"grid_out_size must be divisible by patch_grid={g}, got {self.grid_out_size}"
            )
        if self.subcells < 1:
            raise ValueError("subcells must be >= 1")
        if self.grid_out_size != g * self.subcells:
            raise ValueError("grid_out_size must equal patch_grid * subcells")
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if not 0 <= self.foreground_class < self.num_classes:
            raise ValueError(
                f"foreground_class must be in [0, {self.num_classes}), got {self.foreground_class}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd must be divisible by n_head ({self.n_embd}, {self.n_head})")
        if self.enc_n_layer < 1:
            raise ValueError(f"enc_n_layer must be >= 1, got {self.enc_n_layer}")
        if self.n_embd < 4 or self.n_embd % 4 != 0:
            raise ValueError(f"n_embd must be a positive multiple of 4, got {self.n_embd}")
        cond = self.c1_task_conditioning
        if cond not in ("flat", "factorised", "cond_tokens"):
            raise ValueError(f"c1_task_conditioning must be flat|factorised|cond_tokens, got {cond!r}")
        if cond == "cond_tokens" and self.n_tasks < 1:
            raise ValueError("cond_tokens conditioning requires n_tasks >= 1 (per-task loss weights)")
        if self.grid_loss not in ("ce", "ce_dice"):
            raise ValueError(f"grid_loss must be ce|ce_dice, got {self.grid_loss!r}")
        if self.ce_dice_weight < 0:
            raise ValueError(f"ce_dice_weight must be >= 0, got {self.ce_dice_weight}")
        if self.rel_pos_bias not in ("none", "first", "all"):
            raise ValueError(f"rel_pos_bias must be none|first|all, got {self.rel_pos_bias!r}")
        if self.rel_pos_max_offset < 0:
            raise ValueError(f"rel_pos_max_offset must be >= 0, got {self.rel_pos_max_offset}")
        if self.token_head_enabled:
            if self.n_token_classes < 2:
                raise ValueError("token_head_enabled requires n_token_classes >= 2")
            if self.c1_task_conditioning != "cond_tokens":
                raise ValueError("C5 token head requires cond_tokens conditioning")
        elif self.n_token_classes != 0:
            raise ValueError("n_token_classes must be 0 when token_head_enabled is False")
        if self.wave_change_group_ce and self.wave_change_ce_beta > 0:
            raise ValueError("use wave_change_group_ce or wave_change_ce_beta, not both")
        if self.wave_change_ce_beta < 0:
            raise ValueError(f"wave_change_ce_beta must be >= 0, got {self.wave_change_ce_beta}")
        if self.wave_change_pad_px < 0:
            raise ValueError(f"wave_change_pad_px must be >= 0, got {self.wave_change_pad_px}")
        if self.wave_change_pad_px > 0 and not self.wave_change_group_ce:
            raise ValueError("wave_change_pad_px requires wave_change_group_ce")

    @property
    def patch_grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def subcells(self) -> int:
        return self.grid_out_size // self.patch_grid

    @property
    def num_patches(self) -> int:
        g = self.patch_grid
        return g * g

    @property
    def n_cond_task_slots(self) -> int:
        """Task condition tokens in the prefix (cond_ids shape is (B, this))."""
        return 2 if self.c1_task_conditioning == "cond_tokens" else 0

    @property
    def n_cond_token_slots(self) -> int:
        """Alias: task cond slots only (not including C5 readout)."""
        return self.n_cond_task_slots

    @property
    def n_prefix_tokens(self) -> int:
        """Task cond prefix plus optional readout slot (C5 M2)."""
        extra = 1 if self.token_head_enabled else 0
        return self.n_cond_task_slots + extra

    @property
    def readout_token_index(self) -> int:
        """Encoder sequence index of the readout token (after task cond slots)."""
        if not self.token_head_enabled:
            raise RuntimeError("readout_token_index undefined without token_head_enabled")
        return self.n_cond_task_slots

    @property
    def encoder_seq_len(self) -> int:
        """Patch tokens plus full prefix (259 @ P=16 with cond_tokens + readout)."""
        return self.num_patches + self.n_prefix_tokens

    @property
    def rel_pos_n_spatial_buckets(self) -> int:
        k = self.rel_pos_max_offset
        return (2 * k + 1) ** 2

    @property
    def rel_pos_n_buckets(self) -> int:
        return self.rel_pos_n_spatial_buckets + 1

    # cond_emb indices: 0 = pad (loader atom id −1); 1…12 = atom ids 0…11.
    n_cond_emb: int = 13


def _smoke() -> None:
    cfg = LocalGridViTConfig()
    assert cfg.patch_grid == 16 and cfg.subcells == 4 and cfg.num_patches == 256
    assert cfg.grid_out_size == 64 and cfg.num_classes == 2
    onehot = LocalGridViTConfig(in_chans=13)
    assert onehot.in_chans == 13
    tok = LocalGridViTConfig(n_tasks=9, c1_task_conditioning="cond_tokens")
    assert tok.encoder_seq_len == 256 + 2
    assert tok.rel_pos_n_buckets == 226  # K=7 → 225 spatial + prefix bucket
    from plain_gpt_module.rel_pos_bias import build_rel_pos_bucket_index
    b = build_rel_pos_bucket_index(patch_grid=16, n_prefix=2, max_offset=7)
    assert b.shape == (258, 258) and int(b[0, 2].item()) == 225 and int(b[2, 2].item()) == 112
    try:
        LocalGridViTConfig(img_size=255)
    except ValueError:
        pass
    else:
        raise AssertionError("expected bad img_size to fail")
    print(
        f"local_config smoke ok: g={cfg.patch_grid} s={cfg.subcells} "
        f"N={cfg.num_patches} K={cfg.num_classes} enc={cfg.enc_n_layer}"
    )


if __name__ == "__main__":
    _smoke()
