"""Per-patch linear head → dense H×W class grid (LOCAL_MODEL_GUIDE §7c)."""
from __future__ import annotations

import torch
import torch.nn as nn


def fold_patch_logits(logits: torch.Tensor, patch_grid: int, subcells: int) -> torch.Tensor:
    """(B, N, subcells²·K) with N = patch_grid² → (B, patch_grid·subcells, patch_grid·subcells, K)."""
    if logits.ndim != 3:
        raise ValueError(f"expected logits (B, N, C), got shape {tuple(logits.shape)}")
    B, n, tail = logits.shape
    g, s = patch_grid, subcells
    if n != g * g:
        raise ValueError(f"expected N={g * g}, got {n}")
    if tail % (s * s) != 0:
        raise ValueError(f"last dim must be divisible by subcells²={s * s}, got {tail}")
    k = tail // (s * s)
    x = logits.view(B, g, g, s, s, k)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, g * s, g * s, k)


class GridHead(nn.Module):
    """Each patch token predicts a subcells×subcells tile of class logits on the output grid."""

    def __init__(
        self,
        n_embd: int,
        patch_grid: int,
        subcells: int,
        num_classes: int,
    ):
        super().__init__()
        if patch_grid < 1 or subcells < 1 or num_classes < 1:
            raise ValueError("patch_grid, subcells, num_classes must be >= 1")
        self.patch_grid = patch_grid
        self.subcells = subcells
        self.num_classes = num_classes
        self.out_size = patch_grid * subcells
        self.proj = nn.Linear(n_embd, subcells * subcells * num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B, patch_grid², n_embd) → logits (B, out_size, out_size, num_classes)."""
        return fold_patch_logits(self.proj(feats), self.patch_grid, self.subcells)


def _smoke() -> None:
    B, g, s, n_embd, k = 2, 16, 4, 384, 2
    head = GridHead(n_embd, g, s, k)
    feats = torch.randn(B, g * g, n_embd)
    out = head(feats)
    assert out.shape == (B, g * s, g * s, k), out.shape
    assert head.out_size == 64
    # alignment: one patch, one subcell, one class
    raw = torch.zeros(1, g * g, s * s * k)
    r, c, sr, sc, cls = 2, 5, 1, 2, 1
    pidx = r * g + c
    flat = (sr * s + sc) * k + cls
    raw[0, pidx, flat] = 1.0
    folded = fold_patch_logits(raw, g, s)
    assert folded.shape == (1, 64, 64, k)
    assert folded[0, r * s + sr, c * s + sc, cls].item() == 1.0
    assert folded[0, 0, 0, cls].item() == 0.0
    n_params = sum(t.numel() for t in head.parameters())
    print(
        f"grid_head smoke ok: out={tuple(out.shape)} "
        f"out_size={head.out_size} head_params={n_params:,}"
    )


if __name__ == "__main__":
    _smoke()
