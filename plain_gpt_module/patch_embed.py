"""Patch embedding: (B, in_chans, H, W) → (B, N, n_embd) (LOCAL_MODEL_GUIDE §2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """One shared linear map per non-overlapping P×P patch (same weights at every grid cell)."""

    def __init__(self, in_chans: int, patch_size: int, n_embd: int):
        super().__init__()
        if in_chans < 1:
            raise ValueError(f"in_chans must be >= 1, got {in_chans}")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if n_embd < 1:
            raise ValueError(f"n_embd must be >= 1, got {n_embd}")
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.n_embd = n_embd
        patch_dim = patch_size * patch_size * in_chans
        self.proj = nn.Linear(patch_dim, n_embd)

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        p = self.patch_size
        if height % p or width % p:
            raise ValueError(
                f"H and W must be divisible by patch_size={p}, got ({height}, {width})"
            )
        return height // p, width // p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, cin, H, W = img.shape
        if cin != self.in_chans:
            raise ValueError(f"expected in_chans={self.in_chans}, got {cin}")
        p = self.patch_size
        gh, gw = self.grid_size(H, W)
        x = img.view(B, cin, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, gh * gw, cin * p * p)
        return self.proj(x)


def _smoke() -> None:
    B, cin, H, W, p, n_embd = 2, 4, 256, 256, 16, 384
    pe = PatchEmbed(cin, p, n_embd)
    img = torch.randn(B, cin, H, W)
    out = pe(img)
    n = (H // p) * (W // p)
    assert out.shape == (B, n, n_embd), out.shape
    # one patch: manual flatten must match one row of out
    gh, gw = H // p, W // p
    r, c = 3, 5
    patch = img[0, :, r * p : (r + 1) * p, c * p : (c + 1) * p]
    manual = pe.proj(patch.reshape(-1))
    idx = r * gw + c
    assert torch.allclose(out[0, idx], manual, atol=1e-5)
    # wrong H → clear error
    try:
        pe(torch.randn(B, cin, 255, W))
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for bad H")
    n_params = sum(t.numel() for t in pe.parameters())
    print(
        f"patch_embed smoke ok: {tuple(out.shape)} "
        f"patch_dim={p * p * cin} proj={cin * p * p * n_embd + n_embd:,} params"
    )


if __name__ == "__main__":
    _smoke()
