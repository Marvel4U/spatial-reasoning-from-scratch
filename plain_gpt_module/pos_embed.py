"""Fixed 2-D sine-cosine position codes for ViT patch grids (LOCAL_MODEL_GUIDE §4c)."""
from __future__ import annotations

import torch


def sincos_2d(
    g: int,
    n_embd: int,
    *,
    temperature: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Patch-grid positions as vectors of width ``n_embd`` (same as transformer hidden size).

    Spatial layout is always 2-D: each of the N = g×g patches has a row index and a column
    index. Those two coordinates are encoded into **one vector of length n_embd**, not into
    "2 dimensions" of output.

    Encoding uses ``n_freq = n_embd // 4`` wavelengths in a geometric series (temperature).
    For each frequency, row and column each get sin and cos → 4 × n_freq = n_embd numbers:
    [x_sin, x_cos, y_sin, y_cos] concatenated. More frequencies → finer position detail in
    the vector; n_embd must be divisible by 4 (e.g. 384 → 96 frequencies per axis pair).

    Returns (N, n_embd) in row-major patch order.
    """
    if g < 1:
        raise ValueError(f"g must be >= 1, got {g}")
    if n_embd < 4 or n_embd % 4 != 0:
        raise ValueError(f"n_embd must be a positive multiple of 4, got {n_embd}")
    dev = device if device is not None else "cpu"
    y_idx, x_idx = torch.meshgrid(
        torch.arange(g, device=dev, dtype=dtype),
        torch.arange(g, device=dev, dtype=dtype),
        indexing="ij",
    )
    n_freq = n_embd // 4
    freq_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    omega = 1.0 / (temperature ** (torch.arange(n_freq, device=dev, dtype=freq_dtype) / n_freq))
    x = x_idx.reshape(-1, 1).to(freq_dtype) * omega[None, :]
    y = y_idx.reshape(-1, 1).to(freq_dtype) * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


def _smoke() -> None:
    g, n_embd = 16, 384
    pos = sincos_2d(g, n_embd)
    n = g * g
    assert pos.shape == (n, n_embd), pos.shape
    assert pos.dtype == torch.float32
    uniq = torch.unique(pos, dim=0)
    assert uniq.shape[0] == n, f"expected {n} unique rows, got {uniq.shape[0]}"
    i_center = 8 * g + 8
    sim = pos @ pos[i_center]
    assert sim[i_center].item() > sim[0].item()
    assert sim[i_center].item() > sim[-1].item()
    pos_hot = sincos_2d(g, n_embd, temperature=100.0)
    sim_hot = pos_hot @ pos_hot[i_center]
    assert sim_hot[0].item() < sim[0].item()
    print(f"pos_embed smoke ok: shape={tuple(pos.shape)} n_freq={n_embd // 4} dtype={pos.dtype}")


if __name__ == "__main__":
    _smoke()
