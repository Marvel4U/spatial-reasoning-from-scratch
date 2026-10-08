"""Smoke tests for C5 wave training pad noise (_wave_pad_noise). CPU-only."""
from __future__ import annotations

import torch

from spatial_data.dataset_c4_gpu import _shift_wave_pad_cells, _wave_pad_noise


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _plus(g: int = 32) -> torch.Tensor:
    m = torch.zeros(g, g, dtype=torch.bool)
    m[g // 2, g // 2 - 2 : g // 2 + 3] = True
    m[g // 2 - 2 : g // 2 + 3, g // 2] = True
    return m


def test_shift_cells_moves_mask_without_wrap():
    g = 16
    m = torch.zeros(g, g, dtype=torch.bool)
    m[5, 7] = True
    s = _shift_wave_pad_cells(m, 0, 1)
    assert not m[5, 8] and s[5, 8] and int(s.sum()) == 1
    s2 = _shift_wave_pad_cells(m, -1, 0)
    assert s2[4, 7] and int(s2.sum()) == 1


def test_p_zero_leaves_pad_unchanged():
    pad = _plus().unsqueeze(0).repeat(4, 1, 1)
    out = _wave_pad_noise(pad, _gen(0), p=0.0, thicken=1.0, shift=1.0, drop=1.0, add=1.0)
    assert torch.equal(out, pad)


def test_thicken_only_adds_cells():
    pad = _plus().unsqueeze(0)
    out = _wave_pad_noise(pad, _gen(1), p=1.0, thicken=1.0, shift=0.0, drop=0.0, add=0.0)
    assert int(out.sum()) > int(pad.sum()) and (out & ~pad).any() and not (pad & ~out).any()


def test_drop_only_removes_lit_cells():
    pad = _plus().unsqueeze(0).repeat(8, 1, 1)
    n0 = int(pad[0].sum())
    out = _wave_pad_noise(pad, _gen(2), p=1.0, thicken=0.0, shift=0.0, drop=0.5, add=0.0)
    assert (out.sum(dim=(1, 2)) < n0).all() and (out & pad).any()


def test_add_only_speckles_empty_ground():
    pad = _plus().unsqueeze(0).repeat(16, 1, 1)
    out = _wave_pad_noise(pad, _gen(3), p=1.0, thicken=0.0, shift=0.0, drop=0.0, add=0.05)
    extra = out & ~pad
    assert int(extra.sum()) > 0 and not (extra & pad).any()


def test_shift_only_translates_cardinally():
    g = 32
    pad = torch.zeros(64, g, g, dtype=torch.bool)
    pad[:, g // 2, g // 2] = True
    out = _wave_pad_noise(pad, _gen(4), p=1.0, thicken=0.0, shift=1.0, drop=0.0, add=0.0)
    for i in range(pad.shape[0]):
        a, b = pad[i], out[i]
        assert int(a.sum()) == 1 and int(b.sum()) == 1
        dy = (b.nonzero(as_tuple=False)[0, 0] - a.nonzero(as_tuple=False)[0, 0]).item()
        dx = (b.nonzero(as_tuple=False)[0, 1] - a.nonzero(as_tuple=False)[0, 1]).item()
        assert (dy, dx) in {(-1, 0), (1, 0), (0, -1), (0, 1)}


def test_combined_recipe_differs_from_clean():
    pad = _plus().unsqueeze(0).repeat(32, 1, 1)
    kw = {"p": 0.5, "thicken": 0.5, "shift": 0.5, "drop": 0.05, "add": 0.002}
    out = _wave_pad_noise(pad, _gen(5), **kw)
    touched = out != pad
    assert touched.any(dim=(1, 2)).float().mean() > 0.2
