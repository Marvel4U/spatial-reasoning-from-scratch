import torch

from plain_gpt_module.wave_pad_ce import IGNORE_TARGET, change_group_cell_weights


def test_change_lit_weighs_more_than_stay_unlit():
    fg = 1
    g = 4
    img = torch.zeros(1, 17, 8, 8)
    img[0, 16, 0:2, 0:2] = 1.0
    targets = torch.zeros(1, g, g, dtype=torch.long)
    targets[0, 0, 0] = fg
    targets[0, 0, 1] = fg
    w = change_group_cell_weights(img, targets, foreground_class=fg)
    assert w is not None
    assert w[0, 0, 1] > w[0, 2, 2]


def test_no_scratchpad_channel_returns_none():
    img = torch.zeros(1, 16, 8, 8)
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    assert change_group_cell_weights(img, targets, foreground_class=1) is None


def test_change_pad_px_upweights_halo_neighbor():
    fg = 1
    g = 8
    img = torch.zeros(1, 17, g, g)
    img[0, 16, 0, 0] = 1.0
    targets = torch.zeros(1, g, g, dtype=torch.long)
    targets[0, 0, 0] = fg
    targets[0, 0, 1] = fg
    w0 = change_group_cell_weights(img, targets, foreground_class=fg, change_pad_px=0)
    w1 = change_group_cell_weights(img, targets, foreground_class=fg, change_pad_px=2)
    assert w0 is not None and w1 is not None
    assert w1[0, 0, 2] > w0[0, 0, 2]
