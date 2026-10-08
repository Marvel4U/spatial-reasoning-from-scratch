"""C5 rung 0/1: 17-channel encoding with scratchpad, ray targets equal to the numpy reference."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_data.channels import in_chans_for_encoding
from spatial_data.c4_tasks import get_c4_task, path_pix_to_grid_numpy, rasterise_segment_numpy, C4TargetKind
from spatial_data.dataset_c1_gpu import build_image_from_crops
from spatial_data.dataset_c4_gpu import _rays_targets, polyline_scratch


def test_encoding_sp_has_17_channels_marker_at_15_scratch_at_16():
    assert in_chans_for_encoding("onehot_v4") == 16
    assert in_chans_for_encoding("onehot_v4_sp") == 17
    layers = torch.zeros(2, 3, 256, 256, dtype=torch.uint8)
    scratch = torch.zeros(2, 256, 256); scratch[0, 10, 20] = 1.0
    img = build_image_from_crops(layers, "onehot_v4_sp", marker_col=torch.tensor([[5, -1], [7, 9]]),
                                 marker_row=torch.tensor([[6, -1], [8, 10]]), scratch=scratch)
    assert img.shape == (2, 17, 256, 256)
    assert img[0, 15].sum() > 0 and img[1, 15].sum() > 0, "marker plane is channel 15"
    assert img[0, 16, 10, 20] == 1.0 and img[1, 16].sum() == 0, "scratchpad is channel 16"
    v4 = build_image_from_crops(layers, "onehot_v4", marker_col=torch.tensor([[5, -1], [7, 9]]), marker_row=torch.tensor([[6, -1], [8, 10]]))
    assert torch.equal(v4, img[:, :16]), "the first 16 channels are identical to onehot_v4"


def test_rays_task_spec():
    sp = get_c4_task("rays")
    assert sp.target_kind == C4TargetKind.RAYS and sp.padded_cond_ids()[1] == 21


def test_rays_targets_match_numpy():
    rng = np.random.default_rng(0)
    B, K = 4, 6
    markers = rng.uniform(10, 240, size=(B, 2, 2)).astype(np.float32)
    fan = rng.uniform(0, 255, size=(B, 2, K, 2)).astype(np.float32)
    fan_n = rng.integers(0, K + 1, size=(B, 2))
    got = _rays_targets(torch.from_numpy(markers), torch.from_numpy(fan), torch.from_numpy(fan_n), 64).numpy().astype(bool)
    for b in range(B):
        ref = np.zeros((64, 64), bool)
        for m in range(2):
            for q in fan[b, m, : fan_n[b, m]]:
                ref |= path_pix_to_grid_numpy(rasterise_segment_numpy(markers[b, m], q))
        assert np.array_equal(ref, got[b]), f"sample {b}"


def test_polyline_scratch_draws_connected_pixels():
    poly = torch.tensor([[[10.5, 10.5], [100.5, 10.5], [100.5, 80.5], [-1, -1]]])
    s = polyline_scratch(poly, torch.tensor([3]))
    assert s[0, 10, 10:101].sum() == 91 and s[0, 10:81, 100].sum() == 71
    assert s.sum() == 91 + 71 - 1
