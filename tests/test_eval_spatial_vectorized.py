"""Vectorized eval_spatial batch metrics match the reference loop."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import config
from eval_spatial import _grid_metrics_batch, _grid_metrics_batch_reference

KEYS = (
    "mean_iou_fg",
    "exact_cell_accuracy",
    "pred_fg_cells_mean",
    "pred_fg_cells_median",
    "pred_fg_full_patch_frac",
    "pred_fg_zero_frac",
    "true_cell_recall",
    "pred_fg_in_true_patch_mean",
    "global_argmax_exact",
    "global_argmax_patch_hit",
    "subcell_hit_given_patch",
    "global_argmax_dist_cells_mean",
    "global_argmax_dist_cells_median",
)


def _compare_batch(pred, tgt, am, tidx, *, fg_class=1, s=4, g=64):
    ref = _grid_metrics_batch_reference(
        pred, tgt, am, tidx, fg_class=fg_class, s=s, G=g,
    )
    m = _grid_metrics_batch(pred, tgt, am, tidx, fg_class=fg_class, s=s, G=g)
    b = pred.shape[0]
    vec = {
        "mean_iou_fg": (m["iou_sum"] / b).item(),
        "exact_cell_accuracy": (m["exact_cell"].float() / b).item(),
        "pred_fg_cells_mean": m["pred_fg_counts"].float().mean().item(),
        "pred_fg_cells_median": m["pred_fg_counts"].float().median().item(),
        "pred_fg_full_patch_frac": (m["pred_fg_counts"] == s * s).float().mean().item(),
        "pred_fg_zero_frac": (m["pred_fg_counts"] == 0).float().mean().item(),
        "true_cell_recall": (m["true_cell_recall"].float() / b).item(),
        "pred_fg_in_true_patch_mean": (
            (m["fg_in_true_patch_sum"] / m["fg_in_true_patch_n"]).item()
            if int(m["fg_in_true_patch_n"].item()) else 0.0
        ),
        "global_argmax_exact": (m["global_argmax_exact"].float() / b).item(),
        "global_argmax_patch_hit": (m["patch_hit"].float() / b).item(),
        "subcell_hit_given_patch": (
            (m["subcell_hit_given_patch"].float() / m["patch_hit"]).item()
            if int(m["patch_hit"].item()) else 0.0
        ),
        "global_argmax_dist_cells_mean": (m["argmax_dist_sum"] / b).item(),
        "global_argmax_dist_cells_median": m["argmax_dists"].float().median().item(),
    }
    tol = {"global_argmax_dist_cells_mean": 1e-4, "global_argmax_dist_cells_median": 1e-4}
    for k in KEYS:
        assert ref[k] == pytest.approx(vec[k], rel=0, abs=tol.get(k, 1e-6)), k


def _random_batch(b=16, g=64, fg=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    pred = torch.randint(0, 2, (b, g, g), generator=gen)
    tgt = torch.zeros(b, g, g, dtype=torch.long)
    for i in range(b):
        r = torch.randint(0, g, (1,), generator=gen).item()
        c = torch.randint(0, g, (1,), generator=gen).item()
        tgt[i, r, c] = fg
    flat = torch.randn(b, g * g, generator=gen)
    am = flat.argmax(dim=1)
    tidx = tgt.reshape(b, -1).argmax(dim=1)
    return pred, tgt, am, tidx


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_vectorized_matches_reference_random(seed):
    config.patch_size = 16
    config.img_size = 256
    config.grid_out_size = 64
    pred, tgt, am, tidx = _random_batch(seed=seed)
    _compare_batch(pred, tgt, am, tidx)


def test_vectorized_matches_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    config.patch_size = 16
    config.img_size = 256
    config.grid_out_size = 64
    pred, tgt, am, tidx = _random_batch(b=32, seed=99)
    pred, tgt, am, tidx = [x.cuda() for x in (pred, tgt, am, tidx)]
    _compare_batch(pred, tgt, am, tidx)
