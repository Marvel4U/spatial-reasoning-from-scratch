"""C5 at 512 m windows (4 Oct 2026): the wave field in uint16 (walking distances beyond 508 m), the loader at grid 256,
a 256 m checkpoint rebuilt for 512 m windows (wave_rollout_eval.py), --halves route windows, own pads at grid 256.
The CPU tests run anywhere; the loader tests need cuda and the 512 m Zuid + Centrum store with its wave data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from spatial_data import c4_routes
from spatial_data.c4_tasks import TOK_BACK, WAVE_HORIZON_M, get_c4_task
from spatial_data.c5_wave import (UNREACHED, UNREACHED16, cell_minmax_numpy, dist_dtype, fronts_upto_numpy, start_disc_numpy,
                                  unreached_mask, wave_for_sample)
from spatial_data.dataset_c4_gpu import _wave_cell_minmax

STORE = ROOT / "data/amsterdam/zuid_centrum/crops/v4ns/c4_512_20k"
G, C = 256, 512


# ---------------------------------------------------------------- CPU: the field

def _comb(size: int = 700) -> np.ndarray:
    """A district with long walls: going from the left to the right column of the window means walking far more
    than 508 m (the uint8 limit of the 256 m stores)."""
    bld = np.zeros((size, size), bool)
    for k, x in enumerate(range(150, 600, 60)):
        if k % 2 == 0:
            bld[:size - 40, x:x + 6] = True        # wall from the top, gap at the bottom
        else:
            bld[40:, x:x + 6] = True               # wall from the bottom, gap at the top
    return bld


def test_dtype_follows_the_window():
    assert dist_dtype(256) == np.uint8 and dist_dtype(512) == np.uint16


def test_wave_beyond_508_m_is_kept_at_512():
    bld = _comb()
    markers = np.array([[20.5, 300.5], [490.5, 300.5]], np.float32)       # window px; origin (100, 100)
    d2, goal_m, path = wave_for_sample(bld, (100, 100), C, markers)
    assert d2.dtype == np.uint16 and d2.shape == (C // 2, C // 2)
    assert goal_m > 1000, "the comb forces a long walk"
    reached = ~unreached_mask(d2)
    assert reached.any() and (d2[reached].max() * 2.0) > 508, "distances beyond 508 m survive (uint8 would clip them)"
    assert (d2[~reached] == UNREACHED16).all()
    # the goal's cell holds about the goal's walking distance, not a clipped value
    gx, gy = int(markers[1, 0]) // 2, int(markers[1, 1]) // 2
    assert abs(d2[gy, gx] * 2.0 - goal_m) <= 4.0
    # the fronts up to the goal pass reach the goal cell's neighbourhood
    dmin, dmax = cell_minmax_numpy(d2, G)
    n_goal = int(np.floor(goal_m / WAVE_HORIZON_M)) + 1
    assert n_goal > 13
    ring = fronts_upto_numpy(dmin, dmax, n_goal, WAVE_HORIZON_M)
    assert ring[max(gy - 20, 0):gy + 21, max(gx - 20, 0):gx + 21].any(), "the last front lies near the goal"
    assert len(path) > 0 and np.array_equal(path[0], markers[1].astype(np.int16))


def test_wave_at_256_stays_uint8():
    bld = np.zeros((500, 500), bool)
    d2, goal_m, _ = wave_for_sample(bld, (100, 100), 256, np.array([[10.5, 10.5], [200.5, 10.5]], np.float32))
    assert d2.dtype == np.uint8 and abs(goal_m - 190) < 1.0
    assert unreached_mask(np.array([UNREACHED], np.uint8)).all()


def test_loader_minmax_reads_int16_as_numpy_reads_uint16():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 600, (2, G, G)).astype(np.uint16); d[:, :10, :10] = UNREACHED16
    t = torch.from_numpy(d.view(np.int16))                              # as _load_wave hands it to the GPU
    dmin, dmax = _wave_cell_minmax(t, G)
    for b in range(2):
        rmin, rmax = cell_minmax_numpy(d[b], G)
        assert np.array_equal(dmin[b].numpy(), rmin.astype(np.float32))
        assert np.allclose(dmax[b].numpy(), rmax.astype(np.float32))
    assert torch.isinf(dmin[:, :10, :10]).all()


# ---------------------------------------------------------------- CPU: --halves windows

def test_halves_sampler_keeps_val_west_and_test_east():
    split = {"block_m": [446.0, 448.25], "val_blocks": [[0, 2]], "test_blocks": [[1, 0]]}
    old = c4_routes.CROP_PX
    c4_routes.CROP_PX = 512
    try:
        s = c4_routes.WindowSampler(split, 1793, 1784, np.random.default_rng(0), halves=True)
        for _ in range(200):
            dx, _dy = s.draw("val"); assert 0 <= dx and dx + 512 <= 1784 // 2
            dx, _dy = s.draw("test"); assert dx >= 1784 - 1784 // 2 and dx + 512 <= 1784
        blocks = c4_routes.WindowSampler(split, 1793, 1784, np.random.default_rng(0))
        with pytest.raises(ValueError, match="smaller than"):
            blocks.draw("val")                                          # a 512 m window does not fit a 446 m block
        blocks.draw("train")                                            # train windows only avoid the blocks
    finally:
        c4_routes.CROP_PX = old


# ---------------------------------------------------------------- CPU: a 256 m checkpoint in a 512 m model

def test_eval_geometry_keeps_the_cell_size():
    from diagnostics.wave_rollout_eval import eval_geometry
    assert eval_geometry(256, 128, 512) == (512, 256)
    assert eval_geometry(256, 128, 256) == (256, 128)
    assert eval_geometry(256, 128, 512, grid=128) == (512, 128)
    with pytest.raises(ValueError):
        eval_geometry(256, 128, 512, grid=300)


def test_256_weights_load_unchanged_into_a_512_model():
    from plain_gpt_module.local_config import LocalGridViTConfig
    from plain_gpt_module.local_grid_vit import LocalGridViT
    kw = dict(in_chans=17, patch_size=16, enc_n_layer=2, n_embd=64, n_head=4, rel_pos_bias="all", num_classes=2, balanced_ce=False,
              n_tasks=3, c1_task_conditioning="cond_tokens", token_head_enabled=True, n_token_classes=41)
    small = LocalGridViT(LocalGridViTConfig(img_size=256, grid_out_size=128, **kw))
    big = LocalGridViT(LocalGridViTConfig(img_size=512, grid_out_size=256, **kw))
    sd_s, sd_b = small.state_dict(), big.state_dict()
    assert sd_s.keys() == sd_b.keys() and all(sd_s[k].shape == sd_b[k].shape for k in sd_s), "no learned weight depends on the window"
    big.load_state_dict(sd_s, strict=True)
    x = torch.zeros(1, 17, 512, 512); cond = torch.zeros(1, 2, dtype=torch.long)
    logits, _ = big(x, cond_ids=cond, task_id=torch.zeros(1, dtype=torch.long))
    assert logits.shape == (1, 256, 256, 2)
    # what is new to the network at 512 m: the sin-cos rows beyond its 16 x 16 patch grid
    assert big.pos_embed.shape[0] == 32 * 32 and small.pos_embed.shape[0] == 16 * 16
    p32 = LocalGridViT(LocalGridViTConfig(img_size=512, grid_out_size=256, **dict(kw, patch_size=32)))
    assert p32.state_dict()["patch_embed.proj.weight"].shape != sd_s["patch_embed.proj.weight"].shape, "P=32 cannot take P=16 weights"


# ---------------------------------------------------------------- cuda: the real 512 m store at grid 256

needs_store = pytest.mark.skipif(not torch.cuda.is_available() or not (STORE / "c5_wave_val.npz").is_file(),
                                 reason="cuda and c5_wave_val.npz of the 512 m store required")


def _setup():
    from spatial_data.dataset_c4_gpu import load_c4_route_index, load_district_planes_v4
    dev = torch.device("cuda")
    return (load_district_planes_v4(STORE.parent, dev), load_c4_route_index(STORE, "val", dev),
            np.load(STORE / "c5_wave_val.npz"), np.load(STORE / "c4_routes_val.npz"))


def _ref_pad(w, z, r, k):
    dmin, dmax = cell_minmax_numpy(w["dist"][r], G)
    return start_disc_numpy(z["markers"][r, 0], G, C) | fronts_upto_numpy(dmin, dmax, k, WAVE_HORIZON_M)


@needs_store
def test_store_is_uint16_and_loader_views_int16():
    district, val, w, z = _setup()
    assert w["dist"].dtype == np.uint16 and w["dist"].shape[1:] == (G, G)
    assert val["wave_dist"].dtype == torch.int16 and val["crop_px"] == C
    assert val["wave_dist"].device.type == "cpu" and val["wave_dist"].is_pinned(), "the 512 m field stays in pinned CPU memory"
    assert (val["wave_dist"] < 0).any(), "unreached cells read -1"
    from spatial_data.dataset_c4_gpu import _wave_dist_rows
    ix = torch.tensor([3, 1, 4], device="cuda")
    rows = _wave_dist_rows(val, ix)
    assert rows.device.type == "cuda" and np.array_equal(rows.cpu().numpy(), w["dist"][[3, 1, 4]].view(np.int16))


@needs_store
def test_first_and_reach_passes_at_grid_256():
    from spatial_data.dataset_c4_gpu import C4GpuValLoader
    district, val, w, z = _setup()
    vl = C4GpuValLoader(get_c4_task("wave_first"), "onehot_v4_sp", G, 16, district=district, route_index=val)
    img, tgt, _tix, _cond, meta = vl.next_batch_with_meta()
    assert img.shape[-1] == C and tgt.shape[-1] == G
    for i, r in enumerate(meta["route_ix"].tolist()):
        assert np.array_equal(img[i, 16].cpu().numpy()[::2, ::2] > 0, _ref_pad(w, z, r, 0)), f"pad in, route {r}"
        assert np.array_equal(tgt[i].cpu().numpy() == 1, _ref_pad(w, z, r, 1)), f"target, route {r}"
    # the reach pass of the longest routes: fronts beyond 508 m are drawn
    long_r = np.argsort(-w["goal_m"])[:4]
    from spatial_data.dataset_c4_gpu import C4MultiTaskGpuValLoader
    ml = C4MultiTaskGpuValLoader(["wave_reach"], "onehot_v4_sp", G, 1, district=district, route_index=val)
    for r in long_r.tolist():
        n_goal = int(np.floor(w["goal_m"][r] / WAVE_HORIZON_M)) + 1
        assert n_goal > 13
        _img, tgt_t, _tid, _cond, tok = ml.fetch_wave_pass(r, "wave_reach")
        t = tgt_t.cpu().numpy().reshape(G, G) == 1
        assert np.array_equal(t, _ref_pad(w, z, r, n_goal)), f"reach target, route {r}"
        assert int(tok) == TOK_BACK


@needs_store
def test_own_pads_teacher_rollout_at_grid_256():
    """--teacher store at grid 256: fed back, the teacher's outputs are the teacher's inputs (the mapping holds)."""
    from spatial_data.c5_own_pads import rollout_own_pads, unpack_pads
    from spatial_data.dataset_c4_gpu import load_c4_route_index, load_district_planes_v4
    district = load_district_planes_v4(STORE.parent, torch.device("cuda"))
    route = load_c4_route_index(STORE, "val", torch.device("cpu"), max_items=6)
    res = rollout_own_pads(route, district, encoding="onehot_v4_sp", grid=G, batch=4, step_fn=None, log=None)
    assert res["pads"].shape[1:] == (G, G // 8)
    assert (res["iou"][res["slot_key"] >= 0] == 1.0).all()
    w = np.load(STORE / "c5_wave_val.npz"); z = np.load(STORE / "c4_routes_val.npz")
    r = 0; n_goal = int(res["n_goal"][r]); o = int(res["offset"][r])
    first = unpack_pads(torch.from_numpy(res["pads"][o:o + 1]))[0].numpy()
    assert np.array_equal(first, _ref_pad(w, z, r, 1)), "slot 0 = the output of the first forward pass"
    reach = unpack_pads(torch.from_numpy(res["pads"][o + n_goal - 1:o + n_goal]))[0].numpy()
    assert np.array_equal(reach, _ref_pad(w, z, r, n_goal))
