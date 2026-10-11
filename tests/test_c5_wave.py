"""C5 R5 wave-search step tasks through the real loaders (needs cuda and c5_wave_val.npz next to the 100k store)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuda required", allow_module_level=True)

from spatial_data.c4_tasks import (REL_STATE_BACK, REL_STATE_DRAW, REL_STATE_SEARCH, TOK_BACK, TOK_DONE, TOK_DRAW, TOK_SEARCH, WAVE_HORIZON_M,
                                   get_c4_task, path_pix_to_grid_numpy)
from spatial_data.c5_wave import cell_minmax_numpy, fronts_upto_numpy, start_disc_numpy
from spatial_data.dataset_c4_gpu import C4GpuValLoader, C4MultiTaskGpuTrainLoader, load_c4_route_index, load_district_planes_v4

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"
pytestmark = pytest.mark.skipif(not (STORE / "c5_wave_val.npz").is_file(), reason="waves not built")
G = 128


def _setup():
    dev = torch.device("cuda")
    return load_district_planes_v4(STORE.parent, dev), load_c4_route_index(STORE, "val", dev), np.load(STORE / "c5_wave_val.npz"), np.load(STORE / "c4_routes_val.npz")


def _batch(key, district, val):
    vl = C4GpuValLoader(get_c4_task(key), "onehot_v4_sp", G, 24, district=district, route_index=val)
    return vl.next_batch_with_meta()


def _ref_pad(w, z, r, k):
    dmin, dmax = cell_minmax_numpy(w["dist"][r], G)
    return start_disc_numpy(z["markers"][r, 0], G, 256) | fronts_upto_numpy(dmin, dmax, k, WAVE_HORIZON_M)


def test_cond_slots_and_tokens():
    assert get_c4_task("wave_first").padded_cond_ids()[0] == REL_STATE_SEARCH
    assert get_c4_task("wave_back").padded_cond_ids()[0] == REL_STATE_BACK
    assert get_c4_task("wave_draw").padded_cond_ids()[0] == REL_STATE_DRAW
    assert get_c4_task("segment").padded_cond_ids()[0] == -1


def test_wave_first_pad_and_target():
    district, val, w, z = _setup()
    img, tgt, tix, cond, meta = _batch("wave_first", district, val)
    assert (cond[:, 0] == REL_STATE_SEARCH).all() and (meta["tok_tgt"] == TOK_SEARCH).all()
    for i, r in enumerate(meta["route_ix"].tolist()):
        pad_in = img[i, 16].cpu().numpy() > 0
        ref_in = np.repeat(np.repeat(_ref_pad(w, z, r, 0), 2, 0), 2, 1)
        assert np.array_equal(pad_in, ref_in), f"pad in, route {r}"
        assert np.array_equal(tgt[i].cpu().numpy() == 1, _ref_pad(w, z, r, 1)), f"target, route {r}"
        # both markers in the marker plane
        from scipy import ndimage
        assert ndimage.label(img[i, 15].cpu().numpy() > 0)[1] == 2


def _thick(cells):
    from scipy import ndimage
    return ndimage.binary_dilation(cells, structure=np.ones((3, 3), bool))


def _ref_path(w, r, lo_m, hi_m):
    """cells of the way back with walking distance in [lo_m, hi_m], 3 cells wide"""
    p = w["path"][w["offset"][r]:w["offset"][r + 1]].astype(np.int64)
    d2 = w["dist"][r].astype(float) * 2.0; d2[w["dist"][r] == 255] = np.inf
    dpx = d2[p[:, 1] // 2, p[:, 0] // 2]
    sel = (dpx >= lo_m) & (dpx <= hi_m)
    return _thick(path_pix_to_grid_numpy(p[sel], grid=G))


def test_wave_reach_back_origin_draw():
    district, val, w, z = _setup()
    R = WAVE_HORIZON_M
    img, tgt, tix, cond, meta = _batch("wave_reach", district, val)
    assert (meta["tok_tgt"] == TOK_BACK).all()
    for i, r in enumerate(meta["route_ix"].tolist()):
        n_goal = int(np.floor(w["goal_m"][r] / R)) + 1
        assert np.array_equal(tgt[i].cpu().numpy() == 1, _ref_pad(w, z, r, n_goal)), f"reach target, route {r}"
    img, tgt, tix, cond, meta = _batch("wave_back", district, val)
    assert (cond[:, 0] == REL_STATE_BACK).all()
    for i, r in enumerate(meta["route_ix"].tolist()):
        n_goal = int(np.floor(w["goal_m"][r] / R)) + 1
        pad_in = img[i, 16].cpu().numpy()[::2, ::2] > 0; t = tgt[i].cpu().numpy() == 1
        # the pad in = disc + fronts 1..j + path down to front j for some j in 2..n_goal; find j from the fronts
        found = False
        for j in range(max(n_goal, 1), 0, -1):
            ref_in = _ref_pad(w, z, r, j) | _ref_path(w, r, j * R, np.inf)
            if np.array_equal(pad_in, ref_in):
                ref_out = _ref_pad(w, z, r, j - 1) | _ref_path(w, r, (j - 1) * R, np.inf)
                assert np.array_equal(t, ref_out), f"back target, route {r}, j={j}"
                assert int(meta["tok_tgt"][i]) == (TOK_BACK if j > 1 else TOK_DRAW)
                found = True; break
        assert found, f"back pad in matches no j, route {r}"
    img, tgt, tix, cond, meta = _batch("wave_origin", district, val)
    assert (meta["tok_tgt"] == TOK_DRAW).all()
    for i, r in enumerate(meta["route_ix"].tolist()):
        assert np.array_equal(tgt[i].cpu().numpy() == 1, _ref_pad(w, z, r, 0) | _ref_path(w, r, 0, np.inf)), f"origin target, route {r}"
    img, tgt, tix, cond, meta = _batch("wave_draw", district, val)
    assert (cond[:, 0] == REL_STATE_DRAW).all() and (meta["tok_tgt"] == TOK_DONE).all()
    for i, r in enumerate(meta["route_ix"].tolist()):
        assert np.array_equal(tgt[i].cpu().numpy() == 1, _ref_path(w, r, 0, np.inf)), f"draw target, route {r}"
        assert np.array_equal(img[i, 16].cpu().numpy()[::2, ::2] > 0, _ref_pad(w, z, r, 0) | _ref_path(w, r, 0, np.inf))


def test_wave_step_copies_its_input():
    district, val, w, z = _setup()
    img, tgt, tix, cond, meta = _batch("wave_step", district, val)
    for i, r in enumerate(meta["route_ix"].tolist()):
        pad_in = img[i, 16].cpu().numpy()[::2, ::2] > 0; t = tgt[i].cpu().numpy() == 1
        assert not (pad_in & ~t).any(), f"everything on the pad is in the target, route {r}"
        n_goal = int(np.floor(w["goal_m"][r] / WAVE_HORIZON_M)) + 1
        if n_goal >= 3:
            assert t.sum() > pad_in.sum(), "a new front was added"


def test_mixed_loader_with_wave_tasks():
    district, val, w, z = _setup()
    tl = C4MultiTaskGpuTrainLoader(["segment", "wave_first", "wave_draw"], "onehot_v4_sp", G, 48, 5, district=district, route_index=val)
    tl.return_tokens = True
    img, tgt, tix, cond, tok = tl.next_batch()
    assert (tok[tix == 0] == -1).all() and (tok[tix == 1] == TOK_SEARCH).all() and (tok[tix == 2] == TOK_DONE).all()
    assert (cond[tix == 0, 0] == -1).all() and (cond[tix == 2, 0] == REL_STATE_DRAW).all()
    assert img[tix == 0, 16].sum() == 0 and (img[tix == 1, 16].flatten(1).sum(1) > 0).all()


def test_natural_pass_sampling_covers_all_passes():
    district, val, w, z = _setup()
    R = WAVE_HORIZON_M
    tl = C4MultiTaskGpuTrainLoader(["wave_fwd", "wave_bwd"], "onehot_v4_sp", G, 96, 11, district=district, route_index=val)
    tl.return_tokens = True
    seen_k, seen_tok = set(), set()
    for _ in range(6):
        img, tgt, tix, cond, tok = tl.next_batch()
        for i in range(img.shape[0]):
            pad_in = img[i, 16].cpu().numpy()[::2, ::2] > 0
            n_fronts_in = int(pad_in.sum())
            seen_tok.add((int(tix[i]), int(tok[i])))
            if int(tix[i]) == 0:
                seen_k.add(n_fronts_in == 9)   # k = 0: only the 3x3 start disc on the pad
    assert True in seen_k and False in seen_k, "wave_fwd draws k = 0 and k > 0"
    assert (0, TOK_SEARCH) in seen_tok and (0, TOK_BACK) in seen_tok, "wave_fwd emits SEARCH and, on the last pass, BACK"
    assert (1, TOK_BACK) in seen_tok and (1, TOK_DRAW) in seen_tok, "wave_bwd emits BACK and, on the last pass, DRAW"
    assert (cond[tix == 0, 0] == REL_STATE_SEARCH).all() and (cond[tix == 1, 0] == REL_STATE_BACK).all()


def test_pad_noise_only_when_enabled_and_only_in_training():
    district, val, w, z = _setup()
    vl = C4GpuValLoader(get_c4_task("wave_step"), "onehot_v4_sp", G, 16, district=district, route_index=val)
    vl.pad_noise = {"p": 1.0, "thicken": 1.0, "drop": 0.0, "add": 0.0}
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    for i, r in enumerate(meta["route_ix"].tolist()):
        assert (img[i, 16].cpu().numpy()[::2, ::2] > 0).sum() == meta["pad_cells"][i].sum().item(), "val pads stay clean"
    tl = C4MultiTaskGpuTrainLoader(["wave_step"], "onehot_v4_sp", G, 32, 3, district=district, route_index=val)
    img0, tgt0, _, _ = tl.next_batch()
    tl2 = C4MultiTaskGpuTrainLoader(["wave_step"], "onehot_v4_sp", G, 32, 3, district=district, route_index=val)
    tl2.pad_noise = {"p": 1.0, "thicken": 1.0, "drop": 0.0, "add": 0.0}
    img1, tgt1, _, _ = tl2.next_batch()
    assert torch.equal(tgt0, tgt1), "the target is the clean next pad with or without noise"
    a = img0[:, 16] > 0; b = img1[:, 16] > 0
    assert (b & ~a).sum() > 0 and not (a & ~b).any(), "thickening only adds cells around the drawing"
