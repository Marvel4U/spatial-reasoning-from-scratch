"""C5 key trajectories: the builder on a synthetic crossroads, and the GPU target against the stored pixels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatial_data.c5_trajectories import trajectories_for_window

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"


def _crossroads(width: int = 12) -> np.ndarray:
    """256 px window, four building blocks, a plus-shaped street of `width` px through the centre."""
    m = np.ones((256, 256), bool); c = 128; h = width // 2
    m[c - h:c + h, :] = False; m[:, c - h:c + h] = False
    return m


def test_crossroads_gives_one_trajectory_per_arm():
    trajs = trajectories_for_window(_crossroads(), np.array([[128.5, 128.5], [128.5, 128.5]]))[0][0]
    assert len(trajs) == 4
    ends = np.array([t[-1] for t in trajs])
    # one end per arm, each about 100 m out along the street centre
    arms = {("E" if x > 200 else "W" if x < 56 else "S" if y > 200 else "N" if y < 56 else "?") for x, y in ends}
    assert arms == {"N", "E", "S", "W"}


def test_side_street_gets_its_own_branch():
    m = _crossroads()
    m[:, :] = True; m[122:134, :] = False            # one long street W-E
    m[134:256, 170:176] = False                      # a narrow side street going south, 42 px east of the marker
    trajs = trajectories_for_window(m, np.array([[128.5, 128.5], [128.5, 128.5]]))[0][0]
    assert len(trajs) == 3                           # west, east, and the side street
    assert any(t[-1][1] > 180 for t in trajs), "one trajectory runs down the side street"


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available() or not (STORE / "c5_traj_val.npz").is_file(), reason="needs cuda and c5_traj_val.npz")
def test_gpu_target_matches_stored_pixels():
    from spatial_data.c4_tasks import get_c4_task, path_pix_to_grid_numpy
    from spatial_data.dataset_c4_gpu import C4GpuValLoader, load_c4_route_index, load_district_planes_v4
    dev = torch.device("cuda")
    val = load_c4_route_index(STORE, "val", dev)
    vl = C4GpuValLoader(get_c4_task("trajectories"), "onehot_v4_sp", 64, 32, district=load_district_planes_v4(STORE.parent, dev), route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    z = np.load(STORE / "c5_traj_val.npz")
    for i, r in enumerate(meta["route_ix"].tolist()):
        ref = path_pix_to_grid_numpy(z["pix"][z["offset"][r]:z["offset"][r + 1]].astype(np.int64))
        assert np.array_equal(tgt[i].cpu().numpy() == 1, ref), f"sample {r}"


@pytest.mark.skipif(not torch.cuda.is_available() or not (STORE / "c5_traj_val.npz").is_file(), reason="needs cuda and c5_traj_val.npz")
def test_one_marker_variant():
    from scipy import ndimage
    from spatial_data.c4_tasks import get_c4_task, path_pix_to_grid_numpy
    from spatial_data.dataset_c4_gpu import C4GpuValLoader, C4MultiTaskGpuTrainLoader, load_c4_route_index, load_district_planes_v4
    dev = torch.device("cuda")
    val = load_c4_route_index(STORE, "val", dev); district = load_district_planes_v4(STORE.parent, dev)
    vl = C4GpuValLoader(get_c4_task("trajectories_one"), "onehot_v4_sp", 64, 32, district=district, route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    z = np.load(STORE / "c5_traj_val.npz")
    for i, r in enumerate(meta["route_ix"].tolist()):
        _, n_blobs = ndimage.label(img[i, 15].cpu().numpy() > 0)
        assert n_blobs == 1, "one marker disc"
        sl = slice(z["offset"][r], z["offset"][r + 1])
        ref = path_pix_to_grid_numpy(z["pix"][sl][z["pix_tag"][sl] < 32].astype(np.int64))
        assert np.array_equal(tgt[i].cpu().numpy() == 1, ref), f"sample {r}"
    # mixed with the two-marker task: two discs there, one here
    tl = C4MultiTaskGpuTrainLoader(["trajectories", "trajectories_one"], "onehot_v4_sp", 64, 64, 5, district=district, route_index=val)
    img, tgt, tix, cond = tl.next_batch()
    for i in range(img.shape[0]):
        _, n_blobs = ndimage.label(img[i, 15].cpu().numpy() > 0)
        a, b = val["markers"][0, 0], val["markers"][0, 1]   # discs may touch only if markers are < 7 px apart (min 40 m)
        assert n_blobs == (1 if int(tix[i]) == 1 else 2)


@pytest.mark.skipif(not torch.cuda.is_available() or not (STORE / "c5_traj_r40_val.npz").is_file(), reason="needs cuda and c5_traj_r40_val.npz")
def test_step_tasks_scratchpad_and_target():
    from scipy import ndimage
    from spatial_data.c4_tasks import get_c4_task, path_pix_to_grid_numpy
    from spatial_data.dataset_c4_gpu import C4GpuValLoader, load_c4_route_index, load_district_planes_v4
    dev = torch.device("cuda")
    val = load_c4_route_index(STORE, "val", dev); district = load_district_planes_v4(STORE.parent, dev)
    z = np.load(STORE / "c5_traj_r40_val.npz")
    for rnd, key in ((1, "trajectories_one_r40"), (2, "trajectories_step2_r40"), (4, "trajectories_step4_r40")):
        vl = C4GpuValLoader(get_c4_task(key), "onehot_v4_sp", 64, 16, district=district, route_index=val)
        img, tgt, tix, cond, meta = vl.next_batch_with_meta()
        for i, r in enumerate(meta["route_ix"].tolist()):
            _, n_blobs = ndimage.label(img[i, 15].cpu().numpy() > 0)
            assert n_blobs == 1
            pk, ok = ("pix", "offset") if rnd == 1 else (f"pix{rnd}", f"offset{rnd}")
            px = z[pk][z[ok][r]:z[ok][r + 1]].astype(np.int64)
            if rnd == 1:
                px = px[z["pix_tag"][z[ok][r]:z[ok][r + 1]] < 32]
            assert np.array_equal(tgt[i].cpu().numpy() == 1, path_pix_to_grid_numpy(px)), f"{key} target, sample {r}"
            ref = np.zeros((256, 256), bool)
            for q in range(1, rnd):
                pq, oq = ("pix", "offset") if q == 1 else (f"pix{q}", f"offset{q}")
                pp = z[pq][z[oq][r]:z[oq][r + 1]].astype(np.int64)
                if q == 1:
                    pp = pp[z["pix_tag"][z[oq][r]:z[oq][r + 1]] < 32]
                ref[pp[:, 1], pp[:, 0]] = True
            assert np.array_equal(img[i, 16].cpu().numpy() > 0, ref), f"{key} scratchpad, sample {r}"
