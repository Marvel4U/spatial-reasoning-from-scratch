"""c2 targets, district sampling, val determinism, decision rule, cached stats."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from spatial_data.c2_tasks import (  # noqa: E402
    ATOM_KEYS,
    C2_CORE_KEYS,
    STATS_FILENAME,
    TS2_KEYS,
    TS2_SINGLE_KEYS,
    cell_target_numpy,
    load_task_stats,
    task_keys_for_set,
    decision_threshold,
    fg_ce_weight,
    get_c2_task,
    make_task,
    wrong_cond_ids_padded,
)
from spatial_data.dataset_c2_gpu import (  # noqa: E402
    C2DistrictSampler,
    C2GpuValLoader,
    c2_targets_from_planes,
    cond_id_table,
    held_out_rects,
    load_crop_planes,
)

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v3b"
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LAYERS = ("noise", "surface", "estab")


def _random_planes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n, 3, 256, 256), dtype=np.uint8)


def test_atom_ids_and_keys_are_stable():
    assert ATOM_KEYS[0] == "noise_0" and ATOM_KEYS[6] == "surface_2" and ATOM_KEYS[11] == "estab_3"
    spec = make_task("surface_2", "noise_0")
    assert spec.key == "noise_0+surface_2" and spec.name == "quiet sidewalk"
    assert spec.padded_cond_ids() == (0, 6)
    assert get_c2_task("surface_3").padded_cond_ids() == (7, -1)
    with pytest.raises(ValueError):
        make_task("noise_0", "noise_2")  # same layer


def test_wrong_cond_ids_flips_one_atom_same_layer():
    spec = get_c2_task("noise_0+surface_2")
    a, b = spec.padded_cond_ids()
    w0 = wrong_cond_ids_padded(spec, slot=0)
    assert w0[1] == b and w0[0] // 4 == a // 4 and w0[0] != a
    w1 = wrong_cond_ids_padded(spec, slot=1)
    assert w1[0] == a and w1[1] // 4 == b // 4 and w1[1] != b


def test_gpu_targets_match_numpy_reference():
    """Vectorised GPU targets == pixel-AND-then-majority reference, singles included."""
    planes = _random_planes(6, seed=3)
    t = torch.from_numpy(planes).to(DEV)
    keys = list(C2_CORE_KEYS) + ["surface_3", "estab_3", "noise_1"]
    for key in keys:
        spec = get_c2_task(key)
        cond = torch.tensor([spec.padded_cond_ids()] * 6, dtype=torch.long, device=DEV)
        got = c2_targets_from_planes(t, cond).cpu().numpy()
        ref = cell_target_numpy({k: planes[:, i] for i, k in enumerate(LAYERS)}, spec)
        assert (got == ref.astype(np.int64)).all(), key


def test_gpu_targets_mixed_tasks_in_one_batch():
    planes = _random_planes(8, seed=11)
    t = torch.from_numpy(planes).to(DEV)
    specs = [get_c2_task(k) for k in TS2_KEYS]
    tix = torch.arange(8, device=DEV) % len(specs)
    cond = cond_id_table(specs, DEV)[tix]
    got = c2_targets_from_planes(t, cond).cpu().numpy()
    for i in range(8):
        ref = cell_target_numpy(
            {k: planes[i: i + 1, j] for j, k in enumerate(LAYERS)}, specs[int(tix[i])],
        )
        assert (got[i] == ref[0].astype(np.int64)).all()


@pytest.mark.skipif(not CROPS.exists(), reason="de_pijp v3b cropset not present")
def test_district_windows_never_touch_held_out_ground():
    """10k on-the-fly train origins obey the crops.py rule, and cover train ground broadly."""
    from spatial_data.dataset_c2_gpu import load_district_planes
    rects = held_out_rects(CROPS)
    assert len(rects) == 4
    sampler = C2DistrictSampler(load_district_planes(CROPS, DEV), rects, seed=5)
    dx, dy = sampler.sample_origins(10_000)
    assert dx.shape[0] == 10_000
    for x0, y0, x1, y1 in rects:
        hit = (dx < x1) & (dx + 256 > x0) & (dy < y1) & (dy + 256 > y0)
        assert not bool(hit.any())
    assert 0 <= int(dx.min()) and int(dx.max()) <= sampler.max_dx
    # broad coverage: many distinct origins, spread over most of the admissible range
    origins = {(int(a), int(b)) for a, b in zip(dx.tolist(), dy.tolist())}
    assert len(origins) > 9_000
    assert int(dx.max()) - int(dx.min()) > 0.8 * sampler.max_dx
    assert int(dy.max()) - int(dy.min()) > 0.8 * sampler.max_dy
    assert sampler.n_origins > 500_000


@pytest.mark.skipif(not CROPS.exists(), reason="de_pijp v3b cropset not present")
def test_val_grid_is_deterministic_and_complete():
    crops = load_crop_planes(CROPS, "val", DEV, max_items=12)
    specs = [get_c2_task(k) for k in C2_CORE_KEYS]
    loader = C2GpuValLoader(crops, specs, "onehot", 64, 256, 7)
    assert len(loader) == 12 * len(specs)
    def collect():
        pairs, n = [], 0
        for img, tgt, tix, cond in loader.iter_batches():
            pairs.append((tix.cpu(), tgt.cpu()))
            n += int(tgt.shape[0])
        return pairs, n
    a, na = collect()
    b, nb = collect()
    assert na == nb == len(loader)
    for (ta, ga), (tb, gb) in zip(a, b):
        assert torch.equal(ta, tb) and torch.equal(ga, gb)
    seen = torch.cat([t for t, _ in a])
    assert torch.bincount(seen, minlength=len(specs)).tolist() == [12] * len(specs)


def test_decision_rule_threshold():
    assert decision_threshold(1.0) == pytest.approx(0.5)
    for w in (0.5, 2.5, 11.4, 41.0):
        assert decision_threshold(w) == pytest.approx(w / (1.0 + w))


@pytest.mark.skipif(not (CROPS / STATS_FILENAME).exists(), reason="c2 task stats not cached")
def test_task_set_ts2_singles():
    stats = load_task_stats(CROPS)
    keys = task_keys_for_set("ts2_singles", stats)
    assert keys == list(TS2_SINGLE_KEYS)


def test_cached_stats_match_c2_plan():
    """C2_PLAN §3: quiet sidewalk f ~ 7.7 %, quiet food & drink ~ 0.76 % (within 10 %)."""
    stats = json.loads((CROPS / STATS_FILENAME).read_text())
    assert stats["n_crops"] == 2000
    assert stats["tasks"]["noise_0+surface_2"]["f"] == pytest.approx(0.077, rel=0.10)
    assert stats["tasks"]["noise_0+estab_3"]["f"] == pytest.approx(0.0076, rel=0.10)
    assert stats["tasks"]["noise_0+surface_3"]["f"] == pytest.approx(0.218, rel=0.10)
    assert stats["tasks"]["surface_3+estab_0"]["f"] == pytest.approx(0.274, rel=0.10)
    w = fg_ce_weight("noise_0+surface_2", stats, alpha=0.5)
    assert w == pytest.approx(((1 - 0.0767) / 0.0767) ** 0.5, rel=0.05)
    assert fg_ce_weight("noise_0+surface_2", stats, alpha=1.0) > w
