"""C5 R5 own pads (spatial_data/c5_own_pads.py): store format, the pass-identity mapping, mixing in the train loader.
The CPU tests run anywhere; the loader tests need cuda and c5_wave_val.npz next to the 100k store (val routes stand
in for train routes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_data.c4_tasks import WAVE_KINDS, WAVE_KEYS
from spatial_data.c5_own_pads import (TASK_KEYS, gather_own_pads, load_own_pads, own_pad_source_slot, pack_pads, pass_position,
                                      save_own_pads, unpack_pads)
from spatial_data.wave_search_schedule import wave_pass_schedule

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"
G = 128


def _gen(seed: int, dev="cpu") -> torch.Generator:
    g = torch.Generator(device=dev); g.manual_seed(seed)
    return g


# ---------------------------------------------------------------- CPU: format and mapping

def test_pack_matches_numpy_packbits_and_round_trips():
    pads = torch.rand(5, G, G, generator=_gen(0)) < 0.3
    packed = pack_pads(pads)
    assert np.array_equal(packed.numpy(), np.packbits(pads.numpy(), axis=-1))
    assert torch.equal(unpack_pads(packed), pads)


def test_store_round_trip(tmp_path):
    # three routes with 2 n_goal + 1 slots each; route 1 is unreachable (no slots)
    rng = np.random.default_rng(1)
    n_goal = np.array([2, 0, 3]); counts = np.where(n_goal > 0, 2 * n_goal + 1, 0)
    offset = np.concatenate([[0], np.cumsum(counts)])
    pads = rng.random((offset[-1], G, G)) < 0.2
    path = tmp_path / "c5_own_pads_train.npz"
    save_own_pads(path, pads=np.packbits(pads, axis=-1), offset=offset, n_goal=n_goal, slot_key=np.zeros(offset[-1]),
                  slot_k_in=np.zeros(offset[-1]), slot_j=np.zeros(offset[-1]), iou=np.ones(offset[-1]), meta={"run": "x"})
    own = load_own_pads(path, p=1.0, device=torch.device("cpu"))
    assert own["n_routes"] == 3 and own["meta"]["run"] == "x"
    route_ix = torch.tensor([0, 0, 1, 2, 2, 7])                  # 7 is beyond the store
    slot = torch.tensor([0, 4, 0, 6, -1, 0])
    use, got = gather_own_pads(own, route_ix, slot, _gen(2))
    assert use.tolist() == [True, True, False, True, False, False]
    for i, rs in enumerate([(0, 0), (0, 4), None, (2, 6), None, None]):
        if rs is None:
            assert not got[i].any()
        else:
            assert np.array_equal(got[i].numpy(), pads[offset[rs[0]] + rs[1]])


def test_store_checks_n_goal_against_the_route_store(tmp_path):
    offset = np.array([0, 5]); path = tmp_path / "s.npz"
    save_own_pads(path, pads=np.zeros((5, G, G // 8), np.uint8), offset=offset, n_goal=np.array([2]), slot_key=np.zeros(5),
                  slot_k_in=np.zeros(5), slot_j=np.zeros(5), iou=np.zeros(5), meta={})
    load_own_pads(path, p=0.5, device=torch.device("cpu"), goal_m=torch.tensor([45.0]))       # n_goal 2: fine
    with pytest.raises(ValueError):
        load_own_pads(path, p=0.5, device=torch.device("cpu"), goal_m=torch.tensor([130.0]))  # n_goal 4: another store


def _kind_code(task_key: str) -> int:
    return WAVE_KINDS.index(WAVE_KEYS[task_key][1])


@pytest.mark.parametrize("n_goal", [1, 2, 3, 5, 8])
def test_source_slot_is_the_previous_pass_of_the_schedule(n_goal):
    """For every pass of the teacher's schedule, the slot that feeds it is where the previous pass wrote its output."""
    sched = wave_pass_schedule(n_goal)
    written = {}                                                   # slot -> index in the schedule of the last writer
    for i, wp in enumerate(sched):
        k_in = wp.k_in if wp.k_in is not None else (0 if wp.task_key == "wave_first" else n_goal - 1)
        j = wp.j_back if wp.j_back is not None else 1
        src = int(own_pad_source_slot(torch.tensor([_kind_code(wp.task_key)]), torch.tensor([k_in]), torch.tensor([j]),
                                      torch.tensor([n_goal]))[0])
        if wp.task_key == "wave_first" or (n_goal == 1 and wp.task_key == "wave_reach"):
            assert src == -1, "k_in = 0: the start disc is never replaced"
        else:
            assert written[src] == i - 1, f"{wp.task_key} reads slot {src}, written by pass {written[src]}, not {i - 1}"
        written[pass_position(wp.task_key, wp.k_in, wp.j_back, n_goal)] = i
    assert sorted(written) == list(range(2 * n_goal + 1)), "a route fills exactly 2 n_goal + 1 slots"


def test_source_slot_for_the_natural_samplers():
    """wave_fwd / wave_bwd (the training samplers) map like the fixed keys of the same k_in / j."""
    fwd, bwd, draw = WAVE_KINDS.index(WAVE_KEYS["wave_fwd"][1]), WAVE_KINDS.index(WAVE_KEYS["wave_bwd"][1]), _kind_code("wave_draw")
    n = torch.tensor([4, 4, 4, 4, 4, 4, 4])
    kind = torch.tensor([fwd, fwd, fwd, bwd, bwd, draw, -1])
    k_in = torch.tensor([0, 1, 3, 0, 0, 0, 2]); j = torch.tensor([1, 1, 1, 4, 1, 1, 1])
    assert own_pad_source_slot(kind, k_in, j, n).tolist() == [-1, 0, 2, 3, 6, 7, -1]


# ---------------------------------------------------------------- cuda: the loader

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not (STORE / "c5_wave_val.npz").is_file(),
                                reason="cuda and the val waves needed")


@pytest.fixture(scope="module")
def val_setup(tmp_path_factory):
    import spatial_data.channels as _ch
    from spatial_data.c5_own_pads import rollout_own_pads
    from spatial_data.dataset_c4_gpu import load_c4_route_index, load_district_planes_v4
    _ch.MARKER_RADIUS_PX = 3
    dev = torch.device("cuda")
    district = load_district_planes_v4(STORE.parent, dev)
    val = load_c4_route_index(STORE, "val", dev, max_items=120)
    val_cpu = load_c4_route_index(STORE, "val", torch.device("cpu"), max_items=120)
    res = rollout_own_pads(val_cpu, district, encoding="onehot_v4_sp", grid=G, batch=64, step_fn=None, log=None)   # the teacher's outputs
    d = tmp_path_factory.mktemp("own")
    kw = dict(offset=res["offset"], n_goal=res["n_goal"], slot_key=res["slot_key"], slot_k_in=res["slot_k_in"],
              slot_j=res["slot_j"], iou=res["iou"], meta={"run": "teacher"})
    save_own_pads(d / "teacher.npz", pads=res["pads"], **kw)
    save_own_pads(d / "ones.npz", pads=np.full_like(res["pads"], 255), **kw)
    return district, val, d, res


def _train_loader(district, val, seed, own=None, noise=None):
    from spatial_data.dataset_c4_gpu import C4MultiTaskGpuTrainLoader
    tl = C4MultiTaskGpuTrainLoader(["wave_fwd", "wave_bwd", "wave_draw"], "onehot_v4_sp", G, 96, seed, district=district, route_index=val)
    tl.return_tokens = True
    if own is not None:
        tl.own_pads = own
    if noise is not None:
        tl.pad_noise = noise
    return tl


@needs_cuda
def test_teacher_store_iou_is_one(val_setup):
    *_, res = val_setup
    assert np.allclose(res["iou"], 1.0) and (res["slot_key"] >= 0).all(), "every slot written, teacher against itself"


@needs_cuda
def test_teacher_pads_fed_back_reproduce_the_teachers_inputs(val_setup):
    """p = 1 with the teacher's own outputs as the store: the mapping is right iff every input is unchanged."""
    district, val, d, _ = val_setup
    own = load_own_pads(d / "teacher.npz", p=1.0, device=torch.device("cuda"), goal_m=val["wave_goal_m"])
    img, tgt, tix, cond, tok = _train_loader(district, val, 21, own=own).next_batch()   # the loader path runs
    assert (img[:, 16] > 0).any(dim=(1, 2)).all()
    # direct check through _wave_post: replacing with the teacher's outputs leaves pad_in == pad_clean
    from spatial_data.dataset_c4_gpu import _wave_post
    from spatial_data.c4_tasks import C4TargetKind
    ix = torch.arange(val["origin"].shape[0], device="cuda")
    img0 = torch.zeros(ix.numel(), 17, 256, 256, device="cuda"); tgt0 = torch.zeros(ix.numel(), G, G, dtype=torch.long, device="cuda")
    for kind in (C4TargetKind.WAVE_FWD, C4TargetKind.WAVE_BWD, C4TargetKind.WAVE_DRAW):
        for seed in (1, 2):
            x, t, tk, clean = _wave_post(img0, tgt0, [kind] * ix.numel(), val, ix, G, 256, _gen(seed, "cuda"), own_pads=own)
            assert torch.equal(x[:, 16, ::2, ::2] > 0, clean), f"{kind}: the teacher's output of the producing pass is this pass's input"


@needs_cuda
def test_p1_replaces_every_input_but_the_start_disc_and_keeps_the_target(val_setup):
    from spatial_data.dataset_c4_gpu import _wave_post
    from spatial_data.c4_tasks import C4TargetKind
    district, val, d, _ = val_setup
    ones = load_own_pads(d / "ones.npz", p=1.0, device=torch.device("cuda"))
    teacher = load_own_pads(d / "teacher.npz", p=1.0, device=torch.device("cuda"))
    ix = torch.arange(val["origin"].shape[0], device="cuda")
    img0 = torch.zeros(ix.numel(), 17, 256, 256, device="cuda"); tgt0 = torch.zeros(ix.numel(), G, G, dtype=torch.long, device="cuda")
    for kind in (C4TargetKind.WAVE_FWD, C4TargetKind.WAVE_BWD, C4TargetKind.WAVE_DRAW):
        xo, to, ko, co = _wave_post(img0, tgt0, [kind] * ix.numel(), val, ix, G, 256, _gen(5, "cuda"), own_pads=ones)
        xt, tt, kt, ct = _wave_post(img0, tgt0, [kind] * ix.numel(), val, ix, G, 256, _gen(5, "cuda"), own_pads=teacher)
        assert torch.equal(to, tt) and torch.equal(ko, kt) and torch.equal(co, ct), "same draws: same target, token, clean pad"
        full = (xo[:, 16, ::2, ::2] > 0).all(dim=(1, 2))
        disc_only = co.sum(dim=(1, 2)) <= 9                      # forward k_in = 0: the start disc alone
        if kind == C4TargetKind.WAVE_FWD:
            assert torch.equal(full, ~disc_only), "every forward input but k_in = 0 is replaced"
            assert disc_only.any() and (~disc_only).any()
        else:
            assert full.all(), "every backward / draw input is replaced"


@needs_cuda
def test_p0_is_the_loader_without_own_pads(val_setup):
    district, val, d, _ = val_setup
    own = load_own_pads(d / "ones.npz", p=0.0, device=torch.device("cuda"))
    noise = {"p": 0.5, "thicken": 0.5, "shift": 0.5, "drop": 0.05, "add": 0.002}
    a = _train_loader(district, val, 9, own=own, noise=noise).next_batch()
    b = _train_loader(district, val, 9, noise=noise).next_batch()
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@needs_cuda
def test_noise_comes_after_the_own_pad(val_setup):
    """Order: own pad in, then the noise on it. With thicken-only noise at p = 1 on the all-ones store nothing can
    change (a full pad stays full); on the teacher store the pad grows."""
    from spatial_data.dataset_c4_gpu import _wave_post
    from spatial_data.c4_tasks import C4TargetKind
    district, val, d, _ = val_setup
    ix = torch.arange(val["origin"].shape[0], device="cuda")
    img0 = torch.zeros(ix.numel(), 17, 256, 256, device="cuda"); tgt0 = torch.zeros(ix.numel(), G, G, dtype=torch.long, device="cuda")
    noise = {"p": 1.0, "thicken": 1.0, "drop": 0.0, "add": 0.0}
    ones = load_own_pads(d / "ones.npz", p=1.0, device=torch.device("cuda"))
    x, *_ = _wave_post(img0, tgt0, [C4TargetKind.WAVE_DRAW] * ix.numel(), val, ix, G, 256, _gen(3, "cuda"), own_pads=ones, pad_noise=noise)
    assert (x[:, 16] > 0).all()
    teacher = load_own_pads(d / "teacher.npz", p=1.0, device=torch.device("cuda"))
    x, _, _, clean = _wave_post(img0, tgt0, [C4TargetKind.WAVE_DRAW] * ix.numel(), val, ix, G, 256, _gen(3, "cuda"), own_pads=teacher, pad_noise=noise)
    assert ((x[:, 16, ::2, ::2] > 0).sum() > clean.sum())


@needs_cuda
def test_val_loaders_ignore_own_pads(val_setup):
    from spatial_data.c4_tasks import get_c4_task
    from spatial_data.dataset_c4_gpu import C4GpuValLoader, C4MultiTaskGpuValLoader
    district, val, d, _ = val_setup
    ones = load_own_pads(d / "ones.npz", p=1.0, device=torch.device("cuda"))
    for make in (lambda: C4GpuValLoader(get_c4_task("wave_bwd"), "onehot_v4_sp", G, 32, district=district, route_index=val),
                 lambda: C4MultiTaskGpuValLoader(["wave_fwd", "wave_bwd", "wave_draw"], "onehot_v4_sp", G, 32, district=district, route_index=val)):
        a = make(); b = make(); b.own_pads = ones
        for x, y in zip(a.next_batch(), b.next_batch()):
            assert torch.equal(x, y)
        img = b.fetch_wave_pass(3, "wave_draw")[0] if hasattr(b, "fetch_wave_pass") else None
        if img is not None:
            assert not (img[16] > 0).all(), "fetch_wave_pass (the rollout eval) never reads own pads"


def test_start_step_switches_own_pads_on(tmp_path):
    from spatial_data.c5_own_pads import own_pads_start_step, set_own_pads_step
    assert own_pads_start_step({"p": 0.5}, 48000) == 0
    assert own_pads_start_step({"p": 0.5, "start_frac": 1 / 3}, 48000) == 16000
    assert own_pads_start_step({"p": 0.5, "start_frac": 1 / 3, "start_step": 7}, 48000) == 7
    offset = np.array([0, 5]); path = tmp_path / "s.npz"
    save_own_pads(path, pads=np.full((5, G, G // 8), 255, np.uint8), offset=offset, n_goal=np.array([2]), slot_key=np.zeros(5),
                  slot_k_in=np.zeros(5), slot_j=np.zeros(5), iou=np.zeros(5), meta={})
    own = load_own_pads(path, p=1.0, device=torch.device("cpu")); own["start_step"] = 10
    ix, slot = torch.zeros(4, dtype=torch.long), torch.tensor([0, 1, 2, 3])
    set_own_pads_step(own, 9)
    assert not gather_own_pads(own, ix, slot, _gen(0))[0].any(), "before the start step: never"
    set_own_pads_step(own, 10)
    assert gather_own_pads(own, ix, slot, _gen(0))[0].all(), "from the start step on: with probability p"
