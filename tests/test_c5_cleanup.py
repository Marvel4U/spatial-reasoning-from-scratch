"""C5 cleanup pass (spatial_data/c5_cleanup.py): the chain target, the store, the mixing in _wave_post, the lit-cell
weight override and the eval's cleanliness numbers. All on the CPU; the loader tests read the De Pijp val waves
(val routes stand in for train routes) and skip without them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from spatial_data.c5_cleanup import (CLEAN_EXCESS_RATIO, PATH_WIDTH, build_cleanup, chain_mask, cleanliness, cleanup_stats,
                                     cleanup_target, gather_cleanup, load_cleanup, npz_member_memmap, save_cleanup,
                                     shortest_lit_chain, teacher_path_cells, widen)

STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"
G = 32


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(); g.manual_seed(seed)
    return g


def _blank():
    return np.zeros((G, G), bool)


# ---------------------------------------------------------------- the chain and the target rule

def test_path_width_is_the_loaders():
    from spatial_data.c4_tasks import PATH_WIDTH_CELLS
    assert PATH_WIDTH == PATH_WIDTH_CELLS


def _old_route_score(mask, start_c, goal_c, blocked, goal_m, cell_m):
    """route_score as it was before the graph moved to c5_cleanup (4 Oct), kept here as the reference."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g = mask.shape[0]; m = mask.copy()
    for cx, cy in (start_c, goal_c):
        m[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
    idx = -np.ones((g, g), np.int64); ys, xs = np.nonzero(m); idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in offs:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < g) & (x2 >= 0) & (x2 < g)
        ok[ok] &= m[y2[ok], x2[ok]]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]]); w.append(np.full(int(ok.sum()), np.hypot(dy, dx) * cell_m))
    Gm = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys),) * 2)
    s, t = idx[start_c[1], start_c[0]], idx[goal_c[1], goal_c[0]]
    D, pred = dijkstra(Gm, indices=s, return_predecessors=True)
    if not np.isfinite(D[t]):
        return dict(connected=0.0, length_ratio=float("nan"), building_share=float("nan"), success=0.0, lit_cells=float(mask.sum()))
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(int(pred[nodes[-1]]))
    bshare = float(blocked[ys[nodes], xs[nodes]].mean())
    ratio = float(D[t] / max(goal_m, 1e-6))
    return dict(connected=1.0, length_ratio=ratio, building_share=bshare, success=float(ratio <= 1.10 and bshare == 0.0),
                lit_cells=float(mask.sum()))


def test_route_score_unchanged_by_the_move():
    from diagnostics.wave_rollout_eval import route_score
    rng = np.random.default_rng(0)
    for i in range(60):
        mask = rng.random((G, G)) < rng.uniform(0.3, 0.7)
        blocked = rng.random((G, G)) < 0.05
        s = (int(rng.integers(1, G - 1)), int(rng.integers(1, G - 1))); t = (int(rng.integers(1, G - 1)), int(rng.integers(1, G - 1)))
        goal_m = float(rng.uniform(10, 120))
        a = route_score(mask, s, t, blocked, goal_m, 2.0); b = _old_route_score(mask, s, t, blocked, goal_m, 2.0)
        assert a.keys() == b.keys()
        for k in a:
            assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k], (i, k, a[k], b[k])


def _blob_route():
    """A fat blob (rows 10-19, cols 4-27) joining start (5, 15) and goal (26, 15)."""
    pad = _blank(); pad[10:20, 4:28] = True
    return pad, (5, 15), (26, 15)


def test_connected_blob_gives_a_thin_chain_inside_the_blob():
    pad, s, t = _blob_route()
    teacher = _blank(); teacher[0:3, :] = True                     # somewhere else entirely: must not be used
    res = cleanup_target(pad, s, t, _blank(), goal_m=21 * 2.0, cell_m=2.0, teacher=teacher)
    assert res["flag"] == 1 and res["reason"] == 0 and res["ratio"] == pytest.approx(1.0)
    tgt = res["target"]
    assert not (tgt & teacher).any()
    ys, xs = np.nonzero(tgt)
    assert set(ys.tolist()) == {14, 15, 16}, "a straight chain along row 15, 3 cells wide"
    assert xs.min() == 4 and xs.max() == 27
    assert tgt.sum() == 3 * 24 and tgt.sum() < pad.sum() / 3
    ch = shortest_lit_chain(pad, s, t, 2.0)
    assert np.array_equal(tgt, widen(chain_mask(G, ch[0], ch[1], width=1)))


def test_disconnected_drawing_gives_the_teacher_path():
    pad, s, t = _blob_route(); pad[:, 15:17] = False                # a two-column gap
    teacher = _blank(); teacher[14:17, 4:28] = True
    res = cleanup_target(pad, s, t, _blank(), goal_m=42.0, cell_m=2.0, teacher=teacher)
    assert res["flag"] == 0 and res["reason"] == 1 and np.isnan(res["ratio"])
    assert np.array_equal(res["target"], teacher)


def test_chain_through_a_building_cell_gives_the_teacher_path():
    pad = _blank(); pad[15, 4:28] = True                              # a one-cell corridor
    blocked = _blank(); blocked[15, 15] = True                       # an entirely-building cell on it
    teacher = _blank(); teacher[20:23, 4:28] = True
    res = cleanup_target(pad, (5, 15), (26, 15), blocked, goal_m=42.0, cell_m=2.0, teacher=teacher)
    assert res["flag"] == 0 and res["reason"] == 2
    assert np.array_equal(res["target"], teacher)


def test_too_long_chain_gives_the_teacher_path():
    pad, s, t = _blob_route()
    res = cleanup_target(pad, s, t, _blank(), goal_m=30.0, cell_m=2.0, teacher=_blank())   # chain 42 m > 1.10 x 30 m
    assert res["flag"] == 0 and res["reason"] == 3 and res["ratio"] == pytest.approx(42 / 30)


def test_teacher_path_cells_match_the_loader():
    from spatial_data.dataset_c4_gpu import _wave_path_cells
    rng = np.random.default_rng(3)
    crop, g = 256, 128
    paths = [rng.integers(0, crop, (int(rng.integers(5, 60)), 2)) for _ in range(4)] + [np.array([[0, 0], [255, 255], [1, 254]])]
    off = np.concatenate([[0], np.cumsum([len(p) for p in paths])])
    route = dict(wave_off=torch.from_numpy(off), wave_path=torch.from_numpy(np.concatenate(paths)).long())
    got = _wave_path_cells(route, torch.arange(len(paths)), g, crop)
    for i, p in enumerate(paths):
        assert np.array_equal(got[i].numpy(), teacher_path_cells(p, g, crop))


# ---------------------------------------------------------------- the store

def _tiny_store(tmp_path: Path, crop: int = 64, g: int = 32) -> tuple[Path, dict]:
    """3 routes on a 64 px window (grid 32, 2 m cells): route 0 drawing = a blob (own chain), route 1 drawing with a
    gap (teacher), route 2 unreachable (no drawing)."""
    from spatial_data.c5_own_pads import save_own_pads
    from spatial_data.c4_tasks import WAVE_HORIZON_M
    crops = tmp_path / "crops"; store = crops / "c4_tiny"; store.mkdir(parents=True)
    surface = np.zeros((100, 100), np.uint8); surface[0:2, 0:2] = 3
    np.savez(crops / "district_labels_1m.npz", surface=surface)
    (store / "store.json").write_text(json.dumps({"crop_px": crop}))
    markers = np.array([[[10, 30], [52, 30]], [[10, 30], [52, 30]], [[10, 30], [52, 30]]], np.float32)
    origin = np.array([[10, 10], [0, 0], [5, 5]], np.int32)
    np.savez(store / "c4_routes_train.npz", markers=markers, origin=origin)
    goal = np.array([42.0, 42.0, np.inf], np.float32)
    tpath = np.array([[x, 30] for x in range(10, 53)], np.int16)
    np.savez(store / "c5_wave_train.npz", goal_m=goal, path=np.concatenate([tpath, tpath]), offset=np.array([0, 43, 86, 86]))
    n_goal = np.where(np.isfinite(goal), np.floor(goal / WAVE_HORIZON_M) + 1, 0).astype(int)
    counts = np.where(np.isfinite(goal), 2 * n_goal + 1, 0)
    offset = np.concatenate([[0], np.cumsum(counts)])
    pads = np.zeros((offset[-1], g, g), bool)
    blob = np.zeros((g, g), bool); blob[10:20, 4:28] = True
    gap = blob.copy(); gap[:, 15:17] = False
    pads[offset[0] + 2 * n_goal[0] - 1] = blob
    pads[offset[1] + 2 * n_goal[1] - 1] = gap
    pads[offset[0]] = True                                          # another slot: must not be read
    save_own_pads(store / "own.npz", pads=np.packbits(pads, axis=-1), offset=offset, n_goal=n_goal, slot_key=np.zeros(offset[-1]),
                  slot_k_in=np.zeros(offset[-1]), slot_j=np.zeros(offset[-1]), iou=np.zeros(offset[-1]), meta={"run": "tiny"})
    return store, dict(blob=blob, gap=gap, tpath=tpath, goal=goal)


def test_member_memmap_reads_what_np_load_reads(tmp_path):
    store, _ = _tiny_store(tmp_path)
    mm = npz_member_memmap(store / "own.npz", "pads")
    assert np.array_equal(np.asarray(mm), np.load(store / "own.npz")["pads"])


def test_store_build_round_trip_and_gather(tmp_path):
    store, t = _tiny_store(tmp_path)
    res = build_cleanup(store, "own.npz", workers=2, chunk=1, log=None)
    assert res["flag"].tolist() == [1, 0, -1] and res["reason"].tolist() == [0, 1, -1]
    assert np.array_equal(np.unpackbits(res["pad_in"][0], axis=-1).astype(bool), t["blob"]), "slot 2 n_goal - 1 is the input"
    tgt0 = np.unpackbits(res["target"][0], axis=-1).astype(bool)
    assert set(np.nonzero(tgt0)[0].tolist()) == {14, 15, 16} and tgt0.sum() < t["blob"].sum()
    tgt1 = np.unpackbits(res["target"][1], axis=-1).astype(bool)
    assert np.array_equal(tgt1, teacher_path_cells(t["tpath"], 32, 64)), "failed drawing: the teacher's path"
    assert res["lit_in"][0] == t["blob"].sum() and res["lit_out"][0] == tgt0.sum()
    st = cleanup_stats(res)
    assert st["share_own_chain"] == 0.5 and st["with_drawing"] == 2
    path = store / "c5_cleanup_train.npz"
    save_cleanup(path, **{k: res[k] for k in ("pad_in", "target", "flag", "reason", "ratio", "lit_in", "lit_out", "n_goal", "meta")})
    with pytest.raises(ValueError):
        load_cleanup(path, p=1.0, device="cpu", own_pads_name="other.npz")
    with pytest.raises(ValueError):
        load_cleanup(path, p=1.0, device="cpu", grid=64)
    with pytest.raises(ValueError):
        load_cleanup(path, p=1.0, device="cpu", goal_m=torch.tensor([142.0, 42.0, float("inf")]))   # n_goal differs
    cl = load_cleanup(path, p=1.0, device="cpu", own_pads_name="own.npz", grid=32, goal_m=torch.from_numpy(t["goal"]))
    route_ix = torch.tensor([0, 1, 2, 0, 9]); is_draw = torch.tensor([True, True, True, False, True])
    use, pin, tgt = gather_cleanup(cl, route_ix, is_draw, _gen(0))
    assert use.tolist() == [True, True, False, False, False], "draw samples of routes with a drawing only"
    assert np.array_equal(pin[0].numpy(), t["blob"]) and np.array_equal(tgt[0].numpy(), tgt0)
    assert np.array_equal(pin[1].numpy(), t["gap"]) and np.array_equal(tgt[1].numpy(), tgt1)
    assert not pin[2:].any() and not tgt[2:].any()
    cl["p"] = 0.0
    assert not gather_cleanup(cl, route_ix, is_draw, _gen(0))[0].any()


# ---------------------------------------------------------------- the loader: _wave_post on CPU tensors

needs_val = pytest.mark.skipif(not (STORE / "c5_wave_val.npz").is_file(), reason="De Pijp val waves needed")
GL, CROP, B = 128, 256, 48


@pytest.fixture(scope="module")
def val_cpu():
    from spatial_data.dataset_c4_gpu import load_c4_route_index
    return load_c4_route_index(STORE, "val", torch.device("cpu"), max_items=B)


NO_DRAWING = 5                                                       # a draw sample (5 % 3 == 2) whose route has no drawing


def _stores(n: int, reach: torch.Tensor):
    """An own-pad store of all-ones pads (64 slots per route, more than any route uses) and a cleanup store with a
    fixed pattern in (A) and out (Bp); unreachable routes and route NO_DRAWING have no drawing, as in a real store."""
    from spatial_data.c5_own_pads import pack_pads
    n_slot = 64                                                      # more slots than any route uses
    own = dict(pads=torch.full((n * n_slot, GL, GL // 8), 255, dtype=torch.uint8),
               offset=torch.arange(0, (n + 1) * n_slot, n_slot), n_routes=n, p=1.0)
    yy, xx = torch.meshgrid(torch.arange(GL), torch.arange(GL), indexing="ij")
    A = ((yy // 4 + xx // 4) % 2 == 0); Bp = (yy % 16 == 0)
    flag = torch.where(reach, torch.ones(n, dtype=torch.long), torch.full((n,), -1)); flag[NO_DRAWING] = -1
    cl = dict(pad_in=pack_pads(A.expand(n, GL, GL).contiguous()), target=pack_pads(Bp.expand(n, GL, GL).contiguous()),
              flag=flag, n_routes=n, p=1.0)
    return own, cl, A, Bp


def _post(val, kinds, seed, **kw):
    from spatial_data.dataset_c4_gpu import _wave_post
    ix = torch.arange(len(kinds))
    img0 = torch.zeros(len(kinds), 17, CROP, CROP); tgt0 = torch.zeros(len(kinds), GL, GL, dtype=torch.long)
    return _wave_post(img0, tgt0, kinds, val, ix, GL, CROP, _gen(seed) if seed is not None else None, **kw)


def _kinds(n):
    from spatial_data.c4_tasks import C4TargetKind
    cyc = [C4TargetKind.WAVE_FWD, C4TargetKind.WAVE_BWD, C4TargetKind.WAVE_DRAW]
    return [cyc[i % 3] for i in range(n)]


@needs_val
def test_cleanup_governs_draw_and_own_pads_the_other_passes(val_cpu):
    n = int(val_cpu["origin"].shape[0]); kinds = _kinds(n)
    reach = torch.isfinite(val_cpu["wave_goal_m"])
    own, cl, A, Bp = _stores(n, reach)
    x, t, tok, clean = _post(val_cpu, kinds, 7, own_pads=own, cleanup=cl)
    xo, to, toko, cleano = _post(val_cpu, kinds, 7, own_pads=own)            # own pads alone, same draws
    pad = x[:, 16, ::2, ::2] > 0
    draw = torch.tensor([i % 3 == 2 for i in range(n)])
    hits = 0
    for i in range(n):
        if draw[i] and bool(cl["flag"][i] >= 0):
            assert torch.equal(pad[i], A) and torch.equal(t[i] > 0, Bp), "draw: own drawing in, cleanup target out"
            hits += 1
        elif draw[i]:
            assert torch.equal(pad[i], clean[i]) and torch.equal(t[i], to[i]), "no drawing stored: the usual sample"
        else:
            assert torch.equal(x[i], xo[i]) and torch.equal(t[i], to[i]), "fwd / bwd: as with own pads alone"
    assert torch.equal(tok, toko) and torch.equal(clean, cleano), "tokens and the teacher's clean pad unchanged"
    assert hits >= 5 and bool(draw[NO_DRAWING])


@needs_val
def test_cleanup_without_own_pads_and_p0(val_cpu):
    n = int(val_cpu["origin"].shape[0]); kinds = _kinds(n)
    _, cl, A, Bp = _stores(n, torch.isfinite(val_cpu["wave_goal_m"]))
    d = next(i for i in range(n) if i % 3 == 2 and bool(cl["flag"][i] >= 0))
    x, t, _, _ = _post(val_cpu, kinds, 4, cleanup=cl)
    assert torch.equal(x[d, 16, ::2, ::2] > 0, A) and torch.equal(t[d] > 0, Bp)
    x0, t0, _, _ = _post(val_cpu, kinds, 4)
    assert torch.equal(x[0], x0[0]) and torch.equal(t[0], t0[0]), "a forward sample is untouched"
    cl["p"] = 0.0
    xp, tp, _, _ = _post(val_cpu, kinds, 4, cleanup=cl)
    assert torch.equal(xp, x0) and torch.equal(tp, t0), "p = 0 is the loader without cleanup"


@needs_val
def test_noise_hits_the_cleanup_input_and_val_ignores_cleanup(val_cpu):
    from spatial_data.dataset_c4_gpu import F_max_pool
    n = int(val_cpu["origin"].shape[0]); kinds = _kinds(n)
    _, cl, A, Bp = _stores(n, torch.isfinite(val_cpu["wave_goal_m"]))
    d = next(i for i in range(n) if i % 3 == 2 and bool(cl["flag"][i] >= 0))
    noise = {"p": 1.0, "thicken": 1.0, "drop": 0.0, "add": 0.0}
    x, t, _, _ = _post(val_cpu, kinds, 5, cleanup=cl, pad_noise=noise)
    thick = F_max_pool(A.float()[None, None], 3, 1, 1)[0, 0] > 0
    assert torch.equal(x[d, 16, ::2, ::2] > 0, thick) and torch.equal(t[d] > 0, Bp), "noise after the replacement, target clean"
    xv, tv, _, _ = _post(val_cpu, kinds, None, cleanup=cl)
    x0, t0, _, _ = _post(val_cpu, kinds, None)
    assert torch.equal(xv, x0) and torch.equal(tv, t0), "validation (no generator) never reads the cleanup store"


@needs_val
def test_train_loader_carries_the_cleanup_store(val_cpu):
    from spatial_data.dataset_c4_gpu import C4MultiTaskGpuTrainLoader, load_district_planes_v4
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = 3
    district = load_district_planes_v4(STORE.parent, torch.device("cpu"))
    n = int(val_cpu["origin"].shape[0])
    _, cl, A, Bp = _stores(n, torch.isfinite(val_cpu["wave_goal_m"]))

    def make():
        tl = C4MultiTaskGpuTrainLoader(["wave_fwd", "wave_bwd", "wave_draw"], "onehot_v4_sp", GL, 16, 3, district=district,
                                       route_index=val_cpu, train_task_weights=[0.0, 0.0, 1.0])
        tl.return_tokens = True
        return tl

    plain = make().next_batch()
    tl = make(); tl.cleanup = cl
    img, tgt, tix, cond, tok = tl.next_batch()
    assert (tix == 2).all()
    hit = (img[:, 16, ::2, ::2] > 0).flatten(1).eq(A.flatten()).all(1)
    assert hit.sum() >= 1 and torch.equal(tgt[hit] > 0, Bp.expand(int(hit.sum()), GL, GL))
    assert not torch.equal(plain[1], tgt)


# ---------------------------------------------------------------- the lit-cell weight override

def test_fg_weight_override_reaches_loss_and_decision_rule(monkeypatch):
    import config
    from eval_c4 import wave_decision_tau
    stats = {"tasks": {k: {"f": 0.019} for k in ("wave_fwd", "wave_bwd", "wave_draw")}}
    monkeypatch.setattr(config, "c4_task_stats", lambda: stats)
    monkeypatch.setattr(config, "c4_weight_alpha", 0.5)
    monkeypatch.setattr(config, "c4_grid_loss", "ce")
    monkeypatch.setattr(config, "c4_train_task_keys", lambda: ["wave_fwd", "wave_bwd", "wave_draw"])
    monkeypatch.setattr(config, "c5_wave_fg_weight_override", None)
    w0 = config.c4_model_extra_config()["c1_per_task_fg_weights"]
    assert len(w0) == 3 and all(w > 5 for w in w0)
    assert wave_decision_tau("wave_draw") == pytest.approx(w0[2] / (1 + w0[2]))
    monkeypatch.setattr(config, "c5_wave_fg_weight_override", {"wave_draw": 1.0})
    w1 = config.c4_model_extra_config()["c1_per_task_fg_weights"]
    assert w1[:2] == w0[:2] and w1[2] == 1.0
    assert wave_decision_tau("wave_draw") == 0.5 and wave_decision_tau("wave_fwd") == pytest.approx(w0[0] / (1 + w0[0]))


def test_cleanup_specs_resolve():
    import config
    import experiments
    a, b = experiments.c5_wave_cleanup_queue()
    for spec, rnd, sub in ((a, 2, "c4_100k"), (b, 1, "c4_512_20k")):
        assert "cleanup" in spec["id"] and spec["train_steps"] == 16000 and spec["lr_horizon_steps"] == 16000
        ov = spec["overrides"]
        assert all(hasattr(config, k) for k in ov), [k for k in ov if not hasattr(config, k)]
        assert ov["c4_task_keys"] == ["wave_fwd", "wave_bwd", "wave_draw"] and ov["c4_train_task_weights"] == [1 / 3] * 3
        assert ov["c5_wave_own_pads"] == {"path": f"c5_own_pads_train_r{rnd}.npz", "p": 0.5}
        assert ov["c5_wave_cleanup"]["path"] == f"c5_cleanup_train_r{rnd}.npz" and ov["c5_wave_cleanup"]["p"] == 1.0
        assert ov["c5_wave_fg_weight_override"] == {"wave_draw": 1.0}
        assert ov["c4_routes_subdir"] == sub and ov["learning_rate"] == 1e-4 and ov["c5_token_loss_weight"] == 0.1
        assert ov["c5_wave_change_ce_beta"] == 0.0 and ov["c5_wave_pad_noise"]
        assert spec["load_optimizer"] is False and spec["reset_step_counter"] and Path(spec["init_checkpoint"]).name.endswith(".pt")
    assert a["init_checkpoint"].endswith("c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k.pt")
    assert b["init_checkpoint"].endswith("c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512.pt")
    assert b["overrides"]["batch_size"] == 64 and b["overrides"]["grid_out_size"] == 256


# ---------------------------------------------------------------- the eval's cleanliness numbers

def test_cleanliness_of_a_clean_line_and_a_fat_blob():
    line = _blank(); line[14:17, 5:27] = True                        # exactly 3 wide, start (5, 15) to goal (26, 15)
    c = cleanliness(line, (5, 15), (26, 15), 2.0)
    # the chain runs along row 15 from col 5 to 26; widened it covers cols 4..27 of rows 14-16
    assert c["chain_cells"] == 3 * 24 and c["excess_ratio"] == pytest.approx(66 / 72) and c["clean"] == 1.0
    assert c["iou_own_chain"] == pytest.approx(66 / 72)
    blob, s, t = _blob_route()
    c = cleanliness(blob, s, t, 2.0)
    assert c["excess_ratio"] == pytest.approx(240 / 72) and c["clean"] == 0.0 and c["excess_ratio"] > CLEAN_EXCESS_RATIO
    assert c["iou_own_chain"] == pytest.approx(72 / 240)
    gap = blob.copy(); gap[:, 15:17] = False
    c = cleanliness(gap, s, t, 2.0)
    assert c["clean"] == 0.0 and np.isnan(c["excess_ratio"]) and np.isnan(c["iou_own_chain"])
