"""Synthetic maze district (spatial_data/maze_world.py): a tiny district into a temp dir, its maze properties, and the
route builder, the wave builder and the c4 loaders running on it unchanged (CPU only, ~10 s)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

torch = pytest.importorskip("torch")

from spatial_data import c4_routes, c5_wave
from spatial_data import maze_world as MW
from spatial_data.dataset_c4_gpu import load_c4_route_index, load_district_planes_v4

SIZE, N_BLOCKS, CROP = (300, 300), 3, 64


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    out = tmp_path_factory.mktemp("maze") / "crops" / "v4ns"
    doc = MW.write_store(out, SIZE, N_BLOCKS, (8, 12), (2, 6), seed=3, crop_m=CROP, overview=False)
    return out, doc


def test_deterministic_from_seed():
    a = MW.build_maze(SIZE, N_BLOCKS, (8, 12), (2, 6), 3)[0]
    b = MW.build_maze(SIZE, N_BLOCKS, (8, 12), (2, 6), 3)[0]
    c = MW.build_maze(SIZE, N_BLOCKS, (8, 12), (2, 6), 4)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_maze_properties(store):
    out, doc = store
    z = np.load(out / "district_labels_1m.npz")
    assert set(z.files) >= {"noise7", "surface", "estab", "noise"}
    s = z["surface"]
    assert s.shape == (SIZE[1], SIZE[0]) and s.dtype == np.uint8
    assert set(np.unique(s)) == {MW.GROUND_CLASS, MW.BUILDING_CLASS}, "walls are building, everything else plain ground"
    assert not z["noise7"].any() and not z["estab"].any()
    assert ndimage.label(s != MW.BUILDING_CLASS)[1] == 1, "the corridors form one connected component"
    ch = doc["checks"]
    assert ch["corridors_in_range"] and ch["walls_in_range"] and ch["walkable_components"] == 1
    # the outer frame is wall, so nothing walks off the map
    assert (s[0] == 3).all() and (s[-1] == 3).all() and (s[:, 0] == 3).all() and (s[:, -1] == 3).all()


def test_styles_differ():
    """A perfect block is a tree on the cell graph (edges = cells - 1); braiding adds loops."""
    xs = MW.lattice(200, (8, 12), (2, 6), np.random.default_rng(0)); m = MW.Maze(xs, xs)
    rng = np.random.default_rng(1)
    MW.perfect(m, 0, m.ny, 0, m.nx, rng)
    assert m.east.sum() + m.south.sum() == m.ny * m.nx - 1
    m2 = MW.Maze(xs, xs); MW.braided(m2, 0, m2.ny, 0, m2.nx, np.random.default_rng(1), 1.0)
    assert m2.east.sum() + m2.south.sum() > m2.ny * m2.nx - 1
    dead = sum(m2.degree(r, c) == 1 for r in range(m2.ny) for c in range(m2.nx))
    assert dead == 0, "braid 1.0 leaves no dead end"


def test_split_has_every_style_in_val_and_test(store):
    out, doc = store
    split = json.loads((out / "split.json").read_text())
    style = {(b["row"], b["col"]): b["style"] for b in split["blocks"]}
    for which in ("val_blocks", "test_blocks"):
        assert sorted(style[tuple(b)] for b in split[which]) == sorted(MW.STYLES)
    assert not set(map(tuple, split["val_blocks"])) & set(map(tuple, split["test_blocks"]))


def test_routes_waves_and_loaders(store, monkeypatch):
    out, _ = store
    for k, v in dict(CROP_PX=CROP, D_MIN_PX=20.0, D_MAX_PX=60.0).items():
        monkeypatch.setattr(c4_routes, k, v)
    building, split = c4_routes.load_district(out)
    rng = np.random.default_rng(0)
    sampler = c4_routes.WindowSampler(split, *building.shape, rng)
    routes = out / "c4_test_store"; routes.mkdir()
    (routes / "store.json").write_text(json.dumps({"crop_px": CROP}))
    c4_routes.build_split("val", 6, building, sampler, rng, 0, routes)
    dev = torch.device("cpu")
    planes = load_district_planes_v4(out, dev)
    assert planes.shape == (3, SIZE[1], SIZE[0])
    idx = load_c4_route_index(routes, "val", dev)
    assert idx["origin"].shape == (6, 2) and idx["crop_px"] == CROP
    z = np.load(routes / "c4_routes_val.npz")
    val_rects = c4_routes.block_rects_px(split, "val")
    for (dx, dy), mk in zip(z["origin"], z["markers"]):
        assert any(x0 <= dx and dx + CROP <= x1 and y0 <= dy and dy + CROP <= y1 for x0, y0, x1, y1 in val_rects)
        dist2, goal_m, path = c5_wave.wave_for_sample(building, (dx, dy), CROP, mk)
        assert np.isfinite(goal_m) and len(path) > 0, "the wave reaches the goal"
        assert not building[dy + path[:, 1], dx + path[:, 0]].any(), "the way back stays in the corridors"
