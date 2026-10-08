"""Track A export + polyline scorer (spatial_data/track_a_export.py) on a tiny synthetic district, plus 20 real
De Pijp val routes when the store is there. CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spatial_data import track_a_export as X
from spatial_data.c5_wave import wave_for_sample

ROOT = Path(__file__).resolve().parents[1]
REAL_STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"
CROP = 64

# window origin (district px), start, goal (window metres); what the route tests
ROUTES = [
    ((60, 60), (10.5, 30.5), (54.5, 30.5)),     # 0: a block in between -> detour; the straight line crosses it
    ((60, 60), (10.5, 58.5), (54.5, 58.5)),     # 1: free straight line below the block
    ((120, 120), (10.5, 30.5), (54.5, 30.5)),   # 2: a wall through the whole window -> the way back leaves it
    ((0, 0), (25.5, 25.5), (50.5, 50.5)),       # 3: start walled in -> unreachable, skipped
]


def _district() -> np.ndarray:
    s = np.zeros((200, 200), np.uint8)
    s[:, 0:5] = 1                    # a strip of roadway, so the layers style has something to colour
    s[70:110, 80:100] = 3            # block of route 0/1 (window x 20..39, y 10..49)
    s[112:200, 150:154] = 3          # wall of route 2 (window x 30..33, from y -8 down to the district edge)
    s[20:31, 20:31] = 3; s[21:30, 21:30] = 0   # ring around the start of route 3
    return s


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("track_a")
    crops = tmp / "crops"; store = crops / "store"; store.mkdir(parents=True)
    surface = _district()
    np.savez_compressed(crops / "district_labels_1m.npz", surface=surface)
    (store / "store.json").write_text(json.dumps({"crop_px": CROP}))
    origin = np.array([r[0] for r in ROUTES], np.int32)
    markers = np.array([[r[1], r[2]] for r in ROUTES], np.float32)
    np.savez_compressed(store / "c4_routes_val.npz", origin=origin, markers=markers)
    waves = [wave_for_sample(surface == 3, origin[i], CROP, markers[i]) for i in range(len(ROUTES))]
    counts = [len(w[2]) for w in waves]
    np.savez_compressed(store / "c5_wave_val.npz", dist=np.stack([w[0] for w in waves]),
                        goal_m=np.array([w[1] for w in waves], np.float32),
                        path=np.concatenate([w[2] for w in waves]).reshape(-1, 2).astype(np.int16),
                        offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    out = tmp / "export"
    meta = X.export(store, "val", out, px_per_m=2, style="plain", workers=2)
    return dict(store=store, out=out, meta=meta, routes={r["route"]: r for r in X.load_routes(out / "routes.jsonl")},
                surface=surface, tmp=tmp)


def _score(points, rec):
    return X.score_polyline(points, rec, X.load_building_mask(rec))


def test_export_layout(synth):
    m, R = synth["meta"], synth["routes"]
    assert m["n_exported"] == 3 and m["n_skipped_unreachable"] == 1 and sorted(R) == [0, 1, 2]
    assert R[2]["leaves_window"] and not R[0]["leaves_window"]
    assert min(p[1] for p in R[2]["teacher"]) < 0                  # the full way back, above the image
    for r in R.values():
        assert (synth["out"] / r["image"]).is_file() and r["image_px"] == [128, 128]
        assert r["teacher"][0] == r["start"] and r["teacher"][-1] == r["goal"]
        assert r["n_points"] == len(r["teacher"]) >= 2
    assert R[1]["n_points"] == 2                                     # a free straight route is one segment


def test_teacher_scores_100(synth):
    routes = list(synth["routes"].values())
    rep = X.score(routes, {r["route"]: X.baseline_teacher(r) for r in routes})
    assert rep["solved"] == rep["n"] == 3
    assert all(x["length_ratio"] <= 1.0 + 1e-9 for x in rep["routes"])


def test_straight_through_building_fails(synth):
    R = synth["routes"]
    assert _score(X.baseline_straight(R[0]), R[0])["reason"] == "crosses a building"
    assert _score(X.baseline_straight(R[2]), R[2])["reason"] == "crosses a building"
    assert _score(X.baseline_straight(R[1]), R[1])["solved"]


def test_wrong_endpoints(synth):
    r = synth["routes"][1]; ppm = r["px_per_m"]
    (sx, sy), (gx, gy) = r["start"], r["goal"]
    assert _score([[sx + 5 * ppm, sy], [gx, gy]], r)["solved"]                       # 5 m off: within 6 m
    assert _score([[sx + 7 * ppm, sy], [gx, gy]], r)["reason"] == "wrong endpoints"  # 7 m off
    assert _score([[sx, sy], [gx, gy - 7 * ppm]], r)["reason"] == "wrong endpoints"
    assert _score([[gx, gy], [sx, sy]], r)["reason"] == "wrong endpoints"           # reversed: goal first
    # endpoints are checked before buildings: a wrong-endpoint line through the block reports the endpoints
    r0 = synth["routes"][0]
    assert _score([[r0["start"][0], r0["start"][1] + 20 * ppm], r0["goal"]], r0)["reason"] == "wrong endpoints"


def test_too_long(synth):
    r = synth["routes"][1]; ppm = r["px_per_m"]
    (sx, sy), (gx, gy) = r["start"], r["goal"]
    res = _score([[sx, sy], [sx, 5.5 * ppm], [sx, sy], [gx, gy]], r)   # up the free strip and back, then across
    assert res["reason"] == "too long" and not res["crosses"] and res["length_ratio"] > X.MAX_LENGTH_RATIO
    # just under the limit passes: a small dip keeps the length below 1.10 x
    assert _score([[sx, sy], [(sx + gx) / 2, sy + 4 * ppm], [gx, gy]], r)["solved"]


def test_unparseable(synth):
    r = synth["routes"][1]
    for bad in (None, [], [[1, 2]], [[1, 2], [3]], [["a", 2], [3, 4]], "[[1,2],[3,4]]", [[1, 2], [float("nan"), 4]]):
        assert _score(bad, r)["reason"] == "unparseable", bad
    rep = X.score([r], {})                                                  # missing route
    assert rep["failure_counts"]["unparseable"] == 1 and rep["solved"] == 0


def test_douglas_peucker_keeps_endpoints():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(300, 2)), axis=0)
    for tol in (0.0, 0.5, 2.0, 10.0, 1e6):
        k = X.douglas_peucker(walk, tol)
        assert k[0] == 0 and k[-1] == len(walk) - 1 and (np.diff(k) > 0).all()
    assert len(X.douglas_peucker(walk, 1e6)) == 2
    line = np.stack([np.arange(10.0), 2 * np.arange(10.0)], 1)
    assert list(X.douglas_peucker(line, 0.1)) == [0, 9]                   # collinear points go


def test_simplify_splits_only_where_a_chord_hits_a_building():
    # an L around a block: plain DP at a large tolerance cuts through the block, the constrained one keeps the corner
    win = np.zeros((16, 16), bool); win[3:10, 3:10] = True
    mask = np.pad(win, X.MASK_MARGIN_M, constant_values=False)
    up = [(1.5, y + 0.5) for y in range(11, 0, -1)]; right = [(x + 0.5, 1.5) for x in range(2, 12)]
    dense = np.array(up + right)
    assert len(X.douglas_peucker(dense, 20.0)) == 2
    assert X.segments_hit_building(dense[[0, -1]], mask).all()
    p, added = X.simplify_route(dense, mask, 20.0)
    assert added >= 1 and not X.segments_hit_building(p, mask).any()
    assert (p[0] == dense[0]).all() and (p[-1] == dense[-1]).all()


def test_simplified_teacher_never_touches_a_building(synth):
    for r in synth["routes"].values():
        pm = X.image_px_to_window_m(r["teacher"], r["px_per_m"])
        assert not X.segments_hit_building(pm, X.load_building_mask(r)).any()


@pytest.mark.parametrize("ppm", [1, 2])
def test_coordinates_round_trip_and_render(synth, ppm):
    xy = np.array([[0.0, 0.0], [10.5, 30.5], [63.25, 0.75]])
    assert np.allclose(X.image_px_to_window_m(X.window_m_to_image_px(xy, ppm), ppm), xy)
    assert np.allclose(X.window_m_to_image_px(xy, ppm), xy * ppm)
    # the image agrees with the coordinates: marker centres coloured as markers, a block cell as building
    o, s, g = ROUTES[0]
    im = np.asarray(X.render_route(synth["surface"][o[1]:o[1] + CROP, o[0]:o[0] + CROP], np.array([s, g]), ppm, "plain"))
    assert im.shape == (CROP * ppm, CROP * ppm, 3)
    at = lambda p: tuple(im[int(p[1] * ppm), int(p[0] * ppm)])
    assert at(s) == X._hex(X.COLOURS["start"]) and at(g) == X._hex(X.COLOURS["goal"])
    assert at((30.5, 30.5)) == X._hex(X.COLOURS["plain"]["building"])     # window cell (30, 30) is in the block
    assert at((5.5, 5.5)) == X._hex(X.COLOURS["plain"]["ground"])
    # the record's coordinates follow the same convention at this scale
    out = synth["tmp"] / f"export_ppm{ppm}"
    X.export(synth["store"], "val", out, px_per_m=ppm, style="layers", limit=2, workers=1)
    r = X.load_routes(out / "routes.jsonl")[0]
    assert r["image_px"] == [CROP * ppm] * 2 and np.allclose(r["start"], np.array(s) * ppm)
    assert np.allclose(r["goal"], np.array(g) * ppm)
    rep = X.score([r], {r["route"]: r["teacher"]})
    assert rep["solved"] == 1


@pytest.mark.skipif(not (REAL_STORE / "c5_wave_val.npz").is_file(), reason="De Pijp val store not here")
def test_real_val_routes(tmp_path):
    X.export(REAL_STORE, "val", tmp_path, limit=20, workers=2)
    routes = X.load_routes(tmp_path / "routes.jsonl")
    assert len(routes) == 20
    teach = X.score(routes, {r["route"]: X.baseline_teacher(r) for r in routes})
    straight = X.score(routes, {r["route"]: X.baseline_straight(r) for r in routes})
    assert teach["solved"] == 20
    assert straight["solved"] < 20 and straight["failure_counts"]["wrong endpoints"] == 0
