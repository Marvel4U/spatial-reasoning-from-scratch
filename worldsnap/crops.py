"""Cut a district into aligned 256 m label crops (label stack v2 / v3).

Cropsets: ``v2`` draws named establishments as 2 m discs; ``v3`` paints the footprint of the
hosting building instead (see ``estab_footprints.py``); ``v3b`` = v3 with an area cap on hosts;
``v4`` = v3b plus two extra label planes (7-level noise, sun) -- see ``planes.py``. Same seed ->
same crop origins, so crop ids of v2, v3, v3b and v4 cover identical ground; v4's RGB image is
byte-identical to v3b's and differs only by the extra planes inside the ``.npz``.

Two renders of the SAME ground square: 1024 px (0.25 m/px, Track A) and 256 px
(1 m/px, Track B). Both are rasterized independently from the vector layers at
their own resolution -- the coarse one is NOT a downsample of the fine one, so
thin sidewalks obey the same ``all_touched=False`` rule in both frames.

Efficiency: the whole district is rasterized ONCE per layer per resolution into
a uint8 class-index array (~7172 x 7136 at 0.25 m/px, ~51 MB per layer); a crop
is then pure array slicing. Crop origins are snapped to the 1 m grid so the
1024 px and the 256 px crop cover exactly the same ground.

Split: the district bbox is cut into a 4 x 4 grid of equal blocks (row 0 =
north, col 0 = west). Two scattered blocks are val ground, two others are test
ground, the remaining twelve are train ground. A val/test crop lies FULLY
inside one block of its own split, so val and test sit on disjoint ground and
neither touches train. A train crop may straddle train blocks but must not
intersect any held-out block.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from PIL import Image
from rasterio.transform import from_origin

from . import planes as P
from .config import CRS_RD, DISTRICTS, District
from .layers.sunshine import district_grid
from .split_blocks import Blocks, figure as split_figure
from .estab_footprints import estab_pairs
from .tiles import _burn, _clip

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CROPSET = "v2"  # default; tasks_point.py imports it
#: cropset -> establishment geometry. v4 repeats v3b's rule exactly, on purpose:
#: the only difference between the two cropsets must be the extra planes.
ESTAB_GEOMETRY = {"v2": "disc", "v3": "footprint", "v3b": "footprint", "v4": "footprint"}
ESTAB_MAX_HOST_M2 = {"v3b": 1000.0, "v4": 1000.0}  # bigger hosts: a cap-area disc clipped to the footprint (estab_footprints)
CROP_M = 256
FINE_RES, COARSE_RES = 0.25, 1.0
GREY = [0, 85, 170, 255]  # class index -> 8-bit grey value, all three layers

#: Spatial split: 4 x 4 scattered blocks over the district bbox, (row, col) with
#: row 0 = north, col 0 = west. De Pijp: 1784 x 1793 m -> blocks 446.0 x 448.25 m.
N_BLOCKS = 4
VAL_BLOCKS = [(0, 2), (3, 1)]
TEST_BLOCKS = [(1, 0), (2, 3)]
BLOCKS = Blocks(N_BLOCKS, VAL_BLOCKS, TEST_BLOCKS)
SPLIT_COLOUR = {"train": "#ffd400", "val": "#00e5ff", "test": "#ff4040"}

#: R: noise Lden. The 50-55 dB band and "no polygon" both fall in class 0.
NOISE_MEANING = {0: "<55 dB", 1: "55-65 dB", 2: "65-75 dB", 3: ">=75 dB"}
NOISE_CLASS = {"50-55 dB": 0, "55-60 dB": 1, "60-65 dB": 1,
               "65-70 dB": 2, "70-75 dB": 2, "75 dB and above": 3}
#: G: BGT ``functie``. v2 merges the v1 cycle-path class into roadway.
SURFACE_MEANING = {0: "none", 1: "roadway", 2: "sidewalk", 3: "building"}
SURFACE_CLASS = {"rijbaan lokale weg": 1, "rijbaan regionale weg": 1, "OV-baan": 1,
                 "parkeervlak": 1, "inrit": 1, "overweg": 1, "fietspad": 1,
                 "voetpad": 2, "voetgangersgebied": 2, "voetpad op trap": 2}
#: B: named OSM POIs only, 2 m discs, higher class wins. transport/other undrawn.
ESTAB_MEANING = {0: "none", 1: "other named", 2: "shop", 3: "food & drink"}
ESTAB_CLASS = {"services": 1, "culture&leisure": 1, "health": 1, "shop": 2, "food&drink": 3}
ESTAB_RADIUS_M = 2.0
LAYERS = ("noise", "surface", "estab")
MEANINGS = {"noise": NOISE_MEANING, "surface": SURFACE_MEANING, "estab": ESTAB_MEANING}

LEGEND_TEXT = (
    "This is a map image, north up, 256 m across. Each colour channel is one map layer, "
    "encoded as a grey level: 0, 85, 170 or 255.\n"
    "RED = road-traffic noise (Lden): 0 = below 55 dB, 85 = 55-65 dB, 170 = 65-75 dB, "
    "255 = 75 dB or more.\n"
    "GREEN = ground surface: 0 = none, 85 = roadway (incl. cycle path and parking), "
    "170 = sidewalk, 255 = building.\n"
    "BLUE = named establishments, drawn as {estab_shape}: 0 = none, 85 = other named place, "
    "170 = shop, 255 = food & drink.\n"
)


def _noise_pairs(noise: gpd.GeoDataFrame) -> list:
    rows = noise.assign(_c=noise["db_label"].map(NOISE_CLASS)).dropna(subset=["_c"])
    return list(zip(rows.sort_values("_c").geometry, rows.sort_values("_c")["_c"].astype(int)))


def _surface_pairs(streets: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> list:
    rows = streets.assign(_c=streets["functie"].map(SURFACE_CLASS)).dropna(subset=["_c"])
    # Tunnels first (anything above them overwrites), then ascending class, buildings last.
    rows = rows.assign(_tun=(rows["relatieve_hoogteligging"] < 0).astype(int) * -1)
    rows = rows.sort_values(["_tun", "_c"])
    return list(zip(rows.geometry, rows["_c"].astype(int))) + [(g, 3) for g in buildings.geometry]


def _estab_rows(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = pois[pois["name"].notna()].assign(_c=pois["poi_class"].map(ESTAB_CLASS))
    return rows.dropna(subset=["_c"]).sort_values("_c")  # higher class wins


def _district_stack(pairs: dict, ox: float, oy_top: float, w_m: int, h_m: int, res: float) -> dict:
    """Rasterize every layer over the whole district at ``res`` m/px."""
    tr = from_origin(ox, oy_top, res, res)
    h, w = int(h_m / res), int(w_m / res)
    return {"transform": tr, **{k: _burn(v, tr, h, w) for k, v in pairs.items()}}


def build_crops(district: District, n_train: int, n_val: int, n_test: int, seed: int,
                data_root: Path = DATA_ROOT, cropset: str = CROPSET) -> None:
    t0 = time.time()
    ddir = data_root / district.city / district.name
    out = ddir / "crops" / cropset
    out.mkdir(parents=True, exist_ok=True)

    noise = gpd.read_file(ddir / "noise_road_lden.gpkg")
    streets = gpd.read_file(ddir / "streets_bgt.gpkg")
    buildings = gpd.read_file(ddir / "buildings_3dbag.gpkg")
    pois = gpd.read_file(ddir / "pois_osm.gpkg")

    grid = district_grid(district)  # the one definition of the 1 m frame (sunshine.py)
    ox, oy_top, w_m, h_m = grid["ox"], grid["oy_top"], grid["width_m"], grid["height_m"]
    est_pairs, est_stats = estab_pairs(_estab_rows(pois), buildings, ESTAB_GEOMETRY[cropset],
                                       ESTAB_RADIUS_M, ESTAB_MAX_HOST_M2.get(cropset))
    print(f"establishments ({cropset}): {est_stats}")
    planes = P.CROPSET_PLANES[cropset]
    pairs = {"noise": _noise_pairs(noise), "surface": _surface_pairs(streets, buildings),
             "estab": est_pairs}
    if "noise7" in planes:
        pairs["noise7"] = P.noise7_pairs(noise)
    # One classed plane per available date; each crop picks one date below.
    sun_by_date = ({d: P.sun_plane_1m(ddir / f, grid["transform"], (h_m, w_m))
                    for d, f in sorted(P.SUN_DATES.items())} if "sun" in planes else {})
    sun_1m = next(iter(sun_by_date.values()), None)  # placeholder in the stacks; swapped per crop
    print(f"district {w_m} x {h_m} m -> {int(w_m / FINE_RES)} x {int(h_m / FINE_RES)} px fine"
          f" | planes {planes}")

    stacks, tr_json = {}, {}
    for tag, res in (("025m", FINE_RES), ("1m", COARSE_RES)):
        st = _district_stack(pairs, ox, oy_top, w_m, h_m, res)
        tr = st.pop("transform")
        if sun_1m is not None:  # 1 m raster; the fine frame is an honest 4x repeat
            st["sun"] = sun_1m if res == COARSE_RES else P.upsample(sun_1m, int(1 / res))
        stacks[tag] = st
        tr_json[tag] = {"transform": [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], "res_m": res,
                        "width_px": int(w_m / res), "height_px": int(h_m / res)}
        np.savez_compressed(out / f"district_labels_{tag}.npz", **st)
        print(f"  {tag}: rasterized + saved ({time.time() - t0:.0f} s)")
    (out / "district_labels_transforms.json").write_text(json.dumps(
        {"crs": CRS_RD, "origin_nw_rd": [ox, oy_top], "width_m": w_m, "height_m": h_m,
         "frames": tr_json}, indent=1))

    counts = {"train": n_train, "val": n_val, "test": n_test}
    block_recs = BLOCKS.records(ox, oy_top, w_m, h_m)
    (out / "split.json").write_text(json.dumps(
        {"crs": CRS_RD, "origin_nw_rd": [ox, oy_top], "district_m": [w_m, h_m],
         "n_blocks": N_BLOCKS, "block_m": [w_m / N_BLOCKS, h_m / N_BLOCKS], "crop_m": CROP_M,
         "val_blocks": VAL_BLOCKS, "test_blocks": TEST_BLOCKS,
         "rule": "A val/test crop lies fully inside one block of its own split, so val and "
                 "test ground is disjoint; a train crop intersects no held-out block.",
         "blocks": block_recs}, indent=1))
    print(f"split: {N_BLOCKS}x{N_BLOCKS} blocks of {w_m / N_BLOCKS:.1f} x {h_m / N_BLOCKS:.1f} m, "
          f"val {VAL_BLOCKS}, test {TEST_BLOCKS}, train = the rest (never touching a held-out "
          f"block)")

    est_rows = _estab_rows(pois)
    meanings = {**MEANINGS, **P.MEANINGS_EXTRA}
    legend = P.legend_json(
        planes, meanings, est_stats,
        base={"crop_m": CROP_M, "frames": {"1024": FINE_RES, "256": COARSE_RES},
              "grey_by_class": GREY, "channels": dict(zip("RGB", LAYERS))},
        source_mapping={"noise": NOISE_CLASS, "surface": SURFACE_CLASS, "estab": ESTAB_CLASS},
        legend_text=LEGEND_TEXT.format(estab_shape=(
            "small discs" if est_stats["mode"] == "disc"
            else "the footprint of the building that houses them")))
    (out / "legend.json").write_text(json.dumps(legend, indent=1, ensure_ascii=False))

    seen_grey = {k: set() for k in LAYERS}
    frac = {s: {k: np.zeros(P.N_CLASSES[k], dtype=np.int64) for k in planes} for s in counts}
    index_lines, n_bytes = [], 0
    centres: dict[str, list] = {s: [] for s in counts}
    for si, (split, n) in enumerate(counts.items()):
        rng = np.random.default_rng([seed, si])
        sdir = out / split
        sdir.mkdir(exist_ok=True)
        date_rng = np.random.default_rng([seed, si, 621])  # separate stream: origins stay identical
        for i, (dx, dy) in enumerate(BLOCKS.sample_origins(rng, split, w_m, h_m, CROP_M, n)):
            cid = f"{split}_{i:04d}"
            sun_date = None
            if sun_by_date:
                sun_date = sorted(sun_by_date)[int(date_rng.integers(len(sun_by_date)))]
                stacks["1m"]["sun"] = sun_by_date[sun_date]
                stacks["025m"]["sun"] = P.upsample(sun_by_date[sun_date], 4)
            centres[split].append((dx + CROP_M / 2, dy + CROP_M / 2))
            f = {k: stacks["025m"][k][4 * dy:4 * dy + 1024, 4 * dx:4 * dx + 1024] for k in planes}
            c = {k: stacks["1m"][k][dy:dy + 256, dx:dx + 256] for k in planes}
            rgb = np.dstack([np.take(GREY, f[k]).astype(np.uint8) for k in LAYERS])
            for k in LAYERS:
                seen_grey[k] |= set(np.unique(rgb[:, :, LAYERS.index(k)]).tolist())
            for k in planes:
                frac[split][k] += np.bincount(f[k].ravel(), minlength=P.N_CLASSES[k])
            Image.fromarray(rgb).save(sdir / f"{cid}_rgb.png")
            np.savez_compressed(sdir / f"{cid}_labels.npz",
                                **{f"{k}_1024": f[k] for k in planes},
                                **{f"{k}_256": c[k] for k in planes})
            bb = [ox + dx, oy_top - dy - CROP_M, ox + dx + CROP_M, oy_top - dy]
            ests = [{"name": r["name"], "poi_class": r["poi_class"], "class": int(r["_c"]),
                     "col_1024": int((r.geometry.x - bb[0]) / FINE_RES),
                     "row_1024": int((bb[3] - r.geometry.y) / FINE_RES),
                     "col_256": int((r.geometry.x - bb[0]) / COARSE_RES),
                     "row_256": int((bb[3] - r.geometry.y) / COARSE_RES)}
                    for _, r in _clip(est_rows, tuple(bb)).iterrows()
                    if bb[0] <= r.geometry.x < bb[2] and bb[1] < r.geometry.y <= bb[3]]
            meta = {"crop_id": cid, "split": split, "cropset": cropset, "crs": CRS_RD,
                    "sun_date": sun_date,
                    "bbox_rd": [round(v, 3) for v in bb], "origin_m_from_district_nw": [dx, dy],
                    "crop_m": CROP_M, "px_fine": 1024, "px_coarse": 256,
                    "class_fractions_1024": {
                        k: {str(j): round(float(v), 6) for j, v in
                            enumerate(np.bincount(f[k].ravel(),
                                                  minlength=P.N_CLASSES[k]) / f[k].size)}
                        for k in planes},
                    "establishments": ests}
            (sdir / f"{cid}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            n_bytes += sum((sdir / f"{cid}{s}").stat().st_size
                           for s in ("_rgb.png", "_labels.npz", ".json"))
            index_lines.append(json.dumps({"crop_id": cid, "split": split,
                                           "bbox_rd": [round(v, 3) for v in bb],
                                           "origin_m_from_district_nw": [dx, dy],
                                           "n_establishments": len(ests)}))
    (out / "index.jsonl").write_text("\n".join(index_lines) + "\n")
    (ddir / "overlays").mkdir(exist_ok=True)
    png = ddir / "overlays" / f"crops_{cropset}_split.png"
    split_figure(png, np.dstack([np.take(GREY, stacks["1m"][k][::4, ::4]).astype(np.uint8)
                                 for k in LAYERS]), w_m, h_m, block_recs, centres, SPLIT_COLOUR,
                 f"crops/{cropset} spatial split - {N_BLOCKS}x{N_BLOCKS} blocks of "
                 f"{w_m / N_BLOCKS:.0f} x {h_m / N_BLOCKS:.0f} m, EPSG:28992\n"
                 f"val blocks {VAL_BLOCKS}, test blocks {TEST_BLOCKS}; dots = crop centres ("
                 + ", ".join(f"{s} {len(p)}" for s, p in centres.items()) + ")")
    print(f"split figure: overlays/{png.name}")

    for k in LAYERS:
        assert seen_grey[k] <= set(GREY), f"{k}: unexpected grey values {seen_grey[k] - set(GREY)}"
    for split in counts:
        for k in planes:
            tot = frac[split][k].sum()
            print(f"  {split:5s} {k:8s} " + "  ".join(
                f"{i}:{meanings[k][i]}={frac[split][k][i] / tot:.4f}"
                for i in range(P.N_CLASSES[k])))
    print(f"crops: {sum(counts.values())}, on disk {n_bytes / 1e6:.1f} MB "
          f"(+ district stacks), {time.time() - t0:.1f} s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="worldsnap.crops", description=__doc__)
    p.add_argument("--district", required=True, choices=sorted(DISTRICTS))
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=250)
    p.add_argument("--n-test", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--cropset", default=CROPSET, choices=sorted(ESTAB_GEOMETRY))
    args = p.parse_args(argv)
    build_crops(DISTRICTS[args.district], args.n_train, args.n_val, args.n_test,
                args.seed, args.data_root, args.cropset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
