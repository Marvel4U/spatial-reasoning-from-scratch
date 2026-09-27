"""Cut a district into square RGB map tiles (v1 encoding).

One tile = one 8-bit RGB PNG rendered by ``rasterio.features.rasterize`` onto
the tile's own affine transform, so every pixel maps to an exact RD (EPSG:28992)
square. No antialiasing, no matplotlib: a pixel is either inside a polygon or
not, which is what makes the PNG a *label image* a verifier can read back rather
than a picture.

Channels (all three dicts below are the authority; the per-tile JSON copies them
verbatim so a tile is self-describing):

    R  road-traffic noise Lden band        (absence of a polygon = below 50 dB)
    G  surface class, later paint wins     (nothing / road / cycle / walk / bldg)
    B  named establishments, 4 m discs     (higher value wins on overlap)

Grid: origin at the bbox min (SW) corner, partial edge tiles dropped. Row 0 is
the NORTHERNMOST row, because the PNG is north-up and row indices should match
pixel rows; column 0 is the westernmost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # headless rig
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box as shp_box

from .config import CRS_RD, CRS_WGS84, DISTRICTS, District

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TILESET = "rgb_v1"

#: R: ``db_label`` in noise_road_lden.gpkg -> level. No polygon = 0 = below the
#: lowest published band. NOTE: the "50-55 dB" label itself is *inferred* (the
#: city's viewer never draws that band); manifest flags it as
#: ``lowest_band_label_inferred``. Kept as-is, but it is an inference.
NOISE_LEVELS = {
    "50-55 dB": 40,
    "55-60 dB": 80,
    "60-65 dB": 120,
    "65-70 dB": 160,
    "70-75 dB": 200,
    "75 dB and above": 255,
}

#: G: BGT ``functie`` -> level, painted in ascending-level order so a sidewalk
#: overwrites a carriageway and a building footprint overwrites everything.
#: Values present in streets_bgt.gpkg but deliberately NOT painted (stay 0):
#: berm, verkeerseiland, spoorbaan.
SURFACE_LEVELS = {
    "rijbaan lokale weg": 50,
    "rijbaan regionale weg": 50,
    "OV-baan": 50,
    "parkeervlak": 50,
    "inrit": 50,
    "overweg": 50,
    "fietspad": 100,
    "voetpad": 150,
    "voetgangersgebied": 150,
    "voetpad op trap": 150,
}
SURFACE_MEANING = {0: "nothing", 50: "roadway", 100: "cycle path", 150: "sidewalk", 255: "building"}

#: B: ``poi_class`` -> level, only for POIs with a non-null ``name``.
#: transport and other are not drawn.
POI_LEVELS = {"food&drink": 255, "shop": 160, "services": 80, "culture&leisure": 80, "health": 80}
POI_RADIUS_M = 4.0

#: Contact-sheet picks: four hard-coded WGS84 landmarks + two rules over mean R.
#: The landmark coordinates were read back off the OSM POI nodes of the same
#: name in pois_osm.gpkg, so they are the data's own idea of where these are.
CONTACT_POINTS = {
    "Albert Cuypstraat market, west end": (4.8905, 52.3557),
    "metro De Pijp / Ferd. Bolstraat": (4.8920, 52.3530),
    "Sarphatipark": (4.8964, 52.3544),
    "Amstel / canal edge": (4.9045, 52.3527),
}
COLUMN_TITLES = (
    "R noise Lden: 0/40/80/120/160/200/255\n= <50 / 50-55 / ... / 75+ dB",
    "G surface: 0 none, 50 road,\n100 cycle, 150 walk, 255 building",
    "B establishment: 255 food&drink,\n160 shop, 80 services/culture/health",
    "RGB composite",
)
CONTACT_RULES = ("ring-road edge: loudest (max mean R)", "quiet residential: min mean R")

_TO_WGS = Transformer.from_crs(CRS_RD, CRS_WGS84, always_xy=True)
_TO_RD = Transformer.from_crs(CRS_WGS84, CRS_RD, always_xy=True)


def _clip(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Rows whose geometry intersects ``bounds`` (index-accelerated, no clipping)."""
    idx = gdf.sindex.query(shp_box(*bounds), predicate="intersects")
    return gdf.iloc[sorted(idx)]


def _burn(pairs: list, transform, h: int, w: int | None = None) -> np.ndarray:
    """Rasterize ``(geometry, value)`` in order given; later pairs overwrite."""
    out = np.zeros((h, w if w is not None else h), dtype=np.uint8)
    if pairs:
        rasterio.features.rasterize(
            pairs, out=out, transform=transform, all_touched=False, dtype="uint8"
        )
    return out


def _r_pairs(noise: gpd.GeoDataFrame) -> list:
    rows = noise.assign(_lvl=noise["db_label"].map(NOISE_LEVELS)).dropna(subset=["_lvl"])
    rows = rows.sort_values("_lvl")  # louder band wins where bands overlap
    return list(zip(rows.geometry, rows["_lvl"].astype(int)))


def _g_pairs(streets: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> list:
    rows = streets.assign(_lvl=streets["functie"].map(SURFACE_LEVELS)).dropna(subset=["_lvl"])
    # Tunnels (relatieveHoogteligging < 0) are painted FIRST so that any surface
    # object above them overwrites them; +1 (bridges) are kept and painted with
    # the rest. Nothing is dropped on height grounds.
    rows = rows.assign(_tun=(rows["relatieve_hoogteligging"] < 0).astype(int) * -1)
    rows = rows.sort_values(["_tun", "_lvl"], ascending=[True, True])
    pairs = list(zip(rows.geometry, rows["_lvl"].astype(int)))
    return pairs + [(g, 255) for g in buildings.geometry]


def _b_pairs(pois: gpd.GeoDataFrame) -> tuple[list, gpd.GeoDataFrame]:
    rows = pois[pois["name"].notna()].assign(_lvl=pois["poi_class"].map(POI_LEVELS))
    rows = rows.dropna(subset=["_lvl"]).sort_values("_lvl")  # higher value wins
    discs = rows.geometry.buffer(POI_RADIUS_M)
    return list(zip(discs, rows["_lvl"].astype(int))), rows


def _wgs_bbox(b: tuple[float, float, float, float]) -> list[float]:
    x0, y0 = _TO_WGS.transform(b[0], b[1])
    x1, y1 = _TO_WGS.transform(b[2], b[3])
    return [round(v, 6) for v in (x0, y0, x1, y1)]


def build_tiles(district: District, tile_m: float, px: int, data_root: Path = DATA_ROOT) -> None:
    t_start = time.time()
    ddir = data_root / district.city / district.name
    out_dir = ddir / "tiles" / TILESET
    out_dir.mkdir(parents=True, exist_ok=True)
    (ddir / "overlays").mkdir(parents=True, exist_ok=True)

    noise = gpd.read_file(ddir / "noise_road_lden.gpkg")
    streets = gpd.read_file(ddir / "streets_bgt.gpkg")
    buildings = gpd.read_file(ddir / "buildings_3dbag.gpkg")
    pois = gpd.read_file(ddir / "pois_osm.gpkg")
    manifest = json.loads((ddir / "manifest.json").read_text())
    vintages = {
        k: {f: manifest["layers"][k][f] for f in ("vintage", "downloaded_utc", "source_name")}
        for k in ("noise_road_lden", "streets_bgt", "buildings_3dbag", "pois_osm")
    }

    x0, y0, x1, y1 = district.bbox_rd
    ncols, nrows = int((x1 - x0) // tile_m), int((y1 - y0) // tile_m)
    top = y0 + nrows * tile_m  # north edge of the kept grid
    res = tile_m / px
    legend = {
        "R": {"meaning": "road-traffic noise Lden band", "levels": {**{"below 50 dB / no polygon": 0}, **NOISE_LEVELS}},
        "G": {"meaning": "surface class (later paint wins)", "levels": SURFACE_MEANING,
              "functie_to_level": SURFACE_LEVELS},
        "B": {"meaning": f"named establishment, {POI_RADIUS_M:g} m disc", "levels": {**POI_LEVELS, "not drawn": 0}},
    }
    print(f"grid {nrows} rows x {ncols} cols = {nrows * ncols} tiles, {tile_m:g} m, "
          f"{px} px -> {res:.4f} m/px")

    hist = {c: np.zeros(256, dtype=np.int64) for c in "RGB"}
    index, mean_r, n_empty, n_bytes = [], {}, 0, 0
    for r in range(nrows):
        for c in range(ncols):
            b = (x0 + c * tile_m, top - (r + 1) * tile_m, x0 + (c + 1) * tile_m, top - r * tile_m)
            tr = from_origin(b[0], b[3], res, res)
            R = _burn(_r_pairs(_clip(noise, b)), tr, px)
            G = _burn(_g_pairs(_clip(streets, b), _clip(buildings, b)), tr, px)
            bp, prows = _b_pairs(_clip(pois, b))
            B = _burn(bp, tr, px)
            for ch, arr in zip("RGB", (R, G, B)):
                hist[ch] += np.bincount(arr.ravel(), minlength=256)

            tid = f"tile_r{r}_c{c}"
            png = out_dir / f"{tid}.png"
            Image.fromarray(np.dstack([R, G, B])).save(png)
            n_bytes += png.stat().st_size
            mean_r[tid] = float(R.mean())

            est = [
                {"name": row["name"], "poi_class": row["poi_class"], "level": int(POI_LEVELS[row["poi_class"]]),
                 "col": int((row.geometry.x - b[0]) / res), "row": int((b[3] - row.geometry.y) / res)}
                for _, row in prows.iterrows()
                if b[0] <= row.geometry.x < b[2] and b[1] < row.geometry.y <= b[3]
            ]
            n_empty += not est
            (out_dir / f"{tid}.json").write_text(json.dumps({
                "tile_id": tid, "row": r, "col": c, "tileset": TILESET,
                "bbox_rd": [round(v, 3) for v in b], "crs": CRS_RD,
                "bbox_wgs84": _wgs_bbox(b), "tile_m": tile_m, "px": px, "metres_per_pixel": res,
                "transform": [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
                "channels": legend,
                "establishment_counts": {k: sum(e["poi_class"] == k for e in est) for k in POI_LEVELS},
                "establishments": est,
                "data_vintages": vintages,
            }, indent=1, ensure_ascii=False))
            index.append({"tile_id": tid, "row": r, "col": c, "bbox_rd": [round(v, 3) for v in b]})

    (out_dir / "index.json").write_text(json.dumps({
        "tileset": TILESET, "district": district.name, "crs": CRS_RD,
        "grid": {"rows": nrows, "cols": ncols, "n_tiles": nrows * ncols},
        "tile_m": tile_m, "px": px, "metres_per_pixel": res,
        "grid_bbox_rd": [round(v, 3) for v in (x0, top - nrows * tile_m, x0 + ncols * tile_m, top)],
        "channels": legend, "data_vintages": vintages, "tiles": index,
    }, indent=1, ensure_ascii=False))

    # --- checks -------------------------------------------------------------
    allowed = {"R": {0, *NOISE_LEVELS.values()}, "G": set(SURFACE_MEANING), "B": {0, *POI_LEVELS.values()}}
    total = nrows * ncols * px * px
    for ch in "RGB":
        seen = set(np.nonzero(hist[ch])[0].tolist())
        assert seen <= allowed[ch], f"channel {ch} has unexpected levels {sorted(seen - allowed[ch])}"
        print(f"  {ch}: " + "  ".join(f"{lvl}={hist[ch][lvl] / total:.4f}" for lvl in sorted(seen)))
    print(f"tiles with zero establishments: {n_empty}/{nrows * ncols}; "
          f"PNG bytes {n_bytes / 1e6:.1f} MB; {time.time() - t_start:.1f} s")

    _district_overlay(ddir, district, noise, streets, buildings, pois,
                      (x0, top - nrows * tile_m, x0 + ncols * tile_m, top), tile_m, nrows, ncols)
    _contact_sheet(ddir, out_dir, index, mean_r)


def _district_overlay(ddir, district, noise, streets, buildings, pois, gb, tile_m, nrows, ncols,
                      res=2.0) -> None:
    """Whole grid extent rasterised identically at ~2 m/px, with the tile grid on top."""
    w, h = int(round((gb[2] - gb[0]) / res)), int(round((gb[3] - gb[1]) / res))
    tr = from_origin(gb[0], gb[3], res, res)
    rgb = np.dstack([_burn(_r_pairs(noise), tr, h, w),
                     _burn(_g_pairs(streets, buildings), tr, h, w),
                     _burn(_b_pairs(pois)[0], tr, h, w)])
    fig, ax = plt.subplots(figsize=(11, 11 * h / w), dpi=110)
    ax.imshow(rgb, extent=(gb[0], gb[2], gb[1], gb[3]), interpolation="nearest")
    for c in range(ncols + 1):
        ax.axvline(gb[0] + c * tile_m, color="white", lw=0.6, alpha=0.6)
    for r in range(nrows + 1):
        ax.axhline(gb[3] - r * tile_m, color="white", lw=0.6, alpha=0.6)
    for r in range(nrows):
        for c in range(ncols):
            ax.text(gb[0] + c * tile_m + 6, gb[3] - r * tile_m - 6, f"r{r}_c{c}",
                    color="white", ha="left", va="top", fontsize=7,
                    bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.0))
    ax.set_title(f"{district.label} - tiles/{TILESET}, {tile_m:g} m tiles at {res:g} m/px\n"
                 "R = noise Lden band, G = surface class, B = named establishments")
    ax.set_xlabel("RD x (m)")
    ax.set_ylabel("RD y (m)")
    fig.savefig(ddir / "overlays" / f"tiles_{TILESET}_district.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contact_sheet(ddir: Path, out_dir: Path, index: list, mean_r: dict) -> None:
    """6 diverse tiles x (R, G, B, RGB). Picks: 4 hard-coded landmarks + 2 rules."""
    by_id = {t["tile_id"]: t for t in index}
    picks: list[tuple[str, str]] = []
    for label, (lon, lat) in CONTACT_POINTS.items():
        x, y = _TO_RD.transform(lon, lat)
        hit = next((t for t in index if t["bbox_rd"][0] <= x < t["bbox_rd"][2]
                    and t["bbox_rd"][1] <= y < t["bbox_rd"][3]), None)
        if hit and hit["tile_id"] not in dict(picks):
            picks.append((hit["tile_id"], label))
    order = sorted(by_id, key=lambda t: mean_r[t])
    for label, seq in zip(CONTACT_RULES, (order[::-1], order)):
        pick = next(t for t in seq if t not in dict(picks))
        picks.append((pick, label))
    while len(picks) < 6:  # a landmark may have collided with an earlier pick
        picks.append((next(t for t in order[::-1] if t not in dict(picks)), "next loudest"))

    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)), dpi=100)
    for i, ((tid, label), row) in enumerate(zip(picks, axes)):
        rgb = np.asarray(Image.open(out_dir / f"{tid}.png"))
        for k, ax in enumerate(row):
            ax.imshow(rgb[:, :, k] if k < 3 else rgb, cmap="gray" if k < 3 else None,
                      vmin=0 if k < 3 else None, vmax=255 if k < 3 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"{tid}\n{label}", fontsize=7.5)
        if i == 0:  # legend on the top row only, else the titles collide
            for ax, t in zip(row, COLUMN_TITLES):
                ax.set_title(t, fontsize=7)
    fig.suptitle(f"tiles/{TILESET} contact sheet - 6 tiles, 256 m each", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(ddir / "overlays" / f"tiles_{TILESET}_contact_sheet.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("contact sheet: " + ", ".join(f"{t} ({l})" for t, l in picks))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="worldsnap.tiles", description=__doc__)
    p.add_argument("--district", required=True, choices=sorted(DISTRICTS))
    p.add_argument("--tile-m", type=float, default=256.0)
    p.add_argument("--px", type=int, default=1024)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args(argv)
    build_tiles(DISTRICTS[args.district], args.tile_m, args.px, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
