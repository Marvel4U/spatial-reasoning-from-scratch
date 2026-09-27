"""AHN elevation rasters: 0.5 m DTM (maaiveld) and DSM, clipped to a district.

WHICH AHN, and why not the one the survey named
-----------------------------------------------
The survey pointed at the PDOK Atom feeds
``https://service.pdok.nl/rws/ahn/atom/{dtm,dsm}_05m.xml`` and called them
AHN5. They are not. Probed 2026-09-12, the dataset metadata behind those feeds
(NGR record ``41daef8b-155e-4608-b49c-c87ea45d931c``, revised 2026-05-26) says
in as many words: *"Het huidige AHN is versie 4, deze versie is ingewonnen over
de jaren 2020, 2021 en 2022"*, with a temporal extent of 2019-11-30 to
2022-03-25. The AHN dataroom confirms it from the other side: *"Het maaiveld-
(DTM) en oppervlaktemodel (DSM) van het AHN4 zijn via PDOK beschikbaar als
Cloud Optimized GeoTIFF"*. PDOK serves **AHN4**.

AHN5 (and, for the north-east of the country only, AHN6) is published instead
from the programme's own object store, indexed by a national *bladwijzer*
GeoPackage that carries one download URL per kaartblad per product per version.
Kaartblad **25GN1** -- the one covering De Pijp -- has an AHN5 tile whose file
name is prefixed ``2023_``, so AHN5 *is* published for Amsterdam and the brief's
"AHN5 if published, else AHN4" resolves to AHN5. The version is a module
parameter, not a constant, so an AHN4 comparison is one argument away.

This matters beyond version pedantry, because AHN5 changed the resampling:

* AHN2/3/4 DTM = inverse-distance-weighted mean of ground points; AHN5 DTM =
  **unweighted mean**.
* AHN2/3/4 DSM = IDW mean of all points; AHN5 DSM = **highest point** in the
  cell (water excluded).

The DSM change is exactly the one that matters here: a max-value DSM puts roof
ridges and tree tops at their true height instead of averaging them down, which
is what makes the DSM-minus-DTM cross-check against 3DBAG meaningful. Mixing
AHN4 and AHN5 rasters in one analysis would introduce a systematic step; the
manifest records the version for that reason.

Storage and CRS
---------------
AHN ships as EPSG:7415 = RD New + NAP height, a *compound* CRS: the horizontal
component is EPSG:28992 and the pixel values are metres above NAP. The clip is
written with CRS EPSG:28992 to match every other layer in the snapshot, and the
manifest records ``vertical_datum: NAP (EPSG:5709)`` so the dropped half of the
compound CRS is not lost. This is the same trade already made for 3DBAG.

Nodata is the provider's own sentinel, float32 max (~3.4e38), and is kept
verbatim rather than rewritten to NaN: nodata cells are real information (water
is absent from the AHN5 DSM by construction, and the DTM has holes wherever no
point was classified as ground), and the fraction of them is reported.

WHY a windowed read and not a tile download
-------------------------------------------
The two source kaartbladen are 213 MB (DTM) and 257 MB (DSM) for 5 x 6.25 km.
The district needs 3.2 km^2 of that. Both files are internally tiled 256 x 256
DEFLATE GeoTIFFs on a server that honours byte ranges, so GDAL's ``/vsicurl/``
reads only the blocks the window touches -- a few MB instead of 470 MB. The
cost is that the *source* bytes are not archived the way the BGT and bomen
responses are. As a substitute anchor the manifest records, per tile, the
download URL, the HTTP ``Content-Length``, ``Last-Modified`` and ``ETag``
observed at fetch time, plus the sha256 of the clip actually written. The
national bladwijzer index *is* archived, because it is 4 MB and it is the thing
that decides which tile was used.

The output grid is snapped outward to the AHN 0.5 m grid (which is aligned to
whole RD metres), so the clip is a pure subset of the source pixels: no
resampling, no interpolation, no half-pixel shift, and two districts cut from
the same tile share a grid.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.windows import Window
from shapely.geometry import box

from ..config import CRS_RD, District

#: National kaartblad index for AHN2/3/4/5, one row per 5 x 6.25 km TOP sheet,
#: one column per (version, product) holding the download URL. This is the
#: programme's own index; it is what makes the tile choice checkable.
BLADWIJZER_URL = "https://basisdata.nl/hwh-ahn/AUX/bladwijzer.gpkg"
DATAROOM_URL = "https://www.ahn.nl/dataroom"
SOURCE_NAME = "Actueel Hoogtebestand Nederland (AHN), 0.5 m raster, via ahn.nl dataroom"
#: The AHN programme publishes under CC0 1.0 (the PDOK Atom feed for the same
#: programme states ``https://creativecommons.org/publicdomain/zero/1.0/deed.nl``
#: in its ``<rights>`` element, and the NGR record says "Geen beperkingen").
LICENCE = "CC0 1.0 Universal (public domain dedication)"
OWNER = "Stuurgroep AHN (Rijkswaterstaat, waterschappen, provincies)"

#: AHN version -> (DTM column, DSM column) in the bladwijzer attribute table.
VERSION_COLUMNS: dict[str, tuple[str, str]] = {
    "AHN3": ("AHN3_05M_M", "AHN3_05M_R"),
    "AHN4": ("AHN4_05M_M", "AHN4_05M_R"),
    "AHN5": ("AHN5_05M_M", "AHN5_05M_R"),
}
#: Default version: the newest one that covers Amsterdam (see module docstring).
DEFAULT_VERSION = "AHN5"

#: Native resolution of the 0.5 m products, and the grid the clip snaps to.
PIXEL_M = 0.5

#: GDAL settings for range-reading a remote GeoTIFF: do not list the "directory"
#: (there is none, it is an object store) and only trust the extensions we ask
#: for, which stops GDAL probing for sidecar files that do not exist.
_VSICURL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

_UA = {"User-Agent": "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)"}


def _snap_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Grow ``bbox`` outward to whole 0.5 m AHN grid lines."""
    x0, y0, x1, y1 = bbox
    return (
        math.floor(x0 / PIXEL_M) * PIXEL_M,
        math.floor(y0 / PIXEL_M) * PIXEL_M,
        math.ceil(x1 / PIXEL_M) * PIXEL_M,
        math.ceil(y1 / PIXEL_M) * PIXEL_M,
    )


def _tile_headers(session: requests.Session, url: str) -> dict[str, Any]:
    """HTTP provenance for a source tile we deliberately do not archive."""
    try:
        head = session.head(url, allow_redirects=True, timeout=60)
        return {
            "url": url,
            "http_status": head.status_code,
            "content_length": int(head.headers.get("Content-Length", 0)) or None,
            "last_modified": head.headers.get("Last-Modified"),
            "etag": head.headers.get("ETag"),
        }
    except requests.RequestException as exc:  # provenance is best-effort, never fatal
        return {"url": url, "http_error": repr(exc)}


def _bladwijzer(raw_dir: Path | None, instant: str, verbose: bool) -> tuple[gpd.GeoDataFrame, Path | None]:
    """Fetch (and archive) the national kaartblad index."""
    resp = requests.get(BLADWIJZER_URL, headers=_UA, timeout=300)
    resp.raise_for_status()
    path = None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = instant.replace(":", "").replace("-", "")
        path = raw_dir / f"ahn_bladwijzer_{stamp}.gpkg"
        path.write_bytes(resp.content)
        index = gpd.read_file(path)
    else:  # pragma: no cover - the build always archives
        tmp = Path("/tmp") / "ahn_bladwijzer.gpkg"
        tmp.write_bytes(resp.content)
        index = gpd.read_file(tmp)
    if verbose:
        print(f"  [ahn] bladwijzer: {len(index)} kaartbladen, {len(resp.content) / 1e6:.1f} MB")
    return index, path


def _read_clip(
    urls: list[str],
    out_bbox: tuple[float, float, float, float],
    *,
    verbose: bool,
    label: str,
) -> tuple[np.ndarray, rasterio.Affine, float, dict[str, Any]]:
    """Mosaic the windows of ``urls`` that fall inside ``out_bbox``.

    One district normally sits inside a single kaartblad; the loop exists so a
    district straddling a sheet edge does not silently lose half its raster.
    Every tile is on the same 0.5 m grid, so the paste is an integer index
    operation -- there is no resampling anywhere in this function.
    """
    x0, y0, x1, y1 = out_bbox
    width = int(round((x1 - x0) / PIXEL_M))
    height = int(round((y1 - y0) / PIXEL_M))
    transform = rasterio.Affine(PIXEL_M, 0.0, x0, 0.0, -PIXEL_M, y1)

    out: np.ndarray | None = None
    nodata: float | None = None
    per_tile: list[dict[str, Any]] = []

    with rasterio.Env(**_VSICURL_ENV):
        for url in urls:
            t0 = time.time()
            with rasterio.open("/vsicurl/" + url) as ds:
                if nodata is None:
                    nodata = float(ds.nodata)
                    out = np.full((height, width), nodata, dtype="float32")
                assert out is not None
                src_crs_wkt = ds.crs.to_wkt()
                # Overlap of the requested box with this tile, in tile pixels.
                tx0 = max(x0, ds.bounds.left)
                tx1 = min(x1, ds.bounds.right)
                ty0 = max(y0, ds.bounds.bottom)
                ty1 = min(y1, ds.bounds.top)
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                col_off = int(round((tx0 - ds.bounds.left) / PIXEL_M))
                row_off = int(round((ds.bounds.top - ty1) / PIXEL_M))
                win = Window(
                    col_off, row_off,
                    int(round((tx1 - tx0) / PIXEL_M)),
                    int(round((ty1 - ty0) / PIXEL_M)),
                )
                block = ds.read(1, window=win)
                oc = int(round((tx0 - x0) / PIXEL_M))
                orow = int(round((y1 - ty1) / PIXEL_M))
                out[orow:orow + block.shape[0], oc:oc + block.shape[1]] = block
            per_tile.append({
                "url": url,
                "window_px": [win.col_off, win.row_off, win.width, win.height],
                "seconds": round(time.time() - t0, 1),
                "source_crs_is_compound_7415": "COMPD_CS" in src_crs_wkt,
            })
            if verbose:
                print(f"  [ahn] {label}: read {win.width}x{win.height} px "
                      f"in {per_tile[-1]['seconds']}s from {url.rsplit('/', 1)[-1]}")

    if out is None:
        raise RuntimeError(f"no AHN tile overlapped the district for {label}")
    return out, transform, float(nodata), {"tiles": per_tile}


def _write_gtiff(path: Path, array: np.ndarray, transform, nodata: float) -> None:
    """Tiled, DEFLATE-compressed, float32 GeoTIFF in EPSG:28992."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32", crs=CRS_RD, transform=transform, nodata=nodata,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3,
    ) as dst:
        dst.write(array, 1)


def fetch(
    district: District,
    *,
    out_dir: Path,
    raw_dir: Path | None = None,
    instant: str,
    version: str = DEFAULT_VERSION,
    verbose: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write ``dtm_05m.tif`` and ``dsm_05m.tif`` for ``district``.

    Returns the written paths keyed ``"dtm"``/``"dsm"`` and a stats dict.
    """
    t0 = time.time()
    if version not in VERSION_COLUMNS:
        raise ValueError(f"unknown AHN version {version!r}; known: {sorted(VERSION_COLUMNS)}")
    dtm_col, dsm_col = VERSION_COLUMNS[version]

    index, blad_path = _bladwijzer(raw_dir, instant, verbose)
    hits = index[index.intersects(box(*district.bbox_rd))]
    if hits.empty:
        raise RuntimeError(f"no AHN kaartblad covers {district.label}")
    missing = hits[hits[dtm_col].isna() | hits[dsm_col].isna()]
    if len(missing):
        raise RuntimeError(
            f"{version} not published for kaartblad(en) "
            f"{sorted(missing['AHN'])} - rerun with an older version"
        )

    out_bbox = _snap_bbox(district.bbox_rd)
    session = requests.Session()
    session.headers.update(_UA)

    paths: dict[str, Path] = {}
    per_product: dict[str, Any] = {}
    for key, col, fname in (("dtm", dtm_col, "dtm_05m.tif"), ("dsm", dsm_col, "dsm_05m.tif")):
        urls = list(hits[col])
        array, transform, nodata, info = _read_clip(
            urls, out_bbox, verbose=verbose, label=key.upper()
        )
        path = out_dir / fname
        _write_gtiff(path, array, transform, nodata)
        paths[key] = path
        valid = array != nodata
        per_product[key] = {
            **info,
            "http": [_tile_headers(session, u) for u in urls],
            "shape_px": [int(array.shape[0]), int(array.shape[1])],
            "nodata": nodata,
            "nodata_fraction": float(1.0 - valid.mean()),
            "valid_cells": int(valid.sum()),
        }

    stats = {
        "ahn_version": version,
        "kaartbladen": sorted(str(v) for v in hits["AHN"]),
        "bbox_snapped_rd": [round(v, 3) for v in out_bbox],
        "pixel_m": PIXEL_M,
        "vertical_datum": "NAP (EPSG:5709); source CRS EPSG:7415 = RD New + NAP height",
        "capture_year_from_filename": sorted(
            {Path(u).name.split("_", 1)[0] for u in hits[dtm_col] if Path(u).name[:4].isdigit()}
        ),
        "bladwijzer_file": None if blad_path is None else {
            "file": blad_path.name, "bytes": blad_path.stat().st_size, "url": BLADWIJZER_URL,
        },
        "products": per_product,
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
    }
    return paths, stats


def read_masked(path: Path) -> tuple[np.ma.MaskedArray, Any, float]:
    """Read a stored clip as a masked array, plus its transform and nodata."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True)
        return arr, ds.transform, float(ds.nodata)


def ground_surface(
    dtm: np.ma.MaskedArray, *, max_search_distance: float = 400.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ground height everywhere, by filling the DTM's nodata holes.

    WHY this is needed at all -- the single most important thing about this
    layer. An AHN DTM is built *only* from points classified as ground, so
    under every building there are no points and the DTM is nodata. In De Pijp
    that is **49% of all cells**. A naive nDSM = DSM - DTM is therefore masked
    out exactly where buildings are, i.e. it is blank precisely where the
    interesting height signal lives, which is how this was discovered.

    The fix is GDAL's inverse-distance fill (``rasterio.fill.fillnodata``)
    across the holes. It is legitimate here for a physical reason and not as a
    convenience: Amsterdam's ground surface spans about 2 m across the whole
    district (DTM p5 to p95), and a building block is ~40 m wide, so ground
    level under a block is genuinely well determined by the street level around
    it. The operation is exactly interpolation, never extrapolation of measured
    values: it leaves every observed cell bit-identical (asserted below) and
    only invents values where the sensor could not see the ground.

    This surface is **derived** and is deliberately not written to disk: only
    the two provider rasters are stored. Anything computed from it -- the nDSM
    render, the 3DBAG cross-check -- must say so, because a height above this
    surface carries the interpolation's error as well as the DSM's.
    """
    mask = np.ma.getmaskarray(dtm)
    from rasterio.fill import fillnodata

    filled = fillnodata(
        np.asarray(dtm.data, dtype="float32").copy(),
        mask=(~mask).astype("uint8"),
        max_search_distance=max_search_distance,
        smoothing_iterations=0,
    )
    observed = ~mask
    assert np.array_equal(filled[observed], dtm.data[observed]), "fill altered observed cells"
    unfilled = int((filled > 1e30).sum())
    return filled, {
        "method": "rasterio.fill.fillnodata (GDAL inverse-distance), no smoothing",
        "max_search_distance_px": max_search_distance,
        "dtm_nodata_fraction_before_fill": round(float(mask.mean()), 6),
        "cells_still_unfilled": unfilled,
        "observed_cells_unchanged": True,
    }


def hillshade(elev: np.ma.MaskedArray, *, azdeg: float = 315.0, altdeg: float = 45.0,
              vert_exag: float = 5.0) -> np.ndarray:
    """Grey-scale hillshade in [0, 1] for a 0.5 m elevation grid.

    ``vert_exag`` defaults to 5 because Amsterdam's terrain spans ~2 m over the
    district: at the honest 1:1 exaggeration the DTM hillshade is a uniform
    grey and shows nothing but LiDAR noise. The exaggeration is a *rendering*
    choice and is written into the overlay title so the picture cannot be
    mistaken for topography.
    """
    from matplotlib.colors import LightSource

    filled = elev.filled(float(np.ma.median(elev)))
    shade = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(
        filled, vert_exag=vert_exag, dx=PIXEL_M, dy=PIXEL_M
    )
    return np.ma.masked_array(shade, mask=np.ma.getmaskarray(elev))


def sample_points(array: np.ma.MaskedArray, transform, xs, ys) -> np.ma.MaskedArray:
    """Nearest-cell sample of ``array`` at RD coordinates ``xs``/``ys``.

    Points outside the grid come back masked rather than wrapped, so an
    out-of-frame building cannot quietly contribute a wrong value.
    """
    inv = ~transform
    cols, rows = inv * (np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64"))
    cols = np.floor(cols).astype("int64")
    rows = np.floor(rows).astype("int64")
    inside = (rows >= 0) & (rows < array.shape[0]) & (cols >= 0) & (cols < array.shape[1])
    out = np.ma.masked_all(len(cols), dtype="float32")
    out[inside] = array[rows[inside], cols[inside]]
    return out
