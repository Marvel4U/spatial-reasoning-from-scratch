"""Derived layer: sunlit hours at ground level, from 3DBAG building heights.

WHAT IS STORED AND WHY THESE CHOICES (C4_PLAN section 4c)
---------------------------------------------------------
Three rasters on the district's **1 m grid**, byte-for-byte the same frame as
``district_labels_1m`` in every cropset (``crops.district_grid``), so the sun
plane and the label planes are the same pixels and no resampling ever happens
between them:

* ``sunlit_hours_0621_1m.tif`` - hours of direct sun on **21 June 2026**,
* ``sunlit_hours_0321_1m.tif`` - the same for **21 March 2026**,
* ``shadow_0621_1500utc_1m.tif`` - the 15:00 UTC shadow on 21 June, uint8 0/1,
  kept because a single instant is what a "is that terrace in the sun now?"
  question needs and because it is the raster the AHN cross-check runs on.

Definitional choices, all of them arguable, all of them recorded here and in the
manifest entry so a reader can disagree with a specific line rather than with a
number:

1. **Buildings only.** Trees are in the AHN surface model but as one season's
   noisy crowns with no notion of foliage; mixing them into a building shadow
   would make the layer unverifiable against 3DBAG. A tree variant is a separate
   layer, compared, never mixed. The AHN cross-check below quantifies exactly
   what that omission costs.
2. **Height = 3DBAG ``height_m`` = ``b3_h_dak_70p - b3_h_maaiveld``**, the same
   canonical height the buildings layer and the crops use (see
   ``layers/buildings.py``). Buildings whose height is missing are still painted
   into the *footprint* mask (so their pixels are "not ground") but cast no
   shadow; their count is reported rather than filled in. No cleaning.
3. **Flat roofs at the 70th percentile.** A 2.5D extrusion, not the LoD2.2
   solid: a pitched roof shadows as a box at 70 % of its slope. This is the same
   simplification the whole snapshot already makes.
4. **Year 2026.** The solar declination on a given calendar date moves by under
   0.2 deg across the leap cycle, i.e. under 1 % of a June shadow length, but a
   date is not a date without a year.
5. **One sun position for the whole district.** Over 1.8 km the sun's azimuth
   changes by ~0.02 deg. The position is computed at the district bbox centre.
6. **UTC calendar day, 15-minute steps at the step midpoint**, 200 m shadow
   reach - see ``worldsnap.sun`` for what each of those costs.
7. **Building interiors are nodata (NaN)**, never 0. A courtyard that really
   never sees the sun and a roof pixel must not share a value.

THE BUILT-IN CHECK
------------------
``ahn_ndsm_1m`` rebuilds a second height raster from the AHN5 DSM minus the
interpolated DTM ground surface, at 0.5 m, max-reduced to 1 m (max, not mean:
the AHN5 DSM is itself a per-cell highest-point product, and for shadow casting
the highest obstacle is the one that matters). Its shadow at 15:00 UTC on
21 June is compared with the 3DBAG one over ground pixels. The two disagree by
construction - the nDSM contains trees, scaffolding, roof clutter and the DTM
fill's own error, while 3DBAG contains idealised boxes - so the number is a
*characterisation of the definition*, not a validation of either source, and the
two are never averaged or merged.
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import torch
from rasterio.transform import from_origin

from ..config import CRS_RD, District
from ..sun import shadow_mask, sun_position, sunlit_hours
from . import elevation as elevation_layer

RES_M = 1.0
STEP_MINUTES = 15
MAX_DIST_M = 200.0
YEAR = 2026
DATES = {"0621": _dt.date(YEAR, 6, 21), "0321": _dt.date(YEAR, 3, 21)}
SHADOW_EXAMPLE = ("0621", 15)  # (date key, whole hour UTC)
HOURS_NAME = "sunlit_hours_{key}_1m.tif"
SHADOW_NAME = "shadow_0621_1500utc_1m.tif"
FORMULA_SOURCE = ("Astronomical Almanac section C24 low-precision solar formulae "
                  "(as in the NOAA solar calculator / Meeus ch. 25 abridged); "
                  "no refraction, no solar disc radius - see worldsnap/sun.py")
HEIGHT_FIELD = "height_m (= b3_h_dak_70p - b3_h_maaiveld, 3DBAG)"


def district_grid(district: District) -> dict[str, Any]:
    """The district's 1 m frame. MUST stay identical to ``crops.build_crops``.

    Duplicating the two ``floor``/``ceil`` lines would be the classic way to end
    up with a half-pixel shift between the sun plane and the label planes, so
    ``crops`` imports this function instead of repeating them.
    """
    bx0, by0, bx1, by1 = district.bbox_rd
    ox, oy_top = float(np.floor(bx0)), float(np.ceil(by1))
    w_m, h_m = int(np.ceil(bx1) - ox), int(oy_top - np.floor(by0))
    return {"ox": ox, "oy_top": oy_top, "width_m": w_m, "height_m": h_m,
            "transform": from_origin(ox, oy_top, RES_M, RES_M),
            "bbox": (ox, oy_top - h_m, ox + w_m, oy_top)}


def height_rasters(buildings: gpd.GeoDataFrame, grid: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rasterise 3DBAG footprints to (height above ground, building mask) at 1 m.

    Overlapping footprints are painted tallest-last so the taller building wins;
    the mask is painted from *every* footprint, including those without a usable
    height, because "is this pixel ground?" is a question about the footprint,
    not about the attribute.
    """
    shape = (grid["height_m"], grid["width_m"])
    tr = grid["transform"]
    mask = rasterio.features.rasterize(
        ((g, 1) for g in buildings.geometry), out_shape=shape, transform=tr,
        all_touched=False, dtype="uint8").astype(bool)
    usable = buildings["height_m"].notna()
    rows = buildings[usable].sort_values("height_m")  # taller painted last
    height = np.zeros(shape, dtype="float32")
    rasterio.features.rasterize(
        zip(rows.geometry, rows["height_m"].astype("float32")), out=height, transform=tr,
        all_touched=False, dtype="float32")
    stats = {"buildings": int(len(buildings)), "without_height": int((~usable).sum()),
             "non_positive_height": int((buildings["height_m"] <= 0).sum()),
             "footprint_pixel_fraction": round(float(mask.mean()), 6),
             "height_p50_m": round(float(np.median(height[mask & (height > 0)])), 2),
             "height_max_m": round(float(height.max()), 2)}
    return height, mask, stats


def ahn_ndsm_1m(dtm_path: Path, dsm_path: Path, grid: dict) -> tuple[np.ndarray, dict]:
    """AHN5 nDSM (DSM minus interpolated ground) resampled to the district 1 m grid.

    Max over each 2 x 2 block of 0.5 m cells, for the reason in the module
    docstring. The AHN clip is snapped to the 0.5 m grid and the district frame
    to whole metres, so the block boundaries are exact integers and nothing is
    interpolated horizontally; cells the AHN clip does not reach (it can fall
    half a metre short at the north/east edge, because the two snapping rules
    round in opposite directions) stay NaN and are reported.
    """
    dtm, ahn_tr, _ = elevation_layer.read_masked(dtm_path)
    dsm, _, _ = elevation_layer.read_masked(dsm_path)
    ground, fill_info = elevation_layer.ground_surface(dtm)
    ndsm = np.ma.filled(dsm - ground, np.nan).astype("float32")

    half = elevation_layer.PIXEL_M
    col0 = int(round((grid["ox"] - ahn_tr.c) / half))
    row0 = int(round((ahn_tr.f - grid["oy_top"]) / half))
    need = (2 * grid["height_m"], 2 * grid["width_m"])
    win = np.full(need, np.nan, dtype="float32")
    r0, c0 = max(0, row0), max(0, col0)
    r1 = min(ndsm.shape[0], row0 + need[0])
    c1 = min(ndsm.shape[1], col0 + need[1])
    win[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = ndsm[r0:r1, c0:c1]
    outside = np.ones(need, dtype=bool)
    outside[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = False
    blocks = win.reshape(grid["height_m"], 2, grid["width_m"], 2)
    with np.errstate(invalid="ignore"):  # an all-NaN 2x2 block is a legitimate result
        out = np.nanmax(blocks, axis=(1, 3)).astype("float32")
    info = {"resample": "max over 2x2 cells of the 0.5 m AHN grid (AHN5 DSM is itself "
                        "a per-cell highest point)",
            "ground_reference": "DTM with nodata interpolated (elevation.ground_surface)",
            "dtm_nodata_fraction_before_fill": fill_info["dtm_nodata_fraction_before_fill"],
            # Two very different reasons for a NaN, kept apart on purpose: the AHN5 DSM
            # simply has no water (by construction), while the edge strip is the two
            # snapping rules rounding in opposite directions and is one 0.5 m row wide.
            "nan_1m_cells_total": int(np.isnan(out).sum()),
            "cells_05m_outside_ahn_clip": int(outside.sum()),
            "dsm_nodata_is_water": True,
            "ndsm_p50_m": round(float(np.nanmedian(out)), 2),
            "ndsm_max_m": round(float(np.nanmax(out)), 2)}
    return out, info


def example_instant(lat: float, lon: float) -> tuple[_dt.datetime, float, float]:
    """The single instant the example shadow and the AHN cross-check both use."""
    date_key, hour_utc = SHADOW_EXAMPLE
    instant = _dt.datetime.combine(DATES[date_key], _dt.time(hour_utc),
                                   tzinfo=_dt.timezone.utc)
    return (instant, *sun_position(lat, lon, instant))


def compute(height: np.ndarray, mask: np.ndarray, lat: float, lon: float,
            device: str = "cuda") -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Sunlit hours per date plus the example shadow. Returns (hours, shadow, timing)."""
    dev = device if torch.cuda.is_available() else "cpu"
    h = torch.as_tensor(height, device=dev)
    m = torch.as_tensor(mask, device=dev)
    hours, timing = {}, {"device": dev}
    for key, date in DATES.items():
        t0 = time.time()
        hours[key] = sunlit_hours(h, RES_M, lat, lon, date, STEP_MINUTES, MAX_DIST_M,
                                  building=m).cpu().numpy()
        timing[f"seconds_{key}"] = round(time.time() - t0, 1)
    instant, elevation, azimuth = example_instant(lat, lon)
    shadow, _ = shadow_mask(h, RES_M, elevation, azimuth, MAX_DIST_M, building=m)
    timing["example_sun"] = {"utc": instant.isoformat(), "elevation_deg": round(elevation, 2),
                             "azimuth_deg": round(azimuth, 2)}
    return hours, shadow.cpu().numpy(), timing


def ahn_disagreement(ndsm_1m: np.ndarray, reference_shadow: np.ndarray, mask: np.ndarray,
                     lat: float, lon: float, device: str = "cuda") -> dict:
    """Shadow from the AHN nDSM at the example instant vs the 3DBAG shadow.

    NaN cells (water, and the edge strip the AHN clip misses) are set to 0 m:
    "no obstacle known here". They are counted so the reader can see how much of
    the comparison rests on that.
    """
    dev = device if torch.cuda.is_available() else "cpu"
    filled = np.nan_to_num(ndsm_1m, nan=0.0)
    h = torch.as_tensor(filled, device=dev)
    m = torch.as_tensor(mask, device=dev)
    instant, elevation, azimuth = example_instant(lat, lon)
    shadow, _ = shadow_mask(h, RES_M, elevation, azimuth, MAX_DIST_M, building=m)
    ahn = shadow.cpu().numpy()
    ground = ~mask
    n = int(ground.sum())
    # What the two sources actually disagree *about*: how much structure the AHN
    # surface carries on pixels 3DBAG calls ground (street trees, annexes and
    # balconies outside the footprint, and the 2x2 max bleeding a roof one cell out).
    on_ground = ndsm_1m[ground]
    return {"instant_utc": instant.isoformat(),
            "ahn_ndsm_over_3dbag_ground_gt2m": round(float(np.nanmean(on_ground > 2)), 4),
            "ahn_ndsm_over_3dbag_ground_gt8m": round(float(np.nanmean(on_ground > 8)), 4),
            "sun_elevation_deg": round(elevation, 2), "sun_azimuth_deg": round(azimuth, 2),
            "ground_cells": n,
            "shadow_fraction_3dbag": round(float(reference_shadow[ground].mean()), 4),
            "shadow_fraction_ahn_ndsm": round(float(ahn[ground].mean()), 4),
            "disagreement_rate": round(float((ahn[ground] != reference_shadow[ground]).mean()), 4),
            "ahn_shadow_only": round(float((ahn[ground] & ~reference_shadow[ground]).mean()), 4),
            "3dbag_shadow_only": round(float((~ahn[ground] & reference_shadow[ground]).mean()), 4),
            "ndsm_nan_cells_treated_as_zero": int(np.isnan(ndsm_1m).sum())}


def percentiles(hours: np.ndarray) -> dict:
    """p10/p50/p90 of sunlit hours over ground pixels (NaN = building = excluded)."""
    v = hours[~np.isnan(hours)]
    p10, p50, p90 = (round(float(x), 2) for x in np.percentile(v, [10, 50, 90]))
    return {"ground_cells": int(v.size), "p10": p10, "p50": p50, "p90": p90,
            "mean": round(float(v.mean()), 2), "min": round(float(v.min()), 2),
            "max": round(float(v.max()), 2)}


def write_raster(path: Path, array: np.ndarray, grid: dict, dtype: str, nodata) -> Path:
    """Tiled DEFLATE GeoTIFF on the district 1 m frame, EPSG:28992 (as elevation.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype=dtype, crs=CRS_RD, transform=grid["transform"],
                       nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate") as dst:
        dst.write(array.astype(dtype), 1)
    return path
