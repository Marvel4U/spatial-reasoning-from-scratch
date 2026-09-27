"""CLI: fetch -> save -> manifest -> render for one district and one layer set.

    python -m worldsnap.build --district de_pijp --layers buildings,streets_bgt

WHY a single linear command
---------------------------
The snapshot is meant to be reproducible by one command with no hidden state,
so that the manifest's provenance claim ("this file came from that URL on that
date") is something anyone can re-run and check. Each layer is independent;
adding a layer means adding a branch here and a module under ``worldsnap/layers``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import median

import geopandas as gpd

from . import manifest as mf
from .config import CRS_RD, DISTRICTS, District
from shapely.geometry import box as _shp_box

from .layers import buildings as buildings_layer
from .layers import elevation as elevation_layer
from .layers import noise as noise_layer
from .layers import pois as pois_layer
from .layers import streets as streets_layer
from .layers import sunshine as sunshine_layer
from .layers import transit as transit_layer
from .layers import trees as trees_layer
from .render import (
    categorical_palette,
    render_buildings,
    render_categorical,
    render_noise,
    render_points,
    render_pois,
    render_raster,
    render_transit,
)

#: Repo-root-relative data directory (``worldsnap/`` lives at the repo root).
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

#: Construction-year colour range. BAG bouwjaar contains placeholder values
#: (e.g. 1005, 9999) and genuine pre-1850 buildings; clipping keeps the ramp
#: readable instead of letting one outlier flatten the whole district.
YEAR_VMIN, YEAR_VMAX = 1850, 2025

#: Vertical exaggeration for the DTM hillshade. De Pijp spans ~2 m of terrain;
#: at 1x the shaded relief of a Dutch district is a flat grey rectangle, so the
#: overlay would show nothing. Printed in the overlay title so the picture is
#: never mistaken for real topography.
HILLSHADE_VERT_EXAG = 5.0


def _report_buildings(gdf: gpd.GeoDataFrame, district: District) -> None:
    """Print the numbers a reviewer needs to judge whether the snapshot is sane."""
    n = len(gdf)
    area = district.area_km2_rd
    print(f"\n  buildings                : {n}")
    print(f"  bbox area (RD)           : {area:.3f} km^2")
    print(f"  density                  : {n / area:.0f} buildings / km^2")
    for col in ("height_m", "height_50p_m", "height_max_m"):
        s = gdf[col].dropna()
        print(
            f"  {col:<24} : min {s.min():6.2f}  p25 {s.quantile(.25):6.2f}  "
            f"median {s.median():6.2f}  p75 {s.quantile(.75):6.2f}  max {s.max():7.2f}"
            f"   (n={len(s)})"
        )
    yr = gdf["bouwjaar"].dropna()
    print(
        f"  bouwjaar                 : min {yr.min()}  median {yr.median():.0f}  "
        f"max {yr.max()}   (n={len(yr)})"
    )
    print(f"  missing height_m         : {int(gdf['height_m'].isna().sum())}")
    print(f"  missing bouwjaar         : {int(gdf['bouwjaar'].isna().sum())}")
    print(f"  non-positive height_m    : {int((gdf['height_m'] <= 0).sum())}")
    print(f"  bouwjaar outside 1500-2026: {int(((yr < 1500) | (yr > 2026)).sum())}")
    print("  status counts            :", dict(gdf["status"].value_counts()))
    print("  dak_type counts          :", dict(gdf["dak_type"].value_counts()))


def build_buildings(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Fetch 3DBAG buildings, store as GeoPackage, write manifest, render PNGs."""
    ddir = mf.district_dir(data_root, district.city, district.name)
    print(f"[buildings] fetching 3DBAG for {district.label} bbox_rd={district.bbox_rd}")
    gdf, stats = buildings_layer.fetch(district)
    print(f"[buildings] {stats['pages']} pages in {stats['seconds']}s")

    # GeoPackage (not Parquet) as the on-disk format: it is a single file, holds
    # the CRS, and opens directly in QGIS -- which matters because the human
    # spot-check protocol in DESIGN.md assumes someone can eyeball the data.
    out = ddir / "buildings_3dbag.gpkg"
    gdf.to_file(out, layer="buildings", driver="GPKG")

    _report_buildings(gdf, district)

    entry = mf.layer_entry(
        layer="buildings_3dbag",
        source_name=buildings_layer.SOURCE_NAME,
        source_url=buildings_layer.ITEMS_URL,
        licence=buildings_layer.LICENCE,
        vintage=stats["vintage"],
        crs=CRS_RD,
        bbox=district.bbox_rd,
        bbox_crs=CRS_RD,
        feature_count=len(gdf),
        file_path=out,
        district_dir=ddir,
        extra={
            "bbox_wgs84": list(district.bbox_wgs84),
            "height_field": stats["height_field"],
            "fetch_stats": {
                k: stats[k]
                for k in ("pages", "buildings_raw", "duplicates_dropped",
                          "features_without_lod0", "seconds")
            },
        },
    )
    path = mf.update_manifest(ddir, entry)
    print(f"[manifest] {path}")

    overlays = ddir / "overlays"
    vintage = stats["vintage"]
    render_buildings(
        gdf,
        overlays / "buildings_height.png",
        column="height_m",
        title=(
            f"{district.label} - building height above ground "
            f"(3DBAG b3_h_dak_70p - b3_h_maaiveld)\n"
            f"{len(gdf)} buildings | 3DBAG {vintage} | EPSG:28992"
        ),
        cbar_label="height above ground (m)",
        bbox=district.bbox_rd,
        vmin=0,
        vmax=float(gdf["height_m"].quantile(0.99)),  # 99th pct: keep outliers off the ramp
    )
    render_buildings(
        gdf,
        overlays / "buildings_year.png",
        column="bouwjaar",
        title=(
            f"{district.label} - construction year (BAG oorspronkelijkbouwjaar)\n"
            f"{len(gdf)} buildings | 3DBAG {vintage} | EPSG:28992 | "
            f"colour clipped to {YEAR_VMIN}-{YEAR_VMAX}"
        ),
        cbar_label=f"construction year (clipped {YEAR_VMIN}-{YEAR_VMAX})",
        bbox=district.bbox_rd,
        vmin=YEAR_VMIN,
        vmax=YEAR_VMAX,
    )
    print(f"[render] {overlays}/buildings_height.png")
    print(f"[render] {overlays}/buildings_year.png")




def _report_streets(gdf: gpd.GeoDataFrame, district: District) -> None:
    """Print the numbers a reviewer needs, including the ones that look wrong.

    Nothing here filters or repairs the data -- overlaps, multi-part geometries
    and objects hanging over the bbox edge are *properties of the BGT* (and of
    a rectangular clip of it) and must stay visible rather than be tidied away.
    """
    x0, y0, x1, y1 = district.bbox_rd
    print(f"\n  objects                  : {len(gdf)}")
    print("  per featuretype          :", dict(gdf["featuretype"].value_counts()))

    print("\n  functie                          n        area m2     median m2")
    grp = gdf.groupby(["featuretype", "functie"], dropna=False)
    for (ft, fn), sub in sorted(grp, key=lambda kv: -len(kv[1])):
        tag = f"{fn}" + ("" if ft == "wegdeel" else "  [ondersteunend]")
        print(f"    {tag:<34} {len(sub):5d}  {sub['area_m2'].sum():11.1f}  {sub['area_m2'].median():8.2f}")

    voetpad = gdf[(gdf["featuretype"] == "wegdeel") & (gdf["functie"] == "voetpad")]
    from shapely.geometry import box as _box
    clip_box = _box(x0, y0, x1, y1)
    print(f"\n  voetpad (sidewalk) objects : {len(voetpad)}")
    print(f"  voetpad total area         : {voetpad['area_m2'].sum():,.1f} m2  (whole objects)")
    # 466 of 10005 objects stick out over the bbox edge, so the whole-object sum
    # over-counts. Reported next to the clipped sum rather than instead of it: the
    # stored geometry is uncut, and a reader must be able to see both numbers.
    print(f"  voetpad area inside bbox   : {voetpad.geometry.intersection(clip_box).area.sum():,.1f} m2")
    print(f"  bbox area                  : {district.area_km2_rd * 1e6:,.1f} m2 "
          f"({100 * voetpad['area_m2'].sum() / (district.area_km2_rd * 1e6):.1f}% sidewalk)")

    print("\n  bgt_status               :", dict(gdf["bgt_status"].value_counts(dropna=False)))
    print("  relatieve_hoogteligging  :",
          dict(gdf["relatieve_hoogteligging"].value_counts(dropna=False).sort_index()))
    print("  fysiek_voorkomen         :", dict(gdf["fysiek_voorkomen"].value_counts(dropna=False)))
    print("  in_onderzoek non-null    :", int(gdf["in_onderzoek"].notna().sum()))
    print("  eind_registratie non-null:", int(gdf["eind_registratie"].notna().sum()))

    multi = int((gdf.geometry.geom_type != "Polygon").sum())
    invalid = int((~gdf.geometry.is_valid).sum())
    crossing = int((~gdf.geometry.within(clip_box)).sum())
    print(f"\n  non-Polygon geometries   : {multi}   "
          f"({dict(gdf.geometry.geom_type.value_counts())})")
    print(f"  invalid geometries       : {invalid}")
    print(f"  objects crossing bbox    : {crossing}  (kept whole, not clipped)")

    # Overlap: BGT is a planar partition *per height level*, so genuine overlaps
    # at the same relatieve_hoogteligging would be a data defect, while overlaps
    # across levels (bridge over road) are correct and expected.
    pairs = gpd.sjoin(
        gdf[["relatieve_hoogteligging", "geometry"]].reset_index(),
        gdf[["relatieve_hoogteligging", "geometry"]].reset_index(),
        predicate="overlaps",
    )
    pairs = pairs[pairs["index_left"] < pairs["index_right"]]  # each pair once
    is_same = pairs["relatieve_hoogteligging_left"] == pairs["relatieve_hoogteligging_right"]
    same, cross = int(is_same.sum()), int((~is_same).sum())
    # Same-level overlaps would be a planar-partition violation, so measure them
    # rather than just count them: in De Pijp all of them are zero-area slivers
    # along shared boundaries, i.e. floating-point noise, not real double cover.
    inter = gdf.geometry[pairs["index_left"]].reset_index(drop=True).intersection(
        gdf.geometry[pairs["index_right"]].reset_index(drop=True)
    ).area
    same_area = float(inter[is_same.to_numpy()].sum())
    print(f"  overlapping pairs        : {same} same height level "
          f"(total overlap area {same_area:.4f} m2), {cross} across levels")


def build_streets_bgt(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Fetch BGT road surfaces, store as GeoPackage, write manifest, render PNG."""
    ddir = mf.district_dir(data_root, district.city, district.name)
    instant = mf.utc_now_iso()
    print(f"[streets_bgt] fetching BGT {'+'.join(streets_layer.FEATURE_TYPES)} "
          f"for {district.label} bbox_rd={district.bbox_rd} at {instant}")
    gdf, stats = streets_layer.fetch(district, instant=instant, raw_dir=ddir / "raw")
    print(f"[streets_bgt] {stats['pages']} pages in {stats['seconds']}s")

    out = ddir / "streets_bgt.gpkg"
    gdf.to_file(out, layer="streets_bgt", driver="GPKG")

    _report_streets(gdf, district)

    # Vintage: the register has no single release version. What the provider
    # states is (a) the instant we asked it to be current at and (b) the span of
    # per-object registration times inside the extract -- both recorded.
    vintage = (
        f"extract {instant}; tijdstip_registratie "
        f"{stats['tijdstip_registratie_min']} .. {stats['tijdstip_registratie_max']}"
    )
    entry = mf.layer_entry(
        layer="streets_bgt",
        source_name=streets_layer.SOURCE_NAME,
        source_url=streets_layer.ITEMS_URL.format(featuretype="wegdeel"),
        licence=streets_layer.LICENCE,
        vintage=vintage,
        crs=CRS_RD,
        bbox=district.bbox_rd,
        bbox_crs=CRS_RD,
        feature_count=len(gdf),
        file_path=out,
        district_dir=ddir,
        extra={
            "bbox_wgs84": list(district.bbox_wgs84),
            "featuretypes": list(streets_layer.FEATURE_TYPES),
            "extract_instant": instant,
            "width_attribute": (
                "none - BGT stores no width; derive from polygon geometry downstream"
            ),
            "raw_archive": [
                {**rf, "dir": (ddir / "raw").relative_to(ddir).as_posix(),
                 "downloaded_utc": instant}
                for rf in stats["raw_files"]
            ],
            "fetch_stats": {
                k: stats[k]
                for k in ("pages", "objects_raw", "duplicates_dropped",
                          "features_without_geometry", "per_featuretype", "seconds")
            },
        },
    )
    path = mf.update_manifest(ddir, entry)
    print(f"[manifest] {path}")

    buildings_path = ddir / "buildings_3dbag.gpkg"
    context = gpd.read_file(buildings_path) if buildings_path.exists() else None
    if context is None:
        print("[render] note: buildings_3dbag.gpkg absent - rendering without context")

    counts = gdf["functie"].fillna("<null>").value_counts()
    voetpad_area = gdf.loc[gdf["functie"] == "voetpad", "area_m2"].sum()
    overlays = ddir / "overlays"
    render_categorical(
        gdf,
        overlays / "streets_bgt_functie.png",
        column="functie",
        title=(
            f"{district.label} - BGT road surfaces by functie "
            f"(wegdeel + ondersteunendwegdeel)\n"
            f"{len(gdf)} objects, {len(counts)} functie values | "
            f"voetpad {int(counts.get('voetpad', 0))} objects / {voetpad_area:,.0f} m2 | "
            f"BGT extract {instant} | EPSG:28992"
        ),
        bbox=district.bbox_rd,
        palette=categorical_palette(
            list(counts.index), streets_layer.FUNCTIE_COLORS
        ),
        context=context,
        legend_ncol=4,
    )
    print(f"[render] {overlays}/streets_bgt_functie.png")


def _report_transit(
    stops: gpd.GeoDataFrame, routes: gpd.GeoDataFrame, stats: dict, district: District
) -> None:
    """Print the numbers a reviewer needs, including the ones that look wrong."""
    import pandas as pd

    dep = stops["departures_0719"]
    print(f"\n  stops in bbox+{stats['buffer_m']:.0f}m  : {len(stops)}"
          f"   (of {stats['stops_in_feed']} nationally)")
    print("  location_type counts     :",
          dict(stops["location_type"].fillna("<null>").value_counts()))
    print(f"  routes serving district  : {stats['routes_serving_district']}"
          f"  ({len(routes)} drawn, {stats['routes_without_geometry']} without usable shape)")
    print("  routes per route_type    :",
          dict(routes["route_type_label"].fillna("<null>").value_counts()))
    print(f"\n  service date counted     : {stats['service_date']}")
    print(f"    rationale              : {stats['service_date_rationale']}")
    print(f"    calendar.txt present   : {stats['has_calendar_txt']}"
          "   (False -> service resolved from calendar_dates.txt alone)")
    print(f"    services active        : {stats['services_active_on_date']}")
    print(f"    stop_time rows 07-19   : {stats['departures_window_all_rows']} total, "
          f"{stats['departures_window_boardable']} boardable, "
          f"{stats['departures_window_arrivals_only']} arrival-only (pickup_type=1, excluded)")

    served = stops[dep > 0]
    print(f"\n  departures_0719          : total {int(dep.sum())} over {len(served)} served stops")
    if len(served):
        q = served["departures_0719"]
        print(f"    over served stops      : min {q.min()}  p25 {q.quantile(.25):.0f}  "
              f"median {q.median():.0f}  p75 {q.quantile(.75):.0f}  max {q.max()}")
        print(f"    implied headway median : {12 * 60 / q.median():.1f} min between departures")
    print(f"  stops with 0 departures  : {int((dep == 0).sum())}"
          "   (stop areas + weekend/rush-only quays; kept, not dropped)")
    print(f"    of which location_type=1: "
          f"{int(((dep == 0) & (stops['location_type'] == '1')).sum())}")
    print(f"  stops with no route at all: {int((stops['n_routes'] == 0).sum())}")

    print("\n  busiest 10 stops (departures 07:00-19:00):")
    for _, r in stops.nlargest(10, "departures_0719").iterrows():
        print(f"    {r['departures_0719']:5d}  {str(r['name'])[:34]:<34} "
              f"[{r['route_short_names']}]")

    print("\n  route_short_name reuse across agencies:")
    dup = routes.groupby("route_short_name")["agency"].nunique()
    dup = dup[dup > 1]
    print(f"    {len(dup)} short names used by >1 agency"
          + (f": {dict(dup)}" if len(dup) else ""))
    off = stops[~stops.geometry.within(_shp_box(*district.bbox_rd))]
    print(f"  stops inside buffer only : {len(off)}  (outside the strict bbox, kept by design)")


def build_transit_gtfs(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Download the national GTFS zip, cut stops+routes+frequency, render PNG."""
    from datetime import datetime, timezone

    ddir = mf.district_dir(data_root, district.city, district.name)
    instant = mf.utc_now_iso()
    today_utc = datetime.now(timezone.utc).date()
    stamp = instant[:10].replace("-", "")
    print(f"[transit_gtfs] {district.label} bbox_rd={district.bbox_rd} "
          f"+{transit_layer.BUFFER_M:.0f}m buffer, at {instant}")

    zip_path, dl_stats = transit_layer.download_feed(ddir / "raw", stamp=stamp)
    stops, routes, stats = transit_layer.fetch(
        district, zip_path=zip_path, today=today_utc
    )
    print(f"[transit_gtfs] parsed in {stats['seconds']}s")

    stops_out = ddir / "transit_stops.gpkg"
    routes_out = ddir / "transit_routes.gpkg"
    stops.to_file(stops_out, layer="transit_stops", driver="GPKG")
    routes.to_file(routes_out, layer="transit_routes", driver="GPKG")

    _report_transit(stops, routes, stats, district)

    fi = stats["feed_info"]
    vintage = (
        f"GTFS feed_version {fi['feed_version']} ({fi['feed_publisher_name']}), "
        f"valid {fi['feed_start_date']}..{fi['feed_end_date']}; "
        f"zip last-modified per HTTP header, extract {instant}"
    )
    shared = {
        "bbox_wgs84": list(district.bbox_wgs84),
        "buffer_m": transit_layer.BUFFER_M,
        "bbox_buffered_rd": [round(v, 3) for v in stats["bbox_buffered_rd"]],
        "extract_instant": instant,
        "service_date": stats["service_date"],
        "service_date_rationale": stats["service_date_rationale"],
        "frequency_window": "07:00:00-19:00:00 local service time, boardable departures only",
        "has_calendar_txt": stats["has_calendar_txt"],
        "feed_info": fi,
        "raw_archive": [{
            "file": zip_path.name,
            "dir": (ddir / "raw").relative_to(ddir).as_posix(),
            "bytes": zip_path.stat().st_size,
            "url": transit_layer.FEED_URL,
            "downloaded_utc": instant,
            **dl_stats,
        }],
        "fetch_stats": {
            k: stats[k] for k in (
                "stops_in_feed", "stops_in_buffered_bbox", "stop_times_rows_scanned",
                "stop_times_rows_at_district_stops", "services_active_on_date",
                "departures_window_all_rows", "departures_window_boardable",
                "departures_window_arrivals_only", "routes_serving_district",
                "routes_without_geometry", "shapes_requested",
                "shapes_with_geometry_in_bbox", "seconds",
            )
        },
    }
    for layer_name, out, n in (
        ("transit_stops", stops_out, len(stops)),
        ("transit_routes", routes_out, len(routes)),
    ):
        entry = mf.layer_entry(
            layer=layer_name,
            source_name=transit_layer.SOURCE_NAME,
            source_url=transit_layer.FEED_URL,
            licence=transit_layer.LICENCE,
            vintage=vintage,
            crs=CRS_RD,
            bbox=district.bbox_rd,
            bbox_crs=CRS_RD,
            feature_count=n,
            file_path=out,
            district_dir=ddir,
            extra=shared,
        )
        print(f"[manifest] {mf.update_manifest(ddir, entry)}")

    buildings_path = ddir / "buildings_3dbag.gpkg"
    streets_path = ddir / "streets_bgt.gpkg"
    context = gpd.read_file(buildings_path) if buildings_path.exists() else None
    carriage = None
    if streets_path.exists():
        s = gpd.read_file(streets_path)
        # Carriageways only: sidewalks and parking bays would swamp the route
        # lines they are supposed to sit on.
        carriage = s[s["functie"].fillna("").str.startswith("rijbaan")
                     | (s["functie"] == "OV-baan")]
    if context is None or carriage is None:
        print("[render] note: buildings/streets absent - rendering with partial context")

    overlays = ddir / "overlays"
    render_transit(
        stops, routes, overlays / "transit.png",
        title=(
            f"{district.label} - transit stops, routes and weekday frequency\n"
            f"{len(stops)} stops / {len(routes)} routes in bbox+"
            f"{transit_layer.BUFFER_M:.0f} m | departures 07:00-19:00 on "
            f"{stats['service_date']} (Tue) | GTFS NL v{fi['feed_version']} | EPSG:28992"
        ),
        bbox=district.bbox_rd,
        type_colors=transit_layer.ROUTE_TYPE_COLORS,
        type_labels=transit_layer.ROUTE_TYPE_LABELS,
        buildings=context,
        carriageways=carriage,
    )
    print(f"[render] {overlays}/transit.png")


def _report_trees(gdf: gpd.GeoDataFrame, stats: dict, district: District) -> None:
    """Print the numbers a reviewer needs, including the ones that look wrong."""
    n = len(gdf)
    area = district.area_km2_rd
    print(f"\n  trees                    : {n}")
    print(f"  bbox area (RD)           : {area:.3f} km^2")
    print(f"  density                  : {n / area:.0f} trees / km^2")

    yr = gdf["jaarVanAanleg"].dropna()
    miss = n - len(yr)
    print(f"\n  jaarVanAanleg missing    : {miss}  ({100 * miss / n:.1f}% of trees)")
    if len(yr):
        print(f"  jaarVanAanleg            : min {yr.min():.0f}  p25 {yr.quantile(.25):.0f}  "
              f"median {yr.median():.0f}  p75 {yr.quantile(.75):.0f}  max {yr.max():.0f}")
        print(f"  jaarVanAanleg outside 1700-2026: {int(((yr < 1700) | (yr > 2026)).sum())}")

    print("\n  top-10 soortnaam (species):")
    for name, c in gdf["soortnaam"].fillna("<null>").value_counts().head(10).items():
        print(f"    {c:5d}  ({100 * c / n:4.1f}%)  {name}")
    print(f"  distinct soortnaam       : {gdf['soortnaam'].nunique(dropna=True)}")
    print(f"  distinct soortnaamTop    : {gdf['soortnaamTop'].nunique(dropna=True)}")

    print("\n  boomhoogteklasseActueel  :")
    for k, c in gdf["boomhoogteklasseActueel"].fillna("<null>").value_counts().sort_index().items():
        print(f"    {c:5d}  {k}")
    print("  stamdiameterklasse       :")
    for k, c in gdf["stamdiameterklasse"].fillna("<null>").value_counts().sort_index().items():
        print(f"    {c:5d}  {k}")
    print("\n  typeObject               :", dict(gdf["typeObject"].fillna("<null>").value_counts()))
    print("  typeSoortnaam            :", dict(gdf["typeSoortnaam"].fillna("<null>").value_counts()))
    print("  typeEigenaarPlus         :", dict(gdf["typeEigenaarPlus"].fillna("<null>").value_counts()))
    print("  beschermingsstatus       :",
          dict(gdf["beschermingsstatus"].fillna("<null>").value_counts()))
    print("  standplaats              :", dict(gdf["standplaats"].fillna("<null>").value_counts()))

    dupe_xy = int(gdf.geometry.duplicated().sum())
    print(f"\n  identical coordinates    : {dupe_xy} features share a point with another")
    print(f"  outside strict bbox      : "
          f"{int((~gdf.geometry.within(_shp_box(*district.bbox_rd))).sum())}")
    print(f"  mutatieDatum span        : {stats['mutatie_datum_min']} .. {stats['mutatie_datum_max']}")


def build_trees(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Fetch the municipal tree register, store as GeoPackage, manifest, render."""
    ddir = mf.district_dir(data_root, district.city, district.name)
    instant = mf.utc_now_iso()
    print(f"[trees] fetching bomen/stamgegevens for {district.label} "
          f"bbox_rd={district.bbox_rd} at {instant}")
    gdf, stats = trees_layer.fetch(district, raw_dir=ddir / "raw", instant=instant)
    print(f"[trees] {stats['pages']} pages in {stats['seconds']}s")

    out = ddir / "trees.gpkg"
    gdf.to_file(out, layer="trees", driver="GPKG")

    _report_trees(gdf, stats, district)

    # Vintage: an asset-management register has no release version. What the
    # provider states is the per-record mutation date, so the span of those
    # inside the extract is the honest vintage, next to the extract instant.
    vintage = (
        f"extract {instant}; mutatieDatum "
        f"{stats['mutatie_datum_min']} .. {stats['mutatie_datum_max']}"
    )
    entry = mf.layer_entry(
        layer="trees",
        source_name=trees_layer.SOURCE_NAME,
        source_url=trees_layer.ITEMS_URL,
        licence=trees_layer.LICENCE,
        vintage=vintage,
        crs=CRS_RD,
        bbox=district.bbox_rd,
        bbox_crs=CRS_RD,
        feature_count=len(gdf),
        file_path=out,
        district_dir=ddir,
        extra={
            "bbox_wgs84": list(district.bbox_wgs84),
            "licence_note": (
                "licence string is copied verbatim from the dataset schema at "
                f"{trees_layer.SCHEMA_URL}; it is not an SPDX identifier"
            ),
            "owner": trees_layer.OWNER,
            "coverage_caveat": (
                "register of MUNICIPALLY MANAGED trees (source system Gisib); "
                "private gardens and courtyards are out of scope by construction"
            ),
            "extract_instant": instant,
            "raw_archive": [{
                **stats["raw_file"],
                "dir": (ddir / "raw").relative_to(ddir).as_posix(),
                "downloaded_utc": instant,
            }],
            "fetch_stats": {
                k: stats[k] for k in (
                    "pages", "trees_raw", "duplicates_dropped",
                    "features_without_geometry", "seconds",
                )
            },
        },
    )
    print(f"[manifest] {mf.update_manifest(ddir, entry)}")

    buildings_path = ddir / "buildings_3dbag.gpkg"
    streets_path = ddir / "streets_bgt.gpkg"
    context = gpd.read_file(buildings_path) if buildings_path.exists() else None
    streets = gpd.read_file(streets_path) if streets_path.exists() else None
    if context is None or streets is None:
        print("[render] note: buildings/streets absent - rendering with partial context")

    n = len(gdf)
    overlays = ddir / "overlays"
    render_points(
        gdf, overlays / "trees.png",
        column="boomhoogteklasseActueel",
        title=(
            f"{district.label} - municipal tree register (bomen/stamgegevens), "
            f"colour = boomhoogteklasseActueel\n"
            f"{n} trees | {n / district.area_km2_rd:.0f} trees/km^2 | "
            f"extract {instant} | EPSG:28992"
        ),
        bbox=district.bbox_rd,
        palette=trees_layer.HOOGTEKLASSE_COLORS,
        category_order=list(trees_layer.HOOGTEKLASSEN),
        buildings=context,
        streets=streets,
        legend_ncol=3,
    )
    print(f"[render] {overlays}/trees.png")



def _report_noise(gdf: gpd.GeoDataFrame, district: District) -> None:
    """Polygon counts and, more meaningfully, area share per dB class."""
    bbox_area = district.area_km2_rd * 1e6
    clipped = gdf.clip(_shp_box(*district.bbox_rd))
    print(f"\n  noise polygons in bbox   : {len(gdf)}")
    print(f"  sources present          : {sorted(set(gdf['bron_label']))}")
    print(f"  periods present          : {sorted(set(gdf['periode']))}")
    print(f"  jaar values              : {sorted(set(gdf['jaar'].dropna()))}")
    for (bron, periode), sub in gdf.groupby(["bron_slug", "periode"]):
        print(f"  -- {bron} / {periode}: {len(sub)} polygons")
        sub_clip = clipped[(clipped["bron_slug"] == bron) & (clipped["periode"] == periode)]
        order = noise_layer.band_order(periode)
        labels = [b for b in order if (sub["db_label"] == b).any()]
        labels += sorted(set(sub["db_label"]) - set(labels))
        for band in labels:
            n = int((sub["db_label"] == band).sum())
            a = float(sub_clip.loc[sub_clip["db_label"] == band, "geometry"].area.sum())
            print(f"       {band:<34} n={n:4d}  area_in_bbox={a / 1e4:8.2f} ha  "
                  f"({a / bbox_area * 100:5.2f}% of bbox)")
    # Printed rather than assumed: summed area vs union area says whether the
    # bands overlap. They come out disjoint for Amsterdam, so the shares above
    # are a real partition of the mapped area -- but that is a property of this
    # publisher, and a different city's END submission may well nest its rings.
    lden = clipped[clipped["periode"] == "lden"]
    total = float(lden.geometry.area.sum())
    union = float(lden.geometry.union_all().area)
    print(f"  lden bands, summed area  : {total / bbox_area * 100:.1f}% of bbox")
    print(f"  lden bands, union area   : {union / bbox_area * 100:.1f}% of bbox "
          f"(sum/union = {total / union:.4f}; 1.0 means the bands are disjoint)")
    print(f"  bbox below lowest band   : {(1 - union / bbox_area) * 100:.1f}% of bbox "
          f"(no polygon = quieter than 50 dB Lden, not missing data)")


def build_noise(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Fetch the Geluidskaart, store one GeoPackage per source x period."""
    ddir = mf.district_dir(data_root, district.city, district.name)
    instant = mf.utc_now_iso()
    print(f"[noise] fetching {noise_layer.TABLE} (citywide, clipped locally) "
          f"for {district.label}")
    gdf, stats = noise_layer.fetch(
        district, raw_dir=ddir / "raw", instant=instant
    )
    print(f"[noise] {stats['features_citywide']} citywide -> "
          f"{stats['features_in_bbox']} in bbox, {stats['seconds']}s")

    _report_noise(gdf, district)

    # One file per source x period rather than one fat table: the four sources
    # are different physical phenomena measured in different units (see the
    # layer docstring on industry dB(A)), and a verifier that asks "how loud is
    # road traffic here at night" should not have to remember a filter.
    for (slug, periode), sub in gdf.groupby(["bron_slug", "periode"]):
        name = f"noise_{slug}_{periode}"
        out = ddir / f"{name}.gpkg"
        sub.reset_index(drop=True).to_file(out, layer="noise", driver="GPKG")
        entry = mf.layer_entry(
            layer=name,
            source_name=noise_layer.SOURCE_NAME,
            source_url=noise_layer.ITEMS_URL,
            licence=noise_layer.LICENCE,
            vintage=noise_layer.VINTAGE,
            crs=CRS_RD,
            bbox=district.bbox_rd,
            bbox_crs=CRS_RD,
            feature_count=len(sub),
            file_path=out,
            district_dir=ddir,
            extra={
                "bbox_wgs84": list(district.bbox_wgs84),
                "representation": "vector isophone band polygons (not a raster)",
                "source_kind": sub["bron_label"].iloc[0],
                "period": periode,
                "unit": sub["unit"].iloc[0],
                "band_counts": {
                    str(k): int(v) for k, v in sub["db_label"].value_counts().items()
                },
                "band_area_m2_in_bbox": {
                    str(k): round(float(v), 1)
                    for k, v in gdf.clip(_shp_box(*district.bbox_rd))
                    .pipe(lambda g: g[(g["bron_slug"] == slug) & (g["periode"] == periode)])
                    .groupby("db_label")
                    .geometry.apply(lambda s: s.area.sum())
                    .items()
                },
                "lowest_band_label_inferred": [
                    noise_layer.DB_CLASSES[periode][i][0]
                    for i in noise_layer.INFERRED_BAND_INDICES[periode]
                    if (sub["legenda"] == i).any()
                ],
                "viewer_url": noise_layer.VIEWER_URL,
                "dataset_url": noise_layer.DATASET_URL,
                "owner": noise_layer.OWNER,
                "note": (
                    "Road noise includes trams (city's own Toelichting); rail is "
                    "heavy rail only; the metro is underground here and absent."
                ),
                "fetch_stats": {
                    k: stats[k]
                    for k in ("features_citywide", "features_in_bbox",
                              "features_without_geometry", "download_bytes",
                              "seconds", "extract_instant", "raw_file")
                },
            },
        )
        print(f"[manifest] {mf.update_manifest(ddir, entry)}  ({name}, n={len(sub)})")

    buildings_path = ddir / "buildings_3dbag.gpkg"
    context = gpd.read_file(buildings_path) if buildings_path.exists() else None
    if context is None:
        print("[render] note: buildings absent - noise overlay drawn without the mask")

    lden = gdf[gdf["periode"] == "lden"]
    overlays = ddir / "overlays"
    render_noise(
        lden,
        overlays / "noise_lden.png",
        title=(
            f"{district.label} - strategic noise map, Lden (24 h) - "
            f"{', '.join(sorted(set(lden['bron_label'])))}\n"
            f"{len(lden)} isophone polygons | Geluidskaart Amsterdam "
            f"{noise_layer.VINTAGE} | EPSG:28992 | buildings masked in grey"
        ),
        bbox=district.bbox_rd,
        band_order=noise_layer.band_order("lden"),
        palette=noise_layer.BAND_COLORS,
        buildings=context,
        legend_ncol=4,
    )
    print(f"[render] {overlays}/noise_lden.png")


def _elevation_cross_check(
    ndsm, transform, buildings: gpd.GeoDataFrame, *, n: int = 200, seed: int = 42
) -> dict[str, object]:
    """Correlate AHN nDSM at building centroids with 3DBAG ``height_m``.

    Two independent products -- a 2023 LiDAR surface model and a reconstruction
    published by TU Delft -- are asked the same question at the same 200 points.
    This is a *validation*, not a correction: nothing is written back, nothing
    is dropped from either layer on the strength of it.

    The known ways this comparison is unfair to both sides are measured rather
    than argued away: a centroid can fall outside its own (L-shaped or
    horseshoe) footprint and then samples a courtyard, and 3DBAG's ``height_m``
    is a 70th-percentile roof height while a single AHN5 DSM cell is the
    *highest* return in 0.25 m^2, so a chimney or a roof edge reads high. The
    nDSM itself rests on an interpolated ground surface under buildings (see
    ``elevation.ground_surface``), which adds the fill's error on top.

    The comparison is also not fully independent in one direction, and saying
    so is the honest framing: 3DBAG reconstructs its heights from AHN point
    clouds in the first place. What this checks is therefore the agreement
    between a *raster* product and a *reconstruction* built from a (different,
    older) generation of the same survey programme -- a consistency check on
    the pipeline and the CRS handling, not an independent ground truth.
    """
    import numpy as np

    sel = buildings.dropna(subset=["height_m"])
    take = min(n, len(sel))
    sel = sel.sample(take, random_state=seed)
    pts = sel.geometry.centroid
    outside = int((~sel.geometry.contains(pts)).sum())

    vals = elevation_layer.sample_points(ndsm, transform, pts.x.values, pts.y.values)
    ok = ~np.ma.getmaskarray(vals)
    a = np.asarray(vals[ok], dtype="float64")
    b = sel["height_m"].to_numpy(dtype="float64")[ok]
    if len(a) < 3:
        return {
            "sample_size_requested": n,
            "sample_size_used": int(len(a)),
            "seed": seed,
            "centroids_outside_own_footprint": outside,
            "nodata_or_out_of_grid_samples": int((~ok).sum()),
            "error": "too few usable samples to correlate",
        }
    diff = a - b
    r = float(np.corrcoef(a, b)[0, 1])
    return {
        "sample_size_requested": n,
        "sample_size_used": int(len(a)),
        "seed": seed,
        "centroids_outside_own_footprint": outside,
        "nodata_or_out_of_grid_samples": int((~ok).sum()),
        "pearson_r": round(r, 4),
        "median_abs_diff_m": round(float(np.median(np.abs(diff))), 3),
        "median_signed_diff_m_ndsm_minus_3dbag": round(float(np.median(diff)), 3),
        "p05_signed_diff_m": round(float(np.percentile(diff, 5)), 3),
        "p95_signed_diff_m": round(float(np.percentile(diff, 95)), 3),
    }


def build_elevation(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Clip AHN DTM/DSM to the district, render, and cross-check against 3DBAG."""
    import numpy as np

    ddir = mf.district_dir(data_root, district.city, district.name)
    instant = mf.utc_now_iso()
    print(f"[elevation] fetching AHN 0.5 m DTM/DSM for {district.label}")
    paths, stats = elevation_layer.fetch(
        district, out_dir=ddir, raw_dir=ddir / "raw", instant=instant
    )
    version = stats["ahn_version"]
    print(f"[elevation] {version} kaartblad(en) {stats['kaartbladen']} "
          f"in {stats['seconds']}s")

    dtm, transform, _ = elevation_layer.read_masked(paths["dtm"])
    dsm, _, _ = elevation_layer.read_masked(paths["dsm"])
    # DSM - DTM straight up is masked wherever the DTM is: under every building.
    # See elevation_layer.ground_surface for why the holes are interpolated and
    # what that costs. Both versions are reported; only the filled one is used.
    ndsm_raw = dsm - dtm
    ground, fill_info = elevation_layer.ground_surface(dtm)
    ndsm = dsm - ground  # masked only where the DSM itself is nodata (water)

    def pct(a, q):
        return float(np.percentile(a.compressed(), q))

    d_p5, d_p50, d_p95 = pct(dtm, 5), pct(dtm, 50), pct(dtm, 95)
    print(f"\n  grid                     : {stats['products']['dtm']['shape_px'][1]} x "
          f"{stats['products']['dtm']['shape_px'][0]} px @ {elevation_layer.PIXEL_M} m")
    print(f"  DTM (m NAP)              : min {dtm.min():6.2f}  p5 {d_p5:6.2f}  "
          f"p50 {d_p50:6.2f}  p95 {d_p95:6.2f}  max {dtm.max():7.2f}")
    print(f"  DTM nodata fraction      : {stats['products']['dtm']['nodata_fraction']:.4f}")
    print(f"  DSM (m NAP)              : min {dsm.min():6.2f}  p50 {pct(dsm, 50):6.2f}  "
          f"p95 {pct(dsm, 95):6.2f}  max {dsm.max():7.2f}")
    print(f"  DSM nodata fraction      : {stats['products']['dsm']['nodata_fraction']:.4f}")
    print(f"  nDSM = DSM-DTM (m)       : min {ndsm.min():6.2f}  p50 {pct(ndsm, 50):6.2f}  "
          f"p95 {pct(ndsm, 95):6.2f}  max {ndsm.max():7.2f}")
    print(f"  nDSM masked fraction     : {float(np.ma.getmaskarray(ndsm).mean()):.4f}"
          f"   (= DSM nodata, i.e. water)")
    print(f"  nDSM cells below -0.5 m  : {int((ndsm < -0.5).sum())}   "
          f"(DSM under ground: kept as-is, not clamped)")
    print(f"  ground fill              : {fill_info['dtm_nodata_fraction_before_fill']:.4f} "
          f"of DTM cells had no ground return (buildings, water) and were "
          f"interpolated; {fill_info['cells_still_unfilled']} still unfilled")
    print(f"  nDSM without the fill    : masked fraction "
          f"{float(np.ma.getmaskarray(ndsm_raw).mean()):.4f} -- blank on every "
          f"building, which is why the fill exists")

    ndsm_stats = {
        "min": round(float(ndsm.min()), 3), "p50": round(pct(ndsm, 50), 3),
        "p95": round(pct(ndsm, 95), 3), "max": round(float(ndsm.max()), 3),
        "masked_fraction": round(float(np.ma.getmaskarray(ndsm).mean()), 6),
        "cells_below_minus_half_m": int((ndsm < -0.5).sum()),
        "ground_reference": "DTM with nodata interpolated; see elevation.ground_surface",
        "ground_fill": fill_info,
        "masked_fraction_without_fill": round(float(np.ma.getmaskarray(ndsm_raw).mean()), 6),
    }

    buildings_path = ddir / "buildings_3dbag.gpkg"
    buildings = gpd.read_file(buildings_path) if buildings_path.exists() else None
    cross = None
    if buildings is not None:
        cross = _elevation_cross_check(ndsm, transform, buildings)
        print("\n  cross-check vs 3DBAG height_m (200 random buildings, nDSM at centroid):")
        for k, v in cross.items():
            print(f"       {k:<42} {v}")
    else:
        print("[elevation] note: buildings absent - cross-check skipped")

    for key, fname, product in (("dtm", "dtm_05m.tif", "DTM (maaiveld, ground)"),
                                ("dsm", "dsm_05m.tif", "DSM (surface)")):
        p = stats["products"][key]
        entry = mf.layer_entry(
            layer=f"elevation_{key}",
            source_name=f"{elevation_layer.SOURCE_NAME} - {version} {product}",
            source_url=p["tiles"][0]["url"],
            licence=elevation_layer.LICENCE,
            vintage=(
                f"{version}; national acquisition 2023-2024, tile file name prefix "
                f"{stats['capture_year_from_filename'] or 'n/a'}"
            ),
            crs=CRS_RD,
            bbox=tuple(stats["bbox_snapped_rd"]),
            bbox_crs=CRS_RD,
            # A raster has no features; the valid-cell count is the closest
            # honest analogue and is what the manifest's count column means here.
            feature_count=p["valid_cells"],
            file_path=ddir / fname,
            district_dir=ddir,
            extra={
                "bbox_wgs84": list(district.bbox_wgs84),
                "representation": "raster GeoTIFF, float32, EPSG:28992, tiled+deflate",
                "ahn_version": version,
                "kaartbladen": stats["kaartbladen"],
                "pixel_m": stats["pixel_m"],
                "shape_px": p["shape_px"],
                "nodata": p["nodata"],
                "nodata_fraction": round(p["nodata_fraction"], 6),
                "vertical_datum": stats["vertical_datum"],
                "requested_bbox_rd": [round(v, 3) for v in district.bbox_rd],
                "source_tiles_http": p["http"],
                "source_tiles_not_archived": (
                    "windowed /vsicurl read; 213 MB + 257 MB national tiles are not "
                    "stored, HTTP Content-Length/Last-Modified/ETag recorded instead"
                ),
                "bladwijzer_file": stats["bladwijzer_file"],
                "owner": elevation_layer.OWNER,
                "fetch_stats": {"seconds": stats["seconds"], "extract_instant": instant,
                                "tiles": p["tiles"]},
                **({"ndsm_stats": ndsm_stats,
                    "cross_check_vs_3dbag": cross} if key == "dsm" else {}),
            },
        )
        print(f"[manifest] {mf.update_manifest(ddir, entry)}  (elevation_{key})")

    overlays = ddir / "overlays"
    shade = elevation_layer.hillshade(dtm, vert_exag=HILLSHADE_VERT_EXAG)
    render_raster(
        dtm,
        overlays / "elevation_dtm.png",
        title=(
            f"{district.label} - {version} DTM 0.5 m (maaiveld, m above NAP)\n"
            f"p5 {d_p5:.2f} | p50 {d_p50:.2f} | p95 {d_p95:.2f} m NAP - colour clipped to "
            f"p5-p95 because the district spans only {d_p95 - d_p5:.2f} m\n"
            f"hillshade at {HILLSHADE_VERT_EXAG}x vertical exaggeration | EPSG:28992"
        ),
        cbar_label=f"DTM height (m NAP), clipped to p5-p95 = {d_p5:.2f}-{d_p95:.2f}",
        bbox=tuple(stats["bbox_snapped_rd"]),
        vmin=d_p5,
        vmax=d_p95,
        cmap="cividis",
        hillshade=shade,
    )
    print(f"[render] {overlays}/elevation_dtm.png")

    n_p99 = pct(ndsm, 99)
    render_raster(
        ndsm,
        overlays / "elevation_dsm_minus_dtm.png",
        title=(
            f"{district.label} - nDSM = {version} DSM - DTM (height above ground, m)\n"
            f"p50 {pct(ndsm, 50):.2f} | p95 {pct(ndsm, 95):.2f} | max {float(ndsm.max()):.1f} m "
            f"- colour clipped to 0-{n_p99:.1f} m (p99)\n"
            f"ground = DTM with its "
            f"{fill_info['dtm_nodata_fraction_before_fill'] * 100:.0f}% nodata "
            f"(under buildings) interpolated | "
            f"compare with buildings_height.png: same extent, independent source"
        ),
        cbar_label=f"height above ground (m), clipped 0-{n_p99:.1f}",
        bbox=tuple(stats["bbox_snapped_rd"]),
        vmin=0.0,
        vmax=n_p99,
        cmap="magma",
        outlines=buildings,
        outline_label="3DBAG footprints (independent source)",
        outline_color="#66ffe0",   # black outlines vanish into a magma raster
        scalebar_color="white",
    )
    print(f"[render] {overlays}/elevation_dsm_minus_dtm.png")

#: The one place a layer is registered: CLI name -> builder.

#: Radius for the OSM<->Overture name-match probe. 30 m is a deliberate
#: over-estimate of a POI positional disagreement (OSM often puts the node on
#: the entrance, Overture on a geocoded address centroid, and De Pijp blocks are
#: ~50 m deep) -- it is a measure of *overlap*, not a conflation rule.
POI_MATCH_RADIUS_M = 30.0

#: The four tag values the "is time-aware question answering viable" question
#: hangs on. Named here because the survey flags opening-hours coverage as the
#: single unmeasured blocker for time-aware questions.
POI_EATING_TAGS = ("amenity=cafe", "amenity=restaurant", "amenity=bar", "amenity=pub")

#: Bin edges for the Overture confidence histogram.
CONFIDENCE_BINS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _poi_name_matches(
    osm: gpd.GeoDataFrame, overture: gpd.GeoDataFrame, radius_m: float
) -> dict[str, object]:
    """Case-insensitive exact name matches between the two sources within ``radius_m``.

    Buffer-and-intersect, not nearest-neighbour: two POIs sharing a name are not
    necessarily each other's nearest neighbour (a chain with two branches on the
    same street would break that), and the question asked is "does the other
    source know this place", which is an *any-within-radius* question.

    This is a lower bound on real overlap and is meant to be: it sees only POIs
    both sources named, and only where the strings agree exactly after
    case-folding. "Albert Heijn" vs "Albert Heijn to go" does not match.
    """
    o = osm[osm["name"].notna()].copy()
    o["nkey_osm"] = o["name"].astype(str).str.strip().str.casefold()
    o = o[o["nkey_osm"] != ""]
    v = overture[overture["name_primary"].notna()].copy()
    v["nkey_ovt"] = v["name_primary"].astype(str).str.strip().str.casefold()
    v = v[v["nkey_ovt"] != ""]

    out: dict[str, object] = {
        "radius_m": radius_m,
        "osm_named": len(o),
        "overture_named": len(v),
        "osm_unnamed": int(len(osm) - len(o)),
        "overture_unnamed": int(len(overture) - len(v)),
    }
    if not len(o) or not len(v):
        return out

    o_pts = o.geometry
    o_buf = o[["nkey_osm"]].copy()
    o_buf = gpd.GeoDataFrame(o_buf, geometry=o_pts.buffer(radius_m), crs=o.crs)
    pairs = gpd.sjoin(o_buf, v[["nkey_ovt", "geometry"]], how="inner", predicate="intersects")
    hits = pairs[pairs["nkey_osm"] == pairs["nkey_ovt"]]

    dists = [
        o_pts.loc[li].distance(v.geometry.loc[ri])
        for li, ri in zip(hits.index, hits["index_right"])
    ]
    out.update({
        "pairs": len(hits),
        "osm_matched": int(hits.index.nunique()),
        "overture_matched": int(hits["index_right"].nunique()),
        "osm_matched_pct": round(100 * hits.index.nunique() / len(o), 1),
        "overture_matched_pct": round(100 * hits["index_right"].nunique() / len(v), 1),
        "median_pair_distance_m": round(float(median(dists)), 2) if dists else None,
        "candidate_pairs_in_radius": len(pairs),
    })
    return out


def _report_pois(
    osm: gpd.GeoDataFrame,
    overture: gpd.GeoDataFrame,
    osm_stats: dict,
    ovt_stats: dict,
    district: District,
) -> dict[str, object]:
    """Print the two-source comparison. Reports only -- nothing is merged."""
    area = district.area_km2_rd

    print(f"\n  -- counts --------------------------------------------------")
    print(f"  OSM POIs                 : {len(osm):5d}   ({len(osm) / area:6.0f} / km^2)")
    print(f"  Overture places          : {len(overture):5d}   "
          f"({len(overture) / area:6.0f} / km^2)")
    print(f"  OSM nodes / ways         : {osm_stats['nodes']} / {osm_stats['ways_centroid']}")
    print(f"  OSM timestamp_osm_base   : {osm_stats['timestamp_osm_base']}")
    print(f"  Overture release         : {ovt_stats['release']}")
    print("  OSM per primary key      :", dict(osm["primary_key"].value_counts(dropna=False)))
    print("  OSM per coarse class     :", dict(osm["poi_class"].value_counts()))
    print("  Overture source datasets :",
          dict(overture["source_datasets"].value_counts(dropna=False).head(6)))

    print(f"\n  -- top 15 primary categories -------------------------------")
    top_osm = osm["primary_tag"].value_counts().head(15)
    top_ovt = overture["category_primary"].value_counts(dropna=False).head(15)
    print(f"    {'OSM primary tag':<34}{'n':>6}   {'Overture categories.primary':<34}{'n':>6}")
    for i in range(15):
        lk, lv = (top_osm.index[i], top_osm.iloc[i]) if i < len(top_osm) else ("", "")
        rk, rv = (top_ovt.index[i], top_ovt.iloc[i]) if i < len(top_ovt) else ("", "")
        print(f"    {str(lk):<34}{str(lv):>6}   {str(rk):<34}{str(rv):>6}")

    print(f"\n  -- OSM opening_hours coverage ------------------------------")
    have = osm["opening_hours"].notna() & (osm["opening_hours"].astype(str).str.strip() != "")
    print(f"  all OSM POIs             : {int(have.sum()):5d} / {len(osm):5d} = "
          f"{100 * have.mean():5.1f}%")
    eat = osm[osm["primary_tag"].isin(POI_EATING_TAGS)]
    eat_have = eat["opening_hours"].notna() & (eat["opening_hours"].astype(str).str.strip() != "")
    print(f"  cafe|restaurant|bar|pub  : {int(eat_have.sum()):5d} / {len(eat):5d} = "
          f"{100 * eat_have.mean() if len(eat) else float('nan'):5.1f}%")
    for tag in POI_EATING_TAGS:
        sub = osm[osm["primary_tag"] == tag]
        if not len(sub):
            continue
        h = sub["opening_hours"].notna() & (sub["opening_hours"].astype(str).str.strip() != "")
        print(f"      {tag:<21}: {int(h.sum()):5d} / {len(sub):5d} = {100 * h.mean():5.1f}%")
    for col in ("cuisine", "outdoor_seating", "website", "phone", "wheelchair",
                "addr:street", "addr:housenumber"):
        c = osm[col].notna().sum()
        print(f"  {col:<24} : {int(c):5d} / {len(osm):5d} = {100 * c / len(osm):5.1f}%")
    print(f"  website/phone present only under contact:* : "
          f"{osm_stats['contact_namespace_only']}  (NOT merged into the columns)")

    print(f"\n  -- Overture confidence -------------------------------------")
    conf = overture["confidence"].dropna()
    print(f"  non-null confidence      : {len(conf)} / {len(overture)}")
    if len(conf):
        qs = conf.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        print("  quantiles                : " +
              "  ".join(f"p{int(q * 100)}={qs[q]:.3f}" for q in qs.index))
        print(f"  min / mean / max         : {conf.min():.3f} / {conf.mean():.3f} / "
              f"{conf.max():.3f}")
        for lo, hi in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
            n = int(((conf >= lo) & (conf < hi)).sum())
            print(f"    [{lo:.1f}, {hi:.1f})            : {n:5d}  "
                  f"({100 * n / len(conf):5.1f}%)  {'#' * int(60 * n / len(conf))}")
        n1 = int((conf >= 1.0).sum())
        print(f"    [1.0, 1.0]             : {n1:5d}  ({100 * n1 / len(conf):5.1f}%)")

    print(f"\n  -- name overlap (report only, nothing merged) --------------")
    m = _poi_name_matches(osm, overture, POI_MATCH_RADIUS_M)
    print(f"  named OSM / Overture     : {m['osm_named']} / {m['overture_named']}  "
          f"(unnamed: {m['osm_unnamed']} / {m['overture_unnamed']})")
    if "pairs" in m:
        print(f"  matching pairs <= {m['radius_m']:.0f} m  : {m['pairs']}  "
              f"(of {m['candidate_pairs_in_radius']} candidate pairs in radius)")
        print(f"  OSM named with a match   : {m['osm_matched']} = {m['osm_matched_pct']}%")
        print(f"  Overture named w/ match  : {m['overture_matched']} = "
              f"{m['overture_matched_pct']}%")
        print(f"  median pair distance     : {m['median_pair_distance_m']} m")
    print("  NOTE: exact case-folded string equality -- a lower bound on real overlap.")
    return m


def build_pois(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Fetch OSM (Overpass) and Overture places, store two GeoPackages, compare, render."""
    ddir = mf.district_dir(data_root, district.city, district.name)
    raw = ddir / "raw"
    instant = mf.utc_now_iso()

    print(f"[pois] OSM via Overpass for {district.label} "
          f"bbox_wgs84={district.bbox_wgs84} at {instant}")
    osm, osm_stats = pois_layer.fetch_osm(district, raw_dir=raw, instant=instant)
    print(f"[pois] {len(osm)} OSM POIs in {osm_stats['seconds']}s")

    print(f"[pois] Overture places for {district.label}")
    overture, ovt_stats = pois_layer.fetch_overture(district, raw_dir=raw, instant=instant)
    print(f"[pois] {len(overture)} Overture places in {ovt_stats['seconds']}s")

    osm_out = ddir / "pois_osm.gpkg"
    ovt_out = ddir / "pois_overture.gpkg"
    osm.to_file(osm_out, layer="pois_osm", driver="GPKG")
    overture.to_file(ovt_out, layer="pois_overture", driver="GPKG")

    match = _report_pois(osm, overture, osm_stats, ovt_stats, district)

    common_extra = {
        "bbox_wgs84": list(district.bbox_wgs84),
        "extract_instant": instant,
        "merge_policy": (
            "NONE - the two POI files are stored side by side and never merged; "
            "conflation is a downstream decision, see worldsnap/layers/pois.py"
        ),
        "cross_source_name_match": match,
    }

    mf.update_manifest(ddir, mf.layer_entry(
        layer="pois_osm",
        source_name=pois_layer.OSM_SOURCE_NAME,
        source_url=osm_stats["endpoint"] or pois_layer.OVERPASS_URL,
        licence=pois_layer.OSM_LICENCE,
        # The replication instant of the OSM database the answer was computed
        # from -- stated by the server, not inferred from a file date.
        vintage=f"OSM database as of {osm_stats['timestamp_osm_base']}",
        crs=CRS_RD,
        bbox=district.bbox_rd,
        bbox_crs=CRS_RD,
        feature_count=len(osm),
        file_path=osm_out,
        district_dir=ddir,
        extra={
            **common_extra,
            "attribution_required": "(c) OpenStreetMap contributors, " +
                                    pois_layer.OSM_COPYRIGHT_URL,
            "one_off_snapshot": (
                "Overpass was called ONCE to produce raw/; it is not a runtime "
                "dependency and must never be called per question or per step"
            ),
            "overpass_query": osm_stats["query"],
            "overpass_generator": osm_stats["generator"],
            "selection": list(pois_layer.POI_SELECTORS),
            "geometry_note": (
                "ways are stored as the Overpass 'center' (bounding-box midpoint, "
                "not polygon centroid); relations are excluded"
            ),
            "derived_columns": {
                "primary_key/primary_value/primary_tag": (
                    "first key present in " + ",".join(pois_layer.PRIMARY_KEY_ORDER)
                ),
                "poi_class": "coarse 7-way class for the overlay legend, see module docstring",
            },
            "contact_namespace_only": osm_stats["contact_namespace_only"],
            "raw_archive": [{
                **osm_stats["raw_file"],
                "dir": raw.relative_to(ddir).as_posix(),
                "downloaded_utc": instant,
            }],
            "fetch_stats": {
                k: osm_stats[k] for k in (
                    "elements_returned", "pois_raw", "duplicates_dropped",
                    "elements_without_coordinate", "pois_without_primary_tag",
                    "nodes", "ways_centroid", "download_bytes", "seconds",
                )
            },
        },
    ))
    mf.update_manifest(ddir, mf.layer_entry(
        layer="pois_overture",
        source_name=pois_layer.OVERTURE_SOURCE_NAME,
        source_url=pois_layer.OVERTURE_S3_URL.format(release=ovt_stats["release"]),
        licence=pois_layer.OVERTURE_LICENCE,
        # Overture releases are immutable: this string plus the bbox is a
        # complete and permanently reproducible download recipe.
        vintage=f"Overture release {ovt_stats['release']}",
        crs=CRS_RD,
        bbox=district.bbox_rd,
        bbox_crs=CRS_RD,
        feature_count=len(overture),
        file_path=ovt_out,
        district_dir=ddir,
        extra={
            **common_extra,
            "release": ovt_stats["release"],
            "docs_url": pois_layer.OVERTURE_DOCS_URL,
            "download_command": ovt_stats["command"],
            "nested_fields_note": (
                "categories.alternate, addresses, websites, phones and sources are "
                "stored as JSON strings; GPKG has no list/struct type and the "
                "nesting is the provider's schema"
            ),
            "raw_archive": [{
                **ovt_stats["raw_file"],
                "dir": raw.relative_to(ddir).as_posix(),
                "downloaded_utc": instant,
            }],
            "fetch_stats": {
                k: ovt_stats[k] for k in (
                    "places_returned", "places_without_geometry",
                    "confidence_min", "confidence_max", "seconds",
                )
            },
        },
    ))
    print(f"[manifest] {ddir / mf.MANIFEST_NAME}")

    buildings_path = ddir / "buildings_3dbag.gpkg"
    streets_path = ddir / "streets_bgt.gpkg"
    context = gpd.read_file(buildings_path) if buildings_path.exists() else None
    streets = gpd.read_file(streets_path) if streets_path.exists() else None
    if context is None or streets is None:
        print("[render] note: buildings/streets absent - rendering with partial context")

    overlays = ddir / "overlays"
    render_pois(
        osm, overture, overlays / "pois.png",
        column="poi_class",
        title=(
            f"{district.label} - points of interest, two sources, not merged\n"
            f"OSM {len(osm)} (dots, colour = coarse class; ODbL, database "
            f"{osm_stats['timestamp_osm_base']}) vs "
            f"Overture places {len(overture)} (hollow rings; CDLA-Permissive-2.0, "
            f"release {ovt_stats['release']})\n"
            f"{match.get('osm_matched_pct', float('nan'))}% of named OSM POIs have an "
            f"exact case-folded name match within {POI_MATCH_RADIUS_M:.0f} m | EPSG:28992"
        ),
        bbox=district.bbox_rd,
        palette=pois_layer.POI_CLASS_COLORS,
        category_order=list(pois_layer.POI_CLASSES),
        buildings=context,
        streets=streets,
        legend_ncol=5,
    )
    print(f"[render] {overlays}/pois.png")


def build_sunshine(district: District, *, data_root: Path = DATA_ROOT) -> None:
    """Derive sunlit-hours and shadow rasters from 3DBAG heights; cross-check vs AHN.

    The first layer in the snapshot that is *computed* rather than downloaded, so
    its manifest entries name this code and its parameters as the source instead
    of a URL, and record the hash of the 3DBAG file the heights came from. Every
    definitional choice lives in ``layers/sunshine.py``'s docstring; the entry
    repeats the ones a reader of the manifest alone would need.
    """
    import numpy as np

    ddir = mf.district_dir(data_root, district.city, district.name)
    grid = sunshine_layer.district_grid(district)
    lon = (district.min_lon + district.max_lon) / 2
    lat = (district.min_lat + district.max_lat) / 2
    buildings = gpd.read_file(ddir / "buildings_3dbag.gpkg")
    height, mask, hstats = sunshine_layer.height_rasters(buildings, grid)
    print(f"[sunshine] {grid['width_m']} x {grid['height_m']} m at "
          f"{sunshine_layer.RES_M} m, sun at lat {lat:.4f} lon {lon:.4f}")
    print(f"  footprints               : {hstats['buildings']} "
          f"({hstats['without_height']} without a height -> not ground, no shadow; "
          f"{hstats['non_positive_height']} non-positive, kept as-is)")
    print(f"  building pixel fraction  : {hstats['footprint_pixel_fraction']:.4f}   "
          f"median height {hstats['height_p50_m']} m, max {hstats['height_max_m']} m")

    t0 = time.time()
    hours, shadow, timing = sunshine_layer.compute(height, mask, lat, lon)
    print(f"  compute                  : {timing['device']}, "
          + ", ".join(f"{k.split('_')[-1]} {v} s" for k, v in timing.items()
                      if k.startswith("seconds_"))
          + f", total {time.time() - t0:.1f} s")
    dist = {k: sunshine_layer.percentiles(v) for k, v in hours.items()}
    for key, d in dist.items():
        print(f"  sunlit hours {key}        : p10 {d['p10']}  p50 {d['p50']}  p90 {d['p90']}  "
              f"(mean {d['mean']}, min {d['min']}, max {d['max']}, n={d['ground_cells']})")

    dtm_p, dsm_p = ddir / "dtm_05m.tif", ddir / "dsm_05m.tif"
    cross = None
    if dtm_p.exists() and dsm_p.exists():
        ndsm, ndsm_info = sunshine_layer.ahn_ndsm_1m(dtm_p, dsm_p, grid)
        cross = {**sunshine_layer.ahn_disagreement(ndsm, shadow, mask, lat, lon), **ndsm_info}
        print("\n  built-in check (C4_PLAN 4c): 21 Jun 15:00 UTC shadow, 3DBAG boxes vs "
              "AHN5 nDSM (buildings AND trees) -- reported, never merged:")
        for k, v in cross.items():
            print(f"       {k:<38} {v}")
    else:
        print("[sunshine] note: AHN rasters absent - cross-check skipped")

    paths = {}
    for key, arr in hours.items():
        paths[key] = sunshine_layer.write_raster(
            ddir / sunshine_layer.HOURS_NAME.format(key=key), arr, grid, "float32", float("nan"))
    paths["shadow"] = sunshine_layer.write_raster(
        ddir / sunshine_layer.SHADOW_NAME, shadow.astype("uint8"), grid, "uint8", 255)

    bag_sha = mf.sha256_file(ddir / "buildings_3dbag.gpkg")
    common = {
        "bbox_wgs84": list(district.bbox_wgs84),
        "representation": "raster GeoTIFF, EPSG:28992, 1 m, tiled+deflate; frame identical "
                          "to district_labels_1m in every cropset",
        "derived": True,
        "derived_from": {"layer": "buildings", "file": "buildings_3dbag.gpkg",
                         "sha256": bag_sha, "height_field": sunshine_layer.HEIGHT_FIELD},
        "method": "2.5D ray march over rasterised 3DBAG heights; worldsnap/sun.py",
        "formula_source": sunshine_layer.FORMULA_SOURCE,
        "buildings_only": "trees excluded on purpose; see cross_check_vs_ahn_ndsm",
        "sun_position_at": {"lat": round(lat, 5), "lon": round(lon, 5),
                            "note": "one position for the whole district (~0.02 deg across it)"},
        "max_shadow_distance_m": sunshine_layer.MAX_DIST_M,
        "shape_px": [grid["height_m"], grid["width_m"]],
        "pixel_m": sunshine_layer.RES_M,
        "height_raster_stats": hstats,
        "nodata_meaning": "inside a building footprint: not ground, neither sunlit nor shadowed",
    }
    for key, path in paths.items():
        is_hours = key in hours
        date_key = key if is_hours else sunshine_layer.SHADOW_EXAMPLE[0]
        entry = mf.layer_entry(
            layer=f"sunshine_{'hours_' if is_hours else 'shadow_'}{date_key}",
            source_name="DERIVED by worldsnap.layers.sunshine from 3DBAG building heights "
                        "(no provider publishes this layer)",
            source_url=buildings_layer.API_ROOT,
            licence=f"derived work of 3DBAG ({buildings_layer.LICENCE})",
            vintage=f"computed {mf.utc_now_iso()} for {sunshine_layer.DATES[date_key]} "
                    f"from 3DBAG as snapshotted in this manifest",
            crs=CRS_RD, bbox=grid["bbox"], bbox_crs=CRS_RD,
            feature_count=int((~mask).sum()),  # ground cells: the closest honest analogue
            file_path=path, district_dir=ddir,
            extra={**common,
                   "date": str(sunshine_layer.DATES[date_key]),
                   **({"quantity": "hours of direct sun at ground level over the UTC day",
                       "step_minutes": sunshine_layer.STEP_MINUTES,
                       "distribution_over_ground": dist[key],
                       "nodata": "NaN"}
                      if is_hours else
                      {"quantity": "1 = ground pixel shadowed by a building at the instant",
                       "instant_utc": timing["example_sun"]["utc"],
                       "sun": timing["example_sun"], "nodata": 255,
                       "cross_check_vs_ahn_ndsm": cross})},
        )
        print(f"[manifest] {mf.update_manifest(ddir, entry)}  ({entry['layer']})")

    overlays = ddir / "overlays"
    d = dist["0621"]
    render_raster(
        np.ma.masked_invalid(hours["0621"]),
        overlays / "sunlit_hours_0621.png",
        title=(
            f"{district.label} - DERIVED: hours of direct sun at ground level, "
            f"21 June {sunshine_layer.YEAR} (UTC day)\n"
            f"p10 {d['p10']} | p50 {d['p50']} | p90 {d['p90']} h - 3DBAG heights "
            f"({sunshine_layer.HEIGHT_FIELD.split(' ')[0]}), buildings only, no trees; "
            f"{sunshine_layer.STEP_MINUTES}-min steps, {sunshine_layer.MAX_DIST_M:.0f} m reach\n"
            f"grey = building footprint (not ground, nodata) | 1 m | EPSG:28992"
        ),
        cbar_label="sunlit hours on 21 June (ground level, direct beam only)",
        bbox=grid["bbox"], vmin=0.0, vmax=float(np.ceil(d["max"])), cmap="inferno",
        # The "hillshade" slot is a plain grey base here: 0.55 under buildings,
        # white elsewhere. With colour_alpha=1 the sun colours stay unwashed and
        # the nodata footprints read as grey instead of as the page background.
        hillshade=np.where(mask, 0.55, 1.0).astype("float32"), colour_alpha=1.0,
        outlines=buildings, outline_label="3DBAG footprints (nodata: not ground)",
        outline_color="#404040", scalebar_color="black",
    )
    print(f"[render] {overlays}/sunlit_hours_0621.png")


LAYER_BUILDERS = {
    "buildings": build_buildings,
    "sunshine": build_sunshine,
    "streets_bgt": build_streets_bgt,
    "transit_gtfs": build_transit_gtfs,
    "trees": build_trees,
    "noise": build_noise,
    "elevation": build_elevation,
    "pois": build_pois,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="worldsnap.build", description=__doc__)
    p.add_argument("--district", required=True, choices=sorted(DISTRICTS))
    p.add_argument(
        "--layers",
        default="buildings",
        help="comma-separated layer names: " + ",".join(LAYER_BUILDERS),
    )
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args(argv)

    district = DISTRICTS[args.district]
    for name in [s.strip() for s in args.layers.split(",") if s.strip()]:
        if name not in LAYER_BUILDERS:
            p.error(f"unknown layer {name!r}")
        LAYER_BUILDERS[name](district, data_root=args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
