"""Public-transport stops, routes and weekday frequency from the national GTFS.

WHY the OVapi national feed
---------------------------
The survey's headline finding for Amsterdam was that transit needs no key: the
whole of Dutch public transport (GVB tram/metro/bus, NS rail, every regional
concession) is published as ONE static GTFS zip, refreshed daily, at
``http://gtfs.ovapi.nl/nl/gtfs-nl.zip`` -- no registration, no token, no quota.
Barcelona's equivalent (TMB) is key-gated; this is the reason De Pijp is the
pilot district.

The zip is ~232 MB compressed / ~1.4 GB expanded, national scope. We keep the
**whole zip** in ``raw/`` rather than a district cut of it: the manifest's
provenance claim is about bytes we received, and a GTFS feed is only meaningful
as a complete, internally-consistent set of tables (a trip is worthless without
its calendar). Disk is cheap; a half-feed is not reproducible.

WHY frequency is *derived* here and not downloaded
--------------------------------------------------
There is no published "how often does a tram stop here" dataset. It is a
computation over ``stop_times`` x ``trips`` x the calendar, and the whole point
of this layer is that the answer becomes verifiable rather than assertable:
``departures_0719`` is a count anyone can recompute from the archived zip.

FEED QUIRKS discovered by probing (these are the reason for the code below)
---------------------------------------------------------------------------
1. **There is no ``calendar.txt``.** The feed ships ``calendar_dates.txt``
   only, i.e. every service day is an explicit dated exception
   (``exception_type=1`` = service added). This is legal GTFS but means the
   usual "monday..sunday flags plus exceptions" logic would silently produce
   ZERO active services. :func:`active_services` therefore treats a missing
   ``calendar.txt`` as an empty weekly base and resolves service purely from
   the dated exceptions -- while still implementing the combined rule, so the
   code stays correct if OVapi ever adds ``calendar.txt``.
2. ``stop_times.txt`` is 1.12 GB. It is streamed in chunks and filtered to the
   district's stop ids on the way past; it is never held in memory whole.
3. Stop ids are **not** integers: ``stops.txt`` mixes numeric quay ids with
   ``stoparea:177908`` parent rows. Everything is read as string; an integer
   dtype would mangle the join.
4. ``location_type=1`` rows (stop *areas*) carry no ``stop_times`` at all, so
   they legitimately end up with ``departures_0719 == 0``. They are kept, not
   dropped -- a zero here means "this is a grouping node", not "no service",
   and the distinction has to stay visible.
5. GTFS times may exceed 24:00:00 (``25:10:00`` = 01:10 on the next service
   day). Times are parsed to seconds-since-service-midnight, not to clock
   times, so the 07:00-19:00 window is unambiguous.
6. ``pickup_type=1`` marks a stop where no one may board -- the terminus of a
   trip. Such rows are *arrivals*, not departures. They are excluded from
   ``departures_0719`` and the difference is reported separately, because the
   naive "count all stop_times rows" number is ~2-4% higher and someone will
   eventually try to reproduce it.

WHY a 300 m buffer
------------------
A stop 50 m outside the rectangle still serves the district; a hard bbox cut
would make "nearest tram stop" wrong for every address near an edge. 300 m is
roughly the Dutch urban stop spacing (and the conventional walking-access
radius), so the buffer guarantees every point inside the bbox has its real
nearest stop present. The value is stored in the manifest, not hard-coded into
a downstream assumption.

Sentinels are kept, not cleaned
-------------------------------
Empty ``stop_code``, empty ``parent_station``, ``platform_code`` values that
are really IFF station abbreviations (``IFF:vwd``) and route_short_names that
repeat across operators are all preserved verbatim. GTFS is a merge of 38
agencies' exports and its inconsistencies are properties of the feed.
"""

from __future__ import annotations

import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, box

from ..config import CRS_RD, CRS_WGS84, District

FEED_URL = "http://gtfs.ovapi.nl/nl/gtfs-nl.zip"
FEED_INDEX_URL = "https://gtfs.ovapi.nl/nl/"
SOURCE_NAME = "GTFS Netherlands (static), OVapi / OpenOV mirror of NDOV Loket"
#: Honest statement: the distribution endpoint publishes no licence text. The
#: upstream is the Dutch national access point (NDOV / GOVI) which exists under
#: the EU MMTIS delegated regulation, i.e. the data is open by law, but no SPDX
#: identifier is asserted anywhere we could verify. Recorded as-is.
LICENCE = (
    "not stated at the distribution endpoint (gtfs.ovapi.nl); upstream is the "
    "Dutch national access point NDOV/GOVI under EU MMTIS regulation "
    "(2017/1926) - open data, no registration; no SPDX licence asserted"
)

#: Walking-access buffer around the district bbox, metres. See module docstring.
BUFFER_M = 300.0

#: Departure-counting window, seconds since service midnight (07:00-19:00).
WINDOW_START_S, WINDOW_END_S = 7 * 3600, 19 * 3600

#: GTFS ``route_type`` -> label, for the values that occur in the NL feed.
ROUTE_TYPE_LABELS: dict[int, str] = {
    0: "tram",
    1: "metro",
    2: "rail",
    3: "bus",
    4: "ferry",
    5: "cable tram",
    6: "aerial lift",
    7: "funicular",
    11: "trolleybus",
    12: "monorail",
}

#: Colours for the render. Amsterdam reading: metro is the thick fast thing,
#: tram is the dense surface network, bus fills in; rail only clips the corner.
ROUTE_TYPE_COLORS: dict[int, str] = {
    0: "#0b7a3b",   # tram
    1: "#c0392b",   # metro
    2: "#1f4e9c",   # rail
    3: "#7d5aa8",   # bus
    4: "#0d8fa8",   # ferry
}

_TO_RD = Transformer.from_crs(CRS_WGS84, CRS_RD, always_xy=True)

#: Chunk size for the streaming pass over stop_times.txt (~1.12 GB).
STOP_TIMES_CHUNK = 2_000_000


def download_feed(raw_dir: Path, *, stamp: str, verbose: bool = True) -> tuple[Path, dict[str, Any]]:
    """Download the national GTFS zip into ``raw_dir`` unless it is already there.

    Kept whole and never re-downloaded for the same ``stamp``: the feed is
    refreshed daily, so a second download mid-build would silently mix two
    vintages into one snapshot.
    """
    import requests  # local import: only the download path needs it

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"gtfs-nl_{stamp}.zip"
    if path.exists():
        if verbose:
            print(f"  [gtfs] reusing existing {path.name} ({path.stat().st_size:,} bytes)")
        return path, {"reused_existing": True}
    if verbose:
        print(f"  [gtfs] downloading {FEED_URL} -> {path.name} (~232 MB)")
    with requests.get(FEED_URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        headers = {k.lower(): v for k, v in resp.headers.items()}
        with path.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    return path, {"reused_existing": False, "http_headers": headers}


def buffered_bbox(district: District, buffer_m: float = BUFFER_M) -> tuple[float, float, float, float]:
    """District bbox in RD metres, grown by ``buffer_m`` on every side."""
    x0, y0, x1, y1 = district.bbox_rd
    return (x0 - buffer_m, y0 - buffer_m, x1 + buffer_m, y1 + buffer_m)


def _gtfs_date(s: str) -> date:
    """``YYYYMMDD`` -> date."""
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def representative_tuesday(feed_start: date, feed_end: date, today: date) -> tuple[date, str]:
    """Pick the weekday the frequency is counted on, and explain the choice.

    Rule, in words, so it can be restated in the manifest: *the first Tuesday
    strictly after the build date; if that falls before the feed's validity
    starts, move forward to the first Tuesday inside it.* Tuesday because it is
    the canonical "ordinary weekday" in transit planning -- no Monday
    post-weekend timetable variants, no Friday evening extras, and it is far
    from Dutch public holidays in any normal feed window. "Strictly after
    today" avoids counting a day that is already half over and makes the choice
    a function of the build date alone.
    """
    d = today + timedelta(days=1)
    while d.weekday() != 1:  # Monday=0, Tuesday=1
        d += timedelta(days=1)
    moved = False
    while d < feed_start:
        d += timedelta(days=7)
        moved = True
    why = (
        f"first Tuesday strictly after build date {today.isoformat()}"
        + (f", advanced into feed validity from {feed_start.isoformat()}" if moved else "")
        + f"; feed valid {feed_start.isoformat()}..{feed_end.isoformat()}"
        + ("" if d <= feed_end else "  [WARNING: chosen date is OUTSIDE feed validity]")
    )
    return d, why


def _read_table(zf: zipfile.ZipFile, name: str, **kwargs: Any) -> pd.DataFrame:
    """Read one GTFS table as all-strings (see quirk 3)."""
    with zf.open(name) as fh:
        return pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""], **kwargs)


def _chunks(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> Iterator[pd.DataFrame]:
    with zf.open(name) as fh:
        yield from pd.read_csv(
            fh, dtype=str, keep_default_na=False, na_values=[""],
            usecols=usecols, chunksize=STOP_TIMES_CHUNK,
        )


def _seconds(series: pd.Series) -> pd.Series:
    """``HH:MM:SS`` (H may exceed 23) -> seconds since service midnight."""
    parts = series.str.split(":", expand=True).astype("float64")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def active_services(
    calendar: pd.DataFrame | None, calendar_dates: pd.DataFrame | None, day: date
) -> set[str]:
    """Service ids running on ``day``, per the GTFS calendar rules.

    Implements the full rule even though this feed only exercises half of it
    (quirk 1): weekly pattern from ``calendar.txt`` restricted to its date
    range, then ``calendar_dates.txt`` exceptions applied on top --
    ``exception_type=1`` adds a service, ``=2`` removes it.
    """
    stamp = day.strftime("%Y%m%d")
    base: set[str] = set()
    if calendar is not None and len(calendar):
        weekday_col = ("monday", "tuesday", "wednesday", "thursday",
                       "friday", "saturday", "sunday")[day.weekday()]
        in_range = (calendar["start_date"] <= stamp) & (calendar["end_date"] >= stamp)
        base = set(calendar.loc[in_range & (calendar[weekday_col] == "1"), "service_id"])
    if calendar_dates is not None and len(calendar_dates):
        today_rows = calendar_dates[calendar_dates["date"] == stamp]
        base |= set(today_rows.loc[today_rows["exception_type"] == "1", "service_id"])
        base -= set(today_rows.loc[today_rows["exception_type"] == "2", "service_id"])
    return base


def _route_shape_lines(
    zf: zipfile.ZipFile,
    shape_ids: set[str],
    clip: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Stream ``shapes.txt`` (240 MB) and build one clipped line per shape id."""
    keep: list[pd.DataFrame] = []
    with zf.open("shapes.txt") as fh:
        for chunk in pd.read_csv(
            fh, dtype={"shape_id": str}, chunksize=STOP_TIMES_CHUNK,
            usecols=["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"],
        ):
            sel = chunk[chunk["shape_id"].isin(shape_ids)]
            if len(sel):
                keep.append(sel)
    if not keep:
        return {}
    pts = pd.concat(keep, ignore_index=True)
    pts = pts.sort_values(["shape_id", "shape_pt_sequence"])
    x, y = _TO_RD.transform(pts["shape_pt_lon"].to_numpy(), pts["shape_pt_lat"].to_numpy())
    pts["x"], pts["y"] = x, y
    clip_box = box(*clip)
    out: dict[str, Any] = {}
    for sid, grp in pts.groupby("shape_id", sort=False):
        if len(grp) < 2:
            continue  # a 1-point shape is not a line; recorded as a dropped shape
        geom = LineString(zip(grp["x"], grp["y"])).intersection(clip_box)
        if not geom.is_empty:
            out[sid] = geom
    return out


def fetch(
    district: District,
    *,
    zip_path: Path,
    today: date | None = None,
    buffer_m: float = BUFFER_M,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Cut stops, routes and weekday frequency for ``district`` out of the feed.

    ``zip_path`` is the archived national zip (downloaded by the builder).
    Returns ``(stops_gdf, routes_gdf, stats)``, both in EPSG:28992.
    """
    t0 = time.time()
    clip = buffered_bbox(district, buffer_m)
    stats: dict[str, Any] = {"buffer_m": buffer_m, "bbox_buffered_rd": list(clip)}

    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        stats["feed_files"] = sorted(members)

        feed_info = _read_table(zf, "feed_info.txt")
        fi = feed_info.iloc[0].to_dict()
        feed_start, feed_end = _gtfs_date(fi["feed_start_date"]), _gtfs_date(fi["feed_end_date"])
        day, day_why = representative_tuesday(feed_start, feed_end, today or date.today())
        stats["feed_info"] = fi
        stats["service_date"] = day.isoformat()
        stats["service_date_rationale"] = day_why
        if verbose:
            print(f"  feed {fi['feed_publisher_name']} v{fi['feed_version']} "
                  f"valid {feed_start}..{feed_end}; counting {day} ({day_why})")

        # --- stops in the buffered bbox -------------------------------------
        stops = _read_table(zf, "stops.txt")
        sx, sy = _TO_RD.transform(
            stops["stop_lon"].astype(float).to_numpy(), stops["stop_lat"].astype(float).to_numpy()
        )
        stops["x"], stops["y"] = sx, sy
        x0, y0, x1, y1 = clip
        inside = (stops["x"] >= x0) & (stops["x"] <= x1) & (stops["y"] >= y0) & (stops["y"] <= y1)
        stops = stops[inside].reset_index(drop=True)
        stop_ids = set(stops["stop_id"])
        stats["stops_in_feed"] = int(len(inside))
        stats["stops_in_buffered_bbox"] = len(stops)
        if verbose:
            print(f"  stops: {len(stops)} of {len(inside)} national in bbox+{buffer_m:.0f}m")

        # --- stop_times for those stops (streamed, quirk 2) -----------------
        cols = ["trip_id", "stop_id", "departure_time", "pickup_type", "stop_sequence"]
        kept: list[pd.DataFrame] = []
        n_rows = 0
        for i, chunk in enumerate(_chunks(zf, "stop_times.txt", cols), start=1):
            n_rows += len(chunk)
            sel = chunk[chunk["stop_id"].isin(stop_ids)]
            if len(sel):
                kept.append(sel)
            if verbose and i % 20 == 0:
                print(f"  [stop_times] chunk {i:3d}  rows scanned {n_rows:,}", flush=True)
        st = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=cols)
        stats["stop_times_rows_scanned"] = n_rows
        stats["stop_times_rows_at_district_stops"] = len(st)
        if verbose:
            print(f"  stop_times: {len(st):,} rows at district stops "
                  f"(of {n_rows:,} national)")

        # --- trips / calendar / routes / agency -----------------------------
        trips = _read_table(
            zf, "trips.txt",
            usecols=["route_id", "service_id", "trip_id", "shape_id", "direction_id"],
        )
        trips = trips[trips["trip_id"].isin(set(st["trip_id"]))].reset_index(drop=True)

        calendar = _read_table(zf, "calendar.txt") if "calendar.txt" in members else None
        cal_dates = _read_table(zf, "calendar_dates.txt") if "calendar_dates.txt" in members else None
        stats["has_calendar_txt"] = calendar is not None
        services = active_services(calendar, cal_dates, day)
        stats["services_active_on_date"] = len(services)

        routes = _read_table(zf, "routes.txt")
        agency = _read_table(zf, "agency.txt")

    # ---- departures per stop on the representative weekday -----------------
    st = st.merge(trips[["trip_id", "route_id", "service_id", "shape_id", "direction_id"]],
                  on="trip_id", how="left")
    st["dep_s"] = _seconds(st["departure_time"])
    in_window = (st["dep_s"] >= WINDOW_START_S) & (st["dep_s"] < WINDOW_END_S)
    on_date = st["service_id"].isin(services)
    boardable = st["pickup_type"].fillna("0") != "1"  # quirk 6

    dep_all = st[in_window & on_date]
    dep = st[in_window & on_date & boardable]
    stats["departures_window_all_rows"] = len(dep_all)
    stats["departures_window_boardable"] = len(dep)
    stats["departures_window_arrivals_only"] = len(dep_all) - len(dep)

    counts = dep.groupby("stop_id").size().rename("departures_0719")

    # Route membership is computed over the WHOLE feed, not just the chosen
    # day: a stop served only at weekends must still list its routes, while its
    # weekday frequency is honestly zero.
    rt = routes.set_index("route_id")
    st_routes = st.dropna(subset=["route_id"]).merge(
        rt[["route_short_name", "route_type"]], left_on="route_id", right_index=True, how="left"
    )
    per_stop_routes = st_routes.groupby("stop_id").agg(
        route_short_names=("route_short_name", lambda s: ",".join(sorted(set(s.dropna())))),
        route_types=("route_type", lambda s: ",".join(sorted(set(s.dropna())))),
        n_routes=("route_id", lambda s: s.nunique()),
        n_trips_feed=("trip_id", "nunique"),
    )

    stops = stops.join(counts, on="stop_id").join(per_stop_routes, on="stop_id")
    stops["departures_0719"] = stops["departures_0719"].fillna(0).astype(int)
    stops["n_routes"] = stops["n_routes"].fillna(0).astype(int)
    stops["n_trips_feed"] = stops["n_trips_feed"].fillna(0).astype(int)
    stops["feed_version"] = fi["feed_version"]
    stops["feed_start_date"] = fi["feed_start_date"]
    stops["feed_end_date"] = fi["feed_end_date"]
    stops["service_date"] = day.isoformat()

    stops_gdf = gpd.GeoDataFrame(
        stops[[
            "stop_id", "stop_name", "stop_code", "location_type", "parent_station",
            "platform_code", "zone_id", "wheelchair_boarding",
            "route_short_names", "route_types", "n_routes", "n_trips_feed",
            "departures_0719", "service_date",
            "feed_version", "feed_start_date", "feed_end_date",
        ]].rename(columns={"stop_name": "name"}),
        geometry=gpd.points_from_xy(stops["x"], stops["y"]),
        crs=CRS_RD,
    )

    # ---- routes serving the district, with geometry -------------------------
    served = sorted(set(st["route_id"].dropna()))
    # One line per route: the shape variant used by the most trips at our
    # stops. A route has many shape variants (per direction, per short-turn);
    # drawing all of them turns the map into hairball. The variant count is
    # kept as an attribute so the simplification is visible, not hidden.
    dominant = (
        st.dropna(subset=["route_id", "shape_id"])
        .groupby(["route_id", "shape_id"])["trip_id"].nunique()
        .reset_index(name="n_trips")
        .sort_values(["route_id", "n_trips"], ascending=[True, False])
    )
    variants = dominant.groupby("route_id")["shape_id"].nunique().rename("n_shape_variants")
    top = dominant.drop_duplicates("route_id").set_index("route_id")

    with zipfile.ZipFile(zip_path) as zf:
        geoms = _route_shape_lines(zf, set(top["shape_id"]), clip)
    stats["shapes_requested"] = len(set(top["shape_id"]))
    stats["shapes_with_geometry_in_bbox"] = len(geoms)

    ag = agency.set_index("agency_id")["agency_name"].to_dict()
    rows = []
    for rid in served:
        meta = rt.loc[rid] if rid in rt.index else None
        sid = top["shape_id"].get(rid)
        geom = geoms.get(sid)
        if geom is None:
            continue  # recorded via routes_without_geometry below
        rtype = None if meta is None else meta.get("route_type")
        rows.append({
            "route_id": rid,
            "route_short_name": None if meta is None else meta.get("route_short_name"),
            "route_long_name": None if meta is None else meta.get("route_long_name"),
            "route_type": None if pd.isna(rtype) else int(rtype),
            "agency_id": None if meta is None else meta.get("agency_id"),
            "agency": None if meta is None else ag.get(meta.get("agency_id")),
            "shape_id": sid,
            "n_shape_variants": int(variants.get(rid, 0)),
            "n_trips_dominant_shape": int(top["n_trips"].get(rid, 0)),
            "geometry": geom,
        })
    routes_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_RD)
    routes_gdf["route_type_label"] = routes_gdf["route_type"].map(ROUTE_TYPE_LABELS)
    stats["routes_serving_district"] = len(served)
    stats["routes_without_geometry"] = len(served) - len(routes_gdf)
    stats["seconds"] = round(time.time() - t0, 1)
    return stops_gdf, routes_gdf, stats
