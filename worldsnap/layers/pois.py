"""Points of interest from two independent providers, stored side by side.

WHY two files and no merge
---------------------------
Every other layer in this snapshot has exactly one authority (BAG for buildings,
BGT for street surfaces, Gisib for trees). POIs have none: there is no open Dutch
business register (see ``survey/raw/03_open_data_barcelona_amsterdam.md`` §2.4 --
Amsterdam has no equivalent of Barcelona's per-premises *cens d'activitats*), so
the best available ground truth is *two crowd/aggregate sources that disagree*.
Merging them here would bake one conflation heuristic into the snapshot and
destroy the evidence. The snapshot therefore stores both verbatim:

* ``pois_osm.gpkg``       -- OpenStreetMap, **ODbL**, community-surveyed, deep
  free-text tagging (opening hours, cuisine, wheelchair, terrace).
* ``pois_overture.gpkg``  -- Overture Maps ``places``, **CDLA-Permissive-2.0**,
  an aggregate dominated by Foursquare/Meta records, with a per-record
  ``confidence`` and a fixed category taxonomy but no opening hours at all.

The licence split is the reason this is worth the extra file: the OSM half is
share-alike, the Overture half is not, so a downstream tool can be built on
Overture alone if ODbL contagion ever becomes inconvenient. Reconciliation (name
matching, dedup, category crosswalk) is a *downstream* decision; ``build.py``
only *reports* the overlap.

(a) OSM via Overpass -- a one-off snapshot, NOT a runtime dependency
---------------------------------------------------------------------
The extract is pulled with **one** POST to the public Overpass instance
(https://overpass-api.de/api/interpreter) and the response is archived verbatim
as gzipped JSON under ``raw/``. Overpass is a shared volunteer service with a
fair-use policy; nothing in this pipeline may ever call it in a loop, per
training step, or per question. It is used exactly the way a Geofabrik download
would be used -- once, to produce bytes on disk -- and the GeoPackage is always
re-derivable from the archived JSON without touching the network.

A Geofabrik ``noord-holland.osm.pbf`` (~200 MB) + pyosmium would be the
determinism-maximal alternative and was the briefed fallback. It was not used:
it needs a new binary dependency (osmium) to extract ~3 k features from a
million-object file, and Overpass hands back the identical objects with an
explicit ``timestamp_osm_base`` -- a *better* vintage statement than a PBF's
file mtime, because it is the replication instant of the database the answer was
computed from. Determinism is preserved the way it is preserved everywhere else
in this pipeline: by archiving the bytes and hashing them, not by hoping a
remote service is reproducible.

Selection: nodes and ways carrying any of ``amenity``, ``shop``, ``tourism``,
``leisure``, ``office``, ``craft``, ``healthcare``, or ``public_transport=station``
(see :data:`POI_SELECTORS`). Ways are reduced to their Overpass-computed
``center`` -- the centroid of the way's bounding box, *not* the polygon centroid;
for a shop footprint the two differ by well under a metre, but it is the
provider's number and is stored as such. Relations are excluded: multipolygon
POIs are rare, and including them would mix a third geometry convention into one
point column for a handful of features.

(b) Overture places
-------------------
``overturemaps download --bbox ... -f geoparquet --type place`` against the
latest release, which resolves the S3 GeoParquet through the STAC catalogue and
returns only the row groups the bbox touches. The release id (e.g.
``2026-08-19.0``) *is* the vintage and is recorded in the manifest; Overture
releases are immutable, so this is the one POI source in the snapshot that is
exactly reproducible by anyone, forever. No key, no account.

What is kept, and what that costs
----------------------------------
OSM rows keep the full tag dictionary as a JSON string (``tags_json``) so that
nothing is lost, *plus* explicit columns for the tags the artifact will actually
query -- ``opening_hours``, ``cuisine``, ``outdoor_seating``, ``website``,
``phone``, ``wheelchair``, ``addr:street``, ``addr:housenumber``. Those columns
are verbatim tag values under their verbatim OSM names (colons included, which
GPKG round-trips fine); they are *copies*, not extractions, and the JSON remains
authoritative.

Deliberately NOT coalesced: OSM stores contact details under both ``website`` /
``phone`` and ``contact:website`` / ``contact:phone``, and the two conventions
coexist in De Pijp. The columns hold the bare tag only; the fetch stats report
how many rows carry the ``contact:*`` form instead, so the reader can see the
size of the omission rather than have it silently repaired.

``primary_key`` / ``primary_value`` is the first key present in
:data:`PRIMARY_KEY_ORDER` -- a derived field, because a POI with
``amenity=cafe`` + ``shop=coffee`` + ``tourism=attraction`` has three equally
valid "categories" and the comparison report needs one. The precedence is an
editorial choice, stated here so it can be argued with.

``poi_class`` is the seven-way coarse class used for the overlay legend
(food&drink / shop / services / culture&leisure / health / transport / other).
It is a rendering aid, not a claim about the world: it is a lookup table over
~100 common tag values with key-level fallbacks, so genuinely unusual tagging
lands in ``other`` and stays visible. Notable judgement calls, all arguable:
education (``amenity=school|university|...``) is filed under *services*;
``amenity=nightclub`` under *culture&leisure*, not *food&drink*; and the many
street-furniture amenities (bench, waste_basket, recycling, drinking_water) fall
to *other*, which is why *other* is large and should stay large.

No cleaning anywhere: duplicate-looking POIs, POIs tagged on a building way *and*
on a node inside it, disused shops still tagged, joke tags -- all kept.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import Point

from ..config import CRS_RD, CRS_WGS84, District

# --------------------------------------------------------------------------- #
# (a) OpenStreetMap via Overpass
# --------------------------------------------------------------------------- #

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#: Mirrors, tried in order if the primary instance is rate-limited or down.
#: Same database, different front end -- the archived JSON records which one
#: answered via its ``generator`` string.
OVERPASS_MIRRORS: tuple[str, ...] = (
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
OSM_SOURCE_NAME = (
    "OpenStreetMap points of interest (nodes + ways-with-centroid), one-off "
    "snapshot via the public Overpass API"
)
#: OSM's own licence. Attribution "(c) OpenStreetMap contributors" is required,
#: and share-alike applies to derived *databases* -- which a snapshot is.
OSM_LICENCE = "ODbL 1.0 (Open Database License) - (c) OpenStreetMap contributors"

#: Overpass selectors, one per union member. Any object matching any of these is
#: a POI for our purposes. ``public_transport`` is value-restricted because
#: ``public_transport=platform|stop_position`` would pull in every tram pole.
POI_SELECTORS: tuple[str, ...] = (
    '["amenity"]',
    '["shop"]',
    '["tourism"]',
    '["leisure"]',
    '["office"]',
    '["craft"]',
    '["healthcare"]',
    '["public_transport"="station"]',
)

#: Precedence for picking the ONE primary tag of a multi-tagged POI. Ordered
#: most-specific-purpose first; ``public_transport`` last because a station that
#: also has ``amenity=...`` is better described by the amenity.
PRIMARY_KEY_ORDER: tuple[str, ...] = (
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "craft",
    "healthcare",
    "public_transport",
)

#: Tags promoted to their own column, under their verbatim OSM names.
OSM_TAG_COLUMNS: tuple[str, ...] = (
    "opening_hours",
    "cuisine",
    "outdoor_seating",
    "website",
    "phone",
    "wheelchair",
    "addr:street",
    "addr:housenumber",
)

#: The ``contact:`` namespace duplicates two of the above. Counted, not merged.
CONTACT_ALIASES: dict[str, str] = {
    "website": "contact:website",
    "phone": "contact:phone",
}

#: Amenity values eaten and drunk on the premises.
FOOD_DRINK_AMENITIES: frozenset[str] = frozenset({
    "restaurant", "cafe", "bar", "pub", "fast_food", "ice_cream", "biergarten",
    "food_court", "juice_bar",
})
#: Amenity values that are culture, entertainment or worship.
CULTURE_AMENITIES: frozenset[str] = frozenset({
    "theatre", "cinema", "arts_centre", "library", "community_centre",
    "place_of_worship", "nightclub", "casino", "studio", "music_venue",
    "exhibition_centre", "events_venue", "gallery", "public_bookcase",
})
#: Amenity values that are health care (``healthcare=*`` is handled by key).
HEALTH_AMENITIES: frozenset[str] = frozenset({
    "pharmacy", "doctors", "dentist", "hospital", "clinic", "veterinary",
    "nursing_home", "social_facility",
})
#: Amenity values that are transport infrastructure or vehicle services.
TRANSPORT_AMENITIES: frozenset[str] = frozenset({
    "parking", "parking_entrance", "parking_space", "bicycle_parking",
    "bicycle_rental", "bicycle_repair_station", "motorcycle_parking",
    "car_rental", "car_sharing", "car_wash", "charging_station", "fuel",
    "taxi", "ferry_terminal", "bus_station", "boat_rental", "boat_sharing",
})
#: Amenity values that are a commercial or public service counter.
SERVICE_AMENITIES: frozenset[str] = frozenset({
    "bank", "atm", "bureau_de_change", "post_office", "post_box", "post_depot",
    "police", "fire_station", "townhall", "courthouse", "embassy", "prison",
    "driving_school", "language_school", "music_school", "school", "college",
    "university", "kindergarten", "childcare", "research_institute",
    "coworking_space", "internet_cafe", "funeral_hall", "crematorium",
    "lawyer", "notary", "veterinary_pharmacy",
})

#: Coarse class -> legend colour. Seven classes, fixed order, fixed colours, so
#: successive renders of different districts stay comparable.
POI_CLASSES: tuple[str, ...] = (
    "food&drink", "shop", "services", "culture&leisure", "health",
    "transport", "other",
)
POI_CLASS_COLORS: dict[str, str] = {
    "food&drink": "#e6550d",
    "shop": "#3182bd",
    "services": "#756bb1",
    "culture&leisure": "#e7298a",
    "health": "#31a354",
    "transport": "#8c6d31",
    "other": "#969696",
}


def coarse_class(key: str | None, value: str | None) -> str:
    """Map a primary tag (key, value) onto one of :data:`POI_CLASSES`.

    Value lookups first (an ``amenity`` can be almost anything), then key-level
    fallbacks. Anything unrecognised returns ``"other"`` -- loudly, by being a
    large grey pile on the overlay rather than by disappearing.
    """
    if key == "amenity":
        if value in FOOD_DRINK_AMENITIES:
            return "food&drink"
        if value in CULTURE_AMENITIES:
            return "culture&leisure"
        if value in HEALTH_AMENITIES:
            return "health"
        if value in TRANSPORT_AMENITIES:
            return "transport"
        if value in SERVICE_AMENITIES:
            return "services"
        if value in ("marketplace", "vending_machine"):
            return "shop"
        return "other"
    if key == "shop":
        return "shop"
    if key == "craft":
        return "services"
    if key == "office":
        return "services"
    if key == "healthcare":
        return "health"
    if key == "tourism":
        return "culture&leisure"
    if key == "leisure":
        return "culture&leisure"
    if key == "public_transport":
        return "transport"
    return "other"


def build_overpass_query(
    bbox_wgs84: tuple[float, float, float, float], *, timeout_s: int = 300
) -> str:
    """The single Overpass QL query, built from :data:`POI_SELECTORS`.

    Overpass bboxes are ``(south, west, north, east)`` -- the transposition of
    the ``(min_lon, min_lat, max_lon, max_lat)`` convention used everywhere else
    in this package, which is exactly the kind of thing that should live in one
    function rather than in a string literal.

    ``out center tags`` emits the tag dictionary plus, for ways, a ``center``
    computed by Overpass as the midpoint of the object's bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    bb = f"({min_lat},{min_lon},{max_lat},{max_lon})"
    members = "\n".join(f"  nw{sel}{bb};" for sel in POI_SELECTORS)
    return f"[out:json][timeout:{timeout_s}];\n(\n{members}\n);\nout center tags;"


def fetch_osm(
    district: District,
    *,
    raw_dir: Path | None = None,
    instant: str,
    verbose: bool = True,
    timeout_s: int = 420,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """One Overpass call for ``district``; archive the JSON; parse to points.

    Same contract as every other layer: ``raw_dir`` gets the bytes actually
    received (gzipped) *before* any parsing, so the GeoPackage is always
    re-derivable offline.
    """
    import requests  # local import: keeps the module importable without network deps

    t0 = time.time()
    query = build_overpass_query(district.bbox_wgs84, timeout_s=timeout_s)
    headers = {"User-Agent": "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)"}

    payload: bytes | None = None
    endpoint_used: str | None = None
    errors: list[str] = []
    for url in OVERPASS_MIRRORS:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=timeout_s + 60)
            resp.raise_for_status()
            payload = resp.content
            endpoint_used = url
            break
        except Exception as exc:  # noqa: BLE001 - the mirror list exists for this
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  [osm] mirror failed, trying next: {errors[-1]}", flush=True)
    if payload is None:
        raise RuntimeError("all Overpass mirrors failed:\n  " + "\n  ".join(errors))
    if verbose:
        print(f"  [osm] {len(payload) / 1e6:.2f} MB from {endpoint_used}", flush=True)

    raw_path = None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = instant.replace(":", "").replace("-", "")
        raw_path = raw_dir / f"osm_pois_overpass_{stamp}.json.gz"
        with gzip.open(raw_path, "wb") as fh:
            fh.write(payload)
        # The query itself is provenance too: without it the JSON cannot be
        # re-requested, and "which tags did you ask for" is the first question
        # anyone will have about a POI count.
        (raw_dir / f"osm_pois_overpass_{stamp}.overpassql").write_text(query, encoding="utf-8")

    doc = json.loads(payload)
    osm3s = doc.get("osm3s", {})
    elements = doc.get("elements", [])

    rows: list[dict[str, Any]] = []
    n_no_coord = 0
    n_no_primary = 0
    contact_only = dict.fromkeys(CONTACT_ALIASES, 0)
    for el in elements:
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:  # ways carry their point under "center"
            center = el.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            n_no_coord += 1
            continue
        tags: dict[str, str] = el.get("tags", {}) or {}
        pk = next((k for k in PRIMARY_KEY_ORDER if k in tags), None)
        if pk is None:
            # Can happen when a mirror's selector handling differs; counted, not
            # hidden, and the row is still kept with a null primary tag.
            n_no_primary += 1
        pv = tags.get(pk) if pk else None
        for col, alias in CONTACT_ALIASES.items():
            if col not in tags and alias in tags:
                contact_only[col] += 1
        row: dict[str, Any] = {
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "name": tags.get("name"),
            "primary_key": pk,
            "primary_value": pv,
            "primary_tag": f"{pk}={pv}" if pk else None,
            "poi_class": coarse_class(pk, pv),
        }
        row.update({col: tags.get(col) for col in OSM_TAG_COLUMNS})
        row["n_tags"] = len(tags)
        row["tags_json"] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        row["geometry"] = Point(lon, lat)
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84).to_crs(CRS_RD)
    n_raw = len(gdf)
    # The eight selectors are a union, and Overpass already de-duplicates within
    # one query -- but the union is assembled server-side and a POI matching two
    # selectors has been observed to repeat on some mirrors. (type, id) is the
    # OSM primary key.
    gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"], keep="first").reset_index(drop=True)

    stats = {
        "elements_returned": len(elements),
        "pois_raw": n_raw,
        "duplicates_dropped": n_raw - len(gdf),
        "elements_without_coordinate": n_no_coord,
        "pois_without_primary_tag": n_no_primary,
        "nodes": int((gdf["osm_type"] == "node").sum()),
        "ways_centroid": int((gdf["osm_type"] == "way").sum()),
        "contact_namespace_only": contact_only,
        "download_bytes": len(payload),
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
        "endpoint": endpoint_used,
        "generator": doc.get("generator"),
        #: The replication instant of the OSM database behind the answer. This
        #: is the vintage -- more precise than any file date.
        "timestamp_osm_base": osm3s.get("timestamp_osm_base"),
        "query": query,
        "raw_file": None if raw_path is None else {
            "file": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "url": endpoint_used,
        },
    }
    return gdf, stats


# --------------------------------------------------------------------------- #
# (b) Overture Maps places
# --------------------------------------------------------------------------- #

OVERTURE_SOURCE_NAME = "Overture Maps Foundation - places theme (GeoParquet on S3)"
#: The places theme only. Overture's other themes are ODbL; places is not, which
#: is the whole reason this second file exists.
OVERTURE_LICENCE = "CDLA-Permissive-2.0 (data) + Apache-2.0 (schema)"
OVERTURE_DOCS_URL = "https://docs.overturemaps.org/guides/places/"
OVERTURE_S3_URL = "s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/"

#: Columns lifted out of the nested Overture structs, in storage order.
OVERTURE_COLUMNS: tuple[str, ...] = (
    "id",
    "name_primary",
    "category_primary",
    "category_alternate_json",
    "confidence",
    "operating_status",
    "address_freeform",
    "address_locality",
    "address_postcode",
    "address_region",
    "address_country",
    "addresses_json",
    "websites_json",
    "phones_json",
    "brand_name",
    "brand_wikidata",
    "sources_json",
    "source_datasets",
    "version",
)


def _json_or_none(value: Any) -> str | None:
    """Compact JSON for a nested Overture value; ``None`` stays ``None``.

    Nested structs are stored as JSON strings rather than exploded into columns
    because GPKG has no list/struct type and because the nesting *is* the
    provider's schema -- flattening it would be an interpretation.
    """
    if value is None:
        return None
    if isinstance(value, list) and not value:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def latest_release(python: str | None = None) -> str:
    """Ask the ``overturemaps`` CLI which release is current.

    Recorded in the manifest as the vintage. Overture releases are immutable, so
    this string plus the bbox is a complete, permanently reproducible recipe --
    the strongest provenance claim of any layer in the snapshot.
    """
    python = python or sys.executable
    out = subprocess.run(
        [python, "-m", "overturemaps", "releases", "latest"],
        check=True, capture_output=True, text=True, timeout=180,
    ).stdout.strip()
    return out.splitlines()[-1].strip()


def fetch_overture(
    district: District,
    *,
    raw_dir: Path | None = None,
    instant: str,
    release: str | None = None,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Download the Overture ``place`` rows for the bbox and flatten them.

    The CLI is invoked as a subprocess rather than imported: it is a downloader,
    its internals are not a public API, and a subprocess keeps the archived
    GeoParquet -- the raw artefact -- as the single interface between download
    and parse. The ``.parquet`` under ``raw/`` is the provider's own bytes; the
    GeoPackage next to it is this module's reading of them.
    """
    import pyarrow.parquet as pq
    import shapely

    t0 = time.time()
    release = release or latest_release()
    if verbose:
        print(f"  [overture] release {release}", flush=True)

    raw_dir = raw_dir or Path(".")
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = instant.replace(":", "").replace("-", "")
    raw_path = raw_dir / f"overture_places_{release}_{stamp}.parquet"

    min_lon, min_lat, max_lon, max_lat = district.bbox_wgs84
    cmd = [
        sys.executable, "-m", "overturemaps", "download",
        "--bbox", f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "-f", "geoparquet",
        "-t", "place",
        "-r", release,
        "-o", str(raw_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"overturemaps download failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    # The CLI drops a resume-state file next to the output; it is scratch, not
    # provenance, and would otherwise be hashed into the raw archive.
    state = raw_path.with_suffix(raw_path.suffix + ".state")
    if state.exists():
        state.unlink()

    table = pq.read_table(raw_path)
    if verbose:
        print(f"  [overture] {table.num_rows} places, "
              f"{raw_path.stat().st_size / 1e6:.2f} MB parquet", flush=True)

    rows: list[dict[str, Any]] = []
    n_no_geom = 0
    for rec in table.to_pylist():
        wkb = rec.get("geometry")
        if wkb is None:
            n_no_geom += 1
            continue
        names = rec.get("names") or {}
        cats = rec.get("categories") or {}
        addrs = rec.get("addresses") or []
        first = addrs[0] if addrs else {}
        brand = rec.get("brand") or {}
        brand_names = (brand.get("names") or {}) if brand else {}
        sources = rec.get("sources") or []
        rows.append({
            "id": rec.get("id"),
            "name_primary": names.get("primary"),
            "category_primary": cats.get("primary"),
            "category_alternate_json": _json_or_none(cats.get("alternate")),
            "confidence": rec.get("confidence"),
            "operating_status": rec.get("operating_status"),
            "address_freeform": first.get("freeform"),
            "address_locality": first.get("locality"),
            "address_postcode": first.get("postcode"),
            "address_region": first.get("region"),
            "address_country": first.get("country"),
            "addresses_json": _json_or_none(addrs),
            "websites_json": _json_or_none(rec.get("websites")),
            "phones_json": _json_or_none(rec.get("phones")),
            "brand_name": brand_names.get("primary"),
            "brand_wikidata": brand.get("wikidata"),
            "sources_json": _json_or_none(sources),
            # Denormalised for the report only: "who actually contributed this
            # POI" is the question a reader asks first, and digging it out of a
            # JSON blob every time is friction.
            "source_datasets": ",".join(
                sorted({s.get("dataset") for s in sources if s.get("dataset")})
            ) or None,
            "version": rec.get("version"),
            "geometry": shapely.from_wkb(wkb),
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84).to_crs(CRS_RD)

    conf = gdf["confidence"].dropna()
    stats = {
        "places_returned": table.num_rows,
        "places_without_geometry": n_no_geom,
        "release": release,
        "confidence_min": float(conf.min()) if len(conf) else None,
        "confidence_max": float(conf.max()) if len(conf) else None,
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
        "command": " ".join(cmd[1:]),
        "raw_file": {
            "file": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "url": OVERTURE_S3_URL.format(release=release),
        },
    }
    return gdf, stats
