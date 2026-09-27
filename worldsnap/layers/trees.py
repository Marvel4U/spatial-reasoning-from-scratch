"""Municipal tree register (``bomen``) from the Amsterdam DSO API.

WHY this endpoint and not maps.amsterdam.nl
--------------------------------------------
The survey pointed at https://maps.amsterdam.nl/bomen/ , which is a *viewer*.
Its ``open_geodata/geojson.php`` download path 404s for the tree layer (probed
2026-09-12), and its per-tree downloads are documented as incomplete. The live
register behind the viewer is published properly as an OGC-ish REST dataset on
the city's DSO API:

    https://api.data.amsterdam.nl/v1/bomen/stamgegevens/

``stamgegevens`` ("trunk data") is the tree passport table -- one row per
municipally managed tree. The source system is **Gisib**, the city's asset
management database, so this is an operational register, not a survey: it
covers trees the municipality *manages*, which excludes private gardens and
courtyards. That bias is the single most important caveat for any density
number computed from this layer.

WHY it needs no key
-------------------
The API advertises an ``X-Api-Key`` header but its own OpenAPI document says
it is "for statistical purposes, not for authentication". Anonymous requests
return full pages. No registration was needed at any point.

WHY server-side bbox and not a citywide download
-------------------------------------------------
The dataset supports ``geometrie[intersects]=<WKT POLYGON>`` with
``Content-Crs: EPSG:28992``, i.e. the filter is expressed directly in the
working CRS and the response comes back in it too (``Accept-Crs``). A district
cut is therefore one filtered crawl, with no citywide file to clip and no
reprojection anywhere in the path -- the same property that made the BGT
endpoint the right choice for streets.

API quirks discovered by probing (these are the reason for the code below)
---------------------------------------------------------------------------
1. Paging is ``page=N`` with a ``_links.next`` href in the GeoJSON envelope.
   The envelope is NOT standard GeoJSON: ``_links`` is a top-level array
   alongside ``features``. We follow the server's href rather than building
   page numbers.
2. ``_format=geojson`` returns a *subset* of the attributes that
   ``_format=json`` exposes (the schema lists ~40 filterable fields; the
   GeoJSON projection ships 20). We take the GeoJSON form deliberately: it is
   the projection the city itself publishes as the spatial product, and every
   field the brief asks for (species, planting year, height class, trunk
   diameter class, type, status) is present in it. The wider set stays one
   ``_format=json`` crawl away if it is ever needed.
3. Coordinates are already EPSG:28992 when ``Accept-Crs`` says so -- verified
   against the collection's declared ``crs`` of EPSG:28992.

Sentinels are kept, not cleaned
-------------------------------
Original Dutch field names are preserved verbatim (``soortnaam``,
``jaarVanAanleg``, ``boomhoogteklasseActueel``, ``stamdiameterklasse``, ...).
Height and diameter are **ordered class labels carrying their own sort key** --
``"e. 15 tot 18 m."``, ``"0,2 tot 0,3 m."`` -- with a Dutch decimal comma and a
leading letter. They are stored as the strings they are; parsing them into
numbers is a lossy decision that belongs downstream, not in the snapshot.
``jaarVanAanleg`` contains nulls and placeholder-looking years; nothing is
imputed or dropped.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import geopandas as gpd
import requests
from shapely.geometry import shape

from ..config import CRS_RD, District

API_ROOT = "https://api.data.amsterdam.nl/v1/bomen"
ITEMS_URL = f"{API_ROOT}/stamgegevens/"
SCHEMA_URL = "https://schemas.data.amsterdam.nl/datasets/bomen/dataset"
PORTAL_URL = "https://data.amsterdam.nl/datasets/2Ml5FpaxwrJTPQ/bomen/"
SOURCE_NAME = (
    "Bomen / stamgegevens (municipal tree register, source system Gisib), "
    "Gemeente Amsterdam DSO API v1"
)
#: Verbatim from the dataset schema's ``license`` field at SCHEMA_URL. Not an
#: SPDX identifier; recorded as the portal states it rather than normalised.
LICENCE = "openbaar, tenzij anders aangegeven / behoudens uitzonderingen"
#: Owner/creator as the schema states them, kept for the attribution line.
OWNER = "Directie V&OR, Gemeente Amsterdam"

#: Server-side page size. 1000 is accepted; larger values are silently capped.
PAGE_LIMIT = 1000

#: Attributes kept, in the order they are stored. Original Dutch names, no
#: renaming: the register's vocabulary is part of the provenance.
_ATTRS: tuple[str, ...] = (
    "id",                              # record id in the register
    "soortnaam",                       # full species name, e.g. Tilia americana
    "soortnaamKort",                   # genus, from the reference table
    "soortnaamTop",                    # Dutch common name, e.g. Linde (Tilia)
    "typeSoortnaam",                   # "Bomen" vs other vegetation object types
    "jaarVanAanleg",                   # planting year
    "boomhoogteklasseActueel",         # height class, ordered label
    "stamdiameterklasse",              # trunk diameter class, ordered label
    "typeObject",                      # free-growing vs constrained
    "standplaats",                     # growing site (paving, green object, ...)
    "standplaatsGedetailleerd",        # finer site description
    "typeEigenaarPlus",                # owner
    "typeBeheerderPlus",               # manager (city department)
    "beschermingsstatus",              # protected-tree status
    "beschermingsstatusGedetailleerd",
    "groeiplaatsBoomId",               # link to the growing-site table
    "gbdBuurtId",                      # GBD neighbourhood code
    "geldigVanaf",                     # record valid from
    "mutatieDatum",                    # last mutation -- the de-facto vintage
)

#: Ordered height classes as the register spells them. The leading letter is
#: the register's own sort key, so a lexical sort is already the right order;
#: listing them explicitly makes an unexpected class visible instead of silently
#: sorting into the middle of the ramp.
HOOGTEKLASSEN: tuple[str, ...] = (
    "a. tot 6 m.",
    "b. 6 tot 9 m.",
    "c. 9 tot 12 m.",
    "d. 12 tot 15 m.",
    "e. 15 tot 18 m.",
    "f. 18 tot 24 m.",
    "g. 24 m. en hoger",
)

#: Light -> dark green ramp over HOOGTEKLASSEN; unknown class renders grey.
HOOGTEKLASSE_COLORS: dict[str, str] = {
    "a. tot 6 m.": "#d9f0c4",
    "b. 6 tot 9 m.": "#addd8e",
    "c. 9 tot 12 m.": "#78c679",
    "d. 12 tot 15 m.": "#41ab5d",
    "e. 15 tot 18 m.": "#238443",
    "f. 18 tot 24 m.": "#005a32",
    "g. 24 m. en hoger": "#00331c",
}


def _wkt_box(bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    return (
        f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
    )


def _pages(
    session: requests.Session,
    bbox_rd: tuple[float, float, float, float],
    *,
    timeout: float = 90.0,
    pause_s: float = 0.05,
) -> Iterator[dict[str, Any]]:
    """Yield successive GeoJSON pages for the district bbox (quirk 1)."""
    url: str | None = ITEMS_URL + "?" + urlencode(
        {
            "_format": "geojson",
            "_pageSize": PAGE_LIMIT,
            "geometrie[intersects]": _wkt_box(bbox_rd),
        }
    )
    while url:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        page = resp.json()
        yield page
        url = next(
            (lk["href"] for lk in page.get("_links", []) if lk.get("rel") == "next"), None
        )
        if url:
            time.sleep(pause_s)


def fetch(
    district: District,
    *,
    raw_dir: Path | None = None,
    instant: str,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Download the tree register inside ``district``'s bbox.

    ``raw_dir`` archives every page verbatim as gzipped JSON-lines, the same
    contract as the BGT layer: the parsed GeoPackage must always be
    re-derivable from bytes actually received.
    """
    bbox = district.bbox_rd
    session = requests.Session()
    session.headers.update({
        "User-Agent": "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)",
        "Accept-Crs": "EPSG:28992",
        "Content-Crs": "EPSG:28992",
    })

    rows: list[dict[str, Any]] = []
    n_pages = 0
    n_no_geom = 0
    raw_path = None
    raw_fh = None
    t0 = time.time()

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = instant.replace(":", "").replace("-", "")
        raw_path = raw_dir / f"bomen_stamgegevens_{stamp}.geojsonl.gz"
        raw_fh = gzip.open(raw_path, "wt", encoding="utf-8")
    try:
        for page in _pages(session, bbox):
            n_pages += 1
            if raw_fh is not None:
                raw_fh.write(json.dumps(page, ensure_ascii=False) + "\n")
            for feat in page.get("features", []):
                geom_json = feat.get("geometry")
                if not geom_json:
                    n_no_geom += 1
                    continue
                props = feat.get("properties", {})
                row: dict[str, Any] = {k: props.get(k) for k in _ATTRS}
                row["api_id"] = feat.get("id")
                row["geometry"] = shape(geom_json)
                rows.append(row)
            if verbose:
                print(f"  [bomen] page {n_pages:3d}  trees so far: {len(rows):6d}", flush=True)
    finally:
        if raw_fh is not None:
            raw_fh.close()

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_RD)
    n_raw = len(gdf)
    # Paging is page-number based and the register mutates; a concurrent write
    # could in principle show one record twice. ``api_id`` is stable.
    gdf = gdf.drop_duplicates(subset="api_id", keep="first").reset_index(drop=True)

    mut = gdf["mutatieDatum"].dropna()
    stats = {
        "pages": n_pages,
        "trees_raw": n_raw,
        "duplicates_dropped": n_raw - len(gdf),
        "features_without_geometry": n_no_geom,
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
        "mutatie_datum_min": str(mut.min()) if len(mut) else None,
        "mutatie_datum_max": str(mut.max()) if len(mut) else None,
        "raw_file": None if raw_path is None else {
            "file": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "url": ITEMS_URL,
        },
    }
    return gdf, stats
