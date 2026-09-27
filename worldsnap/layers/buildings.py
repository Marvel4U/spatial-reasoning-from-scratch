"""Buildings from 3DBAG, via its OGC API Features endpoint.

WHY 3DBAG (and not plain BAG / OSM)
-----------------------------------
3DBAG = BAG footprints (authoritative national register) fused with AHN LiDAR,
so every building carries *measured* roof geometry, not a tag-guessed height.
CC BY 4.0, no registration, national coverage. That combination is what makes
sun/shadow, facade orientation and "how tall is that block" verifiable by
construction rather than by assertion -- the survey calls it the single
strongest argument for choosing Amsterdam.

WHY the OGC API and not the GeoPackage tiles
--------------------------------------------
The API takes an arbitrary bbox, so a district snapshot is one bbox away and
needs no tile-index lookup or hundreds-of-MB download. It was tried first and
works; the tile route stays the documented fallback if the API is ever down.

API quirks discovered by probing (these are the reason for the code below)
--------------------------------------------------------------------------
1. ``bbox`` is interpreted in the collection's *storage* CRS, EPSG:7415
   (= RD New + NAP), **not** WGS84. A WGS84 bbox silently returns 0 features.
   ``bbox-crs`` is advertised but returns HTTP 400, so we always send RD metres.
2. ``offset`` is 1-based; ``offset=0`` returns HTTP 500. We therefore never
   build offsets ourselves -- we follow the server's ``rel="next"`` links.
3. ``numberMatched`` / ``numberReturned`` count **CityObjects**, not features.
   Each building yields a Building plus one BuildingPart, so those numbers are
   roughly 2x the building count. Only ``len(features)`` is trustworthy.
   Verified: crawling the same bbox at limit=100 and limit=7 yields the
   identical set of building ids, so next-link paging is stable.
4. Responses are CityJSONFeature: integer vertices plus a per-response
   ``transform`` (scale/translate) that differs between pages. The transform
   must be applied per page, never cached across pages.

WHY the LoD0 footprint
----------------------
LoD0 is the 2D ground footprint, present for every building and identical to
the BAG ``pand`` polygon. A footprint plus a height attribute is exactly the
2.5D representation the map renderer and the future shadow tool need; the
LoD2.2 solids are far heavier and can be re-fetched later if real roof planes
matter.

WHY ``b3_h_dak_70p - b3_h_maaiveld`` as *the* height
-----------------------------------------------------
All 3DBAG heights are absolute elevations above NAP, so "height above ground"
must be a difference. Of the roof statistics:
  * ``b3_h_dak_max``  -- the highest roof point: sensitive to ridges, dormers
    and stray LiDAR returns, over-reads a typical pitched Amsterdam roof.
  * ``b3_h_dak_50p``  -- median roof point: robust, but under-reads pitched
    roofs because half the roof sits above it.
  * ``b3_h_dak_70p``  -- the conventional compromise in Dutch AHN practice and
    the percentile used for block-model extrusion; close to eaves level for
    flat roofs and sensibly up the slope for pitched ones.
We store all three (plus min and the ridge ``b3_h_nok``) so the choice can be
revisited without re-downloading, and expose 70p as ``height_m``.
"""

from __future__ import annotations

import time
from typing import Any, Iterator
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon

from ..config import CRS_RD, District

API_ROOT = "https://api.3dbag.nl"
ITEMS_URL = f"{API_ROOT}/collections/pand/items"
COLLECTION_URL = f"{API_ROOT}/collections/pand"
LICENCE = "CC BY 4.0"
SOURCE_NAME = "3DBAG (TU Delft 3D geoinformation), collection 'pand', OGC API Features"

#: Server caps the page size at 100 CityObjects (~50 buildings) regardless.
PAGE_LIMIT = 100

#: Attributes kept per building. Everything else in the 3DBAG attribute set is
#: reconstruction diagnostics (rmse, point density, val3dity) we do not need
#: for a map snapshot.
_ATTRS: dict[str, str] = {
    "identificatie": "identificatie",       # BAG pand id, the join key
    "oorspronkelijkbouwjaar": "bouwjaar",   # original construction year
    "status": "status",                     # BAG lifecycle status
    "b3_h_maaiveld": "h_maaiveld",          # ground elevation, NAP m
    "b3_h_dak_min": "h_dak_min",            # roof statistics, NAP m
    "b3_h_dak_50p": "h_dak_50p",
    "b3_h_dak_70p": "h_dak_70p",
    "b3_h_dak_max": "h_dak_max",
    "b3_h_nok": "h_nok",                    # ridge line height, NAP m
    "b3_dak_type": "dak_type",
    "b3_bouwlagen": "bouwlagen",            # storey estimate (only up to 5)
    "b3_opp_grond": "opp_grond",            # ground area m2, 3DBAG's own
}


def collection_vintage(session: requests.Session, timeout: float = 30.0) -> str:
    """Provider-stated dataset version, e.g. ``v2023.10.08``.

    3DBAG republishes periodically and the API exposes no per-feature
    'last modified', so the collection version string is the only vintage the
    provider states. Returns ``"unknown"`` rather than failing the snapshot.
    """
    try:
        meta = session.get(COLLECTION_URL, timeout=timeout).json()
        return str(meta.get("version", {}).get("collection", "unknown"))
    except Exception:
        return "unknown"


def _pages(
    session: requests.Session,
    bbox_rd: tuple[float, float, float, float],
    *,
    timeout: float = 60.0,
    pause_s: float = 0.05,
) -> Iterator[dict[str, Any]]:
    """Yield successive CityJSONFeatureCollection pages for ``bbox_rd``.

    Paging follows the server's ``rel="next"`` link (see module docstring,
    quirk 2). A tiny pause keeps us polite on a free public service.
    """
    url: str | None = f"{ITEMS_URL}?" + urlencode(
        {"bbox": ",".join(f"{v:.3f}" for v in bbox_rd), "limit": PAGE_LIMIT}
    )
    while url:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        page = resp.json()
        yield page
        url = next(
            (lk["href"] for lk in page.get("links", []) if lk.get("rel") == "next"), None
        )
        if url:
            time.sleep(pause_s)


def _footprint(
    boundaries: list,
    vertices: list[list[int]],
    scale: list[float],
    translate: list[float],
) -> Polygon | MultiPolygon | None:
    """Turn a CityJSON LoD0 MultiSurface into a 2D shapely polygon in RD.

    ``boundaries`` is [surface][ring][vertex_index]; ring 0 is the exterior,
    further rings are holes. Vertex coordinates are integers, de-quantised as
    ``v * scale + translate``. Z is dropped: LoD0 sits at ground level and the
    working CRS, EPSG:28992, is 2D by definition.
    """

    def ring(idx: list[int]) -> list[tuple[float, float]]:
        return [
            (
                vertices[i][0] * scale[0] + translate[0],
                vertices[i][1] * scale[1] + translate[1],
            )
            for i in idx
        ]

    polys: list[Polygon] = []
    for surface in boundaries:
        if not surface:
            continue
        shell = ring(surface[0])
        if len(shell) < 3:
            continue
        poly = Polygon(shell, [ring(h) for h in surface[1:] if len(h) >= 3])
        if not poly.is_valid:
            poly = poly.buffer(0)  # repairs self-touching footprints
        if poly.is_empty:
            continue
        polys.append(poly)
    if not polys:
        return None
    if len(polys) == 1 and polys[0].geom_type == "Polygon":
        return polys[0]
    parts = [g for p in polys for g in (p.geoms if p.geom_type == "MultiPolygon" else [p])]
    return MultiPolygon(parts)


def fetch(district: District, *, verbose: bool = True) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Download all 3DBAG buildings intersecting ``district``'s bbox.

    Returns the GeoDataFrame (EPSG:28992) and a dict of fetch statistics for
    the manifest and the run report.
    """
    bbox = district.bbox_rd
    session = requests.Session()
    session.headers["User-Agent"] = (
        "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)"
    )

    rows: list[dict[str, Any]] = []
    n_pages = 0
    n_no_lod0 = 0
    t0 = time.time()

    for page in _pages(session, bbox):
        n_pages += 1
        tr = page["metadata"]["transform"]  # per page! see quirk 4
        scale, translate = tr["scale"], tr["translate"]
        for feat in page["features"]:
            obj = feat["CityObjects"].get(feat["id"])
            if obj is None:
                continue
            lod0 = next(
                (g for g in obj.get("geometry", []) if str(g.get("lod")) == "0"), None
            )
            if lod0 is None:
                n_no_lod0 += 1
                continue
            geom = _footprint(lod0["boundaries"], feat["vertices"], scale, translate)
            if geom is None:
                n_no_lod0 += 1
                continue
            attrs = obj.get("attributes", {})
            row: dict[str, Any] = {out: attrs.get(src) for src, out in _ATTRS.items()}
            row["geometry"] = geom
            rows.append(row)
        if verbose and n_pages % 20 == 0:
            print(f"  page {n_pages:4d}  buildings so far: {len(rows):6d}", flush=True)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_RD)

    # De-duplicate: a snapshot must be a set. Empirically the API does not
    # repeat features across pages, but paging bugs would be silent otherwise.
    n_raw = len(gdf)
    gdf = gdf.drop_duplicates(subset="identificatie", keep="first").reset_index(drop=True)

    # Height above ground = roof elevation - ground elevation (both NAP).
    for p in ("50p", "70p", "max"):
        gdf[f"height_{p}_m"] = gdf[f"h_dak_{p}"] - gdf["h_maaiveld"]
    gdf["height_m"] = gdf["height_70p_m"]  # the documented primary height

    gdf["bouwjaar"] = pd.to_numeric(gdf["bouwjaar"], errors="coerce").astype("Int64")

    stats = {
        "pages": n_pages,
        "buildings_raw": n_raw,
        "duplicates_dropped": n_raw - len(gdf),
        "features_without_lod0": n_no_lod0,
        "seconds": round(time.time() - t0, 1),
        "vintage": collection_vintage(session),
        "height_field": "b3_h_dak_70p - b3_h_maaiveld",
    }
    return gdf, stats
