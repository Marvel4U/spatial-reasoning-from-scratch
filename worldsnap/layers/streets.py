"""Street-surface polygons from the BGT, via PDOK's OGC API Features.

WHY BGT wegdeel
---------------
The BGT (Basisregistratie Grootschalige Topografie) is the Dutch national
large-scale base map: every physical surface in the country as a polygon, with
a legally mandated source holder (`bronhouder`, here Amsterdam = G0363) and a
maximum object size of a few metres. ``wegdeel`` is the road-surface feature
type, split by *function*: ``rijbaan lokale weg`` (carriageway), ``voetpad``
(sidewalk), ``fietspad`` (cycleway), ``parkeervlak`` (parking bay),
``voetgangersgebied`` (pedestrian zone), ``inrit`` (driveway), ``OV-baan``
(tram/bus bed). ``ondersteunendwegdeel`` carries the road's supporting surfaces
(``berm`` = verge, ``verkeerseiland`` = traffic island).

That is the point of this layer: a sidewalk is a *measured polygon*, not an OSM
``sidewalk=both`` tag. Widths, "which side of the street", "how much pavement is
there" become computable from geometry instead of assertable from tags.

NO WIDTH ATTRIBUTE EXISTS
-------------------------
BGT stores no width. Neither IMGeo class carries one. Any width must be derived
later from the polygon itself (skeleton / inscribed-circle / cross-section), and
that derivation is a separate, reviewable step -- deliberately not done here.

WHY the OGC API and not the download API
----------------------------------------
The briefed alternative was PDOK's BGT *download* API
(https://api.pdok.nl/lv/bgt/download/v1_0/, POST a polygon -> poll a job ->
fetch a zip of GML). It works but is asynchronous, returns CityGML that needs a
GML driver and an XSD-shaped attribute mapping, and hands back the whole
national feature-type schema. The OGC API Features endpoint at
https://api.pdok.nl/lv/bgt/ogc/v1 serves the same register as GeoJSON, takes an
arbitrary bbox, needs no registration or key, is declared CC0 1.0 in its own
landing page, and -- decisively -- can return coordinates *natively in
EPSG:28992* (``storageCrs``), so no reprojection is involved at all.

API quirks discovered by probing (these are the reason for the code below)
--------------------------------------------------------------------------
1. **The endpoint serves the full object history, not the current map.** An
   unfiltered bbox crawl of a 400x400 m test square returned 1109 wegdeel
   objects of which only 427 had ``eind_registratie == null``; every one of the
   1109 has ``status == "bestaand"``, so status does *not* separate live from
   superseded. Passing ``datetime=<instant>`` returns exactly the 427. We
   therefore always pass a single extract instant, recorded in the manifest, and
   the snapshot means "the BGT as it stood at that moment".
2. ``bbox`` is WGS84 unless ``bbox-crs`` says otherwise. Both ``crs`` and
   ``bbox-crs`` accept the EPSG:28992 OGC URI and are honoured, so we send and
   receive RD metres.
3. Paging is **cursor**-based (opaque ``cursor=`` token on the ``rel="next"``
   link), and there is no ``numberMatched`` -- the total is only knowable by
   crawling. As with 3DBAG we never construct paging URLs ourselves.
4. Attribute names are snake_cased relative to the IMGeo/GML spelling:
   ``fysiek_voorkomen`` for ``fysiekVoorkomen``, ``relatieve_hoogteligging``
   for ``relatieveHoogteligging``, ``status`` for ``bgt-status``.

Sentinels are kept, not cleaned
-------------------------------
IMGeo encodes "no value" out-of-band: a null attribute is paired with a
``*_leeg`` ("empty") field naming the *reason* (``waardeOnbekend`` = value
unknown, ``geenWaarde`` = no value applies). We store the reason fields
alongside the values verbatim, so a downstream consumer can tell "unknown" from
"not applicable" rather than inheriting a silently imputed value.
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

API_ROOT = "https://api.pdok.nl/lv/bgt/ogc/v1"
ITEMS_URL = f"{API_ROOT}/collections/{{featuretype}}/items"
LANDING_URL = f"{API_ROOT}?f=json"
LICENCE = "CC0 1.0"
SOURCE_NAME = (
    "BGT (Basisregistratie Grootschalige Topografie), PDOK / Kadaster, "
    "OGC API Features"
)

#: OGC URI for EPSG:28992; the API's own ``storageCrs``.
CRS_RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"

#: Road surfaces, and the supporting surfaces that bound them. Both share one
#: attribute schema, so they concatenate into a single layer with a
#: ``featuretype`` discriminator rather than needing two files.
FEATURE_TYPES = ("wegdeel", "ondersteunendwegdeel")

#: Server-side page size. 1000 is accepted and keeps the crawl to a few pages.
PAGE_LIMIT = 1000

#: Attributes kept per object. The remainder of the IMGeo attribute set is
#: codespace URIs (constant per field) and the ``version``/``lv_*`` bookkeeping
#: of the national register, none of which a map snapshot needs.
_ATTRS: dict[str, str] = {
    "lokaal_id": "lokaal_id",                      # BGT object id (bronhouder-scoped)
    "functie": "functie",                          # voetpad / rijbaan lokale weg / ...
    "fysiek_voorkomen": "fysiek_voorkomen",        # surface: open verharding / gesloten / ...
    "status": "bgt_status",                        # IMGeo 'bgt-status' (bestaand / plan / ...)
    "relatieve_hoogteligging": "relatieve_hoogteligging",  # -1 tunnel, 0 ground, +1 bridge
    "plus_functie": "plus_functie",                # optional IMGeo-plus refinement
    "plus_fysiek_voorkomen": "plus_fysiek_voorkomen",
    "in_onderzoek": "in_onderzoek",                # object under investigation
    "op_talud": "op_talud",                        # on an embankment
    "bronhouder": "bronhouder",                    # source holder, G0363 = Amsterdam
    "tijdstip_registratie": "tijdstip_registratie",  # start of this object version
    "eind_registratie": "eind_registratie",        # end of this version (null = current)
    "creation_date": "creation_date",
    "lv_publicatiedatum": "lv_publicatiedatum",
    # Out-of-band "why is this empty" sentinels -- kept verbatim, see docstring.
    "functie_leeg": "functie_leeg",
    "plus_functie_leeg": "plus_functie_leeg",
    "plus_fysiek_voorkomen_leeg": "plus_fysiek_voorkomen_leeg",
    "in_onderzoek_leeg": "in_onderzoek_leeg",
}


def _pages(
    session: requests.Session,
    featuretype: str,
    bbox_rd: tuple[float, float, float, float],
    instant: str,
    *,
    timeout: float = 90.0,
    pause_s: float = 0.05,
) -> Iterator[dict[str, Any]]:
    """Yield successive GeoJSON FeatureCollection pages for one feature type.

    ``instant`` is an ISO-8601 UTC timestamp; see quirk 1 in the module
    docstring -- without it the crawl returns the object *history*.
    """
    url: str | None = ITEMS_URL.format(featuretype=featuretype) + "?" + urlencode(
        {
            "f": "json",
            "limit": PAGE_LIMIT,
            "crs": CRS_RD_URI,
            "bbox-crs": CRS_RD_URI,
            "bbox": ",".join(f"{v:.3f}" for v in bbox_rd),
            "datetime": instant,
        }
    )
    while url:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        page = resp.json()
        yield page
        if page.get("numberReturned", 0) == 0:
            break
        url = next(
            (lk["href"] for lk in page.get("links", []) if lk.get("rel") == "next"), None
        )
        if url:
            time.sleep(pause_s)


def fetch(
    district: District,
    *,
    instant: str,
    raw_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Download BGT road-surface polygons intersecting ``district``'s bbox.

    ``instant`` fixes the temporal slice of the register (quirk 1). If
    ``raw_dir`` is given, every page is appended verbatim to one gzipped
    JSON-lines file per feature type -- the archive the manifest points at, so
    the parsed GeoPackage can always be re-derived from bytes we actually
    received.

    Returns the GeoDataFrame (EPSG:28992) and fetch statistics.
    """
    bbox = district.bbox_rd
    session = requests.Session()
    session.headers["User-Agent"] = (
        "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)"
    )

    rows: list[dict[str, Any]] = []
    per_type: dict[str, int] = {}
    raw_files: list[dict[str, Any]] = []
    n_pages = 0
    n_no_geom = 0
    t0 = time.time()

    for ft in FEATURE_TYPES:
        n_ft = 0
        raw_path = None
        raw_fh = None
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            stamp = instant.replace(":", "").replace("-", "")
            raw_path = raw_dir / f"bgt_{ft}_{stamp}.geojsonl.gz"
            raw_fh = gzip.open(raw_path, "wt", encoding="utf-8")
        try:
            for page in _pages(session, ft, bbox, instant):
                n_pages += 1
                if raw_fh is not None:
                    raw_fh.write(json.dumps(page, ensure_ascii=False) + "\n")
                for feat in page.get("features", []):
                    geom_json = feat.get("geometry")
                    if not geom_json:
                        n_no_geom += 1
                        continue
                    props = feat.get("properties", {})
                    row: dict[str, Any] = {out: props.get(src) for src, out in _ATTRS.items()}
                    row["featuretype"] = ft
                    row["api_id"] = feat.get("id")
                    row["geometry"] = shape(geom_json)
                    rows.append(row)
                    n_ft += 1
                if verbose:
                    print(f"  [{ft}] page {n_pages:3d}  objects so far: {n_ft:6d}", flush=True)
        finally:
            if raw_fh is not None:
                raw_fh.close()
        per_type[ft] = n_ft
        if raw_path is not None:
            raw_files.append(
                {
                    "featuretype": ft,
                    "file": raw_path.name,
                    "bytes": raw_path.stat().st_size,
                    "url": ITEMS_URL.format(featuretype=ft),
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_RD)

    # A snapshot must be a set. Paging is cursor-based, so a cursor restart
    # would silently duplicate; ``api_id`` is the register's stable version id.
    n_raw = len(gdf)
    gdf = gdf.drop_duplicates(subset="api_id", keep="first").reset_index(drop=True)

    gdf["area_m2"] = gdf.geometry.area  # RD is metric: area is m2 by construction

    reg = gdf["tijdstip_registratie"].dropna()
    stats = {
        "pages": n_pages,
        "objects_raw": n_raw,
        "duplicates_dropped": n_raw - len(gdf),
        "features_without_geometry": n_no_geom,
        "per_featuretype": per_type,
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
        "raw_files": raw_files,
        "tijdstip_registratie_min": str(reg.min()) if len(reg) else None,
        "tijdstip_registratie_max": str(reg.max()) if len(reg) else None,
    }
    return gdf, stats


#: Semantic colours for the IMGeo ``FunctieWeg`` / ``FunctieOndersteunendWegdeel``
#: value lists. Chosen so the sanity check the render exists for -- "does a
#: sidewalk trace both sides of every street?" -- is answerable at a glance:
#: carriageway greys recede, ``voetpad`` is the one saturated warm colour.
#: Values not listed here still get a colour (see ``render.categorical_palette``).
FUNCTIE_COLORS: dict[str, str] = {
    # wegdeel
    "rijbaan lokale weg": "#8a8a8a",
    "rijbaan regionale weg": "#6b6b6b",
    "rijbaan autoweg": "#4f4f4f",
    "rijbaan autosnelweg": "#3a3a3a",
    "voetpad": "#e8912b",
    "voetpad op trap": "#a85f12",
    "voetgangersgebied": "#f6d08a",
    "fietspad": "#c0392b",
    "ruiterpad": "#7d5a3c",
    "parkeervlak": "#8e6fb0",
    "inrit": "#c9b6e0",
    "OV-baan": "#2e8b57",
    "spoorbaan": "#444444",
    "overweg": "#8c6d31",
    "verkeersdrempel": "#d94f9a",
    "transitie": "#00a0a0",
    "baan voor vliegverkeer": "#999999",
    "calamiteitendoorsteek": "#b5651d",
    # ondersteunendwegdeel
    "berm": "#9ecf8a",
    "verkeerseiland": "#6fa8dc",
}
