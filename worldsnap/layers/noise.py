"""Strategic noise map (EU END "geluidskaart") from Gemeente Amsterdam.

WHAT the source actually is
---------------------------
``GELUID_2021`` is the city's own EU Environmental Noise Directive round: the
**calculated** annual noise load for the main traffic sources, published as
**isophone band polygons** -- not a raster, and not a per-facade or per-street
table. Each feature carries the band it belongs to, so the dataset is a set of
5 dB *class* polygons, one nested ring set per source x period. The reference
year is stated by the provider as **2021** (the ``jaar`` column, and the viewer
is titled "Geluidskaart 2021"), i.e. the 2022 END reporting round.

WHY this endpoint
-----------------
The survey pointed at https://maps.amsterdam.nl/geluid/ , which is a *viewer*
whose download links are not on the page: the viewer asks
``_maps.min/haal.legenda.php`` for its legend, and the legend rows name the
table (``GELUID_2021``) and the class selector (``SELECTIE``). With the table
name the Maps Data download path resolves:

    https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php
        ?KAARTLAAG=GELUID_2021&THEMA=geluid

That is one 32 MB citywide GeoJSON in WGS84 -- 19218 polygons, all sources and
both periods in one file, distinguished only by ``SELECTIE``. There is no
server-side bbox filter (unlike BGT and bomen), so this is the first layer in
the snapshot where the whole city is downloaded and clipped locally. Rejected
alternatives are listed at the bottom of this docstring.

The class code, and what is *inferred*
--------------------------------------
``SELECTIE`` is ``<bron>_<periode>_<legenda>``:

* ``bron``    -- ``vl`` wegverkeer (road), ``rl`` railverkeer, ``ll``
  luchtvaart (aviation), ``il`` industrie.
* ``periode`` -- ``lden`` (24 h, evening +5 dB and night +10 dB weighted) or
  ``lnight`` (23:00-07:00).
* ``legenda`` -- the band index.

The viewer's legend only labels the bands it *draws*: Lden from index 2
(55-60 dB) upward and Lnight from index 12 (50-55 dB) upward, matching the END
reporting floors. The data additionally contains index **1** (Lden) and **11**
(Lnight), which the map never shows. Those two labels are therefore an
**inference**: the bands are a regular 5 dB ladder, so index 1 is read as
50-55 dB Lden and index 11 as 45-50 dB Lnight. The inference is flagged in the
manifest as ``lowest_band_label_inferred``. Every other label is the city's own
wording, translated.

Two facts from the viewer's own Toelichting change how this layer must be read,
and no amount of staring at the geometry would reveal either:

1. **Trams are inside road noise.** "Wegverkeerslawaai omvat het totale geluid
   van motoren, auto's, bussen, vrachtwagens en trams." There is no separate
   tram layer, and ``rl`` (railverkeer) is heavy rail only, supplied by
   ProRail. A tram corridor therefore shows up as elevated ``vl``, and the
   metro -- underground through De Pijp -- shows up nowhere at all.
2. **Industry uses a different unit.** ``il`` bands are Dutch *etmaalwaarde*
   dB(A), not Lden dB, and ``il_*_99`` / ``il_*_999`` are not contours at all
   but the boundary of a zoned industrial estate and of a 55 dB(A)
   installation. They are kept verbatim and labelled as zoning, not as a band.

No cleaning
-----------
Polygons are kept whole: a feature that pokes out of the district bbox is
stored intact, exactly as BGT and bomen features are, so the stored geometry is
always the provider's geometry.

The bands were *checked*, not assumed: in De Pijp the six Lden band polygons
are mutually disjoint to within 0.02 ha (one sliver shared between the 70-75
and 75+ bands), so they partition the area above the lowest published band
rather than nesting inside one another. They cover 68% of the district bbox;
the remaining 32% is simply below 50 dB Lden and has no polygon at all. Absence
of a band is therefore data ("quieter than the lowest published band"), not a
gap, and nothing is filled in.

Sources tried and rejected
--------------------------
* ``maps.amsterdam.nl/open_geodata/geojson.php`` (the RD-coordinate sibling of
  the endpoint used) -- 404 for this table; only ``geojson_lnglat.php`` and
  ``excel.php`` respond, so the reprojection to RD has to happen here.
* ``api.data.amsterdam.nl/v1/geluid...`` -- the DSO API carries no geluid
  collection; the noise map is a Maps Amsterdam product only.
* Atlas Leefomgeving / RIVM geluidkaarten -- a national aggregation of the same
  END submissions, published as WMS (imagery, unusable as data) and at coarser
  granularity. The municipal dataset is the upstream one.
* ``geluidregister.nl`` -- the statutory register the viewer links to for
  per-building maximum noise sensitivity; behind a login.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import box, shape

from ..config import CRS_RD, CRS_WGS84, District

TABLE = "GELUID_2021"
THEME = "geluid"
ITEMS_URL = (
    "https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php"
    f"?KAARTLAAG={TABLE}&THEMA={THEME}"
)
VIEWER_URL = "https://maps.amsterdam.nl/geluid/"
DATASET_URL = "https://maps.amsterdam.nl/open_geodata/?k=492"
TERMS_URL = "https://maps.amsterdam.nl/open_geodata/terms.php"
SOURCE_NAME = (
    "Geluidskaart Amsterdam 2021 (EU Environmental Noise Directive strategic "
    "noise map), table GELUID_2021, Maps Data / Maps Amsterdam"
)
#: Paraphrase of the Gebruiksvoorwaarden at TERMS_URL, which is a bespoke city
#: licence rather than an SPDX one: reuse for any lawful purpose, commercial and
#: non-commercial, no copyright transferred, attribution appreciated but not
#: required, no warranty, historical versions not retained.
LICENCE = (
    "Gebruiksvoorwaarden Maps Data, Gemeente Amsterdam - free use and reuse for "
    "any lawful purpose incl. commercial; attribution appreciated, not required; "
    "no warranty; see " + TERMS_URL
)
#: The provider's own reference year, from the ``jaar`` column and the map title.
VINTAGE = "2021"
OWNER = "Gemeente Amsterdam - Ruimte en Duurzaamheid"

#: ``bron`` code -> (file-name slug, English label, unit of the bands).
BRONNEN: dict[str, tuple[str, str, str]] = {
    "vl": ("road", "road traffic (incl. trams)", "dB Lden / Lnight"),
    "rl": ("rail", "rail traffic (heavy rail, ProRail)", "dB Lden / Lnight"),
    "ll": ("air", "aviation (Schiphol)", "dB Lden / Lnight"),
    "il": ("industry", "industry", "dB(A) etmaalwaarde"),
}

#: ``legenda`` index -> (label, lower bound dB, upper bound dB or None).
#: Indices 2-6 / 12-16 carry the viewer's own labels; 1 and 11 are inferred (see
#: module docstring). 99 and 999 are industry zoning outlines, not bands.
DB_CLASSES: dict[str, dict[int, tuple[str, float | None, float | None]]] = {
    "lden": {
        1: ("50-55 dB", 50.0, 55.0),
        2: ("55-60 dB", 55.0, 60.0),
        3: ("60-65 dB", 60.0, 65.0),
        4: ("65-70 dB", 65.0, 70.0),
        5: ("70-75 dB", 70.0, 75.0),
        6: ("75 dB and above", 75.0, None),
        99: ("zoned industrial estate (outline)", None, None),
        999: ("55 dB(A) installation (outline)", None, None),
    },
    "lnight": {
        11: ("45-50 dB", 45.0, 50.0),
        12: ("50-55 dB", 50.0, 55.0),
        13: ("55-60 dB", 55.0, 60.0),
        14: ("60-65 dB", 60.0, 65.0),
        15: ("65-70 dB", 65.0, 70.0),
        16: ("70 dB and above", 70.0, None),
        99: ("zoned industrial estate (outline)", None, None),
        999: ("55 dB(A) installation (outline)", None, None),
    },
}
#: Band indices whose label this module inferred rather than read off the map.
INFERRED_BAND_INDICES: dict[str, tuple[int, ...]] = {"lden": (1,), "lnight": (11,)}

#: Sequential warm ramp. The 55 dB and louder bands reuse the city's own hex
#: codes from the viewer legend, so the overlay is directly comparable with
#: maps.amsterdam.nl; the undisplayed lowest band continues the same ramp paler.
BAND_COLORS: dict[str, str] = {
    "45-50 dB": "#FFF7C2",
    "50-55 dB": "#FFE796",
    "55-60 dB": "#FFC669",
    "60-65 dB": "#FF8444",
    "65-70 dB": "#FF0000",
    "70-75 dB": "#B7001F",
    "70 dB and above": "#B7001F",
    "75 dB and above": "#690039",
    "zoned industrial estate (outline)": "#4040FF",
    "55 dB(A) installation (outline)": "#808080",
}


def band_order(periode: str) -> list[str]:
    """Band labels for ``periode``, quietest first -- the legend's ramp order."""
    seen: list[str] = []
    for idx in sorted(DB_CLASSES[periode]):
        label = DB_CLASSES[periode][idx][0]
        if label not in seen:
            seen.append(label)
    return seen


def _decode(periode: str, legenda: Any) -> tuple[str, float | None, float | None]:
    """Band label and dB bounds; an unseen index degrades loudly, not silently."""
    table = DB_CLASSES.get(periode, {})
    try:
        key = int(legenda)
    except (TypeError, ValueError):
        return (f"<unparseable {periode} band {legenda!r}>", None, None)
    if key in table:
        return table[key]
    return (f"<unknown {periode} band {key}>", None, None)


def fetch(
    district: District,
    *,
    raw_dir: Path | None = None,
    instant: str,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Download the citywide noise map and keep what touches ``district``.

    ``raw_dir`` archives the response verbatim (gzipped) before any parsing, the
    same contract as the BGT and bomen layers: the stored GeoPackages must be
    re-derivable from bytes actually received.
    """
    t0 = time.time()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "worldsnap/0.1 (spatial_reasoning_LLM_artifact; research)"}
    )
    resp = session.get(ITEMS_URL, timeout=600)
    resp.raise_for_status()
    payload = resp.content
    if verbose:
        print(f"  [geluid] {len(payload) / 1e6:.1f} MB downloaded", flush=True)

    raw_path = None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = instant.replace(":", "").replace("-", "")
        raw_path = raw_dir / f"{TABLE.lower()}_{stamp}.geojson.gz"
        with gzip.open(raw_path, "wb") as fh:
            fh.write(payload)

    doc = json.loads(payload)
    feats = doc["features"]

    # The endpoint only serves WGS84, so the clip test runs in WGS84 and the
    # geometry is projected to RD afterwards -- the one layer in the snapshot
    # where a reprojection is unavoidable.
    x0, y0, x1, y1 = district.bbox_wgs84
    bb = box(x0, y0, x1, y1)
    rows: list[dict[str, Any]] = []
    n_no_geom = 0
    for feat in feats:
        geom_json = feat.get("geometry")
        if not geom_json:
            n_no_geom += 1
            continue
        geom = shape(geom_json)
        if not geom.intersects(bb):
            continue
        props = feat.get("properties", {})
        periode = props.get("periode")
        label, lo, hi = _decode(periode, props.get("legenda"))
        slug, bron_label, unit = BRONNEN.get(
            props.get("bron"), ("other", f"<unknown bron {props.get('bron')!r}>", "")
        )
        rows.append(
            {
                # provider fields, verbatim
                "jaar": props.get("jaar"),
                "periode": periode,
                "bron": props.get("bron"),
                "legenda": props.get("legenda"),
                "SELECTIE": props.get("SELECTIE"),
                # derived, documented in the module docstring
                "bron_slug": slug,
                "bron_label": bron_label,
                "unit": unit,
                "db_label": label,
                "db_min": lo,
                "db_max": hi,
                "api_id": feat.get("id"),
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84).to_crs(CRS_RD)
    gdf["area_m2"] = gdf.geometry.area

    stats = {
        "features_citywide": len(feats),
        "features_in_bbox": len(gdf),
        "features_without_geometry": n_no_geom,
        "download_bytes": len(payload),
        "seconds": round(time.time() - t0, 1),
        "extract_instant": instant,
        "jaar_values": sorted({int(v) for v in gdf["jaar"].dropna().unique()}),
        "raw_file": None
        if raw_path is None
        else {"file": raw_path.name, "bytes": raw_path.stat().st_size, "url": ITEMS_URL},
    }
    return gdf, stats
