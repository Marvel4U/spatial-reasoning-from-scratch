# Pilot snapshot — Amsterdam, De Pijp (built 11–12 Sep 2026)

Purpose: prove that a frozen, local, per-layer "world snapshot" of a real district can be built from open data with no keys, no cleaning, and a per-layer manifest, and that every layer is real, aligned, and visually plausible. Seven layers, all done. No tasks, no model, no training code.

Where: rig `~/Github/spatial_reasoning_LLM_artifact/` — code `worldsnap/` (~4.7k lines incl. long docstrings), data `data/amsterdam/de_pijp/` (418 MB incl. raw archives), overlays in `.../overlays/`, manifest `.../manifest.json`. CLI: `.venv/bin/python -m worldsnap.build --district de_pijp --layers <names>`. Bbox: lon 4.884–4.910, lat 52.345–52.361 (3.196 km²), EPSG:28992 storage.

## Layers

| Layer | Source (no key needed) | Licence | Vintage | Count | Verdict |
|---|---|---|---|---|---|
| buildings_3dbag | 3DBAG OGC API, LoD0 footprint + roof height percentiles, ground height, bouwjaar | CC BY 4.0 | v2023.10.08 | 9,775 (3,058/km²) | Clean; median height 14.9 m; footprint area matches BAG to 0.001 m² |
| streets_bgt | PDOK BGT OGC API, `wegdeel` + `ondersteunendwegdeel` polygons by function | CC0 | extract 2026-09-11 | 10,005 (voetpad 2,425; 0.49 km² sidewalk) | Sidewalks both sides of every street; **no width attribute** — derive from geometry later |
| transit_stops / transit_routes | GTFS NL (OVapi mirror of NDOV), Tue 2026-09-15 07–19 departures per stop | not stated (EU MMTIS open data) | feed v9569, valid to 2026-12-12 | 123 stops / 31 routes | Tram+metro geometry good; 14 bus routes have no shapes → stop-to-stop straight lines |
| trees | Gemeente Amsterdam DSO API, bomen/stamgegevens | "openbaar, tenzij anders aangegeven" | extract 2026-09-11 | 7,650 (2,393/km²) | 468 are stumps; trunk diameter 95% empty; year-0 sentinels |
| noise_road_lden / lnight | Geluidskaart Amsterdam 2021, vector isophone bands (road incl. tram) | Gemeente terms, free reuse | 2021 | 326 / 256 polygons | Bands disjoint; 68% of bbox ≥ 50 dB; metro absent (underground) |
| elevation_dtm / dsm | AHN5 0.5 m, kaartblad 25GN1 (PDOK "05m" tiles are AHN4 — survey corrected) | CC0 | 2023 capture | 3568×3585 px | DTM nodata under buildings (49%); nDSM vs 3DBAG height r=0.855, median |Δ| 0.36 m |
| pois_osm / pois_overture | Overpass one-off snapshot (archived JSON + query) / Overture places 2026-08-19.0 | ODbL / CDLA-Permissive-2.0 | 2026-09-11 / 2026-08-19 | 3,765 / 3,001 | Not merged. **`opening_hours` on 71% of cafés/restaurants/bars** — time-aware tasks are viable in NL. 34% name-match ≤30 m |

## Decisions collected during the build (Marvin's call; none block the next step)
1. Height field: 70th-percentile roof minus ground is canonical; 50p and max stored too. (proposed: keep)
2. District boundary: rectangle, edge-crossing objects kept whole. (proposed: keep)
3. Storage: GeoPackage for vector (QGIS-friendly), GeoTIFF for rasters. (proposed: keep)
4. Transit: one line per route = most-frequent shape variant; two files under one CLI layer. (proposed: keep)
5. Elevation: AHN source tiles (470 MB) not archived, HTTP headers + output sha256 recorded; derived nDSM not stored. (proposed: store nDSM as a product; don't archive tiles)
6. POIs: keep two sources unmerged; editorial 7-class mapping flagged. (proposed: keep; conflation is a task-generator concern)
7. Licensing: GTFS and tree register have no SPDX licence; ODbL share-alike on OSM — deferred per OPEN_DECISIONS (l).
8. Task-design notes: "tree" may need to exclude stumps; road noise includes trams; sidewalk width must be derived.

## Corrections to the survey
- `survey/raw/03`: PDOK `ahn/atom/{dtm,dsm}_05m` is AHN4, not AHN5. AHN5 is on basisdata.nl (hwh-ahn/AHN5), indexed by the national bladwijzer.
- `survey/raw/03`: maps.amsterdam.nl `geojson.php` 404s for bomen and GELUID_2021; use the DSO API for trees and `geojson_lnglat.php?KAARTLAAG=GELUID_2021&THEMA=geluid` for noise.
- Opening-hours coverage is now measured (71% for food & drink in De Pijp), no longer a blocker.

## Next (proposals)
1. Marvin reviews this file + overlays; optionally opens the GeoPackages in QGIS.
2. First derived layer: **shadow** — extrude 3DBAG footprints by height, cast shadows for 2–3 fixed sun positions (pysolar/astral), rasterise at 0.5 m, validate against the AHN5 nDSM-implied geometry. Verifiable by construction.
3. Second derived layer: sidewalk width from BGT polygons (medial axis / inscribed circle).
4. Then: the base-map renderer (the image the model will see) and the first task families.
