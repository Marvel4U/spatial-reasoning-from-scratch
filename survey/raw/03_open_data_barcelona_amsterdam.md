# Open geospatial data survey — Barcelona & Amsterdam

**Purpose:** catalogue snapshotable open data for a frozen local "world snapshot" of one city, so a small VLM can answer verifiable spatial questions via tools (proximity, opening hours, sun/shade, noise, transit frequency). Everything listed must be bulk-downloadable or API-harvestable under permissive terms — no live-API dependency inside the training loop.

**Survey date:** 11 Sep 2026.

**Confidence convention used throughout:**
- **[V]** = verified this session (portal/dataset page or authoritative doc seen).
- **[P]** = plausible / partially verified (dataset exists per search-result text or secondary source; exact format, cadence or license not read from the primary page).
- **[U]** = unverified — asserted from general knowledge, must be checked before relying on it.

Nothing below is invented; where I could not confirm a dataset I say so rather than filling the cell.

---

## 1. BARCELONA

Barcelona's ecosystem has **three tiers**: the city (Open Data BCN + CartoBCN + Geoportal BCN), the metropolitan authority (AMB), and the region (ICGC / Generalitat / IDE Catalunya). Plus the national cadastre (Catastro). Licenses are unusually clean — **CC BY 4.0 nearly everywhere**.

### 1.1 Base map / vector / cartography

| Dataset | Source / portal | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| OSM extract, Catalunya [V] | Geofabrik | .osm.pbf, .osm.bz2, shapefile | ODbL | Bulk HTTP, daily | Standard entry point. Cataluña sub-extract of Spain. | https://download.geofabrik.de/europe/spain.html |
| OSM extract, custom BBox [V] | BBBike | pbf, GeoJSON, GeoPackage, GeoParquet, shapefile, PMTiles | ODbL | Custom polygon/rect extract on demand | Best for "just the municipality" clip; GeoParquet output is convenient for a snapshot pipeline. | https://extract.bbbike.org/ |
| Overture Maps (base, transportation, buildings, places, addresses) [V] | Overture Maps Foundation | GeoParquet on S3/Azure; monthly release | buildings+transportation **ODbL**; **places CDLA-Permissive-2.0 + Apache-2.0** [V] | `s3://overturemaps-us-west-2/release/...`, DuckDB spatial, `overturemaps` CLI | Places theme being non-ODbL matters: you can build a POI tool without share-alike contagion. | https://docs.overturemaps.org/ , https://registry.opendata.aws/overture/ |
| CartoBCN — municipal topographic cartography 1:1,000 (3D), plus other scales [V] | Ajuntament de Barcelona | per-sheet vector products (DGN/DWG/SHP family) | CC BY 4.0 [V] | Web download portal, **no registration needed**, plus an **Atom feed** for updates [V] | Photogrammetric 1:1000 3D base from 2011–2013 flights + field revision + building records. The high-precision municipal base map — kerb lines, street furniture, building outlines. | https://w20.bcn.cat/cartobcn/ |
| CartoBCN product catalogue (machine-readable) [V] | Open Data BCN | CSV | CC BY 4.0 | Bulk CSV | Lists all CartoBCN products — useful to script the snapshot. | https://opendata-ajuntament.barcelona.cat/data/ca/dataset/cataleg-cartobcn |
| Topographic cartography 1:1,000 / 1:5,000, Catalunya [V] | ICGC | raster + vector, per sheet | CC BY 4.0 [V] | Download viewer (`visors.icgc.cat/appdownloads/`), WMS/WMTS/WFS | Regional fallback; metropolitan 1:1000 v2r2 sheets also exist via IDE Catalunya. | https://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Cartographic-geoinformation/Topographic-cartography-11000 |
| Geoportal BCN service catalogue (WMS/WFS) [P] | Ajuntament de Barcelona | OGC services | CC BY 4.0 | WMS/WFS | Entry point for city geoservices; useful where no bulk file exists. | https://www.barcelona.cat/geoportal/en/applications/main-applications/service-catalog |
| Open Data BCN catalogue itself [V] | Ajuntament de Barcelona | CSV/JSON catalogue, CKAN-style dataset URLs | CC BY 4.0 [V] | `/data/{ca,es,en}/dataset/...`, resource downloads | ~450+ datasets claimed [P]. The `cataleg-opendata` dataset is the machine-readable index — script your harvest from it. | https://opendata-ajuntament.barcelona.cat/data/ca/dataset/cataleg-opendata |

### 1.2 Buildings (footprints, heights, use, year)

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Catastro INSPIRE Buildings (BU) [V] | Dirección General del Catastro (ES) | GML (INSPIRE BU), per-municipality | Free reuse / open (Catastro reuse terms) [P] | **ATOM feed per municipality**, plus WFS | Footprints + **number of floors + year of construction + use** as cadastral attributes. Covers Barcelona. QGIS plugin `Spanish_Inspire_Catastral_Downloader` automates it. Geometry = envelope of above-ground volumes per parcel, **excludes overhangs/balconies** — note this if you reason about facades. | http://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.bu.atom.xml |
| CartoBCN 3D topographic 1:1000 [V] | Ajuntament | vector 3D | CC BY 4.0 | Bulk | Building outlines with height/level info from photogrammetry. Closest thing Barcelona has to a municipal LOD1 model. | https://w20.bcn.cat/cartobcn/ |
| AMB solar-map 3D city model (LiDAR-derived roof model) [P] | AMB / Barcelona Regional | web viewer; bulk availability **not confirmed** | unclear [U] | Viewer | Built with LiDAR-based 3D city modelling + cadastre. The underlying model may not be downloadable — treat the *derived* solar potential layer (§1.8) as the usable product. | https://amb.bcnregional.com/ |
| Overture buildings (heights where available) [V] | Overture | GeoParquet | ODbL | S3 bulk | Fills gaps; height attribution in ES is patchy [U]. | https://docs.overturemaps.org/guides/buildings/ |
| Building use / commercial ground-floor census | see §1.4 (cens de locals) | — | — | — | Barcelona's ground-floor premises census effectively gives building-level commercial use. | — |

> **No Catalan equivalent of 3D BAG was found.** ICGC publishes LiDAR and there is a metropolitan 3D model behind the AMB solar map, but I found **no** openly downloadable city-wide LOD2 building model for Barcelona. This is the biggest structural gap vs Amsterdam. **[V on absence-of-finding; not proof of absence]**

### 1.3 Street network / sidewalks / bike / speed

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| OSM highway network [V] | Geofabrik/BBBike | pbf | ODbL | bulk | One-way, `maxspeed`, `sidewalk=*`, `highway=pedestrian`, `cycleway=*`. Sidewalk *width* tags are sparse [U]. | — |
| Bicycle lanes of Barcelona (carril bici) [V] | Open Data BCN | SHP/ZIP, GeoJSON [P] | CC BY 4.0 | bulk | Includes direction of circulation; green routes, cycle paths, corridors. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/carril-bici |
| Bike lanes of other AMB municipalities [V] | Open Data BCN / AMB | vector | CC BY 4.0 | bulk | Metro-area extension. | https://opendata-ajuntament.barcelona.cat/data/ca/dataset/carrils-bici-municipis |
| Zones 30 — streets and zone polygons [V] | Open Data BCN | vector (streets + polygons) | CC BY 4.0 | bulk | Direct speed-limit layer; complements OSM `maxspeed`. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/zones30-carrers |
| Superilles / pedestrianised streets | Open Data BCN / Geoportal | — | CC BY 4.0 | — | **Not confirmed as a standalone open dataset.** The Superblock programme is well documented but I did not verify a downloadable superilla geometry layer. Fall back to OSM `highway=pedestrian` / `living_street`. [U] | — |
| Sidewalk / kerb geometry [P] | CartoBCN 1:1000 | vector | CC BY 4.0 | bulk | Kerb lines exist in the 1:1000 base map, so **sidewalk width is derivable** (kerb-to-facade), but is not published as a ready "sidewalk width" attribute. | https://w20.bcn.cat/cartobcn/ |
| Street sections / street axis (trams de carrer) [P] | Open Data BCN | CSV/vector | CC BY 4.0 | bulk | The noise-by-street-section dataset (§1.9) is keyed on official street sections, implying a published street-section reference layer. | — |

### 1.4 POIs / amenities / opening hours / terraces

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Cens d'activitats econòmiques en planta baixa** (ground-floor economic activity census) [V] | Open Data BCN | CSV (+ coordinates) | CC BY 4.0 | bulk; editions incl. 2016 and later | **Barcelona's killer POI dataset.** Every ground-floor premises in the city with activity classification, address, and active/vacant status. More complete and more authoritative than OSM for retail. Cadence: irregular, roughly per-census-wave [P]. | https://opendata-ajuntament.barcelona.cat/data/es/dataset/cens-activitats-comercials |
| Cens d'activitats i establiments (province) [V] | Diputació de Barcelona open data | CSV/API | open [P] | bulk | Provincial-level complement. | https://dadesobertes.diba.cat/datasets/cens-dactivitats-i-establiments |
| **Terrace authorisations** (terrasses ordinàries en espai públic) [V] | Open Data BCN | CSV/vector [P] | CC BY 4.0 | bulk | Authorised outdoor terraces, seasonal vs annual, per establishment. Genuinely unusual and well suited to "which café terrace has sun at 18:00". | https://opendata-ajuntament.barcelona.cat/data/es/dataset/terrasses-comercos-vigents |
| Opening hours | OSM `opening_hours` tag | pbf | ODbL | bulk | **Coverage not measured.** Almost certainly the weakest link for time-aware questions. The terrace ordinance gives *regulatory* closing times (generally 23:00 Sun–Thu, 00:00–01:00 Fri/Sat/eves, zone-dependent) [P] which can serve as a rule-based fallback. **Measure this before committing.** | — |
| Foursquare OS Places / Overture places | see §3 | Parquet | Apache-2.0 / CDLA-P-2.0 | bulk | Permissive POI layer; some hours attributes [U]. | — |
| Municipal facilities / equipaments [P] | Open Data BCN | CSV/GeoJSON | CC BY 4.0 | bulk | Schools, libraries, health centres, markets. | https://opendata-ajuntament.barcelona.cat/data/en/dataset |

### 1.5 Transit

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| TMB GTFS static (metro + bus) [V] | TMB developer portal | GTFS zip | TMB developer terms (**registration + API key required**) [V] | `https://api.tmb.cat/v1/static/datasets/gtfs.zip`, weekly refresh | Not a plain open download — you must register. Snapshot-friendly once fetched. | https://www.tmb.cat/en/tmb-app-and-other-apps/tools-for-developers |
| TMB GTFS archived versions [V] | Transitland / Mobility Database | GTFS zip | mirrors source | bulk, 100+ archived versions | **Practical escape hatch**: a dated, reproducible snapshot without key management, plus version history. | https://www.transit.land/feeds/f-sp3e-tmb , https://mobilitydatabase.org/feeds/gtfs/mdb-2359 |
| AMB GTFS realtime — bus [V] | Àrea Metropolitana de Barcelona open data | GTFS-RT | AMB open data terms [P] | API | Realtime is irrelevant to a frozen snapshot but useful to characterise actual headways if you harvest a window. | https://www.amb.cat/en/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/gtfs-real-time-bus-service/6332347/11692 |
| FGC GTFS static + GTFS-Realtime [V] | FGC open data / Dades Obertes Catalunya | GTFS zip, GTFS-RT | open (Generalitat open data terms) [P] | direct download, no key seen | Commuter rail into the city. Clean open portal. | https://dadesobertes.fgc.cat/ , https://analisi.transparenciacatalunya.cat/Transport/FGC-GTFS_zip/xzk9-2qky |
| Rodalies / TRAM GTFS | Generalitat / operators | GTFS | [U] | [U] | Exists in the Catalan open-data ecosystem per search results but I did not verify a direct GTFS download URL. | https://analisi.transparenciacatalunya.cat/ |
| Bicing station information + status [V] | Open Data BCN | GBFS-style JSON / CSV | CC BY 4.0 | API + bulk historical | `informacio-estacions-bicing` (static locations) + station-status; historical usage dataset also published. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/informacio-estacions-bicing |

**Frequency/headway reasoning:** derivable from GTFS `stop_times` — no separate frequency dataset needed.

### 1.6 Elevation / terrain / LiDAR

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **MDT 2×2 m** terrain elevation model [V] | ICGC | GeoTIFF, per sheet | **CC BY 4.0** [V] | Download app — select area, get ZIP of sheets | Derived from **2008–2011 LiDAR**. Barcelona has real relief (Montjuïc, Collserola, the Eixample gradient toward the sea), so terrain genuinely matters here. | https://www.icgc.cat/en/Data-and-products/Bessons-digitals-Elevacions/2x2-m-Terrain-elevation-model |
| MDT 5 m / other elevation products [P] | ICGC | GeoTIFF | CC BY 4.0 | download viewer | Coarser alternatives. | https://visors.icgc.cat/appdownloads/ |
| Historical + current LiDAR point clouds [V] | ICGC | **LAZ** | CC BY 4.0 | download viewer, per sheet | Raw point cloud — build your own DSM/nDSM and hence building heights and tree canopy. Important since there is no open LOD2 model. | https://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Digital-twins-Elevations/Historical-LiDAR-data |
| Open ICGC QGIS plugin [V] | ICGC | plugin | — | — | Scriptable access to ICGC downloads; good basis for a reproducible fetch script. | https://plugins.qgis.org/plugins/OpenICGC/ |

### 1.7 Land cover / land use / zoning

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Mapa de Cobertes del Sòl de Catalunya (MCSC) v1.0, 2019–2022 [V] | ICGC / IDE Catalunya | raster/vector | CC BY 4.0 [P] | IDE Catalunya catalogue | Regional land cover. | https://catalegs.ide.cat/geonetwork/srv/api/records/cobertes-sol-v1r0-2019-2022 |
| CatLC multiresolution land cover [V] | published dataset (Sci Data) | raster | open [P] | academic repository | Peer-reviewed Catalan land-cover dataset. | https://www.nature.com/articles/s41597-022-01674-y |
| Urban planning / zoning (PGM, qualificació urbanística) | Ajuntament / AMB | — | — | — | **Not verified.** Barcelona's planning geoportal exists but I did not confirm an open bulk zoning layer. [U] | — |
| Cadastral parcels [V] | Catastro INSPIRE (CP) | GML | open [P] | ATOM | Parcels alongside buildings. | http://www.catastro.minhap.es/INSPIRE/ |

### 1.8 Sun / shadow / solar

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Mapa solar de l'AMB** — PV + thermal potential per building/roof [V] | AMB / Barcelona Regional | web viewer; **bulk download not confirmed** | [U] | viewer | Built from LiDAR-derived 3D city model + cadastre. Per-roof annual energy potential. | https://amb.bcnregional.com/ , https://amb.bcnregional.com/visor_FV/index.html |
| Potencial de radiació solar fotovoltaica, AMB [V] | IDE Catalunya catalogue record | raster [P] | [U] — check the record | catalogue / possibly WMS + download | The most promising *downloadable* route to the solar-potential layer. **Verify license and download before relying on it.** | https://catalegs.ide.cat/geonetwork/geonetwork/api/records/potencial-radiacio-solar-fotovoltaica-area-metropolitana-barcelona |
| Atles de radiació solar a Catalunya [V] | ICAEN | PDF/monograph + data [P] | [U] | — | Regional climatological solar resource. | https://icaen.gencat.cat/ |
| Atles Climàtic Digital de Catalunya — potential radiation [V] | UAB / GRUMETS | raster | academic terms [U] | web | Terrain-driven potential-radiation grids. | https://opengis.grumets.cat/acdc/catala/rad_pot.htm |

> **Verdict for Barcelona:** street-level **shade must be computed**, not downloaded. Published products are *roof* solar potential, not ground/facade shadow. The ingredients exist (ICGC LiDAR → nDSM, or CartoBCN 3D heights) but you build the shadow model yourself. Third-party tools (ShadeMap, Shadowmap, the "Sunseekr" Barcelona bar-sun map) demonstrate feasibility from OSM + terrain but are **not** open datasets you can snapshot. [V]

### 1.9 Noise

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Strategic Noise Map — noise by street section (tramer)** [V] | Open Data BCN | CSV | CC BY 4.0 | bulk; editions 2009/2012/2017/**2022** | Lden/Lday/Levening/Lnight per street section per source (road, rail, industry). **Directly answers "how noisy is this spot" with an official number.** | https://opendata-ajuntament.barcelona.cat/data/en/dataset/tramer-mapa-estrategic-soroll |
| Noise contour maps (isòfones) [V] | Open Data BCN | **GeoPackage** + QML styles | CC BY 4.0 | bulk, 2022 edition confirmed | Isophone polygons, total and per-source, Lden and others. Best layer for spatial queries. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/isofones-mapa-estrategic-soroll |
| Raster noise maps [V] | Open Data BCN | raster | CC BY 4.0 | bulk | Continuous surface — easiest to sample at a point. | https://datos.gob.es/en/catalogo/l01080193-mapas-de-ruido-raster-del-mapa-estrategico-de-ruido-de-la-ciudad-de-barcelona |
| Noise maps by **facade** [V] | Open Data BCN | vector | CC BY 4.0 | bulk | Per-facade noise — extraordinarily good for "which side of this building is quieter". Rare. | https://datos.gob.es/en/catalogo/l01080193-mapas-de-ruido-de-fachadas-del-mapa-estrategico-de-ruido-de-la-ciudad-de-barcelona |
| Acoustic zoning / capacitat acústica [V] | Open Data BCN | CSV + polygons | CC BY 4.0 | bulk | Regulatory max levels per zone — a *rule* layer, deterministic and verifiable. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/capacitat-mapa-estrategic-soroll |
| Population exposed to noise levels [V] | Open Data BCN | tabular | CC BY 4.0 | bulk | END-mandated exposure statistics. | https://data.europa.eu/data/datasets/https-opendata-ajuntament-barcelona-cat-data-dataset-poblacio-exposada-mapa-estrategic-soroll |

**Barcelona's noise layer is the best single thing in this survey.** Four representations (section / contour / raster / facade) of the same EU-END-mandated model, all CC BY 4.0, all bulk, several vintages.

### 1.10 Greenery / trees

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Arbrat viari** (street trees) [V] | Open Data BCN | CSV with coordinates | CC BY 4.0 | bulk, updated regularly [P] | Individual street trees in tree pits, with species and attributes. Framed by the city as providing shade, noise protection, CO2 absorption. | https://opendata-ajuntament.barcelona.cat/data/ca/dataset/arbrat-viari |
| Arbrat zona / arbrat parcs [P] | Open Data BCN | CSV | CC BY 4.0 | bulk | Companion inventories for trees in green zones and parks. | https://opendata-ajuntament.barcelona.cat/data/en/dataset |
| Parks & green spaces [P] | Open Data BCN / OSM | vector | CC BY 4.0 / ODbL | bulk | — | — |
| NDVI | Sentinel-2 (§3) | raster | free & open | bulk | Compute yourself. | — |

### 1.11 Flow / counts / traffic

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Pedestrian counts / footfall | — | — | — | — | **Not found as an open dataset.** No published pedestrian-count or footfall layer for Barcelona was confirmed. [V on absence-of-finding] | — |
| Traffic intensity / IMD | Open Data BCN / Servei Català de Trànsit | [U] | [U] | [U] | Barcelona publishes traffic-related datasets and the Servei Català de Trànsit has an open-data section, but I did not verify a city street-level traffic-intensity layer. | https://transit.gencat.cat/ca/el_servei/dades-estadistiques/dades-obertes/ |
| Bicing usage history [V] | Open Data BCN | CSV | CC BY 4.0 | bulk | Station-usage time series — a genuine flow proxy. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/us-del-servei-bicing |

### 1.12 Air quality, population, tourism

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Air quality measuring stations (XVPCA) [V] | Open Data BCN | CSV | CC BY 4.0 | bulk | Station locations/codes/characteristics. | https://opendata-ajuntament.barcelona.cat/data/en/dataset/qualitat-aire-estacions-bcn |
| Air quality measurements O3/NO2/PM10, 2018–2026 [V] | Open Data BCN | CSV per year/month | CC BY 4.0 | bulk | Point time series, not a surface. A modelled AQ surface exists at AMB/Barcelona Regional but downloadability unconfirmed [U]. | https://opendata-ajuntament.barcelona.cat/ca/novetat-qualitat-aire |
| Population / census [V] | Idescat, INE, Barcelona Datastore | CSV/API | open (CC-BY-ish) [P] | bulk | Fine-grained statistics by *secció censal* and neighbourhood. **No INE 1 km/100 m population grid was verified** — no Spanish equivalent of CBS's 100 m squares was found this session. [V on absence-of-finding] | https://idescat.cat/pub/?id=censph , https://portaldades.ajuntament.barcelona.cat/ |
| Tourism pressure | Open Data BCN | — | — | — | Barcelona publishes tourist-accommodation (HUT licences) and tourism statistics [U] — not verified this session, but a plausible city-specific hook given Barcelona's over-tourism politics. | — |

### 1.13 Imagery

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Territorial orthophoto (25 cm and finer) [V] | ICGC | GeoTIFF/ECW per sheet | **CC BY 4.0** [V] | Download viewer, WMS/WMTS | Excellent, regularly re-flown, permissively licensed. | https://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Image/Territorial-Orthophoto |
| Municipal orthophoto [P] | CartoBCN | raster | CC BY 4.0 | bulk | City-level imagery in the CartoBCN product list. | https://w20.bcn.cat/cartobcn/ |
| Street-level | Mapillary (§3) | JPEG + API | CC BY-SA 4.0 | API | Barcelona Mapillary coverage density **not measured** [U]. | — |
| Satellite | Sentinel-2 (§3) | SAFE/COG | free & open | bulk | — | — |

---

## 2. AMSTERDAM

The Dutch stack is **structurally different and, for this project, structurally better**: national base registries (BAG, BGT, BRT, AHN) are CC0/public-domain-ish, nationally consistent, machine-servable via PDOK, and *designed* to be joined. The city layer (maps.amsterdam.nl, data.amsterdam.nl) sits on top.

### 2.1 Base map / vector

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| OSM extract, Netherlands [V] | Geofabrik / BBBike | pbf (~1.4 GB NL at BBBike) | ODbL | bulk, daily | NL OSM is among the highest-quality OSM in the world; much of BAG/BGT has been imported. | https://download.geofabrik.de/europe/netherlands.html , https://download3.bbbike.org/osm/region/europe/netherlands/ |
| Overture (all themes) [V] | Overture | GeoParquet | ODbL / CDLA-P-2.0 (places) | S3 bulk, monthly | — | https://docs.overturemaps.org/ |
| **BGT** — Basisregistratie Grootschalige Topografie [V] | PDOK / Kadaster | **ZIP (GML), national or tiled 2×2…64×64 km**; also OGC API Features | **open, commonly CC0 / public domain** [P] | **Atom download service**, WFS, OGC API, vector tiles | The national large-scale base map. Contains `wegdeel` polygons typed by function — **roadway vs sidewalk (voetpad) vs cycleway as actual polygons with real geometry**. The killer layer for sidewalk-width reasoning. QGIS `bgt_downloader` plugin exists. | https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-grootschalige-topografie-bgt- |
| **BRT / TOPNL** [V] | PDOK | **GeoPackage**, GML; current + historical years | open [P] | Atom download | Smaller-scale topographic base. | https://www.pdok.nl/introductie/-/article/basisregistratie-topografie-brt-topnl |
| PDOK datasets index [V] | PDOK | — | mostly open/CC0 [P] | WMS/WMTS/WFS/OGC API/vector tiles/Atom | Single national entry point; most datasets streamable without download. | https://www.pdok.nl/datasets |
| maps.amsterdam.nl Open Geodata [V] | Gemeente Amsterdam | **GeoJSON, CSV, Excel, Esri shapefile, MIF/MID, WMS, WFS** [V] | city usage terms (must accept; **not** a standard CC licence) [V] | per-map download | 8 thematic categories: urbanity/housing, energy/sustainability, green/nature, traffic/infrastructure, history/architecture, neighbourhood/amenities, areas. **Portal explicitly states historical datasets are not preserved.** Snapshot early, record the date. | https://maps.amsterdam.nl/open_geodata/ |
| data.amsterdam.nl + Datapunt API [V] | Gemeente Amsterdam | REST / WFS / MVT | open, **no auth required** for many datasets [V] | `api.data.amsterdam.nl/v1/` | Well-documented programmatic API with CSV/GeoJSON export per dataset. | https://data.amsterdam.nl/ , https://api.data.amsterdam.nl/v1/docs/ |

### 2.2 Buildings — the standout

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **3DBAG** — LoD1.2 / LoD1.3 / **LoD2.2** for all ~10 M NL buildings [V] | TU Delft 3D geoinformation | **CityJSON, GeoPackage (per tile + one national SOZip dump), Wavefront OBJ, WMS, WFS** [V] | **CC BY 4.0** [V] | tiled downloads + national dump + web services + `3dbag-scripts` helper | Reconstructed from BAG footprints + AHN LiDAR. Roof-plane heights (`h_dak_*`) in LoD1.3/2.2; 2D roof-surface projections with height references also published. Validated with val3dity. **The single best asset in either city** — real roof geometry means true shadows, facade orientation, sun exposure. | https://3dbag.nl/en/download , https://docs.3dbag.nl/en/ |
| **BAG** — footprints, addresses, **bouwjaar** (construction year), **gebruiksdoel** (use purpose), floor area [V] | PDOK / Kadaster | GML/GeoPackage, WFS, OGC API | open / CC0-ish [P] | Atom + services | Authoritative national register. Use purpose (residential/retail/office/…) and construction year come free. | https://www.pdok.nl/datasets |
| Building age map (bouwjaren) [V] | maps.amsterdam.nl | GeoJSON/CSV/SHP | city terms | bulk | City-styled derivative of BAG. | https://maps.amsterdam.nl/open_geodata/ |
| Non-residential functions map [V] | maps.amsterdam.nl | GeoJSON/CSV/SHP | city terms | bulk | Building-level non-residential use. | https://maps.amsterdam.nl/open_geodata/ |

### 2.3 Street network / sidewalks / bike / speed

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **BGT `wegdeel`** [V] | PDOK | polygons | open [P] | Atom / OGC API | Sidewalk (`voetpad`), roadway (`rijbaan`), cycleway, tram bed as **true polygons** → **real sidewalk and street widths are measurable**, not tag-guessed. Best-in-class for "which side of the street" questions. | https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-grootschalige-topografie-bgt- |
| **NWB Wegen** (national road register) [V] | PDOK / Rijkswaterstaat | vector | open [P] | Atom / services | Road centrelines, fed by BAG + BGT + BRT. | https://www.pdok.nl/introductie/-/article/nationaal-wegen-bestand-nwb-wegen |
| **30 km/u in de stad** — city-wide speed-limit map [V] | maps.amsterdam.nl | GeoJSON/CSV/SHP + WFS | city terms | bulk | Amsterdam went largely 30 km/h; this is the authoritative speed layer. | https://maps.amsterdam.nl/30km/ |
| **Plusnet & Hoofdnet Fiets** (cycle network hierarchy) [V] | maps.amsterdam.nl / data.overheid.nl | GeoJSON/SHP | city terms | bulk | ~400 m grid cycle network with infrastructure status. Also `fietskruispunten` (cycle junctions) and `plushoofdnetten` (all modes). | https://maps.amsterdam.nl/fietsnetten/ , https://data.overheid.nl/dataset/trrngyt7yfedkq |
| OSM (one-way, pedestrian zones, maxspeed) [V] | Geofabrik | pbf | ODbL | bulk | NL OSM tagging is dense and reliable. | — |

### 2.4 POIs / amenities / opening hours

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| OSM POIs + `opening_hours` | Geofabrik | pbf | ODbL | bulk | NL `opening_hours` coverage is **likely better than ES** but **unmeasured** [U]. Measure before committing. | — |
| Non-residential functions [V] | maps.amsterdam.nl | GeoJSON/CSV | city terms | bulk | Function per building/unit — the closest Dutch analogue to Barcelona's ground-floor census, but **much less rich** (no per-premises commercial activity census found). [V on absence-of-finding] | https://maps.amsterdam.nl/open_geodata/ |
| Amenities: sports venues, markets, waste containers, etc. [V] | maps.amsterdam.nl ("Buurt & Voorzieningen") | GeoJSON/CSV/SHP | city terms | bulk | Many small municipal POI layers. | https://maps.amsterdam.nl/open_geodata/ |
| KvK business register | Kamer van Koophandel | — | **not open** [U] | paid/API | The NL business register is *not* open data — a real gap vs Barcelona's cens d'activitats. | — |
| Foursquare OS Places / Overture places (§3) | — | Parquet | permissive | bulk | Primary permissive POI fallback. | — |
| Terrace licences | Gemeente Amsterdam | — | — | — | **Not found.** No Amsterdam equivalent of Barcelona's terrace-authorisation dataset was confirmed. [V on absence-of-finding] | — |

### 2.5 Transit

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **GTFS NL (all operators incl. GVB)** [V] | OVapi / OpenOV | **single national GTFS zip, refreshed daily** | open, **no registration** [V] | `http://gtfs.ovapi.nl/openov-nl/gtfs-openov-nl.zip` and `https://gtfs.ovapi.nl/nl/` | ~2,677 routes / 56,626 stops / 38 operators nationally. Filter to GVB + regional operators. **Far friendlier than TMB's key-gated feed.** | https://gtfs.ovapi.nl/nl/ |
| GTFS-RT NL [V] | OVapi | GTFS-RT | open | API | — | https://gtfs.ovapi.nl/ |
| NDOV loket (raw operator feeds) [P] | NDOV | BISON/KV + GTFS | open | bulk | The national clearinghouse; OVapi is the convenient derivative. | https://ndovloket.nl/ |
| Donkey Republic **GBFS** feed, Amsterdam [P] | Donkey Republic | GBFS JSON (`station_information`, `station_status`) | public GBFS feed, no registration [P] | live JSON | Primary Amsterdam bike-share. | https://www.donkey.bike/cities/amsterdam |
| OV-fiets locations | NS | [U] | [U] | [U] | Exists; open bulk availability not verified. | — |
| Tram/metro lines map [V] | maps.amsterdam.nl (Verkeer & Infrastructuur) | GeoJSON/SHP | city terms | bulk | City-side geometry complementing GTFS. | https://maps.amsterdam.nl/open_geodata/ |

### 2.6 Elevation / terrain / LiDAR — the other standout

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **AHN5 DSM 0.5 m** [V] | AHN / PDOK | GeoTIFF raster tiles | open / public domain [P] | **Atom**: `https://service.pdok.nl/rws/ahn/atom/dsm_05m.xml` | Surveyed **2023–2024**. **AHN5 DSM uses max value per cell** (vs AHN4's weighted average) — better for building/tree tops; note the discontinuity if you mix versions. | https://www.ahn.nl/ |
| **AHN5 DTM 0.5 m** [V] | AHN / PDOK | GeoTIFF | open [P] | `https://service.pdok.nl/rws/ahn/atom/dtm_05m.xml` | Ground surface (squared-IDW resample of ground-classified points). | https://www.pdok.nl/atom-downloadservices/-/article/actueel-hoogtebestand-nederland-ahn |
| AHN point clouds (LAZ) [V] | AHN dataroom | LAZ | open [P] | bulk | Raw point cloud incl. tree canopy. | https://www.ahn.nl/dataroom |
| AHN WMS/WMTS [V] | PDOK | services | open | streaming | — | https://www.pdok.nl/ |

Amsterdam is flat, so *terrain* barely matters — AHN's value here is **surface** height: canopy, bridges, quay walls, and the LiDAR that produced 3DBAG.

### 2.7 Sun / shadow / solar

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Zonatlas** — per-roof solar suitability, NL-wide [V] | Zonatlas (operator; public viewer) | web viewer; **"data producten" licensed separately** [V] | **not open** [P] | viewer free; bulk unclear | Built from AHN + cadastre into a 3D model at ~0.5 m² accuracy, and **explicitly models shadow from dormers, trees and neighbouring buildings**. Methodologically exactly what this project needs, but **treat bulk access as unconfirmed / likely not free**. | https://www.zonatlas.nl/start/toolkits/data-producten/ |
| Zonprojecten — 200 largest roofs [V] | maps.amsterdam.nl | GeoJSON/CSV | city terms | bulk | Small, derived from RVO + 2018 Zonatlas. Not city-wide. | https://maps.amsterdam.nl/zonprojecten/ |
| Solar panels installed (zonnepanelen) [V] | maps.amsterdam.nl / data.overheid.nl | GeoJSON/CSV | open [P] | bulk | Where panels actually are. | https://data.overheid.nl/en/dataset/9yuylj2fqs8i0a |
| Zonnedaken & erfgoed (roof inventory) [V] | maps.amsterdam.nl | GeoJSON | city terms | bulk | Roof-type inventory for heritage areas. | https://maps.amsterdam.nl/zonnedaken_erfgoed/ |

> **Verdict for Amsterdam:** as in Barcelona, ground/facade shadow must be **computed** — but here you have **3DBAG LoD2.2 (CC BY 4.0) + AHN5 DSM 0.5 m**, which is close to ideal input. Shadow computation in Amsterdam is a genuinely solvable, verifiable-by-construction problem. **This is the strongest single argument for choosing Amsterdam.**

### 2.8 Noise

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Geluidskaart Amsterdam 2021** [V] | maps.amsterdam.nl | **GeoJSON (lnglat) + SLD styles**; download links exposed via the map's `Legend.csv` [V] | city terms | bulk (slightly awkward — links live in the legend CSV) | Calculated Lden over 24 h for main traffic sources. EU END-compliant, 5-yearly cycle. | https://maps.amsterdam.nl/geluid/ |
| Geluidscontouren [V] | data.overheid.nl | vector | open [P] | bulk | National / other noise-contour datasets. | https://data.overheid.nl/dataset/wo-7londbtecwg |
| Schiphol aircraft noise (Lden) [V] | Atlas Leefomgeving / RIVM | raster/vector | open [P] | bulk | **Amsterdam-specific and notable**: aircraft noise is a real, spatially structured phenomenon with no Barcelona analogue of similar dominance. | https://www.atlasleefomgeving.nl/geluid-van-schiphol-lden |
| Actieplan geluid technical annex [V] | DBvision / Gemeente | PDF | — | — | Methodology reference. | https://www.dbvision.nl/rapporten/Actieplan_geluid_Amsterdam_2019_2023.pdf |

Good — but **less rich than Barcelona's**: essentially one representation, no per-facade layer found, no confirmed raster.

### 2.9 Greenery / trees

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Bomen** — municipal tree register [V] | maps.amsterdam.nl (WMS from Datapunt, backed by the GISIB asset-management system) | GeoJSON/CSV/SHP + WMS/WFS | city terms | bulk + services | Every municipally managed tree, live from asset management. Species and presumably size/age attributes [P]. | https://maps.amsterdam.nl/bomen/ |
| Bijzondere bomen (special/protected trees) + management advice [V] | maps.amsterdam.nl | GeoJSON | city terms | bulk (per-tree info; downloads incomplete for some trees) | — | https://maps.amsterdam.nl/bomen_bijzonder/ |
| Ecological structure, birds, bats, bat passages [V] | maps.amsterdam.nl "Groen & Natuur" | GeoJSON/SHP | city terms | bulk | Unusually deep ecology layers. | https://maps.amsterdam.nl/open_geodata/ |
| Tree canopy height | AHN5 DSM − DTM | raster | open | computed | Trivially derivable at 0.5 m. Barcelona can do this too from ICGC LiDAR but at coarser and older vintage. | — |
| NDVI | Sentinel-2 | raster | free & open | bulk | — | — |

### 2.10 Pedestrian / traffic flow — Amsterdam's other distinctive asset

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **CMSA — Crowd Monitoring Systeem Amsterdam** [V] | data.amsterdam.nl Datapunt API | **CSV, GeoJSON via WFS; MVT tiles** | **public data, no authorisation required** [V] | `api.data.amsterdam.nl/v1/.../cmsa`, WFS + MVT | Permanent footfall sensors/cameras (De Wallen, Centraal Station bus platform, ferry terminal, …), privacy-preserving counts/heatmaps. **A real, official pedestrian-flow dataset — Barcelona has no verified equivalent.** Coverage is hotspot-limited, not city-wide. | https://api.data.amsterdam.nl/v1/docs/wfs-datasets/cmsa.html , https://api.data.amsterdam.nl/v1/mvt/cmsa/ |
| Parking pressure (parkeerdruk) [V] | maps.amsterdam.nl | GeoJSON/CSV | city terms | bulk | — | https://maps.amsterdam.nl/open_geodata/ |
| Traffic lights, cycle junctions [V] | maps.amsterdam.nl | GeoJSON | city terms | bulk | — | https://maps.amsterdam.nl/ |
| Verkeersintensiteit (car traffic volumes) | Gemeente Amsterdam | [U] | [U] | [U] | data.overheid.nl carries an open *data request* for Amsterdam traffic data, which is evidence it is **not** straightforwardly open. Commercial sources (CityTraffic, HERE) exist but are not snapshotable under permissive terms. [V on the data-request finding] | https://data.overheid.nl/en/community/datarequest/traffic-data-amsterdam |

### 2.11 Air quality, population, land use, other

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| **Luchtmeetnet** air quality (NO2, PM10, PM2.5, O3) [V] | RIVM | OpenAPI JSON; **bulk hourly history at data.rivm.nl** | open [P] | API + bulk | Well-documented OpenAPI incl. forecasts. Bulk historical is the snapshot-friendly route. | https://api-docs.luchtmeetnet.nl/ , https://data.rivm.nl/data/luchtmeetnet/ |
| Hollandse Luchten citizen sensors [V] | Hollandse Luchten | open data | open [P] | bulk | Dense low-cost sensor network in the Amsterdam region — much finer spatial grain than official stations. | https://hollandse-luchten.org/data/ |
| **CBS Vierkantstatistieken 100 m** [V] | CBS / PDOK | GeoPackage etc. via **Atom**; also CBS direct download | open [P] | bulk | **Population + demographics + housing + energy + income + social security + proximity-to-services + density on a 100 m grid** (2022–2024 editions; 500 m variant too). Directly usable as a verifiable answer source. Barcelona has **no verified equivalent grid.** | https://www.pdok.nl/atom-downloadservices/-/article/cbs-vierkantstatistieken , https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/kaart-van-100-meter-bij-100-meter-met-statistieken |
| Zoning / bestemmingsplannen (Omgevingswet DSO) | Ruimtelijke plannen / DSO | [U] | open [P] | [U] | NL planning data must be published digitally and is generally open, but I did not verify the current post-Omgevingswet download route. | — |
| Soil quality, climate impact (heat stress, flooding), gas-free areas [V] | maps.amsterdam.nl "Energie & Duurzaamheid" | GeoJSON/CSV | city terms | bulk | Heat-stress layers are relevant to sun/shade questions. | https://maps.amsterdam.nl/open_geodata/ |
| Monuments, protected cityscapes, WWII sites, street art [V] | maps.amsterdam.nl "Historie & Architectuur" | GeoJSON | city terms | bulk | Rich and city-specific. | https://maps.amsterdam.nl/open_geodata/ |
| Tourism pressure | — | — | — | — | Not verified as a distinct open dataset; CMSA footfall in De Wallen is the de facto proxy. [U] | — |

### 2.12 Imagery

| Dataset | Source | Format | License | Access | Notes | Link |
|---|---|---|---|---|---|---|
| Luchtfoto (aerial ortho) [V] | PDOK | WMTS/WMS; download routes vary | open [P] | services + downloads | National ortho programme, high resolution (sub-10 cm for some products [U]). | https://www.pdok.nl/datasets |
| Street-level | Mapillary (§3) | JPEG + API | CC BY-SA 4.0 | API | Amsterdam Mapillary coverage **not measured** [U], though NL is a strong Mapillary country [U]. | — |
| Satellite | Sentinel-2 (§3) | SAFE/COG | free & open | bulk | — | — |

---

## 3. CROSS-CITY / GLOBAL SOURCES

| Source | Coverage | Layers | Format | License | Access | Notes |
|---|---|---|---|---|---|---|
| **OpenStreetMap** (Geofabrik, BBBike) [V] | both | roads, POIs, buildings, land use, opening_hours, transit | pbf / GeoJSON / GeoParquet / GPKG | **ODbL** (attribution + share-alike on derived databases) | bulk, daily; BBBike custom polygons | Backbone. ODbL affects derived *databases* — fine for a research artifact with attribution, but relevant if you publish the snapshot. |
| **Overture Maps** [V] | both | base, transportation, buildings, places, addresses, divisions | GeoParquet (cloud-native), monthly | **buildings/transportation ODbL; places CDLA-Permissive-2.0 + Apache-2.0** | S3/Azure, DuckDB, `overturemaps` CLI, GEE | The places-under-CDLA split is the key licensing fact: a permissively licensed POI layer. |
| **Foursquare OS Places** [V] | both | 100 M+ global POIs, 22 core attributes | **Parquet** on S3 / HuggingFace / Iceberg | **Apache-2.0** (most permissive POI option) | `hf://datasets/foursquare/fsq-os-places/release/dt=YYYY-MM-DD/...`, Places Portal + Iceberg (DuckDB/Spark/PyIceberg) | Monthly releases with dated partitions → **naturally snapshot-versioned**. Best license/effort ratio for POIs. |
| **Copernicus DEM (GLO-30 / GLO-90)** [P] | both | global DSM 30 m | GeoTIFF | free & open (Copernicus terms; attribution) | AWS Open Data, Copernicus portal | **Far too coarse** vs ICGC 2 m / AHN 0.5 m. Fallback or generalisation experiments only. |
| **GHSL — Global Human Settlement Layer, P2023** [V] | both | GHS-BUILT-S (surface), **GHS-BUILT-H (height)**, GHS-BUILT-V (volume), GHS-BUILT-C, **GHS-POP**, GHS-SMOD; 1975–2020 5-yearly + 2025/2030 projections | GeoTIFF, several resolutions/CRS | **fully open and free**, reuse with source acknowledgement (EC JRC) | direct download portal, HDX, GEE | Gives *comparable* built-up/population/height across both cities — useful as a city-agnostic normalisation layer. Coarse (10 m–1 km depending on product). |
| **Mapillary** [V] | both | crowd-sourced street-level imagery + derived map features | JPEG + API (incl. image radius search) | **CC BY-SA 4.0** (some content CC BY-NC-SA) | API (free, key required) | The only viable open street-level imagery. **Share-alike plus the NC-licensed subset are a real constraint if imagery enters a released training set.** Per-city coverage unmeasured. |
| **Sentinel-2** (Copernicus Data Space Ecosystem) [V] | both | 10–60 m multispectral, full archive | SAFE / COG; also AWS COG registry | **free and open** to all users incl. commercial | CDSE APIs (free with quotas; **large-scale download attracts commercial conditions**), AWS Open Data | NDVI, seasonal greenness, built-up change. The quota note matters if you bulk-download. |
| **Microsoft Global ML Building Footprints** [V] | both | ML-derived footprints | GeoJSONL | **inconsistently documented — ODbL in some places, CDLA-Permissive-2.0 in the GitHub repo** [V on the inconsistency] | GitHub / Azure | **Redundant in both cities** (BAG and Catastro are better). Resolve the license question before any use. |
| **Google Open Buildings** [P] | effectively neither | ML footprints | CSV/Parquet | CC BY 4.0 | GCS / GEE | Africa / South Asia / LatAm focus — **does not usefully cover Europe**. Skip. |
| **VIDA Google–Microsoft combined buildings** [V] | both | merged global footprints | GeoParquet | **dual: CC BY 4.0 or ODbL — user picks** | source.coop | The dual-licence option is handy if you want to avoid ODbL. Still redundant here. |
| **Wikidata** [U] | both | entities, coordinates, admin hierarchy, notable POIs | JSON/RDF dumps | **CC0** | dumps.wikimedia.org | CC0 makes it a safe glue layer for entity names/aliases in NL/CA/ES — useful for multilingual question generation. Not verified this session. |
| **OpenCellID** [U] | both | cell tower locations | CSV | CC BY-SA 4.0, registration required | bulk | Not verified this session; marginal relevance. |
| **INSPIRE / data.europa.eu** [V] | both | harvested metadata for both cities' datasets | — | varies | catalogue | Useful cross-check that a city dataset exists and is formally open (both cities' datasets appear there). |
| **Transitland / Mobility Database** [V] | both | archived, dated GTFS feeds | GTFS zip | mirrors source licence | bulk | **Underrated for this project**: reproducible, dated transit snapshots without operator key management. Essential for Barcelona (TMB key-gated), convenient for NL. |

### License compatibility summary
- **Safest permissive core:** PDOK / BAG / BGT / AHN (open, CC0-ish), ICGC (CC BY 4.0), Open Data BCN (CC BY 4.0), 3DBAG (CC BY 4.0), GHSL (open + attribution), Sentinel-2 (free & open), Foursquare OS Places (Apache-2.0), Overture places (CDLA-P-2.0).
- **Share-alike, handle deliberately:** OSM and Overture buildings+transportation (**ODbL** — share-alike on derived *databases*), Mapillary (**CC BY-SA 4.0**, with a CC BY-NC-SA subset).
- **Ambiguous / verify:** Zonatlas (bulk likely commercial), Microsoft footprints (conflicting ODbL vs CDLA statements), maps.amsterdam.nl (bespoke city usage terms rather than a named CC licence, an accept-terms gate, and an explicit statement that **historical datasets are not preserved**), AMB solar map.
- **Practical rule for this artifact:** if you release the world snapshot publicly, the ODbL components force share-alike on the released database. If you release only *derived question–answer pairs* plus the fetch scripts, the constraint is much lighter. Worth an explicit decision, not a silent one.

---

## 4. QUALITY & GAPS ASSESSMENT

### 4.1 Barcelona

**Unusually rich**
- **Noise.** Best-in-survey. Four parallel representations of the EU-END strategic noise map — per street section, isophone contours (GeoPackage), raster, **and per-facade** — plus acoustic zoning (regulatory limits) and exposed-population statistics; all CC BY 4.0, all bulk, multiple vintages (2009/2012/2017/2022). Per-facade noise is a rare and directly VLM-answerable layer ("which side of this building is quieter").
- **Ground-floor commercial census (cens d'activitats).** An official, exhaustive per-premises register of ground-floor economic activity. No Dutch equivalent found. Makes "what kind of shop is at this address" verifiable against an authority, not just OSM.
- **Terrace authorisations.** Which establishments may put tables on the street, seasonal vs annual. Combined with the terrace ordinance's fixed closing hours this creates a distinctive and *rule-checkable* question family ("which terrace is in sun and still open at 22:00").
- **Licensing.** CC BY 4.0 almost uniformly across city (Open Data BCN, CartoBCN) and region (ICGC). Clean.
- **Terrain actually matters.** Real relief (Montjuïc, Collserola, the Eixample gradient) makes elevation questions non-trivial, unlike Amsterdam.
- **Orthophoto + LiDAR from ICGC** under CC BY 4.0, with good download tooling (viewer + QGIS plugin).
- **City morphology.** The Eixample grid with chamfered corners is a *gift* for spatial reasoning: regular, orientable, with unambiguous NE/SE/SW/NW facade geometry — sun/shade questions have clean verifiable answers and the grid regularity makes the task learnable.

**Thin or missing**
- **No open LOD2 (or even reliable LOD1) 3D building model.** The biggest gap. A metropolitan 3D model exists behind the AMB solar map but its bulk availability and licence are unconfirmed. You would have to **build heights yourself** from ICGC LiDAR (the 2 m MDT rests on **2008–2011** LiDAR — over a decade old) or CartoBCN's 2011–2013 photogrammetry, or use cadastral floor counts as a proxy. That is real, non-trivial work before any sun/shadow tool can exist.
- **Transit feed is key-gated.** TMB requires registration; the clean path is Transitland / Mobility Database archives.
- **No pedestrian-count / footfall dataset found.**
- **No verified population grid.** Spanish statistics come by census section, not a CBS-style 100 m grid.
- **Traffic intensity unverified** at street level.
- **Opening hours** depend entirely on OSM coverage, which is unmeasured and likely sparser than in NL [U].
- **Multilingual friction:** catalogue and field names in Catalan/Spanish with inconsistent English coverage — a tractable but real data-engineering tax.

### 4.2 Amsterdam

**Unusually rich**
- **3DBAG (CC BY 4.0, LoD2.2, CityJSON/GPKG/OBJ/WFS, national dump + tiles).** Decisive. Real roof geometry for every building means **shadow, sun exposure, facade orientation and sky-view factor are computable from open data under a permissive licence** — precisely what "which side gets afternoon sun" needs, and a verifiable-by-construction reward signal.
- **AHN5 DSM/DTM at 0.5 m** (2023–2024 survey) plus raw LAZ, via clean PDOK Atom feeds. Canopy height, quay walls, bridges all derivable.
- **BGT `wegdeel` polygons.** Sidewalks, roadways and cycleways as **true polygons**, nationally consistent. **Real measured sidewalk and street widths** rather than OSM's sparse `width`/`sidewalk` tags. The best street-geometry layer in the survey.
- **BAG.** Authoritative footprints + construction year + use purpose + floor area, open, joinable by design to 3DBAG.
- **CBS Vierkantstatistieken 100 m.** Population, demographics, housing, energy, income, proximity-to-services on a 100 m grid. Directly verifiable answers about density and amenity access.
- **CMSA footfall data**, public and unauthenticated via a documented API with WFS/MVT/CSV/GeoJSON export. An actual pedestrian-flow layer, albeit hotspot-only.
- **GTFS with no registration** (OVapi national feed, daily) plus GTFS-RT.
- **Speed limits as an authoritative city layer** (the 30 km/u map) plus a formal cycle-network hierarchy (Plusnet/Hoofdnet).
- **Schiphol aircraft noise** — a distinctive, spatially structured phenomenon with no Barcelona analogue.
- **Documented, modern APIs.** `api.data.amsterdam.nl/v1/` and PDOK's OGC API / Atom / vector-tile stack are pleasant to script against; nearly everything national is CC0-ish.

**Thin or missing**
- **No open business register.** KvK is not open data; the "non-residential functions" map is much coarser than Barcelona's per-premises activity census. Weakens POI-authority questions — you fall back to OSM / Foursquare / Overture.
- **No terrace-licence dataset found.**
- **Noise is thinner than Barcelona's**: one Geluidskaart 2021 GeoJSON (with download links awkwardly buried in the map's legend CSV), no per-facade layer, no confirmed raster.
- **Solar/shade products are not open**: Zonatlas is methodologically perfect but its data products appear to be licensed/commercial. You compute shade yourself — which, given 3DBAG + AHN5, is entirely fine.
- **Car traffic intensity is not clearly open** (an outstanding data request on data.overheid.nl is evidence against easy availability).
- **maps.amsterdam.nl licensing is a bespoke accept-terms regime**, not a named CC licence, and the portal **explicitly does not preserve historical datasets** — so anything you want must be snapshotted now, with the snapshot date recorded.
- **Flat terrain** removes an entire question family (elevation/slope) — arguably a narrowing of task diversity.
- **Canal geometry is a hidden difficulty**: "within 300 m" by straight line is frequently wrong in Amsterdam because bridges are sparse, so network distance and Euclidean distance diverge sharply. This is either a bug or a **feature** — it makes naive distance heuristics fail and rewards genuine tool use.

### 4.3 Head-to-head, for this specific artifact

| Criterion | Barcelona | Amsterdam |
|---|---|---|
| 3D buildings / heights (→ sun & shade) | **weak** — no open LOD2; build from 2008–2013-vintage LiDAR/photogrammetry | **outstanding** — 3DBAG LoD2.2, CC BY 4.0 |
| Elevation | very good (ICGC 2 m, CC BY) but dated | outstanding (AHN5 0.5 m, 2023–24) — though the city is flat |
| Sidewalk / street geometry | derivable from CartoBCN kerbs | **outstanding** — BGT polygons, measured widths |
| POIs with authority | **outstanding** — cens d'activitats + terrace licences | weak — no open business register |
| Opening hours | OSM-only, unmeasured, likely sparse | OSM-only, unmeasured, likely better |
| Noise | **outstanding** — 4 representations incl. per-facade | good — single contour layer |
| Transit | good but key-gated (use Transitland archives) | **outstanding** — open national GTFS, no key |
| Pedestrian flow | not found | **available** (CMSA, hotspot coverage) |
| Population grid | not found | **outstanding** — CBS 100 m |
| Licensing cleanliness | **outstanding** — CC BY 4.0 nearly everywhere | very good nationally (CC0-ish); city layer has bespoke terms and no history preservation |
| Terrain-based questions | **rich** (real hills) | impoverished (flat) |
| Geometry that helps the task | **Eixample grid** — regular, orientable, chamfered corners | canals — Euclidean vs network distance diverge sharply (a hard, interesting failure mode) |
| Language friction | Catalan / Spanish | Dutch, but with strong English documentation |

**Blunt read:** if the artifact's headline question family is **sun/shade and street-side geometry**, Amsterdam wins outright on 3DBAG + BGT + AHN and offers the cleaner engineering path to a working snapshot. If the headline is **amenities, opening hours, terraces and noise** — the "which café is open, quiet and sunny" flavour — Barcelona has data no Dutch city offers, at the cost of building your own height model first. The honest hybrid: **Amsterdam for the geometry-heavy Stage-1 tools, Barcelona as the second city for cross-city generalisation**, which also happens to be a defensible portfolio story.

---

## 5. THINGS TO VERIFY BEFORE COMMITTING (explicit open items)

1. **OSM `opening_hours` tag coverage** for each city — count tagged vs untagged amenities. This determines whether time-aware questions are viable at all. Neither city has an official opening-hours dataset.
2. **Whether the AMB metropolitan 3D model / solar-potential raster is bulk-downloadable and under what licence** (start from the IDE Catalunya catalogue record).
3. **Exact PDOK licence strings** for BGT / BAG / AHN (repeatedly described as "open / CC0-ish"; confirm per dataset, do not assume).
4. **maps.amsterdam.nl terms of use** — whether redistribution inside a released dataset is permitted.
5. **Mapillary coverage density** per city, and whether any needed imagery falls under the CC BY-NC-SA subset.
6. **Whether ICGC has a more recent LiDAR campaign** than the 2008–2011 basis of the 2 m MDT.
7. **Microsoft building-footprints licence** (ODbL vs CDLA-Permissive-2.0, documented inconsistently) — moot if unused.
8. **Barcelona traffic intensity and tourism / HUT datasets** — both plausible but unverified.
9. **NL zoning under the Omgevingswet / DSO** — current download route unverified.
10. **ODbL implications** for whatever you actually release (snapshot database vs QA pairs + scripts).
