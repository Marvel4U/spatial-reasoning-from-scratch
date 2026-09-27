# 08 — Behavioural / outcome data for urban raster layers, and prior work on "masked layer" modelling

Research note, 22 Sep 2026. Scope: Amsterdam (De Pijp) first, then Barcelona / Paris / Vienna.
Constraint assumed throughout: **everything must be snapshot-able to disk under an open licence** — training cannot call live APIs.

Honesty notes up front (things I could NOT verify, stated plainly):

- The **Fietstelweek** bulk download hosts appear to be **dead**. `opendata.cyclingintelligence.eu` does not resolve (DNS `ENOTFOUND` on 22 Sep 2026); `bikeprint.nl/fietstelweek/` returned no readable content. The R package `loreabad6/ftw` that wrapped the download states the URL is unavailable. So the single most attractive dataset in your list is, as far as I can tell, **not currently downloadable from a public URL** — see §A1.1 for what to do about it.
- I did not test-download any large file; sizes below are either stated by the publisher or explicitly marked as my estimate.
- Licence statements are what the portal says; I did not read full legal terms except where noted.
- Vienna is the weakest of the four cities in my search results; I found little concrete behavioural open data and say so rather than guessing.

---

## PART A — behavioural / outcome data

### A1. Cycling

#### A1.1 Fietstelweek / CyclePRINT / BikePRINT (2015, 2016, 2017)

| field | value |
|---|---|
| What it measures | GPS-tracked cycling trips of ~40–55k volunteers during one week (first edition 14–20 Sep 2015, >375k trips, ~1.2M km). Map-matched to an OSM-derived network. Derived per-link quantities: **intensity** (cyclists/link), **average speed**, **delay / waiting time** (incl. at signalised intersections), plus route-level records. |
| Spatial resolution | Per network **edge** (OSM-linked road segment) and **node** (intersection). National coverage incl. Amsterdam. |
| Temporal | One week per edition; editions 2015, 2016 (2017 announced — I could not confirm that 2017 data was ever published as open data). |
| Format | `edges.gpkg` (MULTILINESTRING), `nodes.gpkg` (POINT), `routes.csv`, one set per year. |
| Size | Not stated anywhere I found. My estimate: order 0.5–3 GB per year. **Unverified.** |
| Access | Was `opendata.cyclingintelligence.eu` (BUas / Breda University) and the BikePRINT server. **Both appear offline as of today.** The `ftw` R package (`downloader_ftw()`, `clip_ftw()`) is still on GitHub with two small sample areas bundled (Texel, Smallingerland) — not Amsterdam. |
| Licence | Described in secondary sources as open for non-commercial reuse; **"not usable commercially without permission of the Nationale Fietstelweek consortium"**. Not a clean open licence. |
| Verdict | **Maybe — high value, currently unobtainable.** Per-link cycling intensity/speed/delay is *exactly* the "no algorithm captures this" target you want, and it rasterises trivially. But you must first obtain the files. Practical route: email BUas / CyclePRINT (pure.buas.nl project page "Cycleprint") or NDW, and ask academic groups that published on it (e.g. the MDPI *Sensors* 2023 paper on Dutch bicycle delay estimation) for a copy. Until a file is in your hands, do not put this on the critical path. |

Links: [ftw R package](https://github.com/loreabad6/ftw) · [downloader_ftw docs](https://rdrr.io/github/loreabad6/ftw/man/downloader_ftw.html) · [CyclePRINT project, BUas](https://pure.buas.nl/en/projects/cycleprint/) · [The Urban Future project page](https://theurbanfuture.com/project/nationale-fietstelweek/)

#### A1.2 NDW bicycle data (Fietsdataplatform / Dexter open data)

| field | value |
|---|---|
| What it measures | Counts and speeds of cyclists at **fixed counting locations** (loops, radar, camera), collected nationally by NDW with Tour de Force and road authorities. Counts by speed class, vehicle class, width class. NDW also ingests "floating bike data" (GPS) — see Talking Bikes. |
| Spatial resolution | **Points** (counting sites), not a network-wide field. Density in Amsterdam is unknown to me — I did not enumerate the sites. |
| Temporal | Aggregates per 5 min / hour / week / month / quarter depending on site. Dexter historical DB covers 2010→now for motor traffic; bicycle network integrated into NWB since early 2022. Older count data is being retro-published ("uniform ontsluiten van diverse oude telgegevens" — in progress). |
| Format | CSV, DATEX II v3 (MST/MDP profiles), Fiets API (Data + GIS endpoints). |
| Access | [dexter.ndw.nu/opendata/bicycle](https://dexter.ndw.nu/opendata/bicycle) and [opendata.ndw.nu](https://opendata.ndw.nu/) — bulk files, no registration for open data. (Dexter is a JS app; I could not read its file list via plain fetch. Verify manually.) |
| Licence | NDW open data, stated as open for everyone; NDW's open-data factsheet is the authority. |
| Verdict | **Maybe.** Great as *validation points* ("does my predicted cycling-intensity raster correlate with real counters?"), poor as a dense raster target — a few dozen points in Amsterdam cannot supervise a per-pixel model. Use as eval, not as the training target. |

#### A1.3 Talking Bikes (floating bike data, national)

>1M bicycle rides/year of GPS data, nationally distributed, collected via Talking Traffic partners (Siemens/RingRing). Documented in *Sensors* 23(24):9664 (2023), "Bicycle Data-Driven Application Framework … Nationwide Sparse GPS Data". **Access route unclear; I found no public bulk download.** Routed through NDW/Tour de Force. Verdict: **no for now** — not demonstrably open/downloadable; worth one email to NDW because it is the live successor to Fietstelweek.
Links: [MDPI paper](https://www.mdpi.com/1424-8220/23/24/9664) · [Fietsberaad overview of bicycle data](https://fietsberaad.nl/Kennisbank/Overzicht-Fietsdata)

#### A1.4 Snuffelfiets (Utrecht province)

| field | value |
|---|---|
| What it measures | ~500+ citizen bikes with sensors: PM1/PM2.5/PM10, temperature, humidity, **GPS position and speed**, road-surface vibration. Raw per-measurement records. |
| Spatial / temporal | Point measurements along ridden routes; since June 2019; a new raw file published **every Monday** for the previous week. |
| Format / access | CSV on the CIP Dataplatform: [ckan.dataplatform.nl/dataset/raw-snifferbike-snuffelfiets-data](https://ckan.dataplatform.nl/dataset/raw-snifferbike-snuffelfiets-data). Also via Samen Meten. |
| Coverage | **Province of Utrecht, not Amsterdam.** |
| Verdict | **No for Amsterdam; interesting as a future city.** But note it is a genuinely open, genuinely behavioural GPS dataset with air-quality side channels — if you ever add Utrecht, this is the best raw cycling-trace corpus in NL that is actually downloadable. |

#### A1.5 Amsterdam municipal bicycle counts (fietstellingen)

Manual/automatic counts on average autumn workdays, **2011–2022**, published via Vervoerregio Amsterdam ([maps-vervoerregio.nl/fietstellingen](https://maps-vervoerregio.nl/fietstellingen/)) and the municipality. Point locations, one number per screenline per year. Verdict: **no as a training target** (far too sparse, annual), **yes as a sanity check**.

Also on data.overheid.nl: **Fietsnetwerk Amsterdam** (the policy cycling network, [dataset](https://data.overheid.nl/dataset/trrngyt7yfedkq)) — that is infrastructure, not behaviour.

#### A1.6 Bike-share: what is actually open

| system | what is open | verdict |
|---|---|---|
| **Donkey Republic (Amsterdam)** | Live **GBFS** feed (station/vehicle positions + availability). No historical trip archive that I could find. Amsterdam shared-mobility reporting exists as PDFs (e.g. [Rapportage deelmobiliteit 2022](https://openresearch.amsterdam/image/2023/1/3/las22115_rapportage_deelmobiliteit_2022_wcag_03.pdf)), not as data. | **Maybe, only if you self-snapshot.** Poll the GBFS feed yourself for N weeks → you own a derived time series of pickups/drop-offs per station. That *is* real behaviour and it *is* snapshot-able. Cost: you must start now and wait. |
| **OV-fiets** | NS publishes availability per station via its API (key required). No open trip data. | **No.** |
| **FlickBike** | Defunct. | **No.** |
| **Bicing (Barcelona)** | Gold standard for history: [station status archive](https://opendata-ajuntament.barcelona.cat/data/en/dataset/estat-estacions-bicing), ~500 stations, **every ~4 minutes, March 2019 → present**, monthly CSV/7z, CC-BY 4.0. ~250M rows. Plus [station information](https://opendata-ajuntament.barcelona.cat/data/en/dataset/informacio-estacions-bicing). Note: the older "[Ús del servei Bicing](https://opendata-ajuntament.barcelona.cat/data/en/dataset/us-del-servei-bicing)" (Aug 2018–Mar 2019) is **city-wide aggregate only, no station dimension** — useless spatially. | **Yes (Barcelona).** Differencing station status gives per-station net flow per 4 min → a dense, honest behavioural target at ~500 points across the city. |
| **Vélib (Paris)** | Official [real-time GBFS](https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/), ODbL, **no official history**. Community archive: [lovasoa/historique-velib-opendata](https://github.com/lovasoa/historique-velib-opendata), every 15 min since Dec 2019, single `stations.zip` release asset. | **Yes (Paris), with a caveat** — the history is a third-party scrape; licence inherits ODbL from Paris Data but provenance is a volunteer repo. Document that. |
| **WienMobil Rad (Vienna)** | Citybike Wien ended 1 Apr 2022; the old `Citybike Standorte Wien` dataset on data.gv.at was **deleted** and replaced by a "Wien Mobil Räder" station dataset (28 Jun 2024). I found **station locations only, no usage history**. | **No** (as far as I could verify). |

#### A1.7 Strava Metro — explicitly **not** usable

Free to public agencies and (via an annual academic programme) to university researchers, but the [Metro Terms of Use](https://metro.strava.com/terms) **prohibit sharing, redistributing, reselling or publicly posting the underlying data**; only compliant "Licensee Reports" may be shared. Verdict: **no** — incompatible with an open portfolio artifact. Do not build on it even if you could get access, because you could not ship the data or a reproducible pipeline.

---

### A2. Walking / footfall

#### A2.1 Amsterdam CMSA (Crowd Monitoring Systeem Amsterdam) — **the best open footfall dataset I found**

| field | value |
|---|---|
| What it measures | Anonymised **pedestrian counts** ("aantalPassanten") from 2D/3D counting sensors, count cameras, wifi sensors and beacons at busy locations. |
| Spatial | Per sensor / per location, with geometry. **I pulled the `cmsa/locatie` table (~185 rows): it is dominated by TV cameras and ANPR travel-time cameras on bridges, squares, museums and the A10/A2/A9 ring; no Ferdinand Bol / Albert Cuyp / Sarphatipark / Gerard Dou entries.** Nearest named site to De Pijp is "Rijksmuseum" (northern edge). Several rows have null geometry. So: **counting coverage is the centre + Wallen, not De Pijp.** (The `crowdmonitor/passanten` sensor set may differ slightly from `cmsa/locatie`; verify by listing distinct `naamLocatie` in the passanten table before you rely on this.) |
| Temporal | Per **hour**, also aggregated to day and week (`periode` field, `datumUur` timestamp). I saw live records spanning at least Mar 2023 → Oct 2023 in one query; full range not established. |
| Format / access | REST: `https://api.data.amsterdam.nl/v1/crowdmonitor/passanten/` (fields: `id, sensor, periode, naamLocatie, datumUur, aantalPassanten, gebied, geometrie`). Sensor/location/marking metadata: `https://api.data.amsterdam.nl/v1/cmsa/{sensor,locatie,markering}`. CSV + GeoJSON + WFS + MVT exports. |
| Licence / auth | "Openbare data", no authorization needed. **Caveat: the portal is moving to mandatory API keys** ("optional from mid-September", becoming mandatory) — get your snapshot and a key soon. |
| Size | Small. Tens of sensors × hourly × a few years ≈ low millions of rows at most; easily paginated to local parquet. |
| Verdict | **Yes for Amsterdam-centre, effectively no for De Pijp.** Even where it exists it is sparse point supervision, not a dense raster — use it as (a) a held-out *evaluation* target for a predicted "busyness" field over the centre, or (b) a sparse-supervision head (loss only at observed pixels). Be honest in the README that this is tens of points, not a city-wide ground truth, and that it does not cover your first study district. |

Links: [dataset docs](https://api.data.amsterdam.nl/v1/docs/datasets/cmsa.html) · [WFS docs](https://api.data.amsterdam.nl/v1/docs/wfs-datasets/cmsa.html) · [sensor map](https://maps.amsterdam.nl/cmsa/) · [Druktebeeld in the national algorithm register](https://algoritmes.overheid.nl/nl/algoritme/16268148)

Related but **not open**: Amsterdam's *Druktebeeld* also consumes commercial mobile-phone panel data (Resono-type). Those layers are not published. City-level projects CityFlows (AMS Institute) and openresearch.amsterdam "Crowd Counting" document method, not data.

#### A2.2 Other cities

- **Paris**: [Comptage vélo — données compteurs](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/) (hourly, rolling 13 months, updated D-1), [historical counters](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/), and [multimodal counting](https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/) (bike, scooter, motorcycle, car, truck, bus — hourly, thermal sensors). ODbL. **Verdict: yes, best-in-class permanent counter network among the four cities**, still point data. Paris has no equivalently open *pedestrian* counter set that I could confirm.
- **Barcelona**: [Detall dels aforaments de mobilitat](https://opendata-ajuntament.barcelona.cat/data/ca/dataset/aforaments-detall) (traffic/mobility counting detail, monthly + day-type averages, CSVs roughly 2017–2024) plus [the counting-equipment inventory](https://opendata-ajuntament.barcelona.cat/data/ca/dataset/aforaments-descriptiu). These are **vehicle/mobility counters**; I could not confirm a dedicated open **pedestrian** counter series. Verdict: maybe, point data.
- **Vienna**: the city operates **18 permanent bicycle counting stations** (since 2003, five added 2021); counts are published via data.wien.gv.at / [radverkehrszaehlung](https://www.wien.gv.at/verkehr/radverkehrszaehlung). Network layers ([Hauptradverkehrsnetz](https://www.data.gv.at/katalog/dataset/1ea3d3e8-fa07-4c37-af68-eb588d439de2)) and the Austrian [GIP](https://gip.gv.at/en/index.html) transport graph are open. **No open pedestrian counters found.** Verdict: Vienna is the weakest of the four for behavioural data; if you need a fourth city, consider swapping it for Utrecht (Snuffelfiets) or Paris-plus.
- Eco-Compteur publishes an [open-data page](https://www.eco-compteur.com/donnees-ouvertes-open-data) covering some cities (Paris, Toulouse named) — worth checking whether Barcelona/Vienna sites are included. Unverified.

---

### A3. Generic open GPS traces

#### A3.1 OpenStreetMap public GPS traces

- **Bulk**: [planet.openstreetmap.org/gps/](https://planet.openstreetmap.org/gps/) — the only bulk dumps are **`gpx-planet-2013-04-09.tar.xz` (21 GB, 848k GPX files, ~2.6T points)** and two older "simple GPS points" CSV/TXT dumps (15 GB / 7 GB, 2012). **No dump since 2013.** Licence on that page: "OpenStreetMap and contributors", **CC-BY-SA 2.0** (note: *not* ODbL — the traces page still states CC-BY-SA).
- **Per-area**: the OSM API `GET /api/0.6/trackpoints?bbox=...&page=N` returns public traces for a bbox, 5000 points/page — fine for one district, slow but snapshot-able. `iandees/planet-gpx-dump` is a tool to re-dump traces.
- **Mode labels**: essentially **absent**. OSM traces have optional free-text tags/description; there is no reliable mode field. Also heavily biased toward mappers (surveying walks, car drives along unmapped roads), not representative travel behaviour.
- **Verdict**: **maybe / weak.** Density over central Amsterdam is probably decent (long-mapped city) but it is 2013-vintage in bulk form, unlabelled by mode, and mapper-biased. Usable as a *coverage* prior ("where do people physically go"), not as a behavioural target you can defend.

#### A3.2 GeoLife, MDC, and friends

- **[GeoLife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367) (Microsoft Research Asia)**: 182 users, Apr 2007 – Aug 2012, mostly **Beijing**, 17,621 trajectories, ~1.2M km, 48k+ hours; 91% logged at 1–5 s / 5–10 m; **many trajectories carry transport-mode labels** {biking, walking, running, bus, car, taxi, train, subway, airplane}. Freely downloadable (also mirrored on Kaggle). **No Amsterdam/NL coverage** → **no** for this project except as a methodological reference for mode inference.
- **Nokia MDC (Lausanne)**: access by application/agreement, not open, Lausanne only → **no**.
- I found **no open, mode-labelled, city-scale human GPS corpus for Amsterdam or the Netherlands.** That is the honest state of the world: the NL datasets that exist (Fietstelweek, Talking Bikes) are either offline or gated.

---

### A4. Routable networks for the PATH task

**Recommendation: OSM as primary, NWB as an official cross-check, Amsterdam's own loop/fiets network as a third opinion.**

| source | what it is | licence / access | fitness for "shortest legal route" |
|---|---|---|---|
| **OpenStreetMap** | Full multimodal graph with `highway=*`, `access/foot/bicycle/motor_vehicle`, `oneway`, `oneway:bicycle`, `bicycle=dismount`, `cycleway:*`, barriers, `maxspeed`. | ODbL, free. Extracts: Geofabrik NL, BBBike custom bbox. Tools: `osmnx` (`network_type='walk'|'bike'|'drive'` already encodes access rules), `pyrosm`, **OSRM** and **Valhalla** (both run fully offline from a local `.pbf` — this is what you want, no live API). | **Best available.** Amsterdam/NL is among the best-tagged areas in OSM; cycleways are mapped as separate ways with the NL-specific conventions ([NL:Cycleway](https://wiki.openstreetmap.org/wiki/NL:Cycleway), [NL:Tagging van Nederlandse wegen](https://wiki.openstreetmap.org/wiki/NL:Tagging_van_Nederlandse_wegen)). |
| **NWB (Nationaal Wegenbestand)** | Official national road file, all named/numbered public roads; **cycle network added from BRT since early 2022**. Publisher states it is "actueel, routeerbaar en nauwkeurig". | **Public Domain** (explicitly free incl. commercial). Shapefile + GeoPackage from [nationaalwegenbestand.nl/nwb-downloaden](https://www.nationaalwegenbestand.nl/nwb-downloaden); also PDOK WMS/WFS/OGC API/ATOM. [Handleiding](https://docs.ndw.nu/en/handleidingen/nwb/) | **Good for cars/cycling, weak for pedestrians.** NWB has no sidewalk geometry and limited legal-access semantics per mode. Best use: authoritative link IDs to join NDW counts onto, and a check that OSM has not invented/omitted a street. |
| **BGT** | Surfaces (sidewalk, cycle path, roadway polygons) — **not a graph**. | Public domain-ish, PDOK. | Not routable as-is. But it is the right thing to *rasterise* for a pedestrian cost surface, and you already have it. |
| **Amsterdam "Loop- en fietsnetwerk Amsterdam"** | Municipal **walking-and-cycling network graph** linked to BAG addresses, built for accessibility / location-allocation analysis. GeoPackage in RD (EPSG:28992) and WGS84, WFS API, quarterly updates (last seen 23 Apr 2025). **CC-BY 4.0.** [dataset](https://data.overheid.nl/dataset/7hgzsrxqwsgqhw) | **Strong candidate and under-used.** An official pedestrian+bike graph for exactly your study area, with address linkage — excellent for generating PATH ground truth and for cross-validating OSM-derived routes. |
| **Fietsersbond routeplanner network** | Cyclist-union routing network with surface/comfort attributes. | **Could not verify an open download.** Treat as closed unless you find otherwise. | Unknown. |

**How to derive "legal route" per mode (concrete recipe).**

- *Pedestrian*: OSM ways where `highway ∈ {footway, path, pedestrian, steps, living_street, residential, service, track, unclassified, tertiary, secondary, primary}` minus `foot ∈ {no, private}`; include `highway=cycleway` **only where `foot != no`** — the NL default is that walking on a cycle path is allowed when there is no adjacent footway ([NL cycleway wiki](https://wiki.openstreetmap.org/wiki/NL:Cycleway)), so a country-default table matters. Exclude motorway/trunk and their links. Ignore `oneway` (except `oneway:foot=yes`). Handle `barrier=gate/bollard` with `foot` access. Use [OSM tags for routing / access restrictions](https://wiki.openstreetmap.org/wiki/OSM_tags_for_routing/Access_restrictions) for the NL default table — note the wiki's own warning that **most routers use worldwide defaults, not the NL defaults**, so if you use OSRM/Valhalla out of the box your "legal" route may be subtly wrong. Safer: build the graph yourself with `osmnx`/`pyrosm` and an explicit NL access table you can print in the README.
- *Cycling*: `highway=cycleway` plus roads with `bicycle != no`; respect `oneway=yes` **unless** `oneway:bicycle=no` (very common in Amsterdam); treat `bicycle=dismount` as either forbidden or penalised — pick one and document it.
- *Car*: `motor_vehicle/access != no`, respect `oneway`, exclude `highway ∈ {footway, cycleway, pedestrian, steps, path}` and `access=private`. Amsterdam's one-way maze makes this the most "interesting" mode for the PATH task.

**Tagging quality in Amsterdam**: I did not find a published quantitative audit of NL foot/bicycle access completeness. Qualitatively: NL is a heavily-mapped, cycling-obsessed community, cycleways are near-complete, `oneway:bicycle=no` is widely used, and sidewalk mapping is *partial* (many Amsterdam streets have no separate `footway=sidewalk` way, so pedestrian routing falls back to walking along the carriageway centreline). **That is the main correctness risk for your PATH task**: pedestrian shortest paths will follow street centrelines, not sidewalks, so crossings and one-sided sidewalks are invisible. Mitigation options: (a) accept centreline routing and say so; (b) cross-check against the municipal Loop- en fietsnetwerk; (c) use BGT sidewalk polygons to build a surface-based cost raster and route on the raster instead of a graph — which actually fits a raster model better and makes the target verifiable by construction.

---

### A5. Other outcome layers reflecting human decisions (Amsterdam, open)

| dataset | what it measures | resolution / years | format & access | licence | verdict |
|---|---|---|---|---|---|
| **Horeca exploitatievergunningen met terrasgeometrie** — [data.overheid](https://data.overheid.nl/dataset/gsy50tekojkcgw), [API docs](https://api.data.amsterdam.nl/v1/docs/datasets/horeca.html) | Granted hospitality permits: name, address, category, date, **terrace polygons**, exemptions, opening hours for venue and terrace | Per establishment, polygon geometry; current state (history = permit dates) | WFS CSV/GeoJSON: `api.data.amsterdam.nl/v1/wfs/horeca/?...TYPENAMES=exploitatievergunning-terrasgeometrie` | Amsterdam open data | **Yes.** Terrace geometry is a genuine revealed-preference signal (where sitting outside is worth paying for) that no rule derives from buildings+noise. Small file, clean polygons, rasterises to a m²-of-terrace-per-cell layer. Strong "predict this layer from the others" task. |
| **Meldingen Openbare Ruimte (MORA / SIA "Signalen")** — [docs](https://api.data.amsterdam.nl/v1/docs/datasets/meldingen.html), [WFS](https://api.data.amsterdam.nl/v1/wfs/meldingen/v1) | 311-style citizen complaints: waste, noise/nuisance from venues, boats, people/groups, cleanliness, roads & street furniture, green & water | Point (often coarsened), **mid-2018 → present**, per report with timestamp + category | REST/WFS, CSV/GeoJSON, no auth | Public/open data | **Yes — arguably the single best behavioural raster target in Amsterdam.** Dense, city-wide, multi-year, multi-category → you get several count rasters (noise complaints, litter, broken street furniture) that are pure human behaviour. Caveats to document: reporting propensity is itself socially biased, and some report types are excluded (split reports). Bias is a *feature* for a portfolio artifact if you name it. |
| **Functiekaart / niet-woonfuncties** — [data.overheid](https://data.overheid.nl/data/dataset/cwujx-uxu9r8sg), [map](https://maps.amsterdam.nl/functiekaart/) | All non-residential functions: offices, retail, hospitality, social/recreational, parking, transit | Per BAG object (building/unit), snapshot; Amsterdam-wide | Download via [maps.amsterdam.nl/open_geodata](https://maps.amsterdam.nl/open_geodata/) and data.amsterdam.nl | Amsterdam open data | **Yes.** Higher quality and better typed than OSM/Overture POIs for Amsterdam, and derived from registrations (real economic decisions). Best used as the "POI layer" to be masked and predicted. |
| **BRON — verkeersongevallen** — [ongevallen](https://data.overheid.nl/en/dataset/53580-verkeersongevallen-nederland----ongevallen--rws-), [wegvakgeografie](https://data.overheid.nl/en/dataset/53579-verkeersongevallen-nederland---wegvakgeografie--rws-) | Police-registered road crashes, with severity, modes involved, location linked to NWB | Point/segment, **2003-01-01 → 2024-12-31**, ZIP per year and a 2022–2024 combined file | ZIP of geo-formats from Rijkswaterstaat geoservices; also on ArcGIS Hub | Open (RWS) | **Maybe→yes.** Real outcomes, 20+ years, network-linked. Caveat: severe under-registration of cycling-only crashes in NL is well known; STAR/BRON coverage of light injuries is poor. As a raster target, aggregate to crash-density per cell over many years or you get an almost-empty map. |
| **Inside Airbnb** — [Amsterdam](https://insideairbnb.com/amsterdam/), [get the data](https://insideairbnb.com/get-the-data/) | Short-stay listings: location (jittered ~150 m), price, room type, reviews (a proxy for bookings), availability calendar | Per listing; quarterly snapshots, long history; Amsterdam snapshot 15 Jun 2026, Barcelona 24 Jun 2026 (Paris & Vienna are also covered by the project; I did not see them in the fetched excerpt) | `listings.csv(.gz)`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson` | **CC-BY 4.0** | **Yes, with an asterisk.** Covers all four target cities — rare. Real market behaviour (price, occupancy proxy). Asterisk: **coordinates are deliberately jittered**, so it is only honest at ≥150–200 m cell size. Also scraped, not authoritative. |
| **Parkeervakken + parkeerdruk + garage occupancy** — [parkeervakken](https://data.overheid.nl/en/dataset/d6rmg5cdgbfp2q), [parkeerdruk](https://data.overheid.nl/dataset/t_86lm6sbcimbg), [garages live](https://data.overheid.nl/dataset/9orkef6t-au29g) | Parking bay geometry & type (incl. disabled, loading, taxi); **parkeerdruk** = occupancy pressure + permit waiting-list length per neighbourhood; garages publish free spaces every 5 min | Bays = polygons; parkeerdruk = per buurt, periodic reports; garages = ~20 points, 5-min live | WFS/WMS; garages live feed; NPR tariffs at opendata.rdw.nl | Open | **Maybe.** Parkeerdruk per buurt is coarse but is a real scarcity signal. Garage occupancy is only ~20 points and live-only (you'd have to self-snapshot). Bay geometry is infrastructure, not behaviour. |
| **KvK handelsregister (business openings/closings over time)** | — | — | — | **Closed / paid** | **No.** Substitute for churn: (a) Functiekaart snapshots across years; (b) **Overture Maps Places** monthly releases — every release is kept as GeoParquet on S3/Azure and browsable via their STAC catalog, so you can diff release *N* against release *N−12* to get openings/closings. Licence since Sep 2025 is per-record by source (CDLA-Permissive-2.0, ODbL for OSM-derived, Apache-2.0 for Foursquare-derived) — you must carry the per-record licence. [Overture data repo](https://github.com/OvertureMaps/data) · [cloud sources](https://docs.overturemaps.org/getting-data/cloud-sources/) |
| **WOZ (property values)** | Per-address assessed value | — | WOZ-waardeloket is **view-only per object**, explicitly not bulk/automated extraction; LV WOZ bulk is authorised users only; legislative change would be needed to open it | Public ≠ open | **No at address level.** **Yes at neighbourhood level**: [CBS *Kerncijfers wijken en buurten*](https://www.cbs.nl/nl-nl/reeksen/publicatie/kerncijfers-wijken-en-buurten) (yearly 2004–2026, StatLine OData + Excel, CC-BY) publishes mean WOZ per buurt/wijk — coarse but legitimate and joins to CBS buurt polygons. Note CBS suppresses the value where a buurt has <20 dwellings or <85% WOZ coverage, so expect holes. |
| **Tourism pressure** | — | — | Amsterdam publishes visitor statistics at district level (onderzoek.amsterdam.nl); the fine-grained inputs are commercial phone panels | Mixed | **Maybe, coarse only.** Inside Airbnb + CMSA + terrace area are better proxies. |
| **Shared-mobility drop-offs** | See §A1.6 — GBFS live only | — | Self-snapshot | Operator feeds | **Maybe, if you start polling now.** |

---

### A6. Practical shortlist for a raster model (my ranking)

1. **Meldingen Openbare Ruimte (MORA/Signalen)** — dense, multi-year, multi-category, city-wide, genuinely behavioural, trivially rasterised, open, no API key drama. Best ratio of "real human behaviour" to "effort".
2. **Horeca permits + terrace polygons** — small, clean, geometric, revealed-preference; perfect masked-layer target that no rule over buildings/noise reproduces.
3. **CMSA hourly pedestrian counts** — the only real open footfall measurement in Amsterdam; use as sparse supervision and as held-out validation of a predicted busyness field. Two caveats: sensors sit in the centre/Wallen, **not in De Pijp**, and the API is moving to mandatory keys — grab a snapshot now.

If you want a third *dense* layer instead of CMSA, the honest alternative for De Pijp is **Inside Airbnb review-density** (jittered to ~150 m, but city-wide and multi-city) or **Overture Places year-over-year churn**.

Honourable mentions: Inside Airbnb (only dataset that spans all four cities), Bicing 4-minute station history (Barcelona's answer to the missing Fietstelweek), Paris counter network.

Explicitly parked: Fietstelweek (offline), Talking Bikes (gated), Strava Metro (licence forbids it), WOZ per address (not open), OSM GPS traces (2013 bulk, no modes).

---

## PART B — has the "masked urban layer" idea been done?

<!--PART_B-->

