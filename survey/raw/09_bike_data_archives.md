# Archived / mirrored Dutch + EU bike-route data — search report

Date of search: 22 Sep 2026. All URLs tested from this machine unless marked "not verified".
Everything below is either (a) verified by me fetching bytes, or (b) explicitly flagged as unverified.

---

## 0. Headline result

**The full Nationale Fietstelweek 2015/2016/2017 open dataset is recoverable from the Internet Archive.**
The original host `opendata.cyclingintelligence.eu` is dead (DNS `ENOTFOUND`, and the Wayback Machine holds
**no** data files for it — only HTML/CSS). But `www.bikeprint.nl/fietstelweek/` — the *other* distribution
point, the one the `ftw` R package scraped — was crawled by the Internet Archive on **2020-01-23**, and the
crawler **followed the download links and captured all 12 zip files** (~850 MB total). I downloaded one 5.4 MB
zip in full and `unzip -t` reports no errors; I range-probed the three largest and all return HTTP 206 with a
valid `PK\x03\x04` header.

---

## 1. PRIMARY TARGET — Nationale Fietstelweek (CyclePRINT / BikePRINT)

### 1.1 Status of the original hosts

| Host | Status today | Wayback holdings |
|---|---|---|
| `opendata.cyclingintelligence.eu` | **DNS does not resolve** | Page captures only; CDX for `cyclingintelligence.eu` (domain match) shows *zero* non-HTML/CSS/image captures. Only extra host found: `ndata.cyclingintelligence.eu` (one 308 redirect, 2024-06-13). **No data recoverable here.** |
| `www.bikeprint.nl` | Domain resolves but is now a **parked / spam domain** (captures from 2023-2026 show `ads.txt`, `.well-known/*`, generated junk). The `/fietstelweek/` app is gone. | **Data files captured 2020-01-23.** |
| `fietstelweek.nl` | Now a generic cycling blog (`fietstelweek.nl/faq/`), not the project. | n/a |

### 1.2 The archived download page (licence text, verbatim)

`http://web.archive.org/web/20200123130606/http://www.bikeprint.nl/fietstelweek/`
Title: *"Download bestanden Nationale Fietstelweek 2015, 2016 en 2017"*. Licence stated on the page (Dutch, my
paraphrase in brackets):

- *"Een bronvermelding bij het gebruik van de data wordt opgenomen"* [attribution required]
- *"Van de data afgeleide producten als open data beschikbaar worden gesteld"* [**derived products must be
  released as open data** — a share-alike-ish clause]
- *"Een mail wordt gestuurd aan bussche.d@buas.nl (Dirk Bussche) waarin wordt aangegeven voor welke toepassing
  de data gebruikt wordt"* [notify the maintainer by e-mail what you use it for]
- *"De Fiets Telweek dataset mag niet voor commerciële doeleinden gebruikt worden zonder toestemming van het
  consortium Nationale Fiets Telweek"* [**non-commercial without consortium permission**]

**This is not an OSI/CC-open licence.** It is "open data, attribution, share-alike, non-commercial, notify us".
For a public portfolio artifact: attribution + releasing derived products openly is fine; the NC clause means
don't sell it and don't ship it inside anything commercial. The notification e-mail address
(`bussche.d@buas.nl`, Breda University of Applied Sciences) may or may not still work — I did not test it, and
sending mail is out of scope here. Marvin should decide whether to send that mail before publishing.

### 1.3 The 12 archived zip files (all verified present, `application/zip`, HTTP 200 in CDX)

Snapshot timestamp `20200123130606` for all except where noted. Download pattern — prepend
`http://web.archive.org/web/<timestamp>id_/` to the original URL (the `id_` suffix gives the raw bytes without
the Wayback toolbar wrapper), and URL-encode `/` in the `h=` parameter as `%2F`:

| # | `bestand=` | Content (page label) | Bytes (CDX) |
|---|---|---|---|
| 1 | 0 | Routes 2015 met uren en weekdagen | 152,069,508 |
| 2 | 3 | Links 2015 met intensiteiten en snelheden — Web Mercator | 38,489,984 |
| 3 | 6 | Links 2015 met intensiteiten en snelheden — Rijksdriehoek (EPSG:28992) | 39,793,778 |
| 4 | 9 | Knopen 2015 met wachttijden — Web Mercator | 3,825,699 |
| 5 | 12 | Knopen 2015 met wachttijden — Rijksdriehoek | 5,309,313 |
| 6 | 15 | **Routes 2016 met uren en weekdagen** (ts `20200123130628`) | 197,501,146 |
| 7 | 18 | Links 2016 met intensiteiten en snelheden — Web Mercator | 98,763,469 |
| 8 | 21 | Links 2016 met intensiteiten en snelheden — Rijksdriehoek | 101,311,727 |
| 9 | 24 | Knopen 2016 met wachttijden — Web Mercator | 21,997,033 |
| 10 | 27 | Knopen 2016 met wachttijden — Rijksdriehoek | 28,838,610 |
| 11 | 30 | **Gecombineerde links met snelheden 2015–2017** — Web Mercator | 94,137,128 |
| 12 | 33 | Voorbeeldbestanden met ruwe data (raw GPS sample) | 69,387 |

Full URL template:
```
http://web.archive.org/web/20200123130606id_/http://www.bikeprint.nl/fietstelweek/download.php?bestand=<N>&d=1579784748&h=<HASH>
```
Hashes (from the archived page source): 0→`15OrhMfJZjIII`, 3→`15Ry.Vfobz.fs`, 6→`15J5TzhFUmNbc`,
9→`15ER9Cr.pA17w`, 12→`155VmUGRF59LI`, 15→`15qapb8VhC%2Fi.`, 18→`155t.4Hd2DZFY`, 21→`15auI8Bm.Fniw`,
24→`15yo.f08CokM.`, 27→`15EOPf%2Fy63TuQ`, 30→`15sF4fFGbeAX6`, 33→`15MuNgG%2FE9DXI`.
There is also a second capture of `bestand=0` at ts `20200123130258` (152,069,506 B) as a fallback.

Note: 2017 is present **only** as the combined 2015–2017 link file (#11). There is no separate 2017 routes or
nodes file on the page. Earlier Wayback captures of `download.php` (2016, 2017) returned a 316-byte HTML error
page — the `d=`/`h=` link signature had expired. **Only the 2020-01-23 crawl has real payloads.**

### 1.4 Verification I actually performed

- `bestand=33` (69 KB) — full download, HTTP 200, `application/zip`, unzips to `routes1.csv` (95,799 B),
  `routes2.csv` (226,443 B), `beschrijving.txt`.
- `bestand=12` (5.4 MB) — **full download, `unzip -t` → "No errors detected"**. Contents:
  `home/goudappel/knopen-28992.{shp,shx,dbf,prj}`, 15.1 MB uncompressed, ESRI shapefile, EPSG:28992.
- `bestand=0`, `15`, `30` — HTTP range request `0-2000000` → HTTP 206, 2,000,001 bytes, `application/zip`,
  magic bytes `PK\003\004`. **Full integrity of the three big files is NOT yet verified** — download them and
  `unzip -t` before relying on them.
- `documentatie.pdf` — recovered in full (144,204 B) from ts `20160607040851`, 2 pages.

Local copies already in the scratchpad: `ftw_sample.zip`, `ftw_sample/` (extracted), `knopen2015_rd.zip`,
`ftw_documentatie.pdf`, `ftw_page.html`.

### 1.5 Schema (from the recovered `documentatie.pdf` — "Gebruiksaanwijzing open dataset Fiets Telweek")

**Links met intensiteiten en snelheden** (shapefile, line geometry, one row per network link):
- `ID` — link id
- `SNELHEID` — mean speed of all cyclists recorded on that link
- `SNELHEID_R` — *relative* speed = fraction of desired speed (10th percentile of the surroundings). Multiply
  by ~20 for a km/h-ish figure.
- `INTENSITEIT` — **number of recorded bicycle trips per link over the whole Fietstelweek** (Mon 14 – Sun 20
  Sep 2015 for the 2015 file). ← *this is the per-segment count raster target.*

**Knopen met wachttijden** (shapefile, point geometry):
- `ID`, `KNOOP` (node position in network), `TIJD` (delay per node in **seconds**, averaged over cyclists).

**Routes met uren en weekdagen** (CSV, one row per (route, link)):
- `routeid` — route id; a route is typically many links, one record per link
- `linknummer` — FK into the Links dataset
- `richting` — `t` = travelled with digitisation direction, `f` = against
- `uur` — 0–23, hour of day
- `weekdag` — 0–6 (0 = Sun 20 Sep, 1 = Mon 14 Sep, … 6 = Sat 19 Sep 2015)

**Voorbeeldbestanden ruwe data** (the 69 KB sample, 2015): columns `id, tijd, lon, lat, speed, heading` —
raw GPS points, WGS84, sub-minute but irregular sampling ("not every second, to save battery"). Heavily
de-identified: clipped to two small areas (Tilburg Stappegoorweg/Ringbaan Zuid; Den Bosch Paleiskwartier),
±400 m trimmed off route ends, times jittered ±60 min, person-id and mode dropped, route ids randomised so
they **cannot** be joined to the open network data. It is a *format* sample, not usable data. The real raw
GPS point cloud was never published openly — only the map-matched network aggregates were.

### 1.6 Verdict for a raster model over Amsterdam

**Best available target, by a wide margin.**
- *Route density per street segment*: directly available. `Links … INTENSITEIT` is literally counts per link
  for a full week, national coverage including all of Amsterdam, in EPSG:28992 (Rijksdriehoek — already the
  right projection for a metric Dutch raster; no reprojection distortion). Clip to an Amsterdam bbox, rasterise
  `INTENSITEIT` by burning line geometry into a grid. Two independent years (2015, 2016) + a combined
  2015–2017 file → natural train/holdout or a temporal-consistency check.
- *Origin–destination pairs*: **not directly available.** The routes CSV has no OD columns and start/end links
  are not flagged; you would have to reconstruct a path per `routeid` by ordering its `linknummer`s along the
  network (the CSV rows are not guaranteed ordered) and take the terminal nodes. Doable but real work, and the
  reconstruction is lossy at loops/self-intersections.
- *Secondary channels you get for free*: mean speed and relative speed per link, and node delay in seconds —
  three more continuous rasters over the same geometry. Useful as auxiliary targets or as "reasoning" features.
- *Caveats to document honestly*: the intensities are **app-user counts, not a census** — smartphone-app
  self-selection, one week in September, weather-dependent, systematically under-representing children,
  elderly and short trips. Penetration differed between 2015 and 2016 (the `ftw` package README notes the 2015
  and 2016 attributes are "not completely equal"), so absolute counts are not comparable across years without
  normalisation. And the licence is NC + share-alike-ish.

### 1.7 Secondary mirrors / derivatives found

- **`github.com/loreabad6/ftw`** (R package, Lorena Abad Crespo, Univ. Salzburg). Repo size ~8.6 MB. Ships
  **real clipped Fietstelweek data** under `inst/extdata/{2015,2016}/{Texel,Smallingerland}/` as
  `edges.gpkg`, `nodes.gpkg`, `routes.csv`. Texel and Smallingerland only — **no Amsterdam**. Value: (a) proof
  of the post-processing target format (GeoPackage + CSV) and (b) `R/download_clip.R` documents the internal
  file names inside the zips: 2015 `links/home/goudappel/links-met-snelheden-900913.shp`,
  `knopen/home/goudappel/knopen-900913.shp`, `routes/routes.csv`; 2016 `netwerk-2016-900913.shp`,
  `knopen-2016-900913.shp`, `routes2016.csv`. Its `downloader_ftw()` scrapes `bikeprint.nl/fietstelweek/` and
  is therefore **dead** — but the scraping logic can be repointed at the Wayback URLs above.
  Licence of the repo itself: not checked.
- **`github.com/sweerstra/ads-fietstelweek`** — a student web app about bike parking. Repo size 18.8 MB. I did
  not confirm it contains Fietstelweek data; the description ("looking for a bike shed in your village") makes
  it unlikely to be the count data. Low priority.
- **`movecatalog.ugent.be/dataset/fietstelweek`** (Move CKAN @ UGent) — appears in search results as holding a
  "fietstelweek – data sample [level 0]" resource. **I could not verify**: the host refuses HTTPS
  (`ECONNREFUSED` on 443) and its CKAN API over plain HTTP returned nothing. Possibly a Flemish
  (Belgian) Fietstelweek, which is a *different* count week from the Dutch one. Worth one manual browser check.
- **`dataplatform.nl` / `data.overheid.nl`** — several "Fietstellingen" entries exist, but these are
  loop/permanent-counter datasets from individual municipalities, **not** the Fietstelweek network data. I did
  not find a Fietstelweek entry with a live resource on data.overheid.nl.

### 1.8 Where it is NOT (honest negatives, all searched)

Zenodo (`q=fietstelweek` → 0 records), Hugging Face datasets (`search=cycling`, `search=bicycle` → nothing
remotely related; hits are battery-cycling, cycling-safety chat data, VQA), Kaggle (no fietstelweek dataset
surfaced), GitHub repo search (`fietstelweek` → exactly 2 repos, both above; `bikeprint OR cyclingintelligence`
→ 0), 4TU.ResearchData (searched "cycling bicycle GPS" → 25 hits, all aerodynamics / eye-tracking / HMI
studies, no route data). **Not checked directly** (search-engine coverage only, no API query): figshare,
DANS EASY / Data Station, Mendeley Data, Harvard Dataverse, OSF, Dryad. DANS is the most plausible remaining
Dutch home and is the one gap I would close next.

---

## 2. Other Dutch bike GPS / route data

### 2.1 Snuffelfiets (Province of Utrecht) — GPS + air-quality sensor, ongoing

- What: ~500+ volunteer cyclists in Utrecht province since Jun 2019, bike-mounted sensor logging **GPS
  position plus PM2.5, temperature, humidity** per measurement. Genuinely point-level with coordinates.
- Where: `https://data.overheid.nl/dataset/snuffelfietsdata-openbaar` (the slug 404s on direct fetch — the
  live page is reachable via the portal search; **URL not verified by me**). Underlying store is CKAN at
  `dataplatform.nl` / `ckan.dataplatform.nl` — **my direct API call was refused (`ECONNREFUSED` on 443)**, so
  I could not read resource URLs, sizes or the licence field. Example helper code:
  `github.com/CivityNL/Snuffelfiets-examples`.
- Access pattern (reported by sources, unverified): the full dump is too large to download in one go; you page
  through the **CKAN DataStore API** in chunks.
- Coverage: Utrecht province. **Not Amsterdam.**
- Verdict: **wrong city**, and it is an air-quality dataset that happens to carry GPS. Useful as a *second*
  region for generalisation testing, or as a methodological precedent (it has been used for COVID cycling-change
  studies). Not a substitute for Fietstelweek over Amsterdam.

### 2.2 NDW bicycle data (Nationale Databank Wegverkeersgegevens) — the official successor

- NDW took over bicycle data collection after Fietstelweek folded: framework agreement 2019–2023 with 13
  suppliers, a new bicycle-count exchange standard, first contracts for the Rotterdam–The Hague metro region
  and South Holland province, weekly deliveries.
- Docs: `https://docs.ndw.nu/producten/fietsdata/` (and `/en/producten/fietsdata/`). Open data root
  `https://opendata.ndw.nu/` — **confirmed live, HTTP 200**. There is a dedicated open bicycle portal with
  richer aggregation queries than the car data.
- **I did not verify** the exact fiets endpoint, file formats, sizes or licence. This is the biggest
  unexplored lead for *current* Dutch data and deserves a follow-up pass.
- Verdict: likely **counts at counting points**, not network-wide per-link route density, and initially
  Rotterdam/South Holland rather than Amsterdam. Probably not a drop-in raster target, but worth 30 minutes.

### 2.3 Amsterdam municipal data

- `maps.amsterdam.nl/fietsnetten/` — Plusnet / Hoofdnet Fiets (bike network hierarchy), GeoJSON downloads via
  `maps.amsterdam.nl/open_geodata/`. `maps.amsterdam.nl/fietsroutes_wegwijzer/` — signposted routes.
  data.overheid.nl entries: "Fietsnetwerk Amsterdam" (`/dataset/trrngyt7yfedkq` and
  `/dataset/196aeef9-eeba-4442-b2df-1ca3e9f9f9f1`), "Loop- en fietsnetwerk Amsterdam" (`/dataset/7hgzsrxqwsgqhw`).
  I guessed at a direct GeoJSON URL and got a 404, so **use the portal pages, not a constructed URL.**
- `maps-vervoerregio.nl/fietstellingen/` — Vervoerregio Amsterdam bike counts, **2017–2022**, counting-point
  based. Format/licence/bulk-download **not verified**.
- Verdict: the network geometry is **exactly what you want as the base grid / rasterisation target geometry**
  for Amsterdam. It carries no intensity. Pair it with Fietstelweek intensities (or with OSM).

### 2.4 Bike-share trip / OD data in NL

Searched OV-fiets, Donkey Republic, Mobike Rotterdam. **No open trip-level or OD dataset found.** NS opened
an OV-fiets *availability* API (station stock), not trips. Donkey Republic exposes GBFS (live station/vehicle
status) — third-party scrapes exist (`bikesharemap.com/rotterdam/`, `/amsterdam/`) but these are live views,
not archives. Verdict: **nothing usable** for OD pairs in NL.

---

## 3. Barcelona / Paris / Vienna

### 3.1 Barcelona — Bicing (VERIFIED via CKAN API, the strongest non-Dutch find)

**Dataset A — `estat-estacions-bicing` ("New Bicing stations status of Barcelona city")**
- URL: `https://opendata-ajuntament.barcelona.cat/data/en/dataset/estat-estacions-bicing`
- API: `https://opendata-ajuntament.barcelona.cat/data/api/3/action/package_show?id=estat-estacions-bicing`
- Licence: **CC-BY-4.0** (`license_id: CC-BY-4.0`) — clean, commercial-safe, unlike Fietstelweek.
- **88 resources.** Monthly `.7z` files, naming `YYYY_MM_<CatalanMonth>_BicingNou_ESTACIONS.7z`.
  Coverage runs **2019-03 through 2026-04** (still being updated — most recent resource 2026-04).
- Size per month: **~17–21 MB compressed**. Examples:
  `2019_09_Setembre…` 17,127,124 B; `2019_12_Desembre…` 17,529,064 B; `2026_03_Marc…` 21,561,252 B;
  `2026_04_Abril…` 20,331,187 B. Whole archive ≈ **1.5 GB compressed**.
- Download URL pattern (per-resource UUID, no guessable pattern — enumerate via the API):
  `https://opendata-ajuntament.barcelona.cat/data/dataset/6aa3416d-ce1a-494d-861b-7bd07f069600/resource/<resource-uuid>/download`
  e.g. 2019-12: `…/resource/29ef34c6-1dde-49c8-a636-28a51ba647fe/download`
- Content: per-station snapshot of mechanical/electric bikes and free docks, **~every 4 minutes**, ~500+ stations.
- Also live: `Estat_Estacions_Bicing_securitzat_json` (GBFS-style real-time JSON).

**Dataset B — `informacio-estacions-bicing`** (station metadata: id, name, lat/lon, capacity). 89 resources,
monthly `.7z` ~8–9 MB, same UUID scheme under dataset `bd2462df-6e1e-4e37-8205-a4b8e7313b84`, CC-BY-4.0.
You need this to geolocate Dataset A.

**Dataset C — `us-del-servei-bicing`** ("Bicing service use, Aug 2018 – Mar 2019"). 16 resources, CSV + 7z,
**tiny** (~265 KB/month CSV, e.g. `2019_01_Gener_BICING_US.csv` 269,477 B). At that size it is monthly/daily
**aggregate usage counts**, not per-trip records. Dataset UUID `4a469cf6-dbab-4aa1-b492-aa0af9af93c9`.
CC-BY-4.0.

**Also**: `kaggle.com/datasets/edomingo/bicing-stations-dataset-bcn-bike-sharing` — a third-party Kaggle
repackaging of the station data. Size/licence **not verified**; the official portal is the better source.

**Verdict (Barcelona):** clean licence, huge time span, and the one thing it is *not* is route data. Station
status gives you **no OD pairs and no per-street-segment density**. You can *infer* OD flows by differencing
consecutive station snapshots (a well-known but noisy trick — rebalancing trucks corrupt it badly, and a
4-minute interval merges concurrent departures), and you cannot recover the path between stations at all.
As a target for a raster model over street segments: **not usable without strong modelling assumptions.**
As a target for a *station-level* density raster or an OD-matrix task: usable, honestly caveated.

### 3.2 Paris

- **Vélib trip data: not open.** The city portal has exactly 2 Vélib datasets (verified via the Opendatasoft
  Explore API v2.1): `velib-emplacement-des-stations` (1,518 station records) and
  `velib-disponibilite-en-temps-reel` (1,518 records, real-time). Both **ODbL**. Both are *snapshots* — there
  is no historical archive of station status published by the city, and no trip-level release. (Contrast with
  Barcelona, which does archive monthly.) Vélib' Métropole publishes **GBFS 1.0** live feeds, no key,
  refreshed every minute: `velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole`. Live only.
- **`comptage-velo-historique-donnees-compteurs`** — *"Comptage vélo – Historique – Données Compteurs et Sites
  de comptage"*, `https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/`.
  Verified via API: licence **ODbL** (`opendatacommons.org/licenses/odbl/`), last modified 2026-02-20.
  Data **from 1 Jan 2016 onward**, two file families, both as compressed CSV: (a) per counter, with direction,
  hourly; (b) per counting site, no direction, quarter-hourly. The API reports `records_count: 0` because the
  payload lives in attachment files rather than the searchable index — **file sizes not verified.**
  Companion live dataset: `comptage-velo-donnees-compteurs`; multimodal variant
  `comptage-multimodal-comptages`.
- **Verdict (Paris):** the counter history is genuinely good *time-series* data but it is **~100 fixed points,
  not a network**. A raster built from it would be almost entirely empty. No route data, no OD. Paris is the
  weakest of the four cities for this purpose.

### 3.3 Vienna

- **`https://www.wien.gv.at/data/ogd/ma46/radverkehrszaehlungen.csv`** — **verified live, HTTP 200**,
  `Last-Modified: 2026-09-04`, `Content-Length` 0x6245e = **402,014 bytes**. Daily counts per permanent
  counting station (Argentinierstraße, Donaukanal, Lassallestraße, …), MA 46 magnetic-loop sensors.
  Licence not read from the header, but Vienna OGD is uniformly **CC-BY-4.0 Stadt Wien** — treat as CC-BY but
  confirm on the catalogue page.
- Network geometry on data.gv.at (all Stadt Wien, CC-BY-4.0, dataset pages verified present in search results,
  files not downloaded): `Hauptradverkehrsnetz Wien` (`/katalog/dataset/1ea3d3e8-fa07-4c37-af68-eb588d439de2`),
  `Hauptradverkehrsnetz Planung Wien` (`…/4973cdb9-e4a8-4de1-a3b9-e23ab41af527`),
  `Radfahranlagen Wien` (`…/5e6175cd-dc44-4b32-a64a-1ac4239a6e4a`),
  `Citybike Standorte Wien` (`…/stadt-wien_citybikestandortewien`).
- **WienMobil Rad** (the current operator, ~3,000 bikes, all 23 districts): I found a 2022 Open Data MeetUp
  presentation about it but **no open trip or historical-status dataset**. Live GBFS exists via the operator;
  not verified.
- **Verdict (Vienna):** same shape as Paris — good network geometry, counts at a handful of points, no routes,
  no OD. Usable as a *negative control* city or for the geometry, not as a density target.

---

## 4. Generic GPS-trace sources

### 4.1 OpenStreetMap public GPS traces — VERIFIED WORKING, and the best fallback for Amsterdam

- **`planet.gpx` bulk dump is stale — last build 2013** (`wiki.openstreetmap.org/wiki/Planet.gpx`). Regional
  extracts exist but lag. Tooling: `github.com/iandees/planet-gpx-dump`.
- **But the live API works and I tested it.** `GET https://api.openstreetmap.org/api/0.6/trackpoints?bbox=<W,S,E,N>&page=<n>`
  returns GPX. Test over central Amsterdam `bbox=4.88,52.36,4.90,52.38&page=0`: **HTTP 200, 522,280 bytes,
  5,000 `<trkpt>` elements**, each with lat/lon and an ISO timestamp, grouped into `<trk>` with a trace name,
  description and uploader URL. 5,000 points per page; page through until empty. Bbox is capped at 0.25 deg².
  Amsterdam municipality (~4.72–5.07 E, 52.28–52.43 N) is ~1–2 tiles, so a full sweep is feasible but you must
  rate-limit politely.
- Licence: **ODbL** (OSM contributor terms), attribution + share-alike, commercial use allowed.
- Coverage/years: everything ever uploaded publicly, 2005→now; the sample I pulled included a trace from
  June 2026.
- **The catch:** traces are **unlabelled by mode.** There is no cycling flag. The dump mixes walking, cycling,
  driving, boats, flights and GPS noise. You would have to infer mode from speed/acceleration statistics per
  segment — doable (and defensible as a methods contribution), but it is inference, not ground truth. Volume
  is also self-selected toward OSM mappers, i.e. heavily biased to mapping-worthy streets.
- **Verdict: viable fallback and a good cross-check.** Map-match traces to the OSM street graph, count
  traversals per segment, and you get a route-density raster over Amsterdam with a clean licence. It is
  strictly worse than Fietstelweek as a *cycling* target (no mode label, no count normalisation) but strictly
  better in licence terms, and it is live rather than archaeological. Strong candidate for a second,
  independent target to show the model is not overfitting one data-generating process.

### 4.2 Negatives

- **GeoLife** — Beijing only, as expected. Not relevant.
- **Mobile Data Challenge (Nokia/Lausanne)** — access was always by application and the programme has long
  since closed; no open bike-labelled release. Not pursued further.
- **2020–2026 open bike-trajectory datasets** — I found none that is both open and covers Amsterdam/
  Barcelona/Paris/Vienna. The recent literature that uses large cycling-GPS corpora
  (e.g. 134,169 trips / 6,523 cyclists; the Rotterdam map-matching study; the MDPI bicycle-delay paper on
  "nationwide sparse GPS data") consistently works from **Fietstelweek, Snuffelfiets or proprietary
  app data**, and none of those papers publishes the trajectories. This is worth stating plainly in the
  README: there is effectively **one** open, network-wide, per-segment Dutch cycling-intensity dataset, and
  its host is dead.

---

## 5. Recommendation

1. **Pull all 12 Wayback zips now** (~850 MB) before the Internet Archive's holdings or availability change —
   IA was intermittently returning "Temporarily Offline" during this very session. Verify each with `unzip -t`.
   Keep a checksummed local mirror; consider re-publishing the Amsterdam clip (the licence *requires* derived
   products be released as open data, so this is aligned, not a risk) with clear provenance.
2. **Primary target**: `INTENSITEIT` per link, 2015 + 2016, clipped to Amsterdam, EPSG:28992, rasterised.
   Secondary channels: `SNELHEID_R` and node `TIJD`.
3. **Second, independent target**: OSM public GPS traces via the trackpoints API, map-matched, mode-inferred.
   Clean ODbL, live, and it lets you claim the method is not tied to one dead dataset.
4. **Do not** build the artifact on Bicing/Vélib/WienMobil — they are station-status, not routes.
5. **Open gaps I did not close**: DANS EASY / Data Station, figshare, Mendeley, Dataverse, OSF, Dryad (no
   direct API query); the UGent Move CKAN fietstelweek entry (host refused HTTPS); NDW's fiets open-data
   endpoint specifics; whether `bussche.d@buas.nl` still receives the notification the licence asks for; and
   full-file integrity of the three largest archived zips.
