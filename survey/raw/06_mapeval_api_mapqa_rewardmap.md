# MapEval-API, MapQA, RewardMap — "a public metric we do not have to build"
**Survey note 06 — compiled 11 Sep 2026.** Follow-up to `05_mapagent_spatialagent_peria_deepdive.md`.

Scope: can we adopt an existing public benchmark as the external, third-party metric for our artifact, instead of only reporting numbers on our own Amsterdam generator?

> **Verification status.** Section A facts marked "verified locally" were obtained by downloading the actual artefacts on this machine: the HF dataset (`MapEval/MapEval-API`, 300 rows), the MapEval-API evaluation harness source, and the 62.5 MB MapQaTor Postgres dump that *is* the offline cache. Leaderboard numbers come from the arXiv HTML render of the papers and are marked **[paper-table, re-check against PDF]**. Anything I could not confirm is marked **[UNVERIFIED]**.

---

## 0. Recommendation up front

**Adopt MapEval-API as the external metric for Stage 2/3. Do not adopt it as the primary metric, and do not design the task around it.**

Reasons, in order of weight:

1. It is the only public map-agent benchmark that is **fully offline, licence-clean enough to run, tool-based, and has a published leaderboard with frontier models, open models, two agent scaffolds (MapAgent, Spatial-Agent) and a human number** — i.e. every comparison row we would want, already populated by other people.
2. Running it costs **€0 in map-API spend and no Google key** (verified locally: the cache is a public Postgres dump; the `/custom` tool endpoints read from it). Cost is LLM inference only — for our own 3–8B model on the rig or on the Hetzner box, effectively free.
3. It is **textual-only** (no map image in the API split) and its tool schema is Google-Places-shaped, so our agent will need a thin adapter. The adapter is cheap (5 tools, place_id-keyed) — **but it means MapEval-API measures our model's *tool-use policy and compositional reasoning*, not our vision or our metric-geometry capability.** That is exactly the right scope for an OOD generalisation claim and exactly the wrong scope for a headline claim.
4. Honest caveat we must state in the README: PERIA's OOD gain on MapEval was **+1.5 points**. Expect a small number. Plan to report it either way.

Companion recommendation: **MapQA is the better template for our generator** (OSM + SQL templates + verifiable answers) but a weak eval for us (CC BY-**NC**, free-form entity-name answers, SoCal/Illinois only, no released generator code). **RewardMap is the best available code reference for our GRPO stage** (MIT, VeRL, GRPO, difficulty-aware composite reward, actually runs) — read its reward and curriculum code, do not use its data.

---

# Part A — MapEval-API in detail

## A1. Who made it, and is it alive?

**MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models**
Mahir Labib Dihan\*, Md Tanvir Hassan\*, Md Tanvir Parvez, Md Hasebul Hasan, Md Almash Alam, Muhammad Aamir Cheema, Mohammed Eunus Ali, Md Rizwan Parvez (\*equal contribution).
arXiv [2501.00316](https://arxiv.org/abs/2501.00316) — v1 31 Dec 2024, v2 6 Jun 2025 — **ICML 2025 (Spotlight)**, [PMLR v267 dihan25a](https://proceedings.mlr.press/v267/dihan25a.html). Paper licence **CC BY 4.0**.

Institutions: **BUET** (Bangladesh University of Engineering and Technology), **QCRI / HBKU** (Qatar), **Monash University**, Islamic University Bangladesh, Bangladesh Computer Council. Acknowledgement names **QCRI** for API access and GPU compute. No other funder is listed. Corresponding: `mahirlabibdihan@gmail.com`, `mohammed.eunus.ali@gmail.com`, `mparvez@hbku.edu.qa`.

This is the same BUET/QCRI group that produced **MapQaTor** (the annotation tool that built the data, [arXiv 2412.21015](https://arxiv.org/abs/2412.21015), ACL 2025 Demo) and **MapAgent** ([arXiv 2509.05933](https://arxiv.org/abs/2509.05933), Findings of EACL 2026 — see note 05). So there is a live research programme around it, even though the benchmark itself is frozen.

**Follow-up versions / v2 / 2026 update: none found.** Searched arXiv, the project site, the GitHub org and the HF org. The benchmark is at v2 of the *paper* (Jun 2025), not v2 of the *data*. **[UNVERIFIED that no private v2 exists — but nothing public.]**

**Maintenance status — honest read (verified locally via GitHub API):**

| Repo | Stars | Last push | Licence |
|---|---|---|---|
| [MapEval/MapEval-API](https://github.com/MapEval/MapEval-API) | 2 | **25 Nov 2024** | **none** |
| [MapEval/MapEval-Textual](https://github.com/MapEval/MapEval-Textual) | 1 | 28 Mar 2025 | **none** |
| [MapEval/MapEval-Visual](https://github.com/MapEval/MapEval-Visual) | 4 | 2 May 2025 | **none** |
| [mapeval.github.io](https://mapeval.github.io/) | — | 8 Jan 2026 | — |
| [mapqator/mapqator-backend](https://github.com/mapqator/mapqator-backend) (the cache) | 0 | 28 Sep 2025 | **none** |

**Effectively unmaintained as code; alive as an artefact.** The HF datasets carry **Apache-2.0** on their cards; every GitHub repo, including the one holding the cached database, has **no LICENSE file** (= all rights reserved by default). The MapQaTor *paper* claims Apache-2, but no LICENSE file exists in the org. Practical consequence: **run their code, do not vendor it.** Our harness should be our own code reading the HF dataset.

**Leaderboard:** [mapeval.github.io](https://mapeval.github.io/) hosts static tables (Tables 1–3 of the paper), not a live submission leaderboard. There is no submission process. Anyone reporting a MapEval number is self-reporting.

## A2. The API split, precisely

### Items and categories (verified locally, `datasets.load_dataset("MapEval/MapEval-API")`)

- **300 items**, one split (`test`), one config (`benchmark`). File size 136 kB.
- Category counts — these match the paper's Table 4 exactly:

| `classification` | count |
|---|---|
| `nearby` | 83 |
| `trip` | 67 |
| `routing` | 66 |
| `poi` (Place Info) | 64 |
| `unanswerable` | 20 |
| **total** | **300** |

- **Option counts:** 295 items have 4 options, 3 items have 3, 2 items have 2.
- `id` range 34–593 (sparse; ids are MapQaTor database keys, not indices).

### Field schema (verified locally)

```
id             : int64          # MapQaTor query id
question       : string
options        : list[string]   # 2-4 strings, NO "unanswerable" option included
answer         : int64          # 1-BASED index into options; 0 == "unanswerable"
classification : string         # poi | nearby | routing | trip | unanswerable
```

The 1-based convention is confirmed two ways: the answer-index distribution is `{0:20, 1:74, 2:73, 3:71, 4:62}` — exactly 20 zeros, matching the 20 `unanswerable` items — and the harness builds ground truth as `item["answer"]["correct"] + 1`.

**There is no tool/context field in the API split on HF.** The cached tool responses live in a separate database (see A2.3). What *is* available per item, in the sibling dataset, is the fully rendered textual context — see next.

### The Textual/API relationship — an important and underdocumented fact

**Verified locally: `MapEval-Textual` and `MapEval-API` are the same 300 questions.** All 300 `id`s overlap; 297 of 300 question strings are byte-identical (3 differ by minor rephrasing); the category and answer-index distributions are identical.

The difference is purely the information channel:

- **MapEval-Textual** hands the model a pre-rendered `context` string (140–11.7k chars) containing everything needed.
- **MapEval-API** gives the model *nothing* and makes it fetch the same facts through 5 tools.

That is a genuinely useful property for us: **(Textual accuracy − API accuracy) on identical items isolates tool-use ability from reasoning ability.** That is a clean, cheap ablation axis for our post-training story, and nobody in notes 05/06 has reported it as such.

### The 5 cached tools — exact schemas (verified locally from `Tools.py` / `FormattedTools.py`)

All five are LangChain `BaseTool`s pointed at `http://localhost:5000/api` (the MapQaTor backend), with a hardcoded JWT in the header. Everything is keyed on **Google `place_id` strings**, not coordinates.

| Tool | Inputs (pydantic) | Backend route | Output returned to the agent |
|---|---|---|---|
| `PlaceId` (a.k.a. Text/Place Search) | `placeName: str` | `GET /map/search?query=` | a single `place_id` string (first result), or `"Incorrect place name. Please use the same name as in the question."` |
| `PlaceDetails` | `placeId: str` | `GET /map/details/custom/{id}` | dict, nulls stripped: `place_id, name, formatted_address, phone_number, geometry{location{lat,lng}}, price_level (Free…Very Expensive), opening_hours (weekday_text[]), rating, user_ratings_total, delivery, dine_in, reservable, takeout, serves_beer/breakfast/brunch/dinner/lunch/vegetarian_food/wine, wheelchair_accessible_entrance` |
| `TravelTime` (Distance Matrix) | `originId, destinationId, travelMode ∈ {driving, walking, bicycling, transit}` | `GET /map/distance/custom` | **a single human-readable duration string** (e.g. `"11 mins"`) — `matrix[0][0].duration.text`. Distance is *not* returned by this tool. |
| `Directions` | `originId, destinationId, travelMode` | `GET /map/directions/custom` | `routes[]`, each `{label, duration ("11 mins"), distance ("4.8 mi"), steps[] (HTML-marked-up turn instructions)}` |
| `NearbyPlaces` | `placeId, type (from a fixed 99-value Google type vocabulary), rankby ∈ {prominence, distance} (default distance), radius: int\|None` | `GET /map/nearby/tool` | `results[]` of `{place_id, name, opening_hours, price_level, rating, user_ratings_total}` |

Notes that matter for an adapter:
- `NearbyPlaces` enforces Google's own constraint: `rankby=distance` forbids `radius`; the tool returns a scolding string rather than results if you violate it. **No distance value is returned with nearby results** — only an implicit ordering.
- `type` must come from `types.json`, a **99-entry Google Places type list** (`restaurant`, `atm`, `mosque`, `subway_station`, …). It is not free text and it is not OSM tagging.
- The whole API surface is **place_id-centric**: you cannot pass lat/lng anywhere. Coordinates only *come out* (via `PlaceDetails.geometry`).

### How "offline cached" actually works (verified locally — this is the important part)

The cache is **public and downloadable**: [`mapqator/mapqator-backend/database/dump.sql`](https://github.com/mapqator/mapqator-backend/blob/main/database/dump.sql), 62,550,214 bytes. Their README says it explicitly:

> "If you want the cached data that was used for MapEval benchmark you should use: `psql -U postgres -d mapqator -a -f database\dump.sql`"

Row counts I measured in that dump, against what the paper claims:

| Table | rows in public dump | paper's figure |
|---|---|---|
| `places` (PlaceDetails) | **11,943** | 13,354 |
| `distance` (TravelTime) | **964** | 1,142 |
| `directions` | **222** | 317 |
| `nearby` / `nearby_places` | **415** / 6,455 | 481 |
| `dataset` (questions + contexts) | 316 | 300 (+16 deleted) |
| `human` (human annotations) | 230 | — |
| `evaluations` (all model verdicts) | 18,675 | — |
| `models` | 22 | — |

**The public dump is an earlier snapshot than the paper's final cache** (roughly 10–30% fewer rows per table). It also covers only **261 of the 300** API question ids in its `dataset` table. So: mostly reproducible offline, **not bit-exactly reproducible**. Say so if we report a number.

**Does it need a Google key? No — but the code needs patching.** The `/custom` and `/tool` routes are cache-first, and `routes/mapRoutes.js` only injects a key if `GOOGLE_MAPS_API_KEY` is set in `.env`. With no key set, two handlers are broken at HEAD (verified by reading `controllers/mapController.js`):

1. `searchText` (backing the `PlaceId` tool) — the no-key branch is `else if (local.success …)` but `const local = await mapRepository.searchText(...)` is declared *inside* that branch → **ReferenceError / TDZ**. Fix: hoist the `const local` above the `if`.
2. `getDetailsCustom` — the local-cache lookup is **commented out** at HEAD, so with no key it returns `400 "Can't find the place in the local database"`. Fix: uncomment the block.

That's a ~5-line patch to run the whole thing with zero external spend. `searchNearbyTool`, `getCustomDirections`, `getDistanceCustom` are already cache-first and work as-is.

**Hygiene warning:** `dump.sql` contains a `users` table with a `google_maps_api_key` column (7 rows) and password hashes, and `Tools.py` ships a hardcoded JWT. Do not re-publish the dump, and do not commit it to our repo.

### Cities / countries

The paper reports **180 cities across 54 countries for the full 700-item benchmark**; it does **not** break this down per split. Spot-checking the API split shows Cusco (PE), Columbus OH (US), Copenhagen/Nørreport (DK), Greece, Middle-Eastern markets — i.e. genuinely global, heavily weighted to tourist-legible places. **[Per-split city/country counts: UNVERIFIED — not reported.]**

### Answer format and scoring rule (verified locally from `Evaluator2.py`)

Prompt construction, verbatim logic:

```
<question> + "Choose the answer from the following options (1/2/3/4).
So, the output format will be \"^^Option_Number^^\".
Choose the correct answer from the following options: "
   [if the item is unanswerable:]  "Option0: Unanswerable, "
   "Option1: <opt>, Option2: <opt>, ..."
```

So **"Option0: Unanswerable" is only offered on the 20 unanswerable items** — it is not a standing option. That is a leak: a model that notices the option list length can infer the item is a trap. Minor (20/300) but worth knowing.

Scoring: the agent is a LangChain `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` ReAct loop over the 5 tools; the final answer is parsed with `re.search(r"\^\^(.*?)\^\^")` and the **first digit character** in that capture is taken as the choice. Verdicts:

- `right` — parsed option == ground truth
- `wrong` — parsed a different non-zero option
- `invalid` — (a) no `^^…^^` match, (b) the agent crashed / hit a parsing error or dead loop, (c) the model answered `0` on an answerable item, **and (d) any response at all on an item whose ground truth is 0** — the harness marks unanswerable items `invalid` at this stage and resolves them elsewhere.

Metric is plain **accuracy = correct / 300**. **Invalid counts as not-correct** — there is no refusal credit, no partial credit, no tie handling. A model that fails to emit `^^n^^` is simply wrong. **Implication for us: format compliance is part of the score.** A small model that can't hold the `^^n^^` protocol over an 11-turn ReAct loop will be scored as wrong, which is a real and legitimate part of what the benchmark measures — and something GRPO format reward directly trains.

### Licence and download

- Data: HF cards say **Apache-2.0** for all three splits. Paper: **CC BY 4.0**.
- Code: **no licence anywhere** (MapEval org, MapQaTor org).
- Underlying content is **derived from Google Maps Platform responses**; an Apache-2.0 dataset card does not override Google's ToS. Standard practice for research evaluation; **do not** use it as a source for anything we redistribute. (Our own data must be OSM/ODbL — repeat of note 05's conclusion.)
- Download: `load_dataset("MapEval/MapEval-API")` → [HF MapEval/MapEval-API](https://huggingface.co/datasets/MapEval/MapEval-API). Cache: [dump.sql](https://github.com/mapqator/mapqator-backend/blob/main/database/dump.sql).

## A3. Five real examples, verbatim

All five printed directly from the downloaded HF dataset. Remember: `answer` is **1-based**, `0` = unanswerable. The API split ships **no context** — I have added the item's cached tool payload from the MapQaTor dump for the `routing` example to show what the tools actually return.

**(1) `trip` — id 479**
```json
{
  "id": 479,
  "question": "I am currently staying at Hostel El Grial in Cusco, Peru. I want to visit Sacsayhuamán for 1.5 hours, Qorikancha Temple for 1 hour, the Cusco Cathedral for 1.5 hours, and the Mercado Central de San Pedro for 1 hour. I have 6 hours available. I will leave my hostel at 8 am. Give the most optimized order to visit these places. I will drive my car.",
  "options": [
    "Saqsaywaman -> Cusco Cathedral -> Mercado Central de San Pedro -> Qorikancha (35 mins)",
    "Cusco Cathedral -> Qorikancha -> Mercado Central de San Pedro -> Saqsaywaman (37 mins)",
    "Saqsaywaman -> Qorikancha -> Mercado Central de San Pedro -> Visit Cusco Cathedral (48 mins)",
    "Mercado Central de San Pedro -> Qorikancha -> Cusco Cathedral -> Saqsaywaman (33 mins)"
  ],
  "answer": 1,
  "classification": "trip"
}
```
(Note the option strings embed the total travel time — so this is TSP-with-opening-hours *and* an arithmetic check.)

**(2) `routing` — id 334**
```json
{
  "id": 334,
  "question": "I am driving to Brassica in Bexley Via E Whittier St. After reaching Alum Creek Dr, where should I go next?",
  "options": [
    "Turn left onto College Ave",
    "Turn right onto E Main St",
    "Turn right onto E Whittier St",
    "Turn right onto US-33 E/E Livingston Ave"
  ],
  "answer": 2,
  "classification": "routing"
}
```
Its cached tool context, from `dataset.context_json` in the dump (abridged — the full payload contains WALKING/DRIVING/BICYCLING/TRANSIT route sets):
```json
{"distance_matrix":{},
 "places":{"ChIJg6iPe8WPOIgRwxmos7YS8Lo":{"selectedAttributes":["formatted_address"]},
           "ChIJZ81nMVuIOIgRyPDZM29rCAo":{"selectedAttributes":["formatted_address"]}},
 "nearby_places":{},
 "current_information":{"time":null,"day":"","location":"ChIJg6iPe8WPOIgRwxmos7YS8Lo"},
 "directions":{"ChIJg6iPe8WPOIgRwxmos7YS8Lo":{"ChIJZ81nMVuIOIgRyPDZM29rCAo":{
   "DRIVING":{"routes":[
     {"label":"I-70 E","duration":"11 mins","distance":"4.8 mi","steps":[ ... ]},
     {"label":"E Whittier St","duration":"12 mins","distance":"4.1 mi","steps":[
        "Head <b>north</b> on <b>S High St</b> toward <b>Shumacher Alley</b>",
        "Turn <b>right</b> onto <b>E Whittier St</b>",
        "Turn <b>left</b> onto <b>Lockbourne Rd</b>",
        "Turn <b>right</b> onto <b>US-33 E</b>/<wbr/><b>E Livingston Ave</b>",
        "Turn <b>left</b> onto <b>Alum Creek Dr</b>",
        "Turn <b>right</b> onto <b>E Main St</b>",
        "Turn <b>left</b> onto <b>College Ave</b>..."]},
     {"label":"E Main St","duration":"13 mins","distance":"4.0 mi","steps":[ ... ]}],
    "showSteps":true}}}}}
```
**Critical observation: the origin ("South Wind Motel") appears nowhere in the question.** It is hidden environment state (`current_information.location`). In the Textual split it is handed over in the context ("Current location of user is **South Wind Motel**."). In the API split the agent has no way to learn it from the 5 tools. A non-trivial number of routing/nearby items are **underspecified without the textual context**, which is one plausible reason API scores sit below Textual scores. **Any adapter we build must decide how to inject this "current location" state.** [I did not quantify how many items are affected — **UNVERIFIED**, worth a 20-minute script if we adopt the benchmark.]

**(3) `nearby` — id 357**
```json
{
  "id": 357,
  "question": "I'm at Khansa Market and feeling really hungry. Where can I get something to eat quickly?",
  "options": ["Shawarma Juha", "Souq Al-Lail Al-Qadim Restaurant",
              "Kfteraa Cham Shawarma Arabic on coal ", "Torch Restaurant"],
  "answer": 4,
  "classification": "nearby"
}
```

**(4) `poi` (Place Info) — id 589**
```json
{
  "id": 589,
  "question": "Which pair of places is farthest from each other in Greece?",
  "options": ["Athens and Thessaloniki", "Athens and Crete",
              "Thessaloniki and Corfu", "Athens and Rhodes"],
  "answer": 4,
  "classification": "poi"
}
```
This one is the contamination exhibit: **answerable from pretraining world knowledge with no tool call at all.** See A4(d).

**(5) `unanswerable` — id 497**
```json
{
  "id": 497,
  "question": "Is there any publicly accessible lake around Nørreport where people can arrange a program?",
  "options": ["Udekørende fysioterapi v/Elie Poulsen", "Haveskriver",
              "Christian Liljedahl", "Madsen Jørgen Lund"],
  "answer": 0,
  "classification": "unanswerable"
}
```
The distractors are plausible-looking but unrelated business names pulled from the cache. The correct behaviour is to emit `^^0^^` (Option0: Unanswerable, which the harness appends for these items only).

## A4. Current leaderboard on the API split

**MapEval paper, Table 4** (direct ReAct agent over the 5 tools) **[paper-table, re-check against PDF]**:

| Model | Overall | Place Info | Nearby | Routing | Trip | Unanswerable |
|---|---|---|---|---|---|---|
| **Claude-3.5-Sonnet** | **64.00** | 68.75 | 55.42 | 65.15 | 71.64 | 55.00 |
| GPT-4-Turbo | 53.67 | 62.50 | 50.60 | 60.61 | 50.75 | 25.00 |
| GPT-4o | 48.67 | 59.38 | 40.96 | 50.00 | 56.72 | 15.00 |
| Gemini-1.5-Pro | 43.33 | 65.63 | 30.12 | 40.91 | 34.33 | 65.00 |
| Gemini-1.5-Flash | 41.67 | 51.56 | 38.55 | 46.97 | 34.33 | 30.00 |
| Llama-3.2-90B *(best open, paper)* | 39.67 | 54.69 | 37.35 | 39.39 | 35.82 | 15.00 |
| Llama-3.1-70B | 37.67 | 53.13 | 32.53 | 42.42 | 31.34 | 15.00 |
| Mixtral-8x7B | 27.67 | 32.81 | 18.07 | 27.27 | 38.81 | 15.00 |
| GPT-3.5-Turbo | 27.33 | 39.06 | 22.89 | 33.33 | 19.40 | 15.00 |
| Gemma-2.0-9B | 27.00 | 35.94 | 14.46 | 28.79 | 26.87 | 45.00 |
| GPT-4o-mini | 23.00 | 28.13 | 14.46 | 13.64 | 43.28 | 5.00 |

**With agent scaffolds (from note 05)** — note these mix backbones, so the columns are not apples-to-apples:

| System | Backbone | MapEval-API |
|---|---|---|
| **MapAgent** (EACL Findings 2026) | Qwen-2.5-72B | **~72%** (best published) |
| MapAgent | GPT-3.5-Turbo | ~70% |
| **Spatial-Agent** (ACL 2026) | GPT-5 | **71.88** |
| Spatial-Agent | Qwen2.5-72B | 53.41 *(best open-source in that paper)* |
| Spatial-Agent | Qwen-14B, SFT+DPO | **60.58** ← the most relevant row for us |
| Spatial-Agent | Qwen-14B base | 49.59 |
| Spatial-Agent | GPT-4o-mini | 45.15 (vs 23.00 direct) |
| Direct (no scaffold), best | Claude-3.5-Sonnet | 64.00 |

**Human:** the paper reports **86.67% human accuracy on MapEval-Textual** and does *not* report a separate human number for the API split (humans were given the textual context, so the Textual number is the operative ceiling for these 300 questions).

*Derived cross-check, my own computation, treat with caution:* the public dump's `human` table has 230 annotations from **3 annotators**; scored against the HF ground truth, 198 matched annotations give **75.76%** overall (nearby 84.13, poi 100.00, routing 72.73, trip 72.31, **unanswerable 11.11** on 9 items). This is *below* the published 86.67, almost certainly because the public dump predates the final annotation round. **[UNVERIFIED — do not quote this number; quote 86.67 with the Textual qualifier.]** The interesting part is qualitative and probably robust: **humans are also bad at the unanswerable category.**

Summary for our purposes: **the band we would be competing in is 23–72%.** A 14B SFT+DPO model reaches 60.6. A 3–8B model post-trained on *our* Amsterdam task and evaluated zero-shot on MapEval-API would plausibly land somewhere in the 25–50 range; anything above the ~40 mark would be a genuinely reportable result against Llama-3.2-90B's 39.67.

## A5. Assessment for our project

### (a) Can our agent be evaluated on it under OUR tool API? — Adapter needed, and it is buildable.

**No, not directly.** MapEval-API's protocol is: 5 tools, **place_id-keyed**, Google Places type vocabulary, human-readable string outputs ("11 mins", "4.8 mi"), ReAct loop, `^^n^^` answer format. Our planned tools (per note 05: `geocode`, `place_search`, `directions`, `distance_matrix`, `nearest`, `within_radius`, `haversine`, `bearing`, `open_at_time`, …) are **coordinate-keyed, OSM-tagged, and return numbers**.

An adapter in the *other* direction is the right move: **wrap the MapQaTor cache behind our own tool signatures.** That is the honest design — we evaluate our model under the interface it was trained on, against a world backed by their cached data.

Is the cached data rich enough? Mostly:

| Our tool | Backable by the cache? |
|---|---|
| `geocode(name) -> id` | Yes — `places.name` / `formatted_address` text match. |
| `place_details(id)` | Yes — 11,943 rows with hours, rating, address, and **`geometry.location` lat/lng**. |
| `haversine(a,b)`, `bearing(a,b)`, `bearing_to_direction` | **Yes — computable from cached coordinates.** This is a free win: we can serve *metric* answers the original benchmark could not. |
| `directions(o,d,mode)` | Only for the **222** cached (origin, destination, mode) triples. Any other pair → miss. |
| `distance_matrix(o,d,mode)` | Only for the **964** cached pairs. |
| `nearest`, `within_radius` | Partially — `nearby` has 415 cached (location, type, rankby, radius) queries; arbitrary radii are not answerable. Can be partly synthesised from cached coordinates. |
| `open_at_time(place, t)` | Yes — `opening_hours.weekday_text` is cached. |
| `tsp_tw` | Yes on top of the cached matrix, for the trip items. |

**The blocker is the sparse routing/distance cache**: 222 direction rows and 964 distance rows for 300 questions means the cache was populated *exactly* for the annotated question paths, and only 261/300 ids are present in the public dump's `dataset` table. Our agent will make tool calls the annotators never made, and get cache misses. Two honest options:
- **(i)** Report on the subset that the public cache fully supports, stating the subset size. Clean, defensible, smaller n.
- **(ii)** Back the misses with our own OSM/OSRM world for those cities. More work, and it breaks comparability with published numbers — so it would be a *separate, clearly-labelled* number, not a leaderboard claim.

I recommend **(i)** plus, in parallel, the trivially cheap **MapEval-Textual** run (no tools at all, just long-context MCQ — a genuine zero-infrastructure external number we can report on day one).

Plus the hidden-state problem from A3(2): we must decide how "current location" is injected. Simplest defensible rule: **inject `current_information` as a system-prompt preamble for every item, for every system we compare.** State it in the README.

### (b) Is it textual only? — Yes.

MapEval-**API** has **no images**. Zero visual grounding. Our VLM will be evaluated on it as a text model with tools. That is fine, but it means **MapEval-API cannot validate the vision half of our artifact.** MapEval-**Visual** is the image split (see A6).

### (c) What it does NOT test that we care about

| Capability we want to claim | Tested by MapEval-API? |
|---|---|
| Metric distances in metres, with a checkable number | **No.** Distances come back as `"4.8 mi"` strings on cached routes only; no tool returns a computed metric distance. Answers are MCQ, so precision is never scored. |
| Bearings / cardinal relations ("is A north of B") | **No.** No tool, no question type. |
| Visual grounding on a rendered map | **No.** (Visual split, separate.) |
| Layers: noise, shadow, elevation, land use, greenness | **No, entirely absent.** Google Places has no such attributes. This is our differentiator and the benchmark simply cannot see it. |
| Reading geometry (polygons, street network topology) | **No.** Places are points; routes are step strings. |
| Multi-turn tool-use policy under a fixed API | **Yes — this is its strength.** |
| Compositional/multi-hop reasoning over fetched facts | **Yes.** |
| Abstention / knowing when the world model is insufficient | **Yes**, weakly — 20 items, with an option-count leak. |
| Route planning with time windows (TSP-TW) | **Yes** — the 67 `trip` items are the best part of the benchmark. |

**So: MapEval-API tests roughly the "tool policy" axis of our artifact and none of the "world model / geometry / vision" axes.** Use it as an *OOD transfer* metric, never as the headline.

### (d) Contamination and memorised world knowledge — a real and quantifiable risk

The benchmark is built on **real, famous, tourist-legible places**, published **December 2024**, with the questions in plain text on HF and GitHub. Three distinct exposures:

1. **Direct test-set contamination.** Any model with a post-Dec-2024 cutoff may have seen the questions *and answers* (the HF dataset is public and the GitHub `dataset.json` includes `answer.correct`). Claude-3.5-Sonnet (Jun 2024 snapshot) predates it; GPT-5, Qwen3, and anything we would use as a teacher **do not**. **If we use a frontier teacher to synthesise SFT trajectories, we must not let it touch MapEval items** — keep MapEval strictly held out from every stage of training data generation.
2. **World-knowledge shortcut.** Example (4) above — "Which pair of places is farthest from each other in Greece?" — is answerable from pretraining alone. An unknown fraction of the 64 `poi` items are like this. A model can score without using a single tool, which **inflates the score of large, well-read models and deflates the measured value of good tool use.** A cheap and genuinely interesting diagnostic we could run and report: **score our model and baselines with the tools disabled**, and report the gap. That "no-tool baseline" number is, as far as I can find, **not published by anyone** for MapEval-API — it would be a small original contribution costing ~€1.
3. **Temporal drift.** The cache is a Nov-2024 Google snapshot. Opening hours, ratings and business existence have moved on. The paper concedes this ("more suitable for archival purposes"). Harmless for us since we use the cache, not live data — but it means the benchmark is a *frozen world*, which is philosophically the same design choice as our Amsterdam snapshot. That's a nice line for the README.

**Our own generator is structurally immune to (1) and (2)** — Amsterdam OSM snapshot, programmatically generated questions, answers computed from geometry. That contrast is worth making explicitly, and MapEval-API is the foil that makes it concrete.

### (e) Is it a good "metric we don't have to develop", and for which stage?

**Yes, with a narrow scope and stage-gated.**

| Stage | Use MapEval? | What for |
|---|---|---|
| **Stage 0 — before any training** | **Yes, immediately.** | Run **MapEval-Textual** (no infra: 300 MCQs + context, pure prompting) on our chosen base model (Qwen2.5-VL-3B/7B or Qwen3-VL) and on a frontier model. ~€2 and one evening. Anchors the story before a single GPU hour is spent, and tells us whether the base model can even hold an MCQ format. |
| **Stage 1 — SFT** | **Set up, don't optimise.** | Stand up the patched MapQaTor backend + our tool adapter. Report base-vs-SFT on MapEval-API as a secondary number. |
| **Stage 2/3 — GRPO** | **Yes — this is the payoff.** | "Trained only on synthetic Amsterdam, our 7B model scores X on MapEval-API, a third-party ICML 2025 benchmark we did not build, vs Y for the untrained base, Z for Llama-3.2-90B and 64.0 for Claude-3.5-Sonnet." That sentence is the portfolio asset. |
| **Headline metric** | **No.** | Our own Amsterdam/Rotterdam/Utrecht held-out generator remains the primary metric — it is the only thing that measures metric geometry, layers and visual grounding. |

**Expectation management, to be written into the README before we see the number:** PERIA's tool-use post-training moved MapEval OOD by **+1.5 points**. A large transfer gain is not the base case. The honest framing is "does Amsterdam-only RLVR transfer to a Google-Maps-shaped world at all?", and **a small or zero gain, reported clearly, is still portfolio-grade** under the ground rules.

**Effort estimate to make MapEval-API runnable end-to-end:** restore the dump (~15 min), patch the two handlers (~15 min), write our tool adapter + ReAct harness reading the HF dataset (~1 day), first eval pass (< 1 h wall clock, €0 map spend). Call it **1.5 days**. Cheap for what it buys.

## A6. MapEval-Visual (400 items) — brief

- **Format** (from the HF card): 400 rows; fields `context` (relative path to a PNG in `Vdata/`), `question`, `options`, `answer` (int), `classification`, `url` (the source Google Maps URL). Images ship as a separate `Vdata.zip` (~345 MB). Categories: **Place Info 121, Nearby 91, Counting 88, Routing 80, Unanswerable 20**. Note "Counting" replaces "Trip" relative to the other splits.
- **Are they Google Maps screenshots? Yes** — literally, with the source Google Maps URL retained per item; zoom levels 8.0–21.0, mean 15.26.
- **Licence implications.** Card says Apache-2.0; the images are Google Maps screenshots, so the card does not and cannot grant rights in the underlying content. Evaluating on them is normal research practice. **We must not** put these images in our training data, fine-tune on them, or redistribute them, and we must not render our own data in a Google-lookalike style. Our data: OSM/ODbL, rendered by us.
- **Can a small VLM be evaluated fairly on it? Yes — and this is a pleasant surprise.** I downloaded three images and measured them: **1166×868, 1770×820, 1050×540** (RGBA PNG, 0.4–1.1 MB). That is *ordinary* resolution — nothing like ReasonMap's 5839×5449 average. A 3–8B VLM at 1024–1536 px native tiling can read these without heroic tiling tricks. **[Sample of 3 — a full distribution scan would take 10 minutes if we adopt it.]**
- Published ceiling: Claude-3.5-Sonnet **61.65**, best open Qwen2.5-VL-72B **60.35**, MapAgent+Qwen-2.5-VL-72B **72.30**, human **82.23** (per the HF card).
- **Verdict:** MapEval-Visual is the only public benchmark in this survey that tests *our* combination of "small VLM + map image + MCQ" at a resolution we can actually serve. **Recommend running it alongside MapEval-API in Stage 2/3.** But it is map-as-picture with no tools and no world — it will not reward our tool policy, only our visual reading.

---

# Part B — MapQA (arXiv 2503.07871, SIGSPATIAL 2025)

**MapQA: Open-domain Geospatial Question Answering on Map Data.** Zekun Li, Malcolm Grossman, Ehsan Qasemi, Mihir Kulkarni, Muhao Chen, Yao-Yi Chiang. arXiv [2503.07871](https://arxiv.org/abs/2503.07871), 10 Mar 2025; [ACM SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3764174). Data licence **CC BY-NC-SA 4.0** (paper) / **CC BY-NC 4.0** (repo README) — the two statements disagree; either way **non-commercial**.

### How the 3,154 questions are generated

**Nine SQL query templates run against a PostGIS database loaded from an OSM dump (downloaded 18 Sep 2023).** The queried table is the standard `osm2pgsql` **`planet_osm_point`** table, using columns `name`, `osm_id`, the amenity-type tag, and the `way` geometry column; spatial predicates are PostGIS functions (`ST_Within`, distance ordering, etc.).

Regions: **Southern California** (606,773 entities, 621 amenity types) and **Illinois** (92,415 entities, 175 amenity types).

An LLM (GPT) is used **only for paraphrasing** — each template is rewritten into several surface phrasings to add linguistic diversity. The *answers* come from SQL, not from the LLM. That is exactly our generator design.

Splits: SoCal **2,206** pairs (1,765 train / 441 test); Illinois **948** pairs, used as a **zero-shot region-transfer set**. Per-type counts are not itemised in the paper. 175 candidate geo-entity types.

### The nine question types, with an example each

| # | Reasoning type | Example question | Answer format |
|---|---|---|---|
| 1 | Adjacency | "What restaurant is adjacent to Luther Burbank Savings?" | geo-entity name |
| 2 | Amenities within a distance | "What are the bars within 50 m of South Park Brewing?" | geo-entity name(s) |
| 3 | Amenities in a radius (type inference) | "What amenities are within a 100 m radius of Yorba Rose?" | amenity type(s) |
| 4 | Type identification | "What amenity is available at Port Police?" | amenity type |
| 5 | Proximity comparison | "Which is closer to ChargePoint, Chicken Maison or Ospi?" | geo-entity name |
| 6 | Directional nearest | "What is the nearest snack_cart west of Colours Wheelchairs?" | geo-entity name |
| 7 | Nearest amenity | "What is the nearest clinic to East Wing?" | geo-entity name |
| 8 | Nearest to a street junction (multi-hop) | "Which driving_school is nearest to the junction of 10th & 10th West?" | geo-entity name |
| 9 | Distance calculation | "How far is 119th & Laflin from 100 Forest Place?" | numeric, metres |

**Answer format:** free-form — entity name, amenity type, or a number in metres. **Not MCQ.** Scoring: Recall@{5,10,50} for retrieval baselines, exact-match accuracy for LLM/text-to-SQL baselines; for type 9 a prediction counts as correct if the **offset is under 100 m**. Each QA pair is coupled to the **geometry** of the referenced geo-entities, not just a textual description — that is the paper's headline claim over GeoQuestions201/1089.

Baselines (**[paper-table, re-check]**): DPR-BERT and DPR-GeoLM retrieval are weak on adjacency (R@50 of 2–4%) and strong on type identification (82–88%); LLM text-to-SQL (GPT-4o) is near-perfect on type 4 and collapses on the multi-hop types 8–9. Spatial-Agent later reported **61.45%** on MapQA vs 13.55% for direct GPT-4o-mini (note 05).

### Is the generator public?

- **Dataset: yes** — [github.com/knowledge-computing/MapQA-dataset](https://github.com/knowledge-computing/MapQA-dataset) (19 stars, 9 commits, last push 10 Mar 2026, **no LICENSE file** despite the README stating CC BY-NC 4.0). Layout: `llm/` (QA files in 9 categories, plus `amenities.csv` and `entities.csv`) and `retrieval/` (train/test in retrieval format).
- **Generator code and the PostGIS database: no.** The paper says a script exists for generating more data; the repo does not ship the SQL templates, the osm2pgsql setup, or a DB dump. **The nine templates in the paper are the specification we would reimplement.**

### Usable as (i) an eval for us?

**Weakly — do not prioritise.** Reasons: non-commercial licence (so we cannot include it in anything we might later want to release un-encumbered, and it constrains how we present it); free-form string answers whose scoring is brittle for a small model; SoCal/Illinois only, so it is a US-suburb distribution far from Amsterdam; and it is designed for text-to-SQL / retrieval, not for a tool-using VLM. If we run anything here, run the **Illinois zero-shot set** as a pure region-transfer probe, and report accuracy with an explicitly-stated matching rule.

### Usable as (ii) a template for our Amsterdam generator?

**Yes — this is its real value to us, and it is high.** It is the published precedent that "SQL/Overpass templates over an OSM snapshot + LLM paraphrase = a legitimate, peer-reviewed QA dataset". Concretely we should:

- **Mirror the nine reasoning types** so our generator has a comparable, citable taxonomy axis for free — and then **extend past them**, which is where our contribution shows: MapQA has no routing, no opening-hours/temporal reasoning, no trip planning, no visual grounding, and its only metric question (type 9) is point-to-point distance with a 100 m tolerance.
- **Steal the 100 m tolerance idea** and generalise it: graded/continuous correctness for metric answers is exactly the shaped reward RLVR wants (cf. PERIA's NDTW clipping).
- **Keep the geometry coupling** — MapQA's stated novelty is attaching geo-entity geometries to each QA pair. We get that by construction, and it is what makes our rewards verifiable.
- **Use `planet_osm_point` / osm2pgsql + PostGIS** as the proven stack, rather than inventing one. (Decision for OPEN_DECISIONS: osm2pgsql+PostGIS vs a local Overpass instance. MapQA is evidence the Postgres path works.)

---

# Part C — RewardMap

**RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning.**
Sicheng Feng, Kaiwen Tuo, Song Wang (Zhejiang University), Lingdong Kong (NUS), Jianke Zhu, Huan Wang (Westlake / Tongji).
[arXiv 2510.02240](https://arxiv.org/abs/2510.02240) (3 Oct 2025) — **ICLR 2026**. Project page [fscdc.github.io/RewardMap](https://fscdc.github.io/RewardMap/), code [github.com/fscdc/RewardMap](https://github.com/fscdc/RewardMap).

**Maturity (verified via GitHub API): MIT licence, 47 stars, 2 forks, last push 22 Feb 2026.** Small but real, actively touched, properly licensed — **the only genuinely reusable RL codebase in this entire survey besides verl-tool.**

**What it is.** Not an agent and **not tool-using**: a single-pass VLM answering questions about high-resolution transit/subway map images. The contribution is the *training recipe*, which is precisely the part we need.

- **Model:** trained on **Qwen2.5-VL-7B-Instruct**; evaluated across Qwen2.5-VL 3B/7B/32B/72B, Kimi-VL-A3B, Seed1.5-VL. **LoRA on the language blocks and the vision projector** (not full-parameter) — directly relevant to our 96 GB single-card budget.
- **Tools:** **none.** No tool sandbox, no map backend.
- **Rewards — the interesting bit.** Composite and difficulty-weighted:
  `R = W_difficulty · (R_format + R_correctness + α · R_detail)`, with **α = 0.5**.
  - `R_format`: output-convention compliance (boxed answer).
  - `R_correctness`: exact match for the VQA tasks; the official ReasonMap route-evaluation algorithm for route planning.
  - **`R_detail`: partial credit** for getting origin/destination stops, route names, transfer stations and segment counts individually right. **This is the sparse-reward fix** — instead of one binary bit per episode, a structured answer earns graded credit per component.
  - `W_difficulty`: per-map difficulty weights `{γ_e, γ_m, γ_h}` (easy/medium/hard) × a per-question weight from transfer count `{β_0, β_1}`.
- **Curriculum:** multi-stage, simple perception → counting → planning, under a "global curriculum principle" (ordered stages) plus a "local stochasticity principle" (randomisation within a stage to avoid overfitting the order).
- **Algorithm/framework:** **GRPO in VeRL**; SFT via LLaMA-Factory; eval via VLMEvalKit; derived from Seg-Zero. Hyperparameters: AdamW, lr 1e-6, KL coef 1e-3, batch 16, **8 responses per query**.
- **Data:** `ReasonMap-Train` **696** samples, `ReasonMap-Plus-Train` **2,570**, `ReasonMap-Plus-Test` **1,448** (ReasonMap-Plus totals 4,018 questions over 30 cities / 13 countries). Released on HF under **Apache-2.0** for academic research; `utils/download_dataset.py` fetches them.
- **Hardware: 8× H800.** More than we have — but with LoRA, batch 16 and 8 rollouts, a scaled-down single-96 GB-card version is plausible.
- **Results (Qwen2.5-VL-7B) [paper-table, re-check]:** ReasonMap weighted 26.22 → **31.51**; ReasonMap-Plus 67.61 → **74.25**; average across 6 benchmarks **+3.47**.

**What is reusable for our GRPO stage (ranked):**

1. **The `R_detail` partial-credit pattern.** Our answers are structured too (route, stop list, POI set, distance) — decomposing correctness into independently-rewarded components is the single cheapest fix for GRPO's zero-variance-group problem, and it is better motivated than a reward-shaping hack because each component is independently verifiable from OSM geometry.
2. **Difficulty-aware reward weighting.** Our generator *knows* each item's difficulty by construction (hop count, number of constraints, POI density). Weighting by it is nearly free and directly ablatable.
3. **The multi-stage curriculum** (perception → counting → planning) maps cleanly onto ours (read the map → count/locate → route/plan), and the "local stochasticity" caveat is a useful detail to copy.
4. **The actual code**, MIT-licensed and runnable: a working GRPO-on-a-VLM VeRL setup with LoRA on language blocks + vision projector, plus their difficulty weighting and reward functions. Even if we end up on verl-tool (which we need for multi-turn tool loops, which RewardMap does not have), **RewardMap is the better first thing to read and run** — it is smaller, licensed, and single-turn.
5. Its **ReasonMap-Plus** data (Apache-2.0) is usable as an extra external eval, but at 5839×5449 average resolution ReasonMap proper is out of reach for a small VLM without tiling — noted in OPEN_DECISIONS already.

**Not reusable:** the tool loop (there isn't one), the world model (there isn't one), and the data domain (transit diagrams, not geographic maps).

---

## Source list

- **MapEval** — [arXiv 2501.00316](https://arxiv.org/abs/2501.00316) · [PMLR v267 dihan25a](https://proceedings.mlr.press/v267/dihan25a.html) · [mapeval.github.io](https://mapeval.github.io/) · [HF MapEval-API](https://huggingface.co/datasets/MapEval/MapEval-API) · [HF MapEval-Textual](https://huggingface.co/datasets/MapEval/MapEval-Textual) · [HF MapEval-Visual](https://huggingface.co/datasets/MapEval/MapEval-Visual) · [GitHub MapEval/MapEval-API](https://github.com/MapEval/MapEval-API)
- **MapQaTor (the cache + annotation tool)** — [arXiv 2412.21015](https://arxiv.org/abs/2412.21015) · [ACL 2025 Demo](https://aclanthology.org/2025.acl-demo.1/) · [mapqator.github.io](https://mapqator.github.io/) · [GitHub mapqator/mapqator-backend](https://github.com/mapqator/mapqator-backend) · [database/dump.sql](https://github.com/mapqator/mapqator-backend/blob/main/database/dump.sql)
- **MapQA** — [arXiv 2503.07871](https://arxiv.org/abs/2503.07871) · [SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3764174) · [GitHub knowledge-computing/MapQA-dataset](https://github.com/knowledge-computing/MapQA-dataset)
- **RewardMap** — [arXiv 2510.02240](https://arxiv.org/abs/2510.02240) · [fscdc.github.io/RewardMap](https://fscdc.github.io/RewardMap/) · [GitHub fscdc/RewardMap](https://github.com/fscdc/RewardMap) · [HF FSCCS/ReasonMap collection](https://huggingface.co/collections/FSCCS/reasonmap)
- **MapAgent** — [arXiv 2509.05933](https://arxiv.org/abs/2509.05933) · [ACL Anthology 2026.findings-eacl.67](https://aclanthology.org/2026.findings-eacl.67/)
- **Spatial-Agent** — [arXiv 2601.16965](https://arxiv.org/abs/2601.16965) · [ACL Anthology 2026.acl-long.679](https://aclanthology.org/2026.acl-long.679/)
