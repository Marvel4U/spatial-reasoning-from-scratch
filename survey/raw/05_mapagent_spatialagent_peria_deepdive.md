# Deep dive: MapAgent / Spatial-Agent / PERIA
**Survey note 05 — compiled 11 Sep 2026**
Scope: practical, hands-on read of three prior works, for (a) possible reproduction/branching and (b) use as baselines for our "small VLM + map tools + GRPO/RLVR on real-city snapshots (Amsterdam first)" artifact.

> **Verification status.** All three papers were read via automated extraction of the arXiv HTML (`arxiv.org/html/...`) plus GitHub/HuggingFace API metadata. Numbers in the tables below were lifted from those renders; **before any number goes into a README or an interview answer, re-check it against the PDF table.** Items I could not verify at all are marked **[UNVERIFIED]**.

---

## 0. One-paragraph orientation

The three works sit on a spectrum that maps almost exactly onto our design space:

| | MapAgent | Spatial-Agent | PERIA |
|---|---|---|---|
| Modality | text + some VLM | text-only reasoning over map APIs | **vision-first (VLM)** |
| Method | prompting only (multi-agent scaffold) | prompting + *optional* SFT+DPO | **SFT → RL (OR-GIGPO)** |
| Tools | 4 coarse Google Maps tools | ~35 fine-grained geo operators | 18 *image* tools (OCR/detect/crop/draw) |
| Backend | live Google Maps Platform | Google Maps (+ OSM/Overpass fallback, cache) | none — tools act on the image |
| Verifiable reward | n/a | n/a (DPO on graph well-formedness) | **yes — binary + NDTW continuous** |
| Code | public, thin, unlicensed | not found | not found |

**Nobody has done our exact thing.** MapAgent and Spatial-Agent are *prompted* agents over a *commercial* map API with no policy learning. PERIA *does* the policy learning, has verifiable rewards, and is a small VLM — but its tools manipulate *pixels*, not a world model; it has no map backend at all. Our artifact = PERIA's training recipe x Spatial-Agent's tool granularity x a **local, offline, OSM-derived city snapshot** instead of Google Maps. That gap is the contribution and it is defensible in an interview.

---

## A. MapAgent

**MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration**
Md Hasebul Hasan, Mahir Labib Dihan, Tanzima Hashem, Mohammed Eunus Ali, Md Rizwan Parvez (BUET / QCRI / HBKU).
arXiv [2509.05933](https://arxiv.org/abs/2509.05933) (v1 7 Sep 2025, rev. Oct 2025, 27 pp) - **Findings of EACL 2026** ([ACL Anthology 2026.findings-eacl.67](https://aclanthology.org/2026.findings-eacl.67/)) - [OpenReview](https://openreview.net/forum?id=T4H85lY19w) - paper licence **CC BY 4.0**.

### A1. What it does

A **hierarchical, prompting-only multi-agent scaffold**. Explicit design claim: flat tool-agent architectures (Chameleon, OctoTools, ReAct) overwhelm the LLM when many map APIs are *similar but subtly different* (e.g. Nearby Search vs Text Search vs Place Details). So MapAgent **decouples planning from execution**:

1. **Planner agent** — decomposes the query into subgoals, routes each to a *module* (not a raw tool).
2. **Map-Service module** — itself an agent (the **Map-Tool Agent**) that adaptively orchestrates the map APIs, **in parallel** where independent (see `parallel_function_implementation.py` in the repo).
3. **Lightweight modules** — solution generator, answer extractor — plain LLM calls, no agent overhead.
4. Supporting components: `src/coordinatoragent/{planner,executor,memory,initializer,formatters}.py`.

**Tool set (4 tools over Google Maps Platform):**

| Tool | Underlying API(s) | Inputs | Outputs |
|---|---|---|---|
| `Trip` | Place Details + Directions | `current_location`, `visiting_places[]`, `travel_mode` | place metadata + step-by-step route instructions for a multi-stop itinerary |
| `Route` | Directions | `origin`, `destination`, `travel_mode`, `alternatives` | distance, estimated duration, navigation steps |
| `Nearby` | Nearby Search | `query`, `location`, `type`, `radius` | place names, ratings, metadata within radius |
| `PlaceInfo` | Place Details | `location_address` | address, opening hours, contact, reviews |

There is also an object-detection tool under `src/coordinatoragent/tools/` used for the visual split.

**Backend: Google Maps Platform only.** No OSM/Overpass path.
**Training: none.** Pure prompting. This is the single most important fact for us — MapAgent is a *scaffold* paper, not a *learning* paper.
**Backbones:** GPT-3.5-Turbo and Qwen-2.5-72B (text); GPT-4o and Qwen-2.5-VL-72B (visual). All via API; no fine-tuning.

### A2. What it can and can't do

Benchmarks: MapEval-Textual, MapEval-API, MapEval-Visual, MapQA.

| Dataset | GPT-3.5-Turbo | Qwen-2.5-72B | GPT-4o | Qwen-2.5-VL-72B |
|---|---|---|---|---|
| MapEval-API | ~70% | ~72% | — | — |
| MapEval-Textual | 72.94% | 76.24% | — | — |
| MapEval-Visual | — | — | 68.95% | 72.30% |
| MapQA | 55.58% | 55.93% | — | — |

Gains over the OctoTools baseline: roughly +10 pts (API), +10 (Textual), +11.22 (MapQA), +4.41 (Visual).

Worth noting: these numbers, with comparatively weak backbones, already beat the MapEval paper's best *unscaffolded* frontier model (Claude-3.5-Sonnet, 64–66%). **The scaffold is worth ~10–20 pts — which is exactly the headroom an RL-trained small model has to compete for.**

**Reported failure modes** (their Table 8; framework-attributable error 2.62–4.36% depending on split):
- Planner module-selection errors: 0.67–2.5%.
- Map-Tool Agent errors (wrong tool, wrong parameter passing): 0.76–1.78%.
- Hallucination when irrelevant parameters are passed to a tool.
- Difficulty decomposing complex multi-hop queries.

Note the arithmetic: overall accuracy is ~70% but attributed *framework* error is only ~4%, so the residual ~26% is reasoning/answer error that their taxonomy does not cover. Be careful not to over-read their error analysis.

**Cost**: reported cheaper than Chameleon/OctoTools on LLM calls, token count and wall-clock, with comparable Google Maps API call counts (Appendix D; specific figures not extracted).

### A3. Can we run it?

- **Code**: [github.com/Hasebul/MapAgent](https://github.com/Hasebul/MapAgent) — **17 stars, 2 forks, 8 commits**, created 21 Jun 2025, **last push 9 Jan 2026**, **NO LICENCE FILE**. That means "all rights reserved" by default: fine to read, run and learn from; **not safe to vendor into our repo**.
- Layout: `src/coordinatoragent/{engine/{base.py,openai.py}, models/, tools/}`, `datasets_dir/{txt_data,img_data,map_qa_data}`, `requirements.txt`, `parallel_function_implementation.py`. **The datasets are vendored in the repo** (`txt_data`: nearby/poi/routing/trip/unanswerable; `img_data` incl. `VData`; `map_qa_data` task_1..task_9 each with `data.json` / `problems.json` / `pid_splits.json`).
- **Keys needed**: OpenAI API key (paid account) **+ Google Maps API key**.
- **Run**:
  ```sh
  cd src
  python run.py --policy_engine gpt-4 --kr_engine gpt-4 --qg_engine gpt-4 --sg_engine gpt-4 \
                --test_split test --test_number -1 \
                --data_root '../datasets_dir/txt_data/trip' --output_root '../new'
  ```
  Four separately configurable engines: policy, knowledge-retrieval, query-generation, solution-generation.
- **Backbone swap**: the engine layer is `engine/base.py` + `engine/openai.py` (a Chameleon-derived abstraction). Adding `engine/openai_compatible.py` pointed at a **vLLM OpenAI-compatible endpoint**, or at Claude, is roughly a one-hour job. This is the cheapest possible way to get a strong *prompted* baseline for our benchmark. **No MCP server** exists for this or any of the three.
- **Cost of one eval pass**: 300 questions (Textual or API) x ~4–8 LLM calls ≈ 1.5–2.5k calls. On a GPT-4o-mini-class model that is **well under $5**; on gpt-4/4o, **$10–40**. Google Maps spend only applies to live splits — the MapEval Textual and API splits are cached (see A4), so those can be run with **zero Google spend**. For reference, 2026 Google Maps pricing is ~$2–$40 per 1k requests by SKU (Places Basic ≈ $17/1k, Advanced ≈ $32/1k) and **Google has removed the universal $200/month credit**; per-SKU free thresholds remain. Budget **EUR 0–30** for one full pass.

### A4. Datasets (these matter more to us than the code)

**MapEval** ([arXiv 2501.00316](https://arxiv.org/abs/2501.00316); [github.com/MapEval](https://github.com/MapEval)) — 700 MCQs, 180 cities, 54 countries, built with the **MapQaTor** annotation web tool ([arXiv 2412.21015](https://arxiv.org/abs/2412.21015)):

| Split | Size | Licence (HF card) | Content |
|---|---|---|---|
| [MapEval-Textual](https://huggingface.co/datasets/MapEval/MapEval-Textual) | 300 rows, 964 kB | **Apache-2.0** | `context` = pre-fetched, human-readable Google Maps info (140–11.7k chars), plus `question`, `options` (2–4), `answer` (index), `classification` (5 types) |
| [MapEval-API](https://huggingface.co/datasets/MapEval/MapEval-API) | 300 rows, 136 kB | **Apache-2.0** | agent gets 5 Google Maps tools — **Text Search, Place Details, Distance Matrix, Directions, Nearby Search** — served from a **cached offline database of 13,000+ locations**, not live APIs. Categories: Place Info / Nearby / Routing / Trip / Unanswerable |
| [MapEval-Visual](https://huggingface.co/datasets/MapEval/MapEval-Visual) | 400 rows, 345 MB | **Apache-2.0** | `Vdata.zip` of **Google Maps screenshots** at zoom 8.0–21.0, source URLs retained; categories POI / Nearby / Routing / Counting / Unanswerable |

Reported ceiling: best model Claude-3.5-Sonnet at 66.33 (Textual) / 64.00 (API) / 61.65 (Visual); **human 86.67 (Textual)** — a ~20-point human gap. No model in their 30-model sweep exceeded 67%.

**Licence caveat, flagged honestly**: the HF cards say Apache-2.0, but MapEval-Visual images are *Google Maps screenshots* and Textual/API content is *derived from Google Maps API responses*. An Apache-2.0 label on the dataset card does not override Google's Terms of Service for the underlying content. For **evaluation/research use this is standard practice and fine**; for anything we *redistribute* (our own rendered-map dataset) we must render from **OSM**, not Google. This is also a positive differentiator for our artifact: an OSM/ODbL snapshot is redistributable, Google tiles are not.

The three `MapEval/*` GitHub repos are **unlicensed** and effectively unmaintained (1 / 4 / 2 stars; last pushes Mar 2025 / May 2025 / Nov 2024). Treat the HF datasets as the artefact, not the code.

**MapQA** ([arXiv 2503.07871](https://arxiv.org/abs/2503.07871); [ACM SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3764174)): **3,154 QA pairs generated by SQL query templates over OpenStreetMap** for Southern California + Illinois; 175 geo-entity types; nine geospatial reasoning types (adjacency, amenities, amenities-around, amenities-around-specific, compare-closer, distance, type identification, neighbourhood inference, …); QA pairs are coupled to **geo-entity geometries**, not just textual descriptions; LLM used to diversify question phrasing.

**MapQA is the most relevant dataset in this entire survey for us**: OSM-grounded, programmatically generated, verifiable answers, redistributable lineage. It is published prior art that our generator design — Overpass/SQL templates over a city snapshot producing verifiable QA — is a legitimate, publishable construction.

---

## B. Spatial-Agent

**Spatial-Agent: Agentic Geo-spatial Reasoning with Scientific Core Concepts**
Riyang Bao, Cheng Yang, Dazhou Yu (Emory), Zhexiang Tang (Rutgers), Gengchen Mai (UT Austin), Liang Zhao (Emory, corresponding).
arXiv [2601.16965](https://arxiv.org/abs/2601.16965) (23 Jan 2026, 15 pp, 4 figs) - **ACL 2026 Long Papers** ([2026.acl-long.679](https://aclanthology.org/2026.acl-long.679/), pp. 14896–14911).

### B1. What it does

Frames geo-QA as a **concept-transformation** problem grounded in spatial information science. NL question -> **GeoFlow Graph** `G = (V, E, lambda, rho)`, a DAG where:
- concept space `C = {Location, Object, Field, Event, Network, Amount, Proportion}` (`lambda` labels nodes)
- functional role space `R = {Extent, TExtent, SubCond, Cond, Support, Measure}` (`rho` assigns roles), with precedence `SubCond < Cond < Support < Measure`
- five well-formedness constraints: **G1** acyclicity; **G2** role ordering along edges; **G3** type compatibility (output type of `v_i` matches input type of `v_j`); **G4** data availability (every edge realisable by an operator); **G5** connectivity from Extent nodes through to Measure nodes.

Five-stage pipeline: (1) spatial-information-theory analysis = concept + role extraction -> (2) concept-transformation drafting via retrieval of a **macro-template** -> (3) GeoFlow Graph construction under constraints -> (4) graph factorisation and operator mapping -> (5) execution and grounded response generation.

**Template library (10 macros)**: Filter-Aggregate-Measure; Object-Field-Measure; Route-Optimize; Geocode-Batch-Compare; Location-Bearing-Classify; Route-Step-Extract; Multi-Route-Compare; Place-Attribute-Query; Multi-Segment-Aggregate; Time-Window-Reverse. Retrieved by cosine similarity between the input question embedding and stored question-graph pairs.

**Operator library (~35 operators — the single most reusable artefact in this paper):**

- *Geocoding*: `geocode(text, anchor, region_hint)` -> (phi, lambda), progressive fallback 10km -> 50km -> 100km; `batch_geocode(names[], anchor)`; `reverse_geocode(phi, lambda)`
- *Places*: `place_search(center, radius_m, type, keyword, min_rating)`; `place_details(place_id)` -> {name, coords, rating, price, hours, phone}; `batch_place_details(places[])` (cache-first)
- *Routing*: `directions(origin, dest, mode in {driving, walking, transit, bicycling}, waypoints)` -> {legs, steps, distance, duration}; `distance_matrix(O[], D[], mode)` -> |O|x|D| of (distance, duration); `compare_routes(routes[], metric in {distance, duration})` -> best index; `filter_routes(routes[], keyword)` (stairs / toll / roundabout); `extract_distance(route)`; `extract_duration(route)`
- *Geometry*: `haversine(phi1, l1, phi2, l2)` (R = 6371 km); `bearing(...)` -> 0–360 deg forward azimuth; `bearing_to_direction(theta)` -> {N, NE, E, SE, S, SW, W, NW}
- *Spatial analysis*: `nearest(anchor, candidates[], metric in {haversine, travel_time})`; `within_radius(center, radius_m, candidates[])`; `pairwise_extremes(locations[])` -> max-distance pair; `filter_places(places[], constraints)` (rating / type / open)
- *Temporal*: `open_at_time(place, datetime)` -> bool (handles cross-midnight); `timezone(phi, lambda, unix_ts)` -> {tz_id, name, utc_offset}; `calculate_finish_time(t0, locations[], stay_durations[], mode)`
- *Trip optimisation*: `tsp_tw(dist_matrix, locations[], service_times[], time_windows[], t0, budget)` -> visit sequence, via **Google OR-Tools** with greedy fallback if infeasible; `steps_analysis(route, landmark)` -> turn counts / manoeuvre stats
- *Local context (cache-first, API fallback, 6 operators)*: `query_local_place`, `query_local_coordinates`, `query_local_routes`, `query_local_travel_time`, `query_local_places_batch`, `query_local_nearby_places`

**Backend**: Google Maps API primary (geocoding, places, directions, distance matrix); **Overpass/OpenStreetMap as fallback** for local context; a local cache database; Google OR-Tools for TSP-with-time-windows.

**Training (presented as optional, two-stage — the interesting bit for us):**
- **Stage 1, SFT** on concept extraction: `L_SFT = -sum log p_theta(V_i | q_i)` — learn to emit concepts with types and functional roles.
- **Stage 2, DPO** on graph well-formedness: preference pairs `(G+, G-)` where `G+` satisfies all five constraints and `G-` violates at least one; standard DPO loss with temperature beta.
- **The preference signal is *structural validity*, not answer correctness** — a cheap, fully programmatic verifier. Directly relevant to our reward design: this is a free auxiliary verifiable reward we can stack alongside answer correctness.

### B2. Results

**MapEval-API** (Place Info / Nearby / Routing / Trip):
- GPT-5: **71.88%** overall (Place Info 85.94, Nearby 53.01, Routing 75.76, Trip 77.61)
- Qwen2.5-72B: **53.41%** (best open-source)
- GPT-4o-mini: **45.15%** vs 23.00% direct (+96.3% relative); per-category relative gains +149.9% Place Info, +133.3% Nearby, +122.1% Routing
- Template ablation: without templates, GPT-4o-mini drops to **39.32%** (-12.9% relative)

**MapQA** (3,154 questions):
- Direct LLM (GPT-4o-mini) 13.55% -> ReAct 43.79% -> Reflexion 53.79% -> **Spatial-Agent 61.45%**; LLaMA-70B 62.45%, Qwen2.5-72B 61.45%.

**SFT/DPO ablation (Qwen-14B, MapEval-API):**

| Configuration | Overall | Relative change |
|---|---|---|
| Base model | 49.59% | — |
| SFT only | 56.84% | +14.6% |
| DPO only | 55.13% | +11.2% |
| **SFT + DPO** | **60.58%** | **+22.2%** |

A 14B model at 60.6 against GPT-5 at 71.9. **This is the closest existing evidence for our artifact's core thesis — post-training takes a small model most of the way to a frontier model on map tasks — and it is only DPO, not RL.** Cite it.

**Failure analysis (68 incorrect predictions):**
- Data quality issues (incomplete/missing API data): **45.6%**
- Search-result mismatch (API returns irrelevant results): **33.8%**
- Concept & role assignment errors: 10.3%
- Response generation (correct execution, wrong option selected): 10.3%
- **Graph construction errors: 0%** (the template library eliminates them)

Authors' own conclusion: *the bottleneck is external API reliability, not reasoning logic.* **This is the strongest single argument for our offline, complete, local world snapshot: it removes ~79% of their error mass by construction.**

**Latency and cost (GPT-4o-mini):** direct LLM 0.6 s; Spatial-Agent 7.5 s (routing), 8.3 s (nearby), 10.4 s (trip); Reflexion 12–14 s. Token use 9,185 in / 1,451 out ≈ **$0.0022/query**; all compared methods under $0.003/query. A full MapEval-API pass therefore costs roughly **$0.70**.

### B3. Can we run it?

- **No code repository found.** Searched arXiv, ACL Anthology, GitHub and author pages; nothing surfaced. **[UNVERIFIED whether code will ever be released — treat as closed-source.]**
- However the appendices are unusually complete: Appendix C (operator specifications), D (fine-tuning details), E (template library), plus execution pseudocode. **Reimplementing the operator library from the paper is a 1–2 day job** and that is what I recommend — we would target our own OSM snapshot anyway, so we would need to rewrite the backend regardless.
- Keys: to run *theirs* you would need Google Maps + an LLM provider. To run *our reimplementation*: **zero external keys**.
- Backbone swap: trivially yes — they already evaluate Gemma-2-9B, Qwen2.5-32B/72B, LLaMA-70B, GPT-3.5-Turbo, GPT-4o-mini, GPT-5.

---

## C. PERIA

**Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning**
Changye Li (Tsinghua), Meng Lu (Virginia Tech), Yi Wu (Tsinghua), Ligeng Zhu (NVIDIA).
arXiv [2606.12830](https://arxiv.org/abs/2606.12830), 11 Jun 2026, cs.CV. PERIA = **PER**ception-**I**nteraction-re**A**son agent.
*(Note: an unrelated 2024 robotics paper also called PERIA lives at peria-for-robotics.github.io — not this work.)*

**This is by far the closest work to our artifact and the one to study line by line.**

### C1. What it does

VLM agent framed as a POMDP: iterative `<think>` / `<action>` / `<answer>` generation, tools executed in a sandbox, observations (including **newly produced images**) appended to the context. Up to **11 turns** (at most 10 interaction turns). The agent first gathers global spatial evidence (perceive), then does fine-grained local verification (interact), then reasons over accumulated observations. All spatial arguments live in a **normalised 1000x1000 coordinate space**.

**Tool sandbox — 18 tools, all operating on images. There is no map/geo backend at all.**

*Perception tools (12) — extract structured visual evidence:*
`text_ocr(image_index)` - `text_spotting(image_index)` - `map_text_ocr(image_index)` - `formula_ocr(image_index)` - `table_ocr(image_index)` - `grounding_dino(image_index, question)` - `auto_segment(image_index)` - `bbox_segment(image_index, bounding_box)` - `text_segment(image_index, text_prompt)` - `exemplar_segment(image_index, bounding_box)` - `concept_count(image_index, text_prompt)` - `presence_check(image_index, text_prompt)`

*Interaction tools (6) — manipulate visual context:*
`image_crop(image_index, bounding_box)` - `image_label(image_index, text, position)` - `draw_line(image_index, coordinates)` - `draw_path(image_index, points)` - `bounding_box(image_index, bounding_box)` - `image_highlight(image_index, bounding_box)`

*Tool implementations:* **PaddleOCR-VL-1.5 (0.9B), GroundingDINO-base, SAM 3.1** — small, local, GPU-resident models, not paid APIs. Good news for us: the tool stack is cheap and self-hostable.

**Backbones:** Qwen3-VL-Thinking **2B / 4B / 8B**. Inference via **vLLM**; training via **Verl-Tool**.

### C2. Training pipeline

**Stage 1 — SFT on synthesised tool-use trajectories.**
- Trajectories synthesised by a proprietary model (**GPT-5**) using **explore-and-exploit sampling under an increasing turn budget**: `n_sample = 3` candidate trajectories per round, up to `T = 11` rounds; the model is *forced* to select a tool, the tool executes, observations accumulate, then it answers. A round terminates when the trajectory matches ground truth, else the budget increases. Effectively rejection sampling against public ground truth.
- **Reasoning diversification**: an open-weight expert, **Qwen3-VL-235B-A22B-Thinking**, rewrites the intermediate *thoughts* while keeping tool calls and observations fixed — decorrelating reasoning style from the GPT-5 teacher.
- Hyperparameters: lr 1e-5, batch 32, **1 epoch**, cosine scheduler.

**Stage 2 — RL with OR-GIGPO (Observation-Relaxed Group-in-Group Policy Optimization).**
- GiGPO computes step-level advantages by grouping *identical states* across rollouts. In multimodal tool use, states contain images and are essentially **never exactly identical**, so the grouping collapses. OR-GIGPO replaces exact state matching with **semantic similarity, threshold delta = 0.9**. Measured effect: *Average Step Advantage Ratio* rises from **9.3% (GiGPO) to 62.3% (OR-GIGPO)**.
- Episode-level and step-level advantages combined with weight **omega = 1.0**.
- Hyperparameters: lr 1e-6, batch 64, **3 epochs**, **N = 4 rollouts per prompt**; max prompt 16,384 tok, max response 32,768 tok, max action 4,096 tok, max observation 8,192 tok.

**Composite reward — fully verifiable, no reward model:**
1. **Repetition penalty** (repeated characters / words / spans),
2. **Format reward** (well-formed `<think>` / `<action>` / `<answer>` tags),
3. **Correctness**: binary for most tasks; **continuous for MapTrace** via normalised Dynamic Time Warping:
   `R_correct = min{1, max{0, (d_high - d_NDTW) / (d_high - d_low)}}`, with `d_low = 0.3`, `d_high = 0.8`.

**Ablations:**

| Variant | Effect (in-distribution) |
|---|---|
| w/o RL (SFT only) | -7.2 |
| w/o SFT (RL from base) | substantially worse |
| w/o tools | imbalance / catastrophic forgetting |
| OR-GIGPO vs GiGPO | +2.8 |
| OR-GIGPO vs GRPO | +2.4 |
| OR-GIGPO vs DAPO | +11.1 |
| w/o perception tools | -11.1 |
| w/o interaction tools | -6.7 |

### C3. Results

| Model | In-dist avg | OOD avg | Overall |
|---|---|---|---|
| Qwen3-8B (base) | 44.1 | 31.7 | 37.4 |
| **PERIA-8B** | **54.1 (+10.0)** | **36.1 (+4.4)** | **44.4 (+7.0)** |
| Qwen3-235B | 49.7 | 39.6 | 44.3 |
| GPT-5 | 55.8 | 39.1 | 46.8 |

PERIA-8B beats comparable-size baselines (VTool-R1-7B, R1-OneVision-7B, Mini-o3) by 7.0–14.8 points.

Map-relevant per-benchmark rows (Table 2 — **verify against the PDF before quoting**):

| Benchmark | Qwen3-8B | PERIA-8B | Delta |
|---|---|---|---|
| ReasonMap | 12.3 | **27.1** | +14.8 |
| ReasonMap-Plus | 65.7 | **72.6** | +6.9 |
| MapTrace | 74.1 | **77.7** | +3.6 |
| MapEval (OOD) | 54.0 | **55.5** | +1.5 |

Headline for us: **an 8B model matches a 235B model overall and closes most of the gap to GPT-5, purely from tool-use post-training.**

Equally important, the honest weak spot: **the OOD gain is only +4.4 overall, and just +1.5 on MapEval.** The learned tool policy is substantially in-distribution-specific. We should expect the same, plan held-out cities from day one, and report it either way.

**Failure-mode diagnosis** (300 cases where GPT-5 succeeds and Qwen3-VL-Thinking fails):
- **Tool-call omission: 55.9–71.6%** (the model simply does not call the tool it needs)
- **Tool-induced errors: 25.0–60.9%** (the model is misled by a tool's output later in reasoning)
- Format errors: 0.6–2.3%

Their framing — *tool use is a policy-learning problem, not a tool-access problem* — is effectively the thesis sentence for our artifact.

### C4. Reproducibility (question 4, answered directly)

- **Code: not released. Weights: not released. Data: not released.** No GitHub link, no HuggingFace link, no project page, and **no "we will release" statement** anywhere in the paper or in search results. **[UNVERIFIED — worth re-checking in a few weeks; NVIDIA-affiliated papers often release later.]**
- **Training-set sizes are never stated** in the paper — only "in-distribution trajectories plus additional map QA data from MapQA". **No GPU count, GPU type, or training hours are reported either.** That is a genuine reproducibility hole and we should say so plainly in our write-up rather than pretending the recipe is fully specified.
- **What the tool trajectories are synthesised from**: the *training splits of the public benchmarks themselves* — MapTrace, ReasonMap, ReasonMap-Plus, Visual Probing, MapQA — with GPT-5 as the explorer/teacher and Qwen3-VL-235B-A22B-Thinking as the thought-rewriter, rejection-sampled against public ground truth. **No proprietary data.** In principle reproducible by anyone with a GPT-5 API budget.
- **The underlying datasets ARE public:**
  - [google/MapTrace](https://huggingface.co/datasets/google/MapTrace) — **CC-BY-4.0**, ~2M annotated paths (the paper uses a **20k subset**); synthetic map images generated by text-to-image models (brochure maps, park directories, shopping malls; simpler office / apartment / campus floor maps); targets are **normalised coordinate lists in [0,1]**. Code: [google-research/MapTrace](https://github.com/google-research/MapTrace).
  - **ReasonMap / ReasonMap-Plus** — high-resolution transit (mostly subway) maps from 30 cities, 1,008 QA pairs in the base set, two question types and three templates; **average image resolution 5839 x 5449 px**; **Apache-2.0**, academic research use; [FSCCS HF collection](https://huggingface.co/collections/FSCCS/reasonmap). ReasonMap-Plus adds counting, true/false and route questions. Related and *public* RL work on the same data: [fscdc/RewardMap](https://github.com/fscdc/RewardMap) (ICLR 2026, [arXiv 2510.02240](https://arxiv.org/html/2510.02240)) — a second, fully open reference for multi-stage RL on map VLMs that we should read next.
  - MapQA and MapEval as in section A4.
- **The training framework is public**: [TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) — **MIT licence**, 1,000+ stars, ~758 commits, active into mid-2026 (TMLR 2026; ICLR 2026 SPOT best paper). Built on verl, with actor rollout fully decoupled from environment interaction, a native **tool-server** design for multi-turn loops, vLLM 0.11 / SGLang backends, OpenAI-compatible endpoints for eval, and documented GRPO / DAPO / Search-R1 recipes. **OR-GIGPO itself is not in it** — we would implement the delta=0.9 observation-relaxed grouping ourselves on top of GiGPO/GRPO.

**Could we reproduce the map-reasoning subset on one 96GB GPU (Hetzner GEX131, RTX PRO 6000 Blackwell, EUR 2/h)?** Honest assessment:

- **SFT of an 8B VLM with LoRA/QLoRA at 96GB: yes, comfortably.** Full-parameter 8B SFT at 32k context is tight but plausible with gradient checkpointing plus ZeRO offload; LoRA is the sane default.
- **RL is where it breaks.** Their recipe is batch 64 x 4 rollouts x up to 32k response tokens x 11 turns, and needs a vLLM rollout server **plus** PaddleOCR-VL + GroundingDINO + SAM 3.1 co-resident (roughly 10–15 GB of tool models). On a single card you must time-share policy and rollout engine. Expect to cut: 8B -> 2B/4B backbone, 32k -> 4–8k response budget, 11 -> 4–6 turns, batch 64 -> 8–16, and to make tools cheap or precomputed. Under those cuts a *map-only* RL run is feasible in the tens of GPU-hours (**~EUR 50–300 per run — my estimate, [UNVERIFIED]**), but it reproduces the *method*, not their numbers.
- **Recommendation: do not attempt to reproduce PERIA.** Reproduce its *recipe shape* — SFT on rejection-sampled teacher trajectories, then GRPO-family RL with a composite verifiable reward — on **our own generated task**, where we control turn count, image size and tool cost. Cite PERIA as the method precedent and use its public MapEval OOD number as a fair external comparison point.
- **What "map reasoning" means in PERIA**: reading *rendered map images* — transit/subway map QA (ReasonMap, ReasonMap-Plus), pixel-level route tracing on synthetic map images (MapTrace), plus OSM-derived MapQA and Google-derived MapEval as OOD. Crucially it is **map-as-picture, never map-as-world**: no geocoding, no routing engine, no POI database, no ground-truth geometry to verify against beyond the image annotation.

---

## 5. What is directly reusable, and the delta to our design

### Directly reusable (high confidence)

1. **MapQA's construction method** — SQL/Overpass templates over OSM plus LLM paraphrase, yielding 3,154 verifiable QA over 9 reasoning types and 175 entity types. This is the published precedent for our Amsterdam generator. Mirroring its nine reasoning types gives us a comparable axis for free.
2. **Spatial-Agent's operator library (section B1)** as the *specification* for our local tool API. Reimplement ~15 of the ~35 against an offline snapshot: `geocode`, `reverse_geocode`, `place_search`, `place_details`, `directions`, `distance_matrix`, `nearest`, `within_radius`, `haversine`, `bearing`, `bearing_to_direction`, `open_at_time`, `filter_places`, `compare_routes`, `tsp_tw`. Every one is exactly computable offline from OSM plus a routing engine (OSRM / Valhalla / pgRouting) — which means **every one yields a verifiable reward**.
3. **Spatial-Agent's structural-validity preference signal** (well-formed plan/graph, checked programmatically) as a *free auxiliary reward channel* in our GRPO mix, alongside answer correctness and format. Their graph-construction error rate of 0% shows how strong a constrained plan space is.
4. **PERIA's whole training recipe**: explore-and-exploit trajectory synthesis with increasing turn budget and rejection sampling against ground truth; thought-rewriting by a second model to decorrelate from the teacher; composite reward (repetition + format + correctness); and **NDTW-style continuous reward for path/route answers** with the `d_low`/`d_high` clipping trick — we will need exactly this for "trace the route" tasks.
5. **OR-GIGPO's observation-relaxed grouping** (semantic match at delta = 0.9 instead of exact state match) — a small, well-motivated, implementable delta over GRPO that we could adopt *and* ablate. Their Average Step Advantage Ratio (9.3% -> 62.3%) is a good diagnostic to reuse regardless.
6. **verl-tool (MIT)** as the RL harness, vLLM for rollouts. Its decoupled tool-server design matches what we need (our "world snapshot" becomes a tool server).
7. **MapEval (3 splits) + MapQA + ReasonMap** as *external, third-party* eval sets — letting us report "our 3–8B model vs Claude-3.5-Sonnet / GPT-4o / MapAgent-scaffolded Qwen-72B on a benchmark we did not build." **This is the single most valuable portfolio item in this survey**: it converts a synthetic-world result into a claim on an independent benchmark.
8. **MapAgent's repo as a prompted baseline** — swap its engine layer to an OpenAI-compatible vLLM endpoint and run the *same backbone we train*, untrained, inside their scaffold. That yields the honest "scaffold vs post-training" comparison that interviewers will ask about. (Licence warning: no LICENCE file -> run it standalone, do not copy code into our repo.)

### The delta (what nobody has done — our contribution)

| Axis | MapAgent | Spatial-Agent | PERIA | **Ours** |
|---|---|---|---|---|
| Model sees a **rendered map image** | only in Visual split | no | **yes** | **yes** |
| Tools query a **world model** (POI / routing / geometry) | yes (Google) | yes (Google) | **no** | **yes (local OSM snapshot)** |
| **Offline, reproducible, redistributable** backend | no | partly | n/a | **yes** |
| **Policy learning** | no | SFT + DPO | **SFT + RL** | **SFT + GRPO** |
| **Verifiable rewards from ground-truth geometry** | no | no | partial (NDTW vs annotation) | **yes, by construction** |
| Small model (3–8B) | no (72B) | 14B in ablation only | **yes** | **yes** |

Concretely, four things we do that none of them do together:

1. **Vision + tools + world state in one loop.** PERIA has vision and tools but no world; MapAgent and Spatial-Agent have world and tools but (mostly) no vision. The interesting failure mode — *the map picture and the tool output disagree* — only exists in our setup, and is a genuinely novel thing to study and report.
2. **Ground truth is generated, not annotated.** Spatial-Agent's #1 failure cause (data quality, 45.6%) and #2 (search mismatch, 33.8%) both vanish when the world is a local snapshot we own. We can generate unbounded verified training data; they cannot.
3. **Reward comes from the simulator, not from a label.** Route optimality, distance, "is it open at time t", "is A north of B" — all exactly checkable, all cheap, all densifiable. Textbook RLVR, for free, from OSM geometry.
4. **An honest OOD story.** PERIA's OOD gain is small (+4.4 overall, +1.5 on MapEval). Held-out cities (train Amsterdam -> test Rotterdam/Utrecht -> test MapEval/MapQA) give us a clean generalisation axis we can report either way, which is exactly the kind of honest-numbers result the portfolio ground rules call for.

### Open questions this survey raises for OPEN_DECISIONS

- **Renderer**: do we render maps ourselves (OSM tiles / custom cartography) or use pre-rendered tiles? Rendering ourselves means we control the visual distribution and can run the "map picture vs tool output conflict" study — but adds a renderer to build.
- **Resolution regime**: ReasonMap images average 5839 x 5449 px. Any claim about "reads a rendered map" must state the resolution regime; a 3–8B VLM at 1024 px will fail those outright. Decide our target resolution early and be explicit about it in the README.
- **Algorithm**: adopt OR-GIGPO or stay with vanilla GRPO? GRPO is simpler to defend and PERIA's own delta over GRPO is only +2.4. *Suggestion*: GRPO for Stage 1, OR-GIGPO as a documented stretch ablation.
- **Baseline budget**: running MapAgent plus a direct-prompt Claude/GPT baseline on MapEval costs roughly EUR 10–40 total. Cheap. Worth doing early to anchor the story before any training happens.
- **Licence hygiene**: MapEval-Visual is Google Maps screenshots. We evaluate on it, we do not imitate it. Our own data must be OSM-derived (ODbL) so it can be released.

---

## Source list

- **MapAgent** — [arXiv abs](https://arxiv.org/abs/2509.05933) - [arXiv HTML](https://arxiv.org/html/2509.05933v1) - [ACL Anthology](https://aclanthology.org/2026.findings-eacl.67/) - [OpenReview](https://openreview.net/forum?id=T4H85lY19w) - [GitHub Hasebul/MapAgent](https://github.com/Hasebul/MapAgent)
- **Spatial-Agent** — [arXiv abs](https://arxiv.org/abs/2601.16965) - [arXiv HTML](https://arxiv.org/html/2601.16965) - [ACL Anthology 2026.acl-long.679](https://aclanthology.org/2026.acl-long.679/)
- **PERIA** — [arXiv abs](https://arxiv.org/abs/2606.12830) - [arXiv HTML](https://arxiv.org/html/2606.12830v1) - [arXiv PDF](https://arxiv.org/pdf/2606.12830)
- **MapEval** — [arXiv 2501.00316](https://arxiv.org/abs/2501.00316) - [GitHub org](https://github.com/MapEval) - [HF Textual](https://huggingface.co/datasets/MapEval/MapEval-Textual) - [HF API](https://huggingface.co/datasets/MapEval/MapEval-API) - [HF Visual](https://huggingface.co/datasets/MapEval/MapEval-Visual) - [MapQaTor arXiv 2412.21015](https://arxiv.org/abs/2412.21015)
- **MapQA** — [arXiv 2503.07871](https://arxiv.org/abs/2503.07871) - [ACM SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3764174)
- **MapTrace** — [HF google/MapTrace](https://huggingface.co/datasets/google/MapTrace) - [GitHub google-research/MapTrace](https://github.com/google-research/MapTrace)
- **ReasonMap / RewardMap** — [HF FSCCS collection](https://huggingface.co/collections/FSCCS/reasonmap) - [GitHub fscdc/RewardMap](https://github.com/fscdc/RewardMap) - [arXiv 2510.02240](https://arxiv.org/html/2510.02240)
- **verl-tool** — [GitHub TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)
- **Google Maps Platform pricing 2026** — [Woosmap breakdown](https://www.woosmap.com/blog/google-maps-api-pricing-breakdown) - [MapAtlas per-SKU](https://mapatlas.eu/blog/google-maps-api-pricing-2026)
