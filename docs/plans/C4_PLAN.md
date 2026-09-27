# C4_PLAN — path finding: two markers, a mode, the shortest legal route; then routes under a cost layer **Report:** `docs/reports/C4_SEGMENT_DETOUR_2026-09-25.md` (c4a/c4b results, overnight batch §7).

Written 22 Sep 2026 from Marvin's request. Status: **preparation started 24 Sep 2026 (§9); m1 solved, its recipe carries over.** Prerequisites on disk: cropset **v4** (RGB + `noise7` + **`sun`** planes in npz). c3 single-marker disk task (`within_20m`) is **not** c4; c4 uses **two marker planes** + OSM graph routes. Order: c2 (done) → c3 (relative + `within_20m`, see `C3_PLAN.md`) → c4 (this).

## 1. Goal

Given the layer stack, **two marked points** and a **mode** (pedestrian first; cycling and car later), mark on the 64 × 64 grid the cells of the **shortest route that is legal for that mode**. The route is computed on a real street graph, so the target is exact and the rule is not ours: "legal" comes from the map's own access rules, and "shortest" from the graph.

**Routing is not ours to write.** OpenStreetMap tags every street with who may use it (`foot`, `bicycle`, `oneway`, `oneway:bicycle`, `access`, road type). `osmnx` builds one graph per mode from those tags (`network_type = walk | bike | drive`: walking ignores one-way rules and drops motorways; cycling keeps cycleways and respects one-way rules except where Amsterdam's usual bicycle exception applies; driving keeps only motor roads and respects one-way and access restrictions). The shortest route is a standard graph search (`networkx`, already on the rig). We write only plumbing: snapshot the graph, sample endpoints, run the search, rasterise the route. The three modes give visibly different routes in Amsterdam: a pedestrian cuts through the park and walks against one-way streets, a cyclist follows the canal-side cycle path, a car goes round the one-way maze.

What c4 adds: the output is a **connected structure** between two distant points, not a per-cell property. A cell on the route can only be known by reasoning about the whole crop (where the endpoints are, which streets connect them, which are allowed). This is the first task where every cell's answer depends on cells far away, and where attention and depth have to carry the work. c3 (crop-relative conditions) is the stepping stone: it needs crop-wide statistics; c4 needs crop-wide structure.

## 2. Decided (carried over from c1/c2)

| | Decision |
|---|---|
| Model | same grid ViT, one-hot input, P = 16 to start; the head stays a per-cell mask |
| Task input | condition tokens; the mode is one more token in the prefix (`mode = pedestrian`), later `bicycle`, `car` |
| Markers | two extra input planes, start and end (the c0 marker channel, twice); the plan of §4 keeps them as planes, not tokens |
| Target | route cells on the 64 × 64 grid (a cell is positive if the route polyline passes through it); IoU plus route-specific metrics (§5) |
| Data | cropset `v3b` layers + a **new routable layer**; targets computed on the GPU-side loader from precomputed routes, not on the fly with a graph library |
| Evaluation | per mode, never pooled; held-out crops as before; a "straight-line" and a "shortest ignoring legality" baseline (§5) |

## 3. Open question: which street graph defines "legal"

Three candidates, from `survey/raw/08_behavioural_data_and_masked_layers.md` §A4:

| Source | What it gives | Licence | Fit |
|---|---|---|---|
| **OpenStreetMap** via `osmnx` (`network_type = walk / bike / drive` already encodes access rules) | full multimodal graph with `foot`, `bicycle`, `oneway`, `oneway:bicycle`, `access` | ODbL | standard, well tooled, offline; sidewalk mapping in Amsterdam is partial, so walking often runs along the street centreline |
| **Amsterdam "Loop- en fietsnetwerk"** (municipal walking + cycling graph, GeoPackage, quarterly) | an official pedestrian/cycle graph linked to addresses | CC BY 4.0 | strong for pedestrian and bicycle; no car mode; untested |
| **NWB** national road file (public domain) | all public roads, routable, cycle network since 2022 | public domain | good cross-check for car; no sidewalks |

Recommendation: **OSM as the primary graph** (one source for all three modes, offline, the per-mode legality recipe in the survey), the municipal network as a comparison for pedestrian routes on a few crops. Marvin decides. Either way the graph is snapshotted once and becomes a layer with a manifest entry like every other.

Two definitional points to settle with the graph choice:
- **Snapping.** Markers are placed on the graph's nearest legal node; the marker plane shows the snapped position, so the target is always reachable from what the model sees.
- **Ties and near-ties.** Two routes of almost equal length are common on a grid-like street plan. Store the length ratio of the second-best route per sample and stratify the evaluation by it, instead of pretending the answer is unique.

## 4. Background: what the data allows

- The BGT surface layer already distinguishes sidewalk (2,425 polygons), cycle path (585), roadway, transit lane in De Pijp. The model can therefore *see* the mode-relevant surfaces; what it cannot see are access rules (one-way, no-entry) that live only in the graph. **Prediction:** pedestrian routes are learnable from the surfaces; car routes will need the one-way information as an input layer (a "car direction" plane) or the task is unfair.
- A 256 m crop holds routes of at most ~360 m; at 4 m cells a typical route is 40–90 cells, i.e. 1–2 % of the grid: rarer than any c1 class. Balancing as in c2 (dampened weight + decision rule) applies unchanged.
- Route rendering: the graph edges are polylines; rasterise the chosen route at 1 m, then "any pixel" per 4 m cell (a route is thin, like the c0 marker, so the majority rule would erase it).
- Routes leaving the crop: endpoints are sampled so that the shortest legal route stays inside the crop (reject otherwise); the share rejected is logged.

## 4b. Extension: routes under a cost layer ("the quiet path", "the sunny path")

OSM knows nothing about noise or sun, but the graph search does not care where an edge's cost comes from. Give every edge a cost of its own and the same search returns the cheapest route under that cost:

- **Quiet path:** sample the noise raster along each edge (mean over its pixels); cost = length × (1 + β · noise_level_above_quiet), one global β (proposal 0.5 per band). The search then trades a detour against quieter streets; β sets the exchange rate and is part of the task definition.
- **Sunny / shaded path:** same with a sunlit-hours raster (§4c); cost rises with shade for "sunny", with sun for "shaded".
- Later: "along shops", "avoiding busy streets" (footfall), each a cost layer.

Task description: the mode token plus one **cost token** (`plain`, `quiet`, `sunny`, `shaded`); the c2 condition vocabulary grows by these entries and nothing else changes. Target: exact, because the cost function is written down and the search is deterministic; the evaluation adds "cost of the predicted route / cost of the true route" to the length ratio of §5. Two routes with nearly equal cost are ties, handled as in §3.

**Prediction:** the cost variants are harder than the plain route by exactly the amount of context they need: the model must read a second layer along the whole candidate route, not only follow the streets. This is the c4 counterpart of c3's crop-wide statistics.

## 4c. Prerequisite for the sunny path and the terrace question: the sunshine layer

**Decided 23 Sep 2026 (Marvin), build started the same day.** Questions and answers that led to it, kept for traceability:

- *Is it just a formula over the building heights?* Half: the sun position for any date / time / latitude is a closed formula. The other half is a geometric test per pixel (is there a building between the pixel and the sun?), done as a ray march along the sun azimuth over a 1 m height raster. Cost: a few hundred million comparisons per sun position, well under a second vectorised on the GPU; a sunlit-hours plane (~50 positions) takes minutes at most.
- *Compute on the spot with a flexible date/time?* Both: **(a) the shadow function stays a callable tool** `shadow(height_raster, date, time)` for the viewer and for later time-specific questions (and the Qwen track's tool API); **(b) summary planes are precomputed** into cropset `v4` (sunlit hours 21 June, 21 March), because they are deterministic and the targets must be reproducible. Shadows are never computed inside the training loop.
- *Which heights?* 3DBAG roof height per building only in v1; trees exist only in the AHN surface model as one-season noisy crowns, so the AHN-based shadow is computed as a **check** (disagreement rate reported), not mixed in.
- *Classes:* first built as 4 ground classes on 21 June (2 / 4 / 6 h), which put 74 % of June ground into one class. **Revised the same day (Marvin):** one plane on a fixed scale 2 … 12 h (7 ground classes + building interior), and the **date varies per crop** (21 June or 21 March, recorded per crop) so the model learns that days differ in sun overall. Owner: Claude; memo `docs/memo/SUNSHINE_LAYER_2026-09-23.md` §5b.

Not built yet; the first *derived* layer of the project (parked in the pilot in favour of raw layers). Inputs exist: 3DBAG building heights (`buildings_3dbag.gpkg`, roof height above ground) and the AHN5 surface model (`dsm_05m.tif`). Sun position is a formula (date, time, latitude); shadow casting is a ray march over a height raster at 1 m: for each sun position, a pixel is in shadow if the line towards the sun hits a building before leaving the district.

Proposed definition (one plane, fits the four-level scheme): **sunlit hours per day at ground level on 21 June**, computed from building shadows at 15-minute steps, classed as `< 2 h`, `2–4 h`, `4–6 h`, `> 6 h`. Variants for later: 21 March / 21 September (the terrace season), and a "sun at 15:00" boolean plane for time-specific questions. Buildings only in the first version: trees are in the AHN surface model but as noisy crowns of one season; a tree variant comes second and is compared, not mixed.

Built-in check: where the computed building shadow disagrees with the AHN surface model's own heights, one of the two sources is wrong; report the disagreement rate per crop rather than hiding it.

Deliverable: `worldsnap/layers/sunshine.py` (builder in the registry, manifest entry with the exact date/step/definition, overlay PNG), a cropset `v4` that adds the sun plane (and the 7-level noise plane c3 needs), and a one-page memo. Owner: Claude, Marvin reviews the definition before it is built.

## 5. Evaluation

Per mode: IoU under the decision rule; **connectivity** (is the predicted mask one connected component joining the two markers?); **length ratio** (length of the predicted path, if connected, over the true shortest length); **legality** (share of predicted cells on surfaces allowed for the mode); **detour stratification** by the second-best-route ratio of §3. Baselines: straight line between the markers (what a model that ignores the streets would draw), shortest route *ignoring* legality (what a model that ignores the mode would draw). The difference between "shortest ignoring legality" and the true target is the part of the task that is about the mode.

## 6. Work list

| # | Item | Owner | Status |
|---|---|---|---|
| A1 | Routable graph layer: snapshot OSM for the district (+ buffer), per-mode subgraphs, manifest, overlay PNG of the pedestrian graph over the BGT surfaces | C | open |
| A2 | Route generator: sample start/end pairs on the graph inside each crop, shortest legal route per mode, rasterise to 1 m, store per crop (`routes.npz`: endpoints, mode, route pixels, length, second-best ratio) | C | open |
| A3 | Loader: two marker planes + mode token + route target on the GPU; on-the-fly crops need routes precomputed on the district graph, so A2 works at district level and the loader clips | C | open |
| B1 | Eval per §5 incl. connectivity and length ratio; the two baselines | C | open |
| B2 | Viewer adapter: markers, true route, predicted mask, error map over the surfaces | C | open |
| D1 | Model: marker planes are already supported (in_chans +2); mode token = one more condition id; nothing else | M | open |
| D2 | Depth comparison (2 vs 4 blocks) becomes mandatory here; expectation that 2 blocks fail on longer routes | M | open |
| D3 | Experiment specs | C | open |
| E1 | Sunshine layer (§4c): builder, manifest, overlay, cropset `v4` | C | **approved 23 Sep, in progress** |
| E2 | Cost-layer routes (§4b): edge costs from the noise / sun rasters, cost token, cost-ratio metric | C | open, after L8a |

## 7. Experiments

| Run | Setup | Question | Pass (proposal) |
|---|---|---|---|
| L8a | pedestrian only, short routes (≤ 100 m), 2 blocks | can the model draw a connected route at all? | connectivity ≥ 0.9, IoU ≥ 0.7 on held-out crops |
| L8b | pedestrian, all lengths, 2 vs 4 blocks | does depth matter for long routes? | IoU by route length; 4 blocks clearly better on long routes, or not |
| L8c | + cycling and car, mode token | does the mode change the route where it should? | legality ≥ 0.95 per mode; car routes respect one-way where the target does |
| L8d | cumulative with c1–c3 tasks | regression | earlier tasks within 0.01 |
| L8e | quiet path (pedestrian, β = 0.5) vs plain path, same endpoints | does the model read the noise layer along the route? | cost ratio ≤ 1.1 on held-out crops; quiet routes differ from plain routes where the target does |
| L8f | sunny / shaded path on cropset `v4` | same question for the sun layer | as L8e |

## 8. Decisions for Marvin

1. The graph source (§3): OSM primary, municipal network as comparison.
2. Whether c4 starts pedestrian-only (proposed) or with all three modes.
3. The sunshine-layer definition of §4c (date, step, classes, buildings only) before it is built.
4. Whether markers stay input planes (proposed) or become coordinate tokens (the option rejected for c0; c4 would be the place to revisit it, since a route needs the endpoints only as anchors).

## 9. Preparation plan (24 Sep 2026, revised the same day: no OSM for the first two rungs)

**Decided 24 Sep (Marvin):** after the m1 experience, c4 does not start from the OSM graph. It starts from targets that are computed from the planes the model already sees, in two rungs, and only if both work does the OSM walking graph (§1–§6 above) come back as a third rung.

| rung | target | what the model must do | obstacle definition |
|---|---|---|---|
| **c4a segment** | the straight line between the two markers | find both markers, draw the line; every other input ignored | none |
| **c4b detour** | the shortest path between the two markers that goes around buildings | as c4a, but read the building plane along the way and bend the path | surface class 3 (building, 33 % of the district) |
| c4c legal (later) | shortest path on a legal walking network | as c4b with a rule the model cannot see | OSM walk graph, or BGT: sidewalk + none, roadway forbidden |

The ladder is one algorithm with three obstacle masks. c4b has an exact target with no external data, no licence, no manifest, and no snapping question. The BGT variant of c4c (walk only on sidewalk and "none" surfaces, i.e. treat roadway as an obstacle too) is the same code again with a different mask and is probably the better third rung than OSM.

### 9.1 What m1 taught, applied here

1. **Marker recipe is settled**: 3-px disc marker, relative position bias in all blocks, 4 blocks, batch 32 (report `docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md`, chapter 7).
2. **Train the hard task inside a mix with the bare marker task.** For m1 that was the disc (4/4 escapes in the mix vs ~½ alone). Here c4a *is* the bare two-marker task, and the mix for c4b is {`within_20m` at K = 2, segment, detour}.
3. **Read which solution the model has at the first eval.** For c4b the shortcut is the segment itself (on a crop with few buildings between the markers the segment is nearly the answer). The probe compares the prediction with the segment and with the true path, and every eval is **stratified by the detour ratio** (path length / straight distance): the samples that matter are the ones where the path bends.
4. **Escape timing needs seeds**; explicit `lr_horizon_steps` in every spec.
5. **Two markers fit the current input**: the marker plane carries K markers per sample already; the segment and the detour are symmetric in start and end, so one plane with K = 2 needs no model change.

### 9.2 The target algorithm for c4b (and what "shortest" means on a raster)

Two ways to compute a shortest path around raster obstacles; the choice matters because of ties.

- **Grid search (A\* on the 1 m pixel grid, 8-connected).** Simple, scipy/numpy only. Its flaw: on a grid, every monotone staircase between two points has the same length, so the shortest path is massively non-unique and the target becomes an arbitrary staircase. Not acceptable as a target.
- **Euclidean shortest path around polygons (visibility graph).** Nodes = the two markers + the corners of the building polygons in the window; an edge between two nodes if the straight segment between them crosses no building; Dijkstra on that graph. The result is the taut string around the buildings: unique in general, straight where it can be, bending only at building corners. This is the target. Polygons come from the building plane by polygonising the 1 m raster within the window (or from `buildings_3dbag.gpkg` directly, same thing since the plane was rasterised from it). Cost per route: tens of ms with a spatial index; 50k routes in well under an hour on the CPU.

Definitions to fix with it: markers are sampled on non-building pixels; endpoint distance 40–220 m; the path must stay inside the window (the window edge is an obstacle); samples whose path is blocked (enclosed courtyard) are rejected and the share logged; the route is rasterised at 1 m along the polyline and becomes cells by "any pixel" (a 1-cell-wide line, like the c0 marker, like m1's thin targets). Stored per sample: window origin, both markers, path polyline, path length, straight distance, detour ratio. Sampling is stratified on the detour ratio (say thirds at 1.0–1.05, 1.05–1.3, > 1.3) so the eval can be read by difficulty.

c4a uses the same store with the polyline replaced by the segment (same endpoints), so the two tasks share samples and the mix is trivial.

### 9.3 Work list (Claude; P1–P3 need no model change and can run now)

| # | Item | Status |
|---|---|---|
| P1 | Sample store: `spatial_data/c4_routes.py` → `crops/v4/c4/c4_routes_{train,val,test}.npz` + stats JSON + overlay PNGs. Lazy A* on convex building corners (simplified outline, pushed 1.2 px), line of sight sampled on the raster at 0.5 px, reachability pre-check by connected components, ellipse 2.2, 600-expansion cap; endpoints ≥ 2 px from buildings, 40–250 m apart; strata ⅓ each at detour 1.0–1.05 / 1.05–1.3 / > 1.3. Smoke 24 Sep: 170 ms/sample, 0 path-hits-building, overlays checked. Rejections logged (blocked courtyards, searches beyond the ellipse/cap). | **built; full store: Marvin runs** |
| P2 | Loader `C4GpuTrainLoader`: draw a sample → cut the 15 planes at its window → paint both markers (disc) → scatter the polyline pixels → cells; task ids for `segment` / `detour`; val = fixed held-out windows from the block split | open |
| P3 | Eval: IoU under the rule; connectivity (one component joining both markers); length ratio; **stratified by detour ratio**; the candidate-mask probe (segment, detour); pred strips with markers, target, prediction over the surface plane | open |
| P4 | Specs: L8a′ = mix {`within_20m` K=2, segment}, 4 blocks, 8k, two seeds (does the two-marker circuit form as fast as the one-marker one?); L8b′ = mix {`within_20m` K=2, segment, detour}, 16k, two seeds, read by detour-ratio stratum; L8c′ = depth 4 vs 8 on L8b′ | open |
| P5 | c4c: BGT-legal variant (roadway as obstacle) = P1 with a second mask; OSM only if c4c-BGT is not enough | later |

Model side (Marvin): nothing for c4a/c4b beyond two new task ids; depth (and the iterative-decoder option of the earlier §9.2, now moved here as D4) after L8c′.

### 9.4 Decisions (Marvin, 24 Sep)

1. Endpoints 40–250 m apart; the maximum distance is a run parameter (runs from max 40 m up to max 250 m); strata ⅓ each at detour 1.0–1.05 / 1.05–1.3 / > 1.3.
2. Target = the 1 m path rasterised, then a 4 m cell is positive if any of its pixels is on the path: a one-cell-wide line.
3. c4a (segment) runs alone first.
4. **Courtyards stay passable in c4b.** The obstacle is the building class only; inner courtyards count as free ground. Paths through courtyards are part of the definition, not an error. "Paved ground only" (roadway + sidewalk, 99.6 % one connected network, courtyards / canals / parks excluded) is the c4c mask.
5. The Euclidean optimum is the target, never a grid metric. Verified:  compares stored paths with a 16-connected grid Dijkstra; after the 24 Sep builder fix (all outline corners as nodes; a convex-only filter had dropped needed corners) 50/50 test samples are within 2.2 % of the grid length. Gallery: .

## 10. First c4 results (Marvin's runs, 25 Sep 2026; read by Claude)

All runs: 2 blocks, batch 128, disc marker, bias in all blocks, 4k steps, 20k-sample store. Val = the 500 stored val samples.

| run | training | val task | plain IoU | IoU after 1-cell dilation | precision / recall within 1 cell | connectivity |
|---|---|---|---:|---:|---:|---:|
| `c4_segment_…cosine4k_seed4` | segment alone | segment | **0.791** | **0.927** | 1.00 / 1.00 | 0.32 |
| `c4_mix20m_segment_…` (cosine / flat) | ½ within_20m K=2 + ½ segment | segment | 0.477 / 0.474 | | | 0.00 / 0.40 |
| `c4_mix20m_segdet_…cosine4k` seeds 4 / 5 | ⅓ within_20m + ⅓ segment + ⅓ detour | detour | 0.338 / 0.304 | 0.538 | 0.79 / 0.70 | 0.12 / 0.11 |
| `c4_mix20m_segdet_…lrflat` seeds 4 / 5 | same, constant LR | detour | 0.245 / 0.260 | | | 0.04 / 0.06 |
| `c4_segdet_…lrflat_seed4` | ½ segment + ½ detour | detour | 0.318 | 0.526 | 0.76 / 0.74 | 0.09 |

Reading (`perf_prototypes/c4_tolerant_metrics.py`; pred strips `runs/diagnostics/<run>/pred_step003999.png`):

1. **c4a is solved.** Every predicted segment cell lies within one cell of the target and every target cell within one cell of a prediction (precision and recall 1.00 at one-cell tolerance); the plain IoU of 0.79 is the price of a one-cell-wide target that a one-cell offset halves. The strips show clean thin lines in the right place. Connectivity 0.32 counts one-cell gaps in a 4-connected sense and is the wrong instrument for a diagonal line at cell resolution.
2. **c4b is under way, not failed.** At 4k steps the detour models draw the right *shape* (the L around the block, the bend at the corner: routes 23, 57, 74 in the strip), blurred and wobbly; 79 % of predicted cells and 70 % of target cells are within one cell of each other. Every curve was still rising at 4k with no plateau; the runs are 3–4× too short. The learning is gradual, unlike the m1 escape: no shortcut basin here, the segment is only the answer where nothing is in the way.
3. **The 20 m disc is not a useful helper for c4.** Segment alone at 4k (0.79) vs the ½ mix (0.48): per segment presentation the same, so the disc neither helps nor hurts, and the ⅓ mix and the segment+detour ½ mix land at the same detour score. The bare task that matters for c4b is the segment, which is already inside every mix.
4. **Cosine beats constant LR at 4k on c4b** (0.30–0.34 vs 0.25–0.26): the annealing sharpens a line target; with curves still rising the flat runs may catch up on a longer horizon. Not decisive.
5. **Metrics for lines**: plain IoU stays the ledger number, but the eval should also report the 1-cell-dilated IoU and precision/recall within 1 cell (E1 below); connectivity should use 8-connectivity on the dilated prediction.

Next (Marvin decides):

- **E1 (Claude)**: add the tolerant metrics and 8-connected connectivity to `harness/eval_c4.py`, stratified by detour ratio.
- **R1**: ½ segment + ½ detour, cosine, **16k**, seeds 4 and 5; the c4b reference. Prediction: dilated IoU > 0.8, plain IoU > 0.6.
- **R2**: R1 with the maximum distance capped at 120 m (short routes) to read whether length is the limit.
- Only if R1 stalls: depth 4 vs 2, and the two-plane marker (start and end distinguishable), which the segment result says is not needed for two symmetric markers.

Addendum 25 Sep, from the dashboard of `c4_mix20m_segdet_…cosine4k_seed4`: (a) no dense building task was in any c4 mix; the patch projection weighs the building channel like every other surface class (~0.35 vs 2.0 for the marker), so a dense `building` task is the first helper to add; (b) train CE 0.27 vs val 0.62 and widening from 1.5k: the 20k store was seen ~8 times by the detour task, so a 100k store (2 h CPU) comes before depth; (c) the pre-clip gradient norm sits above the clip of 1.0 from 1.5k on, a clip of 2–3 is worth one run; (d) both blocks are active (entropies 2.4 / 3.0), which is the case for trying 4 blocks after (a)–(c); (e) the throughput jump at step ~900 was the previous run finishing on the same GPU (17:21), not a loader effect. Near-equal routes are lit twice by the model; the builder should record the second-best ratio so ties are stratified. **The iterative-search direction is written up in `docs/memo/ITERATIVE_SEARCH_2026-09-25.md` and continues in a new conversation.**

**25 Sep batch result** (report §7.1): data (100k store) and depth (4 blocks) are the levers, additive: dilated IoU 0.60 → 0.77, still rising at 16k; building helper hurts, clip irrelevant, route length irrelevant to the overfit. Hard stratum (detour > 1.3) at 0.65 is the one-shot limitation; the iterative track takes over from here.

**25 Sep mix probe (report §7.2):** the segment interferes with the detour (same output form, contradicting rule), the building helper does not; best one-shot c4b = ½ building + ½ detour, 4 blocks, 100k store: dilated IoU 0.81, hardest stratum 0.73. c4 one-shot closed; iterative track next.
