# One table, one rule: wave search vs one-shot vs Qwen (8 Oct 2026)

Status: results. Closes the open item "wave not on the polyline scorer in one table" of
[`TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md`](TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md) §6 and plan §5.
All numbers: [`COMPARISON_TABLE_2026-10-08.json`](COMPARISON_TABLE_2026-10-08.json) (per method and set: n, solved,
bands, failure counts, source file; for the grid models also the raw-chain score and both cell-rule numbers).

## 1. The rule and the conversion

**Rule = the polyline rule** of `spatial_data/track_a_export.py score`, the scorer of the Qwen runs, unchanged:
start and end within 6 m of the markers; no segment crosses a building pixel (samples ≤ 0.5 m on the 1 m raster);
length ≤ 1.10 × the teacher's walking length (`teacher_m`, the walking length of the clearance-priced wave path). The
failure reason is the first failing check in the order unparseable → wrong endpoints → crosses a building → too long.

**Routes.** The Track A exports `data/track_a/{de_pijp,zuid_centrum,maze_v2_houses}_val_plain` (val split, 500 routes
each, 256 m window, 2 px/m). They were exported from the same route stores the grid evals read
(`de_pijp/crops/v4/c4_100k`, `zuid_centrum/crops/v4ns/c4_256_val`, `maze_v2_houses/crops/v4ns/c4_256_val`), and
the route index is the store's `c4_routes_val.npz` order. Checked per route for all 9 converted runs: same origin,
same start and goal (store metres × 2 = the record's pixels), same store; 0 misaligned, 0 records without a drawing.

**Converted from drawings** (marked ◆ below): the one-shot model and both wave-search networks. Their final cell
drawing (one-shot: one forward pass, p ≥ 0.5; wave: free-running from the start disc until DONE, the stepping of
`wave_rollout_eval.py --mode free`) becomes a polyline by `harness/diagnostics/drawing_to_polyline.py`:

1. the shortest 8-neighbour chain start → goal through the lit cells (`c5_cleanup.shortest_lit_chain`, the chain the
   cell rule scores; the 3 × 3 cells around each marker count as lit, as there);
2. its cell centres (2 m cells) → image pixels at 2 px/m; no snapping, no repair;
3. the export's own building-aware Douglas–Peucker at 2 m (the teacher polylines got the same), headline;
   the raw chain is scored too (in the JSON, `raw_chain`).

A drawing whose start and goal do not connect is `points: null` and counts as **unparseable**. Qwen's and the
baselines' polylines are scored as they are.

**n per cell:** 500 everywhere, except the two vanilla rows: 200 (the first 200 De Pijp val routes; those routes are
easier, so the column "first 200" gives every method on the same 200 routes).

## 2. Solved share, polyline rule

| method | De Pijp val | fronts 2–3 (n=112) | 4–5 (n=141) | 6+ (n=247) | first 200 | Zuid + Centrum val | house maze val |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight line (floor) | 27.6 % | 73.2 % | 35.5 % | 2.4 % | 50.5 % | 27.8 % | 21.2 % |
| teacher polyline (ceiling) | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| Qwen3-VL-2B vanilla, two-pass (n=200) | 2.5 % | 5.6 %¹ | 1.5 %¹ | 0.0 %¹ | 2.5 % | – | – |
| Qwen3-VL-4B vanilla, two-pass (n=200) | 24.5 % | 34.7 %¹ | 35.4 %¹ | 1.6 %¹ | 24.5 % | – | – |
| Qwen3-VL-2B + LoRA 1k | 21.4 % | 58.0 % | 27.7 % | 1.2 % | 37.5 % | – | – |
| Qwen3-VL-2B + LoRA 5k | 31.0 % | 74.1 % | 37.6 % | 7.7 % | 50.5 % | – | – |
| Qwen3-VL-2B + LoRA 19.3k | 49.6 % | 81.3 % | 61.7 % | 28.3 % | 64.5 % | – | – |
| ◆ one-shot 64k (1 pass) | 37.4 % | 83.0 % | 39.7 % | 15.4 % | 52.0 % | 31.4 % | 16.4 % |
| ◆ **wave search** (free-running) | **76.4 %** | **92.9 %** | **87.2 %** | **62.8 %** | **89.0 %** | **56.8 %** | **44.0 %** |
| ◆ wave search + cleanup | 46.2 % | 82.1 % | 53.9 % | 25.5 % | 64.5 % | 40.2 % | 32.2 % |

¹ bands of the first 200 routes (n = 72 / 65 / 63).
Zuid + Centrum bands are n = 132 / 163 / 205, house maze n = 205 / 141 / 154 (in the JSON).

Runs: one-shot `c5_os_bld_det_L4_64k_g128_store100k`; wave `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k`;
wave + cleanup `c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g128_store100k`. All grid models were trained
on De Pijp only, so Zuid + Centrum and the house maze are transfer; Qwen was evaluated on De Pijp only.

Sources: baselines `data/track_a/<set>/baselines/{straight,teacher}_report.json`; vanilla
`runs/qwen_vanilla/eval_ab_n200.json`, `eval_ab_4b_n200.json` (mode two-pass); LoRA
`runs/qwen_sft/lora{1000,5000,20000}_v1/eval_val_v1_n500.json`; grid models
`data/track_a/<set>/preds/<run>_{oneshot,wave}_report.json`. Every Qwen and baseline number was re-scored from its
prediction file with today's scorer and matches its source file exactly.

## 3. Why they fail (De Pijp val, share of all routes)

| method | n | unparseable | wrong endpoints | crosses a building | too long | solved |
|---|---:|---:|---:|---:|---:|---:|
| straight line | 500 | 0 | 0 | 72.4 % | 0 | 27.6 % |
| Qwen 2B vanilla, two-pass | 200 | 0 | **92.0 %** | 2.0 % | 3.5 % | 2.5 % |
| Qwen 4B vanilla, two-pass | 200 | 0 | 13.5 % | **51.5 %** | 10.5 % | 24.5 % |
| Qwen 2B + LoRA 1k | 500 | 0 | 39.0 % | 36.4 % | 3.2 % | 21.4 % |
| Qwen 2B + LoRA 5k | 500 | 0 | 19.6 % | **47.0 %** | 2.4 % | 31.0 % |
| Qwen 2B + LoRA 19.3k | 500 | 0 | 11.4 % | **36.4 %** | 2.6 % | 49.6 % |
| ◆ one-shot 64k | 500 | **53.4 %**² | 0 | 9.2 % | 0 | 37.4 % |
| ◆ wave search | 500 | **21.0 %**² | 0 | 2.0 % | 0.6 % | 76.4 % |
| ◆ wave search + cleanup | 500 | 19.2 %² | 0 | **34.0 %** | 0.6 % | 46.2 % |

² For the grid models "unparseable" means the drawing does not connect start and goal (no chain).

What the mix says:

1. **Two families, two failure modes.** Qwen always produces a parseable line with roughly the right ends after
   fine-tuning, and fails on **geometry**: its largest bucket stays "crosses a building" (36–47 %) at every data
   size. The grid models never miss an endpoint (the markers are an input plane and the chain starts at them) and
   rarely cross; they fail by **not connecting**. When the wave search connects, the line is almost always valid
   (382 of the 395 connected routes solve).
2. **The wave search's lead holds under the polyline rule**: 76.4 % vs 49.6 % for the best Qwen and 37.4 % for the
   one-shot model, and the gap is largest on long routes (6+ fronts: 62.8 % vs 28.3 % vs 15.4 %).
3. **The cleanup network does not survive the rule change** (§4): its lines hug buildings.

## 4. Cell rule vs polyline rule on the same drawings

The cell rule (`route_score` of `wave_rollout_eval.py`): connected through lit cells, no cell of the chain *entirely*
building, chain length ≤ 1.10 × `goal_m`. "Same drawings" = the cell rule written by the eval run that produced the
drawings (`runs/diagnostics/<run>/{wave_rollout_eval,oneshot_route_eval}{tag}_pl.json`); "existing" = the JSONs from
before (`wave_rollout_eval_val500.json`, `oneshot_route_eval.json`, `..._{district}_c4_256_val_val.json`). The two
agree to the third decimal in all 9 cases.

| model | set | cell rule (existing) | cell rule (same drawings) | polyline rule (DP 2 m) | polyline (raw chain) | not connected |
|---|---|---:|---:|---:|---:|---:|
| one-shot 64k | De Pijp | 46.0 % | 46.0 % | 37.4 % | 35.6 % | 267 |
| one-shot 64k | Zuid + Centrum | 34.6 % | 34.6 % | 31.4 % | 30.0 % | 325 |
| one-shot 64k | house maze | 17.8 % | 17.8 % | 16.4 % | 16.2 % | 411 |
| wave search | De Pijp | 77.6 % | 77.6 % | 76.4 % | 75.0 % | 105 |
| wave search | Zuid + Centrum | 59.6 % | 59.6 % | 56.8 % | 55.8 % | 183 |
| wave search | house maze | 50.8 % | 50.8 % | 44.0 % | 41.0 % | 214 |
| wave + cleanup | De Pijp | 70.2 % | 70.2 % | 46.2 % | 43.0 % | 96 |
| wave + cleanup | Zuid + Centrum | 52.4 % | 52.4 % | 40.2 % | 36.6 % | 183 |
| wave + cleanup | house maze | 42.2 % | 42.2 % | 32.2 % | 29.6 % | 221 |

Route by route (De Pijp): of the routes the cell rule solves, the polyline rule fails 7 / 388 for the wave search,
43 / 230 for the one-shot model and **120 / 351 for the cleanup network**, all of them as "crosses a building". A few
routes go the other way (the cell rule fails for length, the simplified polyline is short enough): 1 for the wave.

Reading:

- The two rules differ only on the building test (a line through cell centres vs "no cell entirely building") and
  slightly on the length (the DP polyline is shorter than the 8-neighbour chain). For the plain wave search the
  difference is small (−1.2 points on De Pijp, −2.8 Zuid, −6.8 maze, where the lanes are narrow).
- The one-shot model loses 8.6 points on De Pijp: it was trained on the geometric shortest path, which runs along
  walls, so its chain cells are often part building.
- The cleanup network loses 24 points. 1.5 % of its chain cells contain a building pixel, against 0.2 % for the
  plain wave (0.010 for the one-shot). A likely cause: the cleanup targets are the network's own shortest chain,
  accepted under the cell rule, which lets a chain cut through cells that are part building; the shortest chain
  hugs the inside of every corner, and training on those targets teaches lines that hug buildings. This is an
  inference from the numbers, not tested.

**Conversion artifact (sensitivity).** A 2 m cell centre sits on a pixel corner, and the scorer's floor() puts it
in the cell's lower-right pixel. Upper bound of that effect: move every raw chain point to a free pixel of its own
cell, if the cell has one, and re-score. Of the raw "crosses a building" failures this rescues, on De Pijp,
4 / 16 for the wave search, 26 / 55 for the one-shot model and 33 / 186 for the cleanup network (Zuid: 8 / 37, 11 / 25,
27 / 134; maze: 19 / 81, 4 / 8, 16 / 130). The DP simplification already recovers part of this (raw → DP: wave
75.0 → 76.4 %, one-shot 35.6 → 37.4 %). So the one-shot number could be up to ~3 points higher with a friendlier
conversion. The cleanup network's crossings are mostly real, and nothing changes the order of the table.

## 5. What can and cannot be claimed

- Claimable: on the same 500 held-out De Pijp routes, scored by the same polyline rule, a free-running 8.9 M-parameter
  wave search solves 76.4 %. Qwen3-VL-2B + LoRA on 19.3k routes solves 49.6 %, and the one-shot L4 model (one forward
  pass) solves 37.4 %. The search's lead is largest on routes of 6+ fronts.
- Qualifiers: the grid models are converted from drawings by a shortest-chain rule (§1). Their "unparseable"
  is "not connected". Qwen sees a rendered PNG and outputs coordinates in a 0–1000 grid, while the grid models get
  one-hot map planes and a marker plane. Different input modality and output interface, same routes, same scorer.
  Vanilla Qwen is on 200 routes. Transfer (Zuid + Centrum, maze) exists only for the grid models.
- The cleanup network is better than the plain search under the cell rule's cleanliness metrics, but worse under
  the polyline rule. Report it as a negative result of the rule mismatch, not as an improvement.

## 6. Pictures

`runs/diagnostics/c5_os_bld_det_L4_64k_g128_store100k/showcase_vs_search/` (redrawn 8 Oct with the matched one-shot
checkpoints, 256 m: `c5_os_bld_det_L4_64k_g128_store100k`, 512 m: `c5_os_bld_det_from64k_L4_32k_g256_b64_zc512`):
one PNG per showcase route + `contact_sheet.png`. The Qwen 2B + LoRA 19.3k predictions exist only for De Pijp val,
so the five-panel variant (map, teacher, one-shot, search, Qwen polyline with its polyline-rule verdict) exists only
for De Pijp route 333: `de_pijp_c4_100k_val_route333_polyline.png`.

## 7. Reproduce

```
cd harness
PYTHONPATH=. python diagnostics/drawing_to_polyline.py RUN --kind wave|oneshot --export ../data/track_a/de_pijp_val_plain [--rerun]
PYTHONPATH=. python diagnostics/drawing_to_polyline.py RUN --kind ... --export ../data/track_a/zuid_centrum_val_plain \
    --crops ../data/amsterdam/zuid_centrum/crops/v4ns --store c4_256_val
PYTHONPATH=. python diagnostics/drawing_to_polyline.py RUN --kind ... --export ../data/track_a/maze_v2_houses_val_plain \
    --crops ../data/synthetic/maze_v2_houses/crops/v4ns --store c4_256_val
```
(`--save-pads` was added to `wave_rollout_eval.py` and `oneshot_route_eval.py` for this; GPU, ~5–10 min per wave run.)
