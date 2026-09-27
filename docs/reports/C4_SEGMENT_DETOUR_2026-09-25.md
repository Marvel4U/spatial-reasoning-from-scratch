# Report: two markers and a path — the segment task (c4a) and the detour task (c4b)

Written 25 Sep 2026 by Claude from Marvin's runs of 23–25 Sep. Numbers are as recorded in the run ledger and the diagnostics files. The document stands on its own; file pointers are collected in chapter 8. It ends with the overnight batch that Marvin will review and run.

## 0. Glossary

- **Crop, cell, marker, condition token, block, head, relative position bias, disc marker, escape**: as in the m1 report (`docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md`, chapter 0). In short: a 256 m square of Amsterdam at 1 m per pixel as 15 one-hot class planes plus a marker plane; the output is a 64 × 64 grid of 4 m cells; the model is a small vision transformer with 2 blocks; tasks are selected by condition tokens; the marker is a 3-px disc; a learned bias on attention scores that depends on the row/column offset between patches is on in every block.
- **Two markers.** Both markers are drawn into the *same* marker plane. The two tasks of this report are symmetric in start and end, so the model never needs to know which is which.
- **Segment (c4a).** Target: the straight line between the two markers, one cell wide. Every other input is irrelevant.
- **Detour (c4b).** Target: the Euclidean shortest path from marker to marker that does not cross a building, one cell wide. Buildings are surface class 3 of the crop; everything else, including inner courtyards, counts as free ground. Where nothing is in the way the detour *is* the segment.
- **Detour ratio.** Length of the path divided by the straight distance. 1.00 means the segment is clear; the validation set is stratified in thirds at 1.00–1.05, 1.05–1.30 and above 1.30.
- **Plain IoU.** Intersection over union of the predicted and target cells. For a one-cell-wide line a prediction shifted by one cell scores near zero, so plain IoU understates a good line.
- **Tolerant metrics.** IoU after dilating both prediction and target by one cell; precision within one cell (share of predicted cells at most one cell from the target); recall within one cell (share of target cells at most one cell from a prediction).
- **Connectivity.** Whether the predicted cells form one connected component that touches both markers (8-connected from 25 Sep on; the runs in this report used 4-connectivity, which breaks diagonal lines).
- **Store.** The fixed set of training and validation samples: a window position in the district, two markers, the path, its pixels, lengths and detour ratio. 20,000 training, 500 validation, 500 test samples.

## 1. Why these two tasks, and why not a street graph

The plan for c4 was a route on the OpenStreetMap walking graph. After the m1 experience Marvin chose a ladder that needs no external data: first the segment, which is the bare two-marker task, then the shortest path around buildings, which is computed from the building plane the model already sees, and only later a rung with a legal-walking mask or a street graph. Each rung is the same algorithm with a different obstacle definition, and each target is exact.

The segment tests one thing: can the model find both markers and draw a line between them. The detour tests the thing c4 is about: an output where every cell depends on the whole crop, on where both markers are, which buildings stand between them, and which way round is shorter.

## 2. What was built

### 2.1 The target generator

`spatial_data/c4_routes.py` draws a window that respects the held-out block split, samples two endpoints 40–250 m apart on free ground at least 2 px from any building, and computes the Euclidean shortest path around the building polygons on a visibility graph: nodes are the corners of the polygonised building raster (simplified by 1 px, pushed 1.2 px outward so the path clears the wall), two nodes see each other if the segment between them crosses no building pixel (tested on the raster every 0.5 px), and a lazy A\* finds the shortest chain. The path is rasterised at 1 m and stored as a pixel list. Sampling is stratified on the detour ratio; pairs in enclosed courtyards are rejected by a connected-component check; the rejection counts are written next to the store.

Why a visibility graph and not a grid search: on a pixel grid every monotone staircase between two points has the same length, so the "shortest path" would be one of thousands and the target arbitrary. The visibility path is the taut string: straight until it meets a building, along the wall corner to corner, straight to the target once visible.

Two things went wrong on the way and are recorded because they shaped the store:

- A first version kept only "convex" corners as graph nodes. The filter had a sign error on some outlines, corners went missing, and the search returned paths 1.4–2.4 times too long on 58 % of samples: paths swung out to distant corners instead of hugging the near wall. Found by comparing every stored path with a brute-force 16-connected grid search (`perf_prototypes/c4_optcheck.py`). With all corners as nodes every one of 50 test paths is within 2.2 % of the grid length. Gallery of those 50: `crops/v4/c4/c4_gallery_test_50_fixed_builder.png`.
- Courtyards. Several paths run through the inner courtyards of housing blocks, because a courtyard is not a building. Marvin decided to keep them passable in c4b: the task is "around buildings", and "paved ground only" (roadway and sidewalk, 99.6 % one connected network) becomes the next rung.

### 2.2 The loader

The first loader cut windows with a Python loop and two `.item()` calls per sample, and built targets per sample on the CPU. At batch 128 that was about 220 ms per step in the loader against about 90 ms in the model. On 25 Sep the window cut became one gather, the segment rasterisation a batched tensor computation with the same sampling rule as the reference, the path scatter one indexed write, and the mixed-task selection one `torch.where`. Outputs are bit-identical to the numpy reference; the loader takes 39 ms at batch 128. Every run in this report used the old loader; wall times in the tables are therefore about 2.5 times what the same run costs now.

### 2.3 Evaluation

Per validation sample: plain IoU, connectivity, the ratio of the predicted component's length to the true length, and for the detour the IoU of the prediction against the segment (the shortcut). All stratified by detour ratio. From 25 Sep on also the tolerant metrics and 8-connectivity (`harness/eval_c4.py`).

## 3. The runs

All runs: 2 blocks, batch 128, disc marker, bias in all blocks, 4,000 steps, learning rate 1e-3 with cosine decay to 4k unless "constant". The store of 20,000 samples. Validation on the 500 stored samples of the task named in the column.

| run | training mix | val task | plain IoU | connectivity | wall time |
|---|---|---|---:|---:|---:|
| `c4_segment_L2_4k_b128_disc3_relV2_cosine4k_seed4` | segment alone | segment | **0.791** | 0.32 | 11 min |
| `c4_mix20m_segment_…cosine4k` / `…lrflat` | ½ disc 20 m (two markers) + ½ segment | segment | 0.477 / 0.474 | 0.00 / 0.40 | 30 min |
| `c4_mix20m_segdet_…cosine4k_seed4` / `seed5` | ⅓ disc 20 m + ⅓ segment + ⅓ detour | detour | 0.338 / 0.304 | 0.12 / 0.11 | 22 / 19 min |
| `c4_mix20m_segdet_…lrflat_seed4` / `seed5` | same, constant LR | detour | 0.245 / 0.260 | 0.04 / 0.06 | 19 min |
| `c4_segdet_L2_4k_b128_disc3_relV2_lrflat_seed4` | ½ segment + ½ detour, constant LR | detour | 0.318 | 0.09 | 9 min |

Tolerant metrics on the final checkpoints (`perf_prototypes/c4_tolerant_metrics.py`, all 500 validation samples):

| run | plain IoU | dilated IoU | precision within 1 cell | recall within 1 cell |
|---|---:|---:|---:|---:|
| segment alone | 0.791 | **0.927** | **1.000** | **1.000** |
| ⅓ mix, cosine, seed 4 (detour) | 0.338 | 0.538 | 0.789 | 0.699 |
| ½ segment + ½ detour, constant LR (detour) | 0.318 | 0.526 | 0.764 | 0.741 |

## 4. What the segment run shows

The segment is solved at 4k steps. Every predicted cell lies within one cell of the true line and every line cell within one cell of a prediction. The plain IoU of 0.79 is what a one-cell-wide target costs when the prediction sits one cell to the side on part of its length; the strips (`runs/diagnostics/c4_segment_…/pred_step003999.png`) show thin, clean lines exactly where they belong, for every orientation and length. Connectivity of 0.32 is the metric's fault: it used 4-connectivity, which reads a diagonal line as a chain of disconnected cells. Fixed.

Learning was gradual: IoU 0.09 at 250 steps, 0.24 at 1.25k, 0.41 at 1.75k, 0.62 at 2.75k, 0.79 at 4k, still rising. No plateau and no sudden escape, unlike the one-marker tasks. The two identical markers in one plane were no obstacle: the model resolves both and draws the line between them.

## 5. What the detour runs show

The detour is under way, not failed, and the strips say it more clearly than the numbers (`runs/diagnostics/c4_mix20m_segdet_…cosine4k_seed4/pred_step003999.png`). The model draws the right shape: the L around the block, the bend at the building corner, the straight line where nothing is in the way. The lines are blurred and wobbly, and 79 % of predicted cells and 70 % of target cells are within one cell of each other. In the last strip two routes around one house are nearly equally long, and the model lights both; the target holds only one, so that sample scores as an error. The store does not yet record the second-best route length, so such ties cannot be stratified out; that is queued for the builder.

Every detour curve was still rising at 4k with no sign of a plateau: 0.10 at 500, 0.19 at 2k, 0.30 at 3k, 0.34 at 4k for the best run. The runs were three to four times too short for the question they were meant to answer. There is no shortcut basin here: the segment is only the answer where nothing is in the way, so the model has to read the building plane from the start, and it learns gradually rather than by escape.

Three further readings from the diagnostics of the best detour run:

- **Both blocks are active.** The attention entropies recover to 2.4 and 3.0 after the early sink and stay there; in m1 block 0 stayed sunk on the condition tokens. The 2-block model is using all it has, which is the honest argument for trying 4 blocks.
- **The training loss runs away from the validation loss**: 0.27 against 0.62 at the end, widening from step 1.5k. The detour task saw about 170,000 presentations of a 20,000-sample store, eight passes over the same routes. Memorisation has started; a bigger store is cheaper than any model change.
- **The gradient is clipped for the second half of the run.** The pre-clip norm sits above the clip threshold of 1.0 from 1.5k on. Late training is throttled; a larger threshold is worth one run.
- **The patch projection barely weighs the building channel**: its weights are as small as those of every other surface class, about 0.35, while the marker channel sits at 2.0. In m1 the dense food & drink task is what forced the projection to represent the class the conjunction needed. No dense building task was in any c4 mix.

## 6. What did not help, and what the numbers are worth

- **The 20 m disc as a helper.** It was the helper that made m1 reliable, and it does nothing for c4: segment alone reaches 0.79 at 4k, the half mix 0.48 at half the segment presentations, the same per presentation; the ⅓ mix and the segment-plus-detour half mix land at the same detour score. For c4 the helper is the segment, and it is inside every mix already.
- **Constant learning rate** lost to cosine at 4k on the detour, 0.25–0.26 against 0.30–0.34; a line target benefits from the annealing at the end. With curves still rising, longer constant-rate runs may catch up; not decisive.
- **Single seeds.** Every row above is one run (two for the ⅓ mix). The m1 report showed a 3× spread in escape steps between identical runs; here learning is gradual and the spread is probably smaller, but nothing above should be read to better than a few hundredths.
- **The throughput jump** in one dashboard at step 900 was the previous run finishing on the same GPU, not a loader effect; two runs had overlapped for seven minutes.

## 7. The overnight batch

Purpose: turn "under way" into a number for the detour, and answer the four cheap questions before any architecture change. All runs on the vectorised loader, 2 blocks unless stated, batch 128, disc marker, bias in all blocks, cosine to the run length, evaluation every 1,000 steps, diagnostics on, validation on the 500 detour samples with the tolerant metrics. The queue is `c4_overnight_run_queue()` in `harness/experiments.py`; each spec's notes say what it tests. Estimated 9 minutes for the first run and about 35 minutes per 16k run, about 4.5 hours in total.

| id | change | question | expected |
|---|---|---|---|
| N0 `c4n_segdet_L2_4k_seed4_fastloader` | ½ segment + ½ detour, 4k | does the new loader reproduce the old run? | detour IoU ≈ 0.32, dilated ≈ 0.53 |
| N1 `c4n_segdet_L2_16k_seed4`, `…seed5` | same, 16k | the c4b reference and its seed spread | dilated IoU > 0.8, plain > 0.6 |
| N2 `c4n_bld_segdet_L2_16k_seed4`, `…seed5` | ⅓ building + ⅓ segment + ⅓ detour, 16k | does a dense building task lift the detour, as the dense task did for m1? | above N1 if the patch projection was the limit |
| N3 `c4n_segdet_L2_16k_seed4_clip3` | N1 with gradient clip 3.0 | was the clip throttling late training? | faster rise after 1.5k, same or better end |
| N4 `c4n_segdet_L2_16k_seed4_max120m` | N1 on routes with straight distance ≤ 120 m (188 val samples) | is route length the limit? | clearly above N1 if it is |
| N5 `c4n_segdet_L4_16k_seed4` | N1 with 4 blocks | does depth matter when both blocks are already busy? | the m1 lesson says try the cheap levers first; this reads against N1–N3 |
| N6 `c4n_segdet_L2_16k_seed4_store100k` | N1 on a 100,000-sample store | is the train/val gap memorisation? | val CE closer to train CE, higher final IoU |

N6 needs the larger store, which builds on the CPU in about 3.5 hours and does not touch the GPU; started first, it is ready long before the queue reaches N6. Commands, in this order:

```bash
cd ~/Github/spatial_reasoning_LLM_artifact && nohup .venv/bin/python -u -m spatial_data.c4_routes --n-train 100000 --n-val 500 --n-test 500 --overlays 9 --seed 2027 --out data/amsterdam/de_pijp/crops/v4/c4_100k > runs/logs/c4_routes_build_100k.log 2>&1 &
```

Then set `RUN = c4_overnight_run_queue()` in `harness/experiments.py` and start the harness as usual.

Not in the batch, deliberately: the paved-ground rung (c4c), the second-best-route ratio in the builder, and the iterative-search direction (`docs/memo/ITERATIVE_SEARCH_2026-09-25.md`), which is a separate project.

### 7.1 Results of the batch (25 Sep, plus one run Marvin added: 4 blocks on the 100k store)

All on the detour validation set, final checkpoint at 16k unless stated. "dilated" = IoU after one-cell dilation of both; "prec / rec" = within one cell; strata = dilated IoU for detour ratio 1.00–1.05 / 1.05–1.30 / above 1.30; CE = train / val cross-entropy at the end.

| run | plain IoU | dilated | prec / rec | strata (dilated) | CE train / val |
|---|---:|---:|---:|---|---:|
| N0 sanity, 4k | 0.350 | 0.549 | 0.79 / 0.69 | 0.82 / 0.49 / 0.33 | 0.29 / 0.61 |
| N1 reference, seed 4 | 0.395 | 0.602 | 0.84 / 0.74 | 0.85 / 0.55 / 0.40 | 0.10 / 0.60 |
| N1 reference, seed 5 | 0.382 | 0.590 | 0.83 / 0.73 | 0.84 / 0.53 / 0.40 | 0.10 / 0.62 |
| N2 building helper, seed 4 | 0.340 | 0.532 | 0.77 / 0.66 | 0.83 / 0.46 / 0.31 | 0.09 / 0.68 |
| N2 building helper, seed 5 | 0.344 | 0.541 | 0.78 / 0.68 | 0.82 / 0.47 / 0.33 | 0.10 / 0.65 |
| N3 clip 3.0 | 0.380 | 0.591 | 0.82 / 0.73 | 0.84 / 0.53 / 0.40 | 0.09 / 0.64 |
| N4 routes ≤ 120 m (188 val) | 0.419 | 0.653 | 0.85 / 0.80 | 0.83 / 0.51 / 0.33 | 0.01 / 0.62 |
| N5 4 blocks | 0.511 | 0.703 | 0.87 / 0.84 | 0.91 / 0.67 / 0.53 | 0.04 / 0.49 |
| N6 100k store | 0.497 | 0.689 | 0.88 / 0.85 | 0.88 / 0.64 / 0.55 | 0.18 / 0.44 |
| **N7 4 blocks + 100k store** | **0.606** | **0.771** | **0.91 / 0.90** | **0.92 / 0.74 / 0.65** | **0.12 / 0.34** |

N0 reproduced the old run (0.35 vs 0.32–0.34), so the loader rewrite is clean. Seeds 4 and 5 of the reference differ by 0.013, so differences below about 0.02 are noise.

What the batch settles:

1. **Data was the first limit.** The reference overfits (train CE 0.10 against val 0.60, flat from step 6k while train keeps falling). The 100k store alone lifts the detour from 0.60 to 0.69 dilated and halves the gap. Route length made no difference to it (N4 overfits just the same: train CE 0.01).
2. **Depth is the second limit, and the two add up.** 4 blocks alone: 0.70. Both: 0.77, with the curve still rising at 16k (+0.01 per 1k steps at the end) and the val loss still falling. The one-shot ceiling has not been reached.
3. **The building helper hurts** (−0.06), the opposite of the m1 lesson: the detour needs the *shape* of buildings along one line, not a dense building mask, and a third task at ⅓ share simply takes presentations from the detour. **The clip was not throttling** (N3 = N1 within noise).
4. **The hard stratum is the whole story now.** In the best run the straight-ish third scores 0.92, the bending third 0.74, the big-detour third 0.65. Complexity, not blur, is what remains: the strips of N7 show correct shapes with small wobble for single bends (routes 489, 57, 74) and the errors concentrate on routes with several corners. This is the measured version of the one-shot limitation that the iterative design (`docs/memo/ITERATIVE_SEARCH_2026-09-25.md`) is meant to remove.

Recipe for c4b one-shot, as of now: 4 blocks, batch 128, disc marker, bias in all blocks, ½ segment + ½ detour, 100k store, 16k+ steps, cosine, clip 1.0. Numbers to beat for the iterative model: dilated IoU 0.77 overall, 0.65 on detour ratio above 1.3.

### 7.2 The mix probe (25 Sep evening): which task is the odd one out

All at 4 blocks, batch 128, 100k store, 16k steps, seed 4, validation on the detour (dilated IoU; seed noise about 0.013).

| run | segment | building | detour | detour dilated | strata (dilated) | helpers on val |
|---|---|---|---|---:|---|---|
| N7 | ½ | – | ½ | 0.771 | 0.92 / 0.74 / 0.65 | segment 0.965 |
| N8 | – | – | 1 | 0.799 | 0.92 / 0.77 / 0.71 | – |
| **N9** | – | ½ | ½ | **0.813** | 0.92 / 0.78 / 0.73 | building 1.000 |
| N10 | ⅓ | ⅓ | ⅓ | 0.787 | 0.91 / 0.76 / 0.69 | segment 0.972, building 0.999 |

Reading:

1. **The segment is the odd one out.** Removing it gains 0.03 (N7 → N8) and adding it to the building mix costs 0.03 (N9 → N10), both beyond the seed noise. My prediction that it would be neutral was wrong. The likely reason: the segment and the detour share the *output form* (a one-cell line between the same two markers) and the *same input*, but disagree on the rule: the segment goes through buildings, the detour never does. Two tasks that look alike but contradict each other compete for the same circuit. The building mask has a different output form and contradicts nothing.
2. **The building helper is neutral to slightly positive** when the segment is absent (+0.014, borderline), and the earlier −0.06 at 2 blocks on the 20k store (§7.1) was measured in the presence of the segment and in an overfitting regime. Helper effects are small and regime-dependent; the interference rule is the robust finding.
3. **Where the gain lands.** The easy stratum is saturated at 0.92 in all four runs; every difference is in the bending and multi-corner strata (0.65 → 0.73 on the hardest third). The curves of N8 and N9 were still rising at 16k.

**Final one-shot recipe for c4b:** 4 blocks, batch 128, disc marker, bias in all blocks, ½ building + ½ detour (or detour alone), 100k store, 16k+ steps, cosine, clip 1.0. **Numbers to beat for the iterative model: dilated IoU 0.81 overall, 0.73 on detour ratio above 1.3, plain IoU 0.66, connectivity 0.65.**

Diagnostics of N9 (`runs/diagnostics/c4n_bld_det_L4_16k_store100k/dashboard.png`): no clipping after step 2k (median pre-clip norm 0.36); the patch projection puts its largest position-specific weight on the marker (1.4) and the second on the building class (0.9, up from 0.35 in the 2-block runs), i.e. the model now reads buildings inside the patch; the four blocks have separated roles: block 1 holds the sharp marker heads (four heads with entropy 0.7–1.4 and ~0.45 of their attention on each marker), blocks 2 and 3 attend broadly (entropy 4–4.8) and carry the propagation along the building layout, block 0 recovers from an early sink to broad local mixing by 12k.

## 8. Files

- Builder and probes: `spatial_data/c4_routes.py`; `docs/plans/perf_prototypes/c4_optcheck.py` (optimality against a grid search), `c4_gallery.py` (render a store), `c4_tolerant_metrics.py` (tolerant metrics on a checkpoint).
- Store: `data/amsterdam/de_pijp/crops/v4/c4/` (`c4_routes_{train,val,test}.npz`, stats JSON, overlays, the 50-sample gallery); the 100k store goes to `…/c4_100k/`.
- Loader and tasks: `spatial_data/dataset_c4_gpu.py` (vectorised 25 Sep; `building` target kind), `spatial_data/c4_tasks.py` (`segment`, `detour`, `building`), `harness/eval_c4.py` (tolerant metrics, 8-connectivity), `harness/config.py` (`c4_routes_subdir`, `c4_max_straight_m`, `grad_clip`).
- Runs: ids in chapter 3; diagnostics under `runs/diagnostics/<id>/`; the overnight queue in `harness/experiments.py`.
- Plan: `docs/plans/C4_PLAN.md` §9 (preparation and decisions) and §10 (first results and the dashboard addendum).
