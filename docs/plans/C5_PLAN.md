# C5_PLAN — iterative path finding: a scratchpad, output tokens, and a ladder up to a search

Status 1 Oct 2026: **rung 1 redefined as the wave search (§7), agreed with Marvin; data and subtasks next.** Rays and key trajectories are closed as the model's output (§6, §7); the new map (Zuid + Centrum, 512 m windows) and the metre-based eval are in place; the length tokens work; the resolution test A/B is done (§6).

Memo of the 30 Sep – 1 Oct discussion, with the failed suggestions and the figures: `docs/memo/WAVE_SEARCH_2026-10-01.md`.

Written 29 Sep 2026 from Marvin's design (conversation of 28–29 Sep), replacing the wavefront design of `docs/memo/ITERATIVE_SEARCH_2026-09-25.md` §5 and §7 as the working plan; the memo stays as background (precedents, risks).

## 1. Goal

A model that finds a *reasonably short* uninterrupted walking path between two markers the way a person reads a map: look from each end towards the corners of the buildings in the way, extend from the most promising corner, connect when the two ends see each other, pull the result tight. Not the provably shortest path through any labyrinth. Obstacles are buildings, as in the one-shot detour task. Car-like routing from crossing to crossing is out of scope.

Success for a path: it connects both markers, crosses no building pixel, and its length is within 10 % of the optimum. The one-shot detour model is the baseline (C4 report §7.2: dilated IoU 0.81; 0.73 on routes with several corners).

## 2. Decided (29 Sep 2026, Marvin)

1. **Scratchpad.** One new input plane, channel 17, holding drawn lines: the model's own previous output in the iterative rung, a given path in the single-step rungs. Zero for every earlier task, so all earlier tasks run unchanged.
2. **Output tokens.** Besides the mask the model emits tokens: a length estimate in 10 m bins for rungs 2–3, a phase and a stop token for the search. Input tokens keep selecting the task, as before.
3. **The search looks at building corners.** Rays go from a point to the building corners it can see, not in fixed directions: a shortest path around buildings only bends at corners, and corners are what a person looks at on a map.
4. **Cast rays is a rung of its own** (rung 1), the elementary operation of every search pass.
5. **"Best positioned" = the A\* rule** for the first version of the search: the endpoint with the smallest distance travelled so far plus the straight distance to the goal. Deterministic, so each pass has one correct answer; a relaxed "any sensible endpoint" version comes later.
6. **Scoring** of any path output: connects both markers, collision-free, length ratio to the optimum (success ≤ 1.10); cell-wise tolerant IoU reported alongside.
7. **Order.** Scratchpad plane first, then the token output, then rungs 1–4 as single-step tasks; the iterative mechanics only after rungs 1–4 pass.
8. **Both search directions are tried** (open question 3 below is about order, not about whether).

## 3. Open question (resolved 1 Oct 2026)

**Unidirectional, and the direction is in the data:** the start is a point on the initial scratchpad, the marker plane carries the goal only. No second marker plane, no start/goal distinction to learn. Bidirectional search is not needed for the wave formulation (§7) and is dropped for now. The original question, kept for the record:

**Bidirectional or unidirectional search first.** Bidirectional (fronts from both markers, stop when a ray from one front sees a point of the other) needs no start/goal distinction and matches the symmetric marker plane we have; it is the proposal for the first run. Unidirectional (from start to goal, heuristic = straight distance to the goal) needs the two markers to be distinguishable: either a second marker plane (channel 18, "goal") or the goal marked in the scratchpad at pass 0. Both variants are planned; the question is only which runs first. Proposal: bidirectional first, unidirectional as the second run of rung 5, with the goal as a second marker plane.

## 4. The ladder

| rung | task (task token) | input beyond the map and markers | output | what it teaches |
|---|---|---|---|---|
| 0 | infrastructure: empty scratchpad, earlier tasks | scratchpad all zero | as before | the new channel and the token head do not disturb what works |
| 1 | **cast rays** (`rays`) | one or both markers | mask: straight segments from each marker to every building corner it can see, within 80 m | line of sight, the operation every search pass repeats |
| 2 | **length of a straight line** (`length`) | scratchpad: one straight segment | token: length in 10 m bins, 0–360 m (37 classes) | measuring distance on the map |
| 3 | **length of a polyline** (`length`, same token) | scratchpad: a path of 2–6 straight pieces | token: total length, same bins | summing along a path, the "distance so far" of the search |
| 4 | **tighten a path** (`tighten`) | scratchpad: a loose collision-free path that goes around the buildings the same way as the optimum | mask: the taut path | string-pulling, the last step of the search |
| 5 | **find a path** (`search`), iterative | scratchpad: own output of the previous pass (empty at pass 0) | per pass: mask of the current rays and connection; tokens: phase (`explore`, `connect`, `tighten`, `done`) | the search itself |

The straight segment between two markers (the c4a task, solved) stays in every mix as the bare two-marker task.

### 4.1 Targets and data per rung (all derived from the existing c4 store)

- **Corners.** The corners of the simplified building outlines that the c4 builder already uses as graph nodes, pushed 1.2 px outwards; visibility tested on the raster every 0.5 px, as in `spatial_data/c4_routes.py`.
- **Rung 1.** For each store sample and each marker, the visible corners within 80 m, stored as segment endpoints; the mask is the rasterised set of segments. The 80 m radius keeps the fan readable; it is a task parameter, not a cut in the data.
- **Rung 2.** Scratchpad segment = the straight line between the two markers of a store sample (lengths 40–250 m, already stratified) or between a marker and a random visible point for short lengths; target = its length.
- **Rung 3.** Scratchpad path = the stored detour polyline (2–6 pieces in the store) or a random prefix of it; target = its length. No new data.
- **Rung 4.** Loose path = the stored optimal path with 1–3 extra waypoints inserted, each a random free point near the path, kept only if every piece is collision-free and the loose path goes around the same buildings (checked by comparing which side of each building the two paths pass, or equivalently by requiring the tightened loose path to equal the stored path within one cell). Target = the tightened path, computed by the builder's search restricted to a corridor around the loose path.
- **Rung 5.** Teacher trajectories from an instrumented bidirectional A\* over corners: at each expansion the builder records the rays drawn so far, the endpoint chosen next, and the rays that become obsolete. Pass t's input scratchpad = the rays after t−1 expansions; target = the rays after t expansions; at connection, the connecting segment; then the tightened path; then `done`. Typical routes need 2–6 expansions, so a trajectory is 4–10 passes.

### 4.2 Model changes (Marvin)

| # | change | needed from rung |
|---|---|---|
| M1 | input 16 → 17 channels (scratchpad); old checkpoints load with the new channel's weights zero-initialised | 0 |
| M2 | readout slot in the prefix + token head (linear over the token vocabulary), trained with CE next to the mask loss; the mask head unchanged | 2 |
| M3 | task vocabulary: `rays`, `length`, `tighten`, `search`; phase tokens as outputs and, for rung 5, as the next pass's input | 1 / 5 |
| M4 | (unidirectional variant only) channel 18 = goal marker | 5b |

### 4.3 Loader and eval (Claude)

| # | item |
|---|---|
| D1 | scratchpad rendering on the GPU from stored polylines (the segment rasteriser of the c4 loader, generalised to polylines) |
| D2 | rung 1 targets: visible-corner fans per marker, computed once and stored next to the c4 store |
| D3 | rung 2–3 token targets from stored lengths; bins as a task constant |
| D4 | rung 4 loose-path generator and corridor-restricted tightening, stored |
| D5 | rung 5 trajectory recorder in the builder; loader samples a pass index, teacher state in, next state out |
| D6 | eval: rays by tolerant IoU and "rays that cross a building"; lengths by bin accuracy and mean absolute error in metres; paths by connection, collisions, length ratio, stratified by detour ratio; for rung 5 also passes used and stop accuracy |

## 5. Experiments (Marvin triggers)

| run | setup | pass |
|---|---|---|
| R0 | 17-channel model, empty scratchpad, the c4 ½ building + ½ detour recipe | detour within 0.02 of the 16-channel run (0.81 dilated) |
| R1 | rays, mixed with the segment task | rays: tolerant IoU ≥ 0.85, < 5 % of predicted ray cells on buildings |
| R2 | length of a straight line, token head | bin accuracy ≥ 0.8 (±1 bin ≥ 0.98), mean error ≤ 10 m |
| R3 | length of a polyline | as R2 |
| R4 | tighten | length ratio of output to taut ≤ 1.05 on ≥ 90 % of samples, no collisions |
| R5a | search, bidirectional, teacher forcing, routes ≤ 120 m | per-pass ray accuracy as R1; phase-token accuracy ≥ 0.95 |
| R5b | search with own rollouts, all lengths | success (connects, collision-free, ratio ≤ 1.10) above the one-shot model on the several-corner stratum |
| R5c | unidirectional variant | compared with R5b on the same strata |
| R5d | extrapolation: train ≤ 120 m, test 120–250 m with more passes allowed | success does not collapse on the longer routes |

## 6. First runs (29 Sep 2026) and what changed

Runs: 2 blocks, 8k steps for R1–R4 (each mixed with the segment task), 4 blocks, 16k for R0; 64×64 output, 256 m window.

| run | result | reading |
|---|---|---|
| R0 | detour dilated IoU 0.796 (N9: 0.813) | passes (within 0.02): the 17th channel costs nothing |
| R1 rays | rays dilated IoU 0.52, 1 % of ray cells on buildings | fails, and the task is unreadable at this scale: ~16 rays per marker merge into a blob at 4 m cells |
| R2, R3 length | token: 78–81 % exact bin, 99–100 % within one bin, mean error ~2 m (R3 checkpoint, re-evaluated); **segment IoU fell to 0.10** | the token works; the mask target (a copy of the scratchpad) taught "copy what is drawn" and broke the segment task, which then drew only the two marker dots |
| R4 tighten | tighten IoU 0.67, dilated 0.85 | hard to judge: 37 % of loose paths are < 5 % longer than the tight one, 24 % of routes are one straight piece |

Decided (Marvin, 29 Sep):

1. **Length tasks train the token only.** The mask target is gone: every cell of a length sample carries the ignore value (−100), which cross-entropy skips. (Loss note: with per-task weights the ignored cells still count in the mean's denominator; `balanced_ce` and `ce_dice` would need the ignore value handled.)
2. **Rays go only to grazing corners**: corners where the ray, continued 1.5–4.5 px past the corner, stays on free ground. Only these can be a turning point of a shortest path. Val: 15.8 → 6.3 rays per marker (p90 28 → 12). Old fans kept in `c4_100k/c5_fans_allcorners_20260928/`.
3. **Longer, more angled routes are needed for rungs 3 and 4**, which points to a larger window (variant B below).
4. **Resolution test before more rung work**, R0 recipe, patch size 16 throughout:

| variant | window | input | output | cell | tokens |
|---|---|---|---|---|---|
| now | 256 m | 256 px, 1 m | 64×64 | 4 m | 256 |
| A | 256 m | 256 px, 1 m | 128×128 | 2 m | 256 |
| B | 512 m | 512 px, 1 m | 128×128 | 4 m | 1024 |

   A tests whether finer cells make rays and small offsets learnable; B tests the larger window for longer routes (and costs roughly 4–8× per step). A 512 m window at 2 m cells (256×256 output) is the upper end; it is decided after A and B, by extrapolating how much more training each needs.
5. **Eval in metres.** Dilated IoU and precision/recall use a 4 m tolerance at every grid size (one cell at 64, two at 128; the `*_1cell` metric names stay). Task statistics are cached per output grid (`c4_task_stats_g128.json`). Prediction images show the scratchpad and the markers on the input panel, and the length token as text.

## 7. The wave search (decided 1 Oct 2026, Marvin) — replaces rung 1 and the search rung

Full reasoning, dead ends and figures: `docs/memo/WAVE_SEARCH_2026-10-01.md`.

**Why the change.** The rays (R1) and the key trajectories (R1c/R1d, 100 m; the 40 m four-round version is built but not run) fail for the same reasons: the target contains arbitrary choices (which of two equally far points, which side street), the window edge leaked into the targets (fixed by the margin rule, `c5_trajectories.py`), and the model memorises the training ground rather than the rule (train loss falls, val loss flat; R0 drops from 0.80 to 0.64 on the new map). A better loss does not fix an arbitrary target; the target must have no choices.

**The rule (Marvin):** once the network runs, nothing is processed outside it. No route is computed and handed over, no tally of past outputs, no rescaling. The output of pass *i* is the scratchpad and the state token of pass *i+1*, verbatim. The task is one of several the network is meant to use later in an integrated, flexible way; the point is to train the algorithm, not to solve the problem.

**The task package ("extend the wave"):**

| | |
|---|---|
| input layers | map planes; marker plane = the goal only; the start is a point on the initial scratchpad |
| scratchpad in | the previous output: walking distance from the start as far as the wave has got (empty at pass 1 except the start point) |
| token in | SEARCH, or BUILD_PATH once the wave covers the goal |
| output | **the whole next scratchpad**: what was there (copied, or compressed in option 3) plus the next band, one horizon (40 m) further; per-cell classes (10 bins of the pad value + not reached) |
| token out | SEARCH; BUILD_PATH on the pass whose output first covers the goal; DONE after the path pass |
| path pass | BUILD_PATH in → the path from the start to the goal (mask) + DONE; then the tighten rung |

Two value maps of the same wave, to be decided by runs (both derive from one true wave per sample):

- **Option 2, pulses:** every band carries values 0–1 on its own (sawtooth pad); old ground is a pure copy; the path pass hops between bands at the seams.
- **Option 3, one gradient:** the front is always at 1 and the network compresses the older ground each pass so the pad runs from 0 at the start to 1 at the front; the path pass is plain descent. (Not "a little bit of gradient" at the start: the network does not know how far it will have to go.)

**Data:** one clearance-priced wave per sample (the `_Field` of `c5_trajectories.py`), computed on the district with a margin of one horizon and clipped to the window; bands = floor(distance / 40 m); a sample yields one training pair per pass and the loader draws a random pass; the path = the tree's parent links from the goal. Single marker; the goal is the route's second marker. Val: every pass of every route.

**Eval:** per-cell class accuracy and within-one-bin on the new band and on the copied ground separately; token accuracy; for the path pass the tolerant metrics of c4; rollouts on the network's own outputs: share of routes where the token switches on the right pass and the path connects.

**Built 1 Oct (data and loader side):** `spatial_data/c5_wave.py` (one wave per route, 2 m distance field, the way back); tasks `wave_first` / `wave_step` / `wave_reach` / `wave_path` in the loader with fronts as 1-cell rings (crossing rule), the start as a 3x3 disc on the pad, both markers in the marker plane, ONE task id for all wave passes with the state token in condition slot 0 (SEARCH 34 / BUILD_PATH 35) and token classes SEARCH 37 / BUILD_PATH 38 / DONE 39 (40 token classes); eval adds recall on copied cells, recall on new cells, false foreground; specs `c5_wave_queue()`, `c5_wave_mix()`, `c5_wave_night_1oct_queue()` (eight runs, night 1-2 Oct). Tests: `tests/test_c5_wave.py`.

**Caveat found 1 Oct:** adding task ids enlarges the condition-token table; a checkpoint from before is then loaded *without* its trained task embeddings (the loader skips the mismatched key silently). Any evaluation or resume of an older checkpoint must build the model with the checkpoint's table size (`/tmp/c5smokebin/eval_ground.py` does).

**Results 1–2 Oct (details: memo §7):** each step type alone reaches 0.64–0.87 new-cell recall with no false cells (L2 8k); plain mixes collapse to copying for lack of steps per task; with β = 5 on the changed cells the three forward tasks reach single-task quality in one network at 12k (0.84 / 0.70 / 0.62 at the corrected threshold); the rollout shows the fed-back halo derailing pass 3. Built: natural per-pass sampling (`wave_fwd` / `wave_bwd`, `c5_wave_mix_natural()`), pad noise for training (`c5_wave_pad_noise`, off by default).

**Result 2 Oct (details: `docs/memo/WAVE_MIX_TOKEN_LOSS_2026-10-02.md`):** the whole search runs on the network's own scratchpad and tokens. `c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k` (4 blocks, 48k, wave_fwd / wave_bwd / wave_draw 4 : 4 : 1) solves 75 % of 200 De Pijp val routes free-running (92 % at 40–120 m, 77 % at 120–200 m, 54 % beyond; solved = connected, no building cell, at most 10 % longer than the teacher's path); the same recipe at 2 blocks solves 4 %. One route takes 57 ms on the RTX 3060 (10 passes, 2.4 ms per pass). The cause of the failed backward mixes was the token loss: in a mix the token is a real decision, and at weight 1.0 it dominated the loss and blurred the grid; weight 0.1 fixes it (false cells on the hop 180 -> 98 per sample at L2 8k) and learns the token better. Not the cause: ties in the teacher's way back (about 7 false cells per sample), dropping Dice, pad noise, flat lr, the 100k store. Removing the lit-cell weight breaks the rollout (the network under-paints). Recipe: token loss weight 0.1, cosine lr, lit-cell weight on, pad noise, natural sampling, 4 blocks; spec `c5_wave_night_queue_2oct()`.

**Later teachers on the same interface, not changes to it:** goal-aware fronts (extend only what can still beat the best known route); the network deciding when to build the path.

## 8. Next steps

Done 2 Oct: the search rung works end to end on De Pijp val (75 % solved, L4 48k); tools for it under
`harness/diagnostics/` (`wave_rollout_eval.py`, `wave_back_ties.py`, `wave_rollout_gif.py`, `wave_rollout_timing.py`).

1. The new map (Zuid + Centrum): build the wave data for its 256 m val routes (`spatial_data/c5_wave.py`), run
   `wave_rollout_eval.py` on the L4 checkpoint. This is the generalisation number of the rung.
2. The failures (25 %; 46 % on routes of 200 m and more): which pass breaks, hop or erasing; the backward part is
   the weaker half (IoU within 4 m 0.79 against 0.90–0.99 forward, under the teacher's schedule).
3. One longer or deeper run (L4 96k or L6 48k) to tell steps from depth on the long routes.
4. The L4 48k recipe with token loss weight 1.0 as the control that measures what the token fix is worth in the
   full run (not run yet).
5. One forward per pass for grid and token in the rollout; then training on the network's own outputs if the
   failures are drift.
6. Tighten after DRAW; goal-aware fronts and the stop decision as later teachers.
7. The C5 report and the public repo, once 1 is in.

Later, not in this round: zooming with a scratchpad kept in map coordinates and a view token carrying scale and offset, so that a path drawn while zoomed in is visible when zoomed out.
