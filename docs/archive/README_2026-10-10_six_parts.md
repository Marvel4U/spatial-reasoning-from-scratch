# Spatial reasoning from scratch on real city data

<img src="docs/figures/wave_search_512m_house_maze_route008_431m.gif" width="560" alt="A from-scratch 8.9 M-parameter transformer runs a path search on its own scratchpad through a maze it was never trained on">

*__Figure 1.__ A path search run by a network on its own scratchpad. A 4-block, 8.9 M-parameter transformer trained only on Amsterdam finds a 431 m walking route through a synthetic house maze it has never seen (512 m window), in 26 passes. Each frame is one pass: the orange cells are the network's whole output, fed back verbatim as its next input; first a wave of fronts spreads from the start, then the way back is drawn from the goal, then a last pass keeps the path. Leftover cells from the search stay on the drawing. Run `c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512` (Part II).*

This repository is a study of one question: **what spatial reasoning can a small vision transformer learn when the input is not a photo but a set of spatially aligned data layers, and what does it take?**

The input is a 256 m square of Amsterdam drawn as planes: buildings from the 3D building register, surface types from the national base map, the road-noise map, places from OpenStreetMap and Overture, hours of sunshine computed from building heights. Every plane shows a different property of the same ground at the same pixel, one metre per pixel. On top of these the task can place one or two **markers**, points the question refers to ("within 100 m of *here*"). The question itself arrives as two **condition tokens** in front of the image tokens ("food & drink", "within 100 m"). The model answers with a **mask** on a grid of 4 m (later 2 m) cells. Everything in between is a plain transformer of 2 to 4 blocks, 5 to 9 million parameters, trained from scratch on one district. Every task has an exact answer computed from the data, so every number below can be checked.

The tasks form a ladder from "mark the sidewalks" to "draw the shortest path around the buildings from here to there". The README is a series of short posts, each with its question, method, results (headline figure first) and conclusion: [I](#part-i-can-a-vision-in-vision-out-model-read-and-combine-layers) the one-shot ladder, [II](#part-ii-can-a-search-algorithm-run-as-an-iterative-visual-task) the path search on a scratchpad (Figure 1), [III](#part-iii-how-does-a-vision-language-model-do-qwen3-vl-zero-shot-and-fine-tuned) Qwen3-VL zero-shot and fine-tuned, [IV](#part-iv-how-do-they-compare) the comparison under one rule, [V](#part-v-what-we-learned) what we learned, [VI](#part-vi-how-it-is-built-and-how-to-run-it) how it is built and run, limitations, outlook, related work, authorship.

What this is **not**: no reinforcement learning, one city. The vision-language model appears only as a fine-tuned baseline on the same task, not as the system under study. The one-shot path-finding numbers use a one-cell tolerance on one-cell-wide targets and say so; the search is scored by a route rule stated in full in Part II, and all route methods are rescored under one polyline rule in Part IV. The *Limitations* section lists what is not claimed.

---

## Part I. Can a vision-in, vision-out model read and combine layers?

**Question.** Can a small transformer trained from scratch read single layers, combine conditions from several layers, relate them to a marker, and finally draw a path between two markers around buildings, all in one forward pass? And what does each step up the ladder cost?

<img src="docs/figures/F9_results_table.png" width="760" alt="Seven one-shot task levels with input, target and model output">

*__Figure 2.__ The one-shot task ladder. Seven task levels (rows), one validation example each: the input map layers with the markers as white discs (left), the exact target computed from the data (middle, blue cells) and the model's output in one forward pass (right, blue cells). Under each task name is its headline number on the validation set (IoU of predicted and target cells; for the two path tasks, tolerant metrics within one cell, because the targets are one cell wide). All models have 4.8–8.7 M parameters and are trained from scratch on De Pijp, Amsterdam. The reading: the ladder is climbed up to level 7, and on level 7 the score falls with the number of corners the path has to turn.*

### Method

<img src="docs/figures/F00_concept.png" alt="Aligned layers and markers to patch embedding to transformer blocks to a per-cell head; condition tokens select the task">

*__Figure 3.__ The model. Spatially aligned layers of one 256 m crop (buildings, surfaces, noise, places, sunlit hours, and a plane for the markers; 1 m per pixel, one-hot class planes, 16 channels) are cut into 16 × 16 px patches, giving 256 tokens with a 2-D position code. Two condition tokens name the task. A stack of 2–4 transformer blocks (6 heads, width 384, with an optional learned bias per row-and-column offset between patches) attends over all 258 tokens, and a per-cell head turns each patch token into 4 × 4 output cells of a 64 × 64 mask. Part II adds a scratchpad plane and output tokens for the state.*

Each level is one task, or a mix selected by the condition tokens, with exact targets computed from the planes; the path target is the Euclidean shortest path around the building outlines (Part VI). Full numbers, seeds and run ids: `docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md` (markers and the shortcut) and `docs/reports/C4_SEGMENT_DETOUR_2026-09-25.md` (two markers and paths); every configuration is in the run ledger `runs/experiments.json`.

### Results

<img src="docs/figures/F0_hero_detour.png" width="700" alt="Map with two markers, exact shortest path around buildings, and the model's one-pass output">

*__Figure 4.__ The hardest one-shot task, level 7 of Figure 2: the shortest path around buildings between two markers (white discs). Two validation routes, one per row, labelled by how much longer the path is than the straight distance. Input: the map layers with the two markers. Target: the exact shortest path around the buildings, computed geometrically (blue cells). Model output: what the 8.7 M-parameter, 4-block model draws in one forward pass (blue cells). The shapes are right, the bends at the building corners included; the lines wobble by a cell and can stop short of a marker, and the errors concentrate on routes with several corners.*

Layer-only tasks were easy (2k steps, 4.8 M parameters). Every task that refers to a marker stalled until the marker was drawn as a disc and attention got a relative position bias. "Food & drink within 100 m" sat on a shortcut at 0.41 IoU until it was trained in a mix with the plain "within 100 m" task. The shortest path around buildings needed 100k examples, 16k steps and four blocks, and is right on straight routes, four times in five with one bend, and about seven times in ten with several corners. The details, with the probes that found them, are findings 1–5 in Part V.

### Conclusion

A from-scratch vision-in, vision-out transformer reads aligned layers, combines conditions, relates them to markers and draws paths around obstacles in one pass, given the right spatial encoding (disc marker plus relative bias) and enough data. Its limit is the many-corner route: one forward pass has a fixed number of steps, and a longer route needs more. That is the argument for Part II. Findings 1–5 in Part V.

---

## Part II. Can a search algorithm run as an iterative visual task?

**Question.** If one pass runs out of steps, can the same kind of network run a whole search over many passes, reading back only its own drawing, with no code in the loop that decides anything?

<table><tr>
<td><img src="docs/figures/wave_search_256m_de_pijp_route091_solved_final.png" width="380" alt="Final scratchpad of a solved route"></td>
<td><img src="docs/figures/wave_search_256m_de_pijp_route363_failed_final.png" width="380" alt="Final scratchpad of a failed route"></td>
</tr></table>

*__Figure 5.__ How the search succeeds and how it fails: the final scratchpads of two De Pijp validation routes, free-running from the start disc, drawn by the first search network (4 blocks, 48k steps; report §6). Left, solved: route 91, 191 m walking, 11 passes; the path follows the street and turns at the building corner, with a block of leftover cells from the backward passes near the goal, which the success rule ignores. Right, failed: route 363, 291 m walking, 17 passes; the state tokens are right to the end (DONE), and the path runs correctly along the long building, but just before the goal it has a gap and a detached fragment. The search was right; the drawing of the way back lost continuity. Animations: [solved](docs/figures/wave_search_256m_de_pijp_route091_solved.gif), [failed](docs/figures/wave_search_256m_de_pijp_route363_failed.gif); report §6.*

### Method

**What the network does.** One network, one interface: the map planes + a **scratchpad** (one more input plane) + a **state token** go in; the **whole next scratchpad** + the next state token come out. Output *i* is input *i + 1*, verbatim. Once the network runs, nothing is computed outside it: no route handed over, no tally of past outputs, no rescaling, no code that picks the next step. The network's own token decides the phase (SEARCH, BACK, DRAW) and when to stop (DONE). The algorithm it is taught is a breadth-first wave from the start, then a walk back from the goal along the wave, then one pass that keeps only the path (report §1).

**The data rule.** Every route gets one wave: a Dijkstra search over the free 1 m pixels from the start, with a price per metre that rises from 1 in the open to 5 at a wall, run on the district with a 100 m margin and clipped to the window. Front *j* is the ring of 2 m cells where the walking distance crosses *j* × 40 m, and the way back is the wave's parent links from the goal, drawn three cells wide. A route yields one training pair per pass (a forward pass adds the next front, a backward pass extends the path one front inward and erases that front, the draw pass keeps only the path), and the loader draws one pass at random, forward : backward : draw = 4 : 4 : 1 (report §2). The wave has no choices to make, which is why it is learnable where two earlier search targets, rays to building corners and "key trajectories" along streets, were not (`docs/memo/WAVE_SEARCH_2026-10-01.md`).

**The success rule.** *Solved* means: on the final scratchpad there is a chain of lit cells from start to goal that crosses no cell that is entirely building and is at most 1.10 × the teacher's walking length. The teacher's walking length is that of the clearance-priced wave path above, which keeps away from walls, so a legal chain closer to the buildings can be a few percent shorter than the reference. Extra lit cells elsewhere are not penalised. Free-running means: from the start disc, nothing from the teacher after that.

### Results

500 routes per row; 4 blocks, 8.9 M parameters; network `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k` (48k steps, then one round of natural noise, see below; report §4.6 and §4.9).

| routes (256 m window) | solved | 2–3 fronts (40–120 m) | 4–5 fronts (120–200 m) | 6+ fronts (200 m and more) |
|---|---:|---:|---:|---:|
| De Pijp val | **78 %** | 96 % | 89 % | 63 % |
| De Pijp test | **73 %** | 97 % | 85 % | 49 % |
| Zuid + Centrum val, a district never seen in training | **60 %** | 88 % | 61 % | 40 % |
| house maze, never trained on mazes | **51 %** | 84 % | 51 % | 6 % |

Before the round of natural noise the same network (`c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k`) solved 64 / 61 / 44 % of the first three rows (report §4.3). The house maze is a synthetic district of corridors 8–12 m wide between walls 14–30 m thick, in three styles (perfect maze, braided maze with loops, rooms and doors; `spatial_data/maze_world.py`).

More animations, one frame per pass (chosen best cases): [De Pijp, 424 m](docs/figures/wave_search_256m_de_pijp_route333_424m.gif), [never-seen district, 295 m](docs/figures/wave_search_256m_unseen_district_route342_295m.gif), [house maze, 193 m](docs/figures/wave_search_256m_house_maze_route271_193m.gif), [512 m window, 486 m](docs/figures/wave_search_512m_map_route176_486m.gif), [512 m window, 528 m](docs/figures/wave_search_512m_map_route054_528m.gif).

**The loop needs depth.** The same recipe at 2 blocks (5.3 M parameters) is a few points behind the 4-block network teacher-forced (IoU within 4 m on the forward passes 0.87 against 0.92) and solves **4 %** of the first 200 De Pijp val routes free-running, against 75 % for 4 blocks on the same routes. Its backward drawing collapses once it reads its own output (`c5_r5_wave_nat3_tok01_banneal5_noisep1_L2_48k_g128_store100k`; report §4.1–4.3). Teacher-forced scores did not predict rollout success.

**One output token can ruin a dense output.** Mixing the backward passes into one network first failed: the new piece of path came out as a blob ([figure](docs/figures/wave_hop_blob_token_weight_1.png)). Eleven hypotheses were tested one at a time against a reference known to work (ties in the teacher's way back, pad noise, Dice, the learning-rate schedule, the store, the loader, ...). The cause was the token loss: alone, a pass type has a constant output token and its loss vanishes; in a mix the token is a real decision, and one cross-entropy per sample at weight 1.0 outweighed a grid loss averaged over 16,384 cells. At weight **0.1** false cells on the backward hop fell from 180 to 98 per sample, and the token itself was learnt better (0.998 / 1.00 against 0.96 / 0.96 accuracy) (`docs/memo/WAVE_MIX_TOKEN_LOSS_2026-10-02.md`; report §5.5). Two suggestions on the way were wrong and are kept in the report: group-normalised change weights, and removing the lit-cell weight, which breaks the loop because the network then under-paints.

**Natural noise: training on its own drawings.** In the loop the network reads its own previous drawing, mistakes included, while training had shown it the teacher's. One round fixes much of that: run the network under the teacher's schedule over all 100k training routes, store its drawing after every pass (1.16 M passes, 10.5 min), and continue training for 16k steps with half of the input drawings replaced by these stored ones, the target staying the teacher's clean next drawing. That added 14 points on De Pijp val (64 → 78 %), 12 on test (61 → 73 %), 15 on the never-seen district (44 → 60 %); the teacher-forced numbers did not move. Mixing the stored drawings into a from-scratch run instead was 3–5 points worse at three times the compute. A second round (`c5_r5_wave_nat3_tok01_ownp50_r2_L4_16k_g128_store100k`: 80 / 71 / 60 %) did not add a clear further gain (report §4.6).

**Larger windows and the mazes.** At 512 m the routes reach 1.1 km walking and up to 28 fronts. The 256 m network continued at 512 m on the Zuid + Centrum routes, plus one round of natural noise at that size, solves **42 %** of 500 Zuid + Centrum 512 m val routes: 91 / 71 / 53 / 30 / 4 % for 2–3 / 4–5 / 6–8 / 9–12 / 13+ fronts (`c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512`; report §4.7). On the 512 m house maze it solves 40 % before that round and 39 % after (`c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512`, the network of Figure 1). A thin-wall maze (walls 2–6 m) fails zero-shot: 4 % solved, the first pass already draws a round ring straight through the walls (48k network; report §7).

**The cleanup pass.** The draw pass given a real job: input the network's own final drawing, target the shortest chain through its own lit cells (or the teacher's path when that chain is not a success), with a lower lit-cell weight on that pass. At draw weight 3 the network keeps 70 % of De Pijp val solved and triples the share of clean final drawings (at most 1.25 × the cells of the thin chain) from 14 % to 42 %: a trade of 8 points of success for cleanliness, not a free improvement (`c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g128_store100k`; report §4.8). At 512 m the same recipe fails: 18 % solved against 42 % before, and only 30 % of routes finish with DONE (`c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g256_b64_zc512`; report §4.8), so the cleanup pass is a 256 m result and a negative result at 512 m. Under the stricter polyline rule of Part IV it also loses at 256 m.

**Timing.** On the RTX 3060 in bf16, one pass takes 2.4 ms at batch 1; a whole free-running route, one at a time with the loop around it, 57 ms on average (10.3 passes, grid and token as two forwards per pass) (48k network; report §4.5).

### Conclusion

Yes: an 8.9 M-parameter network runs a breadth-first search on its own scratchpad, free-running, with 4 blocks, a balanced token loss and training on its own drawings. It learnt to run the wave around building-sized obstacles, the kind it was trained on, not around obstacles of any shape. Full account: `docs/reports/C5_WAVE_SEARCH_2026-10-03.md`. Findings 6–8 in Part V.

---

## Part III. How does a vision-language model do? Qwen3-VL zero-shot and fine-tuned

**Question.** How far does a pretrained vision-language model get on the same routes, without training and with LoRA fine-tuning on a growing number of routes?

| Qwen3-VL on De Pijp val | n | solved | 6+ fronts | largest failure |
|---|---:|---:|---:|---|
| 2B-Instruct, zero-shot (two-pass) | 200 | 2.5 % | 0 % | misplaced endpoints |
| 4B-Instruct, zero-shot (two-pass) | 200 | 24.5 % | 2 % | straight lines through buildings |
| 2B + LoRA on 1k routes | 500 | 21.4 % | 1.2 % | |
| 2B + LoRA on 5k routes | 500 | 31.0 % | 7.7 % | |
| 2B + LoRA on 19.3k routes | 500 | **49.6 %** | **28.3 %** | crosses a building (36 % of routes) |

*__Table 1.__ Headline of the Qwen track (no figure; the picture is Figure 6 in Part IV). Solved share by the polyline rule (Part IV), straight-line floor 27.6 %. The zero-shot rows are on the first 200 val routes, which hold fewer long routes than the full 500 (Track A report §3.3), so compare them with the fine-tuned rows with care.*

### Method

Done by Marvin (`qwen_reference/`, `qwen_fine_tuning/`, `docs/reports/TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md`). The model sees the route rendered as a 512 × 512 map with a green start and a red goal, and answers with a polyline in its 0–1000 grounding grid, converted to image pixels for scoring.

- **Zero-shot, two-pass.** The zero-shot numbers come from two prompts per route: a first prompt asks the model where the two discs are, and the route prompt is then anchored on those coordinates (Track A report §2). The fine-tuned models get a single route prompt.
- **An own LoRA training loop** before any library: label masking on the answer span, LoRA r 16 on the attention projections of the language model with the vision tower frozen (about 0.3 % of the parameters trainable), AdamW at 2e-4 with cosine decay, effective batch 4, 3 epochs, bf16 with gradient checkpointing, checkpoints with optimizer state and resume, evaluation by generation and the polyline scorer, all on the RTX 3060.

### Results

Zero-shot, Qwen3-VL-2B-Instruct mostly misplaces the endpoints; Qwen3-VL-4B-Instruct finds them and draws straight lines through buildings, on about half of its routes, and lands just below the straight-line floor. Fine-tuned, the 2B model climbs with data, 81 / 62 / 28 % by 2–3 / 4–5 / 6+ fronts at 19.3k routes; the long routes need the data most. What remains is mostly chords through buildings: on 36 % of the 500 routes the polyline crosses a building, and on 11 % an endpoint is off (Track A report §2.2, §3.4).

Not done: the same run through `trl`'s trainer, GRPO, vision-tower LoRA, transfer to the other district and the maze, and A1, the search itself on a VLM backbone (*Outlook*).

### Conclusion

Zero-shot, the 4B model finds the endpoints but does not plan around obstacles, and the 2B model does not ground the endpoints. LoRA on 19.3k routes lifts the 2B model to 49.6 %, most on long routes; its remaining failure is geometric. Finding 9 in Part V.

---

## Part IV. How do they compare?

**Question.** On the same routes and under one rule, how do the one-shot network, the search and Qwen compare, and do they fail the same way?

![The same three De Pijp val routes drawn by five methods](docs/figures/method_comparison_3_routes.png)

*__Figure 6.__ Three De Pijp validation routes (columns) drawn by five methods (rows): Qwen3-VL-4B zero-shot (two-pass, Part III), Qwen3-VL-2B + LoRA on 19.3k routes, the one-shot network (4 blocks, 64k steps, one pass), the wave search (Part II) and the wave search with the cleanup pass. Buildings dark, the method's line or lit cells orange. Columns, picked from the first 200 val routes by the polyline-rule verdicts: route 57, solved by both the best Qwen and the search; route 188, solved by Qwen + LoRA and not by the search; route 192, solved by the search and not by Qwen. Every subtitle is the polyline rule's verdict, with the length of the line as a multiple of the teacher's walking length. That reference is the teacher's clearance-priced walking path, which keeps away from walls, so a legal line closer to the buildings can be a few percent shorter (0.96 × and similar); the rule's cap is 1.10 ×. The scratchpad drawings are converted to polylines as described below. The reading: the Qwen polylines cut through building corners or whole blocks, while the wave search stays on the street and fails by gaps, a drawing that never connects start and goal.*

### Method

<img src="docs/figures/concept_three_methods.png" alt="One-shot, search and vision-language model: input, model, output, what is scored">

*__Figure 7.__ The three methods as three rows, each read left to right as input → model → output → what is scored, with thumbnails of De Pijp val route 192. One-shot: map planes and a marker plane in, one forward pass, one cell drawing out. Search: the same planes plus a scratchpad in, the whole next scratchpad and a state token out, looped until DONE. Vision-language model: a rendered map image and a prompt in, a polyline in a 0–1000 grid out. The two cell drawings are converted to a polyline, and all three are scored by the same polyline rule.*

The one-shot network is the same 4-block transformer without scratchpad, trained to draw the path in one pass for 64k steps, the search's total training budget (`c5_os_bld_det_L4_64k_g128_store100k`). The systems were first scored by different rules (cells on a scratchpad for the two small networks, a polyline for Qwen). The table puts everyone under one rule, the **polyline rule**: start and end within 6 m of the markers, no building crossed, length at most 1.10 × the teacher's clearance-priced walking length (Part II). A drawing becomes a polyline as the shortest chain through its lit cells, simplified as the teacher's lines are (`harness/diagnostics/drawing_to_polyline.py`); a drawing that does not connect start and goal fails.

### Results

Solved share by the polyline rule, De Pijp validation routes (n = 500; the two vanilla rows n = 200 and their fronts columns cover those 200 routes). ◆ = drawn on a scratchpad and converted to a polyline. Full table, the failure mix and every source file: `docs/reports/COMPARISON_TABLE_2026-10-08.md`.

| method | all | 2–3 fronts | 4–5 | 6+ | never-seen district | house maze |
|---|---:|---:|---:|---:|---:|---:|
| straight line between the markers (floor) | 27.6 % | 73 % | 36 % | 2 % | 28 % | 21 % |
| Qwen3-VL-2B, zero-shot (n = 200) | 2.5 % | 6 % | 2 % | 0 % | – | – |
| Qwen3-VL-4B, zero-shot (n = 200) | 24.5 % | 35 % | 35 % | 2 % | – | – |
| Qwen3-VL-2B + LoRA on 1k routes | 21.4 % | 58 % | 28 % | 1 % | – | – |
| Qwen3-VL-2B + LoRA on 5k routes | 31.0 % | 74 % | 38 % | 8 % | – | – |
| Qwen3-VL-2B + LoRA on 19.3k routes | 49.6 % | 81 % | 62 % | 28 % | – | – |
| ◆ one-shot, 4 blocks, 64k steps | 37.4 % | 83 % | 40 % | 15 % | 31 % | 16 % |
| ◆ **wave search**, 4 blocks | **76.4 %** | **93 %** | **87 %** | **63 %** | **57 %** | **44 %** |
| ◆ wave search + cleanup pass | 46.2 % | 82 % | 54 % | 26 % | 40 % | 32 % |

**The rule matters little for the search, a lot for the cleanup.** The search's number hardly depends on the rule (77.6 % under its own cell rule, 76.4 % here): when it connects, the line is valid in 382 of 395 cases. The cleanup network loses 24 points under this rule: its thin lines hug building corners and touch building pixels the cell rule forgives (1.5 % of its chain cells against 0.2 % for the plain search), which points at its targets, accepted under the lenient cell rule. The one-shot numbers could be up to about 3 points higher with a friendlier cell-to-pixel conversion; the ranking does not change.

**One pass against a loop, at the same budget.** Under the search's own cell rule the budget-matched one-shot network solves 46 % of De Pijp val against the search's 78 %, 35 % against 60 % on the never-seen district, and 18 % against 51 % on the house maze; four times the training of the earlier one-shot run (16k → 64k steps) did not move it, and the gap grows with route length everywhere (report §4.9). A single pass has no more steps to spend on a longer route. At 512 m the one-shot network continued at that size solves 19 % of the 512 m routes against the search's 42 %, and 14 % against 40 % on the 512 m house maze (`c5_os_bld_det_from64k_L4_32k_g256_b64_zc512`; report §4.9).

**The failures differ in kind.** Qwen's remaining errors after fine-tuning are mostly chords through buildings: on 36 % of the 500 routes its polyline crosses a building, and on 11 % an endpoint is off; the zero-shot 4B model failed the same way, on about half of its routes (Track A report §2.2, §3.4). The search's failures are breaks: on the 48k networks no connecting chain crosses a building cell, and failed routes end with a gap or a detached fragment in the drawn way back (report §6; Figure 5). The two failure modes are complementary: Qwen always produces a line with roughly the right ends and fails on geometry; the search never misses an endpoint, because the markers are an input plane, and fails on continuity.

### Conclusion

Under one rule the search leads, and the lead is largest on routes of 6+ fronts. Qualifiers: the grid models are converted from drawings by a shortest-chain rule; Qwen sees a rendered image and answers in coordinates while the grid models get one-hot planes and a marker plane (same routes and scorer, different interfaces); the vanilla rows are on 200 routes; transfer exists only for the grid models. Finding 10 in Part V.

---

## Part V. What we learned

### From Part I: reading and combining layers in one pass

**1. The spatial encoding decides whether marker tasks can be learned at all; for tasks without markers it did not matter.** Tasks that only read the layers, "mark the sidewalks", "quiet sidewalks", reached their numbers with plain absolute position codes in 2k steps. Every task that refers to a marker stalled for thousands of steps or never started. Two changes fixed that, and neither works alone: draw the marker as a small disc instead of a single pixel, so the patch projection can see it, and give attention a learned bias for the offset between patches (the Swin-style relative position bias), so "look at the patches around the marker" is cheap to express. With both, "everything within 100 m of the marker" trains in 1.5k steps instead of 6k.

**2. A good score can hide a wrong solution.** "Food & drink within 100 m" has a shortcut: mark every food & drink place and ignore the marker. It scores 0.41 IoU, and ten configurations sat exactly there, including one with twice the parameters and four times the data. The probe that told the difference compares the prediction with each single-condition mask and plots the marked share by distance from the marker: flat for the shortcut, a step at 100 m for the solution. It is cheap and runs at the first evaluation.

**3. Transfer between tasks happens, in both directions.** Two positive cases: a model trained on pairs of conditions composes pairs it never saw together (0.85 IoU on held-out pairs versus 0.01 with the wrong description), and the hard task "food & drink within 100 m" only became reliable when trained *together* with the plain "within 100 m" task: the easy task builds the circuit that routes information from the marker, and the hard task reuses it. Alone, the hard task escaped its shortcut in about half the runs; in the mix, in 10 of 11. One negative case: training the straight line between two markers alongside the shortest path *hurt* the path task. The two tasks share the output form and the input but contradict each other on the rule (through buildings versus around them), and they compete for the same circuit.

**4. What combined tasks cost.** A single-layer task trains in 2k steps with 4.8 M parameters. A task that combines a marker with a layer condition needed a three-task mix and 16k steps, roughly five times the presentations of the hard task itself. Path finding needed 100k examples, 16k steps at batch 128, and twice the model. The growth is in data and steps more than in parameters.

**5. A 9 M-parameter transformer finds shortest paths around obstacles in one pass, reasonably reliably.** Two markers, the building layer, and the exact Euclidean shortest path as the target. After 16k steps on 100k examples the model traces the right route in most cases: where the route is straight it is essentially perfect, where it bends once it is right four times in five, and where it has to turn several corners it is right about seven times in ten. The errors are concentrated in the many-corner routes, which is where a single forward pass runs out of steps; finding 6 is what more steps buy. Two levers moved this task and they add up: five times more training examples, and four blocks instead of two. Two things that sounded helpful did not: a dense "mark the buildings" side task, and a looser gradient clip.

### From Part II: a search on a scratchpad

**6. A loop over its own output needs depth, and only a rollout shows it.** At 2 and 4 blocks the search recipe is a few points apart teacher-forced and 4 % against 75 % free-running (report §4.1–4.3). A teacher-forced score measures one step from a perfect input; the loop compounds every small error over 10–25 passes, and the shallower network cannot repair what it reads back. Free-running evaluation from the first checkpoint on also showed that rollout success keeps rising long after the teacher-forced metrics plateau (18 % at 20k steps, 75 % at 48k; report §4.4).

**7. A single token can starve a dense output.** One cross-entropy per sample for the state token, at weight 1.0, dominated a grid loss averaged over 16,384 cells and blurred the drawing; at 0.1 both the drawing and the token got better. The general point is the balance of a scalar head against a dense head sharing one trunk, and the way to find it was one ingredient at a time against a reference rerun. The fix is not isolated in the final 48k run (*Limitations*).

**8. Showing the network its own mistakes teaches it to repair them.** Half of the input drawings replaced by the network's own, with the clean next drawing as the target, added 12–15 points free-running at no change in teacher-forced quality, most on long routes and on the never-seen district (report §4.6). Training clean first and repairing afterwards beat mixing the noise in from the start, at a third of the compute. Pushed into the draw pass as a cleanup step, the same idea trades success for clean drawings at 256 m and does not carry to 512 m (report §4.8).

### From Part III: the vision-language model

**9. For the fine-tuned VLM, data is the lever, and the long routes need it most.** Qwen3-VL-2B with LoRA goes from 21.4 % at 1k routes to 49.6 % at 19.3k, and on routes of 6+ fronts from 1.2 % to 28.3 %. Fine-tuning fixes the endpoints more than the geometry: what remains is mostly lines through buildings, the same failure as zero-shot 4B at a much higher solve rate.

### From Part IV: the comparison

**10. Under one rule the search leads, and the methods fail in complementary ways.** The search's lead (76.4 % against 49.6 % and 37.4 %) holds under the polyline rule and is largest on long routes. Qwen fails on geometry, chords through buildings with the right ends; the search fails on continuity, gaps in a drawing that stays on the street. A change of rule can also turn an apparent improvement into a loss: the cleanup network, better under the cell rule's cleanliness metrics, solves 46.2 % under the polyline rule against the plain search's 76.4 %.

---

## Part VI. How it is built and how to run it

### How it is built

- **Data** (`worldsnap/`): seven public layers of De Pijp, Amsterdam, downloaded with a manifest (source, licence, vintage, checksum), rasterised at 1 m, cut into 256 m crops; a derived sunshine layer from a sun-position formula and a ray-march over building heights. A second district, Zuid + Centrum, for transfer tests and 512 m windows. Nothing is cleaned; oddities are documented. Licences per layer: `DATA_LICENSES.md`.
- **Tasks** (`spatial_data/`): targets computed from the planes on the GPU; conditions as tokens; markers as a plane; shortest paths from a visibility graph over the building outlines, verified against a grid search. Held-out ground is four scattered blocks of the district no training crop touches. For the search: one wave per route (`c5_wave.py`), the teacher's pass schedule (`wave_search_schedule.py`), the stores of the network's own drawings and cleanup targets (`c5_own_pads.py`, `c5_cleanup.py`), the synthetic maze district (`maze_world.py`), and the export of routes as map images and polylines for the Qwen track (`track_a_export.py`, also the polyline scorer).
- **Model** (`plain_gpt_module/`): 16 px patches, 2-D sine-cosine positions, 2 or 4 blocks of 6 heads at width 384, an optional learned relative position bias per block, two condition tokens, a per-cell head; for the search a scratchpad input plane, a readout token with a state-token head, and the change-weighted grid loss with its β schedule (`wave_pad_ce.py`, `wave_beta_schedule.py`). Written from scratch; `LOCAL_MODEL_GUIDE.md` is the document it was built from.
- **Training and diagnostics** (`harness/`): a run ledger, and a diagnostics suite that records per evaluation the attention entropy and attention-to-marker of every head, per-group update ratios and gradient norms, and prediction strips. For the search: free-running and scheduled rollout evaluation (`diagnostics/wave_rollout_eval.py`), animations (`wave_rollout_gif.py`), timing (`wave_rollout_timing.py`), the ties diagnostic of the teacher's way back (`wave_back_ties.py`), the one-shot model scored by the search's rule (`oneshot_route_eval.py`), and the conversion of drawings to polylines for the comparison (`drawing_to_polyline.py`). Most of the findings above were read off those, not off the loss.
- **Qwen track** (`qwen_reference/`, `qwen_fine_tuning/`, `models/download_qwen.py`): loading, prompts and coordinate conversion, zero-shot evaluation, the LoRA training loop and adapter evaluation. Requirements in `requirements-track-a.txt`.

![Pipeline](docs/figures/F5_pipeline.png)

*__Figure 8.__ The data pipeline. Top: the public layers of De Pijp, Amsterdam (3DBAG buildings, BGT surfaces, the 2021 noise map, OpenStreetMap and Overture places, and the derived sunlit hours) and the RGB class tile they are combined into. Bottom: one 256 m crop as the model sees it (16 input planes, with a marker) and three of the exact targets computed from it on a 64 × 64 grid: sidewalk, within 100 m of the marker, food & drink within 100 m.*

### Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q          # data-dependent tests skip without the sample bundle
```

`REPRODUCE.md` lists the pipeline commands, layers → crops → path store → one run per task level, then the wave data, the search runs, the natural-noise store, the rollout evaluation and the Qwen track, with wall times on an RTX 3060, and how to load a released checkpoint (the GitHub releases hold one model per one-shot task level, and the search networks, the matched one-shot network and the Qwen LoRA adapter).

### Limitations

- One city. The search was trained on one district (later two, at 512 m) and tested on a second district and on synthetic mazes; no other city was tried.
- **The search's headline depends on the route mix and the round.** The 48k network solves 75 % of the first 200 De Pijp val routes and 64 % of all 500, because the 500 hold more long routes; after one round of natural noise 78 % (val) and 73 % (test). The honest single numbers for the training district are 73–78 %.
- **Single seed** for every search configuration (seed 4). The seed spread of the rollout success rate is not measured; the binomial standard error at 500 routes is about 2 points, per length band 2–4.
- **The success rule scores a set of cells**: a chain of lit cells within 1.10 × the teacher's clearance-priced walking length; extra lit cells are not penalised, which is why the cleanup pass was needed to measure clean drawings at all. The comparison table rescores every system under one polyline rule.
- **Long rollouts are sensitive to numerics.** The same checkpoint on the same 200 routes disagrees on 4 routes between a CPU and a GPU run, all of them with 6–9 fronts (177 against 179 solved; `wave_rollout_eval_cleanliness_cpu.json` against `wave_rollout_eval_val500.json` of the best 256 m network). One cell of difference early is fed back for 15–25 passes. All reported numbers are from GPU runs in bf16.
- **The token-loss fix is not isolated** in the final run: the 48k 4-block recipe was not run at token weight 1.0.
- **The search is taught, not discovered.** The network executes a breadth-first wave given by a teacher, it paints much of the window on long routes, and it is neither faster nor better than classical path finding, which solves these maps exactly. Thin-wall mazes fail zero-shot.
- **No vision-language model runs the scratchpad yet**: Qwen was fine-tuned to answer in one shot; the search on a VLM backbone (A1) is planned, not built. No reinforcement learning anywhere.
- The one-shot path-finding numbers use a one-cell tolerance on one-cell-wide targets; the plain IoU is reported next to them. Most one-shot path-finding configurations are single-seed; the reference specification has two seeds 0.013 apart, and no difference below 0.02 is claimed. Courtyards count as free ground in the path tasks, by decision.

### Outlook

The many-corner routes were the argument for a model that runs a search over several passes with its own previous output as an extra input and a learned stop token; that is now built (Part II). It differs from the first design in `docs/memo/ITERATIVE_SEARCH_2026-09-25.md` in what it draws, fronts one cell wide instead of distance values, and in that the state token, not a fixed pass count, decides the phase; the memo stays as background, the working plan was `docs/plans/C5_PLAN.md`.

Next is **A1: the same dense scratchpad interface on a pretrained vision-language backbone** (Qwen3-VL-2B or Qwen3.5-0.8B): map + scratchpad in, the whole next scratchpad + state token out, output *i* = input *i + 1*, with a dense head on the backbone's patch tokens and the state token from a small head (`docs/memo/TRACK_A_BACKBONES_2026-10-01.md`). It asks whether a pretrained backbone learns the loop with less data, or transfers it better to thin walls and new cities, than the from-scratch network. A further idea, not started: let the language model stream a short commentary per pass alongside the scratchpad, so the search can be followed in words while it runs.

### Related work, in one paragraph

Vision-language map benchmarks (MapEval, MapQA, RewardMap) test frontier models on map images; this project is their controlled counterpart, aligned layers with exact targets and a model small enough to open. The relative position bias is Swin's (Liu et al. 2021); point prompts as tokens (Segment Anything, Kirillov et al. 2023) are the road not taken. The shortcut is shortcut learning in the sense of Geirhos et al. 2020; the heads sinking onto the constant condition tokens are the attention-sink phenomenon (Xiao et al. 2023) in a tiny ViT; the mix findings are small data points for the task-grouping literature (Standley et al. 2020). The search builds on Deep Thinking networks (Schwarzschild et al. 2021; Bansal et al. 2022) and neural algorithmic reasoning (Veličković and Blundell 2021); training on the network's own drawings is the scheduled-sampling / DAgger idea (Bengio et al. 2015; Ross et al. 2011) applied to a scratchpad. The Qwen track fine-tunes Qwen3-VL (Qwen team, 2025) with LoRA (Hu et al. 2021).

### Authorship and licence

Marvin Uhlmann, with Claude (Anthropic) and Cursor Composer as assistants; the split is in `AUTHORSHIP.md`. Code is MIT-licensed (`LICENSE`); data sources and their licences are in `DATA_LICENSES.md`. The Qwen3-VL weights are not part of the repository (`models/download_qwen.py` fetches them) and carry their own licence.
