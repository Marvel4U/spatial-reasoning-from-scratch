# LOCAL_EXPERIMENTS — Marvin's decisions and the experimental dimensions for the local model

Set 20 Sep 2026 after working through `LOCAL_MODEL_GUIDE.md`. Marvin implements; this file records what he decided and what he wants to test. Update it as results come in.

## Decisions for the first version
| # | Decision |
|---|---|
| D1 | **Output head: (c) the dense grid** (the model points, 64×64). Later, for harder tasks, combined with (b) the token decoder. *(Marvin wrote "in combination with c"; read as "with b". Correct if wrong.)* |
| D2 | **Question enters via the marker channel.** Coordinate tokens (option B) rejected. |
| D3 | **Position code: fixed 2-D sine-cosine**, to start. |
| D4 | **Size: 2 encoder blocks** to get everything running and see what is obviously missing. Width/heads as in the translator unless changed. |
| D5 | **Layers are fixed in round one.** Layer meta-information and swapping layers in and out: much later. |
| D6 | **First rung: layers + marker in, marker location out** on the grid. Second rung: marker channel plus task information (task embedding, not a channel). |
| D7 | Code: `plain_gpt_module/` and `harness_from_translator/` were copied verbatim; Marvin adapts both himself. |

## Results and closures
| Date | Item | Outcome |
|---|---|---|
| 21 Sep 2026 | **c0 closed** (Marvin's decision) | Pipeline validated: `c0_worldsnap_P4_2k` reaches exact_cell = 1.0 on held-out val after ~200 steps. **E1 finding:** sub-cell decoding difficulty scales with the number of in-patch positions P²: P = 4 solved; P = 8 right patch 100 %, right sub-cell 46 % at ~1.5k steps; P = 16 right patch 98 %, sub-cell 9 % ≈ chance (1/16) at 2k steps, identical on train and val. Mechanism: a linear patchify learns one column per (in-patch position, channel); a single-pixel marker trains each of the 256 columns on 1/256 of the samples, and balanced CE [1, 4095] makes a full-patch blob nearly free. c0 is the *worst case* for this (single-pixel target); c1 masks label every pixel of every patch, so P = 16 is **not** ruled out. Patch size / front end for later rungs is decided by c1 results, not by forcing c0 through at P = 16. Details: `LEARNING_REPORT.md` §3.5–3.6. |
| 21 Sep 2026 | **c1 L5a closed** | `c1_L5a_building_2k`: single-task building mask, no task input, plain CE @ P=16. Val IoU_pure = 1.0, IoU_fg ≈ 0.99 (`LEARNING_REPORT.md` §3.9). Plan: `docs/plans/C1_PLAN.md`. |
| 21 Sep 2026 | **c1 L5b closed** (area tasks) | `c1_L5b_core_four_2k`: four core tasks + **added task embedding** + fixed per-task CE. Val IoU: building 0.97, sidewalk 0.85, noise≥65 dB 0.99, food & drink 0.46 @ 2k steps. **E6a validated**; rare estab task still hard (§3.10). |
| 21 Sep 2026 | **Establishment targets → cropset `v3b`** | disc targets (`v2`): food & drink IoU 0.48; hosting-building footprints with a 1,000 m² clipped-disc cap (`v3b`): 0.81, same model and recipe. Memo `docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md`. |
| 21 Sep 2026 | **c1 L5c (factorised 12, 2 held out)** | trained tasks fine, held-out combinations not composed (IoU ≈ wrong-task level). Inconclusive by design: 12 eval pairs, one class seen with only one layer, two working layers. Redesigned as the c2 held-out test, `docs/plans/C2_PLAN.md` §6. |
| 21 Sep 2026 | **Balancing** | per-task clamp rejected as non-general (Marvin). General recipe: balanced CE (optionally dampened with one global exponent) + decision rule `P(fg) > w/(1+w)`; within ~0.003 of the per-task oracle on L5b/`v3b` (mean 0.905 plain → 0.931 rule). Memo `docs/memo/BALANCING_AND_DECISION_RULE_2026-09-21.md`. |
| 21–22 Sep 2026 | **E2 + E1 on c1 (sidewalk comparison)** | four core tasks on `v3b`, mean IoU over all 250 val crops per task: scalar P16 0.905, **one-hot P16 0.973**, scalar P8 0.947, one-hot P8 0.996. Sidewalk: 0.853 / 0.977 / 0.878 / 1.000. The scalar encoding, not the patch size, was the limit. Details: `docs/plans/C2_PLAN.md` §2a. |

Next: **c2** (cross-layer conjunctions, `docs/plans/C2_PLAN.md`) on `v3b`, one-hot, P = 16; then **c3** crop-relative tasks (`docs/plans/C3_PLAN.md`). Viewer: `python -m analysis.view_sample --task-rung c1 --ckpt …`.

## Decisions after c1 (22 Sep 2026)
| # | Decision |
|---|---|
| D8 | **Input encoding: one-hot** (was scalar). |
| D9 | **Patch size stays P = 16** as the fast default; P = 8 as a quality check when a result matters. |
| D10 | **Cropset `v3b`** (establishments as hosting-building footprints; over-cap hosts get a cap-area disc clipped to the footprint). Data rules must not jump at a threshold. |
| D11 | **Task set is cumulative and versioned**; every rung gets a single-task control run; per-task metrics are never pooled. |
| D12 | **c2 = cross-layer conjunctions** on the usable combinations only; conditions described by a slot / atom table (`docs/plans/C2_PLAN.md` §5). |
| D13 | **Crop-relative tasks ("the quieter sidewalk") are their own rung, c3** (`docs/plans/C3_PLAN.md`); they need crop-wide context and a finer noise channel. |

## Decisions after c2 (22–23 Sep 2026)
| # | Decision |
|---|---|
| D14 | **c2 closed as a success** (22 Sep): condition tokens + one-hot + P = 16; held-out conjunctions 0.85 vs wrong-description 0.01 (`LEARNING_REPORT.md` §3.15). Marvin runs L6d (CE + Dice) himself. |
| D15 | **Order: c3 (crop-relative) → c4 (path finding: two marker planes + mode token, routes from the OSM per-mode graph, later cost layers such as the quiet or sunny path)** (`docs/plans/C3_PLAN.md`, `C4_PLAN.md`). The masked-layer self-supervision idea comes after c4. |
| D16 | **Sunshine layer** (23 Sep): building shadows from 3DBAG heights by ray march; sunlit-hours plane precomputed into **cropset `v4`** (with the 7-level noise plane); the shadow function stays a callable tool for the viewer and later time-specific questions (`C4_PLAN.md` §4c). |
| D17 | **One sun plane, fixed hour scale (2 … 12 h), the date varies per crop** (23 Sep): not one plane per date and not date-specific breaks; the model learns that days differ in sun overall (memo §5b). |

## Experimental dimensions (each is a planned comparison, not a default)
| # | Dimension | Variants | Why it matters here |
|---|---|---|---|
| E1 | Patch size P | 16 / 8 / 4 (with input resolution 256 or 1024 px) | compression of per-pixel detail; P = 4 is expanding, P = 16 compresses 3,328 → 384. Read out against `boundary_dist_px` |
| E2 | Input encoding per layer | one-hot vs index-as-number | no intuition yet (Marvin); ordering is meaningful for noise, meaningless for surface |
| E3 | Layer fusion | all channels in one patch projection vs **each layer as its own tokens + layer embedding** (larger patches to keep T equal) | RGB must be fused early, our layers need not be; the separate-token variant is also the route to swappable layers |
| E4 | Front end | plain linear patchify vs conv stem | buys back locality |
| E5 | Position code | 2-D sin-cos (start) vs learned row+col vs learned 1-D; sin-cos temperature 10000 vs ~100 | our tasks are about *where* |
| E6 | Task conditioning | added task embedding vs extra token | extra token generalises to metadata/text later |
| E7 | Head | grid (start) → grid + token decoder | complementary; share the encoder |
| E9 | Read-out for single answers | **CLS token** vs mean pooling vs indexed patch read-out (guide §7a) | only relevant once a classifier or decoder head joins the grid head; CLS = learned, content-dependent pooling, and the same mechanism as the "extra token" in E6, so one prepended token could serve as both task token and read-out |
| E8 | Depth/width | from 2 blocks upward | find the smallest model that works |

## Task rungs for the grid head (as of 23 Sep 2026; original 20 Sep proposal replaced)
| Rung | Task | Status | Plan |
|---|---|---|---|
| c0 | mark the marker's cell (plumbing: patch order, position, head alignment) | **closed 21 Sep**; solved at P = 4, mechanism of the P = 16 failure understood | `LEARNING_REPORT.md` §3.5–3.8 |
| c1 | one condition selects what to mark: "sidewalk", "noise ≥ 65 dB", "food & drink" | **closed 21–22 Sep**; four core tasks 0.97 mean with one-hot input | `docs/plans/C1_PLAN.md` |
| c2 | conjunction of two conditions from different layers: "quiet sidewalk" | **closed 22 Sep**; composes held-out conjunctions (0.85 vs 0.01 wrong description) | `docs/plans/C2_PLAN.md` |
| c3 | crop-relative conditions: "sidewalk quieter than typical for this crop"; needs crop-wide context and the 7-level noise plane (cropset `v4`) | next; data decision taken, v4 in progress | `docs/plans/C3_PLAN.md` |
| c4 | path finding: two marker planes + mode token → shortest legal route from the OSM per-mode graph; then routes under a cost layer (quiet / sunny path) with the sunshine layer as prerequisite | planned | `docs/plans/C4_PLAN.md` |
| later | marker-relation tasks ("everything within 20 m of the marker", "same class as the marked point") join c3 or c4; masked-layer self-supervision on real behavioural layers after c4 | idea stage | `survey/raw/08_…`, `09_…` |
