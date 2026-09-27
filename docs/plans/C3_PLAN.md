# C3_PLAN (first draft) — crop-relative tasks: "the quieter sidewalk" **Report:** `docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md` (24 Sep, the whole m1 story + proposed runs).

Written 22 Sep 2026 from Marvin's idea. **Status 24 Sep 2026: c3 closed.** Relative tasks pass (§3b); marker tasks m0 pass (§5b-ii, §8.7); the conjunction m1 is solved at val IoU 0.90 with a 2-block model in a ⅓ three-task mix (report `docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md`, §7.10 and chapter 10 for the recipe and the insights). Next rung: c4 (`C4_PLAN.md` §9).

## 1. The idea and why it is a new kind of task

Instead of an absolute condition ("sidewalk below 55 dB") ask a **relative** one: "mark the sidewalk that is quiet *for this crop*". Closer to what a person asks ("where is it quieter around here?") and the target size becomes more stable.

Every rung so far was **local**: a cell's answer depends only on its own 4 × 4 pixels, and the diagnostics showed attention staying mostly idle in c1. A relative task cannot be decided locally: the same cell (say 60–65 dB) is "quiet" in a crop next to the ring road and "loud" in a residential crop. The model has to survey the whole crop first. **This is the first task that needs attention to do real work**, which makes it the natural successor of c2 rather than a variant of it. Absolute "quiet sidewalk" stays in c2.

## 2. What the data allows (measured)

Native noise bands in the source: 50–55, 55–60, 60–65, 65–70, 70–75, ≥ 75 dB, plus "below 50" = 7 levels. The current input merges them into 4.

**A tempting rule that does not work: "sidewalk in the quietest band present in the crop".**

| | 4 bands | 7 native levels |
|---|---|---|
| Crops where all sidewalk is in one level (no answer) | 0.9 % | 0.0 % |
| Quietest level present is the lowest level of the scale | **98 %** | **92 %** |

Well defined, but not relative: in 92–98 % of crops the quietest band present is simply the lowest band, so the task collapses into the absolute c2 task and the model never needs to look at the crop as a whole.

**A rule that is relative: "sidewalk strictly quieter than the median sidewalk noise of this crop".**

| | 4 bands | 7 native levels |
|---|---|---|
| Median level of the crop's sidewalk | L0 36 %, L1 41 %, L2 23 % | L0 25 %, L1 10 %, L2 13 %, L3 28 %, L4 19 %, L5 4 % |
| Target as share of the crop's sidewalk, p10 / p50 / p90 | 0 / 20 / 45 % | 0 / 38 / 48 % |
| Crops with an empty target (median = quietest level) | 36 % | 25 % |
| **Best score a purely local rule can reach** (fixed "level < t", mean IoU over non-empty crops) | **0.78** | **0.76** |

With 7 levels the reference level genuinely moves from crop to crop, and the target is a stable ~40 % of the sidewalk. The last row is the number that makes the task meaningful: a model that ignores the rest of the crop can reach about **0.76**; anything clearly above that *proves* the model uses crop-wide context.

Consequences:
- The task needs the **finer noise channel** (7 levels) as input: a new cropset (`v4` = `v3b` + native noise bands). With one-hot input that is 7 planes instead of 4; nothing else changes. Rule kept from the old project: the target must be computable from what the model sees.
- The local-only baseline of 0.76 is high. Before building, measure one or two alternative rules with a lower local ceiling (e.g. "quietest third of the sidewalk by rank", or "quieter than the crop mean by at least one band") and pick the one with the largest gap between local ceiling and perfect.
- 25 % empty targets are fine in principle (the model must answer "nothing here is quieter than typical"), but they need the empty-target metrics of the evaluation convention.

## 3. What this rung should show

1. **Attention gets used**: attention entropy drops and heads specialise, unlike c1. Prediction to check with the existing probe: a head that aggregates over sidewalk tokens.
2. **Score above the local ceiling** (0.76) on held-out crops.
3. **Depth matters for the first time**: 2 encoder blocks may be too few to first aggregate and then compare; this is the natural place for the depth comparison (E8).
4. A control that isolates the mechanism: evaluate the trained model with attention restricted to the token itself; its score should fall back to about the local ceiling.

### 3b. Diagnostics (measured 22 Sep 2026 — step 4, 2k steps, v4 `onehot_v4`, 2×2 encoder)

**Step 4 score (full val, 250 crops, rule IoU on non-empty targets)**

| Relative rule | Val IoU | Local-only ceiling (v4 train sweep) | Gap vs ceiling |
|---|---:|---:|---:|
| `median_strict` | **0.813** | 0.762 | **+0.051** |
| `quietest_tertile` | **0.826** | 0.588 | **+0.238** |

Both rules beat the oracle local threshold baseline from step 2. Marvin asked to **train and compare both** rather than pick one upfront; tertile has more sweep headroom, median_strict is the tighter relative task (higher ceiling).

Harness: `c3_quieter_sidewalk_median_strict_2k`, `c3_quieter_sidewalk_quietest_tertile_2k` (`harness/experiments.py`). Checkpoints under `runs/checkpoints/`.

**Attention (fixed 64-crop val probe, final checkpoint)**

| Signal | median_strict | quietest_tertile | c1 L5b onehot (ref) |
|---|---:|---:|---:|
| Mean head entropy (init → 2k) | 5.55 → **4.16** | 5.55 → **4.19** | 5.54 → 4.58 |
| Sidewalk key mass / uniform (L0, L1) | **2.0×**, **2.2×** | **2.2×**, **1.7×** | ~2.2×, ~0.8× |

Entropy falls clearly from uniform (~ln 258 ≈ 5.55) but less than c2 L6b (~1.8 at 2k). **Sidewalk aggregation** is the cleaner story: c3 puts ~2× uniform attention mass on sidewalk patch keys; c2 L6b is **below** uniform (~0.5×) on the same probe geometry.

Training-time logging (new runs): `attn_sw_mass/{layer}`, `attn_sw_ratio/{layer}` in `{run_id}.eval.jsonl` (`harness/diagnostics/probe_metrics.py`). Offline: `python -m analysis.c3_probe_attention --run-id …`.

On **relative** c3 tasks (`median_strict`, `quietest_tertile`), use `attn_sw_mass` / `attn_sw_ratio`, not `attn_to_marker` (probe uses crop centre; no marker). On **marker** tasks (`within_20m`, §5b), log `attn_to_marker` instead.

**Attention ablation (full val; C3 §3 point 4)**

Two masks in `LocalGridViT` / `eval_loader_c3`:

- **`patch_local`** (primary): no patch↔patch attention; cond prefix still visible to all tokens.
- **Strict diagonal**: every token self-only (cond tokens isolated from patches — diagnostic only).

| Rule | Full IoU | Patch-local IoU | Drop | Patch-local vs rule ceiling |
|---|---:|---:|---:|---:|
| median_strict | 0.813 | 0.322 | +0.49 | −0.44 |
| quietest_tertile | 0.826 | 0.230 | +0.60 | −0.36 |

Large drop when patch mixing is removed → **cross-patch attention is necessary** for the trained solution. Patch-local IoU is **not** expected to match the step-2 rule ceiling (0.76 / 0.59): that ceiling is a fixed global threshold on pixels, not “patch-local ViT”. Strict diagonal IoU is much lower still (~0.06 / ~0.27). Offline: `python -m analysis.c3_self_attn_ablation --run-id …`.

**§3 checklist vs these runs**

| # | Criterion | Status |
|---|---|---|
| 1 | Attention used | **Partial** — entropy ↓, sidewalk mass ↑; no per-head sidewalk specialist plot yet |
| 2 | Score above ceiling | **Yes** (both rules) |
| 3 | Depth matters | **Not tested** (§5 step 5) |
| 4 | Ablation | **Done** — large drop; ceiling wording in §3 point 4 was too literal for ViT patch-local |

## 4. Task description

A relative condition is a new kind of "atom" next to the (layer, class) conditions of c2: e.g. `noise < crop-median-on(surface = sidewalk)`. With c2's **condition-token** input (C2_PLAN §4), it would be one more token in the task prefix ("quieter than typical here"), combined with `surface = sidewalk`. The new part for c3 is what the model must compute from crop context, not a new description mechanism.

## 4b. Data decision (23 Sep 2026)

c3 targets are generated on the fly from the input planes, like c2's. The only preparation is the input: **cropset `v4`** = the v3b crops and split with (a) the noise plane at the source's 7 native levels and (b) the sunshine plane (`C4_PLAN.md` §4c), built together because both change only the plane set. c1 / c2 are re-run on v4 as a control before c3 starts. Decided by Marvin 23 Sep; build started the same day.

## 5. Order of work (proposal)

1. Finish c2 on `v3b` (one-hot, P = 16). — **done** (L6a–c, commit `ba91c2a`).
2. Measure alternative relative rules; choose one (script above). — **done** sweep on v4 (`analysis/c3_relative_rule_sweep.py`); **both** `median_strict` and `quietest_tertile` trained for step 4 instead of picking one.
3. Cropset `v4` with 7 noise levels; confirm c1 / c2 scores are unchanged on it. — **done** (c2 gate ~0.979, c1 L5b onehot ~0.974 on v4).
4. Single-task run "quieter sidewalk"; compare against the local ceiling; read the attention diagnostics. — **done** (two rules, §3b).
5. Depth comparison (2 / 4 blocks) and the attention-ablation control. — **ablation done** (§3b); **depth (E8) not started**.
6. Marker-relation (m0 disc, then m1 conjunction). — **m0 @ 20 m passes** (§5b-ii); **N4** (§5e); **m1 @ 8k L2** (§5g). **Next:** **E8 L4 8k** (§5g-iii); then N6 / c4.

## 5b. Marker-relation tasks: the `within_20m` failure and the ladder that follows (23 Sep 2026)

**What was run.** `c3_within_20m_2k`: marker plane (one pixel, from the `t0_point_v0` **first** marker per crop) + condition token `(-1, REL_WITHIN_20M)` → mark every cell whose centre falls in a **Euclidean disc of radius 20 m** at 1 m/px (≈40 m diameter), then 4×4 majority to 64×64. Val IoU **≈ 0**.

**Finding: not an encoding bug.** A probe of the training batch (`docs/plans/perf_prototypes/c3_within_probe.py`) shows one marker pixel per sample, a target disc of ~79 cells whose centroid lies within half a cell of the marker, and a fixed marker per crop. The trained model predicts P(fg) < 0.5 on every cell (max 0.32–0.50, mean 0.12); the marker channel's gradient is exactly 0 at the end of the run; the loss flattened at 0.41 from step 500. This is the **all-background collapse of c0**: a single marker pixel lights one of 256 columns per patch and is trained on 1/256 of the samples; with the dampened weight (α = 0.5, target ≈ 1.9 % of cells) the positive weight is only ≈ 7, so "predict nothing" is cheap and nothing pulls the model out. c0 escaped only because its weight was 4,095. Overfitting 64 crops reaches 0.46 IoU, so the task is learnable.

**Fixes (queued in `harness/experiments.py`, `c3_within_20m_fixes()`, ~1 min each):**

| Run | Change | Expectation |
|---|---|---|
| `c3_within_20m_2k_alpha1` | α = 1.0 → positive weight ≈ 50 | escapes the collapse but may over-mark (see the balancing memo); decision rule applies |
| `c3_within_20m_2k_disc3` | marker drawn as a **3 px disc** (`channels.MARKER_RADIUS_PX`, config `marker_radius_px`; default 0 keeps c0 unchanged) | the marker lights several columns per patch and survives the compression; the more likely fix, and what c4 needs anyway (two markers) |

Pass: rule IoU ≥ 0.8 on held-out crops (report a **geometry oracle** too: read marker position → paint the same disc → IoU, to separate label/plumbing from ViT learning). If both fixes fail, the next lever is the 4,096-way softmax used for c0 to *locate* the marker, then the disc as a second head.

**Train-only: several markers at once (§5d).** Val stays one fixed `t0_point_v0` marker per crop so IoU stays comparable to `c3_within_20m_2k`; training may use **`c3_n_markers` > 0** to draw K fresh markers per batch (see below). Queued runs: `c3_within_20m_multimarker()` alongside the two single-marker fixes.

**Task ladder for marker relations (the disc is plumbing, not the goal):**

| Step | Task | Description to the model |
|---|---|---|
| m0 | within 20 m of the marker (the disc) | marker plane + `REL_WITHIN_20M` |
| m1 | **food & drink within 100 m of the marker** (N5; §5c) | marker plane + `REL_WITHIN_100M` + `estab = food & drink`: c2 conjunction, no new mechanism |
| m2 | same surface class as the marked point; quieter than the marked point | relations that read a value *at* the marker and compare elsewhere |
| m3 | nearest food & drink to the marker | a single answer; needs the ranking / argmax behaviour, possibly the token head |

m1 is the target of this rung; m0 must pass first.

## 6. Decisions for Marvin

1. The relative rule (median-based as measured, or an alternative after step 2). — **Open:** both rules run at 2k; tertile scores higher IoU with more sweep headroom; median_strict is the harder relative target (ceiling 0.76). Default for step 5+ TBD after depth sweep or longer train.
2. Cropset `v4` with 7 native noise levels: for c3 only, or as the new default for everything.
3. Whether empty targets stay in (proposed: yes).
4. Whether the marker-relation tasks stay part of c3 or become c4. — **Decided:** marker-relation stays **c3** through **m1** (food & drink **within 100 m**); **m0** is the bare disc family (`within_*m`); **c4** is path-finding (two markers + graph route), not the disc task.
5. **m1 conjunctive marker task** — **m0 @ 20 m passes** (§5b-ii); N4 m0 @ 100 m ~0.79 (§5e). Build m1 with `within_100m` + estab token; target = disc ∧ layer atom; CE + Dice (§6b).

## 5b-ii. m0 result (23 Sep 2026): multi-marker training solves it

Marvin's idea: train with **K markers per crop** (K = 4, fresh per batch, pairwise ≥ 40 m), target = union of the K discs. Reasoning: K× the marker signal per sample and K× the positive share make "predict nothing" expensive. Built as loader options `c3_n_markers` / `c3_marker_min_dist_m` (K = 0 keeps the fixed single marker). All runs scored on the **same fixed single-marker validation set** as the failed run, so the numbers compare directly.

| Run | Markers per sample (train) | Steps | Val IoU | Course |
|---|---|---|---|---|
| `c3_within_20m_2k` (original) | 1, fixed per crop | 2k | 0.000 | collapse |
| `c3_within_20m_2k_alpha1` | 1, fixed; α = 1.0 (w ≈ 50) | 2k | 0.020 | collapse; val loss 1.42 |
| `c3_within_20m_2k_rand1` | 1, fresh per batch | 2k | 0.000 | collapse |
| `c3_within_20m_2k_rand4_disc3` | 4 fresh + 3 px marker disc | 2k | 0.380 | out of the collapse early, then plateau |
| `c3_within_20m_2k_rand4` | 4 fresh | 2k | 0.559 | escapes at step ~1k, still rising |
| **`c3_within_20m_8k_rand4`** | 4 fresh | **8k** | **0.936** | escapes at ~1k, 0.75 at 4k, 0.90 at 6.5k, 0.94 at 8k; val loss 0.005 |

Readings: (1) fresh markers alone (`rand1`) change nothing; the number of markers per sample is what breaks the collapse. (2) A larger positive weight (α = 1) does **not** rescue the single-marker case, so weighting was the wrong lever here. (3) The four-marker model transfers to single-marker inputs (that is what the validation set contains), so it did not learn "there are always four". (4) The 3 px marker disc starts faster but ends lower; not understood, not pursued. (5) The geometry oracle (N1) scores 1.0, and a batch probe confirmed the marker plane matches the target, so both plumbing checks pass.

**m0 passes at 20 m.** Process note: the 2k batch and the 8k follow-up were started by Claude; from here on Marvin triggers runs (memory rule).

## 5c. Radius for marker tasks (measured 23 Sep 2026, `perf_prototypes/c3_radius_stats.py`)

2,000 v4 train crops, the same markers as the `within_20m` run. "Non-empty" = share of samples whose target has at least one cell; "pos" = mean share of the 64 × 64 grid that is positive.

| Radius | Disc alone, share of grid | food & drink within R: non-empty / pos | any establishment: non-empty / pos | sidewalk: non-empty / pos |
|---|---|---|---|---|
| 20 m | 1.8 % | **17 %** / 0.03 % | 37 % / 0.12 % | 86 % / 0.32 % |
| 50 m | 10.2 % | 38 % / 0.18 % | 67 % / 0.70 % | 99 % / 1.67 % |
| 75 m | 20.8 % | 53 % / 0.35 % | 82 % / 1.42 % | 99.8 % / 3.41 % |
| **100 m** | 33.5 % | **66 %** / 0.57 % | **90 %** / 2.25 % | 99.9 % / 5.50 % |

Reading: at 20 m "food & drink within R" is empty in 83 % of samples, so m1 cannot be trained at 20 m. **150 m dropped from the N4 ladder** (disc fills most of a 256 m crop — not a useful pixel task). **N4 sweep: 50 / 75 / 100 m**; m1 still targets **100 m** for estab coverage (table above). **Decision proposal: radius 100 m for m1** (two thirds non-empty for food & drink, nine tenths for any establishment; the disc still covers only a third of the crop, so the answer is not "everything"). The radius becomes a **condition token** (`within_50m`, `within_100m`, …) rather than a constant, so one family covers several distances and m0 can be trained at the same radius as m1. Even at 100 m the positive share is 0.6 %: the targets are rarer than any c1/c2 task, which is the argument for the frequency-free loss below.

## 5d. Multi-marker training (train split only, 23 Sep 2026)

**Problem:** one marker pixel per 256 m crop is a weak, patch-starved signal (§5b). **Idea (Marvin):** on **train** only, place **K ≥ 1 markers per sample**, far enough apart that their radius‑R discs do not merge trivially.

| | Train | Val |
|---|---|---|
| Marker positions | `sample_markers`: K uniform draws per batch item, pairs ≥ `c3_marker_min_dist_m` px apart (default 40 m) | fixed `(col, row)` from `t0_point_v0`, `pick=first` per crop |
| Marker plane | max over K marker discs (or pixels) on channel 16 | single marker |
| Target (m0) | **union** of K discs of radius R (`_within_radius_pix` → `.any(dim=1)`) | one disc around the val marker |
| Config | `c3_n_markers`, `c3_marker_min_dist_m`; `0` = legacy fixed marker | unchanged |

**Semantics:** train teaches “cells within R of **any** marked point on this plane” — a deliberate **surrogate** to increase marker-channel gradient and fg mass; **val** still scores the real single-marker m0. `K=1` random isolates “fresh random point” from “more markers”. Harness: `c3_within_20m_multimarker()` → `…_rand1`, `…_rand4`, `…_rand4_disc3`. Code: `spatial_data/dataset_c3_gpu.py`, `harness/data.py`.

**Not yet:** sampling one of the many **t0** markers per crop per step (real points, one active marker); that is a follow-up if random uniform markers help but val still fails.

**Train K policy by radius (N4, `within_train_markers()`):** val always one fixed `t0` marker. Train only: **K = 4 @ 20 m**, **K = 2 @ 50 m**, **K = 1 @ 75 m and 100 m** — larger discs need fewer overlapping markers so targets do not become “most of the crop”.

## 5e. N4 radius condition tokens — results (Sep 2026)

**Code:** `REL_WITHIN_50/75/100` (cond ids 15/17/16), `n_cond_emb` → 19 (`spatial_data/c3_tasks.py`, `harness/config.py`). Run ids `{task}_rand{K}` with stable resume (`run_id` override). Harness: `c3_n4_within_radius_runs()`, `_c3_within_multimarker()`. Per-run diagnostics: `runs/diagnostics/{run_id}/` (`tier1.jsonl`, `eval.jsonl`, `dashboard.png`, `pred_step*.png`); `python -m diagnostics.plot_run RUN_ID`.

| Run | Train K | Steps | Val IoU (rule) | Notes |
|---|---|---|---|---|
| `within_50m_6k_rand2` → 8k | 2 | 8k | **~0.92** | **m0 @ 50 m passes**; resume kept same run id |
| `within_75m_rand1` | 1 | 8k | ~0.73 | below 0.8 gate |
| `within_100m_rand1` | 1 | 8k | **~0.79** | best single-marker 100 m; still misses 0.8 |
| `within_100m_8k_rand2` | 2 | 8k | ~0.32 | **failed** — K = 2 + 100 m discs ≈ huge union targets |

**100 m learning curve (L2/H6, full recipe, from `within_50m_6k_rand2` eval log — comparable protocol):** IoU **~0.14 @ 4k**, then **~0.35 @ 5k**, **~0.77 @ 6k** — late escape, not a 4k task for the default stack.

**Process fixes (same period):** removed stray `force_restart: True` on step-4 quieter runs (was breaking auto-resume); `_c3_within_multimarker` sets `force_restart: False`. **`force_restart` defaults off** in harness + experiment builders (Sep 2026). `eval_interval` remains 500; resume skips frozen config keys including `eval_interval`.

**Readings for N5:** m1 @ 100 m should use **K = 1** train (same as passing-ish 100 m m0), not K = 2. 100 m m0 at ~0.79 is usable but not a clean m0 pass — m1 adds conjunctive sparsity (§5c); CE + Dice (§6b) as planned.

## 5g. N5 m1 @ 100 m — results (Sep 2026)

**Task:** `food_drink_within_100m` — cond `(estab food & drink, REL_WITHIN_100M)`, target = disc ∧ estab==3, train/val **K = 1** (fixed t0 marker on val). Loss **CE + Dice** @ 0.5 (§6b). Code: `spatial_data/c3_tasks.py`, `c3_targets.py`, `harness/experiments.py` (`c3_m1_food_drink_within_100m_*`).

### 5g-i. Single-task (primary N5 run)

| Run | Stack | Steps | Val IoU (m1) | empty_fp | Notes |
|---|---|---:|---:|---:|---|
| `food_drink_within_100m_rand1` | L2/H6 | 8k | **~0.41** | **~0.73** | geometry oracle **1.0**; preds still **over-mark estab** vs true conjunctive mask |
| `food_drink_within_100m_smoke` | L2/H6 | 2k | ~0.40 | — | smoke sanity |

**Reading:** Task is **learnable** (beats all-background; IoU well above ~0.006 random fg share) but **not solved**: high empty-target false-positive rate → model often marks **food & drink outside the 100 m disc** or on empty-target crops. Qual: `runs/diagnostics/food_drink_within_100m_rand1/pred_step007999.png`. No checkpoint transfer from `within_100m_rand1` on this run (fresh L2).

### 5g-ii. Three-task training mix (negative)

**Motivation:** Single-task m1 may shortcut to “all estab”; mix **dense estab** + **disc** + **conjunctive** in one batch (c2-style random task per sample) with warm-start from **`within_100m_rand1`**.

**Code:** `c3_task_mode=multi`, `N5_MIX_TASK_KEYS = (food_drink, within_100m, food_drink_within_100m)`, `C3MultiTaskGpuTrainLoader` / val task-major + `iter_batches_for_task` for CE snap on m1. Harness: `c3_n5_m1_mix_8k()` → `food_drink_within_100m_mix_8k`.

| Run | Init | Steps | m1 IoU | `food_drink` | `within_100m` | Notes |
|---|---|---:|---:|---:|---:|---|
| `food_drink_within_100m_mix_8k` | `within_100m_rand1` | 8k | **~0.31** | **~0.95** | **~0.25** | **Worse than single m1**; **forgot** init disc (~0.79 → ~0.25); easy task dominates |

Uniform **⅓** sampling + CE+Dice did **not** fix the estab shortcut; it **destroyed** the loaded distance head by step ~500 while lifting dense estab to ~0.84+. **Not** the portfolio recipe; keep as documented interference / catastrophic forgetting.

**Decision (updated §5g-vii):** Portfolio m1 checkpoint is **`food_drink_within_100m_L4_8k_b32_disc3_relV2` ~0.80 @ 8k** (not the pixel **`rand1` ~0.41**). Do **not** ship multi-task mix. Pixel-marker + L4-only stacks are **superseded** for m1.

### 5g-iii. Next lever (E8 on m1)

Marvin (Sep 2026): **L4**, **batch 32**, **8k steps** (~256k sample presentations ≈ 16×8k baseline), **eval every 1k**, **diagnostics off**. **`RUN`:** **`food_drink_within_100m_mix80_L4_8k_b32`** (mix **80/20** via `c3_train_task_weights=[0.1,0.1,0.8]`), then **`food_drink_within_100m_L4_8k_b32`** (m1 only).

### 5g-iv. Diagnosis (Claude, 23 Sep): the m1 model is the "all food & drink" shortcut, and size will not change that

Probe `docs/plans/perf_prototypes/c3_m1_shortcut_probe.py` on the full 250-crop val set, checkpoint `food_drink_within_100m_rand1`, cut-off 0.5 (crop-mean IoU over non-empty crops; this convention gives 0.27 where the ledger's cell accounting gives 0.42, the ranking is the same):

| Mask compared with the target | IoU |
|---|---:|
| all food & drink cells, marker ignored (shortcut ceiling) | 0.287 |
| disc only, establishment ignored | 0.008 |
| **model prediction** | **0.270** |
| model prediction vs the shortcut mask | **0.911** |

Share of food & drink cells the model marks, by distance from the marker: 0.92 (0–50 m), 0.94 (50–100 m), 0.94 (100–150 m), 0.91 (150–200 m), 0.90 (200–400 m). **Flat.** The model does not use the marker at all; `c3_ablation_drop` 0.01 says the same from the other side (no cross-patch information is used). Its IoU is the shortcut ceiling minus noise.

Why the shortcut wins: target foreground is 0.26 % of cells; food & drink is 0.97 %; the disc is 32 %. "Mark all food & drink" reaches recall ~0.9 at precision ~0.27 within 500 steps with a purely patch-local rule. The only gradient that pushes towards distance comes from the 0.7 % of cells that are food & drink *outside* the disc, and it has to build the same marker-routing circuit that took `within_100m_rand1` (where the disc is the whole target) ~6.5k steps to find. Starved twice over, it never starts. The L4/b32 mix80 runs (§5g-iii) show the same plateau in val loss (0.72–0.79 flat to 5k steps, aborted): **depth does not address the cause**, and by E5b (§8.6) neither will the distance bias, since it never shortened the marker-detection phase.

Levers that address the cause (ordered by cost; Marvin decides):

1. **Warm-start m1 single-task from `within_100m_rand1`** (never tried without the mix; §5g-ii lost the disc because dense estab dominated the batch). The distance circuit then exists at step 0 and the model only has to AND it with the establishment plane. The first 500-step eval tells whether it keeps or forgets the disc.
2. **K = 4 markers per train sample** on m1, the lever that solved `within_20m`: four discs give four times the disc-conditioned contrast per sample.
3. Both together.
4. Add to the c3 eval, for any conjunctive spec: IoU of the prediction against each single-atom shortcut and the marked share by distance band. That is the plot that would have shown this at step 500.

Predicted outcome if none of these is done: any architecture lands at the shortcut ceiling (~0.29 crop-mean / ~0.42 ledger).

### 5g-v. E8 results (Marvin's runs, 23 Sep): the prediction held

| Run | Stack | Steps | val IoU (ledger) | patch-local IoU | ablation drop | empty-target FP |
|---|---|---:|---:|---:|---:|---:|
| `food_drink_within_100m_rand1` (§5g-i) | L2, b16, 5.1 M params | 8k | 0.410 | 0.410 | 0.000 | 0.73 |
| `food_drink_within_100m_L4_8k_b32` | L4, b32, 8.7 M params | 8k (4× the presentations) | **0.417** | 0.418 | −0.001 | 0.72 |
| `food_drink_within_100m_mix80_L4_8k_b32` | L4, b32, mix 80/10/10 | aborted at 3k | val loss flat 0.71–0.74 | | | |

Shortcut probe on the L4 checkpoint: IoU vs the "all food & drink" mask **0.901**, marked share of food & drink cells by distance band 0.94 / 0.92 / 0.93 / 0.91 / 0.87 (0–50 … 200–400 m). Same model as L2, twice the parameters, four times the data. Depth and batch bought nothing because the shortcut is the loss minimum the optimiser can reach from a cold start; the gap to the true solution is a circuit-discovery problem, not a capacity problem. V2 (bias in all blocks) on this recipe is running as of 23 Sep; expected to land on the same ceiling for the reason in §8.6 (the bias never shortened the marker-detection phase) and, at K = 1 train = val, without the count shift of §8.6c.

Levers stay as in §5g-iv, items 1–3. The one that gives the fastest verdict is the warm start from `within_100m_rand1`: the first 1k-step eval says whether the loaded disc survives when the only other signal is the establishment plane.

### 5g-vi. Marker disc3 batch + relV2 (Marvin, 23 Sep 2026)

**Input:** `marker_radius_px=3` on the marker plane (val still fixed t0 point per crop). **Round 1:** L2, batch 16, 8k, diagnostics off. **Round 1b:** same + `rel_pos_bias=all` (E5b V2).

**m0 (within_*m) — disc3 fixes pixel starvation; relV2 adds a large step on top:**

| Task | Pixel IoU (prior) | disc3 | disc3 + relV2 |
|---|---:|---:|---:|
| 50 m (train K=2) | ~0.92 @ 8k L2 | **0.895** | **0.955** |
| 75 m | **0.732** | **0.896** | **~0.998** |
| 100 m | **0.801** | **0.949** | **~0.996** |

**Portfolio defaults for bare disc tasks:** **`marker_radius_px=3`**; **`rel_pos_bias=all`**. (Pixel-marker E5b on m1/L4 was neutral — §5g-v; V2 needs a visible marker.)

**m1 conjunctive (`food_drink_within_100m`) — still shortcut-limited @ L2:**

| Run | IoU (ledger) | patch-local drop | shortcut probe IoU(pred, estab) |
|---|---:|---:|---:|
| pixel `rand1` | 0.410 | ~0 | 0.911 |
| `rand1_disc3` | 0.402 | 0.097 | 0.872 |
| `rand1_disc3_relV2` | 0.405 | 0.024 | 0.878 |

Distance band marking stays **flat** (~88–93% of F&D cells marked); IoU vs disc-only ~0.008. **disc3+relV2 did not break the estab-over-marking minimum** at **L2/b16** — see **§5g-vii** for **L4/b32** (IoU **~0.80**, shortcut broken).

**Mix 80/20 @ disc3:** m1 IoU **0.273** (disc3) vs **0.218** (disc3+relV2); aux `food_drink` ~0.91 — still **not** a curriculum.

### 5g-vii. E8 m1: L4 + disc3 + relV2 (Marvin, 23 Sep 2026) — large IoU jump, hard remainder

**Run:** `food_drink_within_100m_L4_8k_b32_disc3_relV2` — 8k, batch 32, L4, eval/1k, `marker_radius_px=3`, `rel_pos_bias=all`, CE+Dice, diagnostics off (~407 s train).

| Metric | Pixel `rand1` / L4 b32 | **L4 disc3 + relV2** |
|---|---:|---:|
| val IoU (ledger, 0.5 rule) | ~0.41–0.42 | **0.799** |
| patch-local IoU | ~0.41 | **0.343** |
| ablation drop (full − patch-local) | ~0 | **0.456** |
| empty-target FP | ~0.73 | **0.077** |
| val CE @ 8k | ~0.69–0.72 | **0.125** |

**What changed:** cross-patch routing is real (large ablation drop); the **“mark all food & drink”** shortcut is **gone** (probe IoU(pred, estab-everywhere) **0.27** vs **~0.91** on pixel runs). Only **~3%** of F&D **outside** the 100 m disc is marked.

**What is still wrong (why m1 stays hard):**

- Shortcut probe (`c3_m1_shortcut_probe.py`): IoU(pred, target) **~0.76** (crop-mean) vs ledger **~0.80** — same ranking, different accounting.
- **Distance profile is no longer flat**, but **inside 0–100 m** the model still marks **~81–90%** of F&D cells — it learned **radius + density**, not yet a **sharp conjunctive** mask.
- IoU(pred, **disc-only**) **~0.007** — predictions are **not** “disc ∧ estab”; they sit between disc-heavy and estab-heavy errors.
- **Target recall** inside the true conjunctive mask **~84%** (probe); the last **~20%** of IoU to a clean portfolio story is **establishment filtering within the disc**, on **0.26% foreground** cells, after the expensive **marker→radius** circuit finally worked.

**Read:** **disc3 + relV2 + depth/data** was necessary; **pixel marker + none/V2 alone was the wrong problem**. Remaining work is **second-stage conjunction** (estab plane), not more L4 on pixel markers.

**Honest portfolio line:** “m1 @ 100 m: **~0.80 IoU** with visible marker + rel bias + L4; broke global estab shortcut; **not** solved to disc∧estab precision.”

**Levers still open (§5g-iv, reprioritized):**

1. **Warm-start m1** from `within_100m_rand1_disc3_relV2` (disc circuit already at ~1.0 IoU) → m1 fine-tune; only train the AND with estab.
2. **K = 4 train markers** on m1 (surrogate that unlocked 20 m).
3. **Both**; plus mandatory **shortcut + distance-band eval** on every m1 checkpoint (item 4 in §5g-iv).

### 5g-vi. Signal budget until the transition (Marvin's question, Claude's numbers, 23 Sep)

`docs/plans/perf_prototypes/c3_signal_coverage.py`. Two quantities per run, both measured from the train loader (40 batches) and multiplied by batch × steps-to-transition (transition = first clear move of val IoU / val loss, 500-step eval grid):

- **positive coverage** = expected number of positive labels a given output cell has received;
- **marker presentations** = K × B × T, how often the patch projection has seen a marker;
- for m1, **disc-signal coverage** = coverage by the cells that only the marker explains (food & drink inside the disc = positives, food & drink outside = negatives). Plain positives ignore the marker, so they are not the useful signal for the conjunction.

| Run | K | fg cells / sample | T | positive coverage | marker presentations | disc-signal coverage |
|---|---:|---:|---:|---:|---:|---:|
| `c3_within_20m_8k_rand4` (pixel) | 4 | 297 | 1.0k | 1.2k | 64k | 1.2k |
| `within_50m_6k_rand2` (pixel) | 2 | 807 | 4.5k | 14k | 144k | 14k |
| `within_75m_rand1` (pixel) | 1 | 845 | 6.5k | 21k | 104k | 21k |
| `within_100m_rand1` (pixel) | 1 | 1358 | 6.5k | 34k | 104k | 34k |
| `within_50m_rand2_disc3_relV2` | 2 | 807 | 1.5k | 4.7k | 48k | 4.7k |
| `within_75m_rand1_disc3_relV2` | 1 | 845 | 1.5k | 5.0k | 24k | 5.0k |
| `within_100m_rand1_disc3_relV2` | 1 | 1358 | 1.5k | 8.0k | 24k | 8.0k |
| `food_drink_within_100m_L4_8k_b32_disc3_relV2` | 1 | 22 (+48 estab outside) | 6.5k | 1.1k | 208k | **3.6k** |
| `food_drink_within_100m_rand1_disc3_relV2` (L2 b16, no escape) | 1 | 23 (+48) | — | 1.5k at 8k | 128k at 8k | **2.2k at 8k** |

Reading:

1. **Pixel-marker runs were not limited by positive coverage** (1.2k → 34k across radii, a 30× spread) but by **marker presentations** (64k–144k, roughly constant). That is the detection phase: what has to accumulate is gradient on the marker column of the patch projection, and a bigger disc does not help with that. Consistent with the c0 starvation finding.
2. **Disc3 + bias runs transition at 4.7k–8k positive coverage** and only ~24k marker presentations. Detection is no longer the bottleneck; what remains looks like a coverage requirement of a few thousand positives per cell.
3. **m1 fits the same budget once the right signal is counted.** Its disc-discriminative cells are 70 per sample instead of ~1000, so per marker it delivers 14× less distance signal. At the L4 escape it had 3.6k disc-signal coverage, the same order as the pure disc tasks at their transition (4.7k–8k), with 9× the marker presentations to get there. The L2 b16 run reached only 2.2k by the end of its 8k steps, below the band, so it never escaping is what the budget predicts, without invoking depth.

Predictions from the budget (single runs, 500-step resolution, so ±30 %):

- **L2 b32 8k** (the §8.7 depth control): 3.6k disc-signal coverage at 6.5k steps, borderline; escape late or not at all.
- **L2 b16 with K = 4 markers**: 4× the disc signal per sample, crosses 4k coverage at ~3.5k steps; should escape by mid-run. This is the cheapest test of the budget hypothesis and of the K lever at once.
- **Any conjunction with a rarer atom** (e.g. a specific POI class at 0.1 % of cells) will need proportionally more markers or steps; the budget gives the number before the run.

## 5f. 50 m @ 4k fast probes — architecture and LR (Sep 2026)

**Goal:** cheap signal at **4k steps** before committing 8k runs. **Fast recipe** (~98–117 s/run on rig): no tier diagnostics, val CE every 1k, **one end spatial eval** (15 batches, not full 250-crop val). Harness helpers: `c3_within_50m_4k_arch_sweep()`, `c3_within_50m_4k_lr3x()` → ids `within_50m_4k_rand2_{L3,H8,L3H8,lr3x}`.

**Compare cautiously:** fast end-only IoU ≠ mid-training full spatial eval every 500 steps (diagnostic runs); use for **ranking levers**, not absolute gates.

| Run | Model | lr | IoU_fg @ 4k (fast end eval) |
|---|---|---:|---:|
| `within_50m_4k_rand2_L3H8` | L3, H8 | 1e-3 | **~0.22** |
| `within_50m_4k_rand2_L3` | L3, H6 | 1e-3 | ~0.18 |
| L2/H6 full recipe (eval log) | L2, H6 | 1e-3 | ~**0.14** |
| `within_50m_4k_rand2_H8` | L2, H8 | 1e-3 | ~0.04 |
| `within_50m_4k_rand2_lr3x` | L2, H6 | **3e-3** | **0.00** |

**`within_50m_4k_rand2_lr3x` detail:** val CE ~0.76 @ 4k (barely down from init); train CE ~0.95; spatial **all-background collapse** (same failure mode as §5b single-marker).

**Takeaways:**

1. **Learning rate:** **1e-3** is the right scale for this recipe; **3× LR @ 4k** does not accelerate escape — it collapses. No 8k lr3x without a gentler schedule (warmup / lower multiplier).
2. **Capacity @ 4k:** shallow-wide / deeper configs **rank higher at 4k** in the fast sweep (L3H8 best), but **L2/H6 @ 8k** already reaches **~0.92** — E8 depth comparison remains optional vs shipping N5.
3. **Multi-marker vs radius:** K = 2 is appropriate @ 50 m; **do not** reuse K = 2 at 100 m (§5e table).

Optional follow-up (not queued): 8k **L3H8** with full 50 m recipe vs 0.92 L2/H6 baseline.

## 6b. Loss decision (Marvin, 23 Sep 2026)

Marvin's concern: a per-task decision threshold does not scale to compound or interactive tasks where the positive share is unknown in advance. Position: the threshold is a consequence of the frequency weight, not of the model; undoing it (`P > w/(1+w)`) is exact but still requires a per-task `w`. **Rule from c3 onward: new tasks train under a frequency-free loss (cross-entropy + Dice, cut-off 0.5) unless shown impossible.** The c2 L6d comparison (Marvin runs it) decides whether the weighted recipe is dropped for the earlier rungs too.

## 7. Next steps (explicit, in order)

| # | Action | Who | Decides |
|---|---|---|---|
| N1 | Add the **geometry oracle** to the c3 eval | Claude | **done** — agreement 1.0 on all runs |
| N2 | Run **`c3_within_20m_fixes()`** (α = 1.0; 3 px marker disc) | Marvin | **done** — disc3 plateau ~0.35 @ 2k (no pass); α1 failed; see §5b-ii |
| N2b | Run **`c3_within_20m_multimarker()`** | Marvin | **done** — **`c3_within_20m_8k_rand4` val IoU 0.929 @ 8k** (§5b-ii); m0 passes |
| N3 | 4,096-way marker softmax then disc | Marvin | **skipped** — N2b passed |
| N4 | Radius tokens + m0 @ 50 / 75 / 100 m | **done** — §5e (50 m pass ~0.92; 100 m ~0.79 @ K=1; K=2 @ 100 m fails) |
| N4b | 50 m @ 4k arch + LR probes | **done** — §5f (L3H8 best in fast sweep; lr 3e-3 collapses) |
| N5 | **m1 @ 100 m**, single-task, K=1, CE+Dice | Marvin | **measured** — §5g-i **~0.41 @ 8k L2** (partial; not pass) |
| N5-mix | Three-task mix + `within_100m` init | Marvin | **done** — §5g-ii **negative** (~0.31 m1) |
| E8 | **L4, 8k, batch 32**, mix 80/20 + m1-only (pixel marker) | Marvin | **done** — §5g-v (m1 ~0.417; mix negative) |
| E8-disc3 | **disc3 batch** L2 + relV2 (10 runs) | Marvin | **done** — §5g-vi |
| E8-disc3-m1 | **`food_drink_within_100m_L4_8k_b32_disc3_relV2`** | Marvin | **done** — §5g-vii **IoU ~0.80** |
| N5b | m1 **warm-start** from `within_100m_*_disc3_relV2` + optional K=4 | Marvin | **not run** — fastest test of remaining conjunctive gap |
| N6 | Record in `LEARNING_REPORT.md` and `LOCAL_EXPERIMENTS.md`; then c4 | both | after N5b or accept §5g-vii number |
| E5b | Relative position bias (§8) | Marvin | **done** — §8.6; **on** with **disc3** (m0 + m1 L4); **off** on **pixel** m1 |

**Next:** N5b (warm-start / K=4) or N6 with §5g-vii as m1 result. Default harness recipe for new marker tasks: **`marker_radius_px=3`**, m0 **`rel_pos_bias=all`**, m1 conjunctive at least **disc3 + relV2** at E8 scale.

## 8. Testing round: relative position bias in attention (E5b), decided 23 Sep 2026

### 8.1 Why

The marker tasks showed a two-phase course: a long plateau while the patch projection learns "a marker is in this patch", then a transition when block-1 attention routes that flag to neighbouring patches (on K=2 / K=1 slow runs, min block-1 entropy often falls from ~5.5 toward ~1–3 as IoU escapes; after the probe fix, heads 1 and 4 of block 1 put 54–62 % of mass on the marker token). **Primary transition metrics:** IoU and patch-local ablation drop — not min entropy alone (`within_100m_rand1` can show very low L1 entropy from ~500 steps while IoU stays flat until ~6.5k). Plateau duration tracks train K: 4 markers ~1k, 2 markers ~3–4k, 1 marker ~5–6k (IoU on `within_75m_rand1`).

A **relative position bias** hands attention distance directly: a learned score add per (Δrow, Δcol) bucket. The disc task is a clean testbed; **prediction:** much shorter plateau and earlier non-zero IoU — not parity with c1 (~0.97 @ 2k).

### 8.2 Where it comes from

- **Shaw, Uszkoreit, Vaswani 2018, "Self-Attention with Relative Position Representations"**: the original idea for 1-D sequences: attention scores get a term from a learned embedding of the offset i − j, clipped to ±K.
- **T5 (Raffel et al. 2020)**: the simplified form used here: a learned scalar bias per offset *bucket* per head, added to the logits before the softmax, shared across layers. T5 has no absolute position embedding at all.
- **Swin Transformer (Liu et al. 2021)**: the **2-D** version this plan proposes: for patches in a window, a table of (2M−1)² learnable biases per head indexed by (Δrow, Δcol); added to the attention logits. Swin's ablation: relative bias beats absolute position embeddings on classification and detection, and adding absolute positions on top *hurts*. BEiT and later ViT variants reuse it.
- **ALiBi (Press et al. 2022)**: the no-parameter cousin: a fixed penalty proportional to distance, with one slope per head. Useful as the control that has no learnable bias but still knows distance.
- Related but different: DeBERTa (content-to-position terms), rotary embeddings (relative position folded into q and k by rotation; what Qwen3-VL uses in 2-D form as MRoPE). Our fixed 2-D sin-cos code is the *absolute* baseline these papers compare against.

### 8.3 Implementation sketch

Optional, **off by default** (`rel_pos_bias = "none"`) so earlier checkpoints load unchanged. Harness + `LocalGridViTConfig`: `rel_pos_bias`, `rel_pos_max_offset` (K), `use_sincos_pos` (V3). Code: `plain_gpt_module/rel_pos_bias.py`, `local_grid_vit.py`.

- **Which blocks:** `"first"` | `"all"` → per-block `nn.Embedding(n_buckets, n_head)`, zero-init (step 3+).
- **K:** clip Δrow/Δcol to ±K → **(2K+1)²** spatial buckets; **+1** bucket for any pair involving a prefix (condition) token. K=7 → 225+1; K=15 covers the full 16×16 patch grid exactly (offsets up to ±15).
- **Threading:** `rel_pos_bucket_idx` (buffer) built in `LocalGridViT.__init__`; passed block → `BidirectionalSelfAttention` (step 4).

### 8.3b How it is built

Attention scores every pair of tokens (i, j) from Q·K, then softmax over j. The change adds a **learned scalar per bucket per head** that depends only on patch geometry:

1. **Precompute once** a buffer of shape **T×T** (`T = n_prefix + num_patches`). For two **patch** tokens, look up the bucket of (Δrow, Δcol) on the patch grid, each axis clipped to ±K → **(2K+1)²** buckets. Any pair where **either** token is a prefix (condition) token uses the **extra** bucket (index (2K+1)²). Built in the model constructor from `patch_grid`, `n_cond_token_slots`, and K (`build_rel_pos_bucket_index` in `rel_pos_bias.py`).
2. **`nn.Embedding(n_buckets, n_head)`** per enabled encoder block — one bias per bucket per head.
3. In **`BidirectionalSelfAttention.forward`:** embed the T×T index table → **(T, T, n_head)**, permute to **(1, n_head, T, T)**, pass to **`scaled_dot_product_attention` as a float `attn_mask`** (float masks are **added** to logits before softmax). If a boolean key-padding mask exists, convert disallowed positions to **−inf** and add to the same float mask.

That is the whole mechanism (~15 lines in forward once wired); nothing else in the model must change for V1. A head can express “two patches to the right” or “mass within four patches” with a few weights — what the disc task needs and what today often takes thousands of steps to approximate from sin-cos inside token contents. The T×T table is also where a **hard distance cutoff** on attention would live later (neighbourhood mask + bias).

Patch tokens still get **fixed 2-D sin-cos** on the patch embed when `use_sincos_pos=True` (default). **V3:** `use_sincos_pos=False` with bias in all blocks (Swin-style ablation).

### 8.4 Experiments (Marvin triggers; each ~1–10 min on the 3060)

Testbed: `within_75m`, **K=1** train (slow IoU escape ~5–6k); control `within_20m`, K=4. Optional sanity: `within_100m` (entropy can decouple from IoU). 8k steps, same seeds, fixed single-marker val. Read: IoU @ 2k/4k/8k, step IoU crosses ~0.2, `attn_to_marker/1/*`, patch-local drop.

| Variant | Setting | Question |
|---|---|---|
| V0 | bias off (today) | baseline curves, already have them |
| V1 | bias in the **first** block only | is one block with distance enough to remove the plateau? |
| V2 | bias in **all** blocks | does the second block need it too? |
| V3 | V2 **without** the sine position code | Swin's claim: relative alone is better than relative + absolute |
| V4 | ALiBi-style fixed slope, no learned table | how much is the *learnability* worth vs merely knowing distance? |
| V5 | K = 3 vs K = 7 vs K = 15 | bias **clips** pairs with \|Δ\| > K; 75 m ≈ 4.7 patches @ 16 m/patch — K=3 (±48 m) is intentionally tight |
| V6 (timing open: right after this round or later, Marvin decides) | **2-D rotary positions** (MRoPE-style: dimension pairs assigned to row / column, q and k rotated by position in every attention layer, no sine code on the content, no parameters) | the "same family as Qwen" control: relative by construction but content-dependent, vs the content-free learned bias |

Pass: IoU escape on `within_75m` moves from ~5–6k to **under ~2k**, final IoU not lower. Regression: c1 core four + c2 core five with V2 within **0.01** IoU (scratch train, same val crops).

| # | Action | Who | Status |
|---|---|---|---|
| E5b-0 | Config + T×T bucket buffer | Claude | **done** (§8.3b) |
| E5b-1 | Embedding + attention forward | Marvin / Claude | **done** |
| E5b-2 | V1 `within_75m` 8k | Marvin | **done** — §8.6 |
| E5b-3 | V2 `within_75m`, V1/V2 `within_50m`, V1/V2 `within_20m` 8k | Marvin | **done** — §8.6 round 2 |
| E5b-4 | §8.6c/d: `within_20m_rand_varK_relV2` | Marvin | **done** — **dead start** (§8.6d); var-K **open** on retry / seed |

### 8.6 Results (measured)

Harness: `c3_rel_pos_e5b_*`, batch `c3_rel_pos_e5b_round2_runs()`. Same recipe as N4 within runs (weighted CE, α=0.5, diagnostics every 500). Compare val rule IoU @ **7999** and **first step with IoU &gt; 0.2** on `runs/diagnostics/{run_id}/eval.jsonl`.

#### 8.6a Round 1 — 75 m V1 only

**`within_75m_rand1_relV1`** (`rel_pos_bias=first`, K=1) vs **`within_75m_rand1`**:

| Step | Baseline IoU | relV1 IoU |
|---:|---:|---:|
| 0–5k | ~0 | ~0 |
| 6500 | ~0.30 | **~0.53** |
| 7000 | ~0.39 | **~0.63** |
| 7999 | **~0.73** | **~0.74** |

First IoU &gt; 0.2: **~6500** both. Mid-escape sharper with V1; **8k ceiling ≈ tie** (+0.008); both below 0.8. Patch-local ablation drop ~**0.72** at end — still cross-patch routing.

#### 8.6b Round 2 — V1/V2 across radii (Sep 2026)

| Task | Baseline (no bias) | relV1 `first` | relV2 `all` |
|---|---:|---:|---:|
| **75 m** K=1 (`within_75m_rand1` …) | **0.729** @ 8k; escape **~6500** | **0.737** @ 8k; escape ~6500 | **0.653** @ 8k; escape **~5000** |
| **50 m** K=2 (`within_50m_6k_rand2` …) | **0.922** @ 8k; escape ~4500 | **0.737** @ 8k (peak **0.772** @ 7500); escape ~3500 | **0.526** @ 8k (peak **0.647** @ 5500, then falls); escape ~3000 |
| **20 m** K=4 (`c3_within_20m_8k_rand4`) | **0.929** @ 8k; escape ~1500 | **0.961** @ 8k; escape ~1500 | **0.027** @ 8k (**collapse** — val CE ~3.2; peak ~0.46 @ 3k only) |

**Patterns (interesting / portfolio-grade):**

1. **§8.4 pass not met.** Nothing moves the long plateau to **&lt;2k**. The only early-escape shift is **75 m V2** (~5k vs ~6.5k), and it **trades away final IoU** (−0.08 vs baseline).

2. **V1 `first` is the safe variant:** ~tie or small win on **75 m** and **20 m**; does **not** rescue **50 m** (already easy without bias — rel runs land ~0.74, well below ~0.92 baseline).

3. **V2 `all` is harmful or unstable here:** worse ceiling on 75 m; **mid-run peak then regression** on 50 m; **hard collapse** on 20 m (not a small IoU dip — runaway val loss). Likely interaction of **two learned bias tables + sin-cos + small data** rather than “distance is bad.”

4. **Early escape ≠ good final model.** 50 m and 75 m V2 can cross IoU 0.2 **before** baseline yet finish far worse — useful anti-pattern for interview storytelling.

5. **Mechanism unchanged:** all successful runs still need **patch mixing** (large patch-local drop when trained). Bias did not turn the disc into a local rule.

**Decision (E5b):** **`rel_pos_bias=none`** only for **pixel-marker m1** (§5g-v). With **`marker_radius_px=3`**: **`rel_pos_bias=all` for m0**; for **m1 at E8 scale**, **disc3 + relV2** is the measured stack (**~0.80 IoU**, §5g-vii). L2 disc3+relV2 without L4 **did not** move m1 (~0.405). Code default stays **none** (zero-init). c1/c2 regression with V2 **not** run.

Diagnostics: `runs/diagnostics/within_*_relV*/` vs baselines `within_75m_rand1`, `within_50m_6k_rand2`, `c3_within_20m_8k_rand4`.

#### 8.6c Probe (Claude, 23 Sep): the V2 "collapse" is a marker-count shift, not an optimisation failure

Evidence from the run logs: in `within_20m_rand4_relV2` the **train** loss keeps falling (0.30 → 0.09 from step 2.5k to 8k) while the **val** loss climbs from 0.08 to 9.6. The bias tables are small (rms 0.1–0.5, logit shifts of order one), so nothing ran away. Train samples carry K = 4 random markers, the val set carries 1 fixed marker. Scoring each checkpoint on train-distribution batches with K = 1 / 2 / 4 markers (`docs/plans/perf_prototypes/c3_marker_count_probe.py`, 20 batches each):

| Run (train K, bias) | val, 1 marker | train K=1 | train K=2 | train K=4 |
|---|---:|---:|---:|---:|
| `c3_within_20m_8k_rand4` (4, none) | 0.961 | 0.962 | 0.960 | 0.957 |
| `within_20m_rand4_relV1` (4, first) | 0.961 | 0.962 | 0.960 | 0.957 |
| `within_20m_rand4_relV2` (4, all) | **0.027** | 0.026 | 0.128 | **0.750** |
| `within_50m_6k_rand2` (2, none) | 0.927 | 0.927 | 0.926 | 0.920 |
| `within_50m_rand2_relV1` (2, first) | 0.718 | 0.718 | **0.809** | 0.747 |
| `within_50m_rand2_relV2` (2, all) | 0.515 | 0.518 | **0.793** | 0.729 |

(The 20 m rand4 baseline row was probed with the same script; its numbers coincide with V1 because both are count-invariant at this precision.)

Reading: **without bias the model is invariant to the number of markers** (all columns equal). **With bias it is best at exactly the K it was trained on** and degrades away from it; "all" degrades hardest (20 m: 0.75 at K = 4 vs 0.03 at K = 1). So the bias did not break training; it let the model learn a solution that depends on how many markers are in the image, and the single-marker val set exposed it. The 75 m runs (K = 1 train and val) show no such gap, which is why they looked fine.

Why the bias makes this shortcut available: a learned bias lets a head keep a **content-free, near-uniform** pattern at no cost (block-1 entropies in the V2 runs sit at 5.4–5.5, i.e. uniform over 258 tokens), which sums the marker flags over the whole crop into a "marker count" feature. With a constant K in training that feature is a constant, and the head/MLP calibrate the logit scale against it; at another K the calibration is off and the output falls to all-background. Content-driven attention (no bias) has to *learn* a uniform pattern and never did. This is a hypothesis consistent with the entropies and the table, not yet a proven circuit.

Consequences:

1. The §8.6 conclusion "V2 harmful" stands for the current recipe, but the cause is a **train/val mismatch we created ourselves** (multi-marker training, single-marker val), which the bias merely exploits. Any recipe change that keeps K fixed in training carries the same risk for other tasks.
2. Cheap fix worth one run before closing E5b: **draw K per sample from {1, 2, 3, 4}** (or per batch) so the count is not constant. Prediction: V1/V2 recover count-invariance and the 20 m V2 run lands near V1's 0.96; whether the plateau still does not move is then a clean answer (it did not move in V1, so probably not).
3. Also reachable from the same loader change: N5 (m1) trains at K = 1 and validates at K = 1, so it is safe as is.

**Probe run (executed):** `within_20m_rand_varK_relV2` — see **§8.6d** (IoU **0** @ 8k, early cond-token sink; **not** a clean test of §8.6c count-invariance). Repeat with new seed optional; keep **`rel_pos_bias=none`** for N5/E8 regardless.

#### 8.6d `within_20m_rand_varK_relV2` (Marvin, 23 Sep): a dead start, not a result about variable K

Run: `rel_pos_bias=all`, K ~ Uniform{1..4} per train sample, val K = 1, 8k steps. IoU 0.000 at every eval, train loss flat at 0.74 (the all-background level) from step 500 to 8k, head logit std 0.01, both bias tables never above rms 0.005.

Checked and ruled out (`docs/plans/perf_prototypes/c3_vark_loader_probe.py`): the new loader is correct. One batch has 1–4 marker pixels per sample, no marker at the corner from the −1 padding slots, target foreground share scales with K, and the overrides differ from `within_20m_rand4_relV2` only in the three marker fields.

What actually happened (`c3_head_sink_probe.py`): at the first eval (step 500) **all 12 heads** had attention entropy 0.0, i.e. every query row put its mass on one token, and that token is one of the two **condition tokens** (fraction of rows whose top key is a prefix token: 0.99–1.00 in both blocks at the final checkpoint; mass on self 0.0). The condition tokens are constant across samples, so once every head reads only them, every token receives the same attention output and no information crosses patches; the per-patch path alone cannot see a marker in a neighbouring patch, the grid head settles on the all-background answer, and nothing feeds a gradient back into the dead query/key weights. The heads drift back towards uniform after step 6.5k but the output never recovers.

This is a **known early failure mode of this model, not of the bias or of variable K**. Count of heads with entropy < 0.5 at the first eval, across all runs with diagnostics:

| Run | collapsed heads at first eval | final IoU |
|---|---:|---:|
| `c3_within_20m_2k_disc3` (no bias) | 11/12 | 0.348 (recovered) |
| `food_drink_within_100m_rand1` (no bias) | 10/12 | 0.417 (recovered) |
| `c3_within_20m_8k_rand4` (no bias) | 6/12 | 0.929 |
| `within_75m_rand1` (no bias) | 6/12 | 0.729 |
| `within_20m_rand4_relV2` | 6/12 | 0.750 on K = 4 (§8.6c) |
| `within_50m_rand2_relV2` | 1/12 | 0.526 |
| **`within_20m_rand_varK_relV2`** | **12/12** | **0.000 (never recovered)** |

Marker tasks sink heads early far more than c1/c2 (0–2/12 there), because for the first few hundred steps the marker plane carries no usable signal and the constant condition tokens are the lowest-variance thing to attend to. Every earlier run had at least one head left and recovered from it. This one had none. The bias runs do not sink more heads than the baselines, so the bias is not the cause; the LR schedule has effectively **no warmup** (`WARMUP_STEPS = 10`, then cosine from 1e-3, `MIN_LR_RATIO = 0.1`), which is the usual reason such sinks form in the first steps.

Consequences:

1. The variable-K question (§8.6c item 2) is **still open**; this run is not evidence either way. A repeat with a different seed is the cheapest way to answer it.
2. Robustness item for the harness, independent of E5b: an LR warmup (a few hundred steps) is the standard guard against early attention sinks; a dead-start detector (all heads entropy < 0.5 at the first eval) would let a run abort at step 500 instead of burning 8k steps. Both are one-line changes; Marvin decides whether either goes in before N5 continues.
3. For c3 single-task runs the condition tokens carry no information at all (one task), so they act purely as sinks. That is worth remembering when reading any c3 entropy plot.

### 8.7 Round 3 (Marvin, 23 Sep): 3-px disc marker + bias in all blocks — the two levers are complementary

All runs `*_disc3_relV2`: `marker_radius_px = 3`, `rel_pos_bias = all`, diagnostics off, otherwise the N4/N5 recipes. The E5b conclusion in §8.6 ("bias does not move the plateau") was confounded: it was measured on a **single-pixel** marker, where the first phase (patch projection learns to detect the marker at all) dominates and the bias cannot act until it is over. With a 3-px disc the marker is detectable from the start, and the bias then does exactly what §8.1 predicted.

| Task | train K | before: best pixel-marker run | disc3 + V2 | val loss < 0.1 first at |
|---|---:|---:|---:|---:|
| `within_50m` | 2 | 0.922 (escape ~4.5k) | **0.955** | 1.5k (vs ~4.5k) |
| `within_75m` | 1 | 0.729 (escape ~6.5k) | **0.998** | 1.5k (vs ~6.5k) |
| `within_100m` | 1 | 0.794 (escape ~6.5k) | **0.996** | 1.5k (vs ~6.5k) |
| m1, L2 b16 | 1 | 0.410 | 0.405 (still the shortcut; distance profile flat 0.88–0.93) | never |
| m1, L2 mix 80/10/10 | mixed | 0.31 | 0.218 | never |
| **m1, L4 b32** | 1 | 0.417 | **0.799** (ledger); crop-mean 0.757 | val loss 0.51 → 0.21 → 0.13 at 6k / 7k / 8k |

Shortcut probe on `food_drink_within_100m_L4_8k_b32_disc3_relV2`: marked share of food & drink cells by distance band **0.90 / 0.81 / 0.03 / 0.02 / 0.04** (0–50 / 50–100 / 100–150 / 150–200 / 200–400 m). The model now uses the marker: a clean step at 100 m, 3 % false positives outside the disc, 84 % of target cells found; ablation drop 0.46 (was 0.00), empty-target FP rate 0.08 (was 0.72). This is the first conjunctive marker task that leaves the shortcut. It left it **late** (val loss only moved between 6k and 7k) and the run ended while still improving, so 0.80 is not its ceiling.

Reading:

1. **Two phases, two levers.** Disc marker = detection phase short; relative bias = routing phase short. Either alone was measured before and did not suffice (disc3 alone at 20 m: 0.35–0.38 at 2k; bias alone: §8.6). Together, pure disc tasks train in ~1.5k steps at K = 1, which was the §8.4 pass criterion (< 2k) after all.
2. **The conjunction still needs something on top.** At L2 the shortcut wins even with both levers; at L4/b32 the model escapes at ~6.5k. The escape is a discovery event, not a capacity effect per se: L4 gives the optimiser more heads/blocks in which a distance circuit can grow while the shortcut occupies the others.
3. The 50 m run (K = 2 train, K = 1 val) reached 0.955 on the single-marker val set, so the count shift of §8.6c is mild when the recipe otherwise works; still vary K when K > 1.
4. §8.6's "V2 harmful" is withdrawn for the disc-marker recipe. The count-shift finding (§8.6c) and the dead-start finding (§8.6d) stand as such.

Open, Marvin's call, in order of expected value:

- **m1 L4 b32 disc3 V2 continued to 16k** (or warm-started from its own 8k checkpoint): val loss was still falling at 8k.
- **m1 L2 with K = 4 markers** disc3 V2: the cheapest way to raise the disc-side gradient without depth; if L2 escapes, capacity was not the point.
- Diagnostics-on repeats (Marvin is running these) to see the escape event: block-1 entropy drop and attn_to_marker at the 6–7k step.
- The marker token (§9 proposal, not yet written) remains the principled fix for the detection phase and the prerequisite for two-marker c4 tasks; with disc3 + bias working, it is no longer urgent for c3, and it should be tested against this recipe, not against the single-pixel baseline.

### 8.8 m1 L4 16k with diagnostics (Marvin, 24 Sep): where the circuit lives, and why the early escape is not the LR schedule

`food_drink_within_100m_L4_16k_b32_disc3_relV2`, same seed (1337) and recipe as the 8k run, horizon 16k, diagnostics on. Val IoU 0.41 → 0.40 → **0.78** at 1k / 2k / 3k, then 0.83–0.86 from 4k to 10k (plateau; val loss still drifting down 0.09 → 0.06). Ablation drop 0.78–0.85, empty-target FP 0.01–0.07.

**Circuit (attn_to_marker, single-marker val batch):** one head, **block 3 head 1**, puts 0.99 of its attention on the marker token from step 3k on and keeps it (0.96–0.99 to 10k); a precursor is visible at 2k (block 3 head 2 at 0.47). Blocks 0–2 never attend to the marker (≤ 0.05). Block 0 heads are sunk to the prefix tokens from 1k on (entropy ≈ 0), block 1 goes uniform (5.5) after the escape. So in the L4 model the distance routing sits in the **last** block, and the first three blocks contribute little beyond per-patch features; the extra depth gave the optimiser spare heads in which a marker head could form, it did not add computation the task needs. This is the evidence on the depth question so far: L2 is not ruled out, and the L2 b32 / L2 K=4 controls in §5g-vi remain the test.

**The LR-horizon explanation for the earlier escape (2–3k vs 6–7k) does not hold up.** Cosine to 16k vs 8k gives LR 0.97 vs 0.87 of peak at step 2k and 0.93 vs 0.72 at 3k, a 10–28 % difference. The 8k run escaped at LR 0.13–0.23 of peak, so a *higher* LR is not what escapes need, and at the steps where this run escaped the 8k run had nearly the same LR and did not. The two runs already differ at step 2k (train CE 0.60 vs 0.70) after matching at 1k (0.685 vs 0.684), i.e. they diverged as trajectories, which with the same seed means GPU nondeterminism (scatter/index kernels, bf16) plus the chaotic timing of a saddle escape. Read: **escape time from the shortcut is a high-variance event**; two samples so far at 2.5k and 6.5k. Setting `lr_horizon_steps` explicitly is good hygiene regardless, but it is not the lever.

**Signal budget (§5g-vi) update:** this run escaped at ~1.4k disc-signal coverage, the 8k run at 3.6k. The budget is a scale (order 10³ per cell), not a threshold; variance across seeds is at least 2–3×.

What would settle escape variance cheaply: the same L4 b32 spec at 8k horizon with two new seeds; if escapes land anywhere between 2k and 7k, the variance is confirmed and single-run timing comparisons stop being evidence.

### 8.5 What it changes if it works

- Marker tasks (m0–m3) and c4 routes become cheap to train; the plateau phenomenon becomes a documented finding rather than a cost.
- The position code becomes an experimental dimension of its own (E5 + E5b): absolute sin-cos vs relative bias vs both, with the marker task as the discriminating benchmark. That is a clean, reportable comparison for the portfolio.
- Two-track note: the Qwen track has rotary 2-D positions (relative by construction); if relative bias is what makes the local model route, the two tracks become more comparable, not less.

## 10. Closing note (24 Sep 2026)

c3 ends here. The final recipe for marker tasks and marker conjunctions, measured in the m1 report: 2 blocks, 6 heads, width 384; batch 32; 3-px disc marker; learned 2-D relative position bias in all blocks; the hard task trained in a uniform three-task mix with its bare marker task; 1e-3 cosine, 16k steps (8k learns it, 16k adds the last hundredths). Verification: candidate-mask probe, outside-disc share, wrong-token test. Everything in §5–§8 above is the trail that led there; the report is the summary to read.
