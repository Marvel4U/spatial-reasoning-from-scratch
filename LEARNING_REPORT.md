# LEARNING_REPORT — Track B local grid ViT (c0 + c1)

Living log of **what we ran**, **what it shows**, and **what it does not show**. Canonical numbers live in `runs/experiments.json`; checkpoints under `runs/checkpoints/`. Specs: `LOCAL_MODEL_GUIDE.md`, `LOCAL_EXPERIMENTS.md`, `IMPLEMENTATION_PLAN.md`, `docs/plans/C1_PLAN.md`, `PLAN.md`.

**Last updated:** 22 Sep 2026 (c2 L6 + **held-out wrong-description** eval; [`C2_PLAN.md`](docs/plans/C2_PLAN.md)).

---

## 1. System under test

| Piece | Status | Notes |
|-------|--------|--------|
| **Model** | `LocalGridViT` ~3.96M params | 2× `EncoderBlock`, `n_embd=384`, `h_heads=6`, patch P=16, grid 64×64, K=2 |
| **Input** | Cin=4 (scalar) or **13 (one-hot)** | **Decided:** one-hot @ P=16 on v3b; c2 runs set `encoding_mode=onehot`; global harness default still scalar (optional flip) |
| **Task input (c2)** | **Condition tokens** (13-way + pad, 2 slots) | Prefix before 256 patches; D1 **done** — gate ≈ one-hot L5b |
| **Cropset** | `v3b` for c1/c2 forward work | Footprint estab + 1000 m² cap; same crop ids as v2 ([`ESTAB_FOOTPRINTS`](docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md)) |
| **Target (c0)** | One fg cell on 64×64 grid | Marker echo — *not* T0 class readout |
| **Loss** | Class-weighted CE (`balanced_ce=True`) | Weights `[1, n_bg/n_fg]` on per-cell CE (~4095:1 for c0) |
| **Harness** | `harness/` | `python harness.py` from `harness/`; logs to `runs/experiments.json` |
| **Data** | Synthetic pools *or* worldsnap | `data_source`: `synthetic` \| `worldsnap`; de_pijp `crops/v3b`, tasks `t0_point_v0` |

**Task rung vs task family:** **c0** = “given marker in input, predict marker cell on grid.” **T0** = “at marker, what is surface/noise class?” — jsonl from `t0_point_v0` supplies **crop_id + marker_256** for c0 today; the **class answer in jsonl is not trained yet**.

**Geometry (default):** P=16 → 16×16 patch tokens; each token predicts a **4×4** subcell block on the output grid (`subcells=4`). c0 is therefore **two coupled subtasks**: (1) which patch contains the marker, (2) which subcell inside that patch.

---

## 2. Implementation progress (for context)

| Phase | Scope | Status |
|-------|--------|--------|
| **A** | `plain_gpt_module/`: pos embed, patch embed, grid head, `local_grid_vit.py` | Done |
| **B1–B3** | `spatial_data/channels.py`, `targets.py`, `dataset_c0.py` | Done |
| **B4** | Harness ↔ worldsnap (`harness/data.py`, config caps) | Done |
| **B5** | Separate `setup.py` | Skipped — device lives in `harness/data.py` |
| **C** | Harness train/eval/checkpoint/resume | Done |
| **Analysis** | Tier1/2 diagnostics + `analysis/view_sample` | Done — [`analysis/README.md`](analysis/README.md), [`docs/plans/ANALYSIS_SUITE_SPEC.md`](docs/plans/ANALYSIS_SUITE_SPEC.md) §0 |
| **Perf** | GPU-resident loader, vectorized eval, thread cap | Done — [`docs/plans/PERF_BASELINE_2026-09-21.md`](docs/plans/PERF_BASELINE_2026-09-21.md) |
| **c1 data/model** | `spatial_data/c1_*`, `task_emb`, multi-task loaders | Done — [`C1_PLAN.md`](docs/plans/C1_PLAN.md) |
| **D** | Sanity ladder L1–L6 | c0 L1–L3; **c1 L5a–L5c** (§3.9–3.12); **c2 L6a–c** **done** §3.15 ([`C2_PLAN.md`](docs/plans/C2_PLAN.md) §7); L6d + polish **optional** |
| **Memos** | Target + eval conventions | [`ESTAB_FOOTPRINTS`](docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md), [`BALANCING_AND_DECISION_RULE`](docs/memo/BALANCING_AND_DECISION_RULE_2026-09-21.md) |

---

## 3. Completed runs (summary)

All on **the RTX 3060 rig** (RTX 3060 12GB) unless noted. Spatial eval uses `harness/eval_spatial.py` (random val batches unless noted).

### 3.1 `c0_overfit_32` — synthetic L1 ✅ (wiring)

| | |
|--|--|
| **Purpose** | Prove model + loss + harness can memorize 32 fixed tiles |
| **Data** | Synthetic; 32 train / 32 val, **`val_same_as_train=True`** |
| **Steps** | 400, batch 32 |
| **Result** | train/val CE → ~4e-6; **IoU_fg=1.0**, **exact_cell=1.0** |
| **Checkpoint** | `runs/checkpoints/c0_overfit_32.pt` |

**Takeaway:** End-to-end consistency. **Does not by itself prove in-patch subcell decoding** — with 32 tiles and heavy repetition the model can memorize tile-specific solutions; re-run with §3.5-style metrics to see pred_fg block structure.

---

### 3.2 `c0_synthetic_1k` — synthetic at scale

| | |
|--|--|
| **Purpose** | c0 on 4096 train / 512 val synthetic tiles, 1000 steps |
| **Result** | train CE ~9e-4, val CE ~0.05; **IoU_fg≈0.37**, **exact_cell≈0.07** |
| **Checkpoint** | `runs/checkpoints/c0_synthetic_1k.pt` |

**Takeaway:** Same CE vs pointing tension as worldsnap, milder. Worth re-eval with decomposed metrics (§4).

---

### 3.3 `c0_worldsnap_smoke` — worldsnap plumbing

| | |
|--|--|
| **Purpose** | End-to-end worldsnap load + 200 steps on full 32k/4k pools |
| **Result** | CE ~0.69 (flat); **IoU_fg≈0**, **exact_cell=0** |
| **Checkpoint** | `runs/checkpoints/c0_worldsnap_smoke.pt` |

**Takeaway:** ~0.1 epoch — plumbing only.

---

### 3.4 `c0_worldsnap_overfit_32` — worldsnap L1 ✅ (wiring)

| | |
|--|--|
| **Purpose** | Same as synthetic L1 on **real** npz (first 32 train jsonl rows) |
| **Data** | `worldsnap_max_train_items=32`, **`val_same_as_train=True`** |
| **Steps** | 400, batch 32 |
| **Result** | CE → ~1e-4; **IoU_fg≈0.96**, **exact_cell≈0.92** (spatial eval 320 grids) |
| **Checkpoint** | `runs/checkpoints/c0_worldsnap_overfit_32.pt` |

**Takeaway:** **Marker channel, npz layers, and c0 targets on de_pijp v2 are aligned.** High exact_cell on 32 memorised tiles still allows shortcutting subcell structure — run decomposed eval on this checkpoint for comparison to §3.5.

---

### 3.5 `c0_worldsnap_full_2k` — full pool, ~one epoch

| | |
|--|--|
| **Purpose** | Train c0 on all **32 000 train / 4 000 val** items from `t0_point_v0` (marker positions only) |
| **Steps** | 2000 × batch 16 ≈ **32k sample presentations** (~1 pass with replacement) |
| **Time** | ~257 s |
| **Headline (legacy eval)** | train/val CE **~0.02**; **IoU_fg≈0.05**, **exact_cell=0.0** |
| **Checkpoint** | `runs/checkpoints/c0_worldsnap_full_2k.pt` |

**Learning curve (val CE):** ~0.73 until ~step 600 → drop **700–1000** (0.47 → 0.04) → tail ~0.015–0.04. The cliff aligns with learning **patch-level** marker presence, not fine subcell placement.

#### Diagnosis (checkpoint eval, 1000 val items, read-only script)

Method: load `c0_worldsnap_full_2k.pt`, iterate shuffled val jsonl (same data as harness), metrics aligned with `eval_spatial.py` (Sep 2026). Reference scratch: `~/.cache/claude_scratch/diag_c0.py` on the RTX 3060 rig.

| Metric | Val (n=1000) | Interpretation |
|--------|----------------|----------------|
| Predicted fg cells == full 4×4 patch block (16 cells) | **87%** | Model often fires **entire subgrid of one patch token** |
| True cell predicted fg (per-cell recall) | **99.9%** | True cell almost always inside predicted fg set |
| Share of predicted fg inside **true patch** | **92%** | Blobs are patch-local, not random scatter |
| **Global argmax** (fg score) in true patch | **97%** | Patch localization generalises to held-out val |
| **Subcell correct, given argmax in true patch** | **7.1%** | ≈ **chance 1/16 = 6.25%** — in-patch position not learned |
| Global argmax exact cell | (high; ~97% patch × ~7% subcell) | Hidden by exact_cell when 16 cells are argmax-fg |

**Train split (same script):** patch hit **96.7%**, subcell given patch **6.3%** — **not a train/val generalisation gap**; same partial solution on both splits.

**Takeaway (revised):** This is a **positive mechanistic result**. Linear patchify + dense balanced CE learns **“marker in this 16×16 px patch”** quickly, then settles on a **low-CE blob** (all 4×4 subcells in that patch). IoU ~0.05 ≈ 1 true / ~16–20 predicted fg cells. CE ~0.02 is **consistent with that blob**, not a misleading log bug: one miss costs ~4095 weight units; fifteen in-patch false positives cost ~15.

**Open detail:** mean predicted fg cells **~31** vs median **16** — ~1/8 of grids have a larger blob (second patch or spillover); not yet characterised by tile type.

---

### 3.6 E1 patch sweep — `c0_worldsnap_P{4,8,16}_2k` (Sep 2026)

**Setup:** Full worldsnap pool (32k / 4k), lr=1e-3, `balanced_ce`, layer cache on. Queue from `experiments.py` / `C0_WORLDSNAP_P_SWEEP`. Same 2k-step budget per P (actual steps below if stopped early).

| Run | P | g×s | Batch | Steps run | Throughput (typ.) |
|-----|---|-----|-------|-----------|-------------------|
| `c0_worldsnap_P4_2k` | 4 | 64×1 | 4 | **889** (skipped after solved) | ~30–35 samples/s (N=4096 tokens) |
| `c0_worldsnap_P8_2k` | 8 | 32×2 | 8 | **1527** (skipped) | ~50–95 samples/s |
| `c0_worldsnap_P16_2k` | 16 | 16×4 | 16 | **1999** (full) | ~115–180 samples/s |

#### End-state spatial eval (harness `c0 detail`)

| Run | val CE | IoU_fg | exact_cell | argmax_patch | full_patch | subcell\|patch | global_argmax_exact |
|-----|--------|--------|------------|--------------|------------|----------------|---------------------|
| **P4** @ 889 | ~0 | **1.0** | **1.0** | **1.0** | 1.0 (1×1) | **1.0** (chance 1.0) | **1.0** |
| **P8** @ 1527 | ~0.003 | **0.25** | **0** | **1.0** | 1.0 (2×2) | **0.46** (chance 0.25) | **0.46** |
| **P16** @ 1999 | ~0.019 | **0.058** | **0** | **0.98** | 0.88 (4×4) | **0.086** (chance 0.062) | **0.084** |

#### Dynamics (what to watch — **`c0 snap`**, not CE alone)

- **P4:** Val CE still ~0.68 at step 100; by **step 200** snaps show **single fg cell**, all metrics 1.0. Task is “which of 4096 cells” with **no in-patch subproblem** (s=1). Early stop after ~200–400 steps is enough for future P4 runs.
- **P8:** By **~400** steps: `argmax_patch=1`, **full 2×2 patch** predicted (4 fg cells) → **IoU=0.25** by construction; `subcell|patch` rises above **0.25** but **`exact_cell` stays 0**. Not “stuck” — same blob mechanism as P16 with smaller blocks.
- **P16:** Matches §3.5: cliff ~600–1000 on CE; end state **median 16 fg cells**, **~98% patch**, **subcell at chance+**.

**Takeaway:** **E1 confirms the mechanism.** In-patch ambiguity scales with **s = 64/P**: shrinking P removes L2b difficulty (P4 passes product metrics on val); P16 does not with current loss. **Low CE + zero exact_cell at P8/P16 is expected**, not a sign that training failed.

**Checkpoints:** `runs/checkpoints/c0_worldsnap_P{4,8,16}_2k.pt`

---

### 3.7 `c0_worldsnap_P16_2k` + diagnostics (throughput vs learning)

**Purpose:** Same c0 recipe as §3.6 P16, with **`diagnostics_enabled=True`** so tier1 JSONL records **`gpu_starvation`** and **`samples_per_s`** (dashboard **plot 7**: “loader vs throughput”). Compare **pre–perf-plan** runs (CPU `next_batch`, high starvation) to **post–perf** (`worldsnap_gpu_resident=True`, `cpu_num_threads=4`, vectorized eval).

| Phase | Harness | Plot 7 (typical) | Notes |
|--------|---------|------------------|--------|
| Pre-perf diagnostics | layer cache, 6 CPU threads, CPU batches | `gpu_starvation` **≈0.85–0.95**, **~100–180 samples/s** | First `c0_worldsnap_P16_2k` diagnostic run; see `runs/diagnostics/c0_worldsnap_P16_2k_plots.png` |
| Post-perf (rerun) | + `worldsnap_gpu_resident`, `eval_interval=250` | expect starvation **≲0.1–0.2**, **~1500–2000 samples/s** @ B=16 | **`experiments.py` → `c0_p16_2k_diagnostics_perf()`** |

**Before overwriting artifacts**, archive the old run:

```bash
mkdir -p runs/diagnostics/archive
mv runs/diagnostics/c0_worldsnap_P16_2k.jsonl runs/diagnostics/archive/c0_worldsnap_P16_2k_preperf.jsonl
mv runs/diagnostics/c0_worldsnap_P16_2k.eval.jsonl runs/diagnostics/archive/c0_worldsnap_P16_2k_preperf.eval.jsonl
mv runs/diagnostics/c0_worldsnap_P16_2k_plots.png runs/diagnostics/archive/c0_worldsnap_P16_2k_preperf_plots.png
# optional: meta, pred PNGs, runs/checkpoints/c0_worldsnap_P16_2k/
```

Train: `cd harness && python harness.py`. Then:

```bash
cd harness
python -m diagnostics.plot_run c0_worldsnap_P16_2k
python -m diagnostics.plot_run c0_worldsnap_P16_2k --compare c0_worldsnap_P16_2k_preperf
# (copy archived *.jsonl back to runs/diagnostics/ with _preperf names for --compare)
```

**Learning metrics (plots 1 & 4)** should track §3.6 P16 after perf fixes; only **wall time and plot 7** change materially. Tier2 probe cost remains non-trivial on a 3060 (~1 min/extra per full CE eval if probe batch 64).

### 3.9 `c1_L5a_building_2k` — L5a dense masks (no task input) ✅

| | |
|--|--|
| **Purpose** | Validate c1 mask targets, GPU loader, plain CE, c1 eval before task conditioning ([`C1_PLAN.md`](docs/plans/C1_PLAN.md) §10 L5a) |
| **Task** | Surface **building**, majority downsampling; marker channel **zeros**; **no** `task_emb` |
| **Setup** | P=16, B=16, 2k steps, `worldsnap_gpu_resident`, de_pijp v2; **2000 train / 250 val crops** (pool cap on rig) |
| **Pass** | Val **IoU_pure_cells = 1.0** @ step 1999 (proposal ≥ 0.9); **IoU_fg ≈ 0.986**, IoU_mixed ≈ 0.945 |

**Takeaway:** Under **dense** supervision, P=16 is **not** stuck in the c0 single-pixel / balanced-CE blob trap. Mixed-cell gap (~5 pp vs pure) is the expected 4×4 downsampling effect, not subcell chance.

### 3.10 `c1_L5b_core_four_2k` — L5b on **v2 discs** ✅ (area pass; estab fail)

| | |
|--|--|
| **Purpose** | Task conditioning (E6a): same encoder, four requests → four masks |
| **Tasks** | building, sidewalk, noise≥65 dB, food & drink (`task_emb`, fixed per-task CE weights, clamp 100) |
| **Cropset** | **v2** — 2 m establishment discs, estab targets use **“any”** pixel rule |
| **Setup** | P=16, B=16, 2k steps, diagnostics on; val **512 fixed (crop, task) pairs** |

**Spatial eval @ step 1999:** IoU_fg **0.828**; pure/mixed **0.818 / 0.760**; P/R **0.837 / 0.986**. Per-task IoU: building **0.969**, sidewalk **0.846**, noise≥65 **0.989**, food & drink **0.463**. `empty_target_fp_rate ≈ 0.73` (estab empties + w≈100 → over-marking). See [`docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md`](docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md).

**Takeaway:** **Task_emb works** on area tasks at P=16. v2 estab failure is largely **geometry + loss weight**, not missing conditioning — superseded on v3b (§3.11).

### 3.11 `c1_L5b_core_four_2k_v3b` — L5b on **v3b footprints** ✅ (full four-task pass)

| | |
|--|--|
| **Purpose** | Same L5b recipe on **building-hosted establishment** labels (1000 m² host cap); [`ESTAB_FOOTPRINTS`](docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md) |
| **Cropset** | **v3b** — estab **majority** downsampling; food & drink fg weight **≈41** (not clamp 100) |
| **Setup** | P=16, B=16, 2k steps; same crop ids as v2 (noise/surface bit-identical) |

**Spatial eval @ step 1999** vs v2 L5b @ same step:

| Metric | v2 disc | v3b footprint |
|--------|---------|---------------|
| IoU_fg | 0.828 | **0.907** |
| IoU_pure / mixed | 0.818 / 0.760 | **0.967 / 0.829** |
| Precision / recall | 0.837 / 0.986 | **0.916 / 0.989** |
| empty_target_fp_rate | 0.73 | **0.27** |
| food & drink IoU | 0.463 | **0.798** |

Area tasks unchanged (~0.97 / ~0.85 / ~0.99). Plots: `runs/diagnostics/c1_L5b_core_four_2k_v3b_plots.png`, compare vs v2 `…_vs_c1_L5b_core_four_2k_compare.png`.

**Takeaway:** Footprint estab turns food & drink into a **region task**; threshold sweep on v2 (memo §1) showed ~half the v2 gap was CE weight — v3b fixes both weight and target scale. **Production default for c2 prep:** one-hot @ P=16 (§3.14) supersedes scalar L5b numbers above for sidewalk and mixed cells.

### 3.12 `c1_L5c_factorised_*` — 12 tasks, factorised `layer_emb + class_emb`, hold-out **surface_0 + estab_2**

| Run | Cropset | Steps | IoU_fg | Held-out IoU (correct / wrong / gap) | Notes |
|-----|---------|-------|--------|--------------------------------------|--------|
| `c1_L5c_factorised_2k` | v2 | 2k | ~0.69 | — | v2 estab still disc/any weights |
| `c1_L5c_factorised_4k` | v2 | 4k | **0.714** | 0.24 / 0.21 / **+0.034** (n=52) | estab_3 IoU **0.25**; gap barely above wrong-task |
| **`c1_L5c_factorised_2k_v3b`** | **v3b** | 2k | **0.731** | 0.24 / 0.14 / **+0.097** (n=52) | estab_1 **0.58**, estab_3 **0.47**; held-out **estab_2** 0.23 (not trained) |

Train on **10/12** atomic (layer, class) pairs; val includes held-out tasks. **Pass (proposal):** held-out IoU ≫ wrong-task baseline — **v3b shows a clearer gap (+0.10)** than v2 @ 4k (+0.03), but absolute held-out IoU (~0.24 on surface_0) is still weak; **compositional generalisation is partial**, not solved.

**v3b @ 2k per-task (spatial eval):** noise **0.97–0.98**, surface building **0.95**, sidewalk **0.78**, estab food **0.47**, estab other **0.58**. Diagnostics: `runs/diagnostics/c1_L5c_factorised_2k_v3b_plots.png`.

**Why c2 replaces L5c-style factorisation:** summing `layer_emb[L] + class_emb[k]` **cannot bind** layer to class. **Decided (22 Sep):** **condition tokens** — one embedding per (layer, class), fixed length-2 prefix before patch tokens ([`C2_PLAN.md`](docs/plans/C2_PLAN.md) §4).

### 3.13 C2 prep — encoding sweep, balancing memo, data/eval plumbing

**P2 ✅ (`C2_PLAN` §5a):** four core tasks on **v3b**, 2k steps, **all 250 val crops per task** (check scripts). Marvin's decision **22 Sep: one-hot, P=16** (P=8 optional for quality checks).

| Variant | mean IoU | sidewalk | food & drink | IoU_mixed | ~train |
|---------|----------|----------|--------------|-----------|--------|
| scalar P=16 (`c1_L5b_core_four_2k_v3b`) | 0.905 | 0.853 | 0.807 | 0.83 | 32 s |
| **one-hot P=16** (`…_v3b_onehot_P16`) | **0.973** | **0.977** | **0.921** | **0.94** | 40 s |
| scalar P=8 | 0.947 | 0.878 | 0.929 | 0.91 | 91 s |
| one-hot P=8 | 0.996 | 1.000 | 0.985 | 0.99 | 99 s |

Sidewalk was limited mainly by **scalar grey-level ambiguity** (0.67 between roadway and building), not patch size alone. Runs: `c1_L5b_core_four_2k_v3b_{onehot_P16,scalar_P8,onehot_P8}` in `experiments.json`.

**Balancing memo** ([`BALANCING_AND_DECISION_RULE_2026-09-21.md`](docs/memo/BALANCING_AND_DECISION_RULE_2026-09-21.md)): weighted CE shifts fg log-odds by `log w`; **decision rule** `P(fg) > w/(1+w)` recovers most of the gap to a per-task oracle threshold on existing checkpoints (e.g. v3b L5b food & drink 0.807 → **0.885** at plain 0.5 vs rule). **c2 eval:** rule + oracle + condition-failure attribution in **`harness/eval_c2.py`** (C2_PLAN §6 B1). **c1 / plot 4:** rule/oracle in shared eval rows still **open** (B2).

**c2 data (22 Sep):** `spatial_data/c2_tasks.py`, GPU loaders + district sampler, cached `c2_task_stats.json`, 21 tests — see C2_PLAN §6 A1–A5, C2.

### 3.15 c2 L6 — condition tokens + conjunction ladder (v3b, one-hot P=16, dampened CE + rule IoU)

**Architecture (D1):** two learned condition embeddings + slot positions prepended to patch tokens (`encoder_seq_len=258`); grid head reads patches only. Batches are `(img, target, task_index, cond_ids)`; per-task loss weights still keyed by `task_index`.

**Runs (fixed 2k train crops unless noted; full val grid 250×tasks):**

| Run | Setup | Mean rule IoU | Pass (C2_PLAN §7) |
|-----|--------|---------------|-------------------|
| **`c2_gate_ts2_singles_2k`** | four `ts2` singles, tokens | **0.978** | ≈ one-hot L5b (~0.973) — **input path validated** |
| **`c2_L6a_quiet_sidewalk_2k`** | one conjunction, **no task input** | **0.948** on `noise_0+surface_2` | ≥ 0.8 — **AND without description works** |
| **`c2_L6b_ts2_2k`** | 4 singles + 5 core conj, tokens | **0.953** mixture mean | Area conj ≥ 0.8; singles mostly track gate ( **`estab_3`** slightly softer ) |
| **`c2_L6c_c2_full_2k`** | 35 tasks, **4 conj held out** of training, **4k steps** | trained **0.899** / held-out **0.852** rule mean | **Pass** on generalisation + wrong-desc (below) |

Checkpoints: `runs/checkpoints/c2_{gate_ts2_singles,L6a_quiet_sidewalk,L6b_ts2,L6c_c2_full}_2k.pt`. Qualitative panel gallery: `runs/diagnostics/l6c_task_gallery/` (`analysis/generate_l6c_gallery.py`, one L6c checkpoint).

**Held-out conjunctions (never sampled in training; evaluated on full val):** `noise_0+surface_1`, `noise_1+surface_3`, `noise_2+surface_2`, `noise_1+estab_2`. Mean rule IoU **~0.85** vs **~0.90** on trained tasks — clear lift over L5c-style factorisation (~0.24 held-out) and over the scalar c1-mask AND baseline (core quiet sidewalk **0.66**, food **0.62**; C2_PLAN §6). Weakest held-out absolute score: **`noise_1+estab_2`** (55–65 dB ∧ shop); three noise×surface held-outs are stronger.

**Wrong-description control (22 Sep, L6c checkpoint):** on each non-empty val crop for the four held-out tasks, second forward with **slot-0 atom flipped** (same layer, class `(c+1) mod 4` via `wrong_cond_ids_padded`); IoU still measured against the **correct** AND target. Implemented in **`harness/eval_c2.py`** (auto when `held_out` is set); re-run: `analysis/run_c2_wrong_description.py`.

| Held-out task | IoU correct (rule) | IoU wrong desc | Gap |
|---------------|-------------------:|---------------:|----:|
| `noise_0+surface_1` | 0.890 | 0.005 | **+0.885** |
| `noise_1+surface_3` | 0.927 | 0.021 | **+0.906** |
| `noise_2+surface_2` | 0.894 | 0.008 | **+0.886** |
| `noise_1+estab_2` | 0.698 | 0.014 | **+0.684** |
| **Pooled** (n=970 non-empty) | **0.856** | **0.012** | **+0.844** |

Wrong IoU ≈ **0.01** → predictions follow the **condition prefix**, not a single map-specific mask (contrast L5c wrong-task gap ~+0.10 at much lower absolute correct IoU).

**L6d (22 Sep, `c2_L6d_ts2_2k`):** same **ts2** as L6b; **`grid_loss=ce_dice`** (plain CE + soft Dice on fg, **no** per-task `f` weights). Primary score = **IoU @ 0.5** (`iou_plain`).

| Metric (mean over 9 ts2 tasks) | **L6b** dampened CE + rule | **L6d** CE + Dice @ 0.5 |
|--------------------------------|---------------------------:|------------------------:|
| `iou_plain` | **0.936** | 0.913 |
| `iou_rule` | **0.953** | 0.906 |

Largest gap on **`noise_0+estab_3`**: L6b plain **0.772** / rule **0.869** vs L6d plain **0.711** / rule **0.688** — rare conj still needs frequency-aware training or rule threshold. Area tasks (building, noise band) nearly tied.

**Takeaway:** CE+Dice without `f` is **close but not free** (~2–5 pts mean IoU at 2k steps); **keep dampened CE + decision rule** as the c3 default unless a later run shows Dice catching up with tuning (`ce_dice_weight`, steps).

Checkpoint: `runs/checkpoints/c2_L6d_ts2_2k.pt`. Harness: `c2_grid_loss`, `experiments.c2_l6d_ts2_2k()`.

**Not run (optional):** per-core-task **train alone** ablations; refreshed **composition baseline** with one-hot c1 checkpoint. **Infra polish:** B2 rule/oracle on c1 plot 4; B3 conjunction viewer in harness (gallery script covers portfolio visuals).

**Takeaways (defensible):**

1. **Binding:** condition tokens fix L5c’s layer/class entanglement; the model uses the **task prefix**, not only implicit AND (L6a vs L6b/L6c).
2. **Recipe:** same 2×384 ViT + one-hot + dampened CE + decision rule scales to **23 conjunctions + 12 singles** without architecture changes — **no hyperparameter rescue needed** for c2 closure.
3. **Generalisation:** atom reuse + held-out **pairs** (~0.85 rule mean) plus **wrong-description gap ~+0.84** on the same crops — composition claim is **defensible in screening**, not only qualitative gallery.
4. **Training data:** district-random origins exist (A4); L6 runs used **fixed 2k train crops** — enough for this ladder; district training is a scale lever for c3, not a c2 fix.

**Marvin’s call (22 Sep):** treat **c2 capability as done** for Track B; proceed toward **c3** without re-tuning the local encoder for c2. Optional doc/eval debt only.

### 3.14 What the diagnostics show about *how* c0 was (half) learned — `c0_worldsnap_P16_2k`, pre-perf run

Read from `runs/diagnostics/c0_worldsnap_P16_2k_plots.png` plus the raw rows in `c0_worldsnap_P16_2k.eval.jsonl` (21 Sep 2026). Learning curves are unaffected by the later perf work; only plot 7 of that figure is stale.

| Plot | Observation | Reading |
|---|---|---|
| 4 (decomposition) | `patch_hit` 0.1 → 0.95 between steps ~600 and ~800; `subcell|patch` at chance (1/16) for the whole run | one sharp phase transition for the coarse skill; the fine skill never starts |
| 3 (patch shared / resid) | marker **shared** grows all run (→ 1.8); marker **per-position** grows to ~1.08 by step ~600, then **flat** | sub-cell learning did not proceed slowly, it **stopped** when the patch-level solution was found. More steps on this recipe would not fix it |
| 2 (true update ratio) | `patch_embed/marker` is the lowest group throughout and ends at **−3.9**, about one decade below all other groups (−3.0 … −3.3) | the group that most needed to learn received the smallest relative updates: "starved group" signature |
| 6 (attention) | entropy: block 0 falls 5.5 → ~1.5 by step ~300; block 1 falls at step ~600–700, **simultaneously with the `patch_hit` jump**. `attn_to_marker` (eval rows): block 1 heads 0, 1, 2, 5 put **0.91–0.95 of all attention mass on the marker's token** at step 1999 (0.004 = uniform at init; ~0.5 at step 700); block 0 heads stay ≤ 0.02 | c0 needs no attention in principle (the answer is local to the marker's token), yet the model solved patch localisation with a **global circuit**: block 1 finds the marker token and every token reads from it. Spec §9 predicted "attention unused for c0": wrong. Relevant for c3 (relations to the marker), where exactly this broadcast is needed |
| 5 (grad norms) | total pre-clip norm 1.5–2.5, i.e. above `GRAD_CLIP = 1.0`, for the first ~600 steps; spikes up to ~60 right after the transition | every early step was clipped, so the effective learning rate was below nominal; the transition is a brief instability that clipping absorbed |
| 6b (residual RMS) | block 1 output RMS grows 1 → ~25 | harmless with the final LayerNorm; watch on longer runs |
| probe | GELU off-fraction 0 → 0.05 (block 0) / 0.22 (block 1) | normal sparsification |

**Takeaway:** the three curves (plots 2, 3, 4) give one consistent mechanism for the E1 result at P = 16: balanced CE is nearly satisfied by the full-patch blob, so after the transition the 256 per-position marker columns get almost no gradient and their growth ends. This supports closing c0 (see `LOCAL_EXPERIMENTS.md`) rather than training longer.

---

## 4. Methods lessons

1. **Unweighted CE** → all-background, low CE, exact_cell=0. Fixed with **`balanced_ce`** (default on).
2. **Overfit val pool mismatch** (different synthetic seed) → train CE→0, val broken. Use **`val_same_as_train=True`** for L1.
3. **Worldsnap data path:** layer cache removed npz reread; **GPU-resident crops** (`worldsnap_gpu_resident`) + **`torch.set_num_threads(4)`** are what unlock **~1900 samples/s** @ P=16 B=32 (see §14). Layer cache alone was **not** a 10× win.
4. **Early stopping:** use **`c0 snap @ step`** — P4 can look “flat” on CE until ~200 then collapses; P8/P16 can show **CE→0.02 while exact_cell=0**.
5. **Diagnostics plot 7:** `gpu_starvation = t_data / (t_data + t_step)` — high values mean the GPU waits on the loader; use to verify perf work before interpreting probe entropy.
6. **exact_cell alone is insufficient for c0** — requires exactly **one** fg argmax cell. Use decomposed metrics (harness since **code 1.1.1**):
   - `pred_fg_cells_mean` / `median`, `pred_fg_full_patch_frac`
   - `true_cell_recall`, `pred_fg_in_true_patch_mean`
   - `global_argmax_patch_hit`, `subcell_hit_given_patch`, `global_argmax_exact`
7. **Balanced CE incentivises full-patch fg blocks** when subcell discrimination is data-starved (~1/s² of the grid per in-patch column at patch size P; **s = 64/P**).

---

## 5. What we **can** draw (defensible claims)

- **Architecture + harness work** on synthetic and worldsnap data.
- **Real v2 npz + marker + grid targets are wired correctly** (worldsnap L1 overfit).
- **After ~1 epoch on 32k crops (P=16, balanced CE), patch localization on val** (`argmax_patch` ~0.97–0.98) **generalises**; **subcell|patch ~0.07–0.09** ≈ chance (1/16).
- **E1 (§3.6):** **P=4** solves full c0 on val (**exact_cell=1**, ~200+ steps). **P=8** solves patch + partial subcell (**global_argmax_exact ~0.46**) but **IoU=0.25** (2×2 blob). **P=16** = patch yes, product pointing no.
- **Mechanism confirmed:** ambiguity lives **inside the patch token**; not a train/val split artifact at P=16.
- **c0 + linear patchify + balanced CE** is a deliberate **measurement** of that interaction — honest for the portfolio.
- **c1 @ P=16 on v3b:** dense masks + task_emb; **one-hot** lifts scalar L5b mean IoU **0.91 → 0.97** (§3.13).
- **v2 estab failure** was target geometry + CE weight, not missing conditioning (§3.10–3.11; estab memo).
- **Weighted CE + analytic decision rule** tested offline on checkpoints (balancing memo); **c2** reports rule IoU in `eval_c2`; **c1 plot 4** still mostly 0.5 until B2.
- **E6:** flat `task_emb` (L5b); L5c factorisation **does not bind** — c2 **condition tokens** work (§3.15).
- **L5c hold-out:** partial on v3b; **c2 L6c** held-out conj mean **~0.85 rule IoU** with atoms seen elsewhere — **stronger** than L5c, **not** perfect on every pair (estab conj).
- **c2 AND across layers** on v3b @ P=16: gate **0.98**, mixture **0.95**, trained mean **~0.90**, held-out **~0.85**, wrong-description IoU **~0.01** (gap **~+0.84** vs correct on held-out crops) (§3.15).
- **T0 class readout not tested**; jsonl used only for crop + marker.

---

## 6. What we **cannot** draw (yet)

- That **P=16 + balanced CE + more epochs alone** will reach **exact_cell≈1** on full val — E1 says subcell stayed at chance+ after 2k; loss or architecture change still required.
- That **L1 overfit proves in-patch decode** on diverse data (only wiring + memorisation).
- That **P=4 success transfers** to default **P=16** production geometry without change — E1 explicitly separates these.
- **Track A vs one-hot local model** on sidewalk — local gap quantified (§3.13); full Track A not run.
- That **held-out c2 conjunctions** are solved at **~1.0 rule IoU** on every pair — **L6c mean ~0.85**; **`noise_1+estab_2`** weaker on correct description (still **+0.68** vs wrong prefix).
- **Other cities / districts** — de_pijp only (v2 + v3b cropsets).

---

## 7. Sanity ladder (`IMPLEMENTATION_PLAN` Phase D, revised)

| Rung | Experiment | Pass criterion | Status |
|------|------------|----------------|--------|
| **L1** | Overfit 32 samples | CE ~0; wiring | **Pass** synthetic + worldsnap (exact_cell high on 32 tiles) |
| **L2a** | c0 **patch** on val | `global_argmax_patch_hit` ≳ 0.95 | **Pass** @ `full_2k` (diagnosis ~0.97 val) |
| **L2b** | c0 **subcell** on val | `subcell_hit_given_patch` ≫ 1/16 | **Fail** @ 2k steps (~0.07 ≈ chance) |
| **L2** (product) | Trivial c0 on val | IoU ≈ 1 **and** exact_cell ≈ 1 | **Fail** (blocked on L2b + blob decode) |
| **L3** | E1 patch sweep | P=4 vs 8 vs 16 | **Done** §3.6 — mechanism confirmed |
| **L4** | E2 scalar vs one-hot | Matched steps | Not started |
| **L5a** | c1 building mask, no task input | IoU_pure ≥ 0.9 | **Pass** §3.9 |
| **L5b** | Four tasks + task_emb | Area tasks IoU ≥ 0.8 | **Pass** v2 §3.10 (food fail); **full pass v3b** §3.11 |
| **L5c** | Factorised 12 tasks, held-out pairs | Held-out ≫ wrong-task | **Partial** §3.12 (binding limit → c2 tokens) |
| **L6a** | c2 quiet sidewalk alone, no task input | Rule IoU ≥ 0.8 | **Pass** **0.948** §3.15 |
| **L6b** | `ts2`: 4 singles + 5 core, condition tokens | singles vs §4 gate; area conj ≥ 0.8 | **Pass** **0.953** mean; singles ≈ gate §3.15 |
| **L6c** | 23 conj, 4 held-out | correct ≫ wrong; vs AND baseline | **Pass** — held-out **0.852**; wrong-desc **0.012**, gap **+0.844** §3.15 |
| **L6d** | L6b with CE+Dice | vs dampened CE + rule | **Done** — L6b wins ~**0.936 vs 0.913** plain; keep CE+rule §3.15 |

---

## 8. Planned harness experiments (backlog)

Edit `experiments.py` → **`RUN`**. Spatial eval + `c0 snap` during train (`train_snap_batches`).

| ID | Intent | Notes |
|----|--------|--------|
| `c0_worldsnap_P{4,8,16}_2k` | E1 sweep | **Done** §3.6 |
| `c0_worldsnap_full_2k` | P=16 baseline | §3.5 (superseded by `P16_2k` harness metrics) |
| `c1_L5a_building_2k` / `c1_L5b_core_four_2k` | c1 ladder v2 | **Done** §3.9–3.10 |
| `c1_L5b_core_four_2k_v3b` / `c1_L5c_factorised_2k_v3b` | c1 v3b scalar | **Done** §3.11–3.12 |
| `c1_L5b_core_four_2k_v3b_onehot_P{16,8}` + scalar_P8 | C2_PLAN P2 | **Done** §3.13 |
| `c1_L5c_factorised_{2k,4k}` | L5c v2 discs | **Done** §3.12 (baseline) |
| **B2** | Rule/oracle in c1 `eval_spatial` + plot 4 | c2 done in `eval_c2.py`; [`BALANCING` memo](docs/memo/BALANCING_AND_DECISION_RULE_2026-09-21.md) |
| **D1–D2** | Condition tokens + 4-tuple batch | **Done** §3.15 |
| **L6a–L6c** | c2 conjunction ladder | **Done** §3.15 |
| **L6d** | CE+Dice vs dampened CE | **Done** §3.15 — dampened CE + rule retained for c3 |
| **L6c wrong-description** | Swap one cond token | **Done** §3.15 (`eval_c2.py`) |
| **Loss ablation @ P=16 (c0)** | Global 4096 softmax or lower fg weight | Optional vs E4 conv stem |
| **`c0_worldsnap_full_10k`** | Long P=16 | Low priority unless loss fixed first |
| L1 / synthetic re-eval | Blob metrics on old ckpts | Optional |

---

## 9. Experimental dimensions (E1–E9)

| ID | Axis | Priority (updated) |
|----|------|---------------------|
| **E1** | Patch P (4 / 8 / 16) | **Done** §3.6 — P4 pass, P8/P16 blob |
| **E2** | Scalar vs one-hot | **Decided** one-hot P=16 on v3b (§3.13); adopt in harness defaults for c2 |
| E3 | Per-layer tokens | Later |
| **E4** | Conv stem vs linear patchify | **Candidate** after E1 + optional loss ablation |
| E5 | Sin-cos temperature | Lower priority |
| **E6** | Added task emb vs extra token | **Added emb done** (L5b); token variant backlog |
| E7–E9 | Decoder, width, readout | T0 / scale |

---

## 10. Task rungs beyond c0

| Rung | Status |
|------|--------|
| **c1** L5a–L5c | **Done** on v3b (scalar + one-hot L5b §3.11–3.13) |
| **c2** L6 | **Closed for capability** (§3.15); optional B2/B3, L6d |
| **c3** | Crop-relative tasks — **next** Track B step; [`C3_PLAN.md`](docs/plans/C3_PLAN.md) |

c0 remains **closed for plumbing**. c1 + **c2** are **shippable** on v3b + one-hot + condition tokens for multi-layer AND + held-out conj evidence.

### 10a. c2 roadmap (from `C2_PLAN`, condensed)

**Task:** mark cells where **two layer conditions hold** (pixel-AND, then majority), e.g. **quiet sidewalk** (~7.7 % cells on v3b; p10–p90 per crop **1–17 %**).

**Work list (§6) — data / eval vs model:**

| Block | Item | Status |
|-------|------|--------|
| A1–A5 | Task defs, cached `f`, GPU loaders, district train, full val grid | **Done** (`c2_tasks.py`, `dataset_c2_gpu.py`) |
| B1 | Per-task rule/oracle eval + condition failure | **Done** (`eval_c2.py`) |
| B2–B3 | Plot 4 + viewer for conjunctions | Open |
| D1–D2 | **Condition tokens** (§4), `spatial_batch` 4-tuple | **Done** §3.15 |
| D3 | L6 experiment specs | **Done** |

**Encoding (§2, §5a):** one-hot P=16 — §3.13; set `encoding_mode=onehot` on c2 runs until harness default flips with D1.

**First mixture `ts2`:** five core conjunctions (§3) + singles **building, sidewalk, food & drink, `noise_2` (65–75 dB)** — not L5b's merged "≥ 65 dB" task.

**Conditioning (§4):** two **condition tokens** (13-way table + pad) + slot positions in front of 256 patches; head sees patches only. Gate: four singles through tokens ≈ one-hot L5b (~0.97).

**Loss / eval:** dampened CE + decision rule; **L6d** CE+Dice. Baselines: c1-mask AND (C2_PLAN §6); **wrong description** measured §3.15; composition vs §4 gate on singles.

**L6c:** hold out 4 of 23 conjunctions (§5c); all 250 val crops per held-out task.

Crop-relative phrasing → **c3** (`C3_PLAN.md`).

---

## 11. Engineering backlog

| Item | Why |
|------|-----|
| **Layer RAM cache** | Done — `layer_cache.py` |
| **GPU-resident worldsnap loader** | Done — `spatial_data/dataset_c0_gpu.py`, `worldsnap_gpu_resident` |
| **Vectorized spatial eval** | Done — `tests/test_eval_spatial_vectorized.py` |
| **Full tensor materialize (CPU)** | Optional `worldsnap_materialize_items` for small pools |
| **`torch.compile`** | ~16% @ B=32 after data fix; off for smokes (`PERF_BASELINE` §Step 6) |
| GEX131 long runs | When loss/arch fork chosen |
| **B2 rule/oracle in plot 4** | c2 scoring in `eval_c2`; c1 rows still 0.5-heavy |
| **D1 condition tokens** | C2_PLAN §4; third mode beside flat/factorised |
| **D2 4-tuple train/eval path** | `cond_ids` from c2 loaders already wired |
| Tier 3 offline probes (SV, linear) | Spec §7; checkpoint series ready |

---

## 12. Analysis suite & performance (reference)

**Analysis (interpret runs):**

| Tool | Path |
|------|------|
| Task viewer (c0/c1 panels A–G) | `python -m analysis.view_sample` — [`analysis/README.md`](analysis/README.md); c1: `--task-rung c1 --c1-task …` |
| Training diagnostics | `diagnostics_enabled=True`; `python -m diagnostics.plot_run RUN_ID` |
| Panel glossary | [`runs/diagnostics/VIEWER_PANELS.md`](runs/diagnostics/VIEWER_PANELS.md) |

**Performance (Sep 2026 plan implemented):** [`docs/plans/PERFORMANCE_PLAN_2026-09-21.md`](docs/plans/PERFORMANCE_PLAN_2026-09-21.md). Measured ladder: [`docs/plans/PERF_BASELINE_2026-09-21.md`](docs/plans/PERF_BASELINE_2026-09-21.md).

| Step | Change | P=16 B=32 on 3060 (indicative) |
|------|--------|--------------------------------|
| 0 | CPU loader, 6 threads | `next_batch` ~240 ms, ~160 samples/s |
| 1 | `cpu_num_threads=4` | ~25 ms batch, ~700 samples/s |
| 2 | `worldsnap_gpu_resident=True` | ~0.7 ms batch, **~1900 samples/s** |
| 4 | Vectorized `eval_spatial` | metric loop ~12 ms/batch vs ~70 ms |
| 5 | `eval_interval=250`, deduped val CE in logs | eval wall share **≪** pre-perf |

Worldsnap experiments in `experiments.py` default **`worldsnap_gpu_resident: True`** and **`eval_interval: 250`**.

---

## 13. How to update this doc

1. Run `python harness.py` (your console).
2. Copy spatial eval **second line** (`c0 detail: …`) and `runs/experiments.json` summary.
3. Update §3 / §7; bump **Last updated**.

**Re-eval old checkpoint without retraining:** load model + `data.load_data()` with matching overrides, call `eval_spatial.eval_model(model, max_batches=…)` (or run scratch diag with checkpoint id).

---

## 14. Open decisions

- **L2 product threshold:** exact_cell vs global-argmax vs L2a/L2b split for README claims.
- **Next intervention @ P=16:** c0 **loss** (global cell softmax / fg weight) vs **E4 conv stem** vs accept **P=4** for plumbing-only c0 (Marvin chooses).
- **Harness defaults:** flip `encoding_mode=onehot` globally vs only c2/`ts2` experiments.
- **§4 gate then L6a vs jump to L6b** (`ts2` mixture).
- **c2 loss:** dampened CE+rule first vs CE+Dice parallel (L6d).
- **Held-out four conjunctions** (C2_PLAN §5c) vs alternatives.
- **T0 readout** vs **c2 L6** on same one-hot encoder.
