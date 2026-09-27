# C4 implementation plan (working doc)

**Purpose:** Turn `C4_PLAN.md` §9 (segment → detour, no OSM first) into shippable stages Marvin and Claude can walk through one at a time.  
**Status:** 24 Sep 2026 — c3 closed; m1 recipe fixed in `docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md` §7.10 / chapter 10.  
**Canonical spec:** `docs/plans/C4_PLAN.md` (§9 overrides §1–§7 until c4a/c4b pass; OSM is **c4c**, later).

---

## 0. What we are building (one sentence per rung)

| Rung | Task id (proposed) | Target | Model must |
|------|-------------------|--------|------------|
| **c4a** | `segment` | Straight line between **two** markers | See both endpoints, ignore buildings |
| **c4b** | `detour` | Euclidean shortest path around **buildings** (visibility graph) | Same endpoints + read surface class 3 |
| **c4c** | (later) | Legal walk on BGT/OSM | Rule not in pixels |

Training recipe (from m1, **do not default to L4**):

- **2 blocks**, 6 heads, 384 wide; batch 32; **3 px disc** markers; **`rel_pos_bias=all`**; CE+Dice @ 0.5.
- **Uniform ⅓ mix** that always includes the **bare marker helper** (`within_20m`, **K = 2** on train).
- **16k steps**, **`lr_horizon_steps = 16000`**, diagnostics on for learning runs.
- Verify: **wrong-token** (when ≥2 tasks), **connectivity + detour strata**, pred strips with route overlay.

---

## 1. Current codebase map

| Piece | Location | State |
|-------|----------|--------|
| Route sample generator (P1) | `spatial_data/c4_routes.py` | **Code done**; full `npz` store **Marvin runs** on rig |
| c3 loader / markers K≥1 | `spatial_data/dataset_c3_gpu.py` | **Reuse** marker sampling + disc painting |
| Task registry | `spatial_data/c3_tasks.py` | **Extend** or add `c4_tasks.py` |
| Harness rung | `harness/config.py`, `harness/data.py` | **`task_rung=c4`** not wired yet |
| Eval | `harness/eval_spatial.py` | **c4 metrics** not wired (P3) |
| Experiments | `harness/experiments.py` | **No c4 builders yet** |

**No model change for c4a/c4b** if we keep two markers on the **existing single marker plane** (K = 2), same as c3 multimarker.

---

## 2. Phased implementation (order matters)

Each phase ends **shippable** (artifact + honest pass/fail). Do not start phase N+1 until the gate in phase N is green or explicitly waived.

### Phase A — Data on disk (P1, Marvin + Claude review)

**Goal:** Fixed train/val/test samples with known geometry.

1. **Smoke store** (already supported by CLI):
   ```bash
   cd ~/Github/spatial_reasoning_LLM_artifact
   python -m spatial_data.c4_routes --n-train 200 --n-val 50 --n-test 50 --overlays 8 \
     --out data/amsterdam/de_pijp/crops/v4/c4
   ```
2. **Inspect:** `c4_routes_{train,val,test}.npz`, `*_stats.json`, `c4_overlay_*.png`.
3. **Full store** (when smoke looks right):
   ```bash
   python -m spatial_data.c4_routes --n-train 20000 --n-val 500 --n-test 500 \
     --out data/amsterdam/de_pijp/crops/v4/c4
   ```

**Gate A:** Overlays show segment (dashed) vs path (solid); stats show detour strata populated; rejection rate logged and acceptable.

**Gate A status (24 Sep 2026):** **Blocked** — cyan dashed segment looks right; **red detour polyline is not acceptable** (route algorithm back to drawing board). Phase B loader can proceed; regenerate `c4_routes_*.npz` after P1 fix before training on detour.

**Decisions (confirm before full store):** endpoint distance 40–250 m, thin 1 m raster target, strata thirds — defaults in `c4_routes.py` match `C4_PLAN.md` §9.2 unless Marvin changes them.

---

### Phase B — Loader + task ids (P2)

**Goal:** One training batch from the c4 store through the existing ViT path.

1. **`spatial_data/c4_tasks.py`** (new, small):
   - `segment`, `detour` `C4TaskSpec`s: cond token ids, human names, tie to target field in npz.
   - Mix keys: `C4_MIX_L8B = ("within_20m", "segment", "detour")` (order matches weight vector).

2. **`spatial_data/dataset_c4_gpu.py`** (new):
   - Load planes from v4 district store (same as c3).
   - Index rows in `c4_routes_{split}.npz` (window origin, marker A/B, path pixels, detour ratio).
   - Paint **K = 2** markers (disc radius from config) on marker plane.
   - Build **64×64** target cells from path pixels (`any pixel` rule).
   - For `segment` task: target = segment cells (same endpoints, precomputed in store or from A/B line).
   - Val: **fixed** indices (full val/test npz), no random windows.

3. **Harness wiring:**
   - `config.task_rung = "c4"`, `c4_routes_dir`, `c4_task_mode`, `c4_task_keys`, weights.
   - `harness/data.py`: branch like c3 → `C4GpuTrainLoader` / batch fn.
   - `spatial_batch.forward`: no change expected if cond ids + shapes match c3.

4. **Smoke test (no training):**
   - Script or pytest: one batch, shapes, marker plane has 2 discs, target sparse, detour ratio in meta.

**Gate B:** Marvin runs smoke; viewer or saved PNG shows inputs + target for one val index.

---

### Phase C — Eval + shortcuts (P3)

**Goal:** Know *which* solution the model found before trusting IoU.

1. **`harness/eval_c4.py`** (or extend `eval_spatial.py` when `task_rung==c4`):
   - **IoU** (nonempty crops), **connectivity** (one component touching both markers),
   - **length ratio** (predicted polyline length / GT, if connected),
   - **Stratify** by stored **detour ratio** (1.0–1.05 / 1.05–1.3 / >1.3),
   - Baselines: **segment mask** IoU vs prediction (shortcut detector for detour task).

2. **Probe:** `docs/plans/perf_prototypes/c4_route_shortcut_probe.py`:
   - Compare pred to segment mask, detour mask, optional “straight line ignore buildings”.

3. **Diagnostics:** tier2 rows include connectivity + stratum; pred strips draw **markers + GT route + pred** on surface plane.

**Gate C:** Eval on a **random init** model runs; metrics finite; shortcut probe on one checkpoint path documented.

**Gate C status (24 Sep 2026):** **`harness/eval_c4.py`** wired in `eval_spatial` (IoU, connectivity, length ratio, detour strata, segment shortcut). Shortcut probe: `docs/plans/perf_prototypes/c4_route_shortcut_probe.py` (default ckpt `c4_loader_smoke`). Tier2 `eval.jsonl` gets `c4_*` flat keys; pred strips show surface + target + P(fg) when diagnostics on.

---

### Phase D — c4a segment (two-marker circuit)

**Goal:** Model draws the segment between two markers (ignores buildings).

| Run | Spec | Question |
|-----|------|----------|
| **D1** | `segment` **alone**, L2, 8k, 2 seeds | Does two-marker routing form at all? |
| **D2** | Mix **⅓** `{within_20m K=2, segment}` ×2 seeds, 8k | Same as m1: does helper task speed escape? |

**Pass (proposal, tune after first runs):** connectivity ≥ 0.9 on val; segment IoU ≥ 0.7; shortcut probe shows segment not detour.

**Gate D:** At least one seed passes; if not, stop and fix markers/attention before detour.

**Phase D wiring (24 Sep 2026):** `C4C3MixGpuTrainLoader` (`dataset_c4_mixed_gpu.py`), `c4_mix_c3_task_keys`, builders `c4_segment_L2_8k_seed` / `c4_mix20m_segment_L2_8k_seed`, queue `c4_phase_d_run_queue()`.

---

### Phase E — c4b detour (main task)

**Goal:** Shortest building-avoiding path; read detour strata.

| Run | Spec | Question |
|-----|------|----------|
| **E1** | Mix **⅓** `{within_20m K=2, segment, detour}`, L2, **16k**, 2 seeds | Full c4b recipe |
| **E1′ (proposed)** | Add **dense building mask** helper in mix (read surface class 3 — “mark all buildings”) to test whether the model learns to read the building plane before detour | Marvin, Sep 2026 |
| **E2** | Same, **L4** vs **L2** (optional) | Only if E1 fails on long detour stratum |

**Pass (proposal):** Overall IoU ≥ 0.6; **high detour stratum** (>1.3) clearly above segment shortcut; connectivity ≥ 0.85.

**Gate E:** Detour beats segment baseline on stratum 3; wrong-token / task-id steering if we add a third token in mix.

---

### Phase F — Hardening (after first pass)

- Wrong-token test for `{within_20m, segment, detour}` (extend `c3_wrong_token_probe.py` pattern).
- **Ceiling run:** 24k, same mix, if IoU still climbing at 16k.
- **c4c:** BGT mask variant in P1 generator + third task id — only after E passes.
- **OSM / mode tokens** (`C4_PLAN.md` §1–§7): separate milestone, not blocking c4b.

---

## 3. Experiment queue sketch (for `experiments.py`)

When phases B–C are done, add builders (names illustrative):

```text
c4_segment_smoke_2k          # 2k, diagnostics, 1 seed — plumbing only
c4_segment_L2_8k_seed{1,2}   # Phase D1
c4_mix20m_segment_L2_8k_s{1,2}  # Phase D2
c4_mix20m_seg_det_L2_16k_s{1,2}  # Phase E1 (reference)
```

Shared overrides helper: `_c4_m1_recipe_overrides()` mirroring report §7.10 (L2, b32, disc3, relV2, diagnostics, eval/1k).

---

## 4. Risks called out in the plan

| Risk | Mitigation |
|------|------------|
| **Segment shortcut** on detour (low detour crops) | Strata in eval; train mix includes segment task so model learns both; report IoU by stratum |
| **Thin target** (1 m line) | Same as m1/c0; keep “any pixel” cell rule; consider thicker band only if IoU ceiling stuck |
| **Two-marker attention** | Copy m1 mix pattern; K=2 within_20m in every mix; attn_to_marker on both marker tokens |
| **Courtyard rejects** in P1 | Already logged; if train pool too small, relax ellipse/cap slightly |
| **Plan doc says L4** in §9.1 | **Implementation uses L2** per m1 report chapter 10 unless a run fails |

---

## 5. Suggested walkthrough (interactive)

1. **You + Claude:** Phase A smoke overlays on rig (~5 min generate + look).
2. **Claude:** Phase B loader + config + unit smoke (you run one batch).
3. **Claude:** Phase C eval + probe stub (you run eval on init ckpt).
4. **Marvin:** Phase D1 one seed overnight if gates B–C green.
5. Review diagnostics → Phase D2/E1 queue.

---

## 6. Files to touch (checklist)

- [ ] `data/.../v4/c4/c4_routes_*.npz` (generated)
- [ ] `spatial_data/c4_tasks.py`
- [ ] `spatial_data/dataset_c4_gpu.py`
- [ ] `harness/config.py` (c4 keys)
- [ ] `harness/data.py`
- [ ] `harness/eval_c4.py` or `eval_spatial.py`
- [ ] `harness/diagnostics/tier2_eval.py` (c4 fields)
- [ ] `harness/experiments.py` (builders + RUN)
- [ ] `docs/plans/perf_prototypes/c4_route_shortcut_probe.py`
- [ ] `tests/test_c4_loader.py` (minimal)
- [ ] `C4_PLAN.md` §9.3 status table (update as phases complete)

---

## 7. Out of scope for first vertical slice

- OSM / osmnx graph (c4c legal)
- Mode tokens pedestrian/cycle/car
- Cost layers quiet/sunny (§4b)
- Marker **coordinate tokens** (SAM-style); stay on disc plane until segment fails
- Iterative / multi-step decoder (D4 in old plan)
