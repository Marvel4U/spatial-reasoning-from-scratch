# Track A Qwen: vanilla baselines and LoRA SFT (Oct 2026)

Status: results report (Marvin). Implements **§2 (vanilla)** and **§3 (fine-tuned)** of
[`docs/plans/TRACK_A_QWEN_PLAN_2026-10-04.md`](../plans/TRACK_A_QWEN_PLAN_2026-10-04.md). Shared scorer and exports:
`spatial_data/track_a_export.py`; inference helpers: `qwen_reference/`; training: `qwen_fine_tuning/`.

## 1. Question and success rule

**Task:** From a top-down map PNG (512×512, De Pijp, green start / red goal), produce a walking polyline as JSON
in Qwen's **0–1000 grounding grid**, converted to image pixels for scoring.

**Eval prompt (locked for SFT and reported FT numbers):** single-shot **`route_v1`** (`qwen_reference/prompts.py`) —
same as plan §2.3 output format. Vanilla A/B also ran **two-pass** (marker probe + anchored route) and **oracle**
(GT endpoints in prompt); those are diagnostic, not the FT success criterion.

**Scorer (polyline rule):** start/end within **6 m** of markers; no building cross (0.5 m sampling on 1 m raster);
length ≤ **1.10×** teacher walking length. Failure reasons: unparseable, wrong endpoints, crosses a building, too
long. Straight-line join of markers on De Pijp val scores **27.6%** (floor); teacher **100%** (ceiling).

**Comparison note:** Our wave search (C5) uses a **cell connectivity** rule on a scratchpad (~**78%** De Pijp val 500
after own-drawings fine-tune). Wave has **not** yet been rescored with this polyline rule for a single table (plan §5,
still open).

---

## 2. Vanilla Qwen (zero-shot / inference only)

**Models:** `Qwen3-VL-2B-Instruct` (primary), optional `Qwen3-VL-4B-Instruct` on RTX 3060 when VRAM free.

**Code:** `qwen_reference/` — load, `route_v1`, coordinate 0–1000 ↔ pixels (`coords.py`), `eval_ab.py`, overlays.

### 2.1 De Pijp val — first 200 routes (A/B, Oct 4)

| Model | Mode | Solved | Failure mix (share of 200) |
|-------|------|-------:|----------------------------|
| **2B** | two-pass | **2.5%** (5/200) | wrong endpoints **92%**, crosses **2%**, too long 3.5% |
| **2B** | oracle | **8.0%** (16/200) | wrong endpoints **79%**, crosses 11%, too long 2% |
| **4B** | two-pass | **24.5%** (49/200) | endpoints 13.5%, **crosses 51.5%**, too long 10.5% |
| **4B** | oracle | **27.0%** (54/200) | endpoints 4.5%, **crosses 54.5%**, too long 14% |

Artifacts: `runs/qwen_vanilla/eval_ab_n200.json`, `eval_ab_4b_n200.json`, preds under `runs/qwen_vanilla/pred_*`.

### 2.2 What vanilla taught us

1. **2B is not a route finder out of the box** — it mostly fails on **endpoint / marker grounding** (~90%+ wrong
   endpoints on two-pass). Output is often parseable JSON; the model does not reliably map discs to the 0–1000 grid.

2. **4B is a different regime** — ~**10×** two-pass solve rate vs 2B on the same 200 routes. Failures shift to
   **geometry**: ~**half** of errors are **chord through buildings** (straight-ish polylines with few vertices). Oracle
   helps endpoints but **does not** fix crosses — the model "sees" the task but not obstacle-aware planning.

3. **Matches plan expectations** (§2): straight/detour split; zero-shot near the **straight-line floor (~28%)** on
   average only for **4B**, not 2B. Task needs **spatial computation + format**, not recognition alone.

4. **Coordinate convention** — 0–1000 grid aligned to image; marker probe works; free-form chat drifts. All training
   and eval use one conversion path in `coords.py`.

5. **Two-pass vs single-shot** — For 2B, two-pass did not beat oracle much; for FT we **standardized on `route_v1`**
   single-shot at eval (decision Oct 2026).

---

## 3. Fine-tuned Qwen (LoRA SFT)

**Goal (plan §3):** Data-scaling curve on De Pijp train → same val scorer; own training loop before `trl`.

### 3.1 Method (implemented)

| Piece | Location | Notes |
|-------|----------|--------|
| Label smoke test | `qwen_fine_tuning/train_batch_smoke.py` | One example, masked CE, decode label span |
| SFT encode + batch | `qwen_fine_tuning/sft_batch.py` | `-100` on user/vision tokens; pad batches |
| Examples | `qwen_fine_tuning/examples.py` | `route_v1` + teacher JSON in 0–1000 |
| Train loop | `qwen_fine_tuning/train_lora.py` | **Main file** for training |
| Eval | `qwen_fine_tuning/eval_lora.py` | Load adapter, `route_v1` generate, score |
| Load adapter | `qwen_fine_tuning/load_lora.py` | Base + PEFT sidecar |

**LoRA:** `q_proj`, `k_proj`, `v_proj`, `o_proj` only; **r=16**, **α=32**, dropout 0.05; ~**0.3%** trainable params.
Vision frozen.

**Optim:** AdamW **lr=2e-4**, cosine + 5% warmup, **batch 2 × grad accum 2** (effective 4), **3 epochs**, bf16,
gradient checkpointing. **RTX 3060**, the project venv (`VENV.md`).

**Data:** `data/track_a/de_pijp_train20k_plain/routes.jsonl`. Rows with **`leaves_window: true` excluded** (~671 of
20k). Maximum usable pool **19 329** routes; "20k" run uses **19 309 train + 20 dev** (first contiguous usable rows).

**Checkpoints:** `--checkpoint-every 500` optimizer steps → `checkpoints/step_XXXXX/` + `training_state.json` /
`training_state.pt` (adapter + AdamW + scheduler). **`--resume`** verified (abort at step 500, continued to 505+).

### 3.2 Training runs (chronological)

| Run dir | Train N | Opt steps | Notes |
|---------|--------:|----------:|--------|
| `runs/qwen_sft/lora100_v1` | 100 | 75 | Pipeline proof; dev-20 **5%** |
| `runs/qwen_sft/lora200_v1` | 200 | 150 | Overnight chain |
| `runs/qwen_sft/lora500_v1` | 500 | 375 | |
| `runs/qwen_sft/lora1000_v1` | 1000 | 750 | |
| `runs/qwen_sft/lora5000_v1` | 5000 | 3750 | Plan **F1** |
| `runs/qwen_sft/lora20000_v1` | 19 309 | 14 483 | Plan **F2** (full usable train) |

Logs: `train_log.jsonl`, `dev_eval.jsonl`, `run_config.json` per run. Shell recipes: `runs/qwen_sft/*.sh`,
`TIMING_ESTIMATE.md`.

### 3.3 Reported numbers — De Pijp val

**Primary metric:** **`route_v1`**, **`eval_lora`**, **`n=500`** (`de_pijp_val_plain`).

| Checkpoint | Val 500 solved | 2–3 goals | 4–5 | 6+ | Top failures (share) |
|------------|---------------:|----------:|----:|---:|----------------------|
| 2B vanilla (two-pass, n=200) | 2.5%† | 5.6% | 1.5% | 0% | wrong endpoints 92% |
| 4B vanilla (two-pass, n=200) | 24.5%† | 35% | 35% | 1.6% | crosses 52% |
| LoRA **1k** | **21.4%** | 58% | 28% | **1.2%** | endpoints 39%, crosses 36% |
| LoRA **5k** | **31.0%** | 74% | 38% | **7.7%** | crosses 47%, endpoints 20% |
| LoRA **~19.3k** | **49.6%** | **81%** | **62%** | **28.3%** | crosses 36%, endpoints 11% |

†Vanilla on **200** routes only (not full 500). JSON: `runs/qwen_sft/lora*/eval_val_v1_n500.json` (FT);
`runs/qwen_vanilla/eval_ab_*.json` (vanilla).

**Val 200 vs 500:** Do not compare across slice sizes. Example: **1k** was **37.5% on val 200** but **21.4% on val 500**
— the first 200 val routes are easier (fewer long / 6+ goals).

**Dev-20 (held-out train routes, generate):** after full train, **60%** (12/20); 5k dev ~35%. Useful for monitoring,
not the headline.

**Loss:** Training CE falls (~1.7 → ~0.4 over full run); **generate + scorer** is the metric that matters.

### 3.4 What fine-tuning taught us

1. **SFT works** — Monotonic improvement with data on val 500: ~**+10 pts** per ~2× data in the 1k→5k→full band;
   **~50%** at full train vs **~2.5%** vanilla 2B.

2. **Data beats width (here)** — Full-train **2B+LoRA (~50% val 500)** beats **4B vanilla (~27% on val 200)** for
   this prompt and scorer.

3. **Hard routes need data** — **6+ goals:** 1.2% (1k) → 7.7% (5k) → **28.3%** (full) on val 500. Easy bands saturate
   first (2–3 → **81%**).

4. **Failure mode shift** — Unparseable → **0%**. Endpoints drop (39% → 11% failure share). **Building crosses**
   remain the largest error bucket at scale (~36–47%) — same qualitative failure as 4B zero-shot, but at much higher
   solve rate.

5. **Engineering** — Smoke test before LoRA; sidecar adapters; resume for ~15k-step overnight on 3060 (~5–18 h
   depending on N); eval separated from HF training loop. **`trl` not yet run** (plan step 4).

6. **Honest gaps** — No F3–F6 (vision LoRA, layers render, 1024 px, transfer). No GRPO yet. Wave not on polyline
   scorer in one table.

---

## 4. Scaling curve (val 500, route_v1)

```
solved_share
  50% |                                    *  (~19.3k LoRA)
  40% |
  30% |                         *
  20% |              *    * (1k)     (~27% 4B vanilla, 200 only)
  10% |
   0% | * (2B vanilla ~2.5%, 200 only)
      +-----|-----|-----|-----|-----|
           1k    5k   10k   15k  ~19k  train routes
```

Straight-line baseline on full val: **27.6%** (plan §1).

---

## 5. Artifact index

| What | Where |
|------|--------|
| Plan (original scope) | `docs/plans/TRACK_A_QWEN_PLAN_2026-10-04.md` |
| Vanilla preds / A/B | `runs/qwen_vanilla/` |
| SFT runs | `runs/qwen_sft/lora*_v1/` |
| Example overlays | `runs/qwen_vanilla/solved_4b_*`, `fail_4b_*` |
| Wave baseline (different rule) | `docs/reports/C5_WAVE_SEARCH_2026-10-03.md` |

---

## 6. Next steps (from plan, not done)

- **F3–F5** ablations; **F6** transfer evals.
- **Wave + c4 detour** scored with `track_a_export` for one comparison table.
- **`trl` SFTTrainer** parity pass.
- Optional **GRPO** once policy failures are mostly scorer-verifiable (crosses / length).
