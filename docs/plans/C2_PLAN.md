# C2_PLAN — conjunctions across layers ("mark quiet sidewalk")

Restructured 22 Sep 2026 (first version 21 Sep). **Read top to bottom: goal → what is decided → how the task is fed in → background → work list → experiments.** The work list (§6) and the experiments (§7) are the plan; everything above them is context.

## 1. Goal

Mark every cell where **two conditions from different layers hold at once**, e.g. "sidewalk **and** noise below 55 dB". Same model, same 64 × 64 output grid as c1. All conditions are **absolute** (fixed classes); crop-relative conditions ("the quieter sidewalk of this crop") are c3, see `C3_PLAN.md`.

What c2 adds over c1: the model must read two channels and combine them in one cell (an AND), the task description has parts, and target size varies a lot between crops.

## 2. Decided

| | Decision |
|---|---|
| Input encoding | **one-hot** (13 channels incl. the unused marker plane). Result that decided it: §5a |
| Patch size | **P = 16** (fast default; P = 8 only as a quality check) |
| Data | cropset **`v3b`**, four noise bands; **no new crops needed**: conjunction targets are computed on the GPU from the existing label planes |
| More training data | training crops are cut **on the fly at random positions from the district label stack** (10 MB, kept on the GPU), never touching the held-out blocks: effectively unlimited distinct crops, same held-out ground. Validation stays the fixed 250 crops |
| Target rule | a pixel is positive if both conditions hold there; a cell is positive if ≥ 8 of its 16 pixels are ("pixel-AND, then majority") |
| Task set | cumulative: c2 tasks are trained **together with the c1 tasks**; task sets are named and versioned; every task's validation items are fixed |
| Which conjunctions | only the usable ones (§5b): 23 of 48. First run: the five core tasks of §3 |
| Loss | balanced cross-entropy with dampened weights `w = ((1 − f)/f)^0.5`, `f` measured from the training crops; decision rule `P(fg) > w/(1+w)` (memo `docs/memo/BALANCING_AND_DECISION_RULE_2026-09-21.md`) |
| Evaluation | per task, never pooled: score under the decision rule, oracle score, precision, recall, empty-target false alarms, pure vs mixed cells, and which of the two conditions failed |
| Harness default encoding | **`encoding_mode` stays `scalar` in `config.py` until D1 lands**; all c2 runs and the §4 gate must set **`onehot`** in experiment overrides (flip global default with D1 if desired) |

## 3. Tasks of the first run

| Task | Conditions | Share of cells | Character |
|---|---|---|---|
| quiet sidewalk | noise < 55 dB ∧ sidewalk | 7.7 % | the product-vision task; thin class × smooth field |
| quiet building | noise < 55 dB ∧ building | 21.8 % | large, easy regions |
| loud roadway | noise 65–75 dB ∧ roadway | 7.1 % | empty in 14 % of crops |
| unoccupied building | building ∧ no establishment | 27.4 % | a negative condition |
| quiet food & drink | noise < 55 dB ∧ food & drink | 0.76 % | rare, empty in 32 % of crops |

Plus four single-condition tasks in the same training mixture — the **`ts2`** singles in `spatial_data/c2_tasks.py`: **building**, **sidewalk**, **food & drink**, and **`noise_2` (65–75 dB only)**. That last one is **not** the same as c1 L5b's "noise ≥ 65 dB" (which spans two noise classes); compare L6b regression against the **one-hot token-input gate** (§4) on these four keys, not against L5b's `noise_ge_65` task.

## 4. How the task is given to the model (decided 22 Sep: condition tokens)

**Decision (Marvin): the task is a short token sequence, handled the way a language model handles its input.** One token per condition (a (layer, class) pair: 12 tokens, plus one "no condition" token), an ordinary embedding table, a short learned position embedding, and the task tokens placed in front of the 256 patch tokens. The encoder is unchanged; the grid head reads only the patch positions. One condition = a c1 task, two = a c2 conjunction. Word-level tokens ("noise", "low") are not needed now; they would also just be tokens.

Why this and not the c1 way (one vector added to every patch): a conjunction has two parts and each class must stay tied to its layer. A summed vector of separate "layer" and "class" parts cannot do that ("noise 0 ∧ surface 2" and "noise 2 ∧ surface 0" give the same sum). With one token per whole condition the problem does not arise.

Implementation notes for D1 (Marvin's code; suggestions, not a spec):
- **Fixed length of two task tokens.** The loader gives `cond_ids` of shape (B, 2) with −1 where there is no second condition. Shift by +1 so that 0 = the "no condition" token and 1…12 = the conditions: `nn.Embedding(13, n_embd)`. With a fixed length no attention mask is needed; T = 258.
- **Positions:** a learned `nn.Embedding(2, n_embd)` for the two task slots. The task tokens get **no** 2-D sine-cosine code; the patch tokens keep theirs unchanged. The loader always emits conditions in layer order (noise, surface, establishment), so slot order carries no extra meaning.
- **Forward:** `x = cat([cond_emb(cond_ids + 1) + slot_pos, patch_embed(img) + pos_2d], dim=1)` → encoder blocks → `ln_enc` → **drop the first two positions** → grid head. The head and the target grid stay exactly as they are.
- Keep it as a third conditioning mode next to `flat` and `factorised`, so the c1 checkpoints still load.
- The per-task loss weights are still looked up by `task_index`; both `task_index` and `cond_ids` come with every batch.
- **First check before any c2 run:** the four c1 single-condition tasks through the token input should reproduce the c1 one-hot result (mean ≈ 0.97). If they do, the input path is right and everything new in c2 is the conjunction itself.
- Worth adding to the probe diagnostics: the share of attention the patch tokens put on the two task slots. With the added vector every patch knew the task at once; now the patches must fetch it in block 1 and use it in block 2, and this number shows whether they do.

## 5. Background (results and measurements the decisions rest on)

### 5a. Encoding vs patch size on c1 (four core tasks, `v3b`, 2,000 steps, all 250 val crops per task)

| Input, patch | building | sidewalk | noise ≥ 65 | food & drink | mean | mixed cells | train time |
|---|---|---|---|---|---|---|---|
| scalar, P = 16 | 0.971 | 0.853 | 0.989 | 0.807 | 0.905 | 0.83 | 32 s |
| **one-hot, P = 16** | 0.991 | **0.977** | 0.995 | 0.929 | **0.973** | 0.94 | 40 s |
| scalar, P = 8 | 0.984 | 0.878 | 0.997 | 0.929 | 0.947 | 0.91 | 91 s |
| one-hot, P = 8 | 1.000 | 1.000 | 0.999 | 0.985 | 0.996 | 0.99 | 99 s |

Sidewalk was limited by the scalar encoding (its grey value sits between roadway and building, so mixtures imitate it), not by the patch size. With one-hot the over-marking largely disappears too. Side note for the two-track comparison: grey-level input costs the local model 0.12 on sidewalk, and it is the only form the Qwen track can receive.

### 5b. Which conjunctions exist (2,000 `v3b` training crops; script `perf_prototypes/c2_target_stats.py`)

- **noise × surface: 13 of 16 usable**, 2–22 % of cells. Excluded: ≥ 75 dB ∧ none / sidewalk / building (almost always empty).
- **noise × establishment: 9 usable but rare**, 0.4–1.4 % of cells, empty in 16–41 % of crops. Excluded: ≥ 75 dB ∧ any establishment (never occurs); any noise ∧ "no establishment" (same as the noise task alone).
- **surface × establishment: degenerate by construction** on `v3b` (establishments are painted on building footprints, so "building ∧ food & drink" is just "food & drink"). Only "building ∧ no establishment" is a real task.
- **Target size depends on the crop:** quiet sidewalk covers 1.0 % of cells in a sparse crop (10th percentile) and 16.9 % in a dense one (90th). This is why §7 compares against a loss that needs no frequencies.
- Composing two c1 masks ("call c1 twice and AND them") differs from the true target on 0.3–6 % of positive cells for the usable tasks: it is a baseline (§7), not the definition.

### 5c. The generalisation test c1 could not deliver

L5c (12 factorised tasks, 2 held out) showed no composition, but could not have: 12 evaluation pairs, one class seen with only one layer, only two working layers. c2 has enough combinations to do it properly: hold out 4 of the 23 conjunctions whose conditions each appear in ≥ 3 trained conjunctions and as single-condition tasks (`< 55 ∧ roadway`, `55–65 ∧ building`, `65–75 ∧ sidewalk`, `55–65 ∧ shop`), evaluate on all 250 val crops, and report correct description vs wrong description vs the composition baseline.

## 6. Work list (what has to exist before the experiments)

Owner "C" = Claude (data / eval side, Marvin reviews), "M" = Marvin (model and training code).

| # | Item | Owner | Status |
|---|---|---|---|
| A1 | Task definitions: conditions, tasks of 1 or 2 conditions, named task sets (`c2_core`, cumulative `ts2`, `c2_full`, held-out list), usability filter | C | **done** 22 Sep: `spatial_data/c2_tasks.py` |
| A2 | Positive share `f` per task **computed from the training crops** and cached; dampened loss weights from it | C | **done** 22 Sep: `c2_tasks.py`, cache `crops/v3b/c2_task_stats.json` |
| A3 | GPU batch builder: one-hot input, conjunction targets, returns image, target, task index **and the condition ids** (so either input variant of §4 can use it) | C | **done** 22 Sep: `spatial_data/dataset_c2_gpu.py` (2.8 ms per batch of 16) |
| A4 | Training crops cut on the fly from the district stack, never touching held-out blocks | C | **done** 22 Sep: `C2DistrictSampler`: 1,094,136 admissible origins |
| A5 | Fixed validation grid: all 250 val crops × every task | C | **done** 22 Sep: `C2GpuValLoader` |
| B1 | Evaluation per task: rule score, oracle score, precision / recall, empty-target false alarms, pure vs mixed cells, **which condition failed** | C | **done** 22 Sep: `harness/eval_c2.py` ( **`task_rung=c2` only** — c1 `eval_spatial` / plot 4 still mostly plain 0.5 until B2 ) |
| B2 | Per-task rule + oracle lines in eval rows and plot 4 (c1 and c2; held-out tasks dashed) | C, in Marvin's diagnostics module after review | open |
| B3 | Viewer adapter for conjunctions: target, prediction, error map, the two part masks | C | open |
| C1 | Composition baseline script ("call c1 twice") | C | **done** 22 Sep: `perf_prototypes/c2_composition_baseline.py` |
| C2 | Tests: GPU targets equal a plain reference; random crops never touch held-out ground; validation is deterministic; decision-rule threshold | C | **done** 22 Sep: `tests/test_c2_tasks.py`, 21 tests pass |
| D1 | **Task input of the model** per §4: condition tokens in front of the patch tokens; grid head reads only patch positions; first check = c1 tasks through the token input | M | **done** 22 Sep |
| D2 | Training loop passes the extra batch element: `spatial_batch` 4-tuple + `cond_tokens` in config; train/eval/diagnostics wired | M | **done** 22 Sep |
| D3 | Experiment specs for §7 in `experiments.py` (gate, L6a–d) | C | **done** 22 Sep (L6d: `grid_loss=ce_dice`) |

**Composition baseline measured (22 Sep)** with the scalar c1 model `c1_L5c_factorised_2k_v3b` (AND of two c1 masks under the decision rule; in brackets the ceiling set by the definitional mismatch): quiet sidewalk 0.66 (0.97), quiet building 0.91 (0.99), loud roadway 0.79 (0.99), unoccupied building 0.92 (0.98), quiet food & drink 0.62 (0.89). These are the numbers a trained c2 model has to beat; a one-hot single-condition checkpoint would give a stronger baseline and does not exist yet.

## 7. Experiments, in order

| Run | Setup | Question | Pass (proposal) |
|---|---|---|---|
| **L6a** | quiet sidewalk alone, no task input | can the model AND two channels at all at P = 16? | rule IoU ≥ 0.8; errors mainly from the sidewalk condition |
| **L6b** | cumulative set **`ts2`** (§3): four singles + five c2 core, condition tokens (§4) | conjunctions in a mixture; do the singles keep their scores? | each **`ts2` single** within **0.01 rule IoU** of the §4 gate run on that key (reference: one-hot P=16 v3b, token input); area conjunctions ≥ 0.8 |
| L6c | all 23 usable conjunctions + single conditions, 4 conjunctions held out | does the model compose conditions it never saw together? | held-out "correct" clearly above "wrong", near the composition baseline |
| L6d | L6b with cross-entropy + overlap (Dice) loss, cut-off 0.5 | a loss that needs no per-task frequencies | per-task comparison decides the default for c3 |

Controls alongside: each core task trained alone (is a low score the task or the mixture?); the composition baseline; wrong-description control for L6c.
