# C1_PLAN — task-conditioned masks on the grid head

Written 21 Sep 2026 after closing c0. Status: **L5a/L5b implemented and run** (21 Sep 2026); L5c/E1-on-c1 open. See `LEARNING_REPORT.md` §3.9–3.10. It explains and specifies; the choices in §12 are Marvin's. Builds on `LOCAL_MODEL_GUIDE.md` (§6, §7c, §13), `LOCAL_EXPERIMENTS.md` (D6, E6, rung c1) and the c0 results in `LEARNING_REPORT.md` §3.5–3.8. Numbers in §3–§4 were measured on 400 training crops of `crops/v2` (script: `docs/plans/perf_prototypes/c1_target_stats.py`).

## 1. What c1 is, and what it is for

**Task:** given the layer stack and a **task id**, mark on the 64 × 64 grid every cell that satisfies the task, e.g. "mark sidewalk", "mark noise ≥ 65 dB", "mark food & drink". Same image, different request → different mask.

**What it tests that c0 could not:**
1. **Task conditioning**: a second input that changes what the same encoder and head produce.
2. **Reading the layers**: c0 only echoed the marker; here the model must decode class values from the input channels.
3. **The patch-size question under dense supervision.** c0 failed at P = 16 because a single-pixel target trained each of the 256 per-position columns on 1/256 of the samples. In c1 every pixel of every patch is labelled in every sample, so that starvation should disappear. Whether P = 16 then resolves 4 m cells is the open question c1 answers (§10).

## 2. What exists and what is missing

| Piece | Now | Needed for c1 |
|---|---|---|
| Model input | image only: `x = patch_embed(img) + pos_embed` | a task input and a way to merge it (§5) |
| Config | `task_rung = "c0"` is a label; the model never sees it | task list, conditioning mode |
| Targets | `c0_target_grid`: one cell from the marker | masks computed from the label planes (§4), on the GPU loader |
| Loss | balanced CE, weight `[1, n_bg/n_fg]` from the batch | breaks for mixed and empty targets (§7) |
| Eval | c0 decomposition (patch hit, sub-cell hit …) | IoU **per task**, pure vs mixed cells, empty-target handling (§9) |
| Viewer | c0 adapter | mask adapter with error map (§9) |

## 3. Candidate tasks, with measured statistics

Fraction of pixels (256 px frame) and of **grid cells** that are positive under two downsampling rules (§4), and share of crops in which the target is completely empty:

| Task | pixels | cells "any" | cells "majority" | any / majority | empty crops |
|---|---|---|---|---|---|
| surface = none | 40.6 % | 54.0 % | 41.1 % | 1.31 | 0 % |
| surface = roadway | 16.9 % | 25.1 % | 18.5 % | 1.36 | 0 % |
| surface = **sidewalk** | 14.7 % | 28.0 % | 16.2 % | **1.73** | 0 % |
| surface = **building** | 27.9 % | 36.9 % | 28.8 % | 1.28 | 0 % |
| noise < 55 dB | 37.4 % | 39.1 % | 37.6 % | 1.04 | 0 % |
| noise 55–65 | 27.4 % | 30.5 % | 27.7 % | 1.10 | 0 % |
| noise 65–75 | 32.3 % | 34.5 % | 32.6 % | 1.06 | 0 % |
| noise ≥ 75 dB | 2.9 % | 3.6 % | 2.9 % | 1.23 | **36.8 %** |
| **noise ≥ 65 dB** (classes 2+3) | 35.2 % | 36.6 % | 35.4 % | 1.03 | 0 % |
| estab = other named | 0.12 % | 0.43 % | 0.08 % | 5.3 | 2.2 % |
| estab = shop | 0.07 % | 0.26 % | 0.05 % | 5.4 | **37.5 %** |
| estab = **food & drink** | 0.10 % | 0.36 % | 0.07 % | 5.2 | 19.5 % |

Two facts that shape the design: **40 % of all 4 × 4 cells contain more than one surface class**, and an establishment disc (~12 px) almost never fills half a cell, so under "majority" establishment targets nearly vanish.

**Proposed core set (four tasks of different character):**

| id | Task | Character |
|---|---|---|
| 0 | mark **building** | common, large regions: the easy case |
| 1 | mark **sidewalk** | thin (2–4 px wide vs 4 px cells): the patch-resolution case |
| 2 | mark **noise ≥ 65 dB** | other layer, smooth large regions, a *threshold* over two classes |
| 3 | mark **food & drink** | rare, point-like: the imbalance case |

Extension (§5c): all 12 (layer, class) pairs with a factorised task embedding.

## 4. Target definition: from 256 × 256 labels to 64 × 64 cells

Each output cell covers 4 × 4 input pixels. Rules:

| Rule | Cell is positive if | Good for | Problem |
|---|---|---|---|
| **any** | ≥ 1 of 16 pixels has the class | point-like things (establishments) | inflates thin classes: sidewalk 16 % → 28 % of cells |
| **majority** | ≥ 8 of 16 pixels | area classes | erases establishments (0.36 % → 0.07 %) and thin sidewalks |
| **centre** | the centre pixel has the class | unbiased fractions | ignores 15 of 16 pixels; noisy at edges |
| **soft** | target = fraction of pixels (0…1), trained with BCE on soft labels | faithful everywhere | changes the loss and the metric (threshold at 0.5 for IoU) |

Proposal for the first version: **majority for surface and noise tasks, any for establishment tasks**; the rule is part of the task definition and is stored with it. Soft targets are the principled alternative and worth one comparison later. On the GPU loader all of these are a comparison plus `reshape(B, 64, 4, 64, 4)` and a sum: no CPU work.

Whatever rule is chosen: look at the targets in the viewer (`--no-model`) before training on them.

## 5. How the task enters the model (E6)

**(a) Added task embedding (proposed start).** `task_emb = nn.Embedding(n_tasks, n_embd)`; `x = patch_embed(img) + pos_embed + task_emb(task_id)[:, None, :]`; `forward(img, task_id, targets=None)` with `task_id` of shape `(B,)`. One vector per task, the same for every image, added to every token, so every token knows the request from layer one. Smallest possible change. Init like the other embeddings (std 0.02); it becomes a new semantic group `task_emb` in the diagnostics.

**(b) Extra token.** One learned vector per task, prepended: T = N + 1. Patch tokens must *read* it through attention. The grid head has to skip slot 0 and the position code needs an entry for it (zeros or a learned vector). More plumbing; generalises to several tokens (layer metadata, a text query). The E6 comparison, after (a) works.

**(c) Factorised embedding.** For tasks of the form "mark class k of layer L": `task_vec = layer_emb[L] + class_emb[k]` (3 + 4 vectors instead of 12). Enables the **held-out-combination test**: train on 10 of the 12 (layer, class) pairs, evaluate on the 2 unseen ones. Success means the model learned what "layer" and "class" mean separately, not 12 memorised behaviours. This is the first concrete step towards layers described by meta-information and swapped in and out. Threshold tasks such as "noise ≥ 65 dB" do not fit this form; keep them out of the factorised experiment or give them their own class vector.

Note on sentences: with exactly n distinct sentences for n tasks, any deterministic encoder yields n fixed vectors, so a trainable (or projected) sentence embedding behaves like the lookup table (a). Differences arise only from a *frozen* encoder's prior similarity structure and for *unseen* tasks; (c) probes the latter without a language model.

## 6. The marker channel in c1

c1 tasks do not use a marker. Proposal: **keep the channel, all zeros** (Cin stays 4 for scalar / 13 for one-hot), so checkpoints and the first layer's layout stay compatible with c3, which needs the marker again (e.g. "mark everything within 20 m of the marker"). Alternative: drop it and reintroduce later (cleaner input, incompatible checkpoints).

## 7. Loss: the current balanced CE does not carry over

Current: class weight `[1, n_bg / n_fg]` computed from the batch. Problems in c1:
1. **Empty targets**: `n_fg = 0` → division by zero. 20–37 % of crops are empty for the rare tasks.
2. **Mixed tasks in one batch**: a batch-level weight is dominated by the common tasks (building 29 % of cells) and leaves food & drink (0.36 %) under-weighted by two orders of magnitude.
3. The `.item()` sync (if still present) returns per step.

| Option | How | Comment |
|---|---|---|
| **Fixed per-task positive weight** (proposed) | table `w[task] = (1 − f_task) / f_task` from training statistics (§3), optionally clamped (e.g. ≤ 100); per-sample class weight `[1, w[task_id]]` applied through a per-cell weight map with `reduction='none'` | deterministic, no division by zero, no sync, interpretable; reference loss values stay fixed per task |
| Per-sample `n_bg/n_fg` with clamp | computed on the device per sample | adapts per image, but empty targets need a special case and the weight is noisy for rare tasks |
| Plain CE for area tasks, weighting only for rare ones | | simplest; area tasks at 16–35 % foreground do not need weighting |
| Dice / focal | | defer until weighting proves insufficient |

c0 lesson to carry over: a large positive weight makes false positives cheap and produced full-patch blobs. With `w ≈ 280` for food & drink expect generous blobs around establishments; the clamp is the knob.

## 8. Sampling

- Task id uniform over the task set per sample (every task gets equal training signal regardless of pixel share); crop uniform.
- Keep empty targets (the model must learn to output nothing), but log the share per task; if a rare task is empty in more than a third of samples, consider sampling crops for that task from those that contain it with probability ~0.8.
- Validation: fixed list of (crop, task) pairs with a fixed seed, identical across runs, stored in the run's meta file.

## 9. Evaluation, diagnostics, viewer

**Metrics, always per task, never pooled:**

| Metric | Definition | Why |
|---|---|---|
| IoU | on non-empty targets | main score |
| empty-target false-positive rate | share of empty-target samples where any cell is predicted; and mean predicted cells | IoU is undefined there |
| precision / recall | per task | separates "blobby" from "misses" (the c0 lesson) |
| **IoU on pure cells vs mixed cells** | pure = all 16 pixels of the cell share one class in the queried layer; mixed = otherwise | the c1 analogue of patch hit / sub-cell hit: pure cells test *reading*, mixed cells test *resolution inside the patch*. 40 % of surface cells are mixed |
| accuracy by position inside the patch | 4 × 4 heat-map of per-sub-cell accuracy | shows whether errors sit at patch borders |

**Diagnostics to watch (existing suite):** new group `task_emb`; `patch_shared` / `patch_resid` per channel, with the prediction that for surface tasks the *surface* channel's per-position part now grows (unlike the marker's in c0); attention entropy. c1 is local in principle (each cell depends on its own 4 × 4 pixels), so attention is not *needed*; after c0, where that expectation was wrong, record what it actually does.

**Viewer:** mask task adapter: target mask, predicted probability, **error map** (true positive / false positive / false negative in three colours), token view of a chosen patch, per-sample IoU. Use `--no-model` to inspect the four tasks' targets before the first training run.

## 10. Experiments, in order

| Rung | Setup | Question | Pass (proposal) |
|---|---|---|---|
| **L5a** | one task only ("mark building"), **no task input** | do dense targets, the GPU target builder, the loss and the mask metrics work at all? | val IoU ≥ 0.9 on pure cells |
| **L5b** | the four core tasks, added task embedding | does conditioning work; which task character is hard? | every area task val IoU ≥ 0.8; the same crop gives different masks for different task ids (viewer) |
| L5c | 12 factorised tasks, 2 combinations held out | compositional generalisation | held-out IoU clearly above a "wrong-task" baseline |
| E1 on c1 | P = 16 / 8 / 4 on L5b | is P = 16 good enough under dense supervision? read from pure-vs-mixed IoU and the sidewalk task | decision on patch size / conv stem for later rungs |
| E2 on c1 | scalar vs one-hot input | now meaningful: the model must decode class values (0, ⅓, ⅔, 1) from one channel vs read a binary plane | |
| E6 | added embedding vs extra token | | |

L5a first: it isolates "masks work" from "conditioning works", the same separation c0 gave for plumbing.

## 11. Predictions to check (so the runs teach something either way)

1. Building and noise ≥ 65 dB reach high IoU quickly at P = 16; errors concentrate on mixed cells.
2. Sidewalk is the hardest area task; its IoU gap between P = 16 and P = 8 is the largest.
3. Food & drink produces blobs (high recall, low precision) under a large positive weight.
4. The surface channel's `patch_resid` grows from the start (dense labels), unlike the marker's in c0.
5. `task_emb` has a high update ratio early and the task vectors separate quickly into distinct directions.
6. Attention entropy stays high (the task is local). Stated as a hypothesis: the same prediction was wrong for c0.

## 12. Decisions for Marvin

1. Task set: the four core tasks of §3, or a different selection.
2. Downsampling rule per task (§4): majority / any as proposed, or soft targets from the start.
3. Conditioning for the first run (§5): added embedding (proposed), extra token, or factorised directly.
4. Marker channel in c1 (§6): keep as zeros (proposed) or drop.
5. Loss (§7): fixed per-task weights (proposed) and the clamp value.
6. Whether L5a (single task, no conditioning) is run first.
7. Pass criteria in §10.

## 13. Code touch list (no code written)

| Where | Change | Side of the authorship line |
|---|---|---|
| `plain_gpt_module/local_config.py`, `local_grid_vit.py` | `n_tasks`, conditioning mode, `task_emb`, `forward(img, task_id, targets)`; loss per §7 | Marvin |
| `spatial_data/` (GPU loader) | task sampling, mask targets from the label planes per §4, fixed val pairs | collaborative |
| `harness/eval_spatial.py` | per-task metrics of §9, vectorised | collaborative |
| `harness/diagnostics/groups.py` | `task_emb` group | Marvin's module; one entry |
| `analysis/` | mask task adapter, error map | collaborative |
| `harness/config.py`, `experiments.py` | `task_rung = "c1"`, task list, weights table, experiments L5a / L5b | Marvin |
| `LOCAL_EXPERIMENTS.md` | record the decisions of §12 when taken | |
