# PERFORMANCE_PLAN — making the local-model runs use the GPU (21 Sep 2026)

Status: **plan, nothing implemented.** Based on measurements on the RTX 3060 rig (RTX 3060, 6 CPU cores) against the harness as of commit `7b57a9d` plus Marvin's uncommitted harness edits of that day. Default config: `c0`, worldsnap `crops/v2` + `t0_point_v0`, batch 32, P = 16, scalar encoding (Cin = 4), bf16 autocast, eager mode, layer cache on.

## 1. Finding in one paragraph

The GPU is idle about **92 % of the time**. One training step costs ~16 ms of GPU work but ~180 ms (up to 435 ms in a second measurement) of CPU time to build the batch. The slow part is not compile and not the analysis maths: it is `next_batch`, and the reason it is slow is **CPU thread contention on tiny tensor operations**, made worse by a desktop app that pins one of the six cores. The same cost is what makes intermediate evaluation look expensive (38 % of run time at `eval_interval = 100`). Keeping the data on the GPU and building batches there was prototyped at **0.74 ms per batch**, which would move throughput from ~160 to ~1,900 samples/s (≈ 12×).

## 2. Measurements

### 2.1 One training step (batch 32, P = 16, eager)

| Part | Time | Note |
|---|---|---|
| GPU: forward + backward + grad clip + AdamW step | **16.0 ms** | = ~2,000 samples/s if data were free. Forward only: 5.4 ms |
| CPU: `train_loader.next_batch()` | **180 ms** (435 ms in a second run) | varies with whatever else the CPU is doing |
| Host → device copy | 4.2 ms | |
| **Observed throughput** | **~160 samples/s** | matches the 115–180 logged in the E1 sweep |

### 2.2 Why `next_batch` is slow

| Piece | Time |
|---|---|
| 32 × `index.sample()` measured in isolation | 20 ms |
| `torch.stack` of 32 images (4 × 256 × 256 float32) | 5 ms |
| `torch.stack` of 32 targets | 0.1 ms |
| → expected total | ~25 ms |
| `next_batch()` as actually measured | 180–435 ms |
| `marker_plane()` alone (one `torch.zeros` + one assignment) | **5 ms per sample** |
| `build_input_from_layers()` per sample | 10.7 ms |
| `scalar_layers_from_arrays()` per sample | 0.1 ms |

The work is tiny; the overhead is not. PyTorch parallelises each small CPU op over **6 intra-op threads on a 6-core machine**, while one core is permanently taken: `/usr/bin/resources` (the GNOME "Resources" system monitor, left open in a remote-desktop session of user `remote`, PID 228052 at the time) ran at **100 % CPU with 24 h of accumulated CPU time and 2.6 GB RAM**. The threads then fight for cores and every tiny op pays a synchronisation penalty.

Evidence that this is the cause: with the intra-op thread count limited, the identical `next_batch` took

| `torch.set_num_threads` | `next_batch` |
|---|---|
| 6 (default) | 180–435 ms |
| 4 | **23 ms** |
| 1 | 38 ms |

### 2.3 `torch.compile`

| | Eager | Compiled |
|---|---|---|
| GPU train step | 16.0 ms | 14.1 ms (**−12 %**) |
| Warm-up (first 3 steps) | 0.4 s | 34 s |
| Forward-only, eval mode, averaged over 30 calls | 5.4 ms | 340 ms (recompilation when switching train/eval mode) |

Compile is currently off (`use_compile = False`), and that is the right setting for now. Two blockers before it is worth enabling:
- `LocalGridViT.forward` computes the balanced-CE class weight as `(n_bg / n_fg).item()`. `.item()` forces a CPU–GPU synchronisation **on every step** (also in eager mode) and is a graph break for the compiler (dynamo warned at `local_grid_vit.py` line 87).
- Switching between train and eval mode triggered recompiles, so frequent evals cancel the gain.

With data at 180 ms per step a 2 ms gain is invisible; after the data fix the GPU step *is* the bottleneck and 12 % becomes real, though still modest: a 4 M-parameter model at batch 32 is limited by kernel-launch overhead, not arithmetic.

### 2.4 Intermediate evaluation

At `eval_interval = 100`, `eval_iters = 20`, spatial eval with 10 batches:

| Per eval | Time | Of which |
|---|---|---|
| `estimate_loss`: 20 train + 20 val batches, forward only | **8.85 s** | 40 × ~180 ms batch building ≈ 7.2 s; the forward passes are ~0.2 s |
| `eval_spatial.eval_loader`, 10 batches | **3.54 s** (20 batches: 5.26 s) | per batch 263 ms: ~185 ms batch building + forward, **~70 ms the per-sample Python metric loop** |
| 100 training steps in between | 20.1 s | |
| **Eval share of wall time** | **38 %** | consistent with the ~30 % observed in real runs |

So the analysis is expensive almost only because every evaluated batch pays the same slow batch building as training. The metric code's own cost is the per-sample loop in `eval_spatial.eval_loader`: about a dozen `.item()` calls per sample (each a GPU sync), ~70 ms per batch. There is also a duplicate: `estimate_loss` makes a validation pass and the spatial eval makes another one that already returns the loss.

Projection once data is fast (§3.2) and nothing else changes: 100 training steps ≈ 1.7 s, `estimate_loss` ≈ 0.2 s, spatial eval ≈ 10 × 70 ms ≈ 0.7–1.5 s, so the *un-vectorised metric loop would then dominate* (eval share ~40–50 %). Vectorising it (§3.4) brings an eval to roughly 0.1–0.2 s, i.e. a few percent.

**Answer to "is the analysis worth its time": yes; it is not what costs the time.** Keep all of it, fix the loader, vectorise the loop, and evaluate less often once runs are fast.

## 3. Actions, in order of value

### 3.1 Limit CPU threads, free the pinned core (one line, do now)
- Add `torch.set_num_threads(4)` at harness start-up (e.g. in `data.setup_device()`), before any tensors are built.
- Close the "Resources" monitor in the remote-desktop session (or kill PID of `/usr/bin/resources`; it belongs to user `remote`, so it was left untouched).
- Expected: `next_batch` 180 → ~23 ms, throughput ~160 → ~700–800 samples/s, eval share drops accordingly. Zero risk. Worth doing even if §3.2 follows, because synthetic pools and any remaining CPU code benefit too.

### 3.2 GPU-resident data, batches built on the GPU (the real fix)
- All 2,000 training crops as uint8 class planes: `(2000, 3, 256, 256)` = **393 MB** of VRAM (val 250 crops: 49 MB). Item tables (crop index, marker row, marker col per jsonl line) as three small integer tensors on the GPU.
- A batch is then pure indexing on the device: sample item indices with `torch.randint`, gather the crops, scale (`/ 3.0` for scalar encoding) or `F.one_hot` for one-hot encoding, write the marker plane and the target grid by advanced indexing. No Python loop over samples, no host → device copy.
- **Prototype, measured: 0.74 ms per batch of 32** (vs 180 ms), values checked against the harness loader (max 1.0 in both). Code: function `gpu_batch` at the end of `docs/plans/perf_prototypes/profile_batch.py` (copy of `~/.cache/claude_scratch/profile_batch.py` on the rig).
- Expected: ~1,900 samples/s at batch 32, P = 16 (GPU-bound from then on). A 2,000-step run: ~4 min → ~20 s.
- Shape of the change: a drop-in loader class exposing `.B` and `.next_batch()` like `C0BatchLoader`, selected by a config flag (e.g. `worldsnap_gpu_resident = True`), so the existing CPU path stays available for comparison and for machines without enough VRAM. For mask tasks (c1+) the targets are computed on the GPU from the same class planes (`planes == class_id`, downsampled to the grid), which removes target building from the CPU as well.
- Equivalence test to write with it: for a fixed list of item indices, the GPU loader and the CPU loader must return identical tensors.
- VRAM note: check that nothing else holds the card (on 20 Sep an idle `minimal_rewriter` web UI held 10.3 GB).

### 3.3 Remove the per-step `.item()` from the model's forward
- Pass the class weight to `F.cross_entropy` as a tensor computed on the device (`torch.stack([one, n_bg / n_fg])`), or as a constant from the config for c0 (always `[1, 4095]` on a 64 × 64 grid with one foreground cell).
- Removes one forced sync per training step and the graph break that blocks `torch.compile`. Small gain alone; prerequisite for §3.6.

### 3.4 Vectorise the metric loop in `eval_spatial.eval_loader`
- Every quantity in the `for b in range(B)` loop is a batched tensor expression: IoU per grid from `(pred_fg & tgt_fg).sum((1, 2))` and `(pred_fg | tgt_fg).sum((1, 2))`; `n_fg = pred_fg.sum((1, 2))`; exact-cell = `(n_fg == 1) & (pred_fg.flatten(1).float().argmax(1) == tidx)`; recall via `pred.flatten(1).gather(1, tidx[:, None])`; patch hit from `am // G // s == tidx // G // s` (rows) and the same for columns; share of predicted foreground inside the true patch via a block mask built from `tidx`.
- Accumulate sums as tensors on the device and call `.item()` once per eval, not per sample.
- Expected: ~70 ms → ~1–2 ms per batch.

### 3.5 Evaluation schedule
- Drop the validation half of `estimate_loss` when a spatial eval runs at the same step (it already returns the validation loss), or fold train-loss estimation into the same routine.
- With fast runs, evaluate by wall-clock budget rather than every 100 steps (e.g. every 250–500 steps), keeping the final eval at full size (`eval_iters_final`).
- Target: evaluation below ~5 % of run time.

### 3.6 Then, and only then: batch size and compile
- **Batch size** is probably the larger lever: a 4 M-parameter model at batch 32 barely loads a 3060. Try 128 and 256 (not measured yet); scale the learning rate or the step budget accordingly and note it in the ledger, since it changes "steps" as a unit.
- **`torch.compile`**: re-measure after §3.2–3.4. Expect ~10–15 % on the GPU step at P = 16. Keep it off for short smoke runs (34 s warm-up) and avoid frequent train/eval mode switches in compiled runs, or compile only the training path.
- **P = 4 is a different regime**: 4,096 tokens, attention-bound at ~30–35 samples/s. Faster data does not help there; larger batch does not fit easily either. This is a cost argument for P = 8/16 or a conv stem, not a loader problem.

## 4. Expected effect (P = 16, batch 32)

| State | Train throughput | 2,000-step run | Eval share |
|---|---|---|---|
| Now | ~160 samples/s | ~4 min | ~38 % |
| After §3.1 | ~700–800 samples/s (estimate from 23 ms batches) | ~1.5 min | ~25–30 % |
| After §3.2 + §3.3 | ~1,900 samples/s | ~35 s | metric loop dominates evals |
| After §3.4 + §3.5 | ~1,900 samples/s | ~35 s | < 5 % |
| After §3.6 | to be measured | | |

The first row and the GPU-step and GPU-batch numbers are measured; the rest are projections from those measurements.

## 5. Caveats

- CPU timings varied by more than 2× between two runs minutes apart (180 vs 435 ms) because they depend on other load on the machine. The GPU-step time and the GPU-side batch time were stable.
- All numbers are for c0 with scalar encoding. One-hot (Cin = 13) makes CPU batch building more expensive still and the GPU step slightly more; on the GPU path the extra cost is one `F.one_hot`.
- Measurements used the harness modules directly with `config.data_source = "worldsnap"`; the profiling scripts write nothing into the repository.

## 6. Reference: profiling scripts and prototype

Copied into the repo so the reference survives a cache clean-up; originals in `~/.cache/claude_scratch/` on the RTX 3060 rig.

| File | What it does |
|---|---|
| `docs/plans/perf_prototypes/profile_harness.py` | Times one training step split into batch building / copy / GPU step, eager and compiled; times `estimate_loss` and the spatial eval; prints the eval share per interval |
| `docs/plans/perf_prototypes/profile_batch.py` | Splits `next_batch` into its pieces, repeats it with 1 and 4 threads, and contains the **GPU-resident batch prototype** (`gpu_batch`, last 20 lines) with a value check against the harness loader |
| `~/.cache/claude_scratch/diag_c0.py` | (earlier) checkpoint diagnostic that produced the patch-hit / sub-cell numbers |

Run from `harness/`: `../.venv/bin/python ../docs/plans/perf_prototypes/profile_harness.py`. Re-running it after each action in §3 is the way to confirm the gain.

## 7. Ownership

§3.1, §3.3, §3.5, §3.6 are small edits in Marvin's harness and model files. §3.2 (loader) and §3.4 (metric vectorisation) are data/eval plumbing on the collaborative side of the authorship split; Claude offered to write §3.2 as a drop-in class for review, or Marvin builds it from the prototype. Open as of 21 Sep.
