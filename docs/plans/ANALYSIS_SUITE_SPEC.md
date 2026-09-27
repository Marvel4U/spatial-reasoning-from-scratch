# ANALYSIS_SUITE_SPEC — understanding how training is progressing

Written 21 Sep 2026 for Marvin to implement when suitable. **Canonical path:** `docs/plans/ANALYSIS_SUITE_SPEC.md` (moved from repo root, 21 Sep 2026). Self-contained design doc: scope is the local grid ViT (`plain_gpt_module/local_grid_vit.py`) trained through `harness/`. It specifies *what to record, how to compute it, how to plot it, and what each plot is for*. §12 has verified snippets; **how to run what exists** is in `analysis/README.md` and `harness/README.md`.

---

## 0. Implementation status (21 Sep 2026, the RTX 3060 rig)

Turn on training diagnostics: `diagnostics_enabled = True` in `harness/config.py` or experiment overrides. Default is **off** so normal runs stay light.

| Spec | Deliverable | Code | Tested on rig |
|---|---|---|---|
| §5 Tier 1 JSONL | `{run_id}.jsonl`, `{run_id}.meta.json` | `harness/diagnostics/tier1.py`, `groups.py` | Yes — synthetic + worldsnap timing; 13 semantic groups; true AdamW `update_ratio/*`; `patch_shared` / `patch_resid`; `gpu_starvation`, `samples_per_s` |
| §6 (a) c0 eval metrics | Keys in `{run_id}.eval.jsonl` | `harness/eval_spatial.py` (`flatten_for_eval_jsonl`) | Yes — wired at CE eval via `tier2_eval.py` |
| §6 (b) probe + hooks | Same eval rows + `probe_sample_ids` in meta | `probe_batch.py`, `probe_metrics.py` | Yes — smoke `smoke_tier2_probe`: init attn entropy ≈ 5.54/head; `attn_to_marker/*`, `gelu_off/*`, `head_logit_std` |
| §6 (c) pred PNGs | `{run_id}/pred_step{step:06d}.png` | `pred_strip.py` | Yes — 8 samples (configurable); smoke with 2 pred rows |
| §6 (d) checkpoint series | `runs/checkpoints/{run_id}/step{N}.pt` weights-only | `checkpoint_series.py` | Yes — `step0.pt` at init (pre–first update); log-spaced list in config |
| §9 plots 1–7 | `{run_id}_plots.png`, compare PNG | `harness/diagnostics/plot_run.py` | Yes — plot 6 uses **mean entropy per block** (per-head values remain in JSONL) |
| §13.1 static task viewer | `python -m analysis.view_sample` | `analysis/` package | Yes — P16 ckpt ~164 ms CUDA forward; panel guide `runs/diagnostics/VIEWER_PANELS.md` |
| §13.2 interactive viewer | — | — | **Not started** |
| §7 Tier 3 offline | linear probes, SV spectra, error maps, … | — | **Not started** (needs checkpoint series — now available) |

**Wiring:** `maybe_run_diagnostics(run_id)` in `harness/train.py` and `harness/harness.py`; tier1 every `diagnostics_interval` steps; tier2 on CE eval (optional subsample `diagnostics_eval_interval`); checkpoints after `opt.step()` (except step 0 saved before the loop).

**Known limitations (documented, not blockers for c0 @ P=16):**

- Tier 2 `record_ce_eval` runs a **second** val pass for spatial metrics (plus probe forward). **c0/c1:** capped at `eval_iters` loader batches (default 20). **c2:** always the **full** crop×task grid (`diagnostics_val_max_batches` → `None`) because the val loader is task-major — partial passes zero out later tasks in per-task IoU. On the 3060, full c2 eval is ~10–15 s per CE eval row; c0 partial pass was ~1 min with probe batch 64.
- **P = 4** (4096 tokens): shrink `diagnostics_probe_batch_size` manually (§10.3); no auto row-chunking yet.
- **Synthetic** probe indices are val items `0 … n_probe−1`; **worldsnap** uses `(diagnostics_probe_seed * 997 + i * 17) % n_val`.
- **`torch.compile` + hooks:** probe hooks register on the unwrapped module; re-verify probe fields if training uses compile.
- **`plot_run` plot 6:** dashboard aggregates heads per block; use JSONL directly for per-head traces.

**Smoke run reference:** `run_id=smoke_tier2_probe` under `runs/diagnostics/` and `runs/checkpoints/smoke_tier2_probe/` (25 steps, synthetic, small probe batch).

---

## 1. Why, and the principle behind the design

On 21 Sep a full-pool c0 run had train/val CE ≈ 0.02 but exact-cell = 0. It took an ad-hoc script to find the cause: the model located the right **patch** 97 % of the time and guessed the **sub-cell inside the patch** at chance (7 % vs 6.25 %), because the marker channel of the patch projection had learned its shared component ("a marker is in this patch", norm 1.82) long before its 256 per-position components (norm 1.05). Loss curves could not show this. A whole-tensor weight histogram could not show it either: the problem lived *inside one matrix*.

Design principle that follows: **every recorded quantity and every plot must answer a named question** (§9), and parameters are grouped by *meaning*, not by tensor (§4). A suite of thirty unread plots is the usual failure; aim for six plots plus prediction images.

## 2. The four classic training diagnostics (what they are)

These come from the standard MLP-debugging toolkit (popularised in Karpathy's "makemore part 3" lecture). For a deep tanh MLP trained with plain SGD one records, on one batch:

1. **Activation distribution per layer**: histogram of each layer's outputs, with mean, std, and **% saturated** (|tanh| > 0.97). Healthy: similar std in every layer, a few % saturated. Shrinking std with depth = signal dies; heavy saturation = gradients die (tanh' ≈ 0).
2. **Gradient distribution per layer** (gradient w.r.t. each layer's output): healthy = similar std across layers. A std that shrinks or grows geometrically with depth = vanishing / exploding gradients.
3. **Weight-gradient distribution per parameter**, with the **grad:data ratio** `grad.std() / weight.std()`: how large the gradient is relative to the weight it will change.
4. **Update-to-data ratio over time**: per parameter and step, `log10( (lr · grad).std() / weight.std() )`, plotted against step with a reference line at **−3**. Rule of thumb: ≈ 10⁻³ is healthy; far below = this parameter is barely training; far above = updates are thrashing the weights. It is the most useful of the four because it is a *time series per parameter*: it shows which parts learn, when.

Diagnostics 1–3 are snapshots (one batch, typically at init and after a little training); 4 is continuous.

## 3. What has to change for this model

| Classic | Problem here | Replacement |
|---|---|---|
| 1. tanh saturation | Pre-LN transformer with GELU: nothing saturates in that sense; LayerNorm fixes the scale entering each sublayer | **Residual-stream RMS after each block** (does the stream grow with depth / over training?), **GELU off-fraction** (share of `mlp.c_fc` outputs < −3, where GELU ≈ 0 and its gradient ≈ 0: dead units), **attention entropy per head** (§6) |
| 2. activation gradients | Carries over, but per-layer output gradients need `retain_grad` | Use **gradient norm per parameter group** instead: same information, no graph changes |
| 3. grad:data ratio | Fine as a snapshot | Keep, per semantic group |
| 4. `lr · grad` update ratio | **Wrong under AdamW.** Adam's step is `lr · m̂ / (√v̂ + ε)`, not `lr · grad`; on the first step it is ±lr for every weight regardless of gradient size. Measured on this model, first step, lr 6e-4: true ratio 10^−1.5 for every group, `lr·grad` formula 10^−3.8 … 10^−4.7, i.e. wrong by 2–3 orders of magnitude and, worse, it *ranks the groups differently* | **Measure the real update**: `(p_after − p_before).std() / p_before.std()` (§5). Weight decay is then included automatically. The −3 line is an SGD heuristic; under AdamW expect roughly −3 … −2 in steady state, higher during warm-up's end. Treat −3 as a reference, not a target; compare groups *to each other* |
| per-tensor grouping | Hides structure inside a matrix (see §1) | **Semantic groups** (§4) |

Two things the classic set does not contain and this project needs: **task-decomposed metrics over time** (§6) and **time accounting** (§5), since runs have been I/O-bound at ~125 samples/s.

## 4. Semantic parameter groups

Only weight matrices (`p.ndim >= 2`); biases and LayerNorm parameters are skipped. Real parameter names of the current model:

| Group | Parameter | Slice |
|---|---|---|
| `patch_embed/<channel>` (one per input channel: noise, surface, estab, marker; with one-hot: one per class plane, or aggregated per layer) | `patch_embed.proj.weight` (384, Cin·P·P) | `W.view(n_embd, Cin, P*P)[:, c]` → (384, 256). Valid because `PatchEmbed` flattens a patch in the order (channel, in_row, in_col) |
| `encoder.{i}.attn.c_attn`, `encoder.{i}.attn.c_proj`, `encoder.{i}.mlp.c_fc`, `encoder.{i}.mlp.c_proj` | as named | whole tensor |
| `head.proj` | `head.proj.weight` (s·s·K, 384) | whole tensor; optionally split by sub-cell later |

For each `patch_embed/<channel>` group additionally record the **shared / per-position decomposition**: with `cols = W[:, c]` of shape (384, 256), `shared = cols.mean(dim=1)` and `resid = cols − shared[:, None]`; log `shared.norm()` and `resid.norm(dim=0).mean()`. "Shared" means *this channel is present somewhere in the patch*; "per-position" means *where in the patch*. This pair is the single most informative weight statistic found so far.

`.view` on a parameter shares storage, and the same view applied to `p.grad` gives the matching gradient slice, so grouping costs nothing.

The group list depends on module names and on the input encoding. Generate it from `model.named_parameters()` plus a channel-name list from the data config; do not hard-code indices.

## 5. Tier 1 — always on, every N steps (N ≈ 10–50), appended to JSONL

One JSON object per recorded step: `runs/diagnostics/{run_id}.jsonl` (same layout as the existing `train_diagnostics.py`).

| Field | Computation | Notes |
|---|---|---|
| `step`, `loss`, `lr` | as now | |
| `grad_norm_total` | value returned by `clip_grad_norm_` | already computed in the train step; shows how often clipping bites |
| `grad_norm/{group}` | `view(p.grad).norm()` | after backward, before clip |
| `param_rms/{group}` | `view(p).pow(2).mean().sqrt()` | slow drift = weight decay vs growth |
| `update_ratio/{group}` | `log10( (after − before).std() / before.std() )` | needs a copy of the group's weights *before* `optimizer.step()`; only on recorded steps. 4 M params → a 16 MB clone every N steps: negligible |
| `patch_shared/{channel}`, `patch_resid/{channel}` | §4 | |
| `t_data`, `t_step` | wall time waiting for `next_batch()` vs the rest of the step | `t_data / (t_data + t_step)` is the **GPU-starvation fraction**. Call `torch.cuda.synchronize()` before reading the clock on recorded steps only |
| `samples_per_s` | as now | |

Cost: a few small reductions per group on recorded steps; one parameter clone. No effect on unrecorded steps. With `torch.compile`, read parameters through `unwrap_compiled(model)`.

## 6. Tier 2 — at every eval

**(a) Task-decomposed metrics.** One score hides which sub-skill is missing. For c0 (all from one forward pass over the val batches; `score = logits[..., 1] − logits[..., 0]`):

| Metric | Definition | Reads as |
|---|---|---|
| `n_fg_mean`, `n_fg_median`, `frac_n_fg_eq_{s·s}` | predicted foreground cells per grid; share of grids with exactly one patch-block worth | blob size; = s·s means "whole patch lit" |
| `recall` | true cell is predicted fg | |
| `argmax_hit` | global argmax of `score` = true cell | "knows where", independent of threshold |
| `patch_hit` | global argmax lies in the true patch (`cell // s` equal in row and col) | coarse localisation |
| `subcell_hit_given_patch` | `argmax_hit` among `patch_hit` cases; chance = 1/(s·s) | fine localisation, the part that was at chance |
| `argmax_dist` | Euclidean distance argmax ↔ target in cells, mean and median | |
| existing `IoU_fg`, `exact_cell` | keep | |

Rule for later rungs: whenever a task is added, split its metric along the sub-skills the task needs (for T0: accuracy binned by `boundary_dist_px` and by class; for c1: IoU per class and per layer; for c3: IoU binned by region size).

**(b) Fixed probe batch.** 64 validation samples chosen once per run with a fixed seed and kept on the GPU. Forward them with hooks (§12) and record:

| Field | Computation |
|---|---|
| `resid_rms/{block}` | RMS of each `EncoderBlock` output |
| `gelu_off/{block}` | share of `mlp.c_fc` outputs < −3 |
| `attn_entropy/{block}/{head}` | mean over queries of `−Σ_j a_ij log a_ij`. Maximum is ln T (5.55 for 256 tokens) = uniform attention; near 0 = every query looks at one token. `F.scaled_dot_product_attention` does not return weights, so recompute them for the probe batch from the hooked attention input and the module's own `c_attn` (§12). Measured at init: 5.54 for all heads, i.e. attention starts uniform |
| `attn_to_marker/{block}/{head}` (c0/c3 only) | mean attention mass that all queries put on the marker's patch token; tells whether attention is used to broadcast the marker |
| `head_logit_std` | std of `score` over the grid |

**(c) Prediction images.** For 8 fixed probe samples save one PNG per eval: input marker position, target grid, `softmax` foreground probability map, side by side, with a consistent colour scale. `runs/diagnostics/{run_id}/pred_step{step:06d}.png`. Watching the blob sharpen (or not) is worth more than any scalar.

**(d) Intermediate checkpoints.** Save at log-spaced steps (e.g. 0, 100, 200, 500, 1k, 2k, 5k, 10k, …) in addition to the final one. Without them Tier 3 has nothing to compare over time. 4 M params ≈ 16 MB each without optimizer state; store weights only for the intermediate ones.

## 7. Tier 3 — offline, from checkpoints

Scripts or a notebook; run when a question comes up, not on every run.

- **Weight structure over time**: the §4 decomposition across the checkpoint series; singular-value spectrum of each `patch_embed/<channel>` slice (how many directions are in use).
- **Linear probes**: freeze a checkpoint, take (i) the patch embedding output and (ii) the final encoder output for the marker's token, fit a logistic regression to predict the marker's in-patch position (256-way, or row and column separately). High probe accuracy with low task accuracy = the information is present but the head does not use it; low probe accuracy = it never made it into the representation. This separates "representation problem" from "read-out problem", which no training-time scalar can.
- **Position-code similarity map**: cosine similarity of one token's position vector against all others, reshaped to the grid. Fixed for sin-cos (a useful picture for choosing the temperature); informative once a learned code is tried (E5): a 2-D blob means the model has found the geometry.
- **Attention maps** for single samples: which tokens the marker's token attends to and which attend to it, per head, on the grid.
- **Error maps**: accuracy as a function of the marker's position in the image and *in the patch* (a 16×16 heat-map of sub-cell accuracy shows whether errors are uniform or e.g. worse at patch borders).

## 8. Files and layout

```
runs/diagnostics/{run_id}.jsonl            Tier 1, one object per recorded step
runs/diagnostics/{run_id}.eval.jsonl       Tier 2 (a)+(b), one object per eval
runs/diagnostics/{run_id}/pred_step*.png   Tier 2 (c)
runs/checkpoints/{run_id}/step{N}.pt       Tier 2 (d), weights only
runs/diagnostics/{run_id}.meta.json        group list, channel names, probe sample ids, config snapshot
```

Flat key names with `/` separators (`update_ratio/patch_embed/marker`) so plotting code can select by prefix. Keep `experiments.json` as the ledger of final results; diagnostics are separate and disposable.

## 9. The plots, each with its question

One figure per run, regenerated from the JSONL files (`cd harness && python -m diagnostics.plot_run RUN_ID`; linear step on x by default, `--log-x-step` for 100k+ runs), plus a compare mode that overlays two runs.

| # | Plot | Question it answers | Healthy | Warning |
|---|---|---|---|---|
| 1 | Loss (train, val) and lr vs step (linear step axis) | Is it training at all; when do phase changes happen | | long plateau at ln 2 (balanced CE, untrained) or at the "trivial solution" value |
| 2 | `update_ratio/{group}` vs step, all groups, reference line −3 | Which parts of the network are learning, and when | groups within about one decade of each other | one group 1–2 decades below the rest = starved; a group above −1.5 after warm-up = thrashing (lower lr or check clipping) |
| 3 | `patch_shared` and `patch_resid` per channel vs step | Is the input layer learning *that* a channel is present, *where* in the patch, or both | both grow for channels the task needs | shared grows, resid flat (the c0 failure) |
| 4 | Task-decomposed metrics vs step | Which sub-skill is learned when | sub-skills rise one after another | one metric at chance while loss looks converged |
| 5 | `grad_norm_total` with the clip threshold, and `grad_norm/{group}` | Is clipping active all the time; does any group get no gradient | clipping occasional | always clipped (lr effectively set by the clip); a group with ~0 gradient |
| 6 | `attn_entropy` per head and `resid_rms` per block vs step | Is attention doing anything; is the residual stream stable | entropy drops below ln T for some heads once the task needs cross-patch information | all heads stay at ln T: attention unused (expected for c0, which needs none; a finding in its own right for c2/c3) |
| 7 | `t_data / (t_data + t_step)` and samples/s | Is the GPU waiting for data | < 10 % | > 50 %: fix the loader before anything else |
| — | Prediction image strip across evals | What is the model actually drawing | | |

## 10. Concerns

1. **Do not port the `lr·grad` formula.** Under AdamW it is off by orders of magnitude and mis-ranks groups (§3). Measure the real update.
2. **Per-tensor statistics are not enough** for the first layer of this model; the semantic slicing in §4 is the point of the suite, not an extra.
3. **Attention weights are not available from flash attention.** Recompute them on the probe batch only (64 × 6 heads × 256² floats ≈ 100 MB at P = 16). At P = 4 (4,096 tokens) that is 64 × 6 × 4096² ≈ 25 GB: reduce the probe batch to 2–4 samples or compute entropy row-chunked.
4. **Hooks and `torch.compile`**: register hooks on the unwrapped model and run the probe forward on it; hooks inside a compiled graph cause recompiles or are skipped.
5. **Timing needs `torch.cuda.synchronize()`**, otherwise GPU work is attributed to whichever later call blocks. Synchronise only on recorded steps.
6. **bf16**: compute diagnostic reductions in float32 (`.float()` before `std`/`norm`); small stds underflow in bf16.
7. **Balanced-CE loss values are not comparable to plain CE.** Put the reference values on plot 1: ln 2 ≈ 0.69 untrained, ≈ 0.5·6.9 ≈ 3.45 for "all background" with weights [1, n_bg/n_fg]. If the loss or its weights change, the reference lines change.
8. **Probe-set leakage is harmless but fixedness matters**: the 64 probe samples must be identical across runs (store ids in `meta.json`), otherwise run-to-run comparisons of Tier 2 are noise.
9. **Volume**: ~40 scalars per record. At N = 10 and 100k steps that is 10k lines ≈ 10 MB. Fine. Do not log histograms in Tier 1; take them offline from checkpoints.
10. **Scope creep**: every new quantity must name the question it answers in §9's table before it is added.

## 11. Recommended order

1. **Timing** (`t_data`, `t_step`) — **done** (`harness/diagnostics/tier1.py`).
2. **Tier 2 (a) c0 metrics** in `eval_spatial.py` — **done**.
3. **Tier 1 groups + true update ratio + patch shared/resid** — **done**; plots 1–3 and 5 via `plot_run.py`.
4. **Prediction images** and **intermediate checkpoints** — **done** (`pred_strip.py`, `checkpoint_series.py`).
5. **Probe batch with hooks** (plot 6) — **done** (`probe_batch.py`, `probe_metrics.py`; plot 6 in `plot_run.py`).
6. Tier 3 pieces on demand; the linear probe first — **not started**.

Historical note: the first draft pointed at `harness_from_translator/train_diagnostics.py`; the live implementation is under `harness/diagnostics/` and is wired into `train.py` / `harness.py`.

## 12. Verified snippets

Run against `LocalGridViT(LocalGridViTConfig())` on the rig, 21 Sep. Illustrations, not a module — production code mirrors this in `harness/diagnostics/`.

```python
# --- semantic groups (§4) -------------------------------------------------
CHANNELS = ["noise", "surface", "estab", "marker"]          # from the data config

def groups(model):
    out = {}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            continue
        if name == "patch_embed.proj.weight":
            for c, cn in enumerate(CHANNELS):
                out[f"patch_embed/{cn}"] = (p, c)
        else:
            out[name.replace(".weight", "")] = (p, None)
    return out

def view(t, c, cfg):                                        # works for p and for p.grad
    return t if c is None else t.view(t.shape[0], cfg.in_chans, cfg.patch_size ** 2)[:, c]

# --- true update ratio (§5), on a recorded step ----------------------------
before = {k: view(p.detach(), c, cfg).float().clone() for k, (p, c) in G.items()}
optimizer.step()
for k, (p, c) in G.items():
    now = view(p.detach(), c, cfg).float()
    rec[f"update_ratio/{k}"] = ((now - before[k]).std() / before[k].std()).log10().item()

# --- shared vs per-position (§4) -------------------------------------------
W = model.patch_embed.proj.weight.detach().float().view(cfg.n_embd, cfg.in_chans, cfg.patch_size ** 2)
cols = W[:, c]; shared = cols.mean(dim=1, keepdim=True)
rec[f"patch_shared/{cn}"] = shared.norm().item()
rec[f"patch_resid/{cn}"]  = (cols - shared).norm(dim=0).mean().item()

# --- probe batch with hooks (§6b); model = the UNWRAPPED module -------------
acts = {}
def save_out(name):
    def f(mod, inp, out): acts[name] = out.detach()
    return f
def save_in(name):
    def f(mod, inp): acts[name] = inp[0].detach()
    return f
handles = []
for i, blk in enumerate(model.encoder):
    handles += [blk.register_forward_hook(save_out(f"resid/{i}")),
                blk.mlp.c_fc.register_forward_hook(save_out(f"mlp_pre/{i}")),
                blk.attn.register_forward_pre_hook(save_in(f"attn_in/{i}"))]
with torch.no_grad():
    model(probe_img)
for h in handles:
    h.remove()

for i, blk in enumerate(model.encoder):
    x = acts[f"attn_in/{i}"].float(); B, T, C = x.shape; nh = blk.attn.n_head
    q, k, _ = blk.attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T, nh, C // nh).transpose(1, 2)
    k = k.view(B, T, nh, C // nh).transpose(1, 2)
    att = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(C // nh), dim=-1)      # (B, nh, T, T)
    entropy = -(att * att.clamp_min(1e-12).log()).sum(-1).mean(dim=(0, 2))     # (nh,)  max = ln T
    rec[f"resid_rms/{i}"] = acts[f"resid/{i}"].float().pow(2).mean().sqrt().item()
    rec[f"gelu_off/{i}"]  = (acts[f"mlp_pre/{i}"] < -3).float().mean().item()
```

## 13. Task viewer — seeing one task the way the model sees it (added 21 Sep)

**Purpose.** Build intuition that tables cannot: what a single token receives at a given patch size, what the model is asked to draw, and what it actually draws. First use: make the patch-size trade-off visible (E1: P = 4 solves c0, P = 8 half, P = 16 only the patch). Later: every new task rung gets looked at here *before* it is trained on, which is also the guard against unanswerable tasks (the failure that ended the predecessor project).

**Usage:** `analysis/README.md`. **Panel guide:** `runs/diagnostics/VIEWER_PANELS.md`.

Scale reminder for reading the pictures: the 256 px input is 1 px = 1 m. A sidewalk is 2–4 px wide, an establishment disc 4 px, a P = 16 token covers 16 × 16 m, an output cell 4 × 4 m.

Patch-size trade-off the viewer should let you *feel* (measured throughputs from the E1 sweep):

| | P = 4 | P = 8 | P = 16 |
|---|---|---|---|
| Tokens N | 4,096 | 1,024 | 256 |
| Attention compute (∝ N²) | 256× | 16× | 1× |
| Throughput on the 3060 | 30–35 samples/s | 50–95 | 115–180 |
| One token sees | 4 × 4 m: a fragment, no context | 8 × 8 m | 16 × 16 m: façade + sidewalk + part of the road |
| In-patch positions to disentangle | 16 | 64 | 256 |
| Output cells per token | 1 | 4 | 16 |

Small patches: almost nothing inside a token, everything must come through attention, which is exactly what makes them expensive. Large patches: rich local context in one token, but fine detail must be unpacked from a compressed vector.

### 13.1 Stage 1 — static figure (**implemented**)

From repo root (needs project venv + `PYTHONPATH` includes repo root; `view_sample` inserts it automatically):

`python -m analysis.view_sample --ckpt runs/checkpoints/X.pt --split val --item 17 [--out fig.png]`

One figure, fixed layout, so figures from different runs can be compared by eye:

| Panel | Content | Why |
|---|---|---|
| A | Input composite: the three layers as RGB (same colour code as the `_rgb.png` crops), marker as a crosshair, **patch grid drawn on top** (lines every P px) | the task as a human sees it, plus how the model cuts it up |
| B | Each input channel on its own (small multiples, grey) | what information exists at all |
| C | **Token view:** the marker's patch cut out and enlarged, one small image per channel, with the s × s sub-cell grid drawn over it and the target sub-cell outlined | this *is* the input of one token. At P = 16 it is a 16 × 16-pixel picture from which one of 16 sub-cells must be named; at P = 4 it is 4 × 4 px and the answer is "me or not me". The single most useful panel for patch-size intuition |
| D | Target grid (64 × 64) | what is asked |
| E | Predicted foreground probability (softmax, fixed colour scale 0–1), global argmax marked, the true patch block outlined | what the model draws; blobs are visible at a glance |
| F | Zoom of D and E on the 3 × 3 patches around the marker | sub-cell behaviour |
| G | Text: checkpoint id, P, N tokens, params, this sample's `n_fg`, patch hit, sub-cell hit, argmax distance | ties the picture to the metrics of §6 |

Modes on top of the same renderer:
- **Compare:** `--ckpt a.pt b.pt c.pt` → one row per checkpoint for the same sample (panels C, E, F per row, A once). With the P = 4 / 8 / 16 checkpoints this is the patch-size picture.
- **Over time:** `--ckpt-series runs/checkpoints/{run_id}/` → one row per saved step: the blob forming and (not) sharpening. Needs the log-spaced checkpoints of §6 (d).
- **Gallery:** `--gallery 24 --sort argmax_dist --worst` → contact sheet of panel E (with the marker) for the worst / best / random samples. Looking at 24 failures at once usually shows a pattern a metric hides (e.g. failures cluster at patch borders or on uniform backgrounds).
- **No model:** `--no-model` → panels A–D only, for inspecting a task definition before any training exists.

### 13.2 Stage 2 — interactive (later, thin shell over stage 1)

Click on the map to place the marker; the model runs (a forward pass is milliseconds) and panels C, E, F update. Controls: crop id, checkpoint(s), patch-grid overlay on/off, channel toggles, later the task id. Value over stage 1: you choose the probe yourself (marker on a sidewalk edge, in a courtyard, at a patch corner, exactly on a patch border) and get an immediate feel for where the model is sure and where it breaks.

Options, in order of effort:
1. **Notebook with matplotlib click events** (`fig.canvas.mpl_connect('button_press_event', ...)` with an interactive backend). Least code; works in a Remote-SSH notebook.
2. **Gradio app on the rig**, opened through the SSH/Tailscale port forward. `gr.Image(...).select(fn)` delivers the clicked pixel (`evt.index` = (x, y)) to a Python callback that returns the updated figure. About a hundred lines. *(Gradio API quoted from memory, not verified on the rig; check the installed version.)*
3. A custom page on top of the existing translator `viewer/`: most work, only worth it if the viewer becomes a demo for the public repo.

Recommendation: 1 or 2. A public demo is a separate, later decision.

### 13.3 Design rules

- **One pure function does the work:** `render_sample(model_or_None, img, target, meta) -> matplotlib Figure` (plus a dict of the per-sample metrics). The CLI, the gallery, the over-time strip and the interactive shell all call it. No plotting logic in the shells.
- **Same code path as training:** build `img` and `target` through `spatial_data` (`build_input_from_npz`, `c0_target_grid`), never a re-implementation, so the viewer shows exactly what the model is trained on, including any bug.
- **Task adapters for later rungs:** the renderer asks a small per-task object for (input tensor, target, how to draw target and prediction). c0: single cell. c1–c3: target mask vs predicted mask, plus an **error map** (true positive / false positive / false negative in three colours) and a per-patch IoU heat-map. T0: bar chart of class probabilities at the point next to panels A and C.
- **Fixed colour scales and fixed layout** across runs; otherwise side-by-side comparison misleads.
- Load checkpoints through the existing `checkpoint_model_config` + `LocalGridViTConfig(**cfg)` path (as `~/.cache/claude_scratch/diag_c0.py` does), with the model switched to inference mode and not compiled.

### 13.4 Where it sits in the order of §11

After step 2 (decomposed metrics) and before the Tier 1 time series: the static viewer needs nothing but a checkpoint and the data loader, and it is the tool that makes the other numbers interpretable. Interactive mode: after c1 exists, when there is more than a single marker cell to look at.

---

**Appendix to §12.** Reference values measured at initialisation (for sanity-checking an implementation of the §12 snippets): 13 groups; true update ratio on the first AdamW step, lr 6e-4: −1.52 (= log10(lr / 0.02)) for standard-init matrices and −1.07 for the scaled-init `c_proj` matrices; marker channel shared norm 0.03 vs per-position 0.39; residual RMS 0.97 and 1.06 after blocks 0 and 1; GELU off-fraction 0; attention entropy 5.54 of a maximum 5.55 in every head.
