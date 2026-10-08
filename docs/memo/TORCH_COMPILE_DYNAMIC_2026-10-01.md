# Memo: `torch.compile`, the `dynamic` flag, and harness diagnostics (1 Oct 2026)

Short record of compile-related harness changes and what we measured on the training machine (RTX 3060, torch 2.11+cu128,
C5 wave fwd3 mix, L4, batch 128). Updated after external review same day (`dynamic=None` recommendation).

## 1. What changed in the repo

**Config (`harness/config.py`).**

- `use_compile` — harness sets True for long runs (`train_steps >= use_compile_min_train_steps`, default 2000)
  unless the spec overrides it.
- **`use_compile_dynamic: bool | None = None`** — passed to `torch.compile(..., dynamic=...)`. **`None` omits the
  PyTorch 2.1+ default** (specialize static shapes first; mark dimensions dynamic only when a recompile is
  triggered by a new size).
- **`use_compile_dynamo_cache_size_limit = 32`** — raised from Dynamo’s default **8** graphs per function so
  harness shape variants (train with targets, eval without, short last val batch, probe, batch-1 paths, rollout)
  do not silently fall back to eager.

**Model build (`harness/grid_vit_adapter.py`).**

- `_maybe_compile(model)` respects `None` / `False` / `True` (never coerces `None` to `False`).
- `_configure_dynamo_for_compile()` applies the cache limit and logs once; use **`TORCH_LOGS=recompiles`** on a
  smoke to count variants.

**Diagnostics (`harness/diagnostics/probe_metrics.py`).**

- Tier-2 **probe metrics** register hooks on the **eager** `LocalGridViT` (`unwrap_compiled(model)`).
- Probe forward must use that eager module. Compiled wrapper forwards **do not run submodule hooks** →
  `KeyError: 'attn_in/0'` at CE eval. **This fix is independent of `dynamic`** and is the best explanation for
  several “compile broke at first eval” incidents alongside probe batch 64 vs train 128.

**Harness (`harness/harness.py`).**

- `_align_probe_batch_for_compile`: when compile is on and probe is enabled, set
  `diagnostics_probe_batch_size = batch_size` **only if the spec did not set probe B**.

**Wave beta overrides (`_c5_wave_long_run_compile_overrides`).**

- Bakes `use_compile_dynamic=None` into `_c5_wave_beta_overrides` for long C5 wave runs.

**Loss inside the compiled graph.**

- Change-weighted CE (`plain_gpt_module/wave_pad_ce.py`) must stay **fully tensorized** in the forward path.
  An earlier Python loop with **`.item()` per sample** caused **graph breaks every step** and largely negated
  compile wins. The vectorized group weights are what make compile worthwhile on beta/cgce runs.

**Smoke spec (`c5_wave_compile_smoke`).**

- 512 steps; explicit `dynamic=False` vs `True` for A/B throughput (see §4). Production default is **`None`**.

## 2. What `dynamic` means here (PyTorch 2.x Inductor)

| Setting | Behaviour | Our harness |
|--------|-----------|-------------|
| **`None` (recommended)** | PyTorch default since 2.1: **static kernels first**; if a new shape appears, recompile and treat that dim as dynamic. | Fixed train grid (128×128, B=128) gets static speed; odd batches (val tail, probe if misaligned, rollout) handled without forcing global `dynamic=True`. |
| **`False`** | Every distinct shape → **new specialized graph**. | Can be fast on the hot path (smoke static ~742 samples/s) but **multiplies graphs**; needs **`cache_size_limit ≥ ~32`** or Dynamo silently goes eager after 8 variants. |
| **`True`** | Up-front shape-polymorphic kernels. | Smoke **~376 samples/s** at step 256 vs static **~742** on 3060; our shapes are fixed in practice — **no benefit**, clear cost. Old hard-coded `dynamic=True` correlated with Inductor backward/tiling failures on lrflat before probe fixes. |

Even at fixed `batch_size=128`, call patterns differ: `targets` present vs absent, eval batching, diagnostics,
rollout. Aligning probe batch with train batch reduces avoidable recompiles; **`dynamic=True` is still wrong**
for this model.

## 3. Interaction map

```text
  training forward (compiled, loss in graph — tensorized CE only)
           │
  eval / pred / rollout (compiled, often targets=None)
           │
  probe metrics (eager core + hooks)  ← mandatory split
           │
  dynamic=None  →  static train path + recompile on odd shapes
  dynamic=True  →  slow polymorphic kernels (avoid)
  dynamic=False →  many specialized graphs; watch cache_size_limit
```

## 4. Results log (1 Oct 2026)

| Run id | `dynamic` | Probe via eager | Outcome |
|--------|-----------|-----------------|--------|
| `…compile_smoke_512` (first static) | False | No | Aborted @ CE eval — `KeyError: attn_in/0` |
| `…compile_smoke_512_static` | False | Yes | **Complete** ~131 s |
| `…compile_smoke_512_dyn` | True | Yes | **Complete** ~236 s (≈half train throughput) |
| `…beta2_lrflat_L4_12k_*` (earlier) | True (hard-coded) | N/A | **Failed** step 0–1 Inductor |

Logs: `runs/logs/c5_wave_compile_smoke_512_r2.log`, `…_512_dyn.log`.

**Conclusion:** Avoid **`dynamic=True`**. Prefer **`None`** for production; **`False`** is fine if you monitor
recompiles. **Probe-on-eager + aligned probe batch** matter more than the flag for correctness.

## 5. Practical defaults

1. **`diagnostics_probe_batch_size = batch_size`** when compile is on, unless the spec sets probe B explicitly.
2. **Probe forward on `unwrap_compiled(model)`** — required with compile.
3. **`use_compile_dynamic=None`** on CUDA long runs (baked in `_c5_wave_beta_overrides`). Do **not** set
   `dynamic=True`.
4. **`use_compile_dynamo_cache_size_limit=32`** when compile is on (automatic in `build_model`).
5. Keep change-weighted CE **vectorized** — no per-sample `.item()` in the loss path.
6. Resume eager↔compile still allowed (`use_compile` in `RESUME_SKIP_CONFIG_KEYS`).

Optional debug: one smoke with `TORCH_LOGS=recompiles` to list graph variants before a GEX131 12k.

## 6. Harness log noise (unrelated to failures)

On compile runs the terminal may show:

- **`_POSIX_C_SOURCE` redefined** — Triton/Inductor JIT including Python + CUDA headers on Ubuntu 24.04; benign.
- **“Online softmax is disabled… split the reduction”** — Inductor lowering choice for attention; informational.

These appeared during **successful** β5 12k step-0 training; not action items unless a run **Traceback**s.

## 7. References in-tree

- Auto compile + probe align: `harness/harness.py`
- Memo context: `docs/memo/CHANGE_WEIGHTED_LOSS_2026-10-02.md`, `docs/plans/PERFORMANCE_PLAN_2026-09-21.md`
