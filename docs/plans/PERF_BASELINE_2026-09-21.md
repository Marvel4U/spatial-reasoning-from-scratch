# Performance baseline — Step 0 (21 Sep 2026)

Recorded **before** Step 1 (`torch.set_num_threads`). Re-run the scripts after each perf step; do not run two profilers at once (they fight for CPU and inflate `next_batch`).

## Environment (the RTX 3060 rig)

| Item | Value |
|------|--------|
| Git | `0efd9fa1f81eda4742367d2355d57a043ddb23c2` on `main` |
| GPU | NVIDIA GeForce RTX 3060 12 GB (~100 MiB used at idle) |
| CPU | 6 cores |
| PyTorch | 2.14.0+cu130, default **6** intra-op threads |
| Harness defaults | `data_source=worldsnap` (in profilers), `cropset=v2`, `taskset=t0_point_v0`, `batch_size=32`, `patch_size=16`, `encoding_mode=scalar`, bf16 CUDA, `use_compile=False` |
| Layer cache | on, not materialized (`materialized: False`) |
| Train / val items | 2000 / 250 (full de_pijp split) |

## Hygiene (manual)

- **`/usr/bin/resources` (GNOME Resources)** was running at baseline (PID 228052, remote-desktop session). It pins ~1 core and amplifies PyTorch CPU op overhead. Close it before trusting `next_batch` timings; killing it may require the `remote` user session.
- Before GPU-resident loader (Step 2): `nvidia-smi` — ensure no other process holds most VRAM.

## Commands

From repo:

```bash
cd harness
../.venv/bin/python ../docs/plans/perf_prototypes/profile_batch.py
../.venv/bin/python ../docs/plans/perf_prototypes/profile_harness.py
```

## `profile_batch.py` (single run, no concurrent jobs)

| Metric | ms |
|--------|-----|
| `next_batch` @ default 6 threads | **238.7** |
| `32× sample()` | 50.7 |
| `torch.stack` imgs | 18.1 |
| Per-sample `build_input_from_layers` (isolated) | 0.128 |
| `next_batch` if `set_num_threads(4)` in-script only | **25.5** |
| `next_batch` if `set_num_threads(1)` in-script only | 37.3 |
| GPU prototype `gpu_batch` (not wired to harness) | **0.74** |
| Prototype vs CPU scaling max | both 1.0 |

**Note:** A second run started while another profiler was loading data reported `next_batch` **1453 ms** — treat as contention, not regression.

## `profile_harness.py` (Step 0)

Not completed at Step 0 (parallel run cancelled).

## After Step 1 (`cpu_num_threads=4` in `setup_device`, commit pending)

Same machine, sequential runs, `profile_harness.py` calls `setup_device()`:

| Metric | Value |
|--------|--------|
| `next_batch` | **24.8 ms** |
| host→device | 4.1 ms |
| GPU train step | 16.0 ms |
| **One step throughput** | **713 samples/s** (data share 64%) |
| `estimate_loss` (20×2) | 1.17 s |
| spatial eval 10 batches | 0.50 s |
| eval share @ interval 100 | 27% |
| Warm `next_batch` (20 iters after 3 warmup) | 24.1 ms |

`profile_batch.py` with `setup_device()`: `threads=4` line **27.9 ms** (first `next_batch` line in same script can still look high under load; trust harness + warm timing).

## Targets (from PERFORMANCE_PLAN_2026-09-21.md)

| After | `next_batch` (CPU path) | throughput @ B=32 |
|-------|-------------------------|-------------------|
| Step 1 | ~23–40 ms | ~700–800 samples/s |
| Step 2 GPU loader | ~1 ms batch on device | ~1900 samples/s |

## After Step 2 (`worldsnap_gpu_resident=True`, `code_version` 1.1.7)

`profile_harness.py` with GPU loader enabled in script:

| Metric | Value |
|--------|--------|
| `load_data` (incl. GPU crop upload) | 6.3 s |
| `next_batch` | **0.7 ms** |
| host→device | ~0 |
| GPU train step | 16.2 ms |
| **One step throughput** | **~1889 samples/s** |
| 100× timed `next_batch` | **0.75 ms** |
| Tests | `pytest tests/test_c0_gpu_loader.py` — 3 passed |

## After Step 4 (vectorized `eval_spatial`, `code_version` 1.1.9)

Same GPU loader + `profile_harness.py`:

| Metric | Value |
|--------|--------|
| spatial eval 10 batches | **0.08 s** (was ~0.24 s post–Step 2) |
| spatial eval 20 batches | **0.24 s** |
| ~per batch (20) | **~12 ms** (metric loop no longer dominant) |
| Tests | `pytest tests/test_eval_spatial_vectorized.py` — 4 passed |

## After Step 5 (`code_version` 1.2.0) — eval schedule

| Change | Detail |
|--------|--------|
| `eval_interval` | **250** (config default + worldsnap experiment overrides; was 100) |
| Duplicate val CE | When `spatial_eval.train_snap_batches > 0`, `estimate_loss` runs **train only**; **`c0 snap`** supplies `val loss` for logs / recorder / tier2 |
| Harness | `on_eval` runs snap **before** recorder; returns patched losses for the step print line |

Rough eval share @ 250 steps × ~16 ms/step ≈ 4 s train vs ~0.15 s CE train-only + ~0.03 s snap ≈ **under 5%** per interval (plus full spatial eval once at end).

## After Step 6 — batch size & compile (3060, P=16, gpu loader, eager)

Script: `docs/plans/perf_prototypes/profile_batch_size.py`

| batch_size | ms/step | samples/s | peak VRAM |
|------------|---------|-----------|-----------|
| 16 | 8.7 | 1834 | 698 MB |
| 32 | 15.5 | 2061 | 867 MB |
| 64 | 29.5 | 2171 | 1188 MB |
| 128 | 57.7 | 2217 | 1836 MB |
| 256 | 113.6 | 2253 | 3125 MB |

All fit on 12 GB; throughput saturates ~**2200–2250 samples/s** (step time scales ~linearly with B).

**`torch.compile`** @ B=32: warm-up ~9 s for 3 steps; step **13.4 ms** vs eager **15.5 ms** (~**16%** faster). Leave `use_compile=False` for smokes; enable for multi-k-step GEX runs. Re-tune LR if you change batch size (not auto-scaled in harness).
