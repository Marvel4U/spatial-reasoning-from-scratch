# harness — local grid ViT experiments

Run scripts from this directory; **paths are anchored to the repo root** (`config.REPO_ROOT`), not `harness/`.

| What | Where |
|------|--------|
| Worldsnap / crops | `../data/amsterdam/de_pijp/...` |
| Experiment log | `../runs/experiments.json` |
| Run checkpoints | `../runs/checkpoints/{run_id}.pt` |
| Tier 1 diagnostics | `../runs/diagnostics/{run_id}/tier1.jsonl` (if `diagnostics_enabled`) |

```bash
cd harness
python train.py          # verbose stage logs by default; --quiet to silence
python train.py -v       # same (verbose is default)
python train.py --quiet  # minimal output
python harness.py        # runs experiments.TO_RUN
```

Edit **`experiments.py`**: parameter blocks (`C0_*`) + open **`RUN`** list (spread builders or inline dicts). Builders live in **`experiment_sweeps.py`**. Defaults: **`config.py`**.

**Data:** `data_source = "synthetic"` (in-memory c0) or `"worldsnap"` (`spatial_data` + `data/amsterdam/de_pijp/crops/{cropset}` + `tasks/{taskset}`). Set `encoding_mode`; `load_data()` syncs `in_chans`. Worldsnap: `worldsnap_cache_layers=True` (default) loads ~2k unique npz layer stacks once (~1 GB); optional `worldsnap_materialize_items=N` prebuilds all `(img,tgt)` when item count ≤ N (e.g. 512 for overfit — not for full 32k). On CUDA, `worldsnap_gpu_resident=True` uploads unique crops once and builds batches on GPU (`spatial_data/dataset_c0_gpu.py`); default in `config.py` is False, worldsnap experiment overrides enable it.

**Adapter:** `grid_vit_adapter.py` → `plain_gpt_module.local_grid_vit.LocalGridViT`.

**Training throughput** (logged each eval): `samples/s`; **`cells/s`** = supervised grid positions per second (batch × 64²), comparable across depth/width like GPT **tokens/s**; **Mpix/s** = input 256² planes/sec (scales with resolution, not channels).

**Spatial eval (c0):** After training, full val metrics + `c0 detail:` line (`eval_spatial.py`). During training, each CE eval also prints **`c0 snap @ <step>:`** when `spatial_eval.train_snap_batches` > 0 (default 3 in `experiments.py`); **`estimate_loss` skips the val split** then and logged `val loss` comes from the snap. Set `train_snap_batches: 0` to restore train+val CE eval. Default **`eval_interval=500`** (config + worldsnap experiments). Key snap fields: `argmax_patch`, `subcell|patch` (chance 1/16 at P=16), `full_patch` = fraction of grids with all 4×4 subcells predicted fg.

**Diagnostics:** Set `diagnostics_enabled = True` in `config.py` or experiment `overrides`. Spec + status: `docs/plans/ANALYSIS_SUITE_SPEC.md` §0. **Per-run folder** `runs/diagnostics/{run_id}/`: `tier1.jsonl`, `eval.jsonl`, `meta.json`, `dashboard.png` (from `plot_run`), `summary.png` (c3 script), `pred_step*.png`, optional `compare_{other}.png`. Tier 1 every `diagnostics_interval` steps; tier 2 at CE evals. Checkpoint series stays `runs/checkpoints/{run_id}/step{N}.pt`. Plots: `cd harness && python -m diagnostics.plot_run RUN_ID [--compare OTHER]`. Legacy flat files: `python -m diagnostics.migrate_layout`. Task viewer: `analysis/README.md`; panel guide: `runs/diagnostics/VIEWER_PANELS.md`.

**Delete one run’s artifacts:** `python -m diagnostics.delete_run RUN_ID` removes `runs/diagnostics/{run_id}/`, legacy flat files if any, checkpoint series, and optionally `runs/checkpoints/{run_id}.pt`. Does **not** change `runs/experiments.json`.
