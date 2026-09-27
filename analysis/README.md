# analysis — visual inspection and offline interpretation

This package is the **human-facing** half of the analysis suite (spec: [`docs/plans/ANALYSIS_SUITE_SPEC.md`](../docs/plans/ANALYSIS_SUITE_SPEC.md) §13). The **training-time** half lives in `harness/diagnostics/` (JSONL, plots, pred strips, checkpoint series). Use both together: scalars tell you *that* something changed; the viewer tells you *what* the model sees and draws.

**Environment:** run from the **repo root** with the project venv activated (same deps as `harness/` — torch, matplotlib, spatial data). Example:

```bash
cd /path/to/spatial_reasoning_LLM_artifact
source .venv/bin/activate   # or harness venv when split
python -m analysis.view_sample --help
```

Panel meanings (A–G): [`runs/diagnostics/VIEWER_PANELS.md`](../runs/diagnostics/VIEWER_PANELS.md).

---

## What you can do today

| Goal | Tool | When to use |
|------|------|-------------|
| Inspect one val/train item with a trained model | `view_sample` + `--ckpt` | Debugging a run, interview prep, comparing patch sizes on the **same** crop |
| Inspect task definition without weights | `view_sample --no-model` | Before training; verify marker/target alignment |
| Compare checkpoints on one item | `--ckpt a.pt b.pt …` | E1 patch-size sweep, ablations |
| Watch training dynamics on one item | `--ckpt-series runs/checkpoints/RUN_ID/` | Needs diagnostics checkpoint series (§6d) |
| Scan many failures/successes | `--gallery N --sort argmax_dist --worst` | Pattern spotting (borders, uniform tiles, etc.) |
| Regenerate loss/metrics dashboards | `cd harness && python -m diagnostics.plot_run RUN_ID` | After a run with `diagnostics_enabled=True` |
| Flip through auto-saved pred strips | Open `runs/diagnostics/{run_id}/pred_step*.png` | Cheap “movie” of the fg blob during training |

**Not implemented here:** Tier 3 offline scripts (linear probes, SV spectra, full attention maps on the grid) — spec §7; checkpoint series is ready for those.

---

## `view_sample` CLI

Default data is **worldsnap** (Amsterdam de Pijp crops + task jsonl). Use **`--synthetic`** for in-memory c0 pools (matches `harness` when `data_source=synthetic`).

### Single sample with model

**c0** (task jsonl index):

```bash
python -m analysis.view_sample \
  --ckpt runs/checkpoints/my_run.pt \
  --split val --item 17 \
  --use-gpu \
  --out runs/diagnostics/view_val0017.png
```

**c1** (crop index in `crops/v2/{split}/*_labels.npz`, sorted list — same as harness c1 loader):

```bash
python -m analysis.view_sample \
  --task-rung c1 --c1-task building \
  --ckpt runs/checkpoints/c1_L5a_building_2k.pt \
  --split val --item 0 --use-gpu \
  --out runs/diagnostics/c1_L5a_val_crop0.png
```

Panels A–G: RGB + channels, focal patch (target centroid), target mask, P(fg), 3×3 zoom, **error map** (TP/FP/FN), metrics (IoU, P/R).

- **`--use-gpu`**: CUDA forward (recommended on the rig); omit for CPU-only.
- **`--item`**: c0 → jsonl row index; c1 → crop index in the split folder.
- Output path defaults to `runs/diagnostics/view_{split}{item:04d}.png`.

### Task-only (no checkpoint)

```bash
python -m analysis.view_sample --no-model --split val --item 17 \
  --patch-size 16 --encoding scalar
```

Panels A–D only. Set **`--patch-size`** when there is no ckpt (otherwise P comes from the checkpoint config).

### Worldsnap paths

Defaults: `data/amsterdam/de_pijp`, cropset `v2`, taskset `t0_point_v0`. Override:

```bash
python -m analysis.view_sample --no-model --item 0 \
  --district-dir data/amsterdam/de_pijp --cropset v2 --taskset t0_point_v0
```

### Compare multiple checkpoints (same sample)

```bash
python -m analysis.view_sample --use-gpu --split val --item 17 \
  --ckpt runs/checkpoints/p4.pt runs/checkpoints/p8.pt runs/checkpoints/p16.pt \
  --out runs/diagnostics/compare_patch_sizes.png
```

Layout: shared panel A; one row per ckpt with C, E, F (see spec §13.1).

### Training series (diagnostics checkpoints)

After a run with diagnostics enabled:

```bash
python -m analysis.view_sample --use-gpu --split val --item 17 \
  --ckpt-series runs/checkpoints/RUN_ID/ \
  --out runs/diagnostics/series_val0017.png
```

Loads every `step*.pt` in the directory in step order.

### Gallery

```bash
python -m analysis.view_sample --use-gpu --ckpt runs/checkpoints/my_run.pt \
  --gallery 24 --sort argmax_dist --worst \
  --out runs/diagnostics/gallery_worst24.png
```

Runs forward on many val items and tiles panel E. **`--worst`**: highest argmax distance first; omit for best-first. Sort key **`patch_hit`** also available.

### Synthetic data

```bash
python -m analysis.view_sample --synthetic --seed 1337 \
  --ckpt runs/checkpoints/smoke.pt --split val --item 0 --use-gpu
```

---

## Training diagnostics (harness)

Enable on a run:

```python
# harness/config.py or experiment overrides
diagnostics_enabled = True
diagnostics_interval = 10          # tier1 JSONL every N steps
diagnostics_eval_interval = None   # None = every CE eval row in eval.jsonl
```

Then train via `python harness/train.py` or `python harness/harness.py`. Artifacts:

| File | Content |
|------|---------|
| `runs/diagnostics/{run_id}/tier1.jsonl` | Tier 1: timing, grad norms, update ratios, patch shared/resid |
| `runs/diagnostics/{run_id}/eval.jsonl` | Tier 2: spatial metrics + probe scalars |
| `runs/diagnostics/{run_id}/meta.json` | Config snapshot, group list, **`probe_sample_ids`** |
| `runs/diagnostics/{run_id}/dashboard.png` | Harness `plot_run` (update ratios, grad norms, …) |
| `runs/diagnostics/{run_id}/summary.png` | C3 `c3_within_diagnostics_plot` (IoU, attn→marker, …) |
| `runs/diagnostics/{run_id}/pred_step*.png` | 8 fixed samples × (input, target, P(fg)) |
| `runs/checkpoints/{run_id}/step{N}.pt` | Weights-only series for viewer + future Tier 3 |

Plot dashboard:

```bash
cd harness
python -m diagnostics.plot_run RUN_ID
python -m diagnostics.plot_run RUN_ID --compare OTHER_RUN_ID
```

Remove all on-disk files for a run (diagnostics + checkpoints, not `experiments.json`): `python -m diagnostics.delete_run RUN_ID` — see [`harness/README.md`](../harness/README.md).

Details and config keys: [`harness/README.md`](../harness/README.md).

**Cost note:** probe forward + extra val pass at each recorded CE eval can dominate wall time on a 3060; reduce `diagnostics_probe_batch_size` for P=4 or long runs.

---

## Package layout

| Module | Role |
|--------|------|
| `view_sample.py` | CLI entrypoint |
| `render.py` | Fixed A–G layout |
| `load_sample.py` | Worldsnap + synthetic loaders (same tensors as training) |
| `load_model.py` | Checkpoint → `LocalGridViT` |
| `c0_adapter.py` | Forward + per-sample c0 metrics for panel G |
| `geometry.py` | Patch / subcell grid math |
| `rgb.py` | Label-plane RGB for panel A |

Design rule from the spec: **one renderer** (`render_sample` / `render_compare`); shells only parse args and call it.

---

## Typical workflow

1. **During training:** turn on `diagnostics_enabled` for runs you will defend in the README.
2. **After eval looks odd:** `plot_run` → if patch_hit ↑ but subcell flat, open **`VIEWER_PANELS.md`** and run `view_sample` on a few `--item`s.
3. **Patch-size or architecture choice:** compare ckpts with `--ckpt` or E1 series with `--ckpt-series`.
4. **Before scaling data or GPU hours:** check tier1 `gpu_starvation` in plots or JSONL; fix loader before interpreting probe entropy.

For the full metric dictionary and plot questions, see **§0–§9** in [`docs/plans/ANALYSIS_SUITE_SPEC.md`](../docs/plans/ANALYSIS_SUITE_SPEC.md).
