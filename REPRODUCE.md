# Reproduce

Everything was run on one machine: Ubuntu, one RTX 3060 (12 GB), 30 GB RAM, Python 3.12, PyTorch 2.14 with CUDA. Wall times below are from that machine.

## 1. Environment

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q        # tests that need data skip when the data is absent
```

## 2. Data: rebuild the district from public sources

The raw layers (about 5 GB) are not in the repository. `worldsnap/` downloads them from the sources listed in `DATA_LICENSES.md`, writes a manifest with licence, vintage and SHA-256 per file, rasterises them at 1 m, and cuts crops. The district is De Pijp, Amsterdam; its bounding box is in `worldsnap/config.py`.

```bash
python -m worldsnap.build --district de_pijp --layers buildings,streets,transit,trees,noise,elevation,pois   # download + rasterise (~30 min, network-bound)
python -m worldsnap.build --district de_pijp --layers sunshine   # derived sunshine layer (GPU ray-march, a few minutes)
python -m worldsnap.crops --district de_pijp --cropset v4 --n-train 2000 --n-val 250 --n-test 250   # crops + district label stack
python -m spatial_data.c4_routes --n-train 100000 --n-val 500 --n-test 500 --out data/amsterdam/de_pijp/crops/v4/c4_100k   # path targets (~3.5 h CPU)
```

All outputs land under `data/amsterdam/de_pijp/`, which is git-ignored. The first three commands are deterministic given the upstream data; the upstream data changes over time (OpenStreetMap, Overture, the building register), so exact numbers may drift by a few thousandths.

## 3. Train one model per task level

Runs are specified as functions in `harness/experiments.py` and executed by the harness from the `harness/` folder; `RUN` at the bottom of that file selects what runs. Each spec names the task, the model size, the store and the seed; the run ledger `runs/experiments.json` records what was actually run, including every configuration used in the README. Approximate wall times on the RTX 3060:

| task level | spec | steps | time |
|---|---|---:|---:|
| one condition on one layer | `c1_l5b_v3b_sidewalk_sweep` (onehot, P = 16) | 2k | 1 min |
| two conditions, held-out pairs | `c2_L6c_c2_full_2k` | 2k | 1 min |
| condition relative to the crop | `c3_quieter_sidewalk_median_strict_2k` | 2k | 1 min |
| within 100 m of a marker | N4 within-radius queue, 100 m, disc marker, bias all | 8k | 3 min |
| food & drink within 100 m (three-task mix) | `m1_report_79_mix33_L2_16k_seed(5)` | 16k | 12 min |
| straight segment between two markers | `c4_c4a_explore_run_queue` (segment only) | 4k | 9 min |
| shortest path around buildings | `c4_mix_probe_run_queue` (½ building + ½ detour, 4 blocks, 100k store) | 16k | 55 min |

Diagnostics (attention entropy and attention-to-marker per head, update ratios, prediction strips) are written to `runs/diagnostics/<run id>/` when `diagnostics_enabled` is set in the spec.

## 4. Probes

Offline scripts under `docs/plans/perf_prototypes/`, each taking a run id:

- `c3_m1_shortcut_probe.py`: prediction against each single-condition mask, and the marked share by distance band (finding 5 in the README).
- `c3_wrong_token_probe.py`: the same crops under each condition token, plus attention-to-marker per head per token.
- `c4_tolerant_metrics.py`, `c4_optcheck.py`, `c4_gallery.py`: the one-cell-tolerant line metrics, the optimality check of stored paths against a grid search, and a gallery of a path store.

## 5. Released checkpoints

One model per task level is attached to the GitHub release as a `.pt` file holding `model_state_dict`, `model_config`, the training record and the source run id. Load from the repository root:

```python
import torch
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
ck = torch.load("c4b_detour.pt", map_location="cpu", weights_only=False)
model = LocalGridViT(LocalGridViTConfig(**ck["model_config"])); model.load_state_dict(ck["model_state_dict"])
```

`docs/figures_make.py` and `docs/figures_make2.py` regenerate every README figure from the ledger, the diagnostics files and these checkpoints.
