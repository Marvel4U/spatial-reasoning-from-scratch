# Reproduce

Everything was run on one machine: Ubuntu, one RTX 3060 (12 GB), 30 GB RAM, Python 3.12, PyTorch 2.14 with CUDA. Wall times below are from that machine.

## 1. Environment

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q        # tests that need data skip when the data is absent
```

The Qwen track needs a few more packages (`transformers`, `peft`, `accelerate`, ...), installed into the same environment: `pip install -r requirements-track-a.txt` (see `VENV.md`).

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

One model per one-shot task level is attached to the GitHub release as a `.pt` file holding `model_state_dict`, `model_config`, the training record and the source run id. Load from the repository root:

```python
import torch
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
ck = torch.load("c4b_detour.pt", map_location="cpu", weights_only=False)
model = LocalGridViT(LocalGridViTConfig(**ck["model_config"])); model.load_state_dict(ck["model_state_dict"])
```

`docs/figures_make.py` and `docs/figures_make2.py` regenerate the one-shot README figures from the ledger, the diagnostics files and these checkpoints.

The search rung adds three checkpoints in the same format, and the Qwen track one LoRA adapter:

| file | run id | what |
|---|---|---|
| `c5_wave_search_256m.pt` | `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k` | the search at 256 m after one round of natural noise (the README's results table) |
| `c5_wave_search_512m.pt` | `c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512` | the search at 512 m after one round of natural noise at that size |
| `c5_oneshot_256m_matched.pt` | `c5_os_bld_det_L4_64k_g128_store100k` | the budget-matched one-shot network (64k steps, no scratchpad) |
| `qwen3vl2b_lora_route_v1_19k/` | Track A `lora20000_v1` | LoRA adapter for Qwen3-VL-2B-Instruct, 19.3k training routes; the base weights are not included |

The evaluation scripts take a run id (for its configuration in `harness/experiments.py`) and `--checkpoint` for the file, so a released checkpoint is evaluated with the commands of section 6.

## 6. The search (rung C5)

**Wave data.** One wave per route of an existing route store (section 2), written next to the store:

```bash
python -m spatial_data.c5_wave --store data/amsterdam/de_pijp/crops/v4/c4_100k --splits train,val,test --workers 4   # train 100k: ~93 min CPU; val 500: ~40 s
```

**The second district and the 512 m routes.** `worldsnap.build` and `worldsnap.crops` with `--district zuid_centrum` (bounding box in `worldsnap/config.py`), then `spatial_data.c4_routes` and `spatial_data.c5_wave` on its stores (`c4_256_val` for the 256 m validation and test routes, `c4_512_20k` for the 512 m training and validation routes).

**The house maze** (synthetic, nothing downloaded):

```bash
python -m spatial_data.maze_world --out data/synthetic/maze_v2_houses/crops/v4ns --wall-m 14,30 --seed 1
python -m spatial_data.c4_routes --crops-dir data/synthetic/maze_v2_houses/crops/v4ns --out data/synthetic/maze_v2_houses/crops/v4ns/c4_256_val \
    --n-train 0 --n-val 500 --n-test 500 --crop-px 256 --d-min 40 --d-max 250 --max-expansions 600
python -m spatial_data.c5_wave --store data/synthetic/maze_v2_houses/crops/v4ns/c4_256_val --splits val,test
```

The thin-wall maze of the report (`maze_v1`) is the same with the defaults `--wall-m 2,6 --seed 0`. The 512 m maze store (`c4_512_val`) is built the same way for 512 m windows.

**Training.** Specs in `harness/experiments.py`, run from `harness/` as in section 3. Training times from the run ledger, RTX 3060:

| step | spec | run id | steps | time |
|---|---|---|---:|---:|
| the search, 4 blocks (2-block control) | `c5_wave_night_queue_2oct()` | `c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k` (`..._L2_...`) | 48k | 2.5 h (1.8 h) |
| natural noise, round 1 | `c5_wave_own_pads_queue()` (the continued run) | `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k` | 16k | 51 min |
| natural noise, round 2 | `c5_wave_natural_noise_round(...)` | `c5_r5_wave_nat3_tok01_ownp50_r2_L4_16k_g128_store100k` | 16k | 52 min |
| 512 m, continued from round 1 | `c5_wave_512_queue(...)` | `c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512` | 16k | 2.6 h |
| 512 m, natural noise | `c5_wave_512_natural_noise_round(...)` | `c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512` | 16k | 2.6 h |
| cleanup pass, draw weight 3 | `c5_wave_cleanup_queue(...)` | `c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g128_store100k` | 16k | 52 min |
| one-shot, budget-matched | `c5_oneshot_matched_queue()` | `c5_os_bld_det_L4_64k_g128_store100k` | 64k | 3.0 h |

**The store of the network's own drawings** (natural noise) and the cleanup targets, from a trained checkpoint, written into the route store:

```bash
python -m spatial_data.c5_own_pads c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k --limit 100000   # 1.16 M passes, 2.3 GB, ~10.5 min GPU
python -m spatial_data.c5_cleanup data/amsterdam/de_pijp/crops/v4/c4_100k --own c5_own_pads_train_r2.npz --tag r2   # ~1 min CPU
```

**Evaluation and animations** (from `harness/`; outputs under `runs/diagnostics/<run id>/`):

```bash
cd harness
RUN=c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k
# free-running and teacher-schedule rollouts on 500 De Pijp val routes (add --checkpoint FILE for a released checkpoint)
PYTHONPATH=. python diagnostics/wave_rollout_eval.py $RUN --n 500 --tag val500
# the same on the test split, the never-seen district, the house maze
PYTHONPATH=. python diagnostics/wave_rollout_eval.py $RUN --n 500 --split test --tag test
PYTHONPATH=. python diagnostics/wave_rollout_eval.py $RUN --n 500 --crops ../data/amsterdam/zuid_centrum/crops/v4ns --store c4_256_val
PYTHONPATH=. python diagnostics/wave_rollout_eval.py $RUN --n 500 --crops ../data/synthetic/maze_v2_houses/crops/v4ns --store c4_256_val
# wall time per pass and per route; one animation
PYTHONPATH=. python diagnostics/wave_rollout_timing.py $RUN
PYTHONPATH=. python diagnostics/wave_rollout_gif.py $RUN --route 333
# the one-shot network under the search's rule
PYTHONPATH=. python diagnostics/oneshot_route_eval.py c5_os_bld_det_L4_64k_g128_store100k --n 500
```

A rollout evaluation of 500 routes at 256 m takes a few minutes (57 ms per route, report §4.5).

## 7. The Qwen track

From `qwen_reference/README.md` and `qwen_fine_tuning/README.md`, run from the repository root. The model weights are fetched once with `python models/download_qwen.py` into `models/hf/` (git-ignored).

```bash
# map images + teacher polylines in the 0-1000 grid (the shared contract), and the straight-line floor
python -m spatial_data.track_a_export export --store data/amsterdam/de_pijp/crops/v4/c4_100k --split val --out data/track_a/de_pijp_val_plain
python -m spatial_data.track_a_export export --store data/amsterdam/de_pijp/crops/v4/c4_100k --split train --limit 20000 --out data/track_a/de_pijp_train20k_plain
python -m spatial_data.track_a_export baseline --routes data/track_a/de_pijp_val_plain/routes.jsonl --kind straight --out straight.jsonl
# zero-shot on 200 routes: two-pass vs oracle endpoints (2B, then 4B)
python -m qwen_reference.eval_ab --limit 200 --quiet --device-map cuda:0
python -m qwen_reference.eval_ab --model models/hf/Qwen3-VL-4B-Instruct --tag 4b --limit 200 --quiet --device-map cuda:0
# LoRA on the full usable training set, a checkpoint every 500 steps (add --resume after an interruption), then 500 val routes
python -m qwen_fine_tuning.train_lora --train-limit 20000 --dev-limit 20 --output-dir runs/qwen_sft/lora20000_v1 --checkpoint-every 500 --device-map cuda:0
python -m qwen_fine_tuning.eval_lora --adapter runs/qwen_sft/lora20000_v1/adapter --export data/track_a/de_pijp_val_plain --limit 500 --device-map cuda:0 --quiet
```

Wall times on the RTX 3060: about 4 s per optimizer step at effective batch 4, so about 0.75 × N steps for N training routes over 3 epochs; the 19.3k run took about 8 h (14,483 optimizer steps); generating and scoring 500 val routes takes 20–30 min (Track A report §3).
