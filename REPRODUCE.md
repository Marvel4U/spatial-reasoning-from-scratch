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

The search rung adds four checkpoints in the same format, the Qwen track one LoRA adapter, and two bundles hold the
evaluation data and the Qwen predictions (contents and sizes: README Table 7; commands: section 8):

| file | run id | what |
|---|---|---|
| `c5_wave_search_256m.pt` | `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k` | the search at 256 m after one round of natural noise (the README's results table) |
| `c5_wave_search_512m.pt` | `c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512` | the search at 512 m after one round of natural noise at that size |
| `c5_wave_search_512m_hero.pt` | `c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512` | the search at 512 m before that round (README Figure 1) |
| `c5_oneshot_256m_matched.pt` | `c5_os_bld_det_L4_64k_g128_store100k` | the budget-matched one-shot network (64k steps, no scratchpad) |
| `qwen3vl2b_lora_route_v1_19k.zip` | Track A `lora20000_v1` | LoRA adapter for Qwen3-VL-2B-Instruct, 19.3k training routes; the base weights are not included |
| `eval_bundle_v0.2.tar.gz` | – | district label stacks, validation route stores, Track A validation exports, per-route results of the paper's runs |
| `qwen_val_predictions_v0.2.tar.gz` | – | Qwen zero-shot and LoRA predictions on De Pijp val with their evaluation reports |

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

## 8. Reproducing the paper's tables from the release

The evaluation numbers of Sections 5–7 of the README can be recomputed from a fresh clone and the release assets alone,
without rebuilding the districts or retraining. Everything below was run on 10 Oct 2026 from a fresh clone of this
branch in `/tmp/verify_pub`, with nothing but the release files added; the results are in the table at the end.

**Release files and where they go.** The scripts find a network by its run id as `runs/checkpoints/<run id>.pt`, so
the released `.pt` files are renamed on placement (or passed with `--checkpoint FILE`):

| release file | place as |
|---|---|
| `c5_wave_search_256m.pt` | `runs/checkpoints/c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k.pt` |
| `c5_wave_search_512m.pt` | `runs/checkpoints/c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512.pt` |
| `c5_wave_search_512m_hero.pt` | `runs/checkpoints/c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512.pt` |
| `c5_oneshot_256m_matched.pt` | `runs/checkpoints/c5_os_bld_det_L4_64k_g128_store100k.pt` |

`eval_bundle_v0.2.tar.gz` holds, in the repository's layout, the district label stacks (`district_labels_1m.npz` and
its legend) of De Pijp, Zuid + Centrum and the house maze, the validation route stores used in the paper (De Pijp
`c4_100k` val; Zuid + Centrum `c4_256_val` val and test and `c4_512_20k` val; house maze `c4_256_val` and `c4_512_val`
val), the three Track A validation exports with their baselines and published polyline predictions, and the
per-route evaluation JSONs, `meta.json` and final drawings (`pads_*.npz`) of the seven runs the paper's search and
one-shot numbers come from. `qwen_val_predictions_v0.2.tar.gz` holds the Qwen predictions and evaluation reports.
Each has a `MANIFEST.md`. The run configurations come from the ledger `runs/experiments.json`, which is in the
repository.

**Setup** (from the repository root; `data/` and `runs/checkpoints/` are git-ignored and do not exist in a clone):

```bash
git clone -b c5-wave-search <repo url> spatial_reasoning && cd spatial_reasoning
# (environment as in section 1)
R=/path/to/release/files
mkdir -p data runs/checkpoints
tar -xzf $R/eval_bundle_v0.2.tar.gz -C /tmp && cp -r /tmp/eval_bundle_v0.2/data/. data/ && cp -r /tmp/eval_bundle_v0.2/runs/. runs/
tar -xzf $R/qwen_val_predictions_v0.2.tar.gz -C /tmp && cp -r /tmp/qwen_val_predictions_v0.2/runs/. runs/
cp $R/c5_wave_search_256m.pt      runs/checkpoints/c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k.pt
cp $R/c5_wave_search_512m.pt      runs/checkpoints/c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512.pt
cp $R/c5_wave_search_512m_hero.pt runs/checkpoints/c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512.pt
cp $R/c5_oneshot_256m_matched.pt  runs/checkpoints/c5_os_bld_det_L4_64k_g128_store100k.pt
```

The bundle's `runs/diagnostics/<run>/*.json` are the published results; the commands below write new files next to
them under the `--tag` given, so the two can be compared route by route (`free_per_route` / `oneshot_per_route`).

**Commands** (GPU, bf16; about 1.5 min per 500 routes at 256 m and 2 min at 512 m on the RTX 3060):

```bash
cd harness; export PYTHONPATH=.
R256=c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k
R512=c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512
RH=c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512
# (a) the 256 m search, free-running, De Pijp val (Table 3)
python diagnostics/wave_rollout_eval.py $R256 --n 500 --mode free --tag _verify_val500
# (b) the never-seen district and the house maze (Table 3)
python diagnostics/wave_rollout_eval.py $R256 --n 500 --mode free --crops ../data/amsterdam/zuid_centrum/crops/v4ns --store c4_256_val --tag _verify_zc256
python diagnostics/wave_rollout_eval.py $R256 --n 500 --mode free --crops ../data/synthetic/maze_v2_houses/crops/v4ns --store c4_256_val --tag _verify_maze256
# (c) the 512 m networks: Zuid + Centrum 512 m val (the run's own store) and the 512 m house maze (Section 5.4)
python diagnostics/wave_rollout_eval.py $R512 --n 500 --mode free --tag _verify_zc512
python diagnostics/wave_rollout_eval.py $R512 --n 500 --mode free --crops ../data/synthetic/maze_v2_houses/crops/v4ns --store c4_512_val --tag _verify_maze512
python diagnostics/wave_rollout_eval.py $RH   --n 500 --mode free --crops ../data/synthetic/maze_v2_houses/crops/v4ns --store c4_512_val --tag _verify_maze512
# (d) the budget-matched one-shot network under the search's cell rule (Section 7.2)
python diagnostics/oneshot_route_eval.py c5_os_bld_det_L4_64k_g128_store100k --n 500 --tag _verify_val500
# (e) the search's drawings as polylines under the polyline rule (Table 5); --rerun recomputes the drawings from the
#     checkpoint instead of reusing the released pads_wave.npz, --pads keeps the released one untouched
python diagnostics/drawing_to_polyline.py $R256 --kind wave --export ../data/track_a/de_pijp_val_plain --rerun --pads /tmp/pads_wave_verify.npz
cd ..
# (f) Qwen + LoRA 19.3k and the straight-line floor under the polyline rule (Tables 4-6)
python -m spatial_data.track_a_export score --routes data/track_a/de_pijp_val_plain/routes.jsonl --pred runs/qwen_sft/lora20000_v1/pred_val_v1_n500.jsonl
python -m spatial_data.track_a_export baseline --routes data/track_a/de_pijp_val_plain/routes.jsonl --kind straight --out /tmp/straight.jsonl
python -m spatial_data.track_a_export score --routes data/track_a/de_pijp_val_plain/routes.jsonl --pred /tmp/straight.jsonl
```

The same `score` command on `runs/qwen_sft/lora{1000,5000}_v1/pred_val_v1_n500.jsonl` gives the 1k and 5k rows, and on
`runs/qwen_vanilla/pred{,_4b}_two-pass_n200.jsonl` with the first 200 lines of `routes.jsonl` the zero-shot rows. The
one-shot and cleanup rows of Table 5 and the transfer columns rescore the released drawings:
`drawing_to_polyline.py RUN --kind wave|oneshot --export ../data/track_a/<export> [--crops DIR --store c4_256_val]`
without `--rerun` (the cleanup network itself is not released).

**Result of the check** (fresh clone, release files only, RTX 3060, bf16):

| check | paper | obtained | routes that differ |
|---|---:|---:|---:|
| (a) 256 m search, De Pijp val, cell rule (Table 3) | 77.6 % (78 %) | 77.6 % | 0 / 500 |
| (b) 256 m search, Zuid + Centrum 256 m val (Table 3) | 59.6 % (60 %) | 59.6 % | 0 / 500 |
| (b) 256 m search, house maze 256 m val (Table 3) | 50.8 % (51 %) | 50.8 % | 0 / 500 |
| (c) 512 m search, Zuid + Centrum 512 m val (Section 5.4) | 42.2 % (42 %) | 42.2 % | 0 / 500 |
| (c) 512 m search, house maze 512 m val (Section 5.4) | 39.2 % (39 %) | 39.2 % | 0 / 500 |
| (c) 512 m search before its noise round, house maze 512 m (Section 5.4) | 39.8 % (40 %) | 39.8 % | 0 / 500 |
| (d) one-shot, budget-matched, De Pijp val, cell rule (Section 7.2) | 46.0 % | 46.0 % | 0 / 500 |
| (e) 256 m search, De Pijp val, polyline rule (Table 5) | 76.4 % | 76.4 % | 0 / 500 (drawings and polylines byte-identical) |
| (f) Qwen3-VL-2B + LoRA 19.3k, polyline rule (Tables 4–6) | 49.6 % | 49.6 % | – |
| (f) straight line between the markers (Table 5) | 27.6 % | 27.6 % | – (baseline file byte-identical) |
| (f) LoRA 1k / 5k; zero-shot 2B / 4B two-pass (n = 200) | 21.4 / 31.0 / 2.5 / 24.5 % | 21.4 / 31.0 / 2.5 / 24.5 % | – |

Every route agrees with the published per-route results, including the long 512 m rollouts, so on the same GPU type,
PyTorch build and bf16 the release reproduces the paper exactly. On other hardware or on the CPU a few long routes
can flip (Section 8.6 of the README: 4 of 200 routes, all with 6–9 fronts, between a CPU and a GPU run of the 256 m
network). Scripts fixed for this check: the evaluation scripts used to apply the data, checkpoint and ledger
directories recorded in a run's `meta.json`, i.e. the training machine's paths; they now keep the clone's own, and
`drawing_to_polyline.py` resolves the repo-relative paths of the released drawings.

Not reproducible from the release alone: the De Pijp **test** row of Table 3 (73 %) and every training number (the
bundle holds validation stores only, and the training stores are 2–7 GB); the 48k network before natural noise, the
2-block control, the second natural-noise round, the cleanup network and the 512 m one-shot network (not released;
their published per-route JSONs are in the bundle, and the cleanup and one-shot drawings can be rescored); the Qwen
generations themselves, which need the base model (`models/download_qwen.py`) and the released adapter
(`qwen_fine_tuning.eval_lora`, section 7, 20–30 min per 500 routes); everything before the search rung (one model per
task level, section 5).
