# Plan 3 Oct 2026: from "it works on De Pijp" to a shippable rung, then natural noise and the maze

Decided 3 Oct (Marvin): do 1–4 below in this order; Claude prepares everything that is not a training run or an
analysis session; Marvin triggers runs and takes part in the test and analysis runs. Track A (Qwen) follows after 4
(`docs/memo/TRACK_A_BACKBONES_2026-10-01.md`, variant A1).

Starting point: `c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k` solves 75 % of 200 De Pijp val routes
free-running (`docs/memo/WAVE_MIX_TOKEN_LOSS_2026-10-02.md`). Training saturates around 25–32k steps on the
teacher-forced metrics (dashboard); whether the free-running success does is measured in step 1.

## 1. The generalisation number (Claude prepares, both read)

- Wave data for the new map's 256 m val/test routes: `python -m spatial_data.c5_wave --store data/amsterdam/zuid_centrum/crops/v4ns/c4_256_val --splits val,test`.
  The 512 m store and the 256 m val store exist already (built 1 Oct); nothing is rebuilt.
- `harness/diagnostics/wave_rollout_eval.py` gets `--crops` / `--store` so a checkpoint can be run on any store.
- Numbers: free-running success on the new map's val routes (and test, once), by length; the same for De Pijp test.
- Success vs training step from the checkpoint series (step 1k, 2k, 5k, 10k, 20k, 48k) on De Pijp val.

## 2. Shippable state (Claude drafts, Marvin reviews)

- `docs/reports/C5_WAVE_SEARCH_2026-10-03.md`: what the rung is, the data rule, the recipe, the numbers of 1, the
  failure curve, what was wrong on the way (from the two memos), limits. Figures: the two animations, the hop blob
  before/after, the success-by-length table.
- README section draft for the public repository (per `docs/plans/PUBLICATION_PLAN.md`); nothing is published
  without Marvin.

## 3. Natural noise = training on the network's own pads (Claude builds, Marvin runs)

- `spatial_data/c5_own_pads.py`: run a checkpoint under the teacher's schedule over the train routes, batched, and
  store the network's pad after every pass (packed bits, one record per route and pass). The pass identity is known
  from the schedule, so the teacher's clean next pad is the target as before.
- Loader: with probability `p` a wave sample's pad in is replaced by the stored own pad of the pass that produces it
  (`config.c5_wave_own_pads = {"path": ..., "p": 0.5}`); the target is unchanged. Val never uses own pads.
  Own pads live on the CPU, packed; the batch gathers what it needs.
- The draw pass trained this way IS the tightening rung on real failure modes: own final pad in, clean path out.
- Specs: (a) continue from the L4 checkpoint with own pads for 16k; (b) from scratch 48k with own pads from the
  start of the second third. Tests for the store, the loader mixing and the alignment of pass identities.
- Read with `wave_rollout_eval.py`; the number that should move is the success on routes of 200 m and more (54 %).

## 4. The maze (Claude builds the world and the pipeline, tinkering together)

- A synthetic "district" in the same files the loaders read (`district_labels_1m.npz` with walls as the building
  class, the crop index and split files, then `c4_routes` and `c5_wave` as for a real district), so that nothing in
  the model or loader changes. Corridors 8–12 m wide (street-like, within the clearance price's range), one or more
  maze styles (perfect maze, braided maze with loops, rooms and doors), 256 m windows.
- Zero-shot: `wave_rollout_eval.py` with the L4 checkpoint on the maze store, success by length; animations.
- If the wave ignores walls or the price, the next step is a short fine-tune with maze routes in the mix (Marvin).

## Order and who

| step | what | who | needs the GPU |
|---|---|---|---|
| 1 | wave data for the new map, eval switches, numbers | Claude | minutes |
| 2 | report and README draft | Claude (draft), Marvin (review) | no |
| 3 | own-pad store, loader, specs, tests; generation of the store | Claude; runs by Marvin | generation minutes; runs hours |
| 4 | maze world, routes, waves, zero-shot eval | Claude; tinkering together | minutes |

Open question for Marvin: for 3, continue from the checkpoint or from scratch first (both specs will exist; (a) is
the cheaper first look).


## Status 4 Oct

1. Done: De Pijp val 500 64 %, test 61 %; Zuid + Centrum val 44 %, test 53 %; free-running success is not saturated
   at 32k (18 % at 20k, 75 % at 48k on the first 200 routes).
2. Drafted with all numbers: `docs/reports/C5_WAVE_SEARCH_2026-10-03.md`, `docs/reports/README_C5_SECTION_DRAFT.md`.
3. Natural noise works: +14 points at home, +16 on the new map, by continuing the 48k network for 16k on its own
   drawings (report §4.6). Round 2 (store from the round-1 checkpoint) is in RUN. The 96k idea is dropped.
4. Maze: thin walls fail zero-shot (4 %: rings through walls never seen); building-thick walls work zero-shot at the
   new-map level (43 %, `data/synthetic/maze_v2_houses`). Parked until the 512 m maps.

Next after round 2: Track A (Qwen3-VL-2B vanilla baseline on the same 500 routes, then A1), the 512 m maps.
