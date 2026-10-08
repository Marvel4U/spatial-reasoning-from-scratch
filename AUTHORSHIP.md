# Authorship

This is Marvin Uhlmann's project. The research direction, the task ladder, every design decision and every training run are his. AI assistants worked in the loop, Claude (Anthropic) and Cursor Composer, and the split is recorded here because it matters for reading the results.

## One-shot rungs (c0–c4, m1; to 27 Sep 2026)

| component | who |
|---|---|
| overarching decisions, project plan, task ladder, what to test and when | Marvin |
| model (`plain_gpt_module/`: the grid vision transformer, relative position bias, conditioning) | Marvin |
| training harness, run ledger, diagnostics suite, experiment specs (`harness/`), and every training run | Marvin, with Cursor Composer support |
| data pipeline, task generators, GPU loaders, path-target builder, probes and analysis scripts (`worldsnap/`, `spatial_data/`, `analysis/`, `docs/plans/perf_prototypes/`) | Claude, reviewed by Marvin |
| reports, plans and this README | drafted by Claude on Marvin's questions, corrected and extended by Marvin |

## The search (C5) and the Qwen track (from 28 Sep 2026)

| component | who |
|---|---|
| the search's interface and its rule (map + scratchpad + state token in, the whole next scratchpad + state token out, nothing computed outside the network), the choice of the wave, the test sequence that found the token-loss cause, what to run and when | Marvin |
| model changes for the search (`plain_gpt_module/`: scratchpad plane, readout token and state-token head, task and token vocabulary) | Marvin |
| the loss: the change-weighted cross-entropy and its β schedule (`plain_gpt_module/wave_pad_ce.py`, `wave_beta_schedule.py`), the token-loss weighting | Marvin |
| training loop and harness changes, run specs (`harness/experiments.py`), and every training run | Marvin |
| the wave data rule and builder (`spatial_data/c5_wave.py`, `c5_trajectories.py`), the teacher's pass schedule (`wave_search_schedule.py`), the loader tasks, pad noise and natural per-pass sampling (`dataset_c4_gpu.py`) | Claude, reviewed by Marvin |
| the stores of the network's own drawings and of cleanup targets, and their mixing into the loader (`c5_own_pads.py`, `c5_cleanup.py`) | Claude, reviewed by Marvin |
| the synthetic maze world (`spatial_data/maze_world.py`) | Claude |
| evaluation and diagnostics for the search: rollout evaluation, animations, timing, the ties diagnostic, the one-shot model under the search's rule, the showcase pictures (`harness/diagnostics/wave_rollout_eval.py`, `wave_rollout_gif.py`, `wave_rollout_timing.py`, `wave_back_ties.py`, `oneshot_route_eval.py`, `showcase_compare.py`, `drawing_to_polyline.py`) | Claude |
| Track A export of routes as map images and polylines, and the polyline scorer (`spatial_data/track_a_export.py`) | Claude |
| the Qwen track code: loading, prompts, coordinate conversion, zero-shot evaluation, the LoRA training loop and adapter evaluation (`qwen_reference/`, `qwen_fine_tuning/`), all its runs, and its report (`docs/reports/TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md`) | Marvin, with Cursor support |
| the search report, memos and plans (`docs/reports/C5_WAVE_SEARCH_2026-10-03.md`, `docs/memo/`, `docs/plans/C5_PLAN.md` and the dated plans), this README | drafted by Claude from Marvin's runs and decisions, reviewed and corrected by Marvin |

Parts of Claude's share were built by Claude sub-agents (Opus and Sonnet models) working to Claude's specifications and reviewed by Claude before Marvin's review. Two of Claude's proposals during the search turned out wrong and are documented as such in the report (§5.3, §5.5).

Every number in the README names the run it comes from; the run ledger (`runs/experiments.json`) holds the configuration of each.
