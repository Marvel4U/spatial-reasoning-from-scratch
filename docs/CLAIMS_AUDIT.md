# Claims audit (11 Oct 2026)

Every number in the public paper, `PAPER.md` (named `PAPER.md.md` until 11 Oct; abstract, Tables 1–7, figure captions, Section 8, plus the Section 5–7 text that
Section 8 summarises) and the headline numbers of `docs/reports/C5_WAVE_SEARCH_2026-10-03.md`,
`COMPARISON_TABLE_2026-10-08.md` and `TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md` (public copies on branch
`c5-wave-search`), compared against the evaluation files of the private repository. Every value below was read
programmatically from the file named; nothing was recomputed by rerunning a model. "Rounded" = equal at the paper's
precision. Audit by Claude; the paper itself was not edited by the audit.

**Paths.** Relative to the private repo `spatial_reasoning_LLM_artifact/`. `diag/X/` = `runs/diagnostics/<run X>/`.
`CMP` = `docs/reports/COMPARISON_TABLE_2026-10-08.json` (identical in both repos). `TA` = `data/track_a/de_pijp_val_plain/`.
`QS` = `runs/qwen_sft/`. `QV` = `runs/qwen_vanilla/`. `LED` = `runs/experiments.json` (`summary.n_params`, seeds).

**Run aliases.**

| alias | run id |
|---|---|
| W48 | `c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k` |
| W2 | `c5_r5_wave_nat3_tok01_banneal5_noisep1_L2_48k_g128_store100k` |
| OWN | `c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k` (256 m search of record) |
| OWNB | `c5_r5_wave_nat3_tok01_banneal5_noisep1_ownp50_L4_48k_g128_store100k` |
| R2 | `c5_r5_wave_nat3_tok01_ownp50_r2_L4_16k_g128_store100k` |
| Z5A | `c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512` (Figure 1) |
| Z5B | `c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512` |
| CU3 / CU1 | `c5_r5_wave_nat3_tok01_cleanup100_dw{3,1}_ownp50_r2_L4_16k_g128_store100k` |
| CU512 | `c5_r5_wave_nat3_tok01_cleanup100_dw3_ownp50_r2_L4_16k_g256_b64_zc512` |
| OS / OS512 | `c5_os_bld_det_L4_64k_g128_store100k` / `c5_os_bld_det_from64k_L4_32k_g256_b64_zc512` |
| OS16 | `c5_r0a_bld_det_L4_16k_store100k_17ch_g128` (the "earlier one-shot run", 16k) |

**Keys.** Rollout files: `free.all.final_success`, bands `free["n_goal 2-3" | "4-5" | "6+" | "6-8" | "9-12" | "13+"].final_success`;
one-shot files the same under `oneshot`. Qwen files: `report.solved_share`, `report.by_n_goal`, `report.failure_share`.
CMP: `rows[method].sets[set].solved_share | by_n_goal | failure_share | first200_solved_share`.

**Route sets.** DP = De Pijp (`de_pijp/crops/v4/c4_100k`), ZC = Zuid + Centrum (`zuid_centrum/crops/v4ns/c4_256_val`),
maze = `maze_v2_houses` (houses), maze v1 = thin walls, ZC512 = `zuid_centrum/crops/v4ns/c4_512_20k` val.
Band sizes (2–3 / 4–5 / 6+): DP val 112 / 141 / 247; DP val first 200: 72 / 65 / 63; DP test 122 / 168 / 210;
ZC val 132 / 163 / 205; maze val 205 / 141 / 154; ZC512 val 43 / 68 / 132 / 165 / 92 (2–3 / 4–5 / 6–8 / 9–12 / 13+);
maze v1 val 42 / 34 / 124 (n = 200). All match the PAPER.md and reports.

**Seeds.** Every grid-model run: seed 4 (LED). Qwen LoRA runs: training seed 42 (`QS/lora*_v1/run_config.json`).
Zero-shot Qwen: no training seed.

## 1. Matrix

Status: **=** exact, **≈** rounded, **✗** mismatch, **?** not found in any file, **~** consistent but indirect.

### 1.1 Abstract and Figure 1

| claim (as written) | PAPER | run | file | key | file value | rule | routes | status |
|---|---|---|---|---|---|---|---|---|
| "2–4 blocks and 5–9 M parameters" (ladder) | 5–9 M | M1/C4 ladder | M1 report §3, Fig 4 caption | n_params | 4.8–8.7 M (Fig 4), 5.1 / 8.7 M (M1) | – | – | ≈ (report level) |
| "8.9 M-parameter network" | 8.9 M | OWN | LED | summary.n_params | 8,856,409 | – | – | ≈ |
| "solves 73–78 % of held-out De Pijp routes" | 73 / 78 | OWN | diag/OWN/wave_rollout_eval_test.json, _val500.json | free.all.final_success | 0.728 / 0.776 | cell | DP test 500 / val 500 | ≈ |
| "60 % in a district never seen in training" | 60 | OWN | diag/OWN/wave_rollout_eval_zuid_centrum_c4_256_val_val.json | free.all.final_success | 0.596 | cell | ZC val 500 | ≈ (0 of 500 windows touch De Pijp, verified below) |
| "51 % in a synthetic house maze" | 51 | OWN | diag/OWN/wave_rollout_eval_maze_v2_houses_c4_256_val_val.json | free.all.final_success | 0.508 | cell | maze val 500 | ≈ |
| "the search solves 76.4 %" | 76.4 | OWN | CMP; TA/preds/OWN_wave_report.json | polyline_simplified.solved_share | 0.764 (382/500) | polyline | DP val 500 | = |
| "Qwen3-VL-2B with LoRA on 19.3k routes 49.6 %" | 49.6 | lora20000_v1 | QS/lora20000_v1/eval_val_v1_n500.json | report.solved_share | 0.496 (248/500) | polyline | DP val 500 | = |
| "19.3k routes" | 19.3k | lora20000_v1 | QS/lora20000_v1/run_config.json | len(train_routes) | 19,309 (+20 dev) | – | – | ≈ |
| "one-pass network at the same training budget 37.4 %" | 37.4 | OS | TA/preds/OS_oneshot_report.json | polyline_simplified.solved_share | 0.374 (187/500) | polyline | DP val 500 | = |
| same budget (64k vs 48k + 16k) | 64k | OS / W48+OWN | LED | steps | 64k / 48k + 16k | – | – | = |
| Fig 1: "4-block, 8.9 M-parameter" | 8.9 M | Z5A | LED; release `c5_wave_search_512m_hero.pt` | n_params | 8,856,409 | – | – | ≈ |
| Fig 1: "431 m walking route" | 431 m | Z5A | data/synthetic/maze_v2_houses/crops/v4ns/c4_512_val/c5_wave_val.npz | goal_m[8] | 431.27 m; solved, stop = done (diag/Z5A/wave_rollout_eval_maze_v2_houses_c4_512_val_val.json, free_per_route[8]) | cell | maze 512 val route 8 | = |
| Fig 1: "in 26 passes" | 26 | Z5A | docs/figures/…route008_431m.gif | frame count | 28 frames (every other GIF has passes + 2 frames) | – | – | ~ (no pass count per route in any eval file) |

### 1.2 Table 3 and Section 5 text (cell rule, single seed 4)

| claim | PAPER | run | file | key | file value | rule | routes | status |
|---|---|---|---|---|---|---|---|---|
| T3 DP val all / 2–3 / 4–5 / 6+ | 78 / 96 / 89 / 63 | OWN | diag/OWN/wave_rollout_eval_val500.json | free.* | 77.6 / 95.5 / 88.7 / 63.2 | cell | DP val 500 | ≈ |
| T3 DP test | 73 / 97 / 85 / 49 | OWN | diag/OWN/wave_rollout_eval_test.json | free.* | 72.8 / 96.7 / 85.1 / 49.0 | cell | DP test 500 | ≈ |
| T3 ZC val | 60 / 88 / 61 / 40 | OWN | diag/OWN/…zuid_centrum_c4_256_val_val.json | free.* | 59.6 / 87.9 / 61.3 / 40.0 | cell | ZC val 500 | ≈ |
| T3 house maze | 51 / 84 / 51 / 6 | OWN | diag/OWN/…maze_v2_houses_c4_256_val_val.json | free.* | 50.8 / 84.4 / 51.1 / 5.8 | cell | maze val 500 | ≈ |
| before natural noise: 64 / 61 / 44 % | 64 / 61 / 44 | W48 | diag/W48/wave_rollout_eval_{val500,test,zuid_centrum_c4_256_val_val}.json | free.all.final_success | 0.640 / 0.610 / 0.442 | cell | DP val / DP test / ZC val, 500 each | ≈ |
| route 91: 191 m, 11 passes | 191 m, 11 | W48 | TA/routes.jsonl | teacher_m, n_goal | 191.1 m, 5 fronts (= 5 + 5 + 1 passes; GIF 13 frames) | – | DP val | = |
| route 363: 291 m, 17 passes | 291 m, 17 | W48 | TA/routes.jsonl | teacher_m, n_goal | 290.9 m, 8 fronts (= 17 passes; GIF 19 frames) | – | DP val | ≈ |
| timing: one pass 2.4 ms | 2.4 | W48 | diag/W48/wave_rollout_timing.json | forward_ms_batch1 | 2.4 | – | – | = |
| timing: route 57 ms, 10.3 passes | 57 / 10.3 | W48 | same | route_ms_mean, passes_mean | 57.4 / 10.29 | – | 55 routes, mean 163 m (shorter than the val median 195 m) | ≈ |
| 2 blocks, 5.3 M parameters | 5.3 M | W2 | LED | summary.n_params | 5,304,769 | – | – | ≈ |
| TF IoU within 4 m fwd 0.87 vs 0.92 | 0.87 / 0.92 | W2 / W48 | diag/{W2,W48}/eval.jsonl (step 47999) | iou_dilated_wave_fwd | 0.866 / 0.920 | TF-IoU 4 m | 1,500 val passes | ≈ |
| 2 blocks solve 4 %, 4 blocks 75 % (first 200) | 4 / 75 | W2 / W48 | diag/{W2,W48}/wave_rollout_eval.json | free.all.final_success | 0.040 / 0.750 | cell | DP val first 200 | = |
| 18 % at 20k, 75 % at 48k | 18 / 75 | W48 | diag/W48/wave_rollout_eval_step20000.json, wave_rollout_eval.json | free.all.final_success | 0.180 / 0.750 | cell | DP val first 200 | = |
| false cells on the backward hop 180 → 98 | 180 / 98 | `…bwd2_ce_noisep1_cosine{,_tok01}_L2_8k_g128_store40k` | diag/…/wave_back_ties_wave_back.json | false_cells_mean, n | 178.8 (**n = 491**) / 97.7 (n = 299) | per-sample false cells, p ≥ 0.5 | val samples | ✗ (179, and different denominators) |
| token accuracy 0.998 / 1.00 vs 0.96 / 0.96 | as written | same two runs | diag/…/eval.jsonl (step 7999) | c5_tok_acc_wave_back / _origin | 0.998 / 1.0 vs 0.96 / 0.96 | tok-acc | val samples | = |
| grid loss over 16,384 cells | 16,384 | – | 128 × 128 grid | – | 16,384 | – | – | = |
| natural noise store: 1.16 M passes, 10.5 min | 1.16 M / 10.5 | W48 | data/amsterdam/de_pijp/crops/v4/c4_100k/c5_own_pads_train_stats.json | passes, seconds_rollout | 1,159,256 / 627.6 s (10.5 min; 10.8 incl. load + save) | – | 100k train routes | ≈ |
| +14 / +12 / +15 points | 14 / 12 / 15 | W48 → OWN | val500 / test / ZC val | free.all.final_success | +13.6 / +11.8 / +15.4 | cell | 500 each | ≈ |
| teacher-forced numbers did not move | – | W48 / OWN | eval.jsonl | iou_dilated_wave_fwd / _bwd | 0.920 / 0.961 → 0.924 / 0.958 | TF-IoU 4 m | 1,500 passes | = |
| from scratch 3–5 points worse | 3–5 | OWNB vs OWN | diag/OWNB/wave_rollout_eval_*.json | free.all.final_success | 74.8 / 69.4 / 54.8 vs 77.6 / 72.8 / 59.6 (−2.8 / −3.4 / −4.8) | cell | 500 each | ≈ |
| second round 80 / 71 / 60 % | 80 / 71 / 60 | R2 | diag/R2/wave_rollout_eval_{val500,test,zuid…}.json | free.all.final_success | 80.0 / 71.4 / 60.4 | cell | 500 each | ≈ |
| 512 m: 42 % | 42 | Z5B | diag/Z5B/wave_rollout_eval_val500.json | free.all.final_success | 0.422 | cell | ZC512 val 500 | ≈ |
| 512 m strata 91 / 71 / 53 / 30 / 4 % | as written | Z5B | same | n_goal 2-3 … 13+ | 90.7 / 70.6 / 53.0 / 30.3 / 4.3 | cell | ZC512 val 500 (43 / 68 / 132 / 165 / 92) | ≈ |
| 512 m house maze 40 % before, 39 % after | 40 / 39 | Z5A / Z5B | diag/…/wave_rollout_eval_maze_v2_houses_c4_512_val_val.json | free.all.final_success | 0.398 / 0.392 | cell | maze 512 val 500 | ≈ |
| thin-wall maze 4 % | 4 | W48 | diag/W48/wave_rollout_eval_maze_v1_c4_256_val_val.json | free.all.final_success | 0.040 | cell | maze v1 val **200** | ≈ |
| cleanup keeps 70 % | 70 | CU3 | diag/CU3/wave_rollout_eval_val500.json | free.all.final_success | 0.702 | cell | DP val 500 | ≈ |
| clean drawings 14 % → 42 % | 14 / 42 | OWN / CU3 | diag/OWN/wave_rollout_eval_cleanliness_cpu.json / diag/CU3/wave_rollout_eval_val500.json | free.all.final_clean | 0.135 (**first 200, CPU**) / 0.416 (500, GPU) | cell + "clean" | **200 vs 500** | ✗ denominator (CU3 on the same first 200: 0.590) |
| "a trade of 8 points of success" | 8 | OWN / CU3 | val500 files | free.all.final_success | 77.6 − 70.2 = 7.4 | cell | DP val 500 | ✗ (7) |
| 512 m cleanup 18 % solved, 30 % DONE | 18 / 30 | CU512 | diag/CU512/wave_rollout_eval_val500.json | final_success, stops.done | 0.184 / 148 of 500 = 29.6 % | cell | ZC512 val 500 | ≈ |

### 1.3 Tables 4–6 and Section 6–7 text

| claim | PAPER | run | file | key | file value | rule | routes | status |
|---|---|---|---|---|---|---|---|---|
| T4 2B zero-shot: 2.5 % / 6+ 0 % / endpoints | 2.5 / 0 | qwen_vanilla 2B | QV/eval_ab_n200.json | modes.two-pass.report | 0.025 (5/200) / 0.0 / wrong endpoints 0.92 | polyline | DP val first 200 | = |
| T4 4B zero-shot: 24.5 % / 6+ **2 %** | 24.5 / 2 | qwen_vanilla 4B | QV/eval_ab_4b_n200.json | modes.two-pass.report | 0.245 (49/200) / **0.0159** (1/63) | polyline | DP val first 200 | ≈ for 24.5; 6+ ≈ at integer precision but the column uses one decimal elsewhere: should read **1.6 %** (CMP and Track A report say 1.6 %) |
| T4 LoRA 1k: 21.4 % / 1.2 % | 21.4 / 1.2 | lora1000_v1 | QS/lora1000_v1/eval_val_v1_n500.json | report | 0.214 / 0.0121 | polyline | DP val 500 | = |
| T4 LoRA 5k: 31.0 % / 7.7 % | 31.0 / 7.7 | lora5000_v1 | QS/lora5000_v1/eval_val_v1_n500.json | report | 0.310 / 0.0769 | polyline | DP val 500 | = |
| T4 LoRA 19.3k: 49.6 % / 28.3 % / crosses 36 % | 49.6 / 28.3 / 36 | lora20000_v1 | QS/lora20000_v1/eval_val_v1_n500.json | report | 0.496 / 0.2834 / 0.364 | polyline | DP val 500 | = |
| T4 caption: straight-line floor 27.6 % | 27.6 | baseline | CMP rows[straight].sets.de_pijp | solved_share | 0.276 on 500 (**0.505 on the first 200**) | polyline | DP val 500 | = (but see over-claim 6) |
| 6.2: 81 / 62 / 28 % by fronts; endpoint off on 11 % | 81 / 62 / 28 / 11 | lora20000_v1 | same | by_n_goal, failure_share | 81.25 / 61.7 / 28.3 / 11.4 | polyline | DP val 500 | ≈ |
| T5 straight line | 27.6 / 73 / **36** / 2 / 28 / 21 | baseline | CMP rows[straight] | sets.* | 27.6 / 73.2 / **35.46** / 2.4 / 27.8 / 21.2 | polyline | 500 each | ✗ 4–5 column: 35.46 → **35 %** |
| T5 Qwen 2B zero-shot (200) | 2.5 / 6 / 2 / 0 | 2B | CMP | sets.de_pijp | 2.5 / 5.56 / 1.54 / 0.0 | polyline | DP val first 200 (72 / 65 / 63) | ≈ |
| T5 Qwen 4B zero-shot (200) | 24.5 / 35 / 35 / 2 | 4B | CMP | sets.de_pijp | 24.5 / 34.7 / 35.4 / 1.6 | polyline | DP val first 200 | ≈ |
| T5 LoRA 1k / 5k / 19.3k | 21.4/58/28/1, 31.0/74/38/8, 49.6/81/62/28 | lora*_v1 | CMP | sets.de_pijp | 21.4/58.0/27.7/1.2, 31.0/74.1/37.6/7.7, 49.6/81.3/61.7/28.3 | polyline | DP val 500 | ≈ |
| T5 one-shot | 37.4 / 83 / 40 / 15 / 31 / 16 | OS | CMP | sets.* | 37.4 / 83.0 / 39.7 / 15.4 / 31.4 / 16.4 | polyline | DP val, ZC val, maze val, 500 each | ≈ |
| T5 wave search | 76.4 / 93 / 87 / 63 / 57 / 44 | OWN | CMP | sets.* | 76.4 / 92.9 / 87.2 / 62.8 / 56.8 / 44.0 | polyline | 500 each | ≈ |
| T5 wave + cleanup | 46.2 / 82 / 54 / 26 / 40 / 32 | CU3 | CMP | sets.* | 46.2 / 82.1 / 53.9 / 25.5 / 40.2 / 32.2 | polyline | 500 each | ≈ |
| T6 all nine rows (failure mix) | as printed | as T5 | CMP | failure_share | identical to one decimal in every cell (e.g. 2B ZS 92.0 / 2.0 / 3.5; 4B ZS 13.5 / 51.5 / 10.5; one-shot 53.4 / 9.2; wave 21.0 / 2.0 / 0.6; cleanup 19.2 / 34.0 / 0.6) | polyline | as T5 | = |
| 7.2: 77.6 % cell vs 76.4 % polyline | 77.6 / 76.4 | OWN | CMP | cell_rule_existing / solved_share | 0.776 / 0.764 | cell / polyline | DP val 500 | = |
| 7.2: valid in 382 of 395 connected cases | 382 / 395 | OWN | TA/preds/OWN_wave_report.json | n_connected, polyline_simplified.solved | 395 / 382 | polyline | DP val 500 | = |
| 7.2: cleanup loses 24 points | 24 | CU3 | CMP | cell 0.702 vs polyline 0.462 | 24.0 | both | DP val 500 | = |
| 7.2: 1.5 % of chain cells touch a building vs 0.2 % | 1.5 / 0.2 | CU3 / OWN | – | – | not in any JSON (`*_pl.json` hold `final_building_share` = cells *entirely* building: 0.0019 / 0.0004) | – | – | ? |
| 7.2: one-shot "up to about 3 points higher" | ~3 | OS | – | – | rescue counts (26 / 55 raw) are in the report text only | – | – | ? |
| 7.2: one-shot vs search, cell rule 46 / 78, 35 / 60, 18 / 51 | as written | OS / OWN | diag/OS/oneshot_route_eval{,_zuid…,_maze…}.json; OWN files | final_success | 46.0 / 77.6, 34.6 / 59.6, 17.8 / 50.8 | cell | DP val, ZC val, maze val, 500 each | ≈ |
| 7.2: 16k → 64k did not move it | – | OS16 / OS | diag/OS16/oneshot_route_eval_val500.json | oneshot.all.final_success | 0.462 → 0.460 (bands moved: 79 / 62 / 23 → 85 / 50 / 26) | cell | DP val 500 | = (overall) |
| 7.2: 512 m one-shot 19 % vs 42 %; maze 14 % vs 40 % | as written | OS512 / Z5B; OS512 / Z5A | diag/OS512/oneshot_route_eval.json, …maze_v2_houses_c4_512_val_val.json | final_success | 0.194 / 0.422; 0.136 / 0.398 (Z5B: 0.392) | cell | ZC512 val, maze 512 val | ≈ (the maze comparison uses the pre-noise network) |
| Fig 8: route 57, 411 m, 11 fronts, longest both solved | as written | lora20000 / OWN | TA/routes.jsonl; per-route verdicts | teacher_m, n_goal | 411.2 m, 11; longest of the 125 first-200 routes both solve | polyline | DP val first 200 | = |
| Fig 8: route 188, 143 m, 4 fronts; search fails, cleanup solves | as written | OWN / CU3 | TA/preds/*_report.json routes[188] | reason | 142.5 m, 4; wave "unparseable", cleanup solved | polyline | – | ≈ |
| Fig 8: route 192, 350 m, 9 fronts; Qwen crosses, one-shot has gaps | as written | – | same | – | 350.4 m, 9; LoRA "crosses a building"; one-shot "unparseable" | polyline | – | ≈ |
| Fig 8: "Solved lengths of 0.95–0.98 ×" | 0.95–0.98 | – | same | length_ratio | wave 0.948 / 0.965, LoRA 0.977 / 0.985, cleanup 0.947 / **0.891** / 0.961 | polyline | – | ✗ (cleanup route 188 is 0.89 ×) |
| Table 7 sizes 35.4 / 35.4 / 35.5 / 35.4 / 25.9 / 37.3 / 0.2 MB and run ids | as written | – | `~/Github/spatial_reasoning_public_release/c5/*` | file size; `source_run_id` in each .pt | 35,449,698 / 35,449,698 / 35,450,363 / 35,384,655 / 25,935,378 / 37,320,309 / 210,662 B; run ids match | – | – | ≈ |

### 1.4 Section 8

| claim | PAPER | run | file | key | file value | rule | routes | status |
|---|---|---|---|---|---|---|---|---|
| 8.2 shortcut 0.41 IoU on ten configurations | 0.41 / 10 | M1 runs | M1 report §3, §7 | val IoU | 0.402–0.428 in ≥ 10 runs (harness accounting; 0.29 in the probe accounting) | IoU | val crops | ≈ (report level) |
| 8.2 / 8.4 2-block 4 % vs 75 % | 4 / 75 | W2 / W48 | wave_rollout_eval.json | final_success | 0.040 / 0.750 | cell | DP val first 200 | = |
| 8.2 cleanup 46.2 % vs 76.4 % | 46.2 / 76.4 | CU3 / OWN | CMP | solved_share | 0.462 / 0.764 | polyline | DP val 500 | = |
| 8.3 escapes in 10 of 11 runs vs about half | 10 / 11 | M1 mix runs | M1 report §9 | escapes | ⅓ mix 8 / 8 + 80/10/10 2 / 3 = 10 / 11; single task 1 of 3 and 1 of 2 seeds | IoU | – | = (report level) |
| 8.4 natural noise +12–15 points | 12–15 | W48 → OWN | val500 / test / ZC | final_success | +13.6 / +11.8 / +15.4 | cell | 500 each | ≈ |
| 8.4 "at a third of the compute" | 1/3 | OWN vs OWNB | LED | steps, training_time_s | 16k vs 48k steps; 3,071 s vs 9,229 s | – | – | = |
| 8.5 21.4 % → 49.6 %; 6+ fronts 1.2 % → 28.3 % | as written | lora1000 / lora20000 | QS/…/eval_val_v1_n500.json | report | 0.214 → 0.496; 0.0121 → 0.2834 | polyline | DP val 500 | = |
| 8.6 75 % of first 200, 64 % of all 500; 78 % / 73 % | as written | W48 / OWN | wave_rollout_eval.json, _val500.json, _test.json | final_success | 0.750 / 0.640; 0.776 / 0.728 | cell | DP val 200 / 500; DP test 500 | ≈ |
| 8.6 seed 4 for every search configuration | 4 | all C5 runs | LED | seed | 4 | – | – | = |
| 8.6 binomial s.e. about 2 points at 500, 2–4 per band | 2 / 2–4 | OWN | val500 bands | – | 1.9 at 77.6 %; 1.8–3.8 over the bands of Table 3 | – | – | = |
| 8.6 CPU vs GPU: 177 vs 179 solved, 4 routes differ, all 6–9 fronts | as written | OWN | diag/OWN/wave_rollout_eval_cleanliness_cpu.json vs _val500.json (first 200) | free_per_route[].solved | 177 vs 179; routes 71, 179, 192, 197 with 8, 8, 9, 6 fronts | cell | DP val first 200 | = |
| 8.6 cleanup at 512 m 18 % vs 42 % | 18 / 42 | CU512 / Z5B | val500 | final_success | 0.184 / 0.422 | cell | ZC512 val 500 | ≈ |
| 8.6 c4 seeds 0.013 apart, no difference below 0.02 | 0.013 / 0.02 | c4 N0 seeds | C4 report §7.1 | dilated IoU | 0.013 | IoU | – | = (report level) |

### 1.5 Report headline numbers

**`C5_WAVE_SEARCH_2026-10-03.md`** (all cell rule unless stated; W48 / W2 unless stated).

| claim | report | file | file value | status |
|---|---|---|---|---|
| §3 parameters 8,856,409 / 5,304,769; training 9,162 / 6,302 s | as written | LED | identical | = |
| §4.1 teacher-forced table (12 cells per depth) | e.g. fwd 0.757 / 0.920 / 0.888 / 1.000 / 0.0030 / 0.988 | diag/W48,W2/eval.jsonl step 47999 | identical to 3 dp in every cell; losses 0.0108 / 0.0076, 0.0233 / 0.0132 | = |
| §4.2 per-pass IoU 4 m and token acc., L4 and L2; 75.0 % / 2.5 % | as written | wave_rollout_eval.json `scheduled` | identical (first 0.991 / 0.954 … draw 0.810 / 0.331) | = |
| §4.3 DONE 98.0 / 83.0, tokens 83.5 / 17.5, passes 10.34 / 11.58 vs 10.16, connected 76.0 / 4.0, solved 75.0 / 4.0, solved + DONE 74.5 / 3.5, length 0.994 / 0.995, IoU 0.807 / 0.383 | as written | wave_rollout_eval.json `free.all` | identical | = |
| §4.3 bands 91.7 / 76.9 / 54.0 (L4), 11.1 / 0 / 0 (L2) | as written | same | identical | = |
| §4.3 500-route table: 64.0 / 61.0 / 44.2 / 53.2, bands, DONE, tokens, per-pass IoU | as written | W48 val500, test, zuid val, zuid test | identical at printed precision | = (but ZC test, see over-claim 1) |
| §4.3 "49 % with 6+ fronts against 32 %" | 49 / 32 | per-route n_goal | 247 / 500 = 49.4; 63 / 200 = 31.5 | ≈ |
| §4.3 new map costs "4–8 points on the short routes and 15–30 on the long ones" | 4–8 / 15–30 | W48 bands | short: 7.8 (val), 5.2 (test); long (6+): 30.8 (val), **2.2 (test)** | ✗ for the test split |
| §4.4 checkpoint series 1.5 / 5.5 / 2.0 / 18.0 / 75.0 %, tokens, lit cells | as written | wave_rollout_eval_step*.json | identical | = |
| §4.5 timing table | as written | wave_rollout_timing.json | identical | = |
| §4.6 natural-noise table (incl. lit cells 339 / 585 / 572, TF 0.924 / 0.958, 0.898 / 0.939) | as written | OWN, OWNB files | identical at printed precision | ≈ |
| §4.6 store: 1.16 M passes, 2.3 GB, 10.5 min, IoU 0.48 | as written | c5_own_pads_train_stats.json | 1,159,256; 2,272.9 MB; 627.6 s; 0.4757 | ≈ |
| §4.7 2.6 % (256 net at 512), 32 %, 42 %, bands, maze 40 / 39 %, fwd 0.88 / bwd 0.93 | as written | diag/OWN/…zuid_centrum_c4_512_20k_val.json; Z5A, Z5B files; Z5B eval.jsonl | 0.026; 0.316; 0.422; 90.7 / 64.7 / 33.3 / 17.6 / 2.2 and 90.7 / 70.6 / 53.0 / 30.3 / 4.3; 0.398 / 0.392; 0.884 / 0.928 | ≈ |
| §4.7 routes to 1.1 km, up to 28 fronts | as written | zuid_centrum/…/c4_512_20k/c5_wave_val_stats.json | goal_m_max 1,105 m; passes_at_40m_max 28 | = |
| §4.8 cleanup table (58 / 70; 42 / 47, 52 / 42; lit 365 / 441; clean 42 %, solved + clean 38 %) | as written | CU1, CU3 files | identical at printed precision | ≈ |
| §4.8 "before cleanup: clean 14 %, solved and clean 14 %" | 14 / 14 | diag/OWN/wave_rollout_eval_cleanliness_cpu.json | 0.135 / 0.135 on the **first 200 routes, CPU** | ✗ denominator (column mixes 200 and 500) |
| §4.8 teacher's schedule 74 % → 56 % (weight 1.0) | 74 / 56 | diag/CU1/wave_rollout_eval_sched500.json | origin_success 0.738 / final_success 0.56 | ≈ |
| §4.8 512 m: 18 %, 30 % DONE, 61 % own chains | as written | CU512 val500; c5_cleanup_train_r2_stats.json | 0.184; 148 / 500; share_own_chain 0.613 | ≈ |
| §4.9 one-shot vs search table (5 rows, bands) | as written | OS, OS512, OWN, Z5B, Z5A files | identical at printed precision; the maze-512 search row is Z5A (pre-noise), the ZC512 row Z5B | ≈ |
| §4.9 "46 % both" (16k and 64k) | 46 / 46 | OS16, OS | 0.462 / 0.460 | ≈ |
| §5.5 counter tests: reference 0.68 (0.65) / 12.5 (13) / 95 % (94 %) | as written | `…wave_back_L2_8k…`, `…wave_back_ref_rerun…` wave_back_ties_wave_back.json | 0.675 (0.648) / 12.5 (13.0) / 94.9 % (93.8 %) | ≈ |
| §5.5 C: 0.88 / 68 / 83 % | as written | – | closest: `…back_ce_noisep1…` 0.874 / 71.1 / 79.6 % (n = 491); `…back_ce_noisep1_lrflat…` 0.888 / 78.1 / 80.5 % | ? |
| §5.5 E, token weight 1.0: 0.73 / 180 / 54 % | as written | `…bwd2_ce_noisep1_cosine_L2_8k…` | 0.719 / 178.8 / 51.8 % (**n = 491**, the others n = 299) | ✗ |
| §5.5 E, 0.1: 0.82 / 98 / 74 %; E, 0: 0.86 / 88 / 77 % | as written | `…_tok01…`, `…_tok0…` | 0.823 / 97.7 / 73.6 %; 0.861 / 88.1 / 76.7 % | ≈ |
| §5.5 hypothesis 1: "~7 false cells per sample, 23 % of the failing run's 167" | 7 / 23 % / 167 | `…bwd2_banneal5_noisep1_L4…` | false_cells_mean 166.8; tie + tie2 shares 0.124 + 0.111 = 23.5 %, i.e. ~39 cells, not 7 | ✗ (internally inconsistent) |
| §5.5 / §5.6 backward reruns 8.5 / 14.5, 12.5 / 9.0, 21.0 / 19.0, 0.5 / 0.5 | as written | [B] files | not re-read in this audit | – |
| §6 failure shares 8 / 23 / 46 %; 94 % of 6+ stop with DONE | as written | W48 wave_rollout_eval.json | 8.3 / 23.1 / 46.0; 0.937 | ≈ |
| §7 thin-wall maze: "500 routes per split" | 500 | W48 maze_v1 files | `routes` = **200** per split | ✗ |
| §7 maze v1 4 % / 3 %, 19 % / 13 % on 2–3 fronts, "0 % beyond" | as written | same | 4.0 / 3.0; 19.0 / 12.8; test 6+ = **0.8 %** (1 route) | ✗ for "0 % beyond" on test |
| §7 first pass IoU 0.36, later 0.17, tokens 0.82; controls 5–6 %, 3–6 m walls 1 % | as written | W48 maze_v1 val (`scheduled`), maze_v1_g1/g2/w3 | 0.35 / 0.17 / 0.818; 6.0 / 5.0; 1.0 | ≈ (0.35 at 2 dp) |
| §7 s.e. about 3 points at 200 routes | 3 | – | 3.06 at 75 % | = |

**`COMPARISON_TABLE_2026-10-08.md`**: every cell of §2 (incl. the "first 200" column: 50.5 / 2.5 / 24.5 / 37.5 / 50.5 /
64.5 / 52.0 / 89.0 / 64.5), §3 and the §4 table (cell 46.0 … 42.2; polyline 37.4 … 32.2; raw chain 35.6 … 29.6; not
connected 267 / 325 / 411 / 105 / 183 / 214 / 96 / 183 / 221) equals `CMP` and the `TA`-family `*_report.json` files
(=). Route-by-route counts 7 / 388, 43 / 230, 120 / 351 and "1 for the wave" equal `cell_vs_polyline` (=). Rule
differences −1.2 / −2.8 / −6.8, −8.6 and −24 (=). Not found in any file (?): the chain-cell building shares
1.5 % / 0.2 % / "0.010" (the last has no unit; probably 1.0 %) and the conversion-sensitivity rescue counts of §4.

**`TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md`**

| claim | report | file | file value | status |
|---|---|---|---|---|
| §2.1 2B two-pass 2.5 % (5/200), 92 / 2 / 3.5 %; oracle 8.0 % (16/200), 79 / 11 / 2 %; 4B two-pass 24.5 % (49/200), 13.5 / 51.5 / 10.5 %; oracle 27.0 % (54/200), 4.5 / 54.5 / 14 % | as written | QV/eval_ab_n200.json, eval_ab_4b_n200.json | identical | = |
| §2.2 "~10× two-pass solve rate" | 10× | same | 24.5 / 2.5 = 9.8 | ≈ |
| §2.2 "~half of errors are chord through buildings" | half of errors | QV/eval_ab_4b_n200.json | 51.5 % of **routes**; 103 / 151 = 68 % of failures | ✗ |
| §2.2 4B "near the straight-line floor (~28 %)" | 28 % | CMP straight `first200_solved_share` | floor on the same 200 routes = **50.5 %** | ✗ |
| §3.1 19,309 train + 20 dev, 14,483 steps; r 16, α 32, lr 2e-4, batch 2 × 2, 3 epochs | as written | QS/lora20000_v1/run_config.json, training_state.json | identical | = |
| §3.3 table (1k / 5k / 19.3k and both zero-shot rows) | as written | QS/*/eval_val_v1_n500.json; QV | identical at printed precision (4B "crosses 52 %" = 51.5) | ≈ |
| §3.3 1k: 37.5 % on val 200 vs 21.4 % on val 500 | 37.5 / 21.4 | QS/lora1000_v1 per route | 75 / 200; 107 / 500 | = |
| §3.3 dev-20 60 % (12/20); 5k ~35 % | 60 / 35 | QS/*/dev_eval.jsonl, last line | 0.60 / 0.35 | = |
| §3.4 "~+10 pts per ~2× data in the 1k→5k→full band" | 10 per 2× | eval files | 1k → 5k: +9.6 for 5× (≈ 4 per doubling); 5k → 19.3k: +18.6 for 3.9× (≈ 9.5 per doubling) | ✗ |
| §3.4 "4B vanilla (~27 % on val 200)" | 27 | QV/eval_ab_4b_n200.json | 27.0 % is the **oracle** mode; two-pass 24.5 % | ✗ |
| §3.4 endpoints 39 % → 11 %; crosses 36–47 % | as written | QS files | 0.39 → 0.114; 0.364–0.47 | ≈ |
| §1 "~78 % De Pijp val 500"; "wave has not yet been rescored" | 78 | OWN val500 | 0.776; the rescoring now exists (comparison report) | ≈; stale sentence |

### 1.6 "Never-seen district" check

Zuid + Centrum's district raster (`zuid_centrum/crops/v4ns/district_labels_transforms.json`, RD x 118,724–123,528,
y 482,885–487,926) contains the whole De Pijp extent (x 120,710–122,494, y 484,227–486,020). Windows were tested for
overlap with that extent from the stores' `origin` (x, y in district metres):

| store | split | window | overlap De Pijp |
|---|---|---|---:|
| `c4_256_val` | val | 256 m | **0 of 500** |
| `c4_256_val` | test | 256 m | **33 of 500** |
| `c4_512_20k` | val | 512 m | 0 of 500 |
| `c4_512_20k` | test | 512 m | 42 of 500 |
| `c4_512_20k` | train | 512 m | 7,139 of 20,000 |

Every PAPER.md number for the "never-seen district" is on ZC **val** (0 overlap): supported. The C5 report's ZC **test**
numbers (53.2 % W48; OWN 74.0 %, not reported) include the 33 overlapping windows; on the 467 clean routes they are
52.7 % (W48) and 73.7 % (OWN), on the 33 overlapping ones 60.6 % and 78.8 %. The 512 m networks were trained on
ZC512 train windows, 36 % of which overlap De Pijp; that is consistent with "trained only on Amsterdam" and
"later two districts", but those networks have no never-seen-district claim.

## 2. Mismatches (correct value and where to fix)

1. **PAPER.md Table 5, straight line, 4–5 fronts: 36 % → 35 %** (CMP 0.3546; double rounding via 35.5).
2. **PAPER.md Table 4, Qwen3-VL-4B zero-shot, 6+ fronts: 2 % → 1.6 %** (1 of 63; the column prints one decimal for every other row; CMP and the Track A report say 1.6 %). The integer "2 %" in Table 5 is fine.
3. **PAPER.md §5.4, "triples the share of clean final drawings (…) from 14 % to 42 %"**: 14 % (0.135) is from `wave_rollout_eval_cleanliness_cpu.json`, the first 200 routes on CPU; 42 % (0.416) is on all 500 on GPU. Same routes: first 200, 13.5 % → 59.0 % (GPU, CU3); for all 500 the pre-cleanup clean share was never measured. Same mix in C5 report §4.8 table ("clean 14 %, solved and clean 14 %").
4. **PAPER.md §5.4, "a trade of 8 points of success" → 7 points** (77.6 → 70.2 = 7.4). Same in C5 report §4.8.
5. **PAPER.md §5.3 and C5 report §5.5, "false cells on the backward hop fell from 180 to 98"**: 178.8 → 179, and the weight-1.0 file is on 491 samples while the 0.1 and 0 runs are on 299 (the report's caption says 299 for all). Either rerun the weight-1.0 diagnostic on the 299-sample set or state the two sample sets. Same file: new-cell recall 0.72 (not 0.73), erased 52 % (not 54 %).
6. **PAPER.md Figure 8 caption, "Solved lengths of 0.95–0.98 ×"**: the cleanup row's route 188 is 0.89 ×; write 0.89–0.98 × or restrict the sentence to the search and Qwen rows.
7. **C5 report §5.5 counter test C (0.88 / 68 / 83 %)**: no file has these values; closest are 0.874 / 71.1 / 79.6 % (n = 491) and 0.888 / 78.1 / 80.5 % (lrflat, n = 299). Name the source run or correct.
8. **C5 report §5.5 hypothesis 1 ("~7 false cells per sample, 23 % of the failing run's 167")**: 23 % of 167 is ~39 cells. One of the two numbers is wrong; the file (`…bwd2_banneal5_noisep1_L4…/wave_back_ties_wave_back.json`) gives tie + tie2 = 23.5 % of 166.8.
9. **C5 report §7, thin-wall maze "500 routes per split" → 200**, and "0 % beyond" → "0 % beyond on val, 0.8 % (1 route) of 6+ on test". First-pass IoU 0.36 → 0.35 (file at 2 dp).
10. **C5 report §4.3, "15–30 [points lost] on the long ones"** holds for val (30.8) but not for test (36.7 → 34.5, 2.2 points). Write "15–30 on val, 2 on test" or drop the range.
11. **Track A report §2.2 item 2, "~half of errors are chord through buildings"** → half of all routes (51.5 %), two thirds of the failures (68 %).
12. **Track A report §3.4 item 1, "~+10 pts per ~2× data"** → about +4 per doubling from 1k to 5k and +9.5 per doubling from 5k to 19.3k.
13. **Track A report §3.4 item 2, "4B vanilla (~27 % on val 200)"** → 24.5 % two-pass (27.0 % is the oracle mode); and on the same first 200 routes LoRA 19.3k solves 64.5 %, not ~50 %.
14. **Track A report §1, "Wave has not yet been rescored with this polyline rule"**: stale; point to `COMPARISON_TABLE_2026-10-08.md`.
15. **Not found in any file** (keep only with a source): PAPER.md §7.2 / comparison report §4 "1.5 % of its chain cells against 0.2 %" (and "0.010" for the one-shot, no unit), and the conversion-sensitivity rescue counts behind "up to about 3 points".

## 3. Over-claims (sentences that say more than the cited evidence)

1. **"The gap grows with route length everywhere"** (PAPER.md §7.2 one-pass paragraph; §8.4 "and the gap grows with length"; C5 report §4.9). Search minus one-shot, cell rule, by 2–3 / 4–5 / 6+ fronts: De Pijp 10.7 / 38.3 / 37.3; Zuid + Centrum 12.1 / 27.0 / 31.7; house maze 44.4 / 46.1 / 5.8; ZC512 16.3 / 27.9 / 22.6; maze 512 43.6 / 57.8 / 7.1. Polyline, De Pijp: 9.9 / 47.5 / 47.4. Only Zuid + Centrum grows monotonically. Suggested: "The gap is smallest on the shortest routes everywhere; beyond 4 fronts it stays at 37–47 points on De Pijp and narrows on the mazes, where both methods fail on long routes."
2. **"most on long routes and on the never-seen district"** (PAPER.md §8.4; C5 report §4.6 "largest on long routes"). Natural-noise gains by band: De Pijp val +8.9 / +17.7 / +13.4, test +8.2 / +13.7 / +12.4, Zuid + Centrum +9.1 / +13.5 / +21.0. Suggested: "most on routes of 4 fronts and more, and most of all on the long routes of the never-seen district."
3. **"Qwen always produces a line with roughly the right ends"** (PAPER.md §7.3; comparison report §3 item 1). Fine-tuned 19.3k: 0 % unparseable but 11.4 % wrong endpoints; zero-shot 2B: 92 %. Suggested: "After fine-tuning, Qwen always produces a parseable line and misses an endpoint on 11 % of routes."
4. **"the search never misses an endpoint, because the markers are an input plane"** (PAPER.md §7.3). The zero is guaranteed by the scoring path: the conversion treats the 3 × 3 cells around each marker as lit and starts the chain at the markers, so a grid model cannot score "wrong endpoints". Suggested: "the grid models cannot miss an endpoint under this conversion (the chain starts at the markers); their failures show up as unconnected drawings."
5. **Section 4: "where it bends once it is right four times in five, and where it has to turn several corners it is right about seven times in ten"**. The source (C4 report N9) gives dilated IoU 0.92 / 0.78 / 0.73 on thirds by detour ratio, not a success frequency and not a corner count. Suggested: "dilated IoU 0.92 on near-straight routes, 0.78 on bending ones and 0.73 on the largest detours." (Section 4 is outside the audited sections but feeds §8.1.)
6. **"Qwen3-VL-4B-Instruct … lands just below the straight-line floor"** (PAPER.md §6.2; Track A §2.2 item 3). The 27.6 % floor is on 500 routes; the zero-shot rows are on the first 200, where the straight line solves 50.5 %. 4B two-pass (24.5 %) is at half the floor on its own routes. Suggested: "lands at half the straight-line floor on the same 200 routes (24.5 % against 50.5 %)."
7. **"One pass against a loop, at the same budget … A single pass has no more steps to spend on a longer route"** (PAPER.md §7.2). The matched one-shot draws on 2 m cells. The Section-4 one-shot network on 4 m cells (`c4n_bld_det_L4_16k_store100k`, `oneshot_route_eval_val500.json`) scores 64.8 % (91 / 84 / 42 %) on the same 500 De Pijp routes under the cell rule at its own 4 m cells, 45.4 % on Zuid + Centrum and 43.6 % on the house maze. Not comparable (coarser cells make the rule more lenient), but it shows the one-pass number depends on output resolution. Suggested: add "at 2 m output cells; the 4 m one-shot of Section 4 scores 65 % under the cell rule at its coarser cells" or name the resolution in the claim.
8. **"The same 4-block transformer without scratchpad"** (PAPER.md §7.1). The one-shot checkpoint has 8,840,240 parameters against 8,856,409 (the difference is exactly the readout token and the 41-way state head) and keeps a 17-channel patch embedding (`model_config.in_chans` = 17 in `c5_oneshot_256m_matched.pt`). Acceptable with "same trunk; no readout token or state head; the scratchpad plane is present but unused" (check the last part against the training config).
9. **"Every result can be rebuilt from public data, the run ledger and released checkpoints"** (PAPER.md §9 heading), "rerun every experiment" (§9.2), "Every number in this paper names the run it comes from" (§9.2). Section 4 numbers name reports, not runs; the release holds the Section 5–7 networks and one network per ladder level, not every experiment; the free-running numbers differ between CPU and GPU on 4 of 200 routes (§8.6); and this audit did not run `REPRODUCE.md` §8. Suggested heading: "Every number in Sections 5–7 can be recomputed from the released checkpoints and evaluation bundle (same GPU class, bf16; rollouts can differ by about 1 % of routes across devices)".
10. **C5 report §4.3 and §7: Zuid + Centrum test labelled "(never seen)"**: 33 of its 500 windows overlap De Pijp. Report on the 467 clean routes (52.7 %) or add "(33 windows overlap De Pijp)". The PAPER.md is not affected (it reports val only).
11. **"transfers in part to a new district and a maze"** (PAPER.md §5.2 heading) and "60 % in a district never seen in training" (abstract): supported (ZC val, 0 of 500 windows overlapping, single seed, one district). No change needed; keep "single seed" near it in the abstract if space allows.
12. **PAPER.md §5.2 timing "a whole free-running route … 57 ms on average"**: measured on 55 routes of mean 163 m walking, shorter than the validation median (195 m). Suggested: add "(55 routes, mean 163 m)".
