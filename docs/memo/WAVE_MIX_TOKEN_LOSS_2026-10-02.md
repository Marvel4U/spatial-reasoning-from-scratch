# Memo: why the backward mix failed, and the first search that runs on its own (2 Oct 2026)

Context: `docs/plans/C5_PLAN.md` §7, `docs/memo/WAVE_SEARCH_2026-10-01.md` (§7 results up to the morning of 2 Oct),
`docs/memo/CHANGE_WEIGHTED_LOSS_2026-10-02.md`. This memo records the search for the cause of the failed `bwd2`
runs, including the hypotheses that were wrong, and the result of the night run that followed.
Figures and animations: `docs/memo/wave_search_2026-10-02/`.

## 1. Result first

`c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k` (4 blocks, 8.9 M parameters, 48k steps): the whole
search in one network — forward, backward and draw passes mixed 4 : 4 : 1 with natural per-pass sampling, token loss
weight 0.1, annealed β 5 on changed cells, pad noise p1, cosine lr, 100k routes of De Pijp.

Free-running on 200 val routes (`harness/diagnostics/wave_rollout_eval.py`): the network starts from the start disc,
its output is the next scratchpad, its own token picks the next state, it stops on DONE. Nothing from the teacher.
A route is solved if the final pad connects start and goal through lit cells, the connection crosses no cell that is
entirely building, and it is at most 10 % longer than the teacher's walking length.

| | L4 48k | L2 48k (same recipe) |
|---|---|---|
| finished with DONE | 98 % | 83 % |
| solved, all 200 | 75 % | 4 % |
| solved, 2–3 fronts (40–120 m, 6 passes) | 92 % | 11 % |
| solved, 4–5 fronts (120–200 m, ~10 passes) | 77 % | 0 % |
| solved, 6+ fronts (200 m and more, ~15 passes) | 54 % | 0 % |
| token sequence identical to the teacher's | 84 % | 18 % |
| passes used / teacher's | 10.3 / 10.2 | 11.6 / 10.2 |
| length of solved routes / teacher's | 0.99 | 1.00 |

Under the teacher's schedule (state from the teacher, pad from the network) the L4 run has IoU within 4 m of
0.99 / 0.93 / 0.90 on first / step / reach and 0.79 / 0.80 / 0.81 on back / origin / draw: the backward part is the
weaker half. Teacher-forced (harness eval): wave_fwd 0.92, wave_bwd 0.96, wave_draw 1.00 IoU within 4 m, tokens
0.99–1.00. L2 is close teacher-forced (0.87 / 0.88 / 1.00) and collapses in the loop: depth decides the rollout.

Limits of the claim: val blocks of the training district; the new map is not tested; the 48k L4 recipe was not run
with token weight 1.0, so the share of the token fix in the 75 % is not measured; the success rule checks the pad as
a set of cells (marker cells count as lit), not a single extracted polyline.

**Wall time** (RTX 3060, bf16, no compile, `harness/diagnostics/wave_rollout_timing.py`): one pass 2.4 ms at batch 1;
2,300 passes/s at batch 128. A whole free-running route, one at a time and with the loop around it, 57 ms on average
(median 53, max 143; 10.3 passes, mean 163 m walking). The eval loop runs the grid and the token as two forwards per
pass; one forward for both would roughly halve that.

## 2. The failure we started from

The `bwd2` runs (wave_back / wave_origin 60 : 40, CE recipe, pad noise, flat lr, 16k) ended at IoU within 4 m of
0.72–0.78 on wave_back against 0.945 for the single-task run, with about 1.2–1.5 % false cells against 0.09 %.
In the strips the new hop (the stretch of path one backward pass adds, ~53 cells) was a patch-shaped blob and the
consumed front was only half erased; in rollouts the blob was fed back and the path dissolved within a few passes.

## 3. Hypotheses, in the order tried

| # | hypothesis | test | verdict |
|---|---|---|---|
| 1 | Ties in the teacher: on flat-price ground many 8-neighbour paths cost the same and Dijkstra's parent links pick one arbitrarily (Claude) | `wave_back_ties.py`: the family of all cheapest paths per hop, false cells attributed to it | Real but small: the family is a median 1.4× the hop; it holds ~7 false cells per sample (59 % of the clean run's 12.5), 23 % of the failing run's 167 |
| 2 | Fronts are rings of walking length, the wave is priced, so the hop is not the cheapest way to the front (Claude) | same script: cheapest front point from the tip | Minor: median shift 1 m, cost gap 2.7 % |
| 3 | The 100k store (Marvin) | array comparison; bwd2 on 40k | No: the first 40k routes are identical in all four arrays; the 40k run fails the same way |
| 4 | Dropping Dice | wave_back alone, CE recipe, no noise (B) | No: at a matched false-cell level B finds more of the hop than the Dice reference (0.72 vs 0.68) |
| 5 | Pad noise | wave_back alone, Dice + noise (A), CE + noise (C) | Costs ~10 points of hop recall, output stays sharp |
| 6 | The lit-cell weight 7.5 of the CE recipe | threshold sweep | Real, factor 3 in false cells at 0.5 (not 20); see §5 |
| 7 | Flat learning rate | C + flat lr (D) | No on one task (78 vs 68 false cells); makes the mix somewhat worse (263 vs 180) |
| 8 | The mix with wave_origin | C + mix (E), mix + flat (F) | Yes: E reproduces the failure at L2 8k; F matches the original |
| 9 | The mixed loader | train batches compared | No: wave_back samples are identical (294 pad, 54 new, 74 erased cells, same token and condition ids) |
| 10 | Confusion of "two fronts left" with the origin case | split by fronts on the pad | No: the damage is the same for 2 to 8 fronts |
| 11 | **The token loss** | E with token weight 0 and 0.1 | **Yes** |

Counter-test numbers (wave_back, L2, 8k, first 40k routes, 299 val samples, read at 0.5):

| run | new-cell recall | false cells per sample | consumed front erased |
|---|---|---|---|
| reference: alone, CE + Dice, no noise (and its rerun) | 0.68 (0.65) | 12.5 (13) | 95 % (94 %) |
| A: alone, CE + Dice, noise | 0.58 | 15.5 | 94 % |
| B: alone, CE recipe, no noise | 0.91 | 38 | 90 % |
| C: alone, CE recipe, noise | 0.88 | 68 | 83 % |
| D: C + flat lr | 0.89 | 78 | 81 % |
| E: C + mix with origin (token weight 1.0) | 0.73 | 180 | 54 % |
| F: E + flat lr | 0.74 | 263 | 39 % |
| E with token weight 0.1 | 0.82 | 98 | 74 % |
| E with token weight 0 | 0.86 | 88 | 77 % |
| bwd2 plain, L4 16k (the original failure) | 0.86 | 259 | 55 % |

The design that found it was Marvin's: test the ingredients on the individual task, one at a time, against a
reference that is known to work, and rerun the reference.

## 4. The mechanism

The network has two outputs behind the same blocks: the grid and one token read at the readout position. The loss
was grid + 1.0 × token.

- Alone, a wave task has a constant token (wave_back always BACK, wave_origin always DRAW). The token loss vanishes
  within a few steps and the blocks are trained by the grid loss only.
- In a mix the token is a decision (BACK or DRAW = is exactly one front left), not trivial on noisy pads.
- The scales differ: the grid loss is a mean over 16,384 cells, most of them easy (0.02–0.04 here); the token loss
  is one cross-entropy per sample (~0.1 here). In E the train loss ends at 0.128 against a grid loss on val of 0.036;
  in C at 0.022 against 0.017. (Train is on noisy pads, val on clean ones, so the gap is not only the token.)
- Effect: uniformly weaker grid output — the hop as a blob, half-erased fronts — in both tasks of the mix, stuck
  from step 1,000 on rather than slow.
- At weight 0.1 the token is learnt better than at 1.0 (0.998 / 1.00 against 0.96 / 0.96 at 8k).

The same holds in any wave mix (SEARCH vs BACK in the forward mixes), so it probably contributed to the earlier
forward and six-task mixes; those were not rerun.

## 5. The bwd2 reruns of the night (L2, 16k, token weight 0.1, cosine, noise, 100k)

| run | back: IoU 4 m / new / false | origin: IoU 4 m / new / false | solved from the teacher's forward pad (200 routes) |
|---|---|---|---|
| before: token weight 1.0, annealed β 5, flat, 40k | 0.73 / 0.72 / 1.4 % | 0.74 / 0.89 / 0.8 % | 9 % |
| annealed β 5 | 0.89 / 0.89 / 0.5 % | 0.92 / 0.95 / 0.2 % | 13 % |
| plain β 0 | 0.89 / 0.90 / 0.4 % | 0.92 / 0.96 / 0.2 % | 21 % |
| annealed β 5, lit-cell weight off (`c4_weight_alpha = 0`) | 0.88 / 0.39 / 0.1 % | 0.93 / 0.59 / 0.1 % | 0.5 % |

- β buys nothing on the backward tasks.
- Removing the lit-cell weight (Claude's suggestion, to calibrate the output at 0.5) is wrong for the loop: the
  network under-paints, lines break, the rollout fails. The slightly fat output the weight produces keeps the pad
  connected. The earlier statement "the anneal calibrates at 0.5" was incomplete for the same reason: the lit-cell
  weight stays on through the anneal, and that turns out to be useful.
- A backward-only L2 network solves at most 21 % from a perfect forward pad; the L4 whole-search network solves 75 %
  from scratch. Depth and steps matter more than the split into packages.

## 6. Recipe as of 2 Oct

Token loss weight 0.1 (`c5_token_loss_weight`); cosine lr over the run; lit-cell weight on (`c4_weight_alpha` 0.5);
pad noise p1; natural per-pass sampling (`wave_fwd` / `wave_bwd` / `wave_draw` 4 : 4 : 1); 4 blocks; 48k steps;
100k routes; annealed β 5 kept for the forward part (not separated in the whole-search run).
Spec: `c5_wave_night_queue_2oct()` in `harness/experiments.py`.

## 7. Tools added (Claude; all under `harness/diagnostics/`)

| script | what it does |
|---|---|
| `wave_back_ties.py RUN [--task wave_back]` | the backward hop on val: threshold sweep, false cells by cause (unerased front / tie family / near the hop / elsewhere), splits by fronts on the pad and hop length, figure |
| `wave_rollout_eval.py RUN [--n 200] [--start 1\|first_back] [--mode both\|scheduled\|free]` | rollouts over many val routes, numbers only: per-pass IoU and tokens under the teacher's schedule; free-running success |
| `wave_rollout_gif.py RUN --route N` | animation of one free-running search |
| `wave_rollout_timing.py RUN` | wall time per pass and per route |

Pitfall hit twice in this work: a standalone script must set `spatial_data.channels.MARKER_RADIUS_PX` from
`config.marker_radius_px` as the harness does; the default draws 1 px markers and the network then cannot see the
goal (forward token accuracy 0.5 instead of 0.99). The two-route rollout in the training log is not a ranking: it
showed 0.49 and 0.39 for the run that solves 75 %.

## 8. Open

- The new map (Zuid + Centrum): wave data for its val routes, then the same eval.
- The 25 % failures: which pass breaks (route 363 in the animation: a gap just before the goal after 17 correct tokens).
- Long routes (54 %): more steps or more depth.
- The ties: straightening the way back would remove ~7 false cells per sample; not needed now.
- One forward per pass for grid and token in the rollout.
