# Memo — class balancing without biased decisions (one rule for all tasks)

21 Sep 2026. Status: **tested on existing checkpoints, not yet in the harness.** Script: `docs/plans/perf_prototypes/c1_logit_correction_check.py`.

## 1. Problem

Rare tasks need a positive weight `w` in the cross-entropy, otherwise the model learns "mark nothing" (the c0 lesson). But a weight makes the model **over-mark**: food & drink at w = 41 had 83 % precision, 99 % recall and marked something on 32 % of crops that contain no such place. Capping `w` for the affected task (the "clamp") treats a symptom, per task. We want **one rule that holds for every task**, because the project builds towards a general multi-task model with a general evaluation.

## 2. Where the over-marking comes from (exact)

With positive weight `w`, the loss is minimised when the model reports, for a cell whose true probability of being positive is `q`:

`p = w·q / (w·q + 1 − q)`  — equivalently, the foreground log-odds are shifted up by `log w`.

So the model is not confused; its output is shifted by a **known** amount. A cell with true probability 50 % is reported as `w / (1 + w)`: 0.71 for w = 2.5, 0.84 for w = 5.2, 0.976 for w = 41, 0.990 for w = 100.

## 3. The rule

Keep the balancing in training. **Undo the shift at decision time:**

> mark a cell if `P(fg) > w_task / (1 + w_task)`  (same thing: `logit_fg − logit_bg − log w_task > 0`)

Same formula for every task, using the same `w` the loss used; nothing tuned. Tasks with `w ≈ 1` keep a cut-off of ≈ 0.5 automatically.

## 4. Test (all 250 val crops per task; IoU on non-empty targets)

"Oracle" = best cut-off from a per-task search: a ceiling for comparison, not a method.

| L5b core four on v3b | w | plain 0.5 | **rule** | oracle |
|---|---|---|---|---|
| building | 2.5 | 0.971 | 0.973 | 0.973 |
| sidewalk | 5.2 | 0.853 | 0.874 | 0.875 |
| noise ≥ 65 dB | 1.8 | 0.989 | 0.989 | 0.990 |
| food & drink | 40.8 | 0.807 | 0.885 | 0.897 |
| **mean** | | **0.905** | **0.931** | **0.934** |

Food & drink under the rule: precision 0.83 → 0.96, recall 0.99 → 0.93, false alarms on empty crops 32 % → 0 %.

| Other checkpoints (mean over trained, non-degenerate tasks) | plain 0.5 | rule | oracle |
|---|---|---|---|
| L5c factorised twelve on v3b (9 tasks) | 0.815 | 0.871 | 0.876 |
| L5b core four on v2 discs (w = 100 for food & drink) | 0.821 | 0.866 | 0.877 |

Per task on L5c/v3b: noise ≥ 75 dB 0.745 → 0.811 (oracle 0.820); other named 0.607 → 0.768 (0.768); food & drink 0.535 → 0.788 (0.790); sidewalk 0.761 → 0.805 (0.807). One regression: roadway 0.831 → 0.809 (oracle 0.832, reached at 0.6 instead of the rule's 0.815). The held-out tasks (`surface_0`, `estab_2`) do not move: their problem is not the cut-off.

The rule recovers almost the whole gap to the oracle with zero tuning. It slightly over-corrects for very large `w` (IoU prefers a marginally lower cut-off than the unbiased one), which argues for smaller weights, §5.

## 5. Dampened balancing (Marvin's proposal) fits with the rule

`w = ((1 − f) / f)^α` with one global exponent, e.g. α = 0.5:

| Task | f (share of positive cells) | α = 1 (now) | α = 0.5 |
|---|---|---|---|
| building | 0.288 | 2.5 | 1.6 |
| sidewalk | 0.162 | 5.2 | 2.3 |
| noise ≥ 75 dB | 0.029 | 33 | 5.8 |
| food & drink (v3b) | 0.017 | 57 | 7.6 |
| c0 marker (1 of 4,096) | 0.00024 | 4,095 | 64 |

The two pieces have separate jobs: **dampening decides how hard training pushes on rare classes** (optimisation); **the decision rule keeps the output unbiased whatever the weight** (calibration). With the rule in place α is a single global knob and no task needs a clamp. Expected benefit of α < 1: smaller residual over-correction and less of the "blob is nearly free" effect seen in c0.

## 6. Proposed evaluation convention (for every task, always)

1. **Score under the general rule** (§3): the number that counts.
2. **Oracle score** from a cut-off search: diagnostic only.
3. The **gap** between them says what kind of problem a task has: large gap = the model knows more than its decisions show (calibration); both low = it does not know (representation). Report per task, never pooled; keep precision, recall and the empty-target false-alarm rate next to IoU.

## 7. Limit, and what comes after

Rule and dampening both need each task's share of positive cells `f`. For c2 / c3 that share varies strongly per sample ("quiet sidewalk" depends on the crop), so a per-task average gets rougher, and a per-sample weight cannot be undone at inference because the target is unknown there. The standard loss that needs **no** frequency information is overlap-based (Dice / soft IoU, usually combined with plain CE). One comparison once c2 exists: [balanced CE + rule] vs [CE + Dice, cut-off 0.5].

## 8. Where it goes in the code (not done)

- `harness/eval_spatial.py` and `analysis/` viewer: decision `P(fg) > w/(1+w)` per sample from its task id (the weights are already in the model config as `c1_per_task_fg_weights`); log rule score and oracle score per task.
- `spatial_data/c1_tasks.py` `per_task_fg_ce_weight`: optional exponent α (default 1.0 = current behaviour); `c1_ce_weight_clamp` can then go to a high safety value.
- Model and training loss: unchanged.
