# Memo: weighting the changed cells in the wave tasks (2 Oct 2026)

Proposal from the 2 Oct discussion; Marvin implements the loss side. Context: `docs/plans/C5_PLAN.md` §7,
`docs/memo/WAVE_SEARCH_2026-10-01.md`.

## 1. Why

In the six-task mix (`c5_r5_wave_mix6_L4_16k_g128_store40k`) the network copies the scratchpad perfectly
(copied-cell recall 0.97–1.00 on every task) but hardly extends it: new-cell recall 0.58 / 0.07 / 0.006 / 0.31 /
0.67 for first / step / reach / back / origin, against 0.81 / 0.65 / 0.64 / 0.64 / 0.87 when each task trains alone.
The tokens are learnt (0.95–1.0). The pad has ~1000 lit cells to keep and ~10–50 to add or remove per pass, so a
network that copies and adds nothing already collects most of the loss; alone, each task climbs out of that basin,
in the mix the shared solution stays in it. Forward/backward competition comes on top (Marvin's three package runs
separate the two effects), but the copy basin is there in every mix.

## 2. What the loss balances today

One axis: lit vs unlit. Per task a foreground weight `w = ((1 - f) / f) ** alpha` from the task statistics
(`spatial_data/c2_tasks.fg_ce_weight`, `c4_weight_alpha = 0.5`, `f` = share of lit cells; for the wave tasks
f ≈ 0.019 at 128×128 → w ≈ 7), applied to cells whose target is the foreground class
(`LocalGridViT._grid_ce_loss`, branch `c1_fg_weight_table`). With `c4_grid_loss = "ce_dice"` the weights are
off and the loss is plain CE + soft Dice; check which branch the wave specs use (`_c4_learning_recipe_overrides`).

## 3. The proposal: a second axis, stay vs change

For a pass with a scratchpad input, every output cell falls in one of four groups:

| | stays as on the pad | changes |
|---|---|---|
| lit in the target | copied front / path cells (~1000) | the new front, the new hop (~10–50) |
| unlit in the target | empty ground (~15,000) | a consumed front (backward passes) |

The two "changes" groups are the work of the pass; the two "stays" groups are the copy. Weight the cells so that
each group present in a sample contributes about equally to that sample's loss, whatever its size:

```
change[b, c] = (target[b, c] == fg) != pad_in[b, c]          # cells that differ between pad in and target
group[b, c]  = 2 * change + (target == fg)                    # 0 stay-unlit, 1 stay-lit, 2 change-unlit, 3 change-lit
n_g[b]       = number of cells of group g in sample b
w[b, c]      = 1 / (n_groups_present[b] * n_g[b])             # each present group sums to 1 / n_groups_present
loss         = sum_c w[b, c] * CE[b, c], averaged over the batch
```

Softer variant with one knob: keep today's lit weight and multiply the changing cells by a factor
`beta` (5–20): `w = w_fg(target) * (1 + beta * change)`, normalised per sample. Less principled, easier to tune.

Points:

- The copy must still be learnt: the "stays" groups keep a weight, they just stop dominating.
- The consumed front counts as a change (group 2); without it a network can leave old fronts standing.
  The eval's `c5_wave_false_fg_*` shows that.
- Only tasks with a pad in have `change` cells; for the plain mask tasks (segment, detour, ...) `pad_in` is
  empty, so group 2 is empty and group 3 equals "lit": the rule degrades to today's lit/unlit balancing.
- Dice: if the run uses `ce_dice`, the weights only enter the CE term; Dice already treats lit cells as a set,
  so the change weight matters less there but the copy basin is the same. Recommendation: CE with the group
  weights, Dice off, for the first comparison.

## 4. Where the change mask comes from (no loader change needed)

The pad in is channel 16 of the input at 1 m; the pad at cell level is its max-pool to the output grid:

```python
s = img.shape[-1] // targets.shape[-1]                                   # 2 at 128x128
pad_cells = F.max_pool2d(img[:, 16:17], s, stride=s)[:, 0] > 0          # (B, g, g) bool
change = ((targets == fg) != pad_cells) & (targets != IGNORE_TARGET)     # (B, g, g) bool
```

This is exactly what the loaders compute for the val meta (`pad_cells`), so train and eval agree. If a
per-cell weight map from the loader is preferred instead, say so and I add it to the batch (as a 6th element
after `tok_tgt`, like the token target was added).

## 5. Reading the result

Per task in `eval.jsonl` (already reported): `c5_wave_copied_recall_*` (should stay ~1.0),
`c5_wave_new_recall_*` (should rise towards the single-task values), `c5_wave_false_fg_*` (should stay ~0),
`c5_tok_acc_*`. Marvin's `wave_eval_dashboard.png` plots them per run. The single-task runs are the reference:
first 0.81, step 0.65, reach 0.64, back 0.64, origin 0.87 (new-cell recall, L2 8k, 128×128).

## 6. Beyond this (for the record)

Two further paths discussed on 2 Oct, in case the weighting alone does not lift the mix: a mixture of experts
by hand (attention heads or adapters masked per state token, joint fine-tuning afterwards), or a mixture of
experts in the MLP with a learned router and a load-balancing term. Both keep the selection inside the
network, since the state token is its own output.
