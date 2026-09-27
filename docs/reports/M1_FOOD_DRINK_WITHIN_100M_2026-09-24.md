# Report: teaching a small vision transformer "food & drink within 100 m of the marker"

Written 24 Sep 2026 by Claude from Marvin's runs; Marvin wrote the results of the second round (chapter 7.1–7.6) and corrected the first draft. All numbers are as recorded in the run ledger and the diagnostics files; nothing is rounded beyond three digits. The document is meant to stand on its own; pointers to code and plan sections are collected in chapter 9.

## 0. Glossary

- **Crop.** One training or validation sample: a 256 m × 256 m square of Amsterdam at 1 m per pixel, given to the model as aligned class planes (noise band, surface type, establishment class), each one-hot encoded, 15 channels in total, plus a 16th channel that carries the marker.
- **Marker.** A point in the crop that the task refers to ("within 100 m of *the marker*"). It is drawn into the 16th input channel, either as a single pixel or as a filled disc of radius 3 px ("disc marker"). Training crops get a fresh random marker every time they are drawn; the 250 validation crops have one fixed marker each.
- **Cell.** The model's output is a 64 × 64 grid over the crop, one cell = 4 m × 4 m. Each cell is labelled foreground or background.
- **The three tasks used in this report.** *Dense food & drink*: mark every cell that is a food & drink establishment, marker ignored. *Disc*: mark every cell within 100 m of the marker, layers ignored. *Conjunction*, the task of this report: mark cells that are both. In run names these appear as `food_drink`, `within_100m` and `food_drink_within_100m`; the plan calls the conjunction "m1".
- **Condition tokens.** The task is told to the model by two extra tokens placed in front of the image tokens (one for the layer condition, one for the relation). A model trained on several tasks at once receives a different pair per sample; that is what "mixed training" or "the mix" means below.
- **Patch, block, head.** The 256 × 256 crop is cut into 16 × 16 patches of 16 px, each projected to a 384-wide token. The encoder is a stack of transformer blocks (2 or 4 here), each with 6 attention heads. Blocks and heads are numbered from 0; "block 3 head 1" is the second head of the fourth block.
- **Relative position bias.** A learned number added to every attention score that depends only on the row and column offset between the two patches (the Swin-transformer form). It hands attention the notion of distance instead of making it infer distance from position codes. "Bias in all blocks" means every block has its own table; "first block only" means only block 0.
- **Shortcut.** A solution the optimiser finds that scores reasonably on the loss without doing the task. For the conjunction the shortcut is "mark every food & drink cell, ignore the marker".
- **Escape.** The moment a run leaves the shortcut and starts using the marker. Measured as the first evaluation at which the conjunction IoU exceeds 0.6 (evaluations are every 1,000 steps unless stated).
- **IoU.** Intersection over union of predicted and target foreground, averaged over validation crops whose target is not empty. The offline probes in this report use a slightly different per-crop accounting than the training harness, which is why the shortcut ceiling reads 0.29 in the probes and 0.41 in the harness; rankings are identical.
- **Patch-local IoU.** IoU of the same model when every patch is only allowed to attend to itself (and to the condition tokens). If it equals the full IoU, the model uses no information from neighbouring patches. A large gap ("ablation drop") means the model routes information across patches, which the conjunction requires.
- **Seed, horizon.** Seed: the random seed of the run. Horizon: the length over which the cosine learning-rate decay is stretched; by default equal to the run length, so an 8k-step run and a 16k-step run have different learning rates at the same step.

## 1. The task and why it matters

Input: one crop with 16 channels. Output: the 64 × 64 grid, foreground = cells that are a food & drink establishment **and** lie within 100 m of the marker. Loss: cross-entropy plus Dice, decision cut-off 0.5. Validation: 250 held-out crops.

The task is small in every sense that matters for learning:

| quantity (validation set) | value |
|---|---:|
| foreground share of cells (the target) | 0.26 % |
| food & drink share of cells | 0.97 % |
| share of cells inside the 100 m disc | 32 % |
| crops whose target is empty | 42 % |
| IoU of the shortcut "mark all food & drink" against the target | 0.287 (0.41 in harness accounting) |
| IoU of "mark the disc" against the target | 0.008 |

The target is a quarter of a percent of the cells, and a rule that ignores the marker already reaches a quarter of the possible IoU. That rule is patch-local: whether a cell is a food & drink establishment can be read off the patch it sits in. The distance to the marker cannot; it needs information from the patch that contains the marker to reach every other patch.

It is the first task in the programme that combines a marker-relative condition with a layer condition. Every task planned after it, routes between two markers, cost layers along a route, "the quieter of the two cafés near the marker", has the same shape. If the model cannot learn this one, nothing after it is reachable.

## 2. First round: twelve runs, two successes

The starting recipe: 2 encoder blocks, 6 heads, width 384 (5.1 M parameters), batch 16, learning rate 1e-3 with cosine decay to the run length and 10 warmup steps, class weights with exponent 0.5, one random marker per training crop drawn as a single pixel, no relative position bias, 8,000 steps.

| # | change from the starting recipe | val IoU | patch-local IoU | false-positive rate on empty crops | outcome |
|---|---|---:|---:|---:|---|
| 1 | none | 0.410 | 0.410 | 0.73 | shortcut |
| 2 | three-task mix, ⅓ each, warm-started from a trained disc model | 0.310 | 0.342 | 0.67 | worse; the loaded disc circuit was gone by step 500 |
| 3 | 4 blocks, batch 32, mix 80/10/10 (two runs) | aborted at 2k and 3k | | | validation loss flat |
| 4 | 4 blocks (8.7 M parameters), batch 32 | 0.417 | 0.418 | 0.72 | shortcut; four times the data changed nothing |
| 5 | 4 blocks, batch 32, bias in all blocks | 0.417 | 0.378 | 0.72 | shortcut |
| 6 | 4 blocks, batch 32, bias, mix 80/10/10 | 0.243 | 0.001 | 0.54 | worse |
| 7 | disc marker | 0.402 | 0.305 | 0.67 | shortcut |
| 8 | disc marker, mix 80/10/10 | 0.273 | 0.213 | 0.61 | worse |
| 9 | disc marker, bias in all blocks | 0.405 | 0.381 | 0.71 | shortcut |
| 10 | disc marker, bias, mix 80/10/10 | 0.218 | 0.152 | 0.63 | worse |
| 11 | **4 blocks, batch 32, disc marker, bias in all blocks** | **0.799** | 0.343 | 0.08 | **escaped around step 6.5k** |
| 12 | as 11, run for 16k steps with the decay stretched to 16k, diagnostics on | **0.883** | 0.001 | 0.02 | **escaped between 2k and 3k**; 0.85 from 4k, 0.88 from 13k |

Run ids, in order: `food_drink_within_100m_rand1`, `…_mix_8k`, `…_mix80_L4_16k` and `…_mix80_L4_8k_b32`, `…_L4_8k_b32`, `…_L4_8k_b32_relV2`, `…_mix80_L4_8k_b32_relV2`, `…_rand1_disc3`, `…_mix80_8k_disc3`, `…_rand1_disc3_relV2`, `…_mix80_8k_disc3_relV2`, `…_L4_8k_b32_disc3_relV2`, `…_L4_16k_b32_disc3_relV2`. Run 11 ran without the diagnostics evaluator, so its escape step comes from the cliff in its validation loss between steps 6k and 7k, not from a logged IoU curve. Run 12's first evaluation above IoU 0.6 is at step 3k; "between 2k and 3k" is the window.

The ten failures fall into two groups. The shortcut group (runs 1, 4, 5, 7, 9) all sit at IoU 0.40 to 0.42 with patch-local IoU equal to the full IoU and a false-positive rate of about 0.7 on crops whose target is empty: the model marks food & drink wherever it is. The mix group (runs 2, 3, 6, 8, 10) is worse than the shortcut at 0.22 to 0.31.

### 2.1 How the shortcut was identified

The IoU alone does not say what a model does. Two offline probes on the checkpoints do. The first compares the prediction with the three candidate masks: the true target, "all food & drink", and "the disc". The second reports, for food & drink cells at increasing distance from the marker, the share the model marks. For run 4, the 4-block model trained on four times the data:

| distance from the marker | 0–50 m | 50–100 m | 100–150 m | 150–200 m | 200–400 m |
|---|---:|---:|---:|---:|---:|
| share of food & drink cells marked | 0.94 | 0.92 | 0.93 | 0.91 | 0.87 |

Flat. The prediction agrees with the "all food & drink" mask at IoU 0.90. The marker plays no role at all. Every run in the shortcut group has this profile.

## 3. What did not work, and what each failure taught

**More depth and more data alone (run 4).** Twice the parameters and four times the sample presentations moved the IoU from 0.410 to 0.417. The shortcut is a minimum the optimiser reaches from a cold start, and capacity does not move it. Lesson: find out which solution a model has before scaling it.

**Mixed training on recipes that cannot solve the task alone (runs 2, 3, 6, 8, 10).** Every mix was worse than the single task, whatever the weights and whether or not a trained disc model was loaded first. The dense task dominates, the disc task at a 10 % share never builds its circuit, and a loaded disc circuit is overwritten within 500 steps. Lesson: interference is real when the network cannot build the marker circuit. As chapter 7 shows, the same mixes work once it can.

**Relative position bias alone, on a pixel marker (run 5).** No change. The bias gives attention the notion of distance, but attention cannot route from a marker that the patch projection has not yet learned to detect. A single pixel is one column out of 256 in that projection and is seen at any given position on 1 in 256 samples. The same starvation was found in the very first experiments of the programme, where a single-pixel target could be located to the right patch but not to the right cell inside it.

**Disc marker alone (run 7).** No change at 2 blocks. The disc makes the marker detectable, but a 2-block model with 12 heads has no spare head for a marker circuit while all heads serve the shortcut.

**Disc marker and bias together at 2 blocks (run 9).** No change either, which was the surprise. On the plain disc task the same pair trains in about 1,500 steps. The difference is the loss landscape: on the plain disc task the disc is the whole target, on the conjunction the shortcut collects most of the reward first.

## 4. What worked

Runs 11 and 12: 4 blocks, batch 32, disc marker, relative position bias in all four blocks. Validation IoU 0.80 after 8k steps, 0.88 after 16k.

The distance profile of run 11, same probe as above:

| distance from the marker | 0–50 m | 50–100 m | 100–150 m | 150–200 m | 200–400 m |
|---|---:|---:|---:|---:|---:|
| share of food & drink cells marked | 0.90 | 0.81 | 0.03 | 0.02 | 0.04 |

A clean step at 100 m: 3 % of food & drink cells outside the disc are marked, 84 % of the target cells inside it are found. Run 12's patch-local IoU is 0.001 against a full IoU of 0.88, so the model depends entirely on information crossing patches. It is doing the task.

Run 12 ran with the diagnostics evaluator, which records, at every evaluation, the attention entropy of every head and the share of each head's attention that lands on the token of the patch containing the marker. It shows how the solution appears:

- By step 1k the IoU is already 0.41, the shortcut level, and the validation loss is still high. Between 2k and 3k the conjunction appears as one event: validation loss 0.83 → 0.11, IoU 0.40 → 0.78. At 2k the loss had got worse while the IoU stayed flat; then both jump.
- **One head does the routing.** Block 3 head 1, in the last of the four blocks, puts 0.99 of its attention on the marker token from step 3k on and keeps it there until the end (0.96 to 0.99 through 16k). A precursor is visible at 2k: block 3 head 2 at 0.47. The heads in blocks 0 to 2 never attend to the marker (0.05 or less).
- The first block's heads all attend to the constant condition tokens from step 1k (entropy near 0), and block 1 goes uniform after the escape. The early blocks supply per-patch features and little else.

## 5. Analysis of the first round

### 5.1 Marker tasks learn in two phases, and each phase has its own lever

Every marker task in the programme has shown the same course: a long plateau, then a sudden transition. The diagnostics separate the two phases. In the first, the patch projection learns that a marker is present in a patch. In the second, one attention head learns to send that information to the patches around it, visible as a drop in that head's entropy and a jump in its attention to the marker token.

The disc marker shortens the first phase, because a filled disc lights several projection columns per patch instead of one. The relative position bias shortens the second, because "attend to patches within a certain offset" becomes a handful of bias weights instead of something attention has to discover from position codes. Either lever alone leaves the other phase as the bottleneck, which is why each looked useless when tested alone and why the pair looked like a discovery when tested together. On the plain disc tasks the pair moved the transition from 4,500 to 6,500 steps down to about 1,500, for radii of 50, 75 and 100 m.

### 5.2 The conjunction adds a third obstacle: the shortcut basin

On the conjunction the pair is necessary but not sufficient at 2 blocks. The optimiser first finds "mark all food & drink", which earns most of the Dice reward with a patch-local rule. Leaving that basin requires a marker head to grow while the rest of the network keeps serving the shortcut. Run 12 shows where that head grew: one head in the last block, with the blocks before it largely idle. The first-round reading was that 4 blocks helped by providing spare heads, not by providing computation the task needs, and that 2 blocks might manage with a different seed or more signal. Chapter 7 tests that.

### 5.3 How much signal a run has seen at the transition

A useful way to compare runs is to count, per output cell, how many positive labels it has received by the transition: foreground share × batch size × steps. For the conjunction the plain positives are the wrong count, because they feed the shortcut; the cells that only the marker explains are food & drink cells inside the disc (22 per crop, the positives) and food & drink cells outside it (48 per crop, negatives that the shortcut gets wrong). That is 70 cells of distance signal per marker, against roughly 1,000 on a plain 100 m disc task.

| runs | transition | marker presentations (markers × batch × steps) | distance-signal coverage per cell |
|---|---:|---:|---:|
| plain disc tasks, pixel marker | 4.5k to 6.5k | 64k to 144k | 1.2k to 34k |
| plain disc tasks, disc marker and bias | 1.5k | 24k to 48k | 4.7k to 8k |
| conjunction, run 11 | ~6.5k | 208k | 3.6k |
| conjunction, run 12 | ~2.5k | 80k | 1.4k |
| conjunction, run 9 (2 blocks, never escaped) | — | 128k by the end | 2.2k by the end |

Two things follow. With a pixel marker the transitions line up on marker presentations, not on positives: the 30-fold spread in positive coverage across radii did nothing, which is the first phase being the bottleneck. With disc marker and bias, transitions land at a few thousand units of the right signal per cell, and the conjunction fits that scale once the right signal is counted. It is a scale, not a threshold: runs 11 and 12 share the architecture but not the schedule, and differ by 2.6× in the coverage at transition.

### 5.4 The escape is a high-variance event

Runs 11 and 12 use the same seed and the same data recipe; run 12 differs in length, in the stretched decay and in having the diagnostics on. Their training losses match at step 1k (0.685 against 0.684) and differ at 2k (about 0.60 against 0.70). One escapes between 2k and 3k, the other around 6.5k. With a fixed seed, residual non-determinism on the GPU (index and scatter kernels, bf16 arithmetic) still moves the timing of a saddle escape. Consequence: a single run's escape step is not evidence for or against a lever; timing comparisons need several seeds at a fixed specification. The second round was designed around this.

### 5.5 The learning-rate schedule is not the explanation

Stretching the cosine decay from 8k to 16k gives a learning rate of 0.97 against 0.87 of peak at step 2k and 0.93 against 0.72 at 3k. Run 11 escaped at 0.13 to 0.23 of peak; run 12 at 0.93 to 0.97. A high learning rate is therefore not what an escape needs, and the two runs differ in too many things for the horizon to be credited. Whether the task is stable under a constant learning rate was a separate question, answered in chapter 7.

### 5.6 The diagnostics that found this

Three offline probes on checkpoints did most of the work, none of them part of the original suite: the shortcut probe (prediction against each candidate mask, plus the distance profile), the head-sink probe (where the one-hot heads look, which separated a dead-start failure from real results), and the signal-coverage count. The suite's own attention-to-marker measurement, after a fix to the token index on 23 Sep, is what located the circuit in block 3 head 1.

## 6. Second round: design

The goal was to turn "two runs worked" into a defensible statement about which ingredients are necessary and how reproducible the result is. Seventeen runs were queued on the evening of 23 Sep and finished by the next morning. Unless stated, each run is the run-11 specification: 4 blocks, batch 32, disc marker, bias in all blocks, 8,000 steps, decay horizon pinned to 8,000, diagnostics on. Each takes about seven minutes on the rig.

1. **Seeds.** The run-11 specification with seeds 1, 2 and 3. Pass: all three escape within 8k. The spread of escape steps is the error bar for everything else.
2. **Ablations at 4 blocks.** The missing cells of the 2 × 2: disc marker without bias (two seeds), and disc marker with bias in the first block only.
3. **Two blocks again.** Batch 32 at 2 blocks (two seeds), so that only depth differs from run 11; and 2 blocks with four markers per training crop, at batch 16 and at batch 32, to raise the distance signal without depth.
4. **Constant learning rate.** 1e-3 and 3e-4 with no decay: does the task escape, and does it hold, without annealing?
5. **Mixed training on the working stack.** The three tasks at ⅓ each for 16k steps; 80/10/10 toward the conjunction for 16k; ⅓ for 24k; and a sequential variant, the disc task alone for 3k then the ⅓ mix for 13k. Pass: each task within 0.03 of its single-task number (about 0.95, 0.99 and 0.85). Also planned: a wrong-token test, feeding the conjunction crops with the other tasks' tokens, and the attention of the marker head under each token.

## 7. Second round: results

Escape step is the first evaluation at which the conjunction IoU exceeds 0.6. For the mix runs all per-task numbers come from the same final evaluation.

A quality note before the tables. Runs 11 and 12 and the best mix models reach patch-local IoU of about 0.001, i.e. fully cross-patch. Several single-task escapes in this round still show patch-local IoU of 0.15 to 0.28: conjunctive by IoU, but not as clean. Patch-local IoU and the shortcut probe should both be read before a run is called solved.

### 7.1 Seeds

| seed | final IoU | patch-local | escape step | verdict |
|---:|---:|---:|---:|---|
| 1 | 0.428 | 0.142 | — | shortcut |
| 2 | 0.415 | 0.245 | — | shortcut |
| 3 | 0.847 | 0.162 | 5,000 | escaped |

One of three. Reproducibility of the single-task recipe at 8k is not established; runs 11 and 12 (seed 1337) remain single-seed demonstrations.

### 7.2 Ablations at 4 blocks

| variant | final IoU | patch-local | escape | read |
|---|---:|---:|---:|---|
| disc marker, no bias, seed 1 | 0.414 | 0.374 | — | the bias is necessary |
| disc marker, no bias, seed 2 | 0.416 | 0.177 | — | |
| disc marker, bias in the first block only | 0.403 | 0.366 | — | the bias is needed in every block |

Together with runs 4 and 5 (pixel marker without and with bias, both at 0.417), the 2 × 2 is complete: only the combination of disc marker and bias in all blocks has ever escaped.

### 7.3 Two blocks

| variant | final IoU | patch-local | escape | read |
|---|---:|---:|---:|---|
| batch 32, seed 1 | 0.375 | 0.243 | — | shortcut |
| batch 32, seed 2 | 0.852 | 0.252 | 7,000 | one of two seeds escapes |
| batch 16, four markers per crop | 0.426 | 0.213 | — | more markers did not help |
| batch 32, four markers per crop | 0.427 | 0.289 | — | |

Two blocks can escape, but less often and later than four; more markers per crop did not replace depth.

### 7.4 Constant learning rate

| learning rate | final IoU | patch-local | escape | read |
|---|---:|---:|---:|---|
| 1e-3 constant | 0.814 | 0.226 | 7,000 | escapes without annealing and holds |
| 3e-4 constant | 0.834 | 0.276 | 3,000 | a lower constant rate escapes as well |

Both hold after the escape with no drift in IoU or validation loss. Annealing is not needed for stability in this range; a threefold higher rate is known from earlier probes to collapse a run at the start.

### 7.5 Mixed training

| variant | steps | dense food & drink | disc | **conjunction** | conjunction patch-local | conjunction escape | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| ⅓ each | 16k | 0.965 | 0.993 | **0.891** | 0.004 | 3,000 | yes |
| 80/10/10 | 16k | 0.958 | 0.990 | **0.880** | 0.006 | 2,000 | yes |
| ⅓ each | 24k | 0.966 | 0.995 | **0.887** | 0.005 | 3,000 | yes |
| disc alone for 3k, then ⅓ for 13k | 3k + 13k | 0.944 | 0.992 | **0.843** | 0.082 | 1,000 | yes, lower |

All three tasks reach their single-task band in one model, and the conjunction is cleaner (patch-local 0.005) than in any single-task run. Weighting toward the conjunction did not beat the uniform mix (0.880 against 0.891). Running 24k instead of 16k did not change the result (0.887 against 0.891), so 16k is not short. The sequential variant escapes earliest but ends lowest and least clean; a curriculum is not clearly better than the mix.

### 7.6 The condition token steers the model

The wrong-token test: the conjunction validation crops, fed with each of the three tokens, and the prediction compared with the three candidate masks.

| model | token fed | IoU vs conjunction target | vs all food & drink | vs disc |
|---|---|---:|---:|---:|
| ⅓ mix, 16k | dense food & drink | 0.267 | **0.958** | 0.008 |
| | disc | 0.008 | 0.008 | **0.970** |
| | conjunction | **0.873** | 0.257 | 0.008 |
| 80/10/10, 16k | dense food & drink | 0.267 | **0.960** | 0.008 |
| | disc | 0.008 | 0.008 | **0.970** |
| | conjunction | **0.884** | 0.267 | 0.008 |

Same image, three tokens, three different and correct outputs. The mix models compute all three functions and the token selects one. The 0.267 under the dense token is exactly the shortcut ceiling, so what the conjunction adds over the dense task is precisely the disc gate.

The attention traces show how. In the ⅓ model, three heads in block 2 attend to the marker at 0.65 to 0.83 when either marker token is present and at 0.00 under the dense token: the routing heads are switched by the task token. One further head attends to the marker under every token (0.84 to 0.88) and is presumably read out only when needed. The 80/10/10 model has the same structure in block 1 (one head at 0.00 / 0.80 / 0.80).

### 7.7 Escape statistics across the round

All runs below use 4 blocks, batch 32, disc marker and bias in all blocks unless stated.

| training | runs | escaped by 8k | escape steps |
|---|---:|---:|---|
| single task, 8k horizon | 5 (three seeds, two constant-rate runs) | 3 of 5 | 3k, 5k, 7k |
| single task, all horizons including runs 11 and 12 | 7 | 5 of 7 | 2.5k to 7k |
| single task at 2 blocks | 4 | 1 of 4 | 7k |
| **any mix containing the disc task** | 4 | **4 of 4** | **1k, 2k, 3k, 3k** |

The disc task inside the mix builds the marker head in about 1,500 steps, as it does alone, and the conjunction reuses it. A single-task run has to grow the same head from the 0.7 % of cells that only the marker explains, and whether that happens inside 8k steps is a coin toss. This is the mechanism guessed at in 5.2, now with numbers. It also reframes the answers in 7.1 to 7.3: "is depth necessary" and "does the seed reproduce" were asked of the hard path; on the mix path neither has been tested yet.

### 7.8 Where the marker head forms

Run 12: block 3. Seed 3: block 1, weakly (0.53). The 2-block escape: block 1, two heads. Constant 3e-4: block 1. The ⅓ mix: block 2. The 80/10/10 mix: block 1. The sequential mix: block 0, after the disc pretraining. Any block can host the circuit. That argues against "depth adds computation the task needs" and for "depth adds spare heads, i.e. more chances for the circuit to form".

### 7.9 Next runs

Agreed with Marvin on 24 Sep. Everything below is the run-11 specification with the ⅓ mix unless stated, decay horizon pinned explicitly, diagnostics on, escape step defined as above.

1. **Mix seeds.** The ⅓ mix and the 80/10/10 mix with two further seeds each, at the 8k horizon. This answers two questions at once: does the 4-of-4 escape rate of the mix hold up across seeds, and is the 16k horizon needed at all, or was that only the single-task path.
2. **Constant learning rate on the mix.** The ⅓ mix at constant 1e-3 and constant 3e-4. The single-task runs showed no instability; this checks that the mix, which carries three losses at once, is as robust.
3. **Two blocks with everything.** The ⅓ mix at 2 blocks, batch 32, 16k, two seeds. If the mix path removes the depth requirement, the smallest model becomes the reference and the depth question is closed; if it does not, 4 blocks is part of the recipe.
4. **The reference recipe.** Until 1 to 3 say otherwise, the ⅓ mix at 4 blocks and 16k is the reference for marker-conjunction tasks: 0.965 / 0.993 / 0.891, patch-local 0.004, tokens verified to steer. The single-task seed matrix is a research result about a harder path, not the recipe. For the route tasks that follow, the bare two-marker task belongs in every mix from the start.

Reads for all of them: per-task IoU curves, escape step, patch-local IoU at the end, and the wrong-token table on the final checkpoint.

### 7.10 Results of the §7.9 runs (24 Sep, evening)

All eight runs finished. Escape step as before (first evaluation with conjunction IoU above 0.6, evaluations every 1k).

| block | run | final conjunction IoU | escape | patch-local IoU |
|---|---|---:|---:|---:|
| ⅓ mix, 4 blocks, 8k horizon | seed 4 | 0.837 | 2k | 0.42 |
| | seed 5 | 0.875 | 2k | 0.36 |
| 80/10/10 mix, 4 blocks, 8k horizon | seed 4 | **0.189** | never | 0.07 |
| | seed 5 | 0.818 | 5k | 0.05 |
| ⅓ mix, 4 blocks, 8k, constant LR | 1e-3 | 0.820 | 2k | 0.38 |
| | 3e-4 | 0.851 | 1k | 0.02 |
| **⅓ mix, 2 blocks, 16k** | seed 4 | **0.907** | 1k | 0.35 |
| | seed 5 | **0.897** | 2k | 0.01 |

**Patch-local IoU is not a cleanliness measure, and the earlier quality note in this chapter over-read it.** It reports what the model outputs when its marker information is cut: 0.35 means the fallback is "mark food & drink", 0.01 means the fallback is "mark nothing". Both are compatible with a fully correct model, and the value moves between evaluations (the 80/10/10 reference: 0.36, 0.16, 0.00, 0.25, 0.31, 0.10, 0.01 across its run). The direct test is whether the full model marks food & drink outside the disc (`docs/plans/perf_prototypes/c3_outside_disc_probe.py`, 16 validation batches under the conjunction token):

| model | patch-local IoU | food & drink marked inside the disc | marked outside the disc |
|---|---:|---:|---:|
| ⅓ mix, 4 blocks, 8k, seed 4 | 0.42 | 0.941 | 0.014 |
| ⅓ mix, 2 blocks, 16k, seed 4 | 0.35 | 0.957 | 0.004 |
| ⅓ mix, 2 blocks, 16k, seed 5 | 0.01 | 0.957 | 0.005 |
| ⅓ mix, 4 blocks, 16k (reference) | 0.00 | 0.956 | 0.004 |

All four are clean; the differences in patch-local IoU are differences in fallback, not in behaviour. The wrong-token test on the 2-block seed-5 model gives 0.881 / 0.954 / 0.970 against the conjunction, the dense mask and the disc, the same as the 4-block reference.

What the eight runs settle:

1. **Depth is not required on the mix path.** Two blocks, ⅓ mix, 16k: 0.907 and 0.897, both seeds, both escaping by 2k, both clean by the outside-disc probe. The 5.1 M-parameter model is now the reference for this task. Depth was a way to win the single-task lottery, and the mix removes the lottery.
2. **The uniform mix is the robust one.** ⅓ mix: 8 of 8 escapes across 4 and 2 blocks, 8k and 16k horizons, cosine and constant LR, five seeds. 80/10/10: 2 of 3, with one outright failure at 8k. Weighting toward the hard task starves the disc task that builds the marker head.
3. **8k versus 16k** is a matter of the last few hundredths of IoU (0.84–0.88 at 8k against 0.89–0.91 at 16k), not of whether the task is learned. Constant LR at 3e-4 reaches 0.85 at 8k and escapes at 1k.
4. Escape statistics over every mix run so far that contains the disc task: **10 of 11**, all by 5k, the median at 2k.

**Reference recipe, final:** 2 blocks, 6 heads, width 384; batch 32; 3-px disc marker; relative position bias in all blocks; three-task ⅓ mix containing the bare disc task; 16k steps, cosine to 16k; expect conjunction IoU about 0.90, the dense task about 0.96, the disc about 0.99, and verify with the wrong-token and outside-disc probes.

## 8. What can be claimed today

- A learned 2-D relative position bias plus a 3-px marker turns the plain disc tasks from 4,500 to 6,500-step problems into 1,500-step problems on this model, at three radii, one run each.
- "Food & drink within 100 m of the marker" is solved by an 8.7 M-parameter, 4-block vision transformer: validation IoU up to 0.89, with a measured step at 100 m in its behaviour, a single identifiable marker head, and, in the mixed model, condition tokens that verifiably select the task.
- Both the disc marker and the bias in every block are necessary at 4 blocks; neither alone, nor the bias in one block, has ever escaped the shortcut.
- Trained alone, the conjunction escapes the shortcut in about half of the runs within 8k steps, at times between 2.5k and 7k; trained together with the plain disc task, it has escaped in every run so far, by 3k steps at the latest.
- Annealing is not required: constant learning rates of 1e-3 and 3e-4 escape and hold.
- Not claimable yet: that the mix result reproduces across seeds; that 4 blocks are needed on the mix path; the ceiling of the best recipe beyond 0.89.

## 9. Files

- Run ledger: `runs/experiments.json`. Diagnostics per run under `runs/diagnostics/<run id>/` (`eval.jsonl`, `tier1.jsonl`, prediction strips); checkpoints under `runs/checkpoints/<run id>.pt`; the step series of run 12 under `runs/checkpoints/food_drink_within_100m_L4_16k_b32_disc3_relV2/`.
- Queues in `harness/experiments.py`: `m1_report_overnight_run_queue()` (§7 batch, done 23 Sep); **`m1_report_79_run_queue()`** (§7.9 next runs).
- Probes, all under `docs/plans/perf_prototypes/`: `c3_m1_shortcut_probe.py` (candidate masks and distance profile), `c3_wrong_token_probe.py` (wrong-token test and per-token marker attention), `c3_head_sink_probe.py`, `c3_signal_coverage.py`, `c3_marker_count_probe.py`, `c3_vark_loader_probe.py`.
- Plan: `docs/plans/C3_PLAN.md`, sections 5g (task results and diagnosis), 8.6 to 8.8 (the position-bias rounds and the diagnostics of run 12).

## 10. Insights, for the next model choices

Measured on this task and this data; each line has the runs behind it in chapters 2, 7 and 7.10.

1. **Depth: 2 blocks are enough** once the task is trained inside the mix (0.907 / 0.897, two seeds). Depth only helped the single-task lottery. Start every new task at 2 blocks.
2. **Relative position bias: required, in every block.** Without it nothing left the shortcut (disc alone 0/2 at 4 blocks; bias in the first block only 0/1).
3. **Marker: a 3-px disc, not a pixel.** The pixel marker starves the patch projection; the bias cannot act on a marker the model cannot see. Both levers together, never one.
4. **Mixing: the strongest lever of all.** Train a hard marker task together with its bare marker task at uniform weights: 8 of 8 escapes across depths, horizons, schedules and seeds, median escape at 2k, against about ½ single-task. Do not weight toward the hard task (80/10/10: 2 of 3, one dead run).
5. **The condition tokens steer.** Same image, three tokens, three correct outputs (0.88 / 0.95 / 0.97), with the routing heads switched off under the token that needs no marker.
6. **Learning rate: 1e-3 cosine is fine, and annealing is not needed.** Constant 1e-3 and 3e-4 both escape and hold; 3× peak collapses at the start. 8k steps learn the task, 16k buys the last few hundredths.
7. **Batch 32 at 4 blocks, batch 32 at 2 blocks: the step is loader-bound**, not model-bound; after the 23 Sep loader fix the GPU is the cost again. Bigger batches are close to free until then.
8. **Escape timing is a random variable** with a spread of at least 3× between identical runs. Never read a lever off one run's escape step.
9. **Diagnostics that decide, in order of use:** the candidate-mask probe (which solution is this?), the outside-disc share (is it clean?), attention-to-marker per head (where is the circuit?), the wrong-token test (do the tokens steer?). Patch-local IoU is a fallback measure and should not be read as cleanliness.
10. **Count the right signal before a run.** For a conjunction the useful labels are the cells only the marker explains; a few thousand per output cell is the scale at which transitions happen. A rarer atom needs proportionally more markers or steps.
