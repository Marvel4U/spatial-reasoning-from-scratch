# Memo: teach the model a search, not a solution (iterative path finding)

Written 25 Sep 2026 from Marvin's idea, after the first c4 runs. Purpose: the starting point for a new conversation; it records the idea, why it is timely, what exists, what could go wrong, and one concrete design to argue about.

## 1. The idea, as stated

A network with two or four blocks is asked to produce the shortest path around buildings in one forward pass. It half-works at 4k steps (C4_PLAN §10): the shapes are right, the lines are blurred, and where two routes are nearly equal it lights both. Asking for arbitrary complexity in one shot invites exactly this. Language models solve hard problems by writing token after token; the spatial equivalent is to let the model *run a search*: branch out from both markers like a slime mould, mark cells at growing distance, and when the two fronts meet, retract everything except the shortest connection. Train the model on the algorithm, step by step, with an extra input plane that carries its own previous state, instead of training it on the final answer only.

## 2. Why it is timely

- The signal budget (M1 report §5.3 and chapter 10, point 10): a one-cell-wide path labels 0.3 % of the cells once. A wavefront labels every cell at every step. Dense supervision is what turned the marker tasks from lotteries into reliable runs.
- The first detour models already show the failure mode the search would remove: near-ties lit twice, blur where the model is unsure which corner to turn.
- Everything else is in place: exact targets from the raster (`spatial_data/c4_routes.py`), a loader that scatters pixel lists on the GPU, the relative position bias that makes local rules cheap to express, and the diagnostics.

## 3. What exists (read before designing)

| work | what it is | what to take from it |
|---|---|---|
| Schwarzschild et al. 2021, *Can you learn an algorithm? Generalizing from easy to hard problems with recurrent networks*; Bansal et al. 2022, *End-to-end algorithm synthesis with recurrent networks: logical extrapolation without overthinking* ("Deep Thinking") | one recurrent block applied T times solves mazes / shortest paths; trained on small mazes, extrapolates to larger ones by iterating longer; the 2022 paper adds a *recall* connection (the raw input re-injected at every iteration) and a progressive loss, which fix "overthinking" | the closest precedent; the recall connection and the incremental-progress loss are the two engineering findings to copy |
| Earle et al. 2023, *Pathfinding Neural Cellular Automata* | a local update rule iterated on the grid learns BFS-like wave propagation; beats GNN baselines out of distribution | the slime mould is a cellular automaton; locality is what makes it extrapolate |
| Tamar et al. 2016, *Value Iteration Networks*; Lee et al. 2018, *Gated Path Planning Networks* | the planning computation (value iteration) built as a recurrent convolution, trained end to end on grid worlds | a planning recurrence can be learned; gating helps optimisation |
| Veličković & Blundell 2021, *Neural algorithmic reasoning*; the CLRS benchmark (2022) | networks trained to execute classical algorithms (BFS, Bellman-Ford, Dijkstra) with *hint* supervision on intermediate states | supervise the intermediate state, not only the answer |
| Universal Transformer (Dehghani 2018), Adaptive Computation Time (Graves 2016), looped transformers (Giannou 2023), chain-of-thought | spend more passes on harder inputs; shared weights across depth | the transformer form of the same idea |

Nothing in this list makes the idea impossible; several of them make it standard. What none of them has is a real city raster with several layers and condition tokens, which is our contribution.

## 4. What could go wrong

1. **Cost.** T passes per sample. With T = 8 and 2 blocks the step costs what a 16-block model costs. Acceptable at our sizes; the loader is no longer the bottleneck.
2. **One-shot shortcut.** Global attention lets the model ignore the recurrence and try to solve everything in pass 1. The test is extrapolation: train on routes ≤ 120 m, evaluate at 250 m with more passes. If the extra passes do not help, the model did not learn the search. The relative bias with a small K (offset clip) is the knob that pushes toward local rules.
3. **Overthinking / drift.** Iterating a learned rule past the point of convergence can degrade the state (Bansal 2022). Recall connection + loss at every pass are the known remedies; a halting signal is the elegant one but not needed first.
4. **Two computations, not one.** Expansion (fronts grow) and retraction (backtrace along the meeting ridge) are different rules. Supervise them separately, or supervise only the distance field and read the path off it by a fixed rule (see 5.2).
5. **Ties.** Near-equal routes are real in a grid city. The distance-field formulation handles them naturally (both routes lie on the ridge); the path formulation must record the second-best ratio and stratify.

## 5. One concrete design to argue about

### 5.1 State and targets

- **State plane**: one extra input channel (channel 17) carrying the model's previous output, zero at pass 0. Model weights shared across passes.
- **Target of pass t**: the **geodesic distance field** from the two markers, computed exactly on the 1 m raster (wavefront / Dijkstra with 16-connectivity or a fast-marching solver for Euclidean distances), then banded into classes like the sun plane (e.g. 8 bands of 20 m, plus "unreached" and "building"). Pass t's target is the field clipped at radius r_t = t · Δr: the front at step t. This is the hint supervision of CLRS in raster form.
- **Final target**: the meeting ridge, cells where d_A + d_B ≤ d_AB + ε, which *is* the shortest path (all shortest paths, so ties come for free), plus optionally the c4b line for comparison with the one-shot models.

### 5.2 Passes

- T fixed at first (T = number of bands), curriculum from short routes.
- Loss on every pass (distance bands, CE + Dice) plus the final ridge loss; the recall connection is free because the raw planes are re-fed at every pass anyway.
- Readout: the path is the ridge of the two predicted fields, a fixed rule, no learned retraction needed in version 1. Version 2 learns the retraction as a second task token.

### 5.3 Evaluation

- Per pass: band accuracy of the front; does the front advance like the true wavefront?
- Final: the tolerant line metrics (C4_PLAN §10) on the ridge, stratified by detour ratio.
- **Extrapolation**: train ≤ 120 m, test at 250 m with T doubled. This is the claim that matters: more thinking, better answer.

### 5.4 What is reused unchanged

Model (2 blocks, disc marker, bias all), the c4 store (windows, markers, buildings), the loader (one more channel and a loop over passes), the diagnostics (per-pass attention-to-marker will show whether the routing head is re-used at every pass).

## 6. Decisions this needs from Marvin

1. Distance field as the supervised state (proposed) vs the raw path as the state.
2. T and Δr for the first run (proposed T = 8, Δr = 20 m, routes ≤ 160 m).
3. Whether the final path is read off the ridge by rule (proposed) or predicted by the model as a second task.
4. Where this sits relative to c4b one-shot: a parallel track for the report ("one-shot vs iterative"), which is the clean portfolio comparison.


## 7. Infrastructure shape (added 25 Sep after Marvin's reading of §1–6)

Marvin's additions to the idea, taken as requirements: (a) a separate input that carries the model's own previous state, and the model has to *learn* that this input is special; (b) a specific way to train the start, because at pass 0 there is no previous state; (c) a learned **stop**: a set of output tokens, not language, that are fed back in the next pass alongside the task tokens, one of which means "done, this is as good as it gets"; (d) training as an iterative problem in the way a language model is trained on text, revealing the solution step by step, with the extra freedom that our model may be rewarded for doing two steps at once; (e) the sequence: search, connection found, retract the other arms, refine the best route, stop. This section gives one concrete shape for all of that. Where two designs are possible, the first is the proposal and the second is named.

### 7.1 The state: what the model reads back and writes out

The state is a small stack of planes at the output resolution (64 × 64 cells), written by the model at pass t and fed back as extra input channels at pass t + 1. Three planes:

| plane | classes | meaning |
|---|---|---|
| `band` | `unknown` + B distance bands | geodesic distance to the nearest marker, banded (Δr = 16 m → B = 16 bands cover 256 m); `unknown` = not reached yet |
| `owner` | `none`, `A`, `B` | which marker's front reached the cell; A = the left-most marker (ties: top-most), a fixed rule the model can compute from the marker plane |
| `path` | `off`, `on` | the retained route; empty during the search, filled at the meeting, the only thing left at the end |

Why this and not the raw path: the band plane is a dense target on every cell at every pass, the owner plane makes the meeting of two fronts an explicit event (a cell whose neighbours are owned by A and B), and the path plane is the retraction target. All three are computable exactly from the store by one wavefront per marker.

Input side: the three planes are one-hot encoded and appended to the 16 image channels (channel count 16 + (B + 1) + 3 + 2 = 36 at B = 16). The model learns that they are special the same way it learned the marker plane: fixed channel positions, and a loss that only makes sense if it reads them. Two things help it: at pass 0 the planes are all `unknown` / `none` / `off`, which is a distinct, recognisable input, and the phase token (7.2) tells it which pass it is in.

Output side: the grid head grows from 2 classes per cell to three groups of logits, (B + 1) + 3 + 2, sliced into the three planes. One head, one trunk; nothing else in the model changes.

### 7.2 Tokens: phase in, phase out, and the stop

A small vocabulary of discrete tokens, produced by the model at every pass and fed back at the next:

- **phase** ∈ {`start`, `explore`, `meet`, `retract`, `done`}: what the model believes the situation is. `done` is the stop criterion: when the model emits it, iteration ends and the `path` plane is the answer.
- **step bin** ∈ {0, 1, 2, 3, 4+}: how many passes have run, coarsely. Cheap insurance against the model losing count.

Mechanics: the two tokens are two more slots in the prefix, next to the two task-condition tokens, embedded from a vocabulary that extends the existing condition vocabulary. For the output, one extra learned **readout slot** is appended to the prefix; its final hidden state goes through a linear head to phase logits and step-bin logits. This is the standard "CLS token" pattern; the grid head keeps reading only the patch slots, as today.

At pass 0 the input tokens are (`start`, bin 0) by construction; the model's job at pass 0 is to seed the fronts and emit `explore`.

### 7.3 Trajectories: the training data

Everything is generated deterministically from the existing c4 store plus one wavefront per marker, and stored per sample as two int8 planes (band from A, band from B) at 64 × 64; every state along the trajectory is derived from these on the GPU by a few tensor operations, so the loader stays fast.

The teacher trajectory for a sample, with d_A and d_B the banded fields, D the band of the true shortest length, and r_t the front radius after t explore passes:

| phase | pass | state target | token target |
|---|---|---|---|
| start | 0 | band = `unknown`, owner = `none`, path = `off` (this is the *input* of pass 0) | (`start`, 0) as input |
| explore | 1 … | band = min(d_A, d_B) where ≤ r_t, else `unknown`; owner = which of the two is smaller where reached, else `none`; path = `off`; r_t = t · Δr | `explore` |
| meet | t* = first t with r_t ≥ D/2 | as explore, plus path = `on` on the ridge cells with d_A + d_B = D (all shortest paths, so ties are honest) | `meet` |
| retract | t* + 1 … t* + R | path unchanged; band and owner cleared from the outside in, one band per pass (r shrinks by Δr) until only the ridge is left | `retract` |
| done | t* + R + 1 | band = `unknown`, owner = `none`, path = the ridge | `done` |

R, the number of retract passes, is a rendering choice: R = 1 clears everything at once and is the fastest; R = r_{t*} / Δr retracts band by band and is the "slime mould pulling its arms back" that Marvin wants to see. Both are exact targets; the choice does not change the computation the model has to learn, only how many passes it spends on the show.

The "refine the best route" step of Marvin's sequence is implicit: the ridge is already the exact set of shortest routes. If a learned refinement is wanted later (for example to pick one of two tied routes), it is a second path plane trained on one canonical choice.

Volume: 20k (or 100k) samples × (t* + R + 1) passes, typically 6 to 20 passes per sample, all derived on the fly. Two int8 planes per sample is 8 KB, negligible.

### 7.4 The loss: "at least one step, more if you can, never wrong"

This is the part that encodes "reward two steps at once" without reinforcement learning, and it answers the question whether the local reward is the hard part: with the distance field it is not, because the correct value of every cell is known.

Per pass, given the input state at radius r_{t−1} (or the model's own state, see 7.5), the target is the **full** field, and cells are grouped by their true distance:

| cells with true distance | what the model must do | loss |
|---|---|---|
| ≤ r_{t−1} + Δr (the mandatory ring: at least one step) | be known and correct | CE against the true band and owner; predicting `unknown` here is an error |
| between r_{t−1} + Δr and r_{t−1} + K·Δr (allowed: up to K steps ahead) | may be `unknown`, but if known must be correct | CE against the true band and owner, masked out where the model predicts `unknown` |
| beyond r_{t−1} + K·Δr | must stay `unknown` | CE against `unknown` (keeps the front a front; K = ∞ removes this row and lets the model jump to the answer if it can) |

Plus: the path plane loss (CE + Dice, as for every thin target) once the phase is `meet` or later; the phase and step-bin token losses (CE) at every pass; and a small **pass cost** term, λ per pass until `done`, which is what makes taking two steps at once worth anything. Without λ the model has no reason to go beyond the mandatory ring; with λ it learns to jump as far as it can be correct. λ is a curriculum knob: 0 while learning the rule, then raised.

Two properties worth stating. The loss is *asymmetric on purpose*: hallucinating a band beyond what can be known is punished, being cautious inside the allowed window is free. And it is *local in the sense that matters*: every term is a per-cell classification against an exact label; the only global quantity is the pass count.

### 7.5 The training loop: teacher forcing first, own rollouts second

Exactly the language-model recipe, with the state in place of the text prefix:

1. **Teacher forcing.** For each sample draw a pass index t at random (weighted toward small t at first, see 7.6), build the *true* state at t − 1 as input, train the model to produce the state and tokens at t under the loss of 7.4. Every pass is an independent training example; no backpropagation through passes. This is cheap (one forward/backward per example, batch 128 as today) and it is how the model learns the rule.
2. **Own rollouts.** Once teacher-forced accuracy is high, mix in examples where the input is the model's *own* state from a rollout (run the model for t passes without gradient, take its output as the input for the trained pass). This is scheduled sampling / DAgger: it teaches the model to correct its own mistakes instead of only the teacher's clean states. Mixing ratio from 0 to about ½ over training.
3. **No BPTT at first.** Backpropagating through several passes (the Deep Thinking way) is an option for later; with the loss of 7.4 each pass has its own exact target, so the gradient does not need to travel through time to be informative. This keeps memory at one pass and the step cost at what it is today.

Rollout length at evaluation: run until the model emits `done` or until T_max = 24 passes.

### 7.6 Curriculum: teach the start first

The start is a distinct sub-task and it is trained as one: in the first phase of training, pass 0 → pass 1 examples (empty state in, first ring around both markers out, `explore` token out) are over-sampled until the model seeds the fronts at the markers reliably; that is the two-marker detection m1 taught us to build first. Then the pass distribution flattens; then the meeting and retraction passes get their share; λ (7.4) is switched on last. Route length follows the same curriculum as c4: routes ≤ 120 m first, all lengths after.

### 7.7 Evaluation

- **Per pass, teacher-forced**: band and owner accuracy on the mandatory ring; hallucination rate beyond the allowed window; phase-token accuracy.
- **Own rollout**: passes until `done` (against the teacher's t* + R + 1); stop accuracy (early / late / never); the final `path` plane under the tolerant line metrics of the c4 report, stratified by detour ratio; ties are automatically handled because the ridge contains all shortest routes.
- **Extrapolation, the claim that matters**: train on routes ≤ 120 m, evaluate on 120–250 m with T_max raised. If the rollout keeps improving with more passes on routes longer than any seen in training, the model learned a search and not a lookup. This is the Deep Thinking test and the one number for the portfolio.
- **Diagnostics**: attention-to-marker per pass (is the routing head reused at every pass?), and a strip that shows the whole rollout of one sample as a film strip, which is also the showmanship output.

### 7.8 Case (b): tentacles

Marvin's second rendering, squiggly branches that grow from the markers, is a different kind of target: there is no single correct tentacle set, so a per-cell supervised loss cannot define it. Making it a real learning problem means either sampling (a stochastic model trained to match a distribution of branch sets, i.e. a diffusion- or flow-style objective) or reinforcement learning (reward when a branch connects, penalty per pass and per cell of growth). Both are much heavier than 7.4 and neither is needed for the computation.

Proposal: keep the computation as in 7.1–7.7 and, if the organic look is wanted, put it in the **rendering** of the trajectory: the teacher can grow the fronts along a randomised subset of directions per pass (a stochastic wavefront that still respects the true distances, so every cell it lights is correct) and the model learns to reproduce that texture. The loss of 7.4 already permits it: anything correct inside the allowed window is fine, so a model that grows a few arms first and fills in later is not punished. That gives the appearance of (b) with the guarantees of (a). A genuine (b) is a separate project after (a) works.

### 7.9 What is actually the hard part

Not the reward: with the distance field every cell has an exact label at every pass. The hard parts, in order of expected trouble:

1. **Own-rollout drift**: the model's state after a few of its own passes looks different from any teacher state; 7.5 step 2 is the remedy and the extrapolation test is where it shows.
2. **Stopping**: `done` too early leaves a broken path, too late is only slow; the token loss plus the pass cost balance it, and the stop-accuracy metric watches it.
3. **The meeting**: the ridge appears in one pass and is the sparsest target in the trajectory; it inherits the thin-line difficulty of c4b. The tolerant metrics and Dice apply; R > 1 gives the model several passes with the ridge as target.
4. **Capacity**: the input has 36 channels and the head three planes; 2 blocks may not suffice, and this is a case where a depth run is justified from the start.

### 7.10 Build order (data and harness Claude, model Marvin, runs Marvin)

| # | item | owner |
|---|---|---|
| I1 | wavefront fields per store sample (`d_A`, `d_B` bands at 64 × 64, 16-connected grid Dijkstra on the 1 m raster, or FMM), written next to the store; teacher trajectories derived on the fly | Claude |
| I2 | loader: state planes one-hot, random pass index with curriculum weights, teacher state in / target state out, token targets; own-rollout mixing as a loader flag that calls the model | Claude |
| I3 | loss of 7.4 with the three rings and λ; metrics of 7.7 | Claude |
| I4 | model: 36-channel input, three-plane grid head, phase and step tokens in the prefix vocabulary, readout slot and token head | Marvin |
| I5 | first run: teacher forcing only, routes ≤ 120 m, R = 1, K = 2, λ = 0; pass: mandatory-ring accuracy > 0.95 and phase accuracy > 0.95 | Marvin |
| I6 | own rollouts, λ > 0, all lengths, R band-by-band; the extrapolation test | both |

Decisions for Marvin before I1: Δr (proposed 16 m → 16 bands), K (proposed 2, i.e. up to two steps ahead), R (proposed 1 for I5, band-by-band for I6), and whether `owner` uses the left-most rule (proposed) or a random assignment that the model must carry through its own state.
