# LOCAL_MODEL_GUIDE — how a vision model works, built from `plain_gpt_module`

Written for Marvin, 20 Sep 2026. Baseline assumed: you understand every line of `plain_gpt_module` and nothing about vision models. This file explains, it does not decide. Snippets are illustrations for you to type, change, or reject; no code for the local model exists anywhere in the repo. Where a real choice exists it is marked **CHOICE** and collected in §10.

---

## 1. The one idea

Your `Seq2Seq.encode` starts like this:

```python
x = self.wte(src_ids) + self.wpe_src(pos)      # (B, T_src) ints  ->  (B, T_src, n_embd)
for block in self.encoder:
    x = block(x, key_padding_mask=src_key_padding_mask)
```

Everything after the first line only ever sees a tensor of shape `(B, T, n_embd)`: a batch of sequences of vectors. The blocks do not know those vectors came from words. A vision transformer (ViT) is the observation that you can produce such a sequence from an image, and then **reuse the rest unchanged**:

```
text :  ids (B, T)            --wte lookup-->        (B, T, n_embd)  + position  -> encoder blocks
image:  pixels (B, Cin, H, W) --cut into patches,
                                one Linear layer-->   (B, N, n_embd)  + position  -> encoder blocks
```

So the whole "vision" part is two things: (a) how pixels become a sequence of vectors, and (b) how to tell the model where each vector sat in the 2-D image. §2 and §4. Everything else is your existing code plus a way to ask a question (§6) and get an answer out (§7).

## 2. Patch embedding: the replacement for tokenizer + `wte`

**Why patches at all.** Attention costs O(T²). A 256×256 image taken pixel by pixel is T = 65,536 tokens, so T² ≈ 4·10⁹ attention entries per head per layer. Hopeless. Cut the image into non-overlapping P×P squares instead: with P = 16 a 256 px image gives a 16×16 grid, N = 256 tokens, the sequence length you trained the translator at.

**What a patch token is.** Take one patch: P·P pixels, each with Cin channel values. Flatten it to one vector of length P·P·Cin and multiply by one learned matrix. That matrix is the entire "vision front end":

```python
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, patch, n_embd):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch * in_chans, n_embd)

    def forward(self, img):                                   # (B, Cin, H, W) float
        B, Cin, H, W = img.shape
        P = self.patch
        x = img.view(B, Cin, H // P, P, W // P, P)            # split H into (grid_row, in_patch_row), same for W
        x = x.permute(0, 2, 4, 1, 3, 5)                       # (B, grid_row, grid_col, Cin, P, P)
        x = x.reshape(B, (H // P) * (W // P), Cin * P * P)    # (B, N, P*P*Cin), patches in row-major order
        return self.proj(x)                                   # (B, N, n_embd)
```

Compare with `wte`: an embedding table is a linear layer applied to a one-hot vector. `wte` maps a 1-of-V vector to n_embd. `PatchEmbed.proj` maps a dense P·P·Cin vector to n_embd. Same role, dense input instead of one-hot. There is no tokenizer and no vocabulary on the input side, which also means **no weight tying** between input and output (your `tie_embeddings` has nothing to tie to on the encoder side).

You will see `nn.Conv2d(Cin, n_embd, kernel_size=P, stride=P)` in every ViT codebase. It is exactly the same operation: a convolution whose stride equals its kernel visits each non-overlapping patch once and applies one shared linear map to it. Same parameter count, same result up to weight layout. The explicit version above is better for understanding; the conv is a drop-in later.

**The number of channels lives only here.** RGB models have Cin = 3 because cameras do. Nothing else in the network depends on it. Your multi-channel idea is literally `in_chans=N`.

**The thing to understand before choosing P** (this matters for our first task). The linear layer sees all P·P·Cin numbers of a patch at once, so no pixel is thrown away at the input. But the patch must be *compressed* into n_embd floats, and everything downstream only has that vector. With P = 16 and, say, 12 one-hot channels, that is 3,072 numbers squeezed into 384. Our first task asks for the class of **one specific pixel**. To answer, the model must keep per-pixel detail recoverable inside the patch vector, for every pixel, because it does not know in advance which one will be asked about. A linear map from 3,072 to 384 dimensions cannot be inverted in general; it works here only because the input is highly structured (few classes, large uniform regions), and it will fail first exactly where you'd expect: near class boundaries, thin sidewalks, 4 px establishment discs. Smaller patches ease this (P = 8 → N = 1,024; P = 4 → N = 4,096) at quadratic attention cost. **CHOICE: patch size**, and it is an experiment worth running rather than guessing: accuracy vs distance-to-boundary (stored per item) will show the effect directly.

## 3. What the input tensor is

One crop on disk (`{id}_labels.npz`) holds integer class rasters per layer, e.g. `surface_256` with values 0..3. Two ways to hand that to `PatchEmbed`:

- **One-hot**: each layer with K classes becomes K binary channels. Three layers × 4 classes → Cin = 12. The linear layer can read "is this pixel sidewalk" as one input number. No false ordering between classes (building is not "3× roadway").
- **Index as a number**: one channel per layer holding 0, 1/3, 2/3, 1. Cin = 3. Compact, and fine for noise, which really is ordered. For surface it imposes an order that means nothing, and the model has to un-learn it.

```python
surface = torch.from_numpy(npz["surface_256"]).long()            # (H, W) ints 0..3
onehot  = F.one_hot(surface, num_classes=4).permute(2, 0, 1)      # (4, H, W)
img     = torch.cat([onehot_noise, onehot_surface, onehot_estab], dim=0).float()   # (12, H, W)
```

**CHOICE: one-hot vs index, per layer.** This conversion is *external* to the model: it belongs in the data loader.

Resolution is also a data-side choice, not a model property: each crop is stored at 1024 px (0.25 m/px) and 256 px (1 m/px). N = (H/P)². A 1024 px input with P = 16 is 4,096 tokens.

## 4. Position in two dimensions

Self-attention without position information is permutation-equivariant: shuffle the input tokens and the outputs shuffle the same way, nothing else changes. You know this from the GPT; it is why `wpe` exists. For an image it means: **the position embedding is the only place the model learns that the picture is 2-D at all.** Patch 17 being directly below patch 1 (in a 16-wide grid) is not built in anywhere.

Three standard options, all added to the patch vectors exactly where you add `wpe_src`:

```python
# (a) learned 1-D table, row-major order. Your wpe_src, unchanged. The original ViT did this.
pos = self.wpe(torch.arange(N, device=x.device))                  # (N, n_embd)

# (b) learned, factorised: one table for the row, one for the column
rows = torch.arange(g, device=x.device).repeat_interleave(g)      # 0,0,..,0,1,1,..   (N,)
cols = torch.arange(g, device=x.device).repeat(g)                 # 0,1,..,g-1,0,1,.. (N,)
pos  = self.row_emb(rows) + self.col_emb(cols)                    # (N, n_embd)

# (c) fixed 2-D sine-cosine: half the dimensions encode y, half encode x, no parameters
def sincos_2d(g, dim, temperature=10000.0):
    y, x = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(dim // 4) / (dim // 4))        # (dim/4,)
    y = y.flatten()[:, None] * omega[None, :]                                 # (N, dim/4)
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)             # (N, dim)
```

(a) works: with enough data the table learns the grid (the ViT paper found no gain from 2-D-aware variants on ImageNet). It has to discover from data that tokens i and i+g are neighbours. (b) and (c) give it the row/column structure for free; (c) has no parameters and, being smooth in x and y, makes "nearby position → similar code" true from step 0. Our tasks are about *where* things are (a queried coordinate, later paths), more than ImageNet ever is, so this choice plausibly matters more for us than in the literature. **CHOICE: position scheme.** Cheap to swap, a good early ablation.

## 5. The encoder: your `EncoderBlock`, unchanged

```python
cfg = Seq2SeqConfig(vocab_size=V, n_embd=384, n_head=6, enc_n_layer=6, dec_n_layer=2, max_tgt_len=8)
self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.enc_n_layer)])
...
x = self.patch_embed(img) + pos
for block in self.encoder:
    x = block(x)                        # key_padding_mask=None
x = self.ln_enc(x)                      # see note below
```

- **No causal mask**, for the reason your translator's encoder has none: there is no generation order among patches. Every patch may look at every other from layer 1. That is `BidirectionalSelfAttention` as it stands.
- **No padding mask.** All images have the same size, so N is constant and `key_padding_mask` stays `None`. One source of bugs gone.
- **A difference worth knowing:** your encoder is pre-LN and `Seq2Seq.encode` returns the residual stream without a final LayerNorm (only the decoder side has `ln_f`). ViTs normally put one LayerNorm on the encoder output before any head reads it. If a head reads the encoder directly (§7a, §7c) you want that `ln_enc`. If only your decoder reads it via cross-attention it has worked without one in the translator, so it is optional there.
- **What comes out:** `(B, N, n_embd)`, one vector per patch, *still in patch order*. The residual stream keeps token identity: output token i is "patch i, now informed by all other patches". This is the fact that makes dense outputs (§7c) possible.

**Inductive bias, the one piece of theory worth carrying around.** A CNN hard-codes two assumptions: nearby pixels matter most (small kernels) and the same pattern means the same thing anywhere (weight sharing → translation equivariance). A ViT hard-codes neither; beyond the patch it must learn locality and translation from data. That is why ViTs lose to CNNs on small datasets and win on huge ones. Our data is effectively unlimited (crops × sampled questions), which is the regime where this weakness mostly disappears. It is also why many small ViTs put a few conv layers *before* the transformer (a "conv stem") instead of one big linear patchify: it buys back locality. **CHOICE (later): plain patchify vs conv stem.** Start plain; it is the version you can fully reason about.

## 6. Getting the question in

The first task: *given a crop and a point, which class is at that point in layer L?* The model needs three pieces of information besides the image: which task, and the point's x and y. Options:

- **(A) Marker channel.** Add one more input channel, zero everywhere and 1 at the queried pixel (or a small blob). The question becomes part of the picture: `Cin = 13`. Locating the point is then trivial for the model (one patch lights up), so this variant tests *reading the layers*, not *finding a coordinate*.
- **(B) Coordinates as tokens.** Give the decoder `[<task>, <x_bin>, <y_bin>]` as its input sequence, using your ordinary `wte`. The model must learn the mapping from the number 137 to "the patch in column 8, and pixel 9 inside it", i.e. it must connect token identity to the position code of §4. That is a real skill, and it is the same skill the Qwen track is being tested on (a coordinate given as text).
- **(C) Both**, as two experimental conditions. The gap between A and B measures how hard coordinate grounding is, separately from reading.

The task id is a token either way (`<surface_at>`, `<noise_at>`), playing the role a language tag plays in a multilingual translator: same encoder, the decoder is told what is wanted. **CHOICE: A, B, or both.**

## 7. Getting an answer out: three heads

**(a) Classification head, no decoder.** The smallest thing that can work.

```python
feats  = self.ln_enc(x)                         # (B, N, n_embd)
pooled = feats.mean(dim=1)                      # (B, n_embd)   global average pooling
logits = self.head(pooled)                      # nn.Linear(n_embd, n_classes) -> (B, n_classes)
loss   = F.cross_entropy(logits, target_class)  # target_class: (B,) ints
```

Instead of the mean, many ViTs prepend one extra learned vector (the "CLS token", `nn.Parameter(torch.zeros(1, 1, n_embd))`, concatenated in front of the N patch tokens so T = N+1) and read the answer from its final state. It works because attention lets that token collect whatever it needs from all patches. Mean pooling is simpler and at small scale at least as good. For *our* task note the problem: the answer lives in one patch, and averaging 256 patch vectors dilutes it. A sharper variant uses the fact from §5 that outputs stay in patch order:

```python
patch_idx = (row // P) * g + (col // P)                         # (B,) which patch holds the point
feat      = feats[torch.arange(B, device=feats.device), patch_idx]   # (B, n_embd)
logits    = self.head(feat)
```

Here the *where* is answered by indexing, so no question encoding is needed for the patch, only for the position inside the patch (marker channel). This is the cleanest possible test of §2's compression question, and a good first rung. It cannot express multi-token answers.

**(b) Your decoder.** `DecoderBlock` and the `decode` path as they are. The "target sentence" is tiny:

```
decoder input :  <surface_at>  <x:137>  <y:052>
targets       :  pad           pad      <surface:2>       # pad_id positions are ignored by your loss
```

Causal self-attention lets the last position see task, x and y; cross-attention lets it read the patches (`memory` = encoder output); `lm_head` + cross-entropy with `ignore_index=pad_id` is exactly `Seq2Seq.forward`. The vocabulary is a few hundred tokens: pad, task ids, coordinate bins, class names. The point of paying for a decoder on a task this small: **the answer format can grow without touching the architecture.** A point answer is two coordinate tokens; a path is a sequence of them (this is the Pix2Seq idea: treat coordinates as words). Classes, counts, points and paths all become "translate image → short token sequence", trained with the loss you already have.

**(c) Dense output grid: the model *points*.** Your 64×64 idea. Since encoder outputs stay aligned with patches, put a linear layer on every patch vector and reshape:

```python
feats  = self.ln_enc(x)                          # (B, N, n_embd), N = g*g
logits = self.mask_head(feats)                   # nn.Linear(n_embd, s*s*K) -> (B, N, s*s*K)
logits = logits.view(B, g, g, s, s, K).permute(0, 1, 3, 2, 4, 5).reshape(B, g*s, g*s, K)   # (B, 64, 64, K)
loss   = F.cross_entropy(logits.reshape(-1, K), target_grid.reshape(-1))    # target_grid: (B, 64, 64) ints
```

With P = 16 (g = 16) each patch predicts an s×s = 4×4 sub-grid, giving 64×64 cells of 4 px (4 m) each; with P = 4 it is one cell per patch and `s = 1`. K = 2 for "mark the quiet sidewalk" (in/out), or K classes. This is semantic segmentation in its simplest form. The target is computed from the label rasters (e.g. `sidewalk & noise<55`, downsampled to 64×64), the score is IoU. No decoder, no tokens; the task id can enter as a learned vector added to every patch token, or as a CLS-like extra token. It is the natural output for area questions and, later, for drawing a path as a mask.

(a) is the quickest to get working, (b) is the general one, (c) is your pointing experiment. They share the encoder, so nothing built for one is wasted. **CHOICE: which first.**

## 8. What is reused, what changes, what is outside the module

| Piece | Status |
|---|---|
| `MLP`, `BidirectionalSelfAttention`, `CrossAttention`, `CausalSelfAttention` | reused as is |
| `EncoderBlock`, `DecoderBlock` | reused as is (they take a `Seq2SeqConfig`; `vocab_size` is a required field even if only the encoder is used) |
| `_init_weights` pattern, `NANOGPT_SCALE_INIT` | reused; it initialises every `nn.Linear` incl. the patch projection. If you ever switch to `nn.Conv2d` for patchify, note that your init function does not touch Conv2d (it would keep PyTorch's default) |
| `configure_optimizers`, `get_lr`, checkpoint code | reused as is |
| `Seq2Seq` | **not edited.** The vision model is a new class next to it. What it replaces is only `encode`'s first line (`wte(src_ids) + wpe_src(pos)` → `patch_embed(img) + pos_2d`) and, depending on the head, the output side |
| `wte`, `wpe_tgt`, `lm_head`, tying | kept only if you use the decoder (§7b); then they serve the tiny answer vocabulary |
| `train()` | **needs a small change or a sibling.** It does `x, y = train_loader.next_batch(); x.to(device); model(x, y)` and logs `B*T` tokens/s. An image batch is (image, question tokens, targets): three tensors, and "tokens per second" becomes "samples per second". The loop body (autocast, accumulation, clipping, LR schedule) is unchanged |
| `TokenBatchLoader` etc. | not applicable. A new loader, **outside the module**, reads crops (`*_labels.npz`) and task items (`tasks/.../*.jsonl`), builds the one-hot tensor (§3), the marker channel or coordinate tokens (§6), and the target |
| Evaluation | outside the module: accuracy overall and binned by `boundary_dist_px`; IoU for the grid head |

So inside `plain_gpt_module` (or a sibling folder, your call) the genuinely new code is `PatchEmbed`, a position function, one model class with a `forward`, and a generalised train step. Roughly 100 to 150 lines.

## 9. One concrete shape walk-through

Illustration only (256 px input, one-hot 12 channels + marker, P = 16, n_embd = 384, decoder head):

```
img                      (B, 13, 256, 256)
PatchEmbed  view/permute (B, 256, 13*16*16 = 3328)
            proj         (B, 256, 384)            + pos_2d (256, 384)
6 x EncoderBlock         (B, 256, 384)            attention matrix per head: 256 x 256
memory                   (B, 256, 384)
decoder in  ids          (B, 3)  -> wte + wpe_tgt -> (B, 3, 384)
2 x DecoderBlock         (B, 3, 384)              cross-attention: 3 queries x 256 keys
lm_head                  (B, 3, V~300)            loss on the last position only
```

Parameter count is dominated by the blocks: about 1.8 M per encoder block at n_embd = 384 (4·C² attention + 8·C² MLP = 12·C²), so 6 layers ≈ 10.6 M, plus the patch projection 3328×384 ≈ 1.3 M. Small enough that the 3060 is not the constraint.

## 10. Sanity ladder (the order that catches bugs cheaply)

1. **Overfit one batch** to zero loss. If it cannot, there is a bug, not a modelling problem.
2. **A task with no question**: "class at the centre pixel". Tests patchify, channels, the head. Should reach ~100 % away from boundaries.
3. **Indexed read-out** (§7a, sharp variant) on the real point task. Tests the compression question of §2 in isolation. Vary P.
4. **Question encoded** (marker channel, then coordinate tokens). Tests §4 and §6.
5. Only then cross-layer tasks and the grid head.

Things worth plotting along the way: accuracy against `boundary_dist_px`; the learned position table's cosine similarity between one patch and all others, reshaped to the grid (you should see a 2-D blob if the model has discovered the geometry); cross-attention weights of the answer position over the 16×16 patches (should peak at the queried patch).

## 11. The decisions that are yours

1. Patch size P (and with it N and the input resolution, 256 or 1024 px).
2. One-hot vs index encoding, per layer.
3. Position scheme: learned 1-D, learned row+col, fixed 2-D sin-cos.
4. How the question enters: marker channel, coordinate tokens, or both as conditions.
5. Which head first: pooled/indexed classifier, your decoder, or the 64×64 grid.
6. Where the code lives: new files inside `plain_gpt_module`, or a sibling package that imports from it.
7. Model size. Your translator's 384 / 6 heads / 6 layers is a sensible default, not a recommendation.

Earlier documents got ahead of you here: the Track B column in `PLAN.md` and the Track B fields in the task items (`task_token`, `marker_256`, `target_token`) are **proposals**, cheap to regenerate once you have decided.

## 12. If you want to see one reference implementation first

`lucidrains/vit-pytorch`, file `simple_vit.py`: about a hundred lines, patchify + fixed 2-D sin-cos + plain encoder + mean pool. Reading it after this guide should hold no surprises. Pix2Seq (arXiv 2109.10852), sections 2–3, for coordinates-as-tokens. More in `survey/raw/07_vision_from_scratch_resources.md`.

---

## 13. Questions and answers (Marvin, 20 Sep)

Decisions taken after this round are in `LOCAL_EXPERIMENTS.md`.

**Is the patch projection the same for every patch?** Yes. One `nn.Linear`, applied along the last axis of `(B, N, P·P·Cin)`, so the same matrix hits all N patches, exactly as one `wte` table serves every position in a sentence. Where a patch sits enters only through the position code added afterwards. This weight sharing is the one spatial prior a plain ViT has: a pattern is embedded identically wherever it occurs.

**What does `permute(0, 2, 4, 1, 3, 5)` do; why 0–5 when patches are 16×16?** The numbers are *axis indices* of a 6-D tensor, not pixel indices. After `view` the axes are `0 batch, 1 channel, 2 grid_row, 3 in_patch_row, 4 grid_col, 5 in_patch_col` with sizes `(B, Cin, 16, 16, 16, 16)`: a 256 px image is 16×16 patches of 16×16 pixels, and both numbers being 16 is a coincidence of 256/16 that makes this confusing. `reshape` can only merge *adjacent* axes in memory order. We need to merge (grid_row, grid_col) into N and (channel, in_patch_row, in_patch_col) into one patch vector, so the axes first have to be reordered to `(B, grid_row, grid_col, Cin, in_row, in_col)`. Without the permute, the reshape would run without error and silently mix pixels from different patches into one vector.

**Is the patch projection pure compression, no skip connection?** Correct: one linear map, no nonlinearity, no path from raw pixels to later layers. Two refinements. (1) It compresses only if P·P·Cin > n_embd. P = 16, Cin = 13 → 3,328 into 384: compressing. P = 4 → 208 into 384: *expanding*, nothing need be lost. (2) From the patch vector onwards everything is a residual stream, `x = x + attn(...)`, so the patch vector itself is carried by skip connections to the last layer; the pixels are not. If the grid head later lacks fine detail, the standard remedy is a skip from the input (or an early layer) to the output head, the U-Net idea. A later option, not a starting point.

**One-hot input vs "evenly distributed activation": where does the embedding happen?** In the patch projection; there is no additional level. A one-hot vector times a matrix selects one column: that is `wte`. A patch of one-hot pixels times the projection matrix selects one column *per pixel* and **sums** them (verified numerically): `patch_vec = Σ_pixels W[:, (in_patch_position, class_of_that_pixel)] + b`. So every (position-in-patch, class) pair owns a learned embedding vector, and a patch is the sum of 256 of them: dense and distributed. This also restates the compression problem: 256 positions × 12 classes = 3,072 vectors living in 384 dimensions cannot all be orthogonal, so the sum is not perfectly decodable. Practical note: many ViTs put a LayerNorm right after the patch projection to fix the scale of that sum.

**Processing the layers independently.** One thing to know first: a single Linear over concatenated channels *already is* "one independent projection per layer, summed", because `W·[a; b] = W_a·a + W_b·b`. Separate per-layer projections that are then added change nothing. Independence only becomes real if the layers **stay separate tokens**: each layer is patchified on its own into N tokens, a learned *layer embedding* is added (like the position code, but saying "I am noise"), and the sequence is L·N long; attention does the fusion. This exists in the literature for multispectral and microscopy images ("ChannelViT"). It is also the natural home for the later wish to swap layers: the layer embedding *is* the meta-information, layers can be dropped or added without touching the patch projection, and the embedding could later come from a description. Cost: L times more tokens, hence larger patches to keep T fixed.

**How can sine position codes locate anything, if a sine is ambiguous?** A single sine is ambiguous, twice over: within a period two positions share a sine value, and the pattern repeats every wavelength. The code fixes both. (1) Every frequency comes as a **sin/cos pair**, which is a point on a circle, i.e. a phase angle: unambiguous within one wavelength. (2) It uses **many frequencies in a geometric series**, here wavelengths from 6.3 positions up to tens of thousands. Think of clock hands, or the digits of a number: the fast ones give fine resolution but wrap around, the slow ones never wrap across the whole range but are coarse. Together they are unique. With only 16 positions per axis most of the slow pairs cover a small fraction of their circle, so they are monotone in position: no ambiguity at all. Two properties make the code *useful* rather than just unique: the dot product of two codes is `Σ_k cos(ω_k (p − q))`, which depends only on the offset and peaks at zero (checked: position 5 has similarity 1.0 with itself, 0.9 with neighbours, decaying to 0.7 far away), so "nearby" means "similar" from step 0; and code(p + Δ) is a fixed rotation of code(p), so a head can learn "look Δ patches to the right" with constant weights. For 2-D, half the dimensions encode x, half y. The model never decodes a position from the code; it only needs codes that are unique and smooth. Tweak to know: with temperature 10000 and 16 positions, 71 of 96 frequency pairs barely move; temperature 100 gives a sharper similarity profile (0.5 far away instead of 0.7). Worth one ablation.

**How are patch content and position combined, and how does the question join them?** Exactly like `tok_emb + pos_emb` in your GPT: elementwise addition, `x = patch_vec + pos`, then into the first block. (Adding rather than concatenating works because in 384 dimensions the network can keep content and position in different subspaces.) The marker needs nothing extra, it is already inside `patch_vec`. Task information can join the same sum as a third term: `x = patch_vec + pos + task_emb(task_id)[:, None, :]` with `task_emb = nn.Embedding(n_tasks, n_embd)`, broadcast to every token. The alternative is one extra token in front (T = N+1); more general (later several tokens: layer metadata, a text query), slightly more plumbing. Proposal for round one: fixed layers, optional marker channel, an integer task id as the added embedding, target = a 64×64 grid.

**What does Cin = 13 mean?** Cin is the number of channels, as you thought: 12 one-hot layer channels plus 1 marker channel. The marker is just one more image plane, `torch.cat([layers, marker[None]], dim=0)`, zero everywhere and 1 at the queried pixel. It enters through the same patch projection: the patch containing it gets one extra column added, and because columns are per in-patch position, the model knows *which pixel inside the patch* is marked.

**First step "just the marker"?** With the grid head a clean first rung is literally: input layers + marker, target = the marker's cell on the 64×64 grid. It tests patchify, axis order, position code and grid-head alignment in one go, with an answer you can see. Task information is *not* part of Cin; it comes in via the task embedding above. Second rung: task id chooses what to mark. Targets for all of these can be computed on the fly from `*_labels.npz`; no new generated data is needed.

**Why pool at all; why not a layer that projects x to the answer?** The encoder gives `(B, N, C)`; a classifier needs `(B, K)`. "Directly a layer" means flatten to `(B, N·C)` and `Linear(N·C, K)`: possible (≈ 393k weights here), but it learns separate weights for every patch position, fixes N forever, and cannot transfer what it learns at one position to another. Mean pooling is the opposite extreme: position-blind, every patch counts 1/N. Good for "is there a café anywhere", poor for "what is at *this* point", where one relevant patch is averaged with 255 irrelevant ones. A CLS token is the middle: a learned, content-dependent weighted average, since attention decides which patches it reads. With the grid head none of this applies: every patch keeps its own output, nothing is pooled.

**What is "memory" in §9?** Not a layer. It is the variable name in your own `Seq2Seq.forward`: `memory = self.encode(...)`, the encoder's output tensor from which the decoder's cross-attention takes K and V. The name comes from the original Transformer paper. It only exists if a decoder is used; with the grid head there is no decoder and no memory.
