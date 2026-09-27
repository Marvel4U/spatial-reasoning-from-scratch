# Vision models from scratch — concept bridge, verified resources, first recipe

**Survey note 07 — compiled 19 Sep 2026.** Audience: Marvin. Companion to `02_training_tooling_and_base_models.md` (the Qwen3-VL fine-tuning track). This note covers the **from-scratch control track**: a small vision(-language) model trained on our own synthetic multi-channel city rasters, native N-channel input, no pretrained backbone.

> **Verification status.** Every GitHub number below was pulled from the GitHub REST API on 19 Sep 2026 (`stargazers_count`, `pushed_at`, `license`) — those are hard facts as of today. Paper facts come from arXiv abstracts/listings. Line counts are taken from the repos' own README claims where they state them and are marked as such — I did **not** clone and run `cloc`. Anything I could not confirm is marked **[UNVERIFIED]**.

---

## 0. Recommendation up front

1. Your first from-scratch model should **not** be a VLM. It should be a **plain ViT encoder with an N-channel patch embedding and a tiny autoregressive token decoder** — roughly a Pix2Seq/Donut-shaped "image in, short token sequence out" model, ~10–25M params. You already own the decoder: it is literally your translator's decoder with cross-attention, with the source encoder swapped from a BPE token embedding to a patch embedding.
2. The three resources that actually matter for you are **lucidrains/vit-pytorch (SimpleViT as the reference module)**, **Beyer et al. "Better plain ViT baselines" (the recipe)**, and **Pix2Seq (the output format)**. Everything VLM-shaped (nanoVLM, seemore, PaliGemma-from-scratch) is *architecture orientation*, not a starting point, because they all assume pretrained SigLIP + pretrained LM — which is exactly what your control track is defined to not have.
3. The one genuinely novel design question — **how do I feed 3–8 aligned channels into a ViT** — is not answered by any VLM tutorial. It is answered by the remote-sensing literature (SatMAE group-wise patch embeds, DOFA wavelength-conditioned hypernetwork). For your case the answer is trivially "just change `in_chans`", and the RS papers only matter if you later want channel-count generalisation.

---

# Part 1 — Concept bridge: from encoder-decoder text transformer to "image in, tokens out"

You have already built every component. What follows is the exact diff list, not a tutorial.

## 1.1 The patch embedding *is* your source embedding table

In your translator, the source side is `nn.Embedding(vocab, d)` applied to `(B, S)` int64 ids → `(B, S, d)`.

In a ViT the source side is:

```
x: (B, C, H, W)  ->  rearrange to (B, N, C*p*p)  ->  nn.Linear(C*p*p, d)  ->  (B, N, d)
```

with `N = (H/p)*(W/p)`. Equivalently and identically, `nn.Conv2d(C, d, kernel_size=p, stride=p)` followed by a flatten — the conv formulation is the same linear map, just expressed as a strided convolution. **This is the only structurally new module in the whole enterprise.** There is no tokenizer, no vocabulary, no embedding lookup: the "token" is a `C*p*p`-dim real vector and the "embedding" is a single dense matrix.

Consequences that trip people up:
- **There is no `<unk>`, no padding token, no variable length.** Sequence length is a deterministic function of image size and patch size. Fixed `H, W` → fixed `N`. That makes the batch collate trivially simple compared to text.
- **Patch size is the compute knob, and it is quadratic.** 256px/p=16 → 256 tokens. 512px/p=16 → 1024 tokens. 1024px/p=16 → 4096 tokens, which at d=384 is already an uncomfortable attention cost on a 3060. Your lever is `p`, not depth.
- **`C*p*p` for 8 channels at p=16 is 2048** — the patch-embed matrix (2048×384 ≈ 0.8M params) is a non-trivial slice of a small model. Fine, but note it.

## 1.2 2D positions

Text: 1D learned or RoPE over position `i`.

Images: the grid is 2D and the model has **zero** knowledge of adjacency after patchification — it is a permutation-invariant set until you add positions. Three options, all defensible:

- **Learned 1D `nn.Parameter(N, d)`** over the flattened grid. This is what the original ViT does. It works; the model recovers 2D structure. Downside: cannot change image size without interpolating the position table.
- **Fixed 2D sin-cos** (`sincos2d`, as in SimpleViT / big_vision). Half the channels encode row, half encode column. Free, no params, generalises to other grid sizes, and is the modern default for small-data training. **Use this.**
- **2D RoPE** — rotate query/key pairs by row angle and column angle in separate channel halves. More work, marginal at your scale; skip for v1.

For a spatial-reasoning task on a metric city raster, positional encoding is not a detail — it is the entire substrate of the task. Fixed 2D sin-cos with a known, documented frequency band is the honest choice, because it lets you state exactly what spatial prior the model was given.

## 1.3 No causal mask in the encoder

Your encoder-decoder translator already has this right: encoder self-attention is bidirectional, decoder self-attention is causal, cross-attention is unmasked. A ViT encoder is *exactly* your bidirectional encoder. Nothing changes except the input embedding and positions. There is also no padding mask, because there is no padding.

The one thing to strip: ViTs use **pre-norm** blocks throughout and typically **no biases** in qkv/MLP in modern recipes, plus a final LayerNorm. If your translator was post-norm, switch — post-norm ViTs are notoriously hard to optimise.

## 1.4 Readout: CLS vs pooling vs cross-attention decoder vs single-stream prefix-LM

Four readouts, in increasing order of what they can express:

| Readout | Shape out | Use when |
|---|---|---|
| **CLS token** | one `d`-vector | classification, single label. Extra learned token prepended; its output row is the summary. |
| **Mean pooling (GAP)** | one `d`-vector | same as CLS, one fewer parameter, and Beyer et al. report it is *better* than CLS in their baselines. Prefer GAP over CLS. |
| **Cross-attention decoder** | token sequence | image → arbitrary-length structured output. Your translator's decoder, verbatim: decoder self-attends causally over the answer tokens and cross-attends over all `N` patch tokens. Donut and Pix2Seq are exactly this. |
| **Single-stream prefix-LM** | token sequence | one transformer stack; `[patch tokens][question tokens][answer tokens]`; bidirectional (non-causal) attention over the prefix, causal over the answer. This is PaliGemma's design, and it is what every modern VLM does. |

The last two are the important distinction for you, and it maps cleanly onto a research decision you already understand:

- **Two-stream (cross-attention)** keeps a clean separation: the encoder is a pure vision tower, the decoder is a pure language head that queries it. Cheaper (patch tokens never appear in the decoder's self-attention), easier to debug (you can probe the encoder in isolation, swap decoders, freeze one side), and it is the design Marvin has already implemented once.
- **Single-stream (prefix-LM)** is what the industry converged on because it lets you reuse a pretrained LM unchanged. In a from-scratch setting that advantage is **zero**. Its remaining advantage is that the question tokens can attend to and be attended by the patch tokens at every layer, which matters for query-conditioned visual grounding ("where is the nearest X to the marker?").

**Recommendation: start two-stream, because you can build it tonight; keep single-stream as experiment #2, because for question-conditioned spatial queries the deep interaction may genuinely win, and having both is a real result to report.**

## 1.5 How the loss is set up

- **Classification head:** `CrossEntropyLoss` over `K` classes on the pooled vector. Label smoothing 0.0–0.1. That is all.
- **Token decoder:** *identical to your translator.* Teacher forcing, shift-by-one, `CrossEntropyLoss(ignore_index=pad)` over the answer tokens only. The image contributes no loss term. The only new decision is whether to also compute loss over the question tokens (in a prefix-LM) — **no, mask them out**; only the answer is supervised.
- **Coordinates:** two options, see §3.5. As tokens → same CE loss, nothing new. As regression → separate L1/smooth-L1 head, which mixes loss scales and is a nuisance. Pix2Seq's whole point is that quantising coordinates into ~500–1000 bins and treating them as vocabulary entries makes everything one CE loss with one decoding path.
- **No contrastive loss anywhere.** CLIP-style training is for learning transferable representations from noisy web pairs; you have a supervised generator with exact labels. Do not import that complexity.

## 1.6 What changes in the data loader

This is where more time goes than you expect.

- **Output of `__getitem__` is a float tensor `(C, H, W)` plus a target**, not an int tensor. No tokenizer on the image side.
- **Normalisation is per-channel and you must own it.** No ImageNet mean/std — your channels are a noise band, a categorical surface class, an establishment density. Compute per-channel mean/std over the generated corpus, or better, define the normalisation *in the generator* so it is exact and documented (per CLAUDE.md: no post-hoc cleaning).
- **Categorical channels must not be fed as scalars.** A "surface class" channel with values {road=1, water=2, building=3} encodes a false ordering. Either one-hot it into separate channels (cheapest, and the reason `C` grows to 8) or give it its own small embedding before the patch projection. **Decide this explicitly; it is a real modelling choice, not plumbing.**
- **Augmentation is different in kind.** Text augmentation you probably skipped entirely. For ViTs on small data it is load-bearing (§3.7). But note: your data is *synthetically unlimited*, which changes the calculus — see §3.7.
- **Channel dropout** (randomly zero a whole input channel) is the multi-channel-specific augmentation, and it doubles as an ablation instrument: it forces the model not to collapse onto one channel, and it lets you measure per-channel contribution at eval.
- **`torch.compile` + channels_last + AMP bf16** matter far more here than in your text work, because the compute is dominated by a small number of large matmuls over 256–1024 tokens.

## 1.7 The mental model in one line

> A ViT is your encoder with `nn.Embedding` replaced by one `Conv2d(C, d, p, stride=p)` and 1D positions replaced by 2D sin-cos. Everything downstream of that — decoder, cross-attention, teacher forcing, CE loss, beam/greedy decode — is code you have already written.

If that sentence is true (it is), then the honest framing of your gap is **not** "I don't know vision models". It is "I have never chosen a patch size, never trained anything that needs augmentation, and never debugged an optimiser that is unstable at init." Those are the three things to actually learn, and §3.7 is about exactly them.

---

# Part 2 — Ranked resources

Ranking criterion: **how much does this shorten the path to a working from-scratch multi-channel image→tokens model on your hardware**, not how famous it is.

### 1. lucidrains/vit-pytorch — `simple_vit.py` as your reference module
- **What:** 30+ ViT variants in single-file PyTorch. The one you want is `SimpleViT` (the Beyer et al. "better baselines" variant: GAP readout, fixed 2D sin-cos positions, no CLS, no dropout) — ~110 lines.
- **Verified:** 25,512 stars; last push **5 Sep 2026** (actively maintained); MIT; pip-installable (`pip install vit-pytorch`). Also contains **CCT** (compact convolutional transformer) and **NaViT** in the same repo, both relevant to you.
- **Why for you:** you will not use it as a dependency — you will read `simple_vit.py` once, type your own, and diff. It is the shortest correct ViT in existence and the variants give you a menu of "what people do when data is small" (CCT, T2T, CrossViT) without reading 10 papers.
- **Time:** 1–2 h to read + type. Another 1 h to skim CCT.
- **Link:** https://github.com/lucidrains/vit-pytorch

### 2. Beyer, Zhai, Kolesnikov — "Better plain ViT baselines for ImageNet-1k" (arXiv 2205.01580)
- **What:** 3-page paper. The minimal set of changes to vanilla ViT training that make it work without exotic regularisation.
- **Verified:** arXiv 2205.01580, May 2022, Google Brain Zürich. Abstract claim: **90 epochs > 76% top-1 in under 7 h on a TPUv3-8; 300 epochs → 80% in under a day**. Code in `google-research/big_vision` (3,541 stars, last push 19 May 2025, Apache-2.0).
- **Why for you:** this is your **hyperparameter prior**, and it is the single highest-value-per-page document in this list. The deltas it establishes — GAP instead of CLS, fixed 2D sin-cos instead of learned positions, no dropout, RandAugment+Mixup only, global gradient clipping at 1.0, large batch — are exactly the defaults you should start from and deviate from only with a reason. It also directly contradicts the folklore that ViTs need heavy regularisation, which matters for your unlimited-synthetic-data case.
- **Time:** 45 min including the appendix table.
- **Link:** https://arxiv.org/abs/2205.01580

### 3. Chen et al. — Pix2Seq (arXiv 2109.10852) + a small PyTorch reimplementation
- **What:** object detection recast as language modelling. Coordinates normalised to [0,1], quantised into a few hundred–few thousand bins, emitted as discrete tokens; each object is 5 tokens `[ymin, xmin, ymax, xmax, class]`.
- **Verified:** arXiv 2109.10852. Official code `google-research/pix2seq` is **TensorFlow 2**. PyTorch reimplementation `moein-shariatnia/Pix2Seq`: 131 stars, MIT, **last push 2 Sep 2023 (stale)**. Chris Hughes wrote a long "lessons from reimplementing Pix2Seq" write-up.
- **Why for you:** this is the **output-format decision** for your whole from-scratch track, and it is the one that makes "a class / a number / coordinates / later a path" all be the *same model with the same loss*. A path is just a longer coordinate sequence. If you adopt Pix2Seq's quantised-coordinate vocabulary on day one, Stage-N "output a route" costs you zero architectural change. Read the paper for the bin-count ablation and the sequence-augmentation trick (noise objects to stop early-EOS); use the PyTorch repo only as a sanity reference, not a base — it is three years stale.
- **Time:** 2 h paper, 1 h repo skim.
- **Link:** https://arxiv.org/abs/2109.10852 · https://github.com/moein-shariatnia/Pix2Seq

### 4. Sebastian Raschka — "Understanding Multimodal LLMs"
- **What:** the canonical explainer of the two VLM architecture families: **Method A, unified embedding decoder** (project image patches into the LM's embedding space and concatenate) vs **Method B, cross-modality attention** (inject image features via cross-attention). Plus a reviewed table of 10 then-recent models (Llama 3.2, Molmo/PixMo, NVLM, Qwen2-VL, Pixtral 12B, MM1.5, Aria, Baichuan-Omni, Emu3, Janus).
- **Verified:** published **3 Nov 2024**, Ahead of AI, ~25–30 min read. Note: **pre-dates Qwen3-VL** — treat the model table as historical, the taxonomy as current.
- **Why for you:** it gives you the vocabulary to defend your two-stream-vs-single-stream choice in an interview, and it is written at exactly your level (no math hand-holding). It is also the fastest way to understand what the Qwen3-VL track is doing architecturally, so the two tracks are comparable in your own head.
- **Time:** 40 min.
- **Link:** https://magazine.sebastianraschka.com/p/understanding-multimodal-llms

### 5. huggingface/nanoVLM
- **What:** the smallest credible *training* codebase for a VLM in pure PyTorch. README claims ~750 lines of model+training logic: vision backbone ~150, language decoder ~250, modality projection ~50, VLM wrapper ~100, train loop ~200 **[line counts are the repo's own claim, not measured by me]**.
- **Verified:** 5,027 stars; Apache-2.0; **last commit 27 Oct 2025** (repo is popular but has been quiet ~11 months — check before depending on it). Blog post 21 May 2025. Default config: SigLIP-B/16-224 (85M) + SmolLM2-135M = **222M params**; ~1.7M samples of `the_cauldron`, **~6 h on one H100 → 35.3% MMStar**. VRAM: ~4.4 GB at batch 1, ~7.6 GB at batch 16, ~38.8 GB at batch 128 (H100 figures). Modality projection = pixel-shuffle + linear, then concatenate into the decoder (Method A).
- **Why for you:** read it for **the training loop and the data/eval plumbing**, not the architecture. It is the best available answer to "what does a real, non-toy VLM training script look like when someone deliberately removed all the framework". The 7.6 GB @ batch-16 figure also tells you a 222M VLM is *3060-feasible*, which is a useful calibration even though your model will be 10× smaller.
- **Caveat:** it is a *fine-tuning-on-pretrained-backbones* repo. Its vision tower is pretrained SigLIP with 3 channels. Neither of those survives contact with your requirements.
- **Time:** 3–4 h to read properly; 1 h to run the Colab.
- **Link:** https://github.com/huggingface/nanoVLM · https://huggingface.co/blog/nanovlm

### 6. Remote-sensing multispectral encoders — SatMAE, DOFA (for N-channel handling only)
- **What:** how the satellite-imagery community feeds 8–13 spectral bands into a ViT.
  - **SatMAE** (arXiv 2207.08051, NeurIPS 2022): two strategies — naive channel-stacking, vs **grouping subsets of bands and giving each group its own patch-embedding matrix and its own token stream**, with a channel-identity positional encoding added on top of the spatial one.
  - **DOFA** (`zhu-xlab/DOFA`: 214 stars, MIT, last push **22 Jul 2026** — alive): a **hypernetwork generates the patch-embedding weights conditioned on each channel's central wavelength** (sinusoidally encoded), so one backbone serves sensors with different band sets.
  - **Clay** (`Clay-foundation/model`: 615 stars, Apache-2.0, last push **11 May 2026**): an open EO foundation model, useful as a worked example of a production N-channel + metadata ViT.
- **Why for you:** this is the only literature that treats "N input channels with distinct semantics" as a first-class design problem. The **relevant takeaway is narrow and you should not over-invest**: for a *fixed* channel set (your case), naive stacking into `in_chans=C` is fine and is what SatMAE's own baseline does. Group-wise embeddings and DOFA-style conditioning only pay off when you want one model to handle *varying* channel sets — which becomes relevant only if you later want "train on 5 channels, evaluate with a 6th added", a genuinely nice ablation.
- **Practical note:** `timm.create_model(..., in_chans=C)` handles arbitrary `C` by expanding/replicating pretrained patch-embed weights. From scratch you don't need that, but it is the one-line escape hatch if you ever want a pretrained-init comparison.
- **Time:** 1 h for SatMAE §3, 30 min for the DOFA README. Do not read Prithvi/Panopticon unless you take up channel-generalisation.
- **Links:** https://arxiv.org/abs/2207.08051 · https://github.com/zhu-xlab/DOFA · https://github.com/Clay-foundation/model

### 7. Xiao et al. — "Early Convolutions Help Transformers See Better" (arXiv 2106.14881)
- **What:** replacing ViT's single large-kernel/large-stride patchify stem with a few stacked stride-2 3×3 convs.
- **Verified:** arXiv 2106.14881 (Xiao, Singh, Mintun, Darrell, Dollár, Girshick, 2021). Claims: **dramatically improved optimisation stability, much reduced sensitivity to optimizer choice (AdamW vs SGD), hyperparameters and schedule length, plus ~1–2% top-1 on ImageNet-1k, at matched flops/runtime**, consistently across 1G–36G flops and IN-1k→IN-21k.
- **Why for you:** you are about to train a from-scratch ViT on a small model, a small budget, and a data distribution nobody has tuned for. Optimisation instability at init is the single most likely way you lose a weekend. A conv stem is ~15 lines and buys you robustness to exactly the hyperparameters you have no prior over. Also: a conv stem gives cheap local inductive bias for free, which is what your rasters have a lot of.
- **Time:** 40 min.
- **Link:** https://arxiv.org/abs/2106.14881

### 8. Steiner et al. — "How to train your ViT?" (arXiv 2106.10270) + Lee et al. "ViT for Small-Size Datasets" (arXiv 2112.13492)
- **What (Steiner):** the large AugReg sweep. Core finding: **increased compute + augmentation/regularisation can substitute for an order of magnitude more training data**; IN-21k+AugReg matches or beats JFT-300M counterparts. Also: **AugReg is not helpful when transferring** a pretrained model. Techniques studied: dropout on intermediate activations, stochastic depth with linearly increasing drop rate, Mixup + RandAugment.
- **What (Lee):** **SPT** (shifted patch tokenization — concatenate spatially shifted copies of the image before patchifying, raising the receptive field of each token) and **LSA** (locality self-attention — learnable temperature + diagonal masking to sharpen attention). Reported **+2.96% average on Tiny-ImageNet across ViTs; +4.08% on Swin**. Code `aanna0701/SPT_LSA_ViT`: 131 stars, **no licence declared**, last push **23 Feb 2022 (stale, and unlicensed → read, do not copy)**.
- **Related:** **CCT / Compact Transformers** (`SHI-Labs/Compact-Transformers`: 545 stars, Apache-2.0, last push 5 Nov 2024) — "Escaping the Big Data Paradigm"; README claims **ViT-quality CIFAR-10 in ~30 min on one GPU**. Also available as a variant inside `vit-pytorch`.
- **Why for you:** these are the "ViTs are data-hungry" fixes. **Read them to know the failure mode, then consciously decide you may not need them** — your generator makes data effectively unlimited, and the small-data ViT literature is entirely about the regime where data is fixed and scarce. CCT is the one to actually keep in your back pocket, because its conv tokenizer + sequence pooling is a strictly-better default than plain ViT at your scale and costs nothing.
- **Time:** 1 h Steiner (skim tables), 45 min Lee, 30 min CCT.
- **Links:** https://arxiv.org/abs/2106.10270 · https://arxiv.org/abs/2112.13492 · https://github.com/SHI-Labs/Compact-Transformers

---

## Also noted (verified, lower priority)

| Resource | Verified facts | Verdict |
|---|---|---|
| **AviSoori1x/seemore** — VLM from scratch, Karpathy style | 260 stars, MIT, created 17 Apr 2024, **last push 6 May 2024 (stale)**. Single-file `seemore.py` + 3 notebooks; from-scratch CLIP-style ViT + projector + **character-level** autoregressive decoder. Sibling `seeMoE` adds an MoE decoder. | The closest in spirit to what you want (genuinely from scratch, no pretrained weights, explicitly homage to makemore). But it is a *toy* — char-level decoder, tiny, stale, and you have already internalised the makemore idiom. **Skim the notebook for 45 min, do not adopt.** |
| **Umar Jamil — "Coding a Multimodal (Vision) Language Model from scratch"** (`hkproj/pytorch-paligemma`) | 634 stars, created 13 Jul 2024, **last push 6 Dec 2024**, **no licence declared**. Accompanies a long-form YouTube walkthrough (`youtube.com/watch?v=vAmKB7iPkWw`). Video length **[UNVERIFIED]** — I did not open YouTube; Jamil's videos are typically 3–6 h. | Best available line-by-line treatment of the **single-stream prefix-LM** design (SigLIP + Gemma, PaliGemma-style, including the non-causal prefix mask). Watch at 1.5× if and when you build experiment #2. Unlicensed → reference only. |
| **nipunbatra/vlm-from-scratch** (`nipunbatra.github.io/vlm-from-scratch`) | 12-part notebook series: minimal VLM, object detection, VQA, multi-task, multi-image, task routing, CoT, referring segmentation, image editing, compression, image generation, OCR→LaTeX. Site claims ~700M params trained across 12 notebooks. Publication/update date **[UNVERIFIED]** — not stated on the index page. Licence **[UNVERIFIED]**. | Parts 2 (detection) and 3 (VQA) and 6 (task routing) map onto your task family unusually well. Worth 2 h of targeted skimming. Freshness unconfirmed — verify before relying on it. |
| **Masoudjafaripour/nanochat-VLM** | **26 stars**, created 23 Oct 2025, **last push 16 Sep 2026 (very fresh)**, licence "Other". Claims end-to-end nanochat-style VLM training (tokenizer → LM → multimodal) "for under $200 in compute". | The most current nanoGPT-idiom VLM repo I found, and the only one that trains the LM too. But 26 stars = unvetted, one-author, unaudited. **Interesting, not dependable. Do not build on it.** |
| **tintn/vision-transformer-from-scratch** | 263 stars, MIT, created 6 Mar 2023, **last push 10 Jun 2024 (stale)**. "A simplified PyTorch implementation of ViT." | Fine, correct, redundant with `simple_vit.py`. Skip unless you want a second opinion while typing. |
| **Brian Pulfer — "Vision Transformers from Scratch (PyTorch)"** | Medium, **published 3 Feb 2022**, MNIST. | The classic step-by-step. Below your level — it explains multi-head attention. **Skip.** |
| **Donut / Pix2Struct** | Donut = OCR-free doc understanding, Swin encoder + BART-ish decoder → JSON. Pix2Struct = ViT encoder + text decoder, pretrained by parsing masked webpage screenshots into simplified HTML. Community consensus in the sources: **Pix2Struct is very hyperparameter-sensitive and converges slower than Donut**; relative quality is task-dependent. `huggingface/pixparse` is an open reproduction effort. | Read as **design precedent** for "pixels in, structured string out, no OCR" — which is precisely your task shape. Their concrete lesson: the output must be a *canonical, terse, deterministic* string grammar, or the decoder wastes capacity learning your formatting. Do not train either. |
| **google-research/big_vision** | 3,541 stars, Apache-2.0, last push **19 May 2025**. Official codebase for ViT, SigLIP, MLP-Mixer, LiT. | JAX. Read the ViT config files for the exact hyperparameters behind resource #2; ignore the rest. |
| **Original ViT paper — Dosovitskiy et al., arXiv 2010.11929** | Not re-fetched today; canonical. | You should read it once for the JFT-300M scaling argument and the "ViTs lack locality inductive bias" framing, because every other paper in this list is a reaction to it. 1 h. |

---

# Part 3 — Practical starting recipe

## 3.1 Recommended first architecture

**Name it `tinyViT-Seq`. Target: ~15M params, trains in <2 h on the 3060.**

```
Input:    (B, C=5, 256, 256)          # start at 256px, not 512/1024
Stem:     conv stem, 3 x [Conv2d(stride=2, k=3) + GN + GELU] -> Conv2d(k=1) to d
          giving 16x16 = 256 tokens   # effective patch 16, but convolutional
Pos:      fixed 2D sin-cos, added
Encoder:  6 layers, d=384, heads=6, mlp_ratio=4, pre-norm, no dropout   (~10.7M)
Readout:  full token sequence (256, 384) -> to decoder cross-attention
Decoder:  4 layers, d=384, heads=6, causal self-attn + cross-attn        (~7.1M)
Vocab:    ~1100 = 1000 coordinate bins + ~50 class words + ~30 task ids
          + digits + <bos>/<eos>/<pad>/<sep>
Loss:     CE over answer tokens only, ignore_index=pad
```

Rationale for each number:
- **d=384, 6 heads** — the ViT-Small shape. Below d=256 attention heads get too narrow; above d=512 you are paying for capacity your task does not need yet.
- **6 encoder / 4 decoder** — deliberately shallow. Depth is the thing to scale *after* you have a working signal, and shallow models expose bugs faster.
- **256 tokens** — attention at 256 tokens is free. At 1024 (512px/p=16) it is 16× the attention flops. Earn the resolution.
- **Params ~18M total.** At bf16 with AdamW that is ~0.3 GB of weights+optimizer state. The 3060's 12 GB is overwhelmingly activations, i.e. batch size — you will fit batch 128–256 easily at 256 tokens.

## 3.2 Conv stem: yes

Use a conv stem, not a plain patchify. Reasons in order: (a) Xiao et al.'s stability result is exactly the risk you face; (b) your rasters have strong local structure a 3×3 conv exploits for free; (c) it costs ~0.5M params and no runtime. **But implement plain patchify too, behind a flag** — "conv stem vs patchify at fixed flops" is a 2-run ablation that produces a real figure for the README, and it is the kind of thing an interviewer will ask whether you checked.

## 3.3 Feeding N channels

**Naive stacking: `in_chans=C`, one patch-embed matrix over `C*p*p`.** That is the answer for v1 and it is what SatMAE's own baseline does. Do not build group-wise embeddings on day one.

Two things that are *not* optional:
1. **One-hot categorical channels in the generator**, not as integer-valued rasters. A surface-class channel with values 1/2/3 asserts road < water < building, and the patch-embed matrix will dutifully learn a monotone ramp. This is the multi-channel equivalent of feeding raw token ids instead of embeddings.
2. **Per-channel normalisation defined in the generator** and written to a config alongside the data, so it is exact, reproducible, and documented — and so the Qwen3-VL track can be told precisely what it is *not* getting.

**Later, if you want the channel-generalisation story:** SatMAE-style group-wise patch embeds + a learned channel-identity embedding added alongside the 2D positional encoding. That gives you "drop a channel at test time" and "add a new channel with a few-shot embedding" as clean experiments. Park it.

## 3.4 Feeding the question

Three options; the trade-off is real and you should run at least two.

| Option | How | Pro | Con |
|---|---|---|---|
| **A. Task-id token** | Prepend a single learned token from a ~30-entry task vocabulary to the decoder. | Trivial. Zero ambiguity. Fastest to a working baseline. | No compositional generalisation — the model memorises 30 behaviours. Says nothing about language. |
| **B. Marker channel + task id** | Add a `C+1`-th channel with a Gaussian blob at the query location; task id as in A. | The clean way to ask *point* queries ("what is at here?") without spending tokens on coordinates, and it keeps the query in the same spatial frame as the data. Very strong for "cross-channel point query". | Only expresses queries that are a location. Cannot express "the nearest bakery to the largest park". |
| **C. Templated text tokens** | Tokenize a short templated question with your own tiny word-level vocab (~200 words); feed it to the decoder as a prefix, or to the encoder in single-stream mode. | The only option that generalises to unseen template combinations, and the only one comparable to what Qwen3-VL sees. | Needs a vocab, needs the templates to be genuinely compositional or it degenerates into option A with extra steps. |

**Recommendation:** build **B** first (fastest path to a non-trivial win, and marker-channel point queries are the sanity ladder's top rung), then **C** as the real system. Report both. The A-vs-C gap is itself a result: it quantifies how much of your task is "30 memorised behaviours" versus actual query understanding, which is a question the Qwen3-VL track cannot answer as cleanly.

## 3.5 Output heads

**One token decoder for everything. Pix2Seq-style quantised coordinates. No regression head.**

- Classes → one token from a small class vocabulary.
- Counts → digit tokens (or a small integer vocabulary up to your max count).
- Coordinates → quantise to **1000 bins per axis** at 256px input (bin ≈ 0.26 px — far below label noise; Pix2Seq found a few hundred to a thousand bins suffices, and bin count is a documented ablation in that paper). Emit `<y><x>`.
- Paths → `<y1><x1><y2><x2>...<eos>`. **Zero architectural change.** This is the whole reason to pay the token-decoder tax now.

**Why not regression:** it forces a second loss with its own scale, a second head, a second decoding path, and it makes the model unable to express multimodality (two equally-good answers average to a wrong one in the middle — a real failure mode for "which corner is nearest"). CE over bins expresses multimodality natively and gives you a calibrated distribution over locations, which is a much better diagnostic and plots beautifully as a heatmap over the raster.

**Do also build a classification head** behind a flag, for the sanity ladder only (§3.8) — a GAP + linear head converges in minutes and isolates "is the encoder learning anything" from "is the decoder learning anything". That separation is worth the 10 lines.

## 3.6 Data volume and training-time estimates

These are **order-of-magnitude estimates, not measurements** — mark them as such in the run ledger and replace them with real numbers after the first run.

- **Throughput, 3060 12 GB, bf16 + `torch.compile` + channels_last, 18M params, 256 tokens in / ~10 tokens out:** expect roughly **600–1,500 samples/s** at batch 128. [UNVERIFIED — measure]
- **Data volume:** with a generator, do not build a fixed dataset — **generate on the fly** and treat "samples seen" as the axis, so there is no train-set memorisation to worry about and the small-data ViT problem simply does not arise. Ballpark targets: **~1M samples** to a convincing signal on a trivial task, **~10–30M samples** for a converged small model on a compositional task.
- **Wall clock on the 3060:** 1M samples ≈ **15–30 min**. 10M ≈ **3–5 h**. 30M ≈ **overnight**. This is genuinely a 3060-sized project at 256px — the RTX PRO 6000 only becomes necessary if you go to 512–1024px, d≥768, or want many seeds in parallel.
- **Critical caveat:** at these throughputs **the generator, not the GPU, will be the bottleneck.** Profile it before you profile the model. Budget real effort for a multi-worker generator or a pre-generated sharded cache (webdataset-style) — this is the single most likely thing to turn a 30-min run into a 6-h run.

## 3.7 Known small-data ViT pitfalls — and why most don't apply to you

The literature's core claim (ViT paper, Steiner, Lee): ViTs lack the locality/translation-equivariance inductive bias of CNNs, so below roughly ImageNet-21k scale they underperform CNNs unless you add heavy AugReg or restore locality architecturally.

**Why this mostly doesn't bind you:** the entire small-data literature assumes a *fixed, finite* dataset where the failure is memorisation. You have a generator. Your data is unlimited and i.i.d.-fresh. **Fresh samples are the strongest regulariser that exists** — with infinite data, train loss ≈ val loss by construction and dropout/stochastic-depth/Mixup are solving a problem you do not have. Say this explicitly in the README; it is a genuinely good argument and it is the kind of reasoning that reads well in an interview.

**What still binds you:**
1. **Optimisation instability at init.** Real and independent of data volume. Mitigate: conv stem (§3.2), AdamW, **global grad clip at 1.0**, ~10k-step linear warmup, and pre-norm blocks. If loss NaNs or plateaus at chance, this is the cause 80% of the time — not the architecture.
2. **LR and weight decay.** Start **lr=1e-3 with batch 256** (ViTs take much higher LRs than you may expect from text at this scale), cosine decay to ~0, **wd=0.1** decoupled, **no weight decay on norms, biases, or positional parameters**. β=(0.9, 0.95).
3. **Sample diversity ≠ sample count.** The failure mode unlimited data does *not* fix: if your generator's distribution is narrow, you get a model that is perfectly fit to a narrow world. **Generator diversity is now a model hyperparameter.** Sweep it explicitly (how many city layouts, how much noise, how many establishment types) and report the curve.
4. **Augmentation you should still use:** random 90° rotations and flips (if your task is rotation/reflection-equivariant — check! "north" tasks are not), random crops/translations, per-channel intensity jitter, and **channel dropout**. Skip Mixup and RandAugment — they are for natural images and will corrupt your categorical channels' semantics.
5. **The CNN-vs-ViT comparison is worth running anyway.** A small ResNet/U-Net baseline at matched params is cheap, is the honest control, and if it beats your ViT on the point-query task that is a *finding*, not a failure. Log it.

## 3.8 Sanity-check ladder

Do not skip a rung. Each rung should take minutes, not hours, and each isolates one failure mode.

| # | Test | Passes when | Fails → suspect |
|---|---|---|---|
| 0 | **Shape/plumbing.** One forward pass, assert every tensor shape. Verify the decoder cannot see future answer tokens (flip one answer token, confirm only later logits change). | Shapes right, causality right. | Mask construction. This bug is silent and ruins everything downstream. |
| 1 | **Overfit one batch.** 32 samples, no augmentation, no weight decay, 500 steps. | Train loss → ~0, exact-match → 100%. | If it can't, the model/loss/optimiser is broken. Nothing else matters until this passes. |
| 2 | **Trivial task, classification head.** "Is the centre pixel a building?" — GAP + linear, one channel would suffice. | >99% within a few thousand steps. | Encoder or normalisation. Check per-channel stats and that categoricals are one-hot. |
| 3 | **Trivial task, token decoder.** Same task, same answer, but emitted as a token through the full decoder. | Matches rung 2. | The decoder/cross-attention path, in isolation. This rung is the whole reason to build the classification head. |
| 4 | **Positional-awareness probe.** "What is the surface class at the marker?" (marker channel, random location). | High accuracy. | 2D positional encoding, or the marker channel's blob scale relative to patch size. Debug by plotting the decoder's cross-attention over the patch grid — it should light up on the marker. |
| 5 | **Coordinate emission.** "Where is the marker?" → `<y><x>`. | Bin error within a few bins. | Coordinate quantisation/detokenisation (off-by-one, y/x order). Plot predicted-vs-true. |
| 6 | **Cross-channel point query.** "What establishment is nearest to the marker?" — requires reading two channels and comparing distances. | Meaningfully above the majority-class baseline. | This is the first *real* task. Compute the majority-class and nearest-neighbour-heuristic baselines **before** you look at the model number, or you will fool yourself. |
| 7 | **Compositional held-out.** Train on template set A, evaluate on unseen template combinations. | Any non-trivial transfer. | This is the headline result, and the honest place to report a negative. |

Log every rung in the run ledger with the same harness you built for the translator — that harness transfers unchanged.

---

# Part 4 — Honest assessment: from-scratch vs fine-tuned Qwen3-VL

## What the from-scratch model CAN show

1. **That you can build and train a vision model end to end.** For the job targets, this is the actual point. "Fine-tuned a VLM with TRL" is a crowded claim; "implemented the patch embedding, the 2D positions, the cross-attention readout, the coordinate tokenizer, and the training loop, and can defend every hyperparameter" is not.
2. **Native N-channel input** — a capability Qwen3-VL structurally cannot have. It eats 3-channel RGB. To give it 8 channels you must either render them into RGB (lossy, and you must choose the colormap — a confound) or tile multiple images (expensive, and the alignment becomes something the model must infer). **This is the from-scratch track's only genuinely exclusive capability and it should be the centre of the story.**
3. **A clean data-scaling curve.** No pretraining contamination, so "accuracy vs samples generated" is interpretable in a way that no fine-tuning curve is.
4. **Exact capacity/latency control.** An 18M model that runs at 1000 samples/s makes a real efficiency argument.
5. **Clean ablations.** Channel dropout, conv-stem on/off, positional-encoding variants, task-id vs text query — each is a controlled 2-run experiment on a model you fully own. You cannot ablate Qwen3-VL's positional encoding.

## What it CANNOT show — be blunt about this

1. **Language.** A from-scratch decoder trained on templated questions is a template classifier with extra steps. It will not generalise to a rephrased question, will not explain itself, will not do chain-of-thought, and will not survive an out-of-distribution query. **Any README language implying "it understands questions" is indefensible.**
2. **World knowledge.** Qwen3-VL knows what a bakery is. Yours knows that channel 4 has high values in some places. Every task requiring semantics not present in the raster is out of reach by construction.
3. **Transfer to real maps.** Trained only on synthetic rasters, it will transfer to real data approximately not at all. State this; do not test it and quietly not report.
4. **Competitive absolute numbers on any public benchmark.** It cannot run MapEval (note 06) at all — no tool-use, no language. The two tracks are not on the same axis.
5. **The agentic/GRPO story.** Tool use, multi-turn, verifiable rewards — the artifact's actual thesis — lives entirely on the Qwen3-VL track. The from-scratch model is a *perception control*, not a second agent.

**The framing that is both honest and strong:** the from-scratch model is not a competitor to Qwen3-VL. It is the **measurement instrument** that tells you how much of the fine-tuned model's performance comes from perception of your rasters versus from pretrained language and world knowledge. Frame it as a control, and every result — including bad ones — becomes informative. Frame it as a competitor and you will be defending an unwinnable comparison in an interview.

## The 2–3 most informative comparisons to run

1. **Matched-perception ceiling: from-scratch (native N channels) vs Qwen3-VL (RGB render of the same scene), on the point/geometry tasks only.**
   The single most informative run. If the 18M from-scratch model beats a fine-tuned 3–8B VLM on "what is nearest to the marker", you have demonstrated that RGB rendering discards task-relevant information and that native multi-channel input recovers it — a clean, defensible, quantified claim about representation, not about scale. If it loses, that is also a real result about how much pretrained visual priors are worth, and it is equally reportable.

2. **Data-scaling curves on the same axis: accuracy vs samples-seen, both tracks.**
   Qwen3-VL should start far higher (pretraining) and saturate early; from-scratch should start at chance and climb. **Where they cross is the quantitative answer to "how much synthetic data is one pretrained VLM worth" for your task family** — a genuinely interesting number that nobody has published for this domain, and cheap to produce since you control the generator.

3. **Query-format ablation on the from-scratch model: task-id token vs templated text.**
   Isolates how much of the task is memorised behaviour versus query understanding. Directly bounds what the language side of Qwen3-VL is actually contributing, and it is the cheapest of the three.

(If you only have budget for two: run 1 and 2. Comparison 1 is the one that produces a headline; comparison 2 is the one that produces the figure.)

---

## Verification log

| Item | Source | Checked |
|---|---|---|
| nanoVLM: 5,027★, Apache-2.0, last commit 27 Oct 2025 | GitHub REST API + commits API | 19 Sep 2026 |
| nanoVLM blog: 21 May 2025, 222M, ~1.7M cauldron samples, ~6 h H100, 35.3% MMStar, VRAM table | huggingface.co/blog/nanovlm | 19 Sep 2026 |
| seemore: 260★, MIT, created 17 Apr 2024, last push 6 May 2024 | GitHub REST API | 19 Sep 2026 |
| vit-pytorch: 25,512★, MIT, last push 5 Sep 2026 | GitHub REST API | 19 Sep 2026 |
| hkproj/pytorch-paligemma: 634★, no licence, last push 6 Dec 2024 | GitHub REST API | 19 Sep 2026 |
| tintn/vision-transformer-from-scratch: 263★, MIT, last push 10 Jun 2024 | GitHub REST API | 19 Sep 2026 |
| big_vision: 3,541★, Apache-2.0, last push 19 May 2025 | GitHub REST API | 19 Sep 2026 |
| Compact-Transformers: 545★, Apache-2.0, last push 5 Nov 2024 | GitHub REST API | 19 Sep 2026 |
| SPT_LSA_ViT: 131★, no licence, last push 23 Feb 2022 | GitHub REST API | 19 Sep 2026 |
| moein-shariatnia/Pix2Seq: 131★, MIT, last push 2 Sep 2023 | GitHub REST API | 19 Sep 2026 |
| DOFA: 214★, MIT, last push 22 Jul 2026 | GitHub REST API | 19 Sep 2026 |
| Clay: 615★, Apache-2.0, last push 11 May 2026 | GitHub REST API | 19 Sep 2026 |
| nanochat-VLM: 26★, created 23 Oct 2025, last push 16 Sep 2026 | GitHub REST API | 19 Sep 2026 |
| Raschka article: 3 Nov 2024, ~25–30 min, 10 papers reviewed | magazine.sebastianraschka.com | 19 Sep 2026 |
| Beyer 2205.01580, Steiner 2106.10270, Lee 2112.13492, Xiao 2106.14881, Chen 2109.10852, SatMAE 2207.08051 | arXiv listings/abstracts | 19 Sep 2026 |

**Explicitly UNVERIFIED:** all line-of-code figures (taken from README claims, not measured); Umar Jamil video length; nipunbatra/vlm-from-scratch publication date and licence; all throughput and wall-clock estimates in §3.6; Donut-vs-Pix2Struct relative quality (community claims from secondary sources, not benchmarked); the original ViT paper (2010.11929) was not re-fetched today.
