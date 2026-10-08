# Memo: the two Track A backbones — Qwen3-VL-2B-Instruct and Qwen3.5-0.8B (1 Oct 2026)

Both downloaded on 1 Oct 2026 into `models/hf/` on the rig (gitignored; `models/download_qwen.py` re-fetches
them). Track A, variant A1 (decided 1 Oct): the wave search runs on a pretrained VLM backbone with our exact
interface — map + scratchpad in, the whole next scratchpad + state token out, output i = input i+1.
This memo records what the two models are, what we have on disk, and where the attachment points are.
Numbers below come from `models/inspect_qwen.py` (module tree and parameter counts on the meta device,
token geometry through the real processor) and from the modelling code; nothing was run on the GPU.

## 1. What is on disk

```
models/
  download_qwen.py          re-fetch both repos (weights, configs, tokenizers; no code)
  inspect_qwen.py           structure / params / token geometry report (CPU, meta device)
  hf/Qwen3-VL-2B-Instruct/  4.26 GB  model.safetensors (bf16), config.json, generation_config.json,
                            preprocessor_config.json, video_preprocessor_config.json, tokenizer.json,
                            tokenizer_config.json, vocab.json, merges.txt, chat_template.json, README.md
  hf/Qwen3.5-0.8B/          1.75 GB  model.safetensors-00001-of-00001.safetensors (+ index.json), same set
                            of configs/tokenizer files, chat_template.jinja, README.md (62 kB model card)
  reference_code/           copies of the transformers 5.16.1 modelling code (Apache 2.0), for reading:
                            configuration_/modeling_/modular_qwen3_vl.py, processing_qwen3_vl.py,
                            configuration_/modeling_/modular_qwen3_5.py
```

Neither Hub repo ships Python: the model code is part of `transformers` (both models have been in the library
since 4.57), which is why the copies in `reference_code/` are taken from the installed package. The `modular_*`
files are the readable source (they declare only what differs from the parent models); `modeling_*` is the
generated, self-contained version that actually runs. For the weights the two files share a format:
one safetensors shard each, bf16, keys `model.visual.*`, `model.language_model.*`, `lm_head.weight`
(tied to the input embedding in both, so the head is not stored twice).

Licences: both Apache 2.0.

Environment (the project venv, uv-managed): `transformers` 5.16.1,
`torch` 2.11+cu128, `huggingface_hub` 1.31, `triton` 3.6. Added on 1 Oct: `torchvision` 0.26+cu128
(`uv pip install --no-deps`; the fast image processor needs it). Not installed: `peft` (needed for LoRA),
`fla` (Flash Linear Attention kernels for the 0.8B's DeltaNet layers; the code falls back to a pure-torch
chunked implementation without it), `trl`/`accelerate` (A2 only).

Pointers: model cards [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B); code and cookbooks
[github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL); the Qwen3-VL technical report (Nov 2025)
and the Qwen3.5 release blog (Feb 2026) describe the training stages, not the recipes.

## 2. The shared template

Both models are the same kind of object: a bidirectional ViT, a 2×2 patch merger, and a causal decoder that
receives the merged image tokens in place of `<|image_pad|>` placeholders. The processor, the vision tower and
the way images enter the LLM are identical in structure between the two; the LLMs differ (§4).

**Processor** (`Qwen2VLImageProcessorFast` + `Qwen3VLProcessor` for both). Images are resized so that both sides
are multiples of 32 px (patch 16 × merge 2), within `shortest_edge` 65 536 px² ... `longest_edge` 16.8 M px²
(i.e. 256×256 up to ~4096×4096), normalised with mean = std = 0.5 per channel. The image is duplicated into
two "frames" (temporal patch 2, the video path with a single image), so each patch arrives as a vector of
3 × 2 × 16 × 16 = 1536 values: `pixel_values` has shape (n_patches, 1536), plus `image_grid_thw` = (1, H/16, W/16).
The number of image tokens follows the pixel count: there is no fixed input size.

| our crop | pixels | ViT patches | LLM image tokens | m per token |
|---|---|---|---|---|
| 256 m at 1 px/m | 256×256 | 16×16 | 64 (8×8) | 32 m |
| 256 m at 2 px/m | 512×512 | 32×32 | 256 (16×16) | 16 m |
| 512 m at 1 px/m | 512×512 | 32×32 | 256 (16×16) | 16 m |

256 m at 2 px/m is our p16 geometry: 256 tokens of 16 m, and a dense head per token predicts 8×8 cells of 2 m,
exactly what the Grid ViT's readout does. The minimal chat prompt around one image adds 11–15 tokens
(sequence length 75 / 79 at 256 px, 267 / 271 at 512 px).

**Vision tower** (`Qwen3VLVisionModel` / `Qwen3_5VisionModel`).
- `patch_embed.proj`: `Conv3d(3 → d_vit, kernel (2, 16, 16), stride same)`; weight (d_vit, 3, 2, 16, 16).
  This is the one place where the input channel count lives.
- `pos_embed`: a learned 48×48 grid of positions (2304 entries), bilinearly interpolated to the image's patch grid
  (`fast_pos_embed_interpolate`); plus 2-D rotary embedding inside the attention (`rotary_pos_emb`).
- `blocks`: pre-norm transformer blocks with **full bidirectional attention** over all patches of an image
  (`is_causal = False`; `cu_seqlens` separates images within a batch). No window attention (that was Qwen2.5-VL).
- `merger`: 2×2 neighbouring patches concatenated (4·d_vit) → LayerNorm → MLP → `out_hidden_size` = the LLM
  width. `linear_fc2` is (d_llm, 4·d_vit) ... (2048, 4096) for the 2B, (1024, 3072) for the 0.8B.

**Into the LLM.** The tokenizer emits `<|vision_start|>`, N × `<|image_pad|>`, `<|vision_end|>`; the model
replaces the placeholders' embeddings with the merged image tokens (`masked_scatter` on `inputs_embeds`,
`get_placeholder_mask`). Positions are multimodal RoPE ("M-RoPE", interleaved variant): each token carries three
position indices (time, height, width); text tokens advance all three together, image tokens take their
(row, column) in the merged grid. So the LLM does know where an image token sits on the map.

**Decoder.** Causal. For a dense readout this is the decisive property: the hidden state at image token (i, j)
has seen only tokens before it in raster order. The grid cannot be read from the image token positions; it has
to be read from tokens placed after the image (and after any text), see §5.

## 3. Qwen3-VL-2B-Instruct

`Qwen3VLForConditionalGeneration`, 2 127.5 M parameters, bf16 file 4.26 GB.

| part | module | params | shape |
|---|---|---|---|
| vision tower | `model.visual` | 407.0 M | 24 blocks, width 1024, 16 heads, MLP 4096, GELU-tanh |
| of which merger | `model.visual.merger` | 25.2 M | 4096 → 2048 |
| of which DeepStack mergers | `model.visual.deepstack_merger_list` | 75.5 M | 3 extra mergers |
| decoder | `model.language_model` | 1 720.6 M | 28 layers, width 2048, MLP 6144 |
| of which embeddings | `embed_tokens` | 311.2 M | 151 936 × 2048, tied with `lm_head` |
| per decoder layer | `layers[i]` | 50.3 M | GQA 16 q / 8 kv heads, head dim 128, RMSNorm, SiLU MLP |

Non-embedding decoder parameters ≈ 1.41 B. RoPE θ 5·10⁶, `mrope_section` [24, 20, 20] interleaved, context
262 144. Vocabulary 151 936; special ids: image pad 151655, vision start/end 151652/151653, `<|im_end|>` 151645
(eos). Chat template: ChatML (`<|im_start|>user ... <|im_end|>`), no thinking block in the Instruct variant.

**DeepStack** (the one architectural novelty over Qwen2.5-VL): the ViT's hidden states after blocks 5, 11 and 17
are passed through three additional mergers (same shape as the main one) and **added to the LLM's hidden states
at the image token positions after decoder layers 0, 1 and 2** (`_deepstack_process`). The LLM therefore sees
low-, mid- and high-level visual features, not only the top of the ViT. For us this matters in two ways: the
image tokens inside the LLM carry more local detail than a single-projection VLM would give (good for a dense
head), and any hook that replaces or augments the image embeddings has to be aware that three more injections
follow in layers 0–2.

## 4. Qwen3.5-0.8B

`Qwen3_5ForConditionalGeneration`, 853.0 M parameters, bf16 file 1.75 GB (the file also holds a one-layer
multi-token-prediction head, `mtp.*`, ~20 M, which the model class ignores on load).

| part | module | params | shape |
|---|---|---|---|
| vision tower | `model.visual` | 100.6 M | 12 blocks, width 768, 12 heads, MLP 3072 |
| of which merger | `model.visual.merger` | 12.6 M | 3072 → 1024 |
| decoder | `model.language_model` | 752.4 M | 24 layers, width 1024, MLP 3584 |
| of which embeddings | `embed_tokens` | 254.3 M | 248 320 × 1024, tied with `lm_head` |
| per decoder layer | `layers[i]` | 21.6 M | see below |

Non-embedding decoder parameters ≈ 0.50 B, i.e. the "0.8B" is 30 % vocabulary. No DeepStack
(`deepstack_visual_indexes = []`): the ViT's top output alone enters the LLM.

**Hybrid decoder.** `layer_types` repeats `[linear, linear, linear, full]` six times: layers 3, 7, 11, 15, 19, 23
are softmax attention, the other 18 are Gated DeltaNet.
- `Qwen3_5Attention`: GQA with 8 query / 2 key-value heads of dim 256, an output gate (`attn_output_gate`), rotary
  on a quarter of the head dim (`partial_rotary_factor` 0.25, θ 10⁷, `mrope_section` [11, 11, 10]).
- `Qwen3_5GatedDeltaNet`: 16 heads with key and value dim 128, a depthwise causal conv of kernel 4 on the
  projected q/k/v, a gated delta-rule recurrence (`A_log`, `dt_bias`, state kept in float32,
  `mamba_ssm_dtype`). In transformers the recurrence runs through `torch_chunk_gated_delta_rule`
  (training/prefill) and `torch_recurrent_gated_delta_rule` (decoding); both are wrapped with
  `use_kernel_func_from_hub_with_fallback`, which swaps in the `fla` Triton kernels when installed.
  Without `fla` the chunked torch path is used: correct, slower, more memory.

The linear layers are a fixed-size state scanned over the sequence; they summarise, they do not look up. Only the
six attention layers can match a query token against a specific image token at a specific position. For the
dense readout (§5) this is the open question about this model: whether six attention layers are enough for 256
query tokens to each find their own patch, or whether the recurrent layers smear the map. The frozen-backbone
probe answers it.

Vocabulary 248 320 (larger than the 2B's); special ids: image pad 248056, vision start/end 248053/248054,
`<|im_end|>` 248044. The chat template is the thinking template: the generation prompt ends in
`<think>\n\n</think>\n\n` (empty thinking block). Irrelevant for A1, where we build the sequence ourselves, but
it means the model's instruction-tuned behaviour expects that block.

## 5. Attachment points for A1 (module paths)

All of this is for the plan page; listed here because the module names are the concrete interface.

1. **Input channels** — `model.visual.patch_embed.proj` (Conv3d, weight (d_vit, 3, 2, 16, 16)). Replace by a
   Conv3d with 3 + k input channels, copy the pretrained weight into channels 0–2, zero-initialise the rest.
   At step 0 the model is the pretrained one. The processor's own channel handling (`pixel_values` rows of
   1536 = 3·2·16·16) has to be bypassed: we produce the patch vectors ourselves from our (3 + k)-channel crops,
   which is simpler than it sounds (unfold + duplicate the frame). Alternative with no change: render the layers
   to RGB and keep the processor.
2. **Scratchpad** — one of the extra channels (or a colour in the RGB variant). Same for both markers.
3. **New tokens** — `model.language_model.embed_tokens` and `lm_head` are tied; `resize_token_embeddings` grows
   both. For A1 the state token can also be a small separate head instead (41 classes as now), which avoids
   touching the 150k/248k-way softmax entirely. Recommended: separate head, read from the first query token.
4. **Query tokens** — K = n_image_tokens learned embeddings appended after `<|vision_end|>` (and after any text),
   given the M-RoPE positions of their patches (same row/column as the image token they read out, a later time
   index). Built as `inputs_embeds`, not through the tokenizer; position ids via the model's `get_rope_index`
   with a hand-made extension for the appended block. They attend causally to everything before them, which is
   the whole map; that is what makes the dense readout bidirectional in effect.
5. **Dense head** — a linear map from the final hidden state of each query token (d_llm) to 8 × 8 × 2 logits per
   16 m patch (at 2 px/m), assembled to the 128×128 grid; our CE + β loss, our eval. `output_hidden_states=True`
   or a forward through `model.model(...)` directly to get the last hidden state without the LM head.
6. **What trains** — probe: query tokens + dense head + state head only (backbone frozen). Then LoRA on the
   decoder's attention and MLP projections (`peft`), the new modules fully, the patch-embed's new channels fully;
   the ViT frozen or LoRA'd last. Full fine-tuning of the 0.8B is within reach on a rented 96 GB GPU.

Memory on the 3060 (12 GB), rough: 2B in bf16 4.3 GB weights, LoRA + gradient checkpointing at 267 tokens and
batch 8 ≈ 8–10 GB, feasible for smoke tests; 0.8B 1.7 GB weights, comfortable. Throughput will be tens of
samples per second against the Grid ViT's ~960; real runs go to a rented 96 GB GPU.

## 6. What to decide next

- Which model carries the first probe. My order: both frozen (cheap), then the one with the better probe gets
  the LoRA run; the 0.8B stays the main line only if its probe is close to the 2B's.
- RGB rendering versus extra channels for the first probe. RGB keeps the processor and everything pretrained
  intact and is the cleaner baseline; the zero-init channels are the better final design.
- `peft` and `fla` into the venv (`uv pip install peft flash-linear-attention`); `fla` only if the 0.8B stays.
