# Track A plan: Qwen3-VL on the path task — vanilla, then fine-tuned (4 Oct 2026)

Marvin builds this in Cursor; Claude delivers the shared data contract (§1) and stays out of the model code.
Background: `docs/memo/TRACK_A_BACKBONES_2026-10-01.md` (the two models, module paths, token geometry),
`docs/reports/C5_WAVE_SEARCH_2026-10-03.md` (our own network's numbers, the success rule).

## 0. The question and the shape of the answer

How well does a pretrained VLM find a walking route between two markers on our maps, (V) as it comes and
(F) after supervised fine-tuning, measured with the same success rule and on the same routes as our 9 M network?

The answer is one table and one plot: solved share by route length (2–3 / 4–5 / 6+ fronts, i.e. 40–120 m,
120–200 m, 200 m+) for: Qwen vanilla, Qwen fine-tuned at 1k / 5k / 20k examples, our one-shot detour model, our wave
search. The interesting quantity is how each curve falls with length.

Output form for Qwen: text. A route is a short polyline, `[[x, y], [x, y], ...]`, start first, goal last. No dense
grid, no new heads: that is A1, after this.

## 1. The shared contract (Claude builds, in `spatial_data/track_a_export.py`)

So that both tracks read the same routes and are scored by the same rule:

- `export`: for a store and split (De Pijp val/test, Zuid + Centrum val/test, De Pijp train for fine-tuning), one
  folder with `images/route_000123.png` and `routes.jsonl`. Per route: image file, window size in m, pixels per m,
  start and goal in image pixels, the teacher's walking length in m, the number of fronts, and the teacher's path as
  a simplified polyline in image pixels (Douglas–Peucker with a tolerance in m, so a typical route is 3–12 points).
- Rendering: buildings one dark colour, ground light, water blue, start and goal as two clearly different discs
  (green / red). Two variants to compare: `plain` (buildings + ground + markers only) and `layers` (the surface
  classes in colours, as our network sees them). 2 px per m (512 px for 256 m) by default; the scale is an argument.
- `score`: reads `routes.jsonl` and a predictions file (`{"route": 123, "points": [[x, y], ...]}` per line, image
  pixels) and reports solved / by length / failure reasons. Rule, mirroring ours: the polyline starts within 6 m of
  the start and ends within 6 m of the goal, no segment crosses a building pixel (1 m raster of the segment), and
  its length is at most 1.10 × the teacher's walking length. Failure reasons are counted separately: unparseable,
  wrong endpoints, crosses a building, too long.
- A unit test with the teacher's own polylines (must score ~100 %) and with straight lines (the trivial baseline:
  its solved share is the floor every model has to beat).

**Delivered 4 Oct** (`spatial_data/track_a_export.py`, `tests/test_track_a_export.py`; folders under `data/track_a/`):
`de_pijp_val_plain`, `de_pijp_test_plain`, `de_pijp_val_layers`, `zuid_centrum_val_plain`, `zuid_centrum_test_plain`,
`maze_v2_houses_val_plain` (500 routes each), `de_pijp_train20k_plain` (20,000). Each holds `images/`, `routes.jsonl`,
`export.json` and `baselines/`. Coordinates are image pixels, origin top-left, x right, y down; at 2 px per m all
markers and teacher points are integers. Polylines run start -> goal. The surface raster has no water class, so the
renderings show ground and buildings only (`layers` adds roadway and sidewalk).

```bash
python -m spatial_data.track_a_export score --routes data/track_a/de_pijp_val_plain/routes.jsonl --pred my_predictions.jsonl
```

The floor and the ceiling (solved share; bands 2–3 / 4–5 / 6+ fronts):

| export | teacher's polyline | straight line | straight, by band |
|---|---:|---:|---|
| De Pijp val | 100 % | 27.6 % | 73 / 36 / 2 % |
| De Pijp test | 100 % | 20.2 % | 59 / 16 / 1 % |
| Zuid + Centrum val | 100 % | 27.8 % | 64 / 25 / 6 % |
| Zuid + Centrum test | 100 % | 25.6 % | 62 / 26 / 7 % |
| house maze val | 100 % | 21.2 % | 47 / 6 / 0 % |

Every straight-line failure is "crosses a building". A model that only finds the markers and joins them scores
20–28 %; anything above that is route finding. Teacher polylines have a median of 7 points (p90 10–14).

Two things to know: (1) on about 3–10 % of the routes the teacher's path leaves the window; those records carry
`leaves_window: true` and polyline points outside the image; for fine-tuning you may want to drop them. (2) The
polyline rule allows paths up to 100 m outside the window, which our own network's rule does not, so it is slightly
more lenient than the 78 % it will be compared with; I will score our network by the polyline rule for the table.

## 2. Vanilla Qwen (V)

**Goal:** the zero-shot and few-shot numbers, and a feeling for what the model sees.

**Environment.** A separate venv for Track A (CLAUDE.md foresaw this): `transformers` (the rig's 5.16 has
`qwen3_vl`), `torch`, `torchvision`, `accelerate`, `peft`, later `trl`; `pillow`. Models are in `models/hf/`.
GPU: the 2B in bf16 needs ~5 GB for inference; my training runs use ~8 GB of the 12, so the two do not fit
together. Either you run when the queue is idle, or this is the moment for the first rented-GPU hour.

**Steps, in the order I would build them.**

1. *Load and talk.* `AutoProcessor` + `AutoModelForImageTextToText` from `models/hf/Qwen3-VL-2B-Instruct`, bf16,
   one image, "describe this image". Read while doing it: `processing_qwen3_vl.py` (how the image becomes
   `<|image_pad|>` placeholders), the chat template (`apply_chat_template`), `generate`. Check the image-token
   count against the memo (512 px → 256 tokens).
2. *Can it see the task?* Three probes before any route: "where is the green disc" / "where is the red disc"
   (compare with the true coordinates), "is the straight line between them free of buildings" (compare with our
   line-of-sight label). This tells you the coordinate convention the model actually uses and how precise it is.
   **Check the convention in the Qwen3-VL cookbook** (grounding examples): Qwen2.5-VL used absolute pixels of the
   resized image, Qwen3-VL as far as I know uses coordinates relative to 0–1000. Whatever it is, convert in one
   place and test it with the marker probe; an off-by-scale here ruins every later number silently.
3. *The route prompt.* System + user message: what the picture shows (legend), the task, the output format (JSON
   list of points), "at most 12 points", "do not cross dark areas". Greedy decoding, `max_new_tokens` ~200.
   A tolerant parser (find the first JSON array; clip to the image; drop duplicates).
4. *Score* with §1 on the 500 De Pijp val routes; then the variants, one change at a time:
   - V1 `plain` rendering vs V2 `layers` rendering
   - V3 one worked example in the prompt (one-shot, as a second image + its answer)
   - V4 image at 1024 px (4× the tokens)
   - V5 the 4B/8B model if a bigger GPU is at hand; Qwen3.5-0.8B as the small end
5. *Read failures by eye*: a script that draws the predicted polyline on the image for 20 routes per failure reason.

**What I expect** (so you can be surprised): markers found within a few metres; straight routes often right;
anything needing a detour mostly "crosses a building". The trivial straight-line baseline will be hard to beat
zero-shot. If that is what happens, it is the result: the task needs spatial computation, not recognition.

**Decisions that are yours:** the prompt wording and legend; whether the few-shot example counts as "vanilla";
which resolution is the reference.

## 3. Fine-tuned Qwen (F)

**Goal:** the same table after supervised fine-tuning on (image, teacher polyline) pairs, and a data-scaling curve.

**Data.** From §1's export of De Pijp train routes: 1k, 5k, 20k examples (nested subsets). One example = the chat
of §2.3 plus the assistant answer = the teacher's simplified polyline in the model's coordinate convention.
Keep a 200-route slice of train out as a quick dev set; the 500 val routes stay for the reported number.

**Method, in the order I would learn it.**

1. *One batch by hand.* Build `input_ids`, `pixel_values`, `labels` for one example with the processor; labels are
   −100 everywhere except the assistant answer (find the assistant span via the chat template; this masking is the
   classic place for a silent bug — print the decoded label tokens). Forward, look at the loss (it should be the
   CE of ~40 answer tokens, a few nats at the start).
2. *LoRA with `peft`.* `LoraConfig` on the decoder's attention projections first (`q_proj`, `k_proj`, `v_proj`,
   `o_proj`), r 16, alpha 32, dropout 0.05; vision tower and merger frozen. Count trainable parameters
   (~0.5 % of the model). Later ablation: MLP projections too; LoRA on the vision tower; merger unfrozen.
3. *Your own loop first.* You have a harness mindset: AdamW on the LoRA parameters, lr 1e-4 to 2e-4, warmup +
   cosine, batch 4–8 with gradient accumulation, bf16 autocast, gradient checkpointing if memory is tight.
   On the 3060 the 2B fits in bf16 for training at ~270 tokens per example with batch 4; 4-bit loading
   (`bitsandbytes`) is the fallback and worth knowing, not needed here.
4. *Then the same with `trl`'s `SFTTrainer`* (or `transformers.Trainer` with a collator that does step 1): the
   point is to see which of your loop's decisions the library takes for you, since knowing that is the point of writing the loop by hand first.
5. *Evaluate by generation*, not by loss: every N steps generate on the dev slice and score with §1. Watch the
   failure reasons shift (unparseable → wrong endpoints → crosses a building → too long).

**Experiments.**

| id | what | why |
|---|---|---|
| F1 | 5k examples, decoder LoRA, 512 px, `plain` | the reference |
| F2 | 1k and 20k | data scaling: does it learn the rule or the routes? |
| F3 | + LoRA on the vision tower / merger unfrozen | is the bottleneck seeing or reasoning? |
| F4 | `layers` rendering | does colour information help or distract? |
| F5 | 1024 px | precision vs tokens |
| F6 | evaluate F1 on Zuid + Centrum and the house maze | transfer, against our 60 % / 43 % |
| F7 (later) | answer = the search written out (fronts as coordinate lists, then the path) | chain of thought as the text scratchpad; the bridge to GRPO |

**What to compare against** (same 500 routes, same rule; numbers as of 4 Oct): wave search after one round of
natural noise 78 % De Pijp val (96 / 89 / 63 % by length), 60 % on the never-seen map; one-shot detour model: to be
scored by the polyline rule too (Claude adds it to the table).

**Pitfalls to expect.** The processor resizes to multiples of 32 px, so the coordinate frame is the resized image;
`min_pixels`/`max_pixels` change the token count silently; padding side and `eos` in batched generation; the chat
template of Qwen3.5 adds an empty thinking block; JSON with a trailing comma; x/y order.

## 4. After this

A1 (the dense scratchpad on the Qwen backbone: query tokens + grid head, memo §5) builds on the fine-tuned
backbone that already reads our maps; F7 leads to GRPO with the scorer as the reward.

## 5. Who does what, in order

| step | who |
|---|---|
| export + scorer + straight-line and teacher baselines (§1) | Claude, now |
| venv, load, probes, route prompt, V1–V4 | Marvin |
| one batch by hand, LoRA, own loop, F1 | Marvin |
| `trl` version, F2–F6 | Marvin |
| our models scored by the polyline rule for the comparison table | Claude |

## 6. Results (Oct 2026)

**Report:** [`docs/reports/TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md`](../reports/TRACK_A_QWEN_VANILLA_AND_SFT_2026-10-08.md)
— vanilla 2B/4B A/B, LoRA scaling (100 → ~19.3k train), val-500 numbers, failure modes, resume/checkpoints, open items.

**Headline (De Pijp val 500, `route_v1`, polyline scorer):** 2B vanilla ~2.5% (200-route A/B); 4B vanilla ~25% (200);
LoRA 2B at 1k / 5k / full train → **21% / 31% / 50%** solved. F1 (5k) and F2 (full train) **done** on rig; F3–F6,
wave-on-polyline table, and `trl` **not done**. Training code: `qwen_fine_tuning/train_lora.py`.
