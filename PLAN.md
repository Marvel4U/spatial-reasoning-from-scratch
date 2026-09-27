# PLAN — current working plan (set 19 Sep 2026)

This file is the live plan. It supersedes the *starting point* of `DESIGN.md` (the P0–P3 ladder and stage structure there remain the long-range frame; what changed is how P0 is defined and that there are now two model tracks). History of how we got here: `survey/SURVEY_2026-09-11.md`, `PILOT_de_pijp_2026-09-12.md`.

## The idea in one paragraph
The world is a stack of **spatially aligned, very simple raster layers** cut from a frozen real-city snapshot (De Pijp first): noise, surface type, establishments, later trees, heights, transit. Each layer has only 3–5 widely spaced classes. A model's first job (the new P0) is to **relate the same place across layers** and talk or point about it: "what is at this point", "where is quiet sidewalk near a café". Everything is exactly verifiable from the rasters. Paths over these layers are the next step after points and areas. Marvin, 19 Sep: "this feels like a bit of a breakthrough … both achievable and useful".

## Two tracks, same data, same tasks, same verifier
| | **Track A — "Qwen"** (fine-tune) | **Track B — "local"** (from scratch) |
|---|---|---|
| Model | small open VLM (Qwen3-VL-class, 2–8B), LoRA SFT, later GRPO | own small vision transformer, ~10–20M params, typed by Marvin |
| Input | layers packed into **RGB** (one layer per channel) at 1024 px + a text legend; question in natural language; later the model **chooses which 3 layers to view** (zoom + layer access as one tool call) | the same layers as **native N channels** (one-hot per class), 256 px; question = marker channel + **command tokens** (no natural language) |
| Output | text (class, number, normalized 0–1000 coordinates) | tokens from a tiny vocabulary (task ids, classes, coordinate bins à la Pix2Seq); **experiment:** a 64×64 output grid so the model can *point* at locations / surfaces / areas |
| Question it answers | can pretrained vision be re-wired to read our channel code ("detangle the colours")? | how much of the task is learnable by a model that does this and nothing else; what does native multi-channel input buy? |
| Role | the portfolio credential (post-trained open model, agentic ladder P1–P3, tools, RL) | measurement instrument and control; home of the multi-channel architecture idea; Marvin's own model code |
| Priority if time is short | **first** | second |

**Status of the Track B column (20 Sep): everything in it is a proposal, not a decision.** Input format, resolution, vocabulary, heads and size are Marvin's to decide after working through `LOCAL_MODEL_GUIDE.md`; the Track B fields in the task items are equally provisional and cheap to regenerate.

Comparability rule: the tracks are compared only on perception-level tasks (point, area, later path), on identical crops and identical ground truth. Tool use, language and RL exist only in Track A.

## Data contract (what the generator must guarantee)
- One **canonical label stack** per crop: integer class rasters per layer. Every model input (RGB PNG for A, one-hot array for B) is derived from it, so both tracks see the same world.
- Same ground footprint for both tracks: **256 m × 256 m**, north up. Track A render 1024 px (0.25 m/px; 1024 is a multiple of Qwen3-VL's 32 px token size, so no resampling — verify in the processor when the model is chosen). Track B render 256 px (1 m/px → 16×16 tokens after the stem).
- Layer classes v2 (3–5 levels, widely spaced; RGB grey values in brackets):
  - noise Lden: <55 dB [0], 55–65 [85], 65–75 [170], ≥75 [255]
  - surface: none [0], roadway incl. cycle path and parking [85], sidewalk/pedestrian [170], building [255]
  - establishments (named OSM POIs, discs radius **2 m**, was 4 m — too large): none [0], other named [85], shop [170], food & drink [255]
- No data cleaning (raw oddities stay; see memory/feedback). The JSON sidecar, not the pixel, is ground truth where discs overlap.
- **Spatial split inside the district: scattered held-out blocks.** The district bbox is cut into a 4×4 grid of equal blocks, indexed (row, col) with row 0 = north, col 0 = west (De Pijp: 446 m × 448 m each). `VAL_BLOCKS = [(0,2), (3,1)]`, `TEST_BLOCKS = [(1,0), (2,3)]`, the other twelve are train ground. A val or test crop lies **fully inside one block of its own split** — so val and test now sit on **disjoint ground** — and a train crop must **not intersect any held-out block** (it may straddle adjacent train blocks). Scattering the held-out ground means all three splits see the same mix of city fabric (canal belt, park, market street), which a single west/east cut did not give. Geometry written to `crops/v2/split.json`, picture in `overlays/crops_v2_split.png`. Held-out *cities* (Barcelona, Paris district, Vienna) come later and are the real generalisation test.
- Known imbalance (v1 numbers): buildings 34 % of pixels, establishments 2 %, cycle path 2 % → questions are **sampled per class**, not per pixel.

## First task family — T0 "point read" (v0, deliberately trivial)
"At this point, what is the {surface | noise band}?" One point, one layer, one class as the answer. Purpose: the sanity rung. For Track A it isolates two skills the predecessor study found broken — locating a coordinate in the image and reading our colour code. For Track B it is the "does the pipeline learn at all" check.
- Track A form: RGB tile + legend + "What is the surface class at (x=412, y=733)?" (normalized 0–1000), answer = class word.
- Track B form: N-channel tile + marker channel + `<task:surface_at>` → class token.
- Every item stores the point's distance to the nearest class boundary, so accuracy can be stratified instead of filtering hard cases out.
Next rungs (not built yet): T1 cross-layer point ("is there food & drink within 20 m of this point"), T2 area / mask ("mark quiet sidewalk" = sidewalk ∧ noise<55 → the 64×64 pointing output), T3 paths (share of a path on sidewalk, mean noise along it).

## Order of work
1. ✅ Snapshot pilot, 7 layers (12 Sep). ✅ RGB tiles v1 (19 Sep).
2. Crop generator v2 (canonical label stack, both renders, spatial split) + T0 items. ← in progress
3. **Baseline before training:** untrained Qwen3-VL-class model (and one frontier model as reference) on T0, a few hundred items. Tells us whether T0 is trivial, impossible, or useful.
4. Track B: Marvin designs and types the local model, starting from `LOCAL_MODEL_GUIDE.md` (further resources: `survey/raw/07_vision_from_scratch_resources.md`; ladder: overfit one batch → T0 → T1).
5. Track A: LoRA SFT on T0/T1 on the 3060 with the smallest model; compare to baseline and to Track B.
6. Then T2/T3, layer-selection tool, and the DESIGN.md ladder (P1–P3, GRPO) on Track A.

## Authorship
**The local model is built only together with Marvin, line by line, in sessions he leads — Claude never builds it ahead (19 Sep).** Marvin types: the local model, its training loop, Track A model loading / LoRA / training / RL code. Collaborative (Claude + agents, Marvin reviews): snapshot layers, crop and task generators, eval harness, renders. Review habit: Marvin reads each new module; modules stay short (target ≤ 250 lines).

## Parked (deliberately not now)
Derived shadow layer, sidewalk width, elevation in tasks, opening hours, licensing posture, MapEval-API / MapReason-OSM as secondary metrics, PERIA-style image tools.
