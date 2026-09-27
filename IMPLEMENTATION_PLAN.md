# IMPLEMENTATION_PLAN — local grid ViT (Track B)

Live specs: `LOCAL_MODEL_GUIDE.md`, `LOCAL_EXPERIMENTS.md`, `PLAN.md`.  
Working style: **one model file per step** in Phase A (you review each diff); **Phase C harness in one batch** when model + data are ready.

Folder names: **`plain_gpt_module/`** (add files only), **`spatial_data/`** (loaders), **`harness/`** (evolved from `harness_from_translator/` template).

---

## Locked for v1 (from LOCAL_EXPERIMENTS)

| ID | Choice |
|----|--------|
| D1 | Dense **64×64 grid head** first; token decoder later |
| D2 | **Marker channel** for location (no coordinate tokens) |
| D3 | Fixed **2-D sin-cos** positions |
| D4 | **2 encoder blocks**; default `n_embd=384`, `h_heads=6` |
| D5 | Fixed layer stack in the input |
| D6 | **c0:** layers + marker → predict marker on grid; **c1:** add task embedding |
| D7 | Adapt copies of `plain_gpt_module` + harness in this repo |

**First encoding default (E2):** scalar class channels (3 layers) + marker → **`Cin = 4`**. One-hot is a harness override, not a separate code path rewrite.

---

## c0 and **K** (grid head classes per cell)

**c0** is rung **c0** in `LOCAL_EXPERIMENTS.md`: a deliberately trivial task — *“given the marker in the input channels, mark the same place on the 64×64 output grid.”* It validates patch order, position code, and head reshaping before the model must read surface/noise semantics.

The grid head outputs, for **each** of the 64×64 cells, a vector of length **K** (then softmax / cross-entropy against a target class per cell):

| K | Meaning | c0 target | Metric |
|---|---------|-----------|--------|
| **K = 2** (recommended start) | Each cell: class **0 = background**, **1 = marker cell** | Exactly one cell (or small neighborhood) is class 1 | IoU for class 1; **exact-cell** accuracy (did the argmax land on the true marker cell?) |
| K = 1 + regression | Not used in v1 | — | — |
| K > 2 (later, c1+) | e.g. background + “sidewalk” + … | Multi-class mask from a layer | IoU per class |

So **K is not “number of map layers”** — it is **how many logits per output pixel** on the 64×64 grid. For c0, **K = 2** is the simplest story: binary segmentation “where is the marker?”

Default config unless changed: **`num_grid_classes = 2`**, **`task_rung = c0`**.

---

## Architecture (first milestone)

```
img (B, Cin, 256, 256)  →  PatchEmbed  →  + sincos_2d  →  2× EncoderBlock  →  ln_enc
                                                              ↓
                                                    GridHead → logits (B, 64, 64, K)
```

- **Reuse (import only):** `EncoderBlock`, `MLP`, `BidirectionalSelfAttention` from `plain_gpt_module.encdec_model`.
- **New under `plain_gpt_module/`:** `pos_embed.py`, `patch_embed.py`, `grid_head.py`, `local_config.py`, `local_grid_vit.py`.
- **Default geometry:** input **256 px**, patch **P = 16** → 16×16 tokens; subcells **s = 4** per patch → **64×64** logits (see LOCAL_MODEL_GUIDE §7c).

Not in v1: decoder, task embedding (c1), token vocabulary.

---

## Phase A — Model primitives (`plain_gpt_module`)

One file per step; do not edit existing module files unless unavoidable. Defer `__init__.py` exports until you want a public API.

| Step | File | Purpose | Exit check |
|------|------|---------|------------|
| **A1** | `pos_embed.py` | `sincos_2d(g, n_embd)` (+ temperature hook for E5) | Shapes `(N, n_embd)`; positions distinct on 16×16 |
| A2 | `patch_embed.py` | `PatchEmbed` (guide §2) | `(B,Cin,H,W) → (B,N,n_embd)` |
| A3 | `grid_head.py` | Patch features → `(B,64,64,K)` | Reshape math matches P=16, s=4 |
| A4 | `local_config.py` | Dataclass: channels, patch, width, depth, grid, K | — |
| A5 | `local_grid_vit.py` | Full module + CE loss on grid | Forward/backward; ~2-block param count |
| A6 | (optional) | `python -m plain_gpt_module.local_grid_vit` smoke | Loss drops on random batch |

**Next up:** A1.

---

## Phase B — Data (`spatial_data/`)

Can start after A2; needs crop npz on disk for real tiles (`data/.../crops/v2/*_labels.npz`). c0 can use synthetic markers early.

| Step | File | Purpose |
|------|------|---------|
| B1 | `channels.py` | npz → `(Cin,H,W)`; `scalar` \| `onehot`; marker plane |
| B2 | `targets.py` | c0: build `(64,64)` int labels from `marker_256`; later c1 masks |
| B3 | `dataset_c0.py` | `__getitem__` → image, target grid |
| B4 | `loader.py` | `next_batch()` for harness; expose `.B` |
| B5 | `setup.py` | device + autocast context |

---

## Phase C — Harness (`harness/`) ✅

Implemented (Sep 2026). Template = `harness_from_translator/`; run from **`harness/`** — see `harness/README.md`.

| File | Role |
|------|------|
| `config.py` | Spatial defaults + `local_grid_vit_config_dict()` |
| `data.py` | Synthetic c0 pools until Phase B |
| `grid_vit_adapter.py` | Model build, train loop, checkpoints |
| `eval_spatial.py` | Val loss, IoU (fg), exact-cell accuracy |
| `run_log.py` | `runs/experiments.json` + resume fingerprint |
| `harness.py` | Experiment driver |
| `experiments.py` | `TO_RUN`: `c0_overfit_32`, `c0_synthetic_1k` |
| `train.py` | Manual entry |

---

## Phase D — Sanity ladder (experiments)

| Rung | Experiment | Pass |
|------|------------|------|
| L1 | Overfit 32 samples | Loss ~0; exact-cell 100% |
| L2 | c0 on val crops | IoU ≈ 1 (trivial task) |
| L3 | E1 patch size sweep | Log vs boundary when T0 points linked |
| L4 | E2 scalar vs one-hot | Compare at matched steps |
| L5a | **c1** building mask only | Dense targets + eval (no task emb) — **done** |
| L5b | **c1** four tasks + task emb | **done** — see `LEARNING_REPORT.md` §3.9–3.10 |
| L5c | Factorised tasks | Backlog (`C1_PLAN.md`) |

Defer: token decoder (E7), ChannelViT (E3), conv stem (E4), Track A baseline.

---

## Experimental dimensions (harness overrides)

| ID | Axis |
|----|------|
| E1 | Patch P and/or input 256 vs 1024 px |
| E2 | Scalar vs one-hot channels |
| E3 | Fused patch embed vs separate tokens per layer + layer emb |
| E4 | Linear patchify vs conv stem |
| E5 | Sin-cos temperature / learned pos |
| E6 | Task emb add vs extra token (c1+) |
| E7 | Grid + decoder |
| E8 | Depth/width |
| E9 | CLS / mean / indexed readout (when classifier head exists) |

---

## Doc sync (when convenient)

- `PLAN.md` Track B column still mentions one-hot globally; **`LOCAL_EXPERIMENTS` + this file** govern implementation (scalar first).
- Record passing run ids in `LOCAL_EXPERIMENTS.md` as rungs clear.
