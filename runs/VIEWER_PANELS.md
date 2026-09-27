# c0 static task viewer — what each panel shows

Figures from `python -m analysis.view_sample` ([spec §13.1](../../docs/plans/ANALYSIS_SUITE_SPEC.md); usage [`analysis/README.md`](../../analysis/README.md)). Scale: 256 px input = 256 m → **1 px = 1 m**. Output grid is **64×64** (≈4 m per cell). Patch size **P** (e.g. 16) defines ViT tokens and in-patch subcells **s = 64 / (256/P) = P/4** (e.g. P=16 → s=4).

**Layout:** left column — **A**, then **B** (channels side-by-side, same mosaic style as **C**), then **C**. Top row centre/right — **D** and **E**. Middle row centre/right — **F** target zoom and **F** prediction zoom. Bottom right — **G** (metrics text).

## A — Input composite

Three layers as **RGB** (same grey code as crop `_rgb.png`: R noise, G surface, B estab), **marker crosshair**, and **patch grid** (lines every P px). The task as a human sees it, plus how the model tiles the image.

## B — Input channels

Full **256×256** planes concatenated horizontally (noise | surface | estab | marker), white dividers between channels — same layout as **C**, but at full image size. Shows what signal exists before patch embedding.

## C — Token view (what one ViT patch “sees”)

The marker lies in one **P×P px** patch (P=16 → 16×16 m). This panel **crops that patch** from every input channel and lays channels **side by side** (noise | surface | estab | marker).

The **white grid** on the first channel is the **s×s subcell grid** inside the patch (P=16 → s=4 → 4×4 subcells, each 4×4 px). The **green box** is the **true** subcell that contains the marker.

This matches the input to **one patch token** before the transformer: “given this local snippet, which of the s² subcells is the answer?” At P=16 it is a 16×16 px collage and a 16-way subcell choice on the output grid; at P=4 the patch is tiny and s=1 (no real in-patch subproblem).

## D — Target grid (what training asks for)

The **64×64** c0 target: **one foreground cell** at the marker, downsampled from image coordinates (same rule as `spatial_data.targets.c0_target_grid`).

The **cyan square** is the **patch block** on the output grid: the s×s cells that belong to the marker’s image patch (for P=16, a 4×4 block = 16 cells). With a perfect single-cell target, that block is exactly the patch containing the one hot cell.

## E — Predicted P(foreground) (what the model “draws”)

**Softmax/sigmoid foreground probability** over the full 64×64 grid (fixed colour scale 0–1). Output only — no target overlays. This is the continuous picture behind argmax, IoU, and “full patch blob” metrics.

- **Red cell** (or **green** if the high end of the colormap is already red): **global argmax** of the fg score, drawn as a single pixel/cell.

Typical P=16 failure mode: a **bright 4×4 blob** (all subcells in one patch lit), with the argmax cell on one corner of that blob — high patch hit, low exact cell. Compare patch block outline on **D** only.

## F — Zoom around the marker (local behaviour)

Not the full 64×64; a crop of **3×3 patch blocks** on the output grid centred on the marker’s block (so **3s × 3s** cells, e.g. 12×12 at P=16).

- **Left:** target in that neighbourhood (usually one fg cell in the centre block).
- **Right:** P(fg) in the same window — whether the blob is tight, fills one block, or spills into neighbours.

Use F when E looks busy globally but you care whether errors are **inside the patch** vs **wrong patch**.

## G — Sample metadata and metrics

Run id, P, token count N, marker position, and per-sample c0 metrics (`n_fg`, patch hit, subcell|patch, argmax distance, etc.) tied to the picture.

---

**Read order:** **C** = perceptual difficulty at the token; **D** = correct answer; **E** = model’s soft map; **F** = patch-neighbourhood detail. Together they explain scalars like “patch hit 0.98, subcell|patch ≈ chance” without re-running eval.

**CLI:** from repo root, `python -m analysis.view_sample --help`. With checkpoint: `--ckpt runs/checkpoints/….pt --split val --item N --use-gpu`. Task only: `--no-model`.
