# Memo — establishments as building footprints (cropsets v3 and v3b)

21 Sep 2026. Status: **implemented and generated, including the area-capped variant v3b (§6, decided the same day).** Code: `worldsnap/estab_footprints.py` (new, 50 lines), `worldsnap/crops.py` (cropset is now a parameter). Data: `data/amsterdam/de_pijp/crops/v3/` and `crops/v3b/`. v2 is untouched and still reproduces bit-exactly with the patched code.

## 1. Problem

In cropsets v1/v2 a named establishment (OSM point) was drawn as a **disc of radius 2 m = 13 m²**. One output cell of the 64 × 64 grid is 4 × 4 m = 16 m², so a disc is *smaller than one cell*:
- almost every establishment target cell is only partly covered (the "any pixel" rule was needed to have targets at all);
- at P = 16 the model has to find a ~4 px dot inside a 16 × 16 px patch and read one of three grey levels (0.33 / 0.67 / 1.0) from it: the same kind of sparse, sub-patch target that made c0 hard;
- the class is so rare (food & drink: 0.36 % of cells) that the loss weight hit its clamp of 100. A weight of 100 tells the model to mark every cell with more than ~1 % chance of being positive. Result in L5b / L5c: recall 0.97, precision 0.49, IoU 0.48 for food & drink; on crops without any such establishment the model still marked something 76–100 % of the time.

A threshold sweep on the saved checkpoints (`docs/plans/perf_prototypes/c1_threshold_sweep.py`) separated the two causes: with a stricter decision threshold (0.95–0.99, exactly where theory puts it for w = 33–100) food & drink rises from 0.48 to 0.66. So about half of the deficit was the loss weight, the rest is genuine difficulty of the tiny target.

## 2. Why not simply larger discs

| Disc radius | Area in grid cells | Discs overlapping another disc |
|---|---|---|
| 2 m (v2) | 0.8 | 5 % |
| 3 m | 1.8 | 32 % |
| 4 m | 3.1 | 50 % |
| 6 m | 7.1 | 63 % |

On the shopping streets establishments sit 5–6 m apart. Larger discs collide, and where they collide the higher class overwrites the lower one: information is destroyed (food & drink paints over the shop next door).

## 3. The fix: paint the hosting building

An establishment occupies a building. Measured on De Pijp (1,414 named establishments, 9,775 3DBAG buildings):

| Measurement | Value |
|---|---|
| Establishments inside a building footprint | 1,351 = 95.5 % |
| Hosting buildings | 1,204 |
| … hosting exactly one establishment | 91 % |
| … hosting establishments of more than one class | 47 = 4 % |
| Footprint of a hosting building, median | 96 m² ≈ 6 grid cells |

The Dutch building register records narrow individual houses, so nearly every establishment maps to its own house. Rules in `estab_footprints.py` (nothing is cleaned):
1. A building takes the **highest class** among the establishments inside it (same priority as before: food & drink > shop > other named).
2. An establishment **outside every footprint** (63 of 1,414: market stalls, kiosks) **keeps its 2 m disc**.
3. Footprints are painted first, discs after, lower class first, so the higher class wins wherever things overlap.

Meaning changes from "an establishment's point is here" to **"this building houses an establishment of class k"**. The street-side position (the entrance) is no longer in this layer; if it matters later it becomes its own layer.

## 4. What v3 looks like

(v3 numbers: all 2,000 training crops; an earlier draft quoted a 400-crop subsample.) Same seed → **identical crop origins and ids as v2**; noise and surface layers are bit-identical (checked on three crops); only the establishment layer differs. So any v2-vs-v3 comparison isolates the target geometry. Figure: `data/amsterdam/de_pijp/overlays/crops_v2_vs_v3_estab.png`.

| | v2 (discs) | v3 (footprints) |
|---|---|---|
| Establishment pixels, district | 0.54 % | 9.0 % |
| … of which on building surface | n/a | 99.7 % |
| food & drink, share of grid cells | 0.36 % ("any" rule) | **2.06 %** (majority rule) |
| shop | 0.26 % | 3.02 % |
| other named | 0.43 % | 3.41 % |
| Implied positive weight (1 − f)/f, food & drink | 277 → clamp 100 | **48** |
| Crops with an empty food & drink target | 19.5 % | 9.1 % |
| Establishment cells that are pure (all 16 px one class) | ≈ 0 | 50.6 % |
| "any" vs "majority" cell count | 5.2× apart | 1.38× apart |

The task becomes a region task like "building" (which the model solves at IoU 0.97): "which buildings carry value k in the establishment channel".

## 5. What has to change on the training side (not done; Marvin's files)

1. `harness/config.py`: `cropset = "v3"` (or as an experiment override).
2. `spatial_data/c1_tasks.py`: for v3 the establishment tasks should use **`downsample="majority"`** (currently `"any"` for the estab layer), and `TASK_CELL_FG_FRACTION` needs per-cropset values (majority rule, all 2,000 training crops): **v3** `estab_0: 0.920`, `estab_1: 0.0341`, `estab_2: 0.0302`, `estab_3 = food_drink: 0.0206`; **v3b** `estab_0: 0.937`, `estab_1: 0.0246`, `estab_2: 0.0251`, `estab_3 = food_drink: 0.0172`. The fractions depend on the cropset, so the table should be keyed by cropset (or computed from the crops at start-up).
3. c0 and the T0 point items keep working: crop ids are identical, only the cropset folder differs.
4. The legend text in `crops/v3/legend.json` already says "drawn as the footprint of the building that houses them".

Suggested experiment: L5b on v3 with majority rule and the new weights, next to the queued v2 clamp runs. Prediction: food & drink IoU well above the 0.66 ceiling reached on v2, with the best decision threshold back near 0.5.

## 6. Very large host buildings → cropset v3b (decided 21 Sep: build it)

A single point inside a huge building paints the whole building:

| Host buildings larger than | Count (of 1,204) | Share of all painted establishment area |
|---|---|---|
| 300 m² | 156 (13 %) | 63.5 % |
| 1,000 m² | 42 (3.5 %) | **43.3 %** |
| 2,000 m² | 21 (1.7 %) | 33.7 % |

The six largest: a 17,373 m² block painted by one POI named "Fvgvb" (a junk entry), "De Nieuwe Schatkamer" 13,371 m², the Rijksmuseum 9,768 m² (food & drink because of its café), "SPAR university" 5,856 m², Hotel Okura 5,078 m², Sporthal De Pijp 4,817 m². This also skews the splits: "other named" is 3.4 % of train pixels but 11.9 % of val pixels because of one such block.

Options:
- **(a) keep as is** (v3 as generated): faithful to the rule "this building houses X"; the Rijksmuseum *is* a building with a café.
- **(b) area cap**: in host buildings above a limit (1,000 m²) each establishment paints a **disc of that same area, clipped to the footprint**, instead of the whole building. A modelling rule, not data cleaning: the establishment stays in the data, only its drawn extent changes. Removes 43 % of the painted area while touching 3.5 % of the hosts. Would be cropset `v3b`, six minutes to generate.
- (c) paint only the part of the footprint within N metres of the point: more faithful for large buildings, more code.

Decision (Marvin, 21 Sep): **(b)**, built as cropset **`v3b`** (`ESTAB_MAX_HOST_M2 = {"v3b": 1000.0}` in `crops.py`, `max_host_m2` in `estab_footprints.estab_pairs`). v3 stays on disk; v2 and v3 were re-verified bit-identical after every change.

**First attempt, withdrawn the same day.** The first v3b let establishments in over-cap hosts fall back to the 2 m disc. Marvin's objection: a 999 m² building painted whole next to a 1,001 m² building shrunk to a 13 m² dot is a wild discontinuity in the data. Correct. That version was overwritten; nothing had been trained on it.

**Rule as built.** Each establishment inside an over-cap host paints a disc of **area = the cap** (radius √(1000/π) = 17.8 m) centred on its point and **clipped to the host footprint**. The painted area therefore grows with the building up to 1,000 m² and stays at that scale beyond it; nothing spills onto streets or neighbours; several establishments in one large host each keep their own patch and class (better than v3, where the whole block took the highest class).

**v3b result:** 42 hosts (3.5 %) are over the cap and contain 67 establishments. Their clipped patches measure 223–998 m², median 588 m² (below 1,000 because establishments sit at the façade, so part of the disc falls outside the building and is cut). No painted shape exceeds the cap. 1,162 buildings stay painted whole; 63 establishments outside any building keep the 2 m disc (unchanged from v3: an inherited small inconsistency, 4.5 % of establishments). Noise and surface identical to v2/v3, same crop ids. Figure: `data/amsterdam/de_pijp/overlays/crops_v2_v3_v3b_estab.png`.

| Majority-rule cell share, all 2,000 train crops | v3 | v3b | implied weight (1 − f)/f, v3 → v3b |
|---|---|---|---|
| other named | 3.41 % | 2.46 % | 28 → 40 |
| shop | 3.02 % | 2.51 % | 32 → 39 |
| food & drink | 2.06 % | 1.72 % | 48 → 57 |
| empty food & drink targets | 9.1 % | 9.8 % | |

Split skew: "other named" was 3.4 % of train vs 11.9 % of val pixels in v3; in v3b it is 2.4 % vs 3.1 %.

**Caveat on the loss weight.** Inverse-frequency weights on v3b are still 39–57. The threshold sweep showed that a weight of that size makes the model over-mark *by construction* (it moves the break-even probability to 1/(1+w) ≈ 2 %). What v3/v3b fix is the geometry: targets of several cells, half of them pure, no collisions. Recommendation: clamp the weight at about 10–20 (`c1_ce_weight_clamp`) and confirm with `c1_threshold_sweep.py` that the best decision threshold sits near 0.5.

## 7. Reproduce

```bash
.venv/bin/python -m worldsnap.crops --district de_pijp --cropset v3    # ~6 min, 240 MB
.venv/bin/python -m worldsnap.crops --district de_pijp --cropset v3b   # v3 + 1,000 m2 host cap
.venv/bin/python -m worldsnap.crops --district de_pijp --cropset v2    # unchanged output
```
