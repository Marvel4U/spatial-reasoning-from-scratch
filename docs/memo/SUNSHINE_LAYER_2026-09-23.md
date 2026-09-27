# Sunshine layer + cropset v4 — 23 Sep 2026

Implements C4_PLAN §4c (item E1) and the 7-level noise plane C3_PLAN §2 asks for.
**First derived layer of the project**: no provider publishes it, so its credibility
rests entirely on the definition below and on code short enough to read.
Owner: Claude. **The definition below was not pre-approved — Marvin reviews it.**

## 1. Definition

**Sunlit hours** = hours on a given **UTC calendar day** during which the centre of the
solar disc is geometrically above the horizon *and* the straight line from a ground
pixel to the sun is not blocked by a building within 200 m. Evaluated at **15-minute
steps, at each step's midpoint**; each unblocked step contributes 0.25 h.

Rasters (all 1 m, EPSG:28992, `data/amsterdam/de_pijp/`):

| file | contents | nodata |
|---|---|---|
| `sunlit_hours_0621_1m.tif` | float32 hours, 21 June 2026 | NaN = building interior |
| `sunlit_hours_0321_1m.tif` | float32 hours, 21 March 2026 | NaN |
| `shadow_0621_1500utc_1m.tif` | uint8 0/1, shadowed at 21 Jun 15:00 UTC | 255 |
| `overlays/sunlit_hours_0621.png` | overlay, sun colour map, buildings grey, scale bar | — |

The frame is **identical to `district_labels_1m`** in every cropset (same `floor`/`ceil`
origin, 1784 × 1793) — `crops.py` now imports `sunshine.district_grid` rather than
repeating the two lines, and `planes.sun_plane_1m` hard-errors on any frame mismatch.

## 2. Formula source

Solar position: *Astronomical Almanac* §C24 low-precision solar formulae (the series
behind the NOAA solar calculator; Meeus ch. 25 abridged), stated accurate to ~0.01° in
RA/dec for 1950–2050. GMST from the USNO linear expression. **Not modelled**:
refraction (~0.5° of apparent lift at the horizon), the disc's 0.27° radius, parallax,
UT1−UTC. Consequence: sunlit hours are under-estimated by a couple of minutes per day.

Checked in `tests/test_sun.py` (run with `-s` to print) against values this code did not
produce:

| | computed | published |
|---|---|---|
| 21 Jun max elevation | 61.09° @ 11:42 UTC, az 179.88° | 61.1° (= 90 − 52.35 + 23.44), due south |
| 21 Dec max elevation | 14.22° | 14.2° |
| 21 Mar max elevation | 38.00° | 38.0° |
| 21 Jun day length | 16.80 h | 16.78 h (almanac: 05:19–22:06 CEST) |

Plus a synthetic sign check (a 10 m column, sun at 45°): west sun → 9–10 m shadow to
the **east**; south sun → to the **north** (lower row index). A mirrored azimuth would
still produce a plausible-looking overlay, so this is asserted, not eyeballed.

## 3. The choices, and why

1. **Buildings only, no trees.** Trees are in the AHN surface model but as one season's
   noisy crowns with no foliage model; mixing them would make the layer unverifiable
   against 3DBAG. §4 quantifies what the omission costs. A tree variant is a separate
   layer, compared, never merged.
2. **Height = 3DBAG `height_m` = `b3_h_dak_70p − b3_h_maaiveld`** — the same canonical
   height the buildings layer and the crops already use. 0 of 9,775 footprints lack it;
   29 are non-positive and are **kept as-is** (no cleaning): they mark their pixels as
   not-ground and cast no shadow.
3. **2.5D flat extrusion.** A pitched roof shadows as a box at 70 % of its slope.
4. **Year 2026**, because a date is not a date without one (declination moves < 0.2°
   across the leap cycle).
5. **One sun position for the whole district** (bbox centre 52.353 N, 4.897 E); azimuth
   varies by ~0.02° over 1.8 km.
6. **200 m shadow reach**, 1 m steps. Low sun therefore loses its longest shadows and
   sunlit hours are slightly over-estimated near sunrise/sunset. The march also stops
   early once `d·tan(elevation)` exceeds the tallest building — that changes nothing.
7. **Zero-padded shifts, not `torch.roll`.** Nothing outside the raster casts a shadow,
   so a 200 m edge band is lit slightly too much. Wrapping would be worse and invisible.
8. **Building interiors are NaN, never 0.** A permanently dark courtyard and a roof
   pixel must not share a value.
9. **v4 sun plane: 5 values, 4 = "not ground (building)".** The plan's four classes
   (`<2 / 2–4 / 4–6 / >6 h`) describe ground; folding buildings into class 3 or class 0
   would both assert something false. The fifth value costs one one-hot plane and is
   *redundant* (the surface plane already marks buildings), so it can be ignored by any
   task generator that prefers to mask on `surface`. Class breaks are half-open
   `[lo, hi)`, so exactly 6.00 h is class 3 — labelled `>=6 h`, not `>6 h`.

## 4. Built-in check vs AHN (C4_PLAN §4c) — reported, never merged

A second shadow at 21 Jun 15:00 UTC (sun 42.79° elevation, 251.77° azimuth) from the
AHN5 nDSM (`dsm_05m − ` interpolated DTM ground, max over 2 × 2 cells to 1 m — max
because the AHN5 DSM is itself a per-cell highest point), compared over the 2,133,400
3DBAG ground pixels:

| | |
|---|---|
| shadow fraction, 3DBAG boxes | **0.2764** |
| shadow fraction, AHN nDSM (buildings **and** trees) | **0.7238** |
| **disagreement rate** | **0.4621** |
| AHN-shadow-only / 3DBAG-shadow-only | 0.4547 / 0.0073 |

The asymmetry is the whole story and it is explained, not mysterious: **41 % of the
pixels 3DBAG calls ground carry > 2 m of structure in the AHN surface, and 26 % carry
> 8 m** — street trees above all, plus annexes, balconies and dormers outside the BAG
footprint, plus the 2 × 2 max bleeding a roof one cell outward. 3DBAG shadows are
almost a subset of AHN shadows (0.7 % the other way), which is what a "buildings only"
definition should look like. 218,869 nDSM cells are NaN (the AHN5 DSM has no water by
construction) and were treated as 0 m obstacle; only 3,568 half-metre cells are outside
the AHN clip at all (one 0.5 m row at the south edge). **The two sources are never
averaged.** Read the number as: *this layer answers "what do the buildings shade?", and
if you want "what is actually in the sun?" you need the tree variant.*

## 5. Distributions over ground pixels (n = 2,133,400)

| date | p10 | p50 | p90 | mean | min | max |
|---|---|---|---|---|---|---|
| 21 June | 3.50 | **9.75** | 14.25 | 9.15 | 0.00 | 16.50 |
| 21 March | 0.00 | **4.50** | 10.50 | 5.08 | 0.00 | 12.00 |

**Flag for Marvin:** on 21 June the plan's class breaks are badly skewed — class 3
(`>=6 h`) holds **74 % of ground** (49.6 % of all pixels). If the sun plane is meant to
carry a task, **21 March is the better-balanced date** (p50 4.5 h sits inside class 2),
or the June breaks want moving to roughly 6 / 10 / 13 h. Both rasters are built, so
switching is a one-line change in `planes.SUN_SOURCE_RASTER` / `SUN_BREAKS`.

## 5b. Revision the same day: one plane, fixed scale, the date varies per crop (Marvin)

The first v4 classed 21 June at breaks 2 / 4 / 6 h, chosen before looking at the data; 74 % of June ground fell into the top class. Two fixes were proposed (use 21 March, or two planes for two dates). **Marvin's decision:** neither. One `sun` plane on **one fixed scale in hours** (breaks 2, 4, 6, 8, 10, 12; classes 0–6 for ground, 7 = building interior), and the **date is a property of the crop**: each crop draws 21 June or 21 March (seeded, recorded as `sun_date` in the crop json; the origin stream is untouched, so crop ids and ground stay identical to v3b). The model has to learn that some days have more sun overall, as it learns that some crops are noisier. More dates can be added the same way; if a task ever needs the date as information, it becomes a condition token.

Resulting v4 sun plane, all 2,000 train crops (1,000 per date): <2 h 11 %, 2–4 9 %, 4–6 8 %, 6–8 7 %, 8–10 10 %, 10–12 11 %, ≥12 11 %, building 33 %. Val/test are within a few points. v3b re-verified bit-identical after the change; regeneration 508 s.

## 6. Cropset v4

`v3b` + two npz-only planes. Same crop origins, ids, split and seed; **same
establishment rule**; the 2,500 RGB PNGs are **byte-identical to v3b's** — the image
contract (3 channels × 4 grey levels) cannot hold 7 noise levels and 5 sun classes, and
Track A does not use v4. The new planes live only in the `.npz`, which is what the GPU
loaders read.

npz keys: `{noise,surface,estab,noise7,sun}_{1024,256}`. The sun plane is computed at
1 m and **repeated 4×** for the 1024 px frame (the vector planes are still rasterised
independently at both resolutions); `legend.json` says so, and lists every plane, its
class count and every class label.

Class fractions (train split, over 1024-px pixels):

| plane | fractions |
|---|---|
| `noise` (4) | .5075 / .2335 / .2347 / .0242 — unchanged from v3b |
| `noise7` (7) | .3670 `<50` / .1406 `50-55` / .1101 / .1234 / .1307 / .1040 / .0242 `>=75` |
| `sun` (5) | .0279 `<2h` / .0612 `2-4h` / .0824 `4-6h` / **.4955** `>=6h` / .3330 not-ground |
| `surface` (4) | .3416 / .1695 / .1559 / .3330 |
| `estab` (4) | .9341 / .0242 / .0247 / .0170 |

Two consistency checks that passed: the 7-level fractions collapse exactly onto the
4-level ones under the documented merge, and `sun == 4` is pixel-identical to
`surface == 3` (two independent rasterisation paths, same footprints).

Disk: **333 MB** for v4 (v3b 247 MB); district stacks 684 KB (1 m) + 4.4 MB (0.25 m).

**v2/v3/v3b are not broken.** Regenerating `v3b` with the new code into a scratch root
reproduced all 2,500 `_labels.npz` and all 2,500 `_rgb.png` byte-identically, and both
`district_labels_{1m,025m}.npz` bit-identically (1 m sha256 `a9246916…e972eac`, as
stored). `split.json`, `index.jsonl` and the transforms JSON are identical;
`legend.json` gains three purely additive keys (`planes`, `n_classes`, `plane_notes`).

## 7. How to regenerate

```bash
cd ~/Github/spatial_reasoning_LLM_artifact
.venv/bin/python -m pytest tests/test_sun.py -s          # almanac + shadow-direction checks
.venv/bin/python -m worldsnap.build --district de_pijp --layers sunshine   # ~40 s, GPU
.venv/bin/python -m worldsnap.crops --district de_pijp --cropset v4 \
    --n-train 2000 --n-val 250 --n-test 250 --seed 0      # ~7 min, CPU
```

`sunshine` needs `buildings_3dbag.gpkg` (and `dtm_05m.tif` + `dsm_05m.tif` for the AHN
check, which is skipped with a note if they are absent). `crops --cropset v4` needs
`sunlit_hours_0621_1m.tif` and fails loudly with the build command if it is missing.
Runtime of the shadow march itself: **2.7 s on the RTX 3060** for both dates (67 + 49
sun positions × ≤ 200 shifted comparisons over 1784 × 1793); the layer's 40 s are
dominated by reading the AHN rasters and the DTM hole fill.

## 8. Files

| file | lines | |
|---|---|---|
| `worldsnap/sun.py` | 226 | new — pure solar geometry + ray march, no I/O |
| `worldsnap/layers/sunshine.py` | 252 | new — rasters, AHN check, stats |
| `worldsnap/planes.py` | 155 | new — v4 plane tables, sun classing, legend assembly |
| `tests/test_sun.py` | 92 | new — 5 tests, all passing |
| `worldsnap/build.py` | +129 | `build_sunshine` + registry entry `sunshine` |
| `worldsnap/crops.py` | 253 → 270 | v4 cropset; v2/v3/v3b output unchanged |
