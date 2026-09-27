"""Label planes that exist only in cropset v4: 7-level noise and the sun plane.

WHY a separate module
---------------------
``crops.py`` owns the crop geometry, the split and the RGB image contract, and
all three of those must stay byte-for-byte what v2/v3/v3b produced. Putting the
new planes here keeps that file short and makes the diff that adds v4 readable:
nothing below changes an existing cropset, and ``CROPSET_PLANES`` is the single
place that says which cropset carries which planes.

WHY the RGB PNG does not change for v4
--------------------------------------
The crop image is a three-channel PNG whose contract is fixed and written into
``LEGEND_TEXT``: one layer per colour channel, four grey levels 0/85/170/255.
Seven noise levels and five sun classes do not fit that contract, and v4 has
five planes for three channels anyway. So the v4 PNG is *identical* to the v3b
PNG (same three layers, same four-level noise), and the new planes live only in
the ``.npz`` label stacks, which is where the GPU loaders read from. Track A
(the image models) is unaffected by v4 by construction; Track B (c3's relative
noise task, c4's sunny path) reads the planes it needs from the npz.

PLANE 1 - noise at the source's seven native levels (C3_PLAN section 2)
-----------------------------------------------------------------------
The Amsterdam road-noise source publishes six Lden bands plus "no polygon".
v2/v3/v3b merge them into four, which collapses 50-55 dB into "no polygon" and
pairs the bands two by two. c3 measured that the merge is what makes the
"quieter than the median of this crop" task collapse into an absolute task (the
quietest band present is the lowest band of the scale in 98 % of crops at four
levels, 92 % at seven), so v4 carries the native levels as an *extra* plane.
The four-level plane stays exactly as it was, in the same npz, so a v4 crop can
reproduce a v3b answer.

PLANE 2 - the sun plane (C4_PLAN section 4c)
--------------------------------------------
Sunlit hours on 21 June, classed into the four levels the plan proposes. The
question the plan leaves open is what to do with building interiors, which have
no ground and therefore no sunlit hours. Three options were on the table:
folding them into class 3 (">= 6 h") is plainly wrong - a roof is not a sunny
courtyard; folding them into class 0 is wrong the same way and additionally
makes "dark" the most common class for a reason that has nothing to do with sun;
so v4 uses a **fifth value, 4 = "not ground (building)"**. It is the simplest
option that states what is true, it costs one one-hot plane, and it is
*redundant* rather than load-bearing: the surface plane already marks buildings
as class 3, so a task generator can equally well mask on that. A model trained
on this plane can never confuse "no sun" with "no ground".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

#: Layers drawn into the RGB crop image, in channel order. Fixed forever.
RGB_LAYERS = ("noise", "surface", "estab")
#: Which planes each cropset stores in its npz files. v2/v3/v3b: the RGB three.
CROPSET_PLANES: dict[str, tuple[str, ...]] = {
    "v2": RGB_LAYERS, "v3": RGB_LAYERS, "v3b": RGB_LAYERS,
    "v4": RGB_LAYERS + ("noise7", "sun"),
}
#: Number of classes per plane; drives the one-hot width and the bincounts.
N_CLASSES = {"noise": 4, "surface": 4, "estab": 4, "noise7": 7, "sun": 8}

#: The source's own Lden bands. 0 is "no polygon", which the noise layer
#: documents as "below 50 dB Lden" (the map simply does not draw quieter areas).
NOISE7_MEANING = {0: "<50 dB (no polygon)", 1: "50-55 dB", 2: "55-60 dB", 3: "60-65 dB",
                  4: "65-70 dB", 5: "70-75 dB", 6: ">=75 dB"}
NOISE7_CLASS = {"50-55 dB": 1, "55-60 dB": 2, "60-65 dB": 3,
                "65-70 dB": 4, "70-75 dB": 5, "75 dB and above": 6}

#: Sunlit-hours classes on ONE fixed scale in hours, whatever the date. Half-open
#: intervals [lo, hi): a pixel with exactly 6.00 h is class 3. Hours come in
#: 0.25 h steps, so only the exact break values are affected.
#: Decision (Marvin, 23 Sep 2026): the date is a property of the CROP, not of the
#: plane. Each crop draws one date from SUN_DATES (seeded); the model has to learn
#: that some days have more sun overall, as it learns that some crops are noisier.
#: The first breaks (2, 4, 6) were set before looking at the data and put 74 % of
#: June ground into one class; 2..12 h covers the June range (p50 9.75 h, max 16.8).
SUN_BREAKS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
SUN_NOT_GROUND = 7
SUN_MEANING = {0: "<2 h", 1: "2-4 h", 2: "4-6 h", 3: "6-8 h", 4: "8-10 h", 5: "10-12 h",
               6: ">=12 h", 7: "not ground (building interior)"}
#: One raster per available date; a crop uses exactly one of them (see crops.py).
SUN_DATES = {"0621": "sunlit_hours_0621_1m.tif", "0321": "sunlit_hours_0321_1m.tif"}
SUN_SOURCE_RASTER = SUN_DATES["0621"]  # kept for the layer overlay / single-date callers

MEANINGS_EXTRA = {"noise7": NOISE7_MEANING, "sun": SUN_MEANING}

#: One sentence per plane for ``legend.json``: what it is and whether it is
#: visible in the RGB image. A reader of the legend alone must be able to tell
#: that two of the five v4 planes exist only in the npz.
PLANE_NOTES = {
    "noise": "road-traffic Lden merged into 4 levels; drawn in the RGB image (R)",
    "surface": "BGT functie; drawn in the RGB image (G)",
    "estab": "named OSM establishments; drawn in the RGB image (B)",
    "noise7": "road-traffic Lden at the source's own 7 native levels; npz only, NOT in the "
              "RGB image, whose contract is 3 channels x 4 grey levels",
    "sun": f"sunlit hours at ground level, classed on one fixed scale {list(SUN_BREAKS)} h, "
           f"half-open [lo, hi); the DATE varies per crop (one of {sorted(SUN_DATES)}, recorded as "
           f"sun_date in the crop json); value {SUN_NOT_GROUND} = not ground "
           f"(building interior, nodata in the raster); computed at 1 m and repeated 4x for "
           f"the 1024 px frame; npz only",
}


def legend_json(planes: tuple[str, ...], meanings: dict, est_stats: dict, *,
                base: dict, source_mapping: dict, legend_text: str) -> dict:
    """Assemble ``legend.json``: every plane, its class count and every class label."""
    return {**base,
            "planes": list(planes),
            "n_classes": {k: N_CLASSES[k] for k in planes},
            "classes": {k: {str(i): v for i, v in meanings[k].items()} for k in planes},
            "plane_notes": {k: PLANE_NOTES[k] for k in planes},
            "source_mapping": {**source_mapping, "estab_geometry": est_stats,
                               **({"noise7": NOISE7_CLASS} if "noise7" in planes else {})},
            "legend_text": legend_text}


def noise7_pairs(noise) -> list:
    """``(geometry, level)`` pairs for the 7-level plane, quietest painted first.

    Same paint order as the 4-level plane (``crops._noise_pairs``): ascending
    level, so where two bands overlap the louder one wins. Rows whose label is
    not a road Lden band (the industry zoning outlines) are dropped by the map,
    exactly as they are for the 4-level plane.
    """
    rows = noise.assign(_c=noise["db_label"].map(NOISE7_CLASS)).dropna(subset=["_c"])
    rows = rows.sort_values("_c")
    return list(zip(rows.geometry, rows["_c"].astype(int)))


def sun_plane_1m(path: Path, transform, shape: tuple[int, int]) -> np.ndarray:
    """Read one sunlit-hours raster and class it into the sun plane (one date).

    The assertion is the whole point of the function: the sun raster and the
    label stack must be the *same pixels*, so a mismatch of transform or shape
    is a hard error rather than a silent half-pixel shift that would only ever
    show up as a model that cannot learn the plane.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; build it first with "
            f"`python -m worldsnap.build --district <d> --layers sunshine`")
    with rasterio.open(path) as ds:
        if ds.shape != shape or tuple(ds.transform)[:6] != tuple(transform)[:6]:
            raise ValueError(f"sun raster frame {ds.shape} {tuple(ds.transform)[:6]} != "
                             f"label frame {shape} {tuple(transform)[:6]}")
        hours = ds.read(1)
    plane = np.digitize(hours, SUN_BREAKS, right=False).astype(np.uint8)
    return np.where(np.isnan(hours), SUN_NOT_GROUND, plane).astype(np.uint8)


def upsample(plane: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour block repeat, for planes that exist only at 1 m.

    The vector planes are rasterised independently at 0.25 m and 1 m so that
    thin sidewalks obey ``all_touched=False`` in both frames (see ``crops``).
    The sun plane cannot be: it is computed from a 1 m shadow march, so its
    0.25 m version is an honest 4x repeat and is documented as such in the
    legend. Track A does not use v4, so nothing depends on it being finer.
    """
    return np.repeat(np.repeat(plane, factor, axis=0), factor, axis=1)
