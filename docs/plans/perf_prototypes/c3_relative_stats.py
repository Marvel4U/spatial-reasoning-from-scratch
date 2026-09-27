"""c3 candidate 'quietest sidewalk in this crop': is it well defined with 4 noise bands? with the source's native bands?
CPU only. Uses the v3b crop origins so the statistics describe the crops we train on."""
import json, sys
from pathlib import Path
import numpy as np
import geopandas as gpd

REPO = Path(__file__).resolve().parents[3] if "perf_prototypes" in __file__ else Path.home() / "Github/spatial_reasoning_LLM_artifact"
sys.path.insert(0, str(REPO))
from worldsnap import crops as C
from worldsnap.config import DISTRICTS

d = DISTRICTS["de_pijp"]; ddir = C.DATA_ROOT / d.city / d.name
noise = gpd.read_file(ddir / "noise_road_lden.gpkg")
col = "db_label" if "db_label" in noise.columns else [c for c in noise.columns if noise[c].astype(str).str.contains("dB").any()][0]
labels = sorted(noise[col].dropna().unique(), key=lambda s: float(str(s).replace(">", "").replace("=", "").split("-")[0].split()[0]))
print("native band labels, quiet -> loud:", labels)
native = {lab: i + 1 for i, lab in enumerate(labels)}                       # 0 = below the lowest mapped band
rows = noise.assign(_l=noise[col].map(native)).dropna(subset=["_l"]).sort_values("_l")
bx0, by0, bx1, by1 = d.bbox_rd
ox, oy = float(np.floor(bx0)), float(np.ceil(by1)); w, h = int(np.ceil(bx1) - ox), int(oy - np.floor(by0))
fine = C._district_stack({"n": list(zip(rows.geometry, rows["_l"].astype(int)))}, ox, oy, w, h, 1.0)["n"]
st = np.load(ddir / "crops/v3b/district_labels_1m.npz"); coarse, surf = st["noise"], st["surface"]
print("native levels incl. 0:", len(labels) + 1, "| pixel share per native level:", np.round(np.bincount(fine.ravel(), minlength=len(labels) + 1) / fine.size, 3).tolist())

idx = [json.loads(l) for l in open(ddir / "crops/v3b/index.jsonl")]
tr = [x for x in idx if x["split"] == "train"]


def stats(nz, name):
    ndist, share_of_sw, share_of_cells, second = [], [], [], []
    for x in tr:
        dx, dy = x["origin_m_from_district_nw"]; sl = (slice(dy, dy + 256), slice(dx, dx + 256))
        sw = surf[sl] == 2
        if sw.sum() < 50: continue
        vals = nz[sl][sw]; lv, cnt = np.unique(vals, return_counts=True)
        ndist.append(len(lv)); share_of_sw.append(cnt[0] / cnt.sum()); share_of_cells.append(cnt[0] / 65536)
        second.append((cnt[0] + cnt[1]) / cnt.sum() if len(lv) > 1 else 1.0)
    ndist, s1, sc = np.array(ndist), np.array(share_of_sw), np.array(share_of_cells)
    print(f"\n{name}: crops used {len(ndist)}")
    print("  distinct noise levels on sidewalk per crop: " + ", ".join(f"{k}: {100 * (ndist == k).mean():.0f}%" for k in range(1, ndist.max() + 1)))
    print(f"  crops where ALL sidewalk is in one level (relative task degenerate): {100 * (ndist == 1).mean():.1f}%")
    print("  'quietest level present' covers this share of the crop's sidewalk: p10 %.0f%%  p50 %.0f%%  p90 %.0f%%" % tuple(np.percentile(s1, [10, 50, 90]) * 100))
    print(f"  crops where the quietest level holds > 80% of the sidewalk (nearly trivial): {100 * (s1 > 0.8).mean():.1f}%")
    print("  target as share of all pixels: p10 %.1f%%  p50 %.1f%%  p90 %.1f%%" % tuple(np.percentile(sc, [10, 50, 90]) * 100))


stats(coarse, "4 bands (current input)")
stats(fine, f"{len(labels) + 1} native levels")


def relativity(nz, name):
    """Is the task really relative? Which level is the quietest / the median per crop, and how big are rank-based targets?"""
    mins, meds, below, atbelow = [], [], [], []
    for x in tr:
        dx, dy = x["origin_m_from_district_nw"]; sl = (slice(dy, dy + 256), slice(dx, dx + 256))
        sw = surf[sl] == 2
        if sw.sum() < 50: continue
        v = nz[sl][sw]; med = int(np.median(v))
        mins.append(int(v.min())); meds.append(med); below.append((v < med).mean()); atbelow.append((v <= med).mean())
    mins, meds, below, atbelow = map(np.array, (mins, meds, below, atbelow))
    top = int(nz.max())
    print(f"\n{name}")
    print("  quietest level present on sidewalk: " + ", ".join(f"L{k}: {100 * (mins == k).mean():.0f}%" for k in range(top + 1) if (mins == k).any()))
    print("  median level of the crop sidewalk:  " + ", ".join(f"L{k}: {100 * (meds == k).mean():.0f}%" for k in range(top + 1) if (meds == k).any()))
    print("  rule [strictly quieter than the crop median]: share of sidewalk p10 %.0f%% p50 %.0f%% p90 %.0f%% | empty in %.0f%% of crops"
          % (*np.percentile(below, [10, 50, 90]) * 100, 100 * (below == 0).mean()))
    print("  rule [at or below the crop median]:           share of sidewalk p10 %.0f%% p50 %.0f%% p90 %.0f%%" % tuple(np.percentile(atbelow, [10, 50, 90]) * 100))


relativity(coarse, "4 bands")
relativity(fine, "7 native levels")


def local_baseline(nz, name):
    """How well can a model do WITHOUT looking at the rest of the crop? Best fixed absolute rule (sidewalk and level < t)
    scored against the relative target (sidewalk strictly quieter than the crop median); pixel level, per-crop IoU."""
    top = int(nz.max()); ious = {t: [] for t in range(1, top + 1)}; n_empty = 0
    for x in tr:
        dx, dy = x["origin_m_from_district_nw"]; sl = (slice(dy, dy + 256), slice(dx, dx + 256))
        sw = surf[sl] == 2
        if sw.sum() < 50:
            continue
        v = nz[sl]; med = int(np.median(v[sw])); tgt = sw & (v < med)
        if not tgt.any():
            n_empty += 1
            continue
        for t in ious:
            pr = sw & (v < t); ious[t].append((pr & tgt).sum() / (pr | tgt).sum())
    print(f"{name}: local-only baseline vs relative target (non-empty crops: {len(ious[1])}, empty: {n_empty})")
    print("  mean IoU of fixed rule [level < t]: " + ", ".join(f"t={t}: {np.mean(v):.2f}" for t, v in ious.items()))


local_baseline(coarse, "4 bands")
local_baseline(fine, "7 native levels")
