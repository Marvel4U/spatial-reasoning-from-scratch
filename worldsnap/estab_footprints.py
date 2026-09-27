"""Establishment geometry for the label crops: discs (v2) or hosting-building footprints (v3, v3b).

Why footprints (docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md): a 2 m disc is 13 m2, smaller than one
4 x 4 m output cell, so almost every establishment target cell was only partly covered, the class
had to be read from a ~4 px dot, and the extreme rarity forced a loss weight of 100 that made the
model over-mark. Larger discs collide on shopping streets (50 % overlap at 4 m). In De Pijp 95.5 %
of named establishments lie inside a 3DBAG building footprint and 91 % of hosting buildings host
exactly one, so painting the hosting building is larger (~96 m2, ~6 cells), collision-free and
closer to the meaning "this building houses a cafe".

Rules (nothing is cleaned): a building takes the HIGHEST class among the establishments inside it;
an establishment outside every footprint (market stalls, kiosks) keeps a small disc; discs are
painted after footprints, lower class first, so the higher class wins wherever things overlap.

Area cap (cropset v3b): one point inside a very large building would paint all of it (42 of 1,204
hosts are > 1,000 m2 and carry 43 % of the painted area; the largest, 17,000 m2, comes from one
junk POI). With ``max_host_m2`` set, each establishment in a host above the cap paints a disc of
AREA ``max_host_m2`` (radius sqrt(cap / pi) = 17.8 m for 1,000 m2) centred on its point and CLIPPED
to the host footprint. So the painted area grows with the building up to the cap and stays at that
scale beyond it: no jump from "whole building" to "tiny dot" at the threshold, and nothing spills
onto streets or neighbouring houses. A drawing rule, not cleaning.
"""

from __future__ import annotations

import math

import geopandas as gpd

GEOMETRY_MODES = ("disc", "footprint")


def estab_pairs(rows: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, mode: str,
                radius_m: float, max_host_m2: float | None = None) -> tuple[list, dict]:
    """``rows``: named establishments with int-able class column ``_c``. Returns the
    ``(geometry, class)`` pairs in paint order and a small stats dict for the legend."""
    if mode not in GEOMETRY_MODES:
        raise ValueError(f"estab geometry mode {mode!r} not in {GEOMETRY_MODES}")
    rows = rows.assign(_c=rows["_c"].astype(int)).sort_values("_c")  # higher class painted last
    if mode == "disc":
        return list(zip(rows.geometry.buffer(radius_m), rows["_c"])), {
            "mode": mode, "n_establishments": len(rows), "radius_m": radius_m}

    b = buildings[["geometry"]].reset_index(drop=True)
    hit = gpd.sjoin(rows[["_c", "geometry"]], b, how="left", predicate="within")
    inside = hit.dropna(subset=["index_right"])
    capped = inside.iloc[0:0]
    if max_host_m2 is not None:
        too_big = b.geometry.area.iloc[inside["index_right"].astype(int)].to_numpy() > max_host_m2
        capped, inside = inside[too_big], inside[~too_big]
    per_building = inside.groupby("index_right")["_c"]
    host_class = per_building.max().astype(int).sort_values()           # building -> winning class
    outside = rows.loc[hit.index[hit["index_right"].isna()].unique()]

    pairs = [(b.geometry.iloc[int(i)], int(c)) for i, c in host_class.items()]
    patches = list(zip(outside.geometry.buffer(radius_m), outside["_c"]))
    if len(capped):
        r_cap = math.sqrt(max_host_m2 / math.pi)
        patches += [(pt.buffer(r_cap).intersection(b.geometry.iloc[int(i)]), int(c))
                    for pt, i, c in zip(capped.geometry, capped["index_right"], capped["_c"])]
    pairs += sorted(patches, key=lambda gc: gc[1])                      # lower class first
    cap_areas = sorted(g.area for g, _ in patches[len(outside):])
    stats = {"mode": mode, "n_establishments": len(rows), "max_host_m2": max_host_m2,
             "n_inside_building": int(inside.index.nunique()) + int(capped.index.nunique()),
             "n_outside_building_as_disc": len(outside), "disc_radius_m": radius_m,
             "n_host_buildings": int(len(host_class)),
             "n_host_buildings_multi_establishment": int((per_building.size() > 1).sum()),
             "n_host_buildings_mixed_class": int((per_building.nunique() > 1).sum()),
             "host_footprint_m2_median": float(b.geometry.iloc[host_class.index.astype(int)].area.median()),
             "n_host_buildings_over_cap": int(capped["index_right"].nunique()),
             "n_establishments_in_capped_hosts": len(capped),
             "capped_patch_m2_min_median_max": [round(cap_areas[0]), round(cap_areas[len(cap_areas) // 2]),
                                                round(cap_areas[-1])] if cap_areas else None}
    return pairs, stats
