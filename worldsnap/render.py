"""Deterministic PNG overlays of a district layer.

WHY render at all, this early
-----------------------------
Two reasons, both about trust rather than beauty. (1) A map is the fastest
possible QA instrument for a geo pipeline: a wrong CRS, a botched CityJSON
de-quantisation or a half-empty bbox is invisible in a dataframe and screaming
in a picture. (2) The eventual training data *is* rendered maps, so the render
path has to exist and be pixel-deterministic from day one.

Pixel determinism: the figure size is derived from the RD bbox aspect ratio at a
fixed output width, the axes are set to exactly the bbox with ``aspect="equal"``,
and RD metres are linear, so the pixel->metre mapping is exact and constant.
That is what later lets a verifier convert a model's pixel click into RD
coordinates without guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # headless: the rig has no display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box as _shp_box

#: Stop-marker AREA (pt^2) = _SIZE_BASE + _SIZE_PER_DEP * departures, so the
#: area -- not the radius -- is proportional to the count, which is the
#: non-exaggerating encoding for a magnitude.
_SIZE_BASE, _SIZE_PER_DEP = 12.0, 0.40

#: Fixed output width. Height follows from the district's aspect ratio.
DEFAULT_WIDTH_PX = 1600
DPI = 100


def _scale_bar(
    ax,
    bbox: tuple[float, float, float, float],
    length_m: float = 100.0,
    color: str = "black",
) -> None:
    """Draw a ``length_m`` bar in the lower-left corner.

    Trivial in RD because the axes are in metres -- which is precisely the
    property we want to make visible to a reader of the PNG. ``color`` exists
    because a black bar is invisible on a dark raster overlay, and a scale bar
    a reader cannot see is worse than none: it implies the map was scaled.
    """
    x0, y0, x1, y1 = bbox
    pad_x = 0.04 * (x1 - x0)
    pad_y = 0.05 * (y1 - y0)
    bx, by = x0 + pad_x, y0 + pad_y
    ax.add_line(Line2D([bx, bx + length_m], [by, by], color=color, lw=3, solid_capstyle="butt"))
    for xx in (bx, bx + length_m):  # end ticks
        ax.add_line(Line2D([xx, xx], [by, by + 0.008 * (y1 - y0)], color=color, lw=3))
    ax.text(
        bx + length_m / 2,
        by + 0.012 * (y1 - y0),
        f"{length_m:.0f} m",
        ha="center",
        va="bottom",
        fontsize=11,
        color=color,
    )


def render_buildings(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    *,
    column: str,
    title: str,
    cbar_label: str,
    bbox: tuple[float, float, float, float],
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Fill building footprints by ``column`` and save a PNG.

    ``bbox`` is the district bbox in EPSG:28992 and fixes the extent, so all
    overlays of one district are pixel-registered with each other.
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)
    fig_w = width_px / DPI
    # +12% height for the title strip and the horizontal colorbar below.
    fig = plt.figure(figsize=(fig_w, fig_w * aspect * 1.12), dpi=DPI)
    ax = fig.add_axes([0.02, 0.09, 0.96, 0.85])

    plot = gdf.dropna(subset=[column])
    plot.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.15,
        edgecolor="#33333366",
        legend=False,
    )
    # Features with no value at all: drawn hollow so gaps are visibly *data*
    # gaps and not rendering failures.
    missing = gdf[gdf[column].isna()]
    if len(missing):
        missing.plot(ax=ax, facecolor="none", edgecolor="#cc0000", linewidth=0.4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=vmin if vmin is not None else float(plot[column].min()),
            vmax=vmax if vmax is not None else float(plot[column].max()),
        ),
    )
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.018])
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label(cbar_label, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def categorical_palette(
    values: list[str], preferred: dict[str, str] | None = None
) -> dict[str, str]:
    """Stable value -> colour map.

    Values named in ``preferred`` keep their semantic colour; anything else --
    including values we did not anticipate -- is assigned from ``tab20`` in
    sorted order, so an unexpected category shows up on the map with a distinct
    colour instead of being silently dropped or merged.
    """
    preferred = preferred or {}
    out = {v: preferred[v] for v in values if v in preferred}
    rest = sorted(v for v in values if v not in preferred)
    cmap = plt.get_cmap("tab20")
    for i, v in enumerate(rest):
        out[v] = matplotlib.colors.to_hex(cmap(i % 20))
    return out


def render_categorical(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    *,
    column: str,
    title: str,
    bbox: tuple[float, float, float, float],
    palette: dict[str, str] | None = None,
    context: gpd.GeoDataFrame | None = None,
    context_label: str = "buildings (context)",
    legend_ncol: int = 4,
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Fill polygons by a *categorical* ``column`` and save a PNG with a legend.

    Same extent contract as :func:`render_buildings` -- the axes are set to the
    district bbox in EPSG:28992 -- so this overlay is pixel-registered with the
    building overlays of the same district.

    ``context`` (e.g. building footprints) is drawn underneath in light grey:
    street polygons alone are an unreadable ribbon soup, and the eye needs the
    blocks to judge whether sidewalks really trace both sides of every street.
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)

    counts = gdf[column].fillna("<null>").value_counts()
    cats = list(counts.index)
    colours = palette or categorical_palette(cats)
    legend_rows = -(-(len(cats) + (1 if context is not None else 0)) // legend_ncol)

    fig_w = width_px / DPI
    # title strip + one legend row per wrapped line of the legend.
    fig = plt.figure(figsize=(fig_w, fig_w * aspect + 0.30 + 0.26 * legend_rows), dpi=DPI)
    bottom = (0.30 + 0.26 * legend_rows) / (fig_w * aspect + 0.30 + 0.26 * legend_rows)
    ax = fig.add_axes([0.02, bottom * 0.75, 0.96, 1 - bottom * 0.75 - 0.06])

    if context is not None and len(context):
        context.plot(ax=ax, facecolor="#e4e4e4", edgecolor="#cfcfcf", linewidth=0.2, zorder=1)

    for cat in cats:
        sel = gdf[gdf[column].fillna("<null>") == cat]
        sel.plot(
            ax=ax,
            facecolor=colours.get(cat, "#999999"),
            edgecolor="#33333355",
            linewidth=0.1,
            zorder=2,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    handles = [
        Patch(facecolor=colours.get(c, "#999999"), edgecolor="#333333", linewidth=0.3,
              label=f"{c}  (n={counts[c]})")
        for c in cats
    ]
    if context is not None and len(context):
        handles.append(
            Patch(facecolor="#e4e4e4", edgecolor="#cfcfcf", label=f"{context_label}  (n={len(context)})")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=legend_ncol,
        fontsize=9,
        frameon=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def render_transit(
    stops: gpd.GeoDataFrame,
    routes: gpd.GeoDataFrame,
    out_path: Path,
    *,
    title: str,
    bbox: tuple[float, float, float, float],
    type_colors: dict[int, str],
    type_labels: dict[int, str],
    buildings: gpd.GeoDataFrame | None = None,
    carriageways: gpd.GeoDataFrame | None = None,
    size_column: str = "departures_0719",
    n_labels: int = 10,
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Route lines coloured by ``route_type``, stops sized by departure count.

    Same extent contract as the other renderers -- axes set to the district
    bbox in EPSG:28992 -- so this overlay is pixel-registered with the building
    and street overlays. Note that stops and route geometry extend past the
    axes because the layer is cut to bbox+buffer, not to the bbox: the map
    shows the district, the data covers the walking catchment around it.

    Marker **area** is proportional to ``size_column``, which is the honest
    encoding for a count (radius-proportional would exaggerate ~quadratically).
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)
    fig_w = width_px / DPI
    fig = plt.figure(figsize=(fig_w, fig_w * aspect + 1.05), dpi=DPI)
    bottom = 0.75 / (fig_w * aspect + 1.05)
    ax = fig.add_axes([0.02, bottom, 0.96, 1 - bottom - 0.07])

    if buildings is not None and len(buildings):
        buildings.plot(ax=ax, facecolor="#ececec", edgecolor="#dcdcdc", linewidth=0.2, zorder=1)
    if carriageways is not None and len(carriageways):
        carriageways.plot(ax=ax, facecolor="#cfcfcf", edgecolor="none", zorder=2)

    # Draw bus first, rail/metro/tram on top: the frequent modes must not be
    # buried under the dense bus network.
    draw_order = sorted(
        routes["route_type"].dropna().unique(), key=lambda t: {3: 0, 2: 1, 4: 1, 0: 2, 1: 3}.get(int(t), 0)
    )
    for rtype in draw_order:
        sel = routes[routes["route_type"] == rtype]
        sel.plot(
            ax=ax,
            color=type_colors.get(int(rtype), "#999999"),
            linewidth=2.2 if int(rtype) in (0, 1) else 1.2,
            alpha=0.85,
            zorder=3,
        )

    sizes = _SIZE_BASE + _SIZE_PER_DEP * stops[size_column].astype(float)
    ax.scatter(
        stops.geometry.x, stops.geometry.y,
        s=sizes, facecolor="#ffffff", edgecolor="#111111", linewidth=0.8, zorder=4,
    )

    # Label the busiest stops that are actually *in frame*: the layer extends
    # into the buffer ring, so a global top-10 would spend most of its labels
    # on stops the reader cannot see. Deduplicated by name, because one busy
    # stop is two or three quays and three identical labels is noise.
    visible = stops[
        stops.geometry.x.between(x0, x1) & stops.geometry.y.between(y0, y1)
    ]
    busiest = (
        visible.sort_values(size_column, ascending=False)
        .drop_duplicates(subset="name")
        .head(n_labels)
    )
    for i, (_, row) in enumerate(busiest.iterrows()):
        dy = 8 if i % 2 == 0 else -14
        ax.annotate(
            f"{row['name']}  {int(row[size_column])}",
            (row.geometry.x, row.geometry.y),
            xytext=(8, dy), textcoords="offset points",
            fontsize=8, zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffffcc", edgecolor="none"),
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    handles: list[Any] = []
    for rtype in sorted(routes["route_type"].dropna().unique()):
        n = int((routes["route_type"] == rtype).sum())
        handles.append(Line2D(
            [], [], color=type_colors.get(int(rtype), "#999999"), lw=2.4,
            label=f"{type_labels.get(int(rtype), rtype)}  (n={n} routes)",
        ))
    # Size key: three reference counts spanning the observed range.
    obs_max = int(stops[size_column].max()) if len(stops) else 0
    for ref in sorted({0, max(1, obs_max // 4), obs_max}):
        handles.append(Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="#ffffff",
            markeredgecolor="#111111",
            markersize=((_SIZE_BASE + _SIZE_PER_DEP * ref) ** 0.5),
            label=f"{ref} departures 07-19",
        ))
    if buildings is not None and len(buildings):
        handles.append(Patch(facecolor="#ececec", edgecolor="#dcdcdc",
                             label=f"buildings (context, n={len(buildings)})"))
    if carriageways is not None and len(carriageways):
        handles.append(Patch(facecolor="#cfcfcf", edgecolor="none",
                             label=f"BGT carriageway (context, n={len(carriageways)})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=4, fontsize=9, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def render_points(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    *,
    column: str,
    title: str,
    bbox: tuple[float, float, float, float],
    palette: dict[str, str],
    category_order: list[str] | None = None,
    buildings: gpd.GeoDataFrame | None = None,
    streets: gpd.GeoDataFrame | None = None,
    markersize: float = 7.0,
    legend_ncol: int = 4,
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Point features coloured by a categorical ``column``, over map context.

    ``category_order`` fixes the legend order for *ordered* categories (a
    height class ramp is meaningless sorted by frequency); values outside it
    are appended, so an unexpected class is visible rather than silently
    merged. Same bbox/extent contract as the other renderers.
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)

    counts = gdf[column].fillna("<null>").value_counts()
    cats = list(category_order or []) + [c for c in counts.index if c not in (category_order or [])]
    cats = [c for c in cats if c in counts.index]
    legend_rows = -(-(len(cats) + 2) // legend_ncol)

    fig_w = width_px / DPI
    fig_h = fig_w * aspect + 0.30 + 0.26 * legend_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    bottom = (0.30 + 0.26 * legend_rows) / fig_h
    ax = fig.add_axes([0.02, bottom * 0.75, 0.96, 1 - bottom * 0.75 - 0.06])

    if buildings is not None and len(buildings):
        buildings.plot(ax=ax, facecolor="#ececec", edgecolor="#dcdcdc", linewidth=0.2, zorder=1)
    if streets is not None and len(streets):
        streets.plot(ax=ax, facecolor="#f4f0e8", edgecolor="#e2dccf", linewidth=0.1, zorder=2)

    for cat in cats:
        sel = gdf[gdf[column].fillna("<null>") == cat]
        ax.scatter(
            sel.geometry.x, sel.geometry.y,
            s=markersize, facecolor=palette.get(cat, "#999999"),
            edgecolor="#1a1a1a", linewidth=0.25, zorder=3,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    handles: list[Any] = [
        Line2D([], [], linestyle="none", marker="o", markersize=6,
               markerfacecolor=palette.get(c, "#999999"), markeredgecolor="#1a1a1a",
               label=f"{c}  (n={counts[c]})")
        for c in cats
    ]
    if buildings is not None and len(buildings):
        handles.append(Patch(facecolor="#ececec", edgecolor="#dcdcdc",
                             label=f"buildings (context, n={len(buildings)})"))
    if streets is not None and len(streets):
        handles.append(Patch(facecolor="#f4f0e8", edgecolor="#e2dccf",
                             label=f"BGT street surfaces (context, n={len(streets)})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=legend_ncol, fontsize=9, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def render_noise(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    *,
    title: str,
    bbox: tuple[float, float, float, float],
    band_order: list[str],
    palette: dict[str, str],
    buildings: gpd.GeoDataFrame | None = None,
    legend_ncol: int = 4,
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Isophone band polygons under a light-grey building mask.

    Two departures from :func:`render_categorical`, both forced by what a noise
    map *is*. (1) Bands are drawn quietest-first, so that wherever two of them
    do touch the louder one wins the pixel; the Amsterdam bands turn out to be
    essentially disjoint, but a painter's algorithm that can only ever err
    towards "louder" is the safe way to render a hazard. Frequency order, which
    :func:`render_categorical` uses, would let an arbitrary polygon count
    decide which hotspot is visible. (2) Buildings are drawn **over** the noise
    rather than under it: the interesting signal is the noise in the street
    canyons and courtyards, and an opaque building mask is what makes the
    street network legible as the thing being coloured.

    The legend reports **area inside the district bbox** per band rather than a
    feature count, because polygon count is an artefact of how the contouring
    happened to split the rings and carries no physical meaning.

    Same extent contract as every other renderer: the axes are the district
    bbox in EPSG:28992, so this overlay is pixel-registered with the others.
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)
    bbox_area = (x1 - x0) * (y1 - y0)

    present = [b for b in band_order if (gdf["db_label"] == b).any()]
    present += sorted(set(gdf["db_label"].dropna()) - set(present))  # unexpected labels
    # Area clipped to the bbox: the stored polygons stick out of the district,
    # and an area number that counted the overhang would not be about De Pijp.
    clipped = gdf.clip(_shp_box(x0, y0, x1, y1))
    area_by_band = clipped.groupby("db_label").geometry.apply(lambda s: float(s.area.sum()))

    legend_rows = -(-(len(present) + (1 if buildings is not None else 0)) // legend_ncol)
    fig_w = width_px / DPI
    fig_h = fig_w * aspect + 0.30 + 0.26 * legend_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    bottom = (0.30 + 0.26 * legend_rows) / fig_h
    ax = fig.add_axes([0.02, bottom * 0.75, 0.96, 1 - bottom * 0.75 - 0.06])

    for z, band in enumerate(present, start=1):
        sel = gdf[gdf["db_label"] == band]
        sel.plot(ax=ax, facecolor=palette.get(band, "#999999"), edgecolor="none", zorder=z)
    if buildings is not None and len(buildings):
        buildings.plot(
            ax=ax, facecolor="#d8d8d8", edgecolor="#b4b4b4", linewidth=0.2,
            zorder=len(present) + 2,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    handles: list[Any] = [
        Patch(
            facecolor=palette.get(b, "#999999"), edgecolor="#333333", linewidth=0.3,
            label=f"{b}  ({area_by_band.get(b, 0.0) / bbox_area * 100:.1f}% of bbox)",
        )
        for b in present
    ]
    if buildings is not None and len(buildings):
        handles.append(
            Patch(facecolor="#d8d8d8", edgecolor="#b4b4b4",
                  label=f"buildings (mask, n={len(buildings)})")
        )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=legend_ncol, fontsize=9, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def render_raster(
    array,
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
    bbox: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    hillshade=None,
    colour_alpha: float = 0.62,
    outlines: gpd.GeoDataFrame | None = None,
    outline_label: str | None = None,
    outline_color: str = "#101010",
    scalebar_color: str = "black",
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """A masked float raster over an optional grey hillshade base.

    ``array`` is a ``numpy.ma.MaskedArray`` in north-up row order whose extent
    is exactly ``bbox`` in EPSG:28992 -- the same extent contract as the vector
    renderers, so an elevation overlay is pixel-registered with the building
    overlays of the same district.

    Masked cells (nodata) are left as the figure's white background, which is
    the honest rendering: a nodata hole must not be mistakable for a height.

    Amsterdam is flat, so the caller is expected to pass a percentile-based
    ``vmin``/``vmax``: a 0-to-max ramp over a district whose terrain spans ~2 m
    renders as one flat colour and shows nothing.
    """
    import numpy as np

    # Masked cells still carry the provider's nodata sentinel (float32 max) in
    # the underlying buffer, and matplotlib's normaliser multiplies it before it
    # checks the mask -- which overflows loudly. NaN is the value matplotlib
    # actually treats as "draw nothing", so convert once, here, at the edge.
    array = np.ma.filled(np.ma.masked_invalid(array), np.nan)
    if hillshade is not None:
        hillshade = np.ma.filled(np.ma.masked_invalid(hillshade), np.nan)

    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)
    fig_w = width_px / DPI
    fig = plt.figure(figsize=(fig_w, fig_w * aspect * 1.12), dpi=DPI)
    ax = fig.add_axes([0.02, 0.09, 0.96, 0.85])
    extent = (x0, x1, y0, y1)

    if hillshade is not None:
        ax.imshow(hillshade, cmap="gray", extent=extent, origin="upper",
                  vmin=0.0, vmax=1.0, interpolation="nearest", zorder=1)
    ax.imshow(array, cmap=cmap, extent=extent, origin="upper", vmin=vmin, vmax=vmax,
              alpha=colour_alpha if hillshade is not None else 1.0,
              interpolation="nearest", zorder=2)
    if outlines is not None and len(outlines):
        outlines.plot(ax=ax, facecolor="none", edgecolor=outline_color, linewidth=0.35,
                      alpha=0.75, zorder=3)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox, color=scalebar_color)
    ax.set_title(title, fontsize=13, pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.018])
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label(cbar_label, fontsize=10)
    if outlines is not None and len(outlines) and outline_label:
        fig.legend(
            handles=[Patch(facecolor="none", edgecolor=outline_color, label=outline_label)],
            loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=9, frameon=False,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def render_pois(
    osm: gpd.GeoDataFrame,
    overture: gpd.GeoDataFrame,
    out_path: Path,
    *,
    column: str,
    title: str,
    bbox: tuple[float, float, float, float],
    palette: dict[str, str],
    category_order: list[str] | None = None,
    buildings: gpd.GeoDataFrame | None = None,
    streets: gpd.GeoDataFrame | None = None,
    markersize: float = 9.0,
    overture_markersize: float = 30.0,
    legend_ncol: int = 5,
    width_px: int = DEFAULT_WIDTH_PX,
) -> Path:
    """Two POI sources on one map: filled dots (OSM) inside hollow rings (Overture).

    Why not two figures. The interesting quantity is *disagreement*, and
    disagreement between point sets is only legible when the points share an
    extent, a scale and a pixel grid. A ring with a dot in it is a POI both
    providers know about; an empty ring is Overture-only; a bare dot is
    OSM-only. Encoding the two sources as fill-vs-outline rather than as two
    colours keeps the whole colour channel free for the OSM class, which is the
    second thing a reader wants.

    Overture is drawn first and larger so its rings frame the OSM dots rather
    than occluding them. Same bbox/extent contract as the other renderers, so
    this overlay is pixel-registered with every other layer of the district.
    """
    x0, y0, x1, y1 = bbox
    aspect = (y1 - y0) / (x1 - x0)

    counts = osm[column].fillna("<null>").value_counts()
    cats = list(category_order or []) + [
        c for c in counts.index if c not in (category_order or [])
    ]
    cats = [c for c in cats if c in counts.index]
    legend_rows = -(-(len(cats) + 3) // legend_ncol)

    fig_w = width_px / DPI
    fig_h = fig_w * aspect + 0.30 + 0.26 * legend_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    bottom = (0.30 + 0.26 * legend_rows) / fig_h
    ax = fig.add_axes([0.02, bottom * 0.75, 0.96, 1 - bottom * 0.75 - 0.06])

    if buildings is not None and len(buildings):
        buildings.plot(ax=ax, facecolor="#ececec", edgecolor="#dcdcdc", linewidth=0.2, zorder=1)
    if streets is not None and len(streets):
        streets.plot(ax=ax, facecolor="#f4f0e8", edgecolor="#e2dccf", linewidth=0.1, zorder=2)

    if overture is not None and len(overture):
        ax.scatter(
            overture.geometry.x, overture.geometry.y,
            s=overture_markersize, facecolor="none", edgecolor="#2b2b2b",
            linewidth=0.45, zorder=3,
        )

    for cat in cats:
        sel = osm[osm[column].fillna("<null>") == cat]
        ax.scatter(
            sel.geometry.x, sel.geometry.y,
            s=markersize, facecolor=palette.get(cat, "#999999"),
            edgecolor="#1a1a1a", linewidth=0.2, zorder=4,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    _scale_bar(ax, bbox)
    ax.set_title(title, fontsize=13, pad=8)

    handles: list[Any] = [
        Line2D([], [], linestyle="none", marker="o", markersize=6,
               markerfacecolor=palette.get(c, "#999999"), markeredgecolor="#1a1a1a",
               label=f"OSM {c}  (n={counts[c]})")
        for c in cats
    ]
    if overture is not None and len(overture):
        handles.append(
            Line2D([], [], linestyle="none", marker="o", markersize=7,
                   markerfacecolor="none", markeredgecolor="#2b2b2b",
                   label=f"Overture place  (n={len(overture)})")
        )
    if buildings is not None and len(buildings):
        handles.append(Patch(facecolor="#ececec", edgecolor="#dcdcdc",
                             label=f"buildings (context, n={len(buildings)})"))
    if streets is not None and len(streets):
        handles.append(Patch(facecolor="#f4f0e8", edgecolor="#e2dccf",
                             label=f"BGT street surfaces (context, n={len(streets)})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=legend_ncol, fontsize=9, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path
