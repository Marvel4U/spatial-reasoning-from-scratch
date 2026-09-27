"""District definitions and CRS conventions.

WHY WGS84 in, RD out
--------------------
Districts are declared in WGS84 (EPSG:4326) because that is the CRS you can
read off any map or OSM URL, so the definitions stay human-checkable. Everything
downstream works in **EPSG:28992 (Amersfoort / RD New)**, the Dutch national
projected CRS: metres, minimal distortion over the Netherlands, and the native
CRS of BAG / BGT / AHN / 3DBAG. Distances and areas are therefore metric and
directly verifiable, which is the whole point for a spatial-reasoning verifier.

(3DBAG's own storage CRS is EPSG:7415 = RD New + NAP height. Its horizontal
component *is* EPSG:28992, so dropping Z gives valid 28992 geometry.)
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer

#: WGS84, the CRS district bboxes are authored in.
CRS_WGS84 = "EPSG:4326"
#: Amersfoort / RD New — the working CRS for all stored layers (metres).
CRS_RD = "EPSG:28992"

_TO_RD = Transformer.from_crs(CRS_WGS84, CRS_RD, always_xy=True)


@dataclass(frozen=True)
class District:
    """A named rectangular study area.

    Attributes:
        name: slug used for file paths, e.g. ``de_pijp``.
        city: slug used for file paths, e.g. ``amsterdam``.
        label: human-readable name for plot titles.
        min_lon, min_lat, max_lon, max_lat: bbox in WGS84 decimal degrees.
    """

    name: str
    city: str
    label: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def bbox_rd(self) -> tuple[float, float, float, float]:
        """Bbox in EPSG:28992 metres, as (min_x, min_y, max_x, max_y).

        The axis-aligned WGS84 rectangle is not exactly axis-aligned in RD, so
        we project all four corners and take the *envelope*. At district scale
        (~2 km) the difference is a few metres, but taking the envelope
        guarantees we never clip inside the requested area.
        """
        xs, ys = zip(
            *(
                _TO_RD.transform(lon, lat)
                for lon in (self.min_lon, self.max_lon)
                for lat in (self.min_lat, self.max_lat)
            )
        )
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def area_km2_rd(self) -> float:
        """Bbox area in km^2 (RD metres), for density sanity checks."""
        x0, y0, x1, y1 = self.bbox_rd
        return (x1 - x0) * (y1 - y0) / 1e6


# De Pijp, Amsterdam.
#
# Bbox verified to contain the three landmarks used as a sanity anchor:
#   Albert Cuypstraat  ~ 4.888-4.899 E, 52.3555 N   (market street, W-E)
#   Sarphatipark       ~ 4.893-4.896 E, 52.3535-52.3555 N
#   metro De Pijp      ~ 4.8926 E, 52.3532 N        (Ceintuurbaan/N-Z lijn)
# all comfortably inside. Bbox kept exactly as briefed - no adjustment needed.
# The north edge (52.361) reaches over the Singelgracht into the Weteringschans
# fringe and the south edge (52.345) into the Amstelkanaal/Rivierenbuurt edge,
# i.e. the rectangle is slightly larger than the administrative buurt. That is
# intentional for a pilot: a rectangle is trivially reproducible, and buildings
# are truncated by a hard bbox rather than by an official boundary polygon.
DE_PIJP = District(
    name="de_pijp",
    city="amsterdam",
    label="De Pijp, Amsterdam",
    min_lon=4.884,
    min_lat=52.345,
    max_lon=4.910,
    max_lat=52.361,
)

#: Registry for the CLI: ``--district <key>``.
DISTRICTS: dict[str, District] = {DE_PIJP.name: DE_PIJP}
