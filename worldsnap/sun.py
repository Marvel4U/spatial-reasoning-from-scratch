"""Solar geometry and building shadows: pure functions, no I/O.

WHY this module exists separately from ``layers/sunshine.py``
-------------------------------------------------------------
Sunlit hours are the project's first *derived* layer: unlike every other layer
so far, no provider publishes the answer and no download can be cited. The
credibility therefore has to come from the code being small enough to read and
from every definitional choice being written down. Splitting the arithmetic
(here) from the district plumbing (``layers/sunshine.py``) is what makes that
possible: this file can be checked against an almanac without touching a raster,
and it is the only place where a wrong sign would silently rotate every shadow.

FORMULA SOURCE
--------------
Solar position follows the *low-precision formulae for the Sun* of the
Astronomical Almanac (section C24; reproduced in the NOAA solar calculator and
in Meeus, "Astronomical Algorithms", ch. 25 as the abridged series). It models
the Sun's apparent ecliptic longitude with two equation-of-centre terms and is
stated by the source to be accurate to about 0.01 deg in right ascension and
declination for 1950-2050. Greenwich mean sidereal time uses the USNO linear
expression in days from J2000.

Deliberately NOT modelled, because each is smaller than the 1 m raster's own
discretisation error and adding them would make the module harder to audit:
  * atmospheric refraction (~0.5 deg of apparent lift at the horizon, ~0.01 deg
    above 30 deg elevation). Consequence: near sunrise and sunset this code puts
    the sun slightly lower than an observer sees it, so sunlit hours are, if
    anything, *under*-estimated by a couple of minutes per day.
  * the solar disc's ~0.27 deg radius (we treat the sun as a point);
  * parallax, nutation, and the difference between UT1 and UTC (< 1 s).

SHADOW MODEL AND ITS LIMITS
---------------------------
A 2.5D ray march over a height-above-ground raster: a ground pixel is shadowed
if, stepping towards the sun in ``res_m`` increments, the height raster at any
shifted position rises above the straight line ``d * tan(elevation)``. This is
exact for vertical extrusions on flat ground and is the standard hillshade /
horizon-angle construction. What it assumes, and what that costs:
  * **Flat ground.** Heights are above *local* ground (3DBAG's roof percentile
    minus its own maaiveld), and De Pijp's terrain spans ~2 m across 1.8 km, so
    the error from ignoring the slope is a few centimetres of shadow length.
  * **Only direct beam.** No sky diffuse, no reflection: "sunlit" means the
    solar disc centre is geometrically visible, not "bright".
  * **Truncation at ``max_dist_m``.** Beyond that distance a building is ignored,
    so very low sun (elevation < atan(h / max_dist)) loses its longest shadows
    and sunlit hours are over-estimated near sunrise and sunset. 200 m at 1 m
    steps is the documented default; the marching also stops early once
    ``d * tan(elevation)`` exceeds the tallest building, which changes nothing.
  * **Nothing outside the raster casts a shadow.** The shift pads with zero, so
    a 200 m band along the district edge is lit slightly too much.
"""

from __future__ import annotations

import datetime as _dt
import math

import torch

#: Days per Julian century is not needed at this precision; J2000.0 as a Julian
#: day number is, and the Unix epoch as a Julian day is how we get there exactly.
_JD_UNIX_EPOCH = 2440587.5
_JD_J2000 = 2451545.0


def julian_day(dt_utc: _dt.datetime) -> float:
    """Julian day number of a timezone-aware UTC datetime.

    Going through the POSIX timestamp avoids hand-rolling a calendar; it is
    exact because ``datetime.timestamp()`` on an aware datetime is defined as
    seconds since 1970-01-01T00:00:00Z (leap seconds are not represented, which
    is what we want: we are using UTC as a stand-in for UT1, error < 1 s).
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC); naive datetimes are ambiguous")
    return dt_utc.timestamp() / 86400.0 + _JD_UNIX_EPOCH


def sun_position(lat_deg: float, lon_deg: float, dt_utc: _dt.datetime) -> tuple[float, float]:
    """Apparent solar (elevation, azimuth) in degrees; azimuth 0 = north, 90 = east.

    Elevation is the *true geometric* altitude of the disc centre above the
    horizontal plane (negative below the horizon, no refraction, see module
    docstring). Azimuth is measured clockwise from north, the convention the
    shadow march below needs and the one every compass uses.

    ``lon_deg`` is positive east. See the module docstring for the source.
    """
    n = julian_day(dt_utc) - _JD_J2000
    mean_long = math.radians((280.460 + 0.9856474 * n) % 360.0)
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360.0)
    # Apparent ecliptic longitude: mean longitude + the two-term equation of centre.
    ecl = mean_long + math.radians(1.915) * math.sin(mean_anom) \
        + math.radians(0.020) * math.sin(2.0 * mean_anom)
    obliquity = math.radians(23.439 - 3.6e-7 * n)
    right_asc = math.atan2(math.cos(obliquity) * math.sin(ecl), math.cos(ecl))
    declination = math.asin(math.sin(obliquity) * math.sin(ecl))

    gmst_hours = (18.697374558 + 24.06570982441908 * n) % 24.0
    # Local hour angle, wrapped to (-180, 180] so "morning" is negative.
    hour_angle = math.radians(
        ((gmst_hours * 15.0 + lon_deg - math.degrees(right_asc)) + 180.0) % 360.0 - 180.0
    )

    lat = math.radians(lat_deg)
    sin_el = math.sin(lat) * math.sin(declination) \
        + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    azimuth = math.degrees(math.atan2(
        -math.cos(declination) * math.sin(hour_angle),
        math.cos(lat) * math.sin(declination)
        - math.sin(lat) * math.cos(declination) * math.cos(hour_angle),
    )) % 360.0
    return elevation, azimuth


def _shift(t: torch.Tensor, d_row: int, d_col: int) -> torch.Tensor:
    """``out[i, j] = t[i + d_row, j + d_col]``, zero outside the raster.

    Deliberately NOT ``torch.roll``: rolling wraps, so a tall block on the west
    edge of the district would cast its shadow onto the east edge. Zero padding
    instead means "no building known out there" -- wrong in a different, honest
    and documented way (see the module docstring's edge-band caveat).
    """
    h, w = t.shape
    out = torch.zeros_like(t)
    r0, r1 = max(0, d_row), min(h, h + d_row)
    c0, c1 = max(0, d_col), min(w, w + d_col)
    if r0 < r1 and c0 < c1:
        out[r0 - d_row:r1 - d_row, c0 - d_col:c1 - d_col] = t[r0:r1, c0:c1]
    return out


def shadow_mask(height_m: torch.Tensor, res_m: float, elevation_deg: float,
                azimuth_deg: float, max_dist_m: float = 200.0,
                building: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Shadow at ground level for one sun position.

    Returns ``(shadowed, not_ground)``, both bool ``(H, W)``. The second tensor
    is the point of the two-value return: a pixel inside a building footprint is
    not a piece of ground at all, so it is neither "sunlit" nor "shadowed" and
    must be excluded from every statistic rather than silently counted as dark.
    ``building`` defaults to ``height_m > 0``; the builder passes the rasterised
    footprints instead, so that a footprint whose 3DBAG height is missing or
    non-positive still counts as "not ground" (it just casts no shadow).

    Sun below the horizon -> everything shadowed (and nothing sunlit), which is
    the physically right answer and keeps the caller free of special cases.

    The march is vectorised over the whole raster: one shifted comparison per
    step, at most ``max_dist_m / res_m`` steps, regardless of raster size.
    Step offsets are rounded to whole pixels, so along a diagonal azimuth two
    consecutive steps can land on the same pixel; that costs a redundant
    comparison and never skips a pixel (a unit step in a direction can advance
    at most one row and one column).
    """
    if height_m.ndim != 2:
        raise ValueError(f"height_m must be (H, W), got {tuple(height_m.shape)}")
    not_ground = (height_m > 0) if building is None else building
    if elevation_deg <= 0.0:
        return torch.ones_like(not_ground), not_ground

    tan_el = math.tan(math.radians(elevation_deg))
    # Nothing beyond h_max / tan(elevation) can ever reach the ray, so stop there.
    reach_m = float(height_m.max().item()) / tan_el
    n_steps = int(min(max_dist_m, reach_m) / res_m)
    sin_a, cos_a = math.sin(math.radians(azimuth_deg)), math.cos(math.radians(azimuth_deg))

    shadowed = torch.zeros_like(not_ground)
    for k in range(1, n_steps + 1):
        # Step k pixels towards the sun: east is +col, north is -row.
        shifted = _shift(height_m, int(round(-k * cos_a)), int(round(k * sin_a)))
        shadowed |= shifted > (k * res_m) * tan_el
    return shadowed, not_ground


def sunlit_hours(height_m: torch.Tensor, res_m: float, lat_deg: float, lon_deg: float,
                 date: _dt.date, step_minutes: int = 15, max_dist_m: float = 200.0,
                 building: torch.Tensor | None = None) -> torch.Tensor:
    """Hours of direct sun at ground level over one **UTC calendar day**.

    UTC handling, stated explicitly because it is a definitional choice: the day
    runs from ``date`` 00:00:00Z to 24:00:00Z, and each step is evaluated at its
    **midpoint** (``t + step/2``), which is unbiased where a sunrise or sunset
    falls inside a step. For the Netherlands (UTC+1/+2 civil time) a UTC day
    contains the whole local daylight period on both dates this layer is built
    for -- 21 June, sun up ~03:19-20:06 UTC, and 21 March, ~05:30-17:35 UTC --
    so "the UTC day" and "the local day" select the same sunshine. That would
    stop being true for a longitude far from Greenwich, hence the note.

    Building-interior pixels come back as NaN, not 0: they are not ground, and a
    zero there would be read as "permanently shaded courtyard".
    """
    if 1440 % step_minutes:
        raise ValueError(f"step_minutes must divide 1440, got {step_minutes}")
    not_ground = (height_m > 0) if building is None else building
    hours = torch.zeros_like(height_m)
    midnight = _dt.datetime(date.year, date.month, date.day, tzinfo=_dt.timezone.utc)
    for minute in range(0, 1440, step_minutes):
        instant = midnight + _dt.timedelta(minutes=minute + step_minutes / 2.0)
        elevation, azimuth = sun_position(lat_deg, lon_deg, instant)
        if elevation <= 0.0:
            continue
        shadowed, _ = shadow_mask(height_m, res_m, elevation, azimuth, max_dist_m, not_ground)
        hours += (~shadowed).to(hours.dtype) * (step_minutes / 60.0)
    return hours.masked_fill(not_ground, float("nan"))


def day_summary(lat_deg: float, lon_deg: float, date: _dt.date,
                minute_step: int = 1) -> dict:
    """Max elevation, its azimuth and instant, and the day length, all in UTC.

    The quantities an almanac publishes, so that this module can be checked
    against one (``tests/test_sun.py``). Day length uses the conventional
    -0.833 deg sunrise altitude, which is the refraction-plus-disc-radius
    allowance almanacs use; ``sun_position`` itself models neither, so the
    allowance is applied here, at the point of comparison, and nowhere else.
    """
    midnight = _dt.datetime(date.year, date.month, date.day, tzinfo=_dt.timezone.utc)
    samples = [(sun_position(lat_deg, lon_deg, midnight + _dt.timedelta(minutes=m)), m)
               for m in range(0, 1440, minute_step)]
    (elev, azim), minute = max(samples, key=lambda s: s[0][0])
    up = sum(1 for (e, _), _ in samples if e > -0.833) * minute_step / 60.0
    return {"max_elevation_deg": round(elev, 2), "azimuth_at_max_deg": round(azim, 2),
            "solar_noon_utc": f"{minute // 60:02d}:{minute % 60:02d}",
            "day_length_h": round(up, 2)}
