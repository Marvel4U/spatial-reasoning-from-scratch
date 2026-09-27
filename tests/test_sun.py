"""``worldsnap.sun`` against published values. Run with ``-s`` to see the numbers.

    .venv/bin/python -m pytest tests/test_sun.py -s

WHY these particular checks. The sunshine layer is derived, so nothing downstream
can catch a wrong sun: a shadow map computed with a mirrored azimuth or a
half-hour time offset still looks like a plausible picture of a city. The three
almanac anchors pin the *time* and the *height* of the sun; the synthetic column
pins the *direction* of the shadow, which is the one thing a plausible-looking
overlay cannot reveal.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldsnap.sun import day_summary, shadow_mask, sun_position, sunlit_hours  # noqa: E402

LAT, LON = 52.35, 4.89  # De Pijp, Amsterdam

#: date -> (max elevation, day length h). Elevations are 90 - lat +/- the solar
#: declination at the solstices/equinox; the 21 June day length is the almanac's
#: sunrise 05:19 / sunset 22:06 CEST = 16 h 47 min = 16.78 h for Amsterdam.
ALMANAC = {
    dt.date(2026, 6, 21): (61.1, 16.78),
    dt.date(2026, 12, 21): (14.2, 7.7),
    dt.date(2026, 3, 21): (38.0, 12.2),
}


def test_sun_position_against_almanac() -> None:
    for date, (elev, day_h) in ALMANAC.items():
        got = day_summary(LAT, LON, date)
        print(f"  {date}  max elevation {got['max_elevation_deg']:6.2f} deg "
              f"(almanac ~{elev}), azimuth at max {got['azimuth_at_max_deg']:6.2f} deg "
              f"(expect ~180), solar noon {got['solar_noon_utc']} UTC, "
              f"day length {got['day_length_h']:5.2f} h (almanac ~{day_h})")
        assert abs(got["max_elevation_deg"] - elev) < 0.5
        assert abs(got["azimuth_at_max_deg"] - 180.0) < 0.5   # max elevation is due south
        assert abs(got["day_length_h"] - day_h) < 0.1


def test_sun_below_horizon_at_local_midnight() -> None:
    elev, _ = sun_position(LAT, LON, dt.datetime(2026, 6, 21, 22, 30, tzinfo=dt.timezone.utc))
    print(f"  21 Jun 22:30 UTC (00:30 CEST): elevation {elev:.2f} deg")
    assert elev < 0.0


def test_shadow_direction_and_length() -> None:
    """One 10 m column, sun at 45 deg: a 10 m shadow cast AWAY from the sun.

    West sun -> shadow to the east (higher column index); south sun -> shadow to
    the north, which is the LOWER row index because the raster is north-up. A
    sign error here would rotate every shadow in the district.
    """
    grid = torch.zeros(21, 21)
    grid[10, 10] = 10.0
    east, not_ground = shadow_mask(grid, 1.0, 45.0, 270.0, 30.0)
    north, _ = shadow_mask(grid, 1.0, 45.0, 180.0, 30.0)
    cols = [i for i, v in enumerate(east[10].tolist()) if v]
    rows = [i for i, v in enumerate(north[:, 10].tolist()) if v]
    print(f"  west sun (az 270)  -> shadow east,  columns {cols[0]}-{cols[-1]}")
    print(f"  south sun (az 180) -> shadow north, rows    {rows[0]}-{rows[-1]}")
    assert cols == list(range(11, 20))   # 9 pixels; the 10th is exactly on the ray
    assert rows == list(range(1, 10))
    assert not_ground[10, 10] and not not_ground[10, 11]


def test_night_is_all_shadow_and_buildings_are_not_ground() -> None:
    grid = torch.zeros(5, 5)
    grid[2, 2] = 10.0
    shadow, not_ground = shadow_mask(grid, 1.0, -5.0, 180.0)
    assert bool(shadow.all()) and int(not_ground.sum()) == 1

    hours = sunlit_hours(grid, 1.0, LAT, LON, dt.date(2026, 6, 21), 60)
    assert torch.isnan(hours[2, 2])                    # building interior -> nodata, not 0
    open_ground = float(hours[0, 0])
    print(f"  open ground next to a 10 m column, 21 Jun, 60-min steps: {open_ground:.2f} h")
    assert 15.0 < open_ground <= 17.0                  # ~16.8 h of daylight, little shading


def test_edges_do_not_wrap() -> None:
    """A tall block on the west edge must not shadow the east edge (no torch.roll)."""
    grid = torch.zeros(9, 9)
    grid[4, 0] = 30.0
    shadow, _ = shadow_mask(grid, 1.0, 45.0, 90.0, 40.0)  # east sun -> shadow runs west
    assert not bool(shadow[4, 8]) and not bool(shadow[4, 1])
