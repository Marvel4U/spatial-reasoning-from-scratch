"""wave_rollout_start_pass_for_route (first_back is route-dependent)."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "harness"))

from spatial_data.c4_tasks import WAVE_HORIZON_M
import diagnostics.wave_search_rollout as wr
from diagnostics.wave_search_rollout import wave_rollout_start_pass_for_route


def test_first_back_pass_index():
    def fake_goal(route_ix: int) -> float:
        return 8.0 * WAVE_HORIZON_M + 1.0  # n_goal=9 -> first back pass 10

    orig = wr._wave_val_goal_m
    wr._wave_val_goal_m = fake_goal
    try:
        assert wave_rollout_start_pass_for_route(0, "first_back") == 10
        assert wave_rollout_start_pass_for_route(363, "first_back") == 10
        assert wave_rollout_start_pass_for_route(0, "reach") == 9
        assert wave_rollout_start_pass_for_route(0, 2) == 2
    finally:
        wr._wave_val_goal_m = orig
