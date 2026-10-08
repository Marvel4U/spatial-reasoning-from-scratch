"""Deterministic pass order for C5 wave-search rollout (teacher forcing schedule at test time)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WavePass:
    label: str
    task_key: str
    tok_out: str  # SEARCH | BACK | DRAW | DONE
    k_in: int | None = None   # wave_step: fixed band index (training picks at random)
    j_back: int | None = None  # wave_back: fixed front index j


def n_goal_from_m(goal_m: float, horizon_m: float = 40.0) -> int:
    if not (goal_m == goal_m) or goal_m <= 0:  # nan / unreachable
        return 1
    return int(goal_m // horizon_m) + 1


def wave_pass_schedule(n_goal: int) -> list[WavePass]:
    """Full search episode for one route: forward fronts, reach, peel fronts, origin, final path."""
    n_goal = max(1, int(n_goal))
    out: list[WavePass] = [WavePass("1 · first front", "wave_first", "SEARCH")]
    for k in range(1, n_goal - 1):
        out.append(WavePass(f"2 · extend (+{k} band)", "wave_step", "SEARCH", k_in=k))
    out.append(WavePass("3 · wave hits goal", "wave_reach", "BACK"))
    for j in range(n_goal, 1, -1):
        tok = "BACK" if j > 1 else "DRAW"
        out.append(WavePass(f"4 · consume front j={j}", "wave_back", tok, j_back=j))
    out.append(WavePass("5 · at start", "wave_origin", "DRAW"))
    out.append(WavePass("6 · path only", "wave_draw", "DONE"))
    return out
