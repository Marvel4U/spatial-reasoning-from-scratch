"""Prompt text for smoke probes and vanilla route V1 (plain map)."""

COORD_LEGEND = (
    "Use a coordinate grid from 0 to 1000: (0, 0) is the top-left corner of the image, "
    "(1000, 1000) is the bottom-right. x increases to the right, y increases downward."
)

MAP_LEGEND = (
    "The image is a top-down map. Light areas are walkable ground. "
    "Dark grey shapes are buildings (not walkable). "
    "Green disc = start. Red disc = goal."
)


def describe_brief() -> str:
    return "Describe this image briefly."


def marker_probe(w: int, h: int) -> str:
    return (
        f"The image is {w}x{h} pixels. {COORD_LEGEND} "
        'Where is the green disc and where is the red disc? '
        'Reply with JSON only: {"green": [x, y], "red": [x, y]}.'
    )


def route_v1(w: int, h: int) -> str:
    return (
        f"{MAP_LEGEND} {COORD_LEGEND} "
        "Find a walking route from the green start to the red goal that avoids dark building areas. "
        "Use only 4–8 waypoints (hard maximum 12). First point at the start, last at the goal; "
        "do not trace the image border. "
        "Reply with JSON only: [[x0,y0],[x1,y1],...] in 0–1000 coordinates."
    )


def route_v2_structured(w: int, h: int) -> str:
    """Single turn: force explicit start/goal then path (same 0–1000 frame as marker probe)."""
    return (
        f"The map image is exactly {w}×{h} pixels. {MAP_LEGEND} {COORD_LEGEND} "
        "All coordinates are on the 0–1000 grid aligned to this image (not a smaller crop). "
        "Task in two steps in one reply:\n"
        "1) Locate the green start and red goal.\n"
        "2) Plan a walking path on light ground only; detour around dark buildings.\n"
        'JSON only: {"start": [x, y], "goal": [x, y], "path": [[x, y], ...]} '
        "where path[0] equals start, path[-1] equals goal, 4–8 points total (max 12). "
        "Do not follow the image border."
    )


def route_oracle_endpoints(w: int, h: int, start_q: list, goal_q: list) -> str:
    """Ground-truth start/goal on 0–1000 grid (oracle upper bound on endpoint confusion)."""
    s = [int(round(start_q[0])), int(round(start_q[1]))]
    g = [int(round(goal_q[0])), int(round(goal_q[1]))]
    return (
        f"The map is {w}×{h} pixels. {MAP_LEGEND} {COORD_LEGEND} "
        f"The green start is at {s} and the red goal is at {g} on the 0–1000 grid. "
        "Find a walking route on light ground from start to goal, detouring around dark buildings. "
        f'JSON only: {{"path": [[x, y], ...]}} with path[0] = {s} and path[-1] = {g}, '
        "4–8 points including endpoints (max 12). Do not trace the image border."
    )


def route_after_markers(w: int, h: int, green: list, red: list) -> str:
    """Second turn after a successful marker probe."""
    return (
        f"The map is {w}×{h} pixels. {MAP_LEGEND} {COORD_LEGEND} "
        f"You already located the green start at {green} and the red goal at {red} (0–1000 grid). "
        "Find a walking route on light ground from start to goal, detouring around dark buildings. "
        f'JSON only: {{"path": [[x, y], ...]}} with path[0] = {green} and path[-1] = {red}, '
        "4–8 points including endpoints (max 12). Stay over the map; do not trace the border."
    )
