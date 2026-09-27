"""Builders only — specs and parameter choices live in experiments.py."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _patch_geometry(patch_size: int, img_size: int = 256, grid_out: int = 64) -> tuple[int, int]:
    g = img_size // patch_size
    s = grid_out // g
    if g * s != grid_out:
        raise ValueError(f"patch_size={patch_size}: grid_out={grid_out} not divisible by g={g}")
    return g, s


def build_c0_worldsnap_p_sweep(spec: dict[str, Any]) -> list[dict]:
    """Expand one sweep spec → list of runnable experiment dicts (spread into RUN)."""
    base = deepcopy(spec["base"])
    base_overrides = deepcopy(base.pop("overrides", {}))
    per_p = spec.get("patch_overrides") or {}
    out: list[dict] = []
    for p in spec["patch_sizes"]:
        g, s = _patch_geometry(p)
        po = deepcopy(per_p.get(p, per_p.get(str(p), {})))
        overrides = {**base_overrides, "patch_size": p, **po}
        out.append({
            **base,
            "id": spec.get("id_template", "c0_worldsnap_P{p}_2k").format(p=p),
            "overrides": overrides,
            "notes": spec.get(
                "notes_template",
                "E1 c0 worldsnap P={p} (g={g}, s={s}); subcell chance=1/{ss}",
            ).format(p=p, g=g, s=s, ss=s * s),
        })
    return out


def as_single(spec: dict[str, Any], *, run_id: str, notes: str = "") -> list[dict]:
    """Wrap one spec as a one-element list (for RUN)."""
    return [{"id": run_id, "notes": notes or run_id, **deepcopy(spec)}]
