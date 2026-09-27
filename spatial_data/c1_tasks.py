"""c1 task definitions (C1_PLAN §3–§4). L5a: building only; L5b: core four; L5c: 12×(layer, class).

Establishment downsampling and CE foreground fractions depend on cropset (v2 discs vs v3/v3b footprints).
See docs/memo/ESTAB_FOOTPRINTS_2026-09-21.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

LayerKey = Literal["noise", "surface", "estab"]
Downsample = Literal["majority", "any", "centre"]

LAYER_INDEX = {"noise": 0, "surface": 1, "estab": 2}
N_C1_LAYERS = 3
N_C1_CLASSES = 4

FOOTPRINT_CROPSETS: frozenset[str] = frozenset({"v3", "v3b", "v4"})

# L5c default holdout (one surface + one estab atomic class); train on the other 10 pairs.
DEFAULT_HELD_OUT_KEYS: tuple[str, ...] = ("surface_0", "estab_2")

# Measured on de_pijp v2 (400 train crops, majority / any per C1_PLAN §3).
_FG_V2: dict[str, float] = {
    "noise_0": 0.376,
    "noise_1": 0.277,
    "noise_2": 0.326,
    "noise_3": 0.029,
    "surface_0": 0.411,
    "surface_1": 0.185,
    "surface_2": 0.162,
    "surface_3": 0.288,
    "estab_0": 0.99,
    "estab_1": 0.0043,
    "estab_2": 0.0026,
    "estab_3": 0.0036,
    "building": 0.288,
    "sidewalk": 0.162,
    "noise_ge_65": 0.354,
    "food_drink": 0.0036,
}

# Establishment layer on v3 footprints (majority rule); memo §4. Noise/surface unchanged vs v2.
_FG_V3_ESTAB: dict[str, float] = {
    "estab_0": 0.93,
    "estab_1": 0.0259,
    "estab_2": 0.0164,
    "estab_3": 0.0239,
    "food_drink": 0.0239,
}

# v3b: same weights until remeasured after 1000 m² host cap (expect close to v3).
_FG_V3B_ESTAB: dict[str, float] = dict(_FG_V3_ESTAB)

# Backward compat alias (v2 table).
TASK_CELL_FG_FRACTION: dict[str, float] = dict(_FG_V2)


@dataclass(frozen=True)
class C1TaskSpec:
    key: str
    layer: LayerKey
    class_id: int
    downsample: Downsample
    description: str
    cropset: str

    @property
    def layer_index(self) -> int:
        return LAYER_INDEX[self.layer]


def task_cell_fg_fractions(cropset: str) -> dict[str, float]:
    if cropset not in ("v2", "v3", "v3b", "v4"):
        raise ValueError(f"unknown cropset for c1 fractions {cropset!r}")
    out = dict(_FG_V2)
    if cropset == "v3":
        out.update(_FG_V3_ESTAB)
    elif cropset in ("v3b", "v4"):
        out.update(_FG_V3B_ESTAB)
    return out


def task_cell_fg_fraction(key: str, *, cropset: str = "v2") -> float:
    table = task_cell_fg_fractions(cropset)
    try:
        return table[key]
    except KeyError as e:
        raise KeyError(key) from e


def _estab_downsample(cropset: str) -> Downsample:
    return "majority" if cropset in FOOTPRINT_CROPSETS else "any"


def _atomic_key(layer: LayerKey, class_id: int) -> str:
    return f"{layer}_{class_id}"


def _atomic_spec(layer: LayerKey, class_id: int, cropset: str) -> C1TaskSpec:
    down: Downsample = _estab_downsample(cropset) if layer == "estab" else "majority"
    names = {
        "noise": ("<55 dB", "55–65 dB", "65–75 dB", "≥75 dB"),
        "surface": ("none", "roadway", "sidewalk", "building"),
        "estab": ("none", "other named", "shop", "food & drink"),
    }
    desc = f"mark {layer} = {names[layer][class_id]}"
    return C1TaskSpec(
        key=_atomic_key(layer, class_id),
        layer=layer,
        class_id=class_id,
        downsample=down,
        description=desc,
        cropset=cropset,
    )


FACTORISED_TWELVE_KEYS: tuple[str, ...] = tuple(
    _atomic_key(layer, c) for layer in ("noise", "surface", "estab") for c in range(N_C1_CLASSES)
)

CORE_FOUR_KEYS: tuple[str, ...] = ("building", "sidewalk", "noise_ge_65", "food_drink")


@lru_cache(maxsize=64)
def get_c1_task(key: str, cropset: str = "v2") -> C1TaskSpec:
    if cropset not in ("v2", "v3", "v3b", "v4"):
        raise ValueError(f"unknown cropset {cropset!r}")
    if key in FACTORISED_TWELVE_KEYS:
        layer: LayerKey = key.split("_", 1)[0]  # type: ignore[assignment]
        class_id = int(key.rsplit("_", 1)[-1])
        return _atomic_spec(layer, class_id, cropset)
    if key == "building":
        return C1TaskSpec(
            key="building", layer="surface", class_id=3, downsample="majority",
            description="mark surface = building", cropset=cropset,
        )
    if key == "sidewalk":
        return C1TaskSpec(
            key="sidewalk", layer="surface", class_id=2, downsample="majority",
            description="mark surface = sidewalk", cropset=cropset,
        )
    if key == "noise_ge_65":
        return C1TaskSpec(
            key="noise_ge_65", layer="noise", class_id=-1, downsample="majority",
            description="mark noise >= 65 dB (classes 2+3)", cropset=cropset,
        )
    if key == "food_drink":
        return C1TaskSpec(
            key="food_drink", layer="estab", class_id=3, downsample=_estab_downsample(cropset),
            description="mark estab = food & drink", cropset=cropset,
        )
    raise KeyError(f"unknown c1 task {key!r}")


def task_specs_for_keys(keys: list[str], *, cropset: str) -> list[C1TaskSpec]:
    return [get_c1_task(k, cropset) for k in keys]


def task_keys_for_mode(mode: str) -> list[str]:
    if mode == "single":
        raise ValueError("single mode uses config.c1_task_key, not task_keys_for_mode")
    if mode == "core_four":
        return list(CORE_FOUR_KEYS)
    if mode == "factorised_twelve":
        return list(FACTORISED_TWELVE_KEYS)
    raise ValueError(f"unknown c1_task_mode {mode!r}")


def train_task_keys(mode: str, held_out: tuple[str, ...]) -> list[str]:
    keys = task_keys_for_mode(mode)
    if mode != "factorised_twelve":
        return keys
    hold = set(held_out)
    unknown = hold - set(keys)
    if unknown:
        raise ValueError(f"held-out keys not in factorised twelve: {sorted(unknown)}")
    if len(hold) != 2:
        raise ValueError(f"L5c expects exactly 2 held-out keys, got {len(hold)}")
    return [k for k in keys if k not in hold]


def train_task_indices(mode: str, held_out: tuple[str, ...], active_keys: list[str]) -> list[int] | None:
    if mode != "factorised_twelve":
        return None
    train_keys = train_task_keys(mode, held_out)
    return [active_keys.index(k) for k in train_keys]


def held_out_task_indices(active_keys: list[str], held_out: tuple[str, ...]) -> list[int]:
    return [active_keys.index(k) for k in held_out]


def compositional_wrong_task_index(correct_tid: int, active_keys: list[str], *, cropset: str = "v2") -> int:
    """Same layer, different class — wrong-task baseline for L5c."""
    spec = get_c1_task(active_keys[correct_tid], cropset)
    wrong_c = (spec.class_id + 1) % N_C1_CLASSES
    wrong_key = _atomic_key(spec.layer, wrong_c)
    return active_keys.index(wrong_key)


def factorised_ids_for_keys(keys: list[str], *, cropset: str = "v2") -> tuple[tuple[int, ...], tuple[int, ...]]:
    layers: list[int] = []
    classes: list[int] = []
    for k in keys:
        s = get_c1_task(k, cropset)
        if s.class_id < 0:
            raise ValueError(f"task {k!r} is not an atomic (layer, class) pair")
        layers.append(s.layer_index)
        classes.append(s.class_id)
    return tuple(layers), tuple(classes)


def task_index(key: str, keys: list[str]) -> int:
    try:
        return keys.index(key)
    except ValueError as e:
        raise KeyError(f"task {key!r} not in {keys}") from e


def per_task_fg_ce_weight(key: str, *, cropset: str = "v2", clamp: float = 100.0) -> float:
    f = task_cell_fg_fraction(key, cropset=cropset)
    w = (1.0 - f) / max(f, 1e-9)
    return min(w, clamp)


def fg_ce_weights_for_keys(keys: list[str], *, cropset: str = "v2", clamp: float = 100.0) -> tuple[float, ...]:
    return tuple(per_task_fg_ce_weight(k, cropset=cropset, clamp=clamp) for k in keys)
