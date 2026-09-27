"""c2 task definitions: conjunctions of conditions across layers (C2_PLAN §3, §6–§7).

A *condition* (atom) is (layer, class): 12 of them, id = layer_index*4 + class_id.
A *task* is 1 atom (= a c1 single-condition task) or 2 atoms from DIFFERENT layers.
Target rule (C2_PLAN §3, Marvin's choice): a PIXEL is positive if every atom holds
there; a CELL (4x4 px) is positive if >= 8 of its 16 pixels are. Single-atom tasks
use the same rule, so estab singles are majority-downsampled on v3b like c1 does.

Positive shares `f` are never typed in: they are measured on the train crops and
cached next to the cropset, because they drive both the CE weight and the decision
rule, and a stale hand-typed number silently biases every decision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAYERS: tuple[str, ...] = ("noise", "surface", "estab")
LAYER_INDEX = {k: i for i, k in enumerate(LAYERS)}
N_CLASSES = 4
CLASS_NAMES = {
    "noise": ("<55 dB", "55-65 dB", "65-75 dB", ">=75 dB"),
    "surface": ("none", "roadway", "sidewalk", "building"),
    "estab": ("none", "other named", "shop", "food & drink"),
}

# Usability filter, C2_PLAN §5b. ratio = f(A and B) / min(f(A), f(B)).
MIN_F = 0.003
MAX_EMPTY_SHARE = 0.5
MAX_RATIO = 0.9

STATS_FILENAME = "c2_task_stats.json"


def atom_id(layer: str, class_id: int) -> int:
    if layer not in LAYER_INDEX or not 0 <= class_id < N_CLASSES:
        raise ValueError(f"bad atom ({layer!r}, {class_id})")
    return LAYER_INDEX[layer] * N_CLASSES + class_id


def atom_key(aid: int) -> str:
    return f"{LAYERS[aid // N_CLASSES]}_{aid % N_CLASSES}"


def atom_from_key(key: str) -> int:
    layer, cls = key.rsplit("_", 1)
    return atom_id(layer, int(cls))


ATOM_KEYS: tuple[str, ...] = tuple(atom_key(a) for a in range(len(LAYERS) * N_CLASSES))

# Human names given in C2_PLAN §3 (plus the ts2 singles).
HUMAN_NAMES: dict[str, str] = {
    "noise_0+surface_2": "quiet sidewalk",
    "noise_0+surface_3": "quiet building",
    "noise_2+surface_1": "loud roadway",
    "surface_3+estab_0": "unoccupied building",
    "noise_0+estab_3": "quiet food & drink",
    "surface_3": "building",
    "surface_2": "sidewalk",
    "estab_3": "food & drink",
    "noise_2": "noise 65-75 dB",
}


@dataclass(frozen=True)
class C2TaskSpec:
    key: str                  # "noise_0+surface_2" or "surface_3"
    atoms: tuple[int, ...]    # 1 or 2 atom ids, ordered by layer index
    name: str                 # human name, or the key if none is defined

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def description(self) -> str:
        parts = [f"{LAYERS[a // N_CLASSES]} = {CLASS_NAMES[LAYERS[a // N_CLASSES]][a % N_CLASSES]}"
                 for a in self.atoms]
        return "mark " + " and ".join(parts)

    def padded_cond_ids(self) -> tuple[int, int]:
        """(a, b) with -1 = "always true" so single-atom tasks use the same target path."""
        return (self.atoms[0], self.atoms[1] if self.n_atoms == 2 else -1)


def make_task(*atom_keys: str) -> C2TaskSpec:
    ids = sorted(atom_from_key(k) for k in atom_keys)
    if not 1 <= len(ids) <= 2:
        raise ValueError("a c2 task has 1 or 2 atoms")
    if len(ids) == 2 and ids[0] // N_CLASSES == ids[1] // N_CLASSES:
        raise ValueError(f"both atoms on the same layer: {atom_keys}")
    key = "+".join(atom_key(a) for a in ids)
    return C2TaskSpec(key=key, atoms=tuple(ids), name=HUMAN_NAMES.get(key, key))


def get_c2_task(key: str) -> C2TaskSpec:
    return make_task(*key.split("+"))


def wrong_cond_ids_padded(spec: C2TaskSpec, *, slot: int = 0) -> tuple[int, int]:
    """Same layer, different class on one atom — wrong-description control for L6c (mirrors L5c)."""
    if spec.n_atoms != 2:
        raise ValueError(f"wrong description requires a conjunction, got {spec.key!r}")
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    a, b = spec.padded_cond_ids()
    aid = a if slot == 0 else b
    layer, cls = aid // N_CLASSES, aid % N_CLASSES
    flipped = layer * N_CLASSES + (cls + 1) % N_CLASSES
    return (flipped, b) if slot == 0 else (a, flipped)


C2_CORE_KEYS: tuple[str, ...] = (
    "noise_0+surface_2",   # quiet sidewalk
    "noise_0+surface_3",   # quiet building
    "noise_2+surface_1",   # loud roadway
    "surface_3+estab_0",   # unoccupied building
    "noise_0+estab_3",     # quiet food & drink
)

# ts2 = the four c1 core tasks + c2 core. NOTE: c1's "noise >= 65 dB" spans classes 2 and 3
# and is therefore NOT a single atom; ts2 uses noise_2 (65-75 dB) in its place.
TS2_SINGLE_KEYS: tuple[str, ...] = ("surface_3", "surface_2", "noise_2", "estab_3")
TS2_KEYS: tuple[str, ...] = TS2_SINGLE_KEYS + C2_CORE_KEYS

# Held-out conjunctions for c2_full (C2_PLAN §5c): each atom still appears in >= 3
# trained conjunctions and as a single-atom task.
C2_FULL_HELD_OUT: tuple[str, ...] = (
    "noise_0+surface_1",   # <55 dB and roadway
    "noise_1+surface_3",   # 55-65 dB and building
    "noise_2+surface_2",   # 65-75 dB and sidewalk
    "noise_1+estab_2",     # 55-65 dB and shop
)


def all_candidate_keys() -> list[str]:
    """12 single atoms + every cross-layer pair (48) — the set `f` is measured for."""
    keys = list(ATOM_KEYS)
    for i, la in enumerate(LAYERS):
        for lb in LAYERS[i + 1:]:
            for ca in range(N_CLASSES):
                for cb in range(N_CLASSES):
                    keys.append(make_task(f"{la}_{ca}", f"{lb}_{cb}").key)
    return keys


def _cells_positive(pix: np.ndarray) -> np.ndarray:
    """(n, 256, 256) bool pixels -> (n, 64, 64) bool cells by 4x4 majority (>= 8 of 16)."""
    n = pix.shape[0]
    blk = pix.reshape(n, 64, 4, 64, 4).transpose(0, 1, 3, 2, 4).reshape(n, 64, 64, 16)
    return blk.sum(-1) >= 8


def cell_target_numpy(planes: dict[str, np.ndarray], spec: C2TaskSpec) -> np.ndarray:
    """Reference target: planes[layer] (n, 256, 256) uint8 -> (n, 64, 64) bool."""
    pix = np.ones_like(planes["noise"], dtype=bool)
    for a in spec.atoms:
        pix &= planes[LAYERS[a // N_CLASSES]] == (a % N_CLASSES)
    return _cells_positive(pix)


def _load_train_planes(crops_dir: Path) -> dict[str, np.ndarray]:
    files = sorted((crops_dir / "train").glob("*_labels.npz"))
    if not files:
        raise FileNotFoundError(f"no train crops under {crops_dir}")
    return {k: np.stack([np.load(f)[f"{k}_256"] for f in files]) for k in LAYERS}


def compute_task_stats(crops_dir: Path) -> dict:
    """Measure f, empty-crop share and per-crop f percentiles for every candidate task."""
    planes = _load_train_planes(crops_dir)
    n = planes["noise"].shape[0]
    tasks: dict[str, dict] = {}
    for key in all_candidate_keys():
        cells = cell_target_numpy(planes, get_c2_task(key))
        per = cells.reshape(n, -1).mean(1)
        tasks[key] = {
            "f": float(cells.mean()),
            "empty_share": float((per == 0).mean()),
            "p10": float(np.percentile(per, 10)),
            "p50": float(np.percentile(per, 50)),
            "p90": float(np.percentile(per, 90)),
        }
    return {"cropset_dir": str(crops_dir), "n_crops": n, "rule": "pixel-AND then 4x4 majority",
            "tasks": tasks}


def load_task_stats(crops_dir: Path, *, rebuild: bool = False) -> dict:
    """Cached stats at crops/<cropset>/c2_task_stats.json; measured once, reused after."""
    path = Path(crops_dir) / STATS_FILENAME
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    stats = compute_task_stats(Path(crops_dir))
    path.write_text(json.dumps(stats, indent=1))
    return stats


def positive_share(key: str, stats: dict) -> float:
    return float(stats["tasks"][key]["f"])


def is_usable(key: str, stats: dict) -> bool:
    """C2_PLAN §5b: drop near-empty tasks and conjunctions that are their own rarer part."""
    spec = get_c2_task(key)
    t = stats["tasks"][key]
    if t["f"] < MIN_F or t["empty_share"] > MAX_EMPTY_SHARE:
        return False
    if spec.n_atoms == 2:
        parts = [stats["tasks"][atom_key(a)]["f"] for a in spec.atoms]
        if min(parts) <= 0 or t["f"] / min(parts) > MAX_RATIO:
            return False
    return True


def usable_conjunction_keys(stats: dict) -> list[str]:
    return [k for k in all_candidate_keys() if "+" in k and is_usable(k, stats)]


def task_keys_for_set(task_set: str, stats: dict) -> list[str]:
    if task_set == "c2_core":
        return list(C2_CORE_KEYS)
    if task_set == "ts2":
        return list(TS2_KEYS)
    if task_set == "ts2_singles":
        return list(TS2_SINGLE_KEYS)
    if task_set == "l6a_quiet_sidewalk":
        return ["noise_0+surface_2"]
    if task_set == "c2_full":
        return usable_conjunction_keys(stats) + list(ATOM_KEYS)
    raise ValueError(f"unknown c2 task set {task_set!r}")


def held_out_keys_for_set(task_set: str) -> tuple[str, ...]:
    return C2_FULL_HELD_OUT if task_set == "c2_full" else ()


def train_task_indices(keys: list[str], held_out: tuple[str, ...]) -> list[int] | None:
    if not held_out:
        return None
    hold = set(held_out) & set(keys)
    return [i for i, k in enumerate(keys) if k not in hold]


def fg_ce_weight(key: str, stats: dict, *, alpha: float = 0.5) -> float:
    """Dampened balanced weight w = ((1-f)/f)**alpha (balancing memo §5). No clamp:
    alpha < 1 already keeps w small, and the decision rule undoes whatever shift it causes."""
    f = positive_share(key, stats)
    return float(((1.0 - f) / max(f, 1e-9)) ** alpha)


def fg_ce_weights_for_keys(keys: list[str], stats: dict, *, alpha: float = 0.5) -> tuple[float, ...]:
    return tuple(fg_ce_weight(k, stats, alpha=alpha) for k in keys)


def decision_threshold(w: float) -> float:
    """Unbiased cut-off for a weighted-CE model: P(fg) > w/(1+w) (balancing memo §3)."""
    return w / (1.0 + w)
