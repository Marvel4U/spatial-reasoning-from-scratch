"""Experiment specs + RUN queue. Builders: experiment_sweeps.py."""

from __future__ import annotations

from typing import Any

import config
from experiment_sweeps import _patch_geometry, as_single, build_c0_worldsnap_p_sweep

SCHEMA_VERSION = 1
RESULTS_PATH = config.RESULTS_PATH


# ── E1 patch sweep (LEARNING_REPORT §3.5) ─────────────────────────────────
# P=4 → s=1 (no in-patch ambiguity). P=16 → s=4 (blob regime).
# batch_size: lower P → more tokens (P=4 → N=4096); tune if OOM on 3060.

C0_WORLDSNAP_P_SWEEP: dict[str, Any] = {
    "patch_sizes": [4, 8, 16],
    "id_template": "c0_worldsnap_P{p}_2k",
    "base": {
        "train_steps": 2000,
        "overrides": {
            "data_source": "worldsnap",
            "cropset": "v2",
            "taskset": "t0_point_v0",
            "worldsnap_max_train_items": None,
            "worldsnap_max_val_items": None,
            "val_same_as_train": False,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "worldsnap_cache_layers": True,
            "worldsnap_gpu_resident": True,
        },
        "spatial_eval": {"max_batches": 20, "train_snap_batches": 3},
    },
    "patch_overrides": {
        4: {"batch_size": 4},
        8: {"batch_size": 8},
        16: {"batch_size": 16},
    },
}


C0_WORLDSNAP_BASE_2K: dict[str, Any] = {
    "train_steps": 2000,
    "overrides": {
        "data_source": "worldsnap",
        "cropset": "v2",
        "taskset": "t0_point_v0",
        "worldsnap_max_train_items": None,
        "worldsnap_max_val_items": None,
        "val_same_as_train": False,
        "batch_size": 16,
        "patch_size": 16,
        "learning_rate": 1e-3,
        "eval_interval": 500,
        "worldsnap_cache_layers": True,
        "worldsnap_gpu_resident": True,
    },
    "spatial_eval": {"max_batches": 20, "train_snap_batches": 3},
}


# ── RUN: open list — add/remove lines; each entry is independent ───────────
def c0_p16_2k_diagnostics_perf() -> dict:
    """P16 full pool, 2k steps, tier1+2 diagnostics, post–perf-plan harness defaults."""
    spec = build_c0_worldsnap_p_sweep({**C0_WORLDSNAP_P_SWEEP, "patch_sizes": [16]})[0]
    spec["id"] = "c0_worldsnap_P16_2k"
    spec["notes"] = (
        "P16 2k + diagnostics + gpu_resident; compare plot 7 to pre-perf run "
        "(archive old runs/diagnostics/c0_worldsnap_P16_2k* as …_preperf first)"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "diagnostics_enabled": True,
        "worldsnap_gpu_resident": True,
        "eval_interval": 500,
    }
    return spec


def c1_l5a_building_2k() -> dict:
    """L5a: mark building only, no task embedding, dense mask plumbing."""
    return {
        "id": "c1_L5a_building_2k",
        "train_steps": 2000,
        "notes": "C1_PLAN L5a: surface building majority mask, marker=0, plain CE",
        "overrides": {
            "task_rung": "c1",
            "c1_task_key": "building",
            "balanced_ce": False,
            "data_source": "worldsnap",
            "cropset": "v2",
            "worldsnap_max_train_items": None,
            "worldsnap_max_val_items": None,
            "val_same_as_train": False,
            "batch_size": 16,
            "patch_size": 16,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "worldsnap_cache_layers": True,
            "worldsnap_gpu_resident": True,
            "diagnostics_enabled": True,
        },
        "spatial_eval": {"max_batches": 20, "train_snap_batches": 3},
    }


def c1_l5c_factorised_4k() -> dict:
    """Continue L5c from 2k checkpoint to 4000 steps (same data/hyperparams)."""
    ckpt = config.REPO_ROOT / "runs" / "checkpoints" / "c1_L5c_factorised_2k.pt"
    spec = c1_l5c_factorised_2k()
    spec["id"] = "c1_L5c_factorised_4k"
    spec["train_steps"] = 4000
    spec["init_checkpoint"] = str(ckpt)
    spec["load_optimizer"] = True
    spec["reset_step_counter"] = False
    spec["notes"] = "L5c factorised: warm-start from c1_L5c_factorised_2k @2k, train to 4k"
    return spec


def c1_l5c_factorised_2k() -> dict:
    """L5c: 12 atomic (layer, class) tasks, factorised emb, 2 held-out pairs (C1_PLAN §5c, §10)."""
    return {
        "id": "c1_L5c_factorised_2k",
        "train_steps": 2000,
        "notes": "C1 L5c: layer_emb+class_emb, train 10/12, hold out surface_0+estab_2",
        "overrides": {
            "task_rung": "c1",
            "c1_task_mode": "factorised_twelve",
            "c1_held_out_task_keys": ("surface_0", "estab_2"),
            "balanced_ce": False,
            "data_source": "worldsnap",
            "cropset": "v2",
            "worldsnap_max_train_items": None,
            "worldsnap_max_val_items": None,
            "val_same_as_train": False,
            "batch_size": 16,
            "patch_size": 16,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "worldsnap_cache_layers": True,
            "worldsnap_gpu_resident": True,
            "c1_val_pairs": 512,
            "c1_ce_weight_clamp": 100.0,
            "diagnostics_enabled": True,
        },
        "spatial_eval": {"max_batches": 20, "train_snap_batches": 3},
    }


def _with_cropset(spec: dict, cropset: str, *, id_suffix: str) -> dict:
    out = {**spec, "id": f"{spec['id']}_{id_suffix}"}
    out["overrides"] = {**spec["overrides"], "cropset": cropset}
    out["notes"] = f"{spec['notes']} [{cropset} estab footprints]"
    return out


def c1_l5b_core_four_2k_v3b() -> dict:
    """L5b on v3b (footprint + 1000 m² cap); majority estab + memo §4 CE weights."""
    return _with_cropset(c1_l5b_core_four_2k(), "v3b", id_suffix="v3b")


def c1_l5b_core_four_2k_v4_onehot_P16() -> dict:
    """C3 step 3: L5b four core on v4 RGB planes (same as v3b), one-hot P16."""
    spec = c1_l5b_core_four_2k_v3b()
    spec["id"] = "c1_L5b_core_four_2k_v4_onehot_P16"
    spec["overrides"] = {
        **spec["overrides"],
        "cropset": "v4",
        "encoding_mode": "onehot",
        "diagnostics_enabled": False,
    }
    spec["notes"] = "C3 regression: L5b core four on v4, one-hot P16 (expect ≈ v3b onehot ~0.97)"
    spec["spatial_eval"] = {"max_batches": None, "train_snap_batches": 3}
    return spec


def c1_l5c_factorised_2k_v3b() -> dict:
    return _with_cropset(c1_l5c_factorised_2k(), "v3b", id_suffix="v3b")


def c1_l5b_core_four_2k() -> dict:
    """L5b: four core tasks + task_emb + fixed per-task CE weights (C1_PLAN §7)."""
    return {
        "id": "c1_L5b_core_four_2k",
        "train_steps": 2000,
        "notes": "C1 L5b: building/sidewalk/noise_ge_65/food_drink, task_emb, per-task CE",
        "overrides": {
            "task_rung": "c1",
            "c1_task_mode": "core_four",
            "balanced_ce": False,
            "data_source": "worldsnap",
            "cropset": "v2",
            "worldsnap_max_train_items": None,
            "worldsnap_max_val_items": None,
            "val_same_as_train": False,
            "batch_size": 16,
            "patch_size": 16,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "worldsnap_cache_layers": True,
            "worldsnap_gpu_resident": True,
            "c1_val_pairs": 512,
            "c1_ce_weight_clamp": 100.0,
            "diagnostics_enabled": True,
        },
        "spatial_eval": {"max_batches": 20, "train_snap_batches": 3},
    }


def c1_l5b_clamp_sweep(clamps=(10.0, 30.0)) -> list[dict]:
    """L5b with a lower positive-weight clamp. Threshold sweep on the clamp=100 checkpoint
    (docs/plans/perf_prototypes/c1_threshold_sweep.py) showed food_drink IoU 0.48 at P>0.5 but 0.66 at P>0.95:
    the large weight biases the model to over-mark. Same recipe as c1_l5b_core_four_2k, only the clamp differs."""
    out = []
    for c in clamps:
        spec = c1_l5b_core_four_2k()
        spec["id"] = f"c1_L5b_core_four_2k_clamp{int(c)}"
        spec["notes"] = f"C1 L5b, fg CE weight clamp {c:g} (was 100): blob / over-marking test"
        spec["overrides"] = dict(spec["overrides"], c1_ce_weight_clamp=float(c))
        out.append(spec)
    return out


def _c2_v3b_2k(**override) -> dict:
    """Shared c2 worldsnap recipe (C2_PLAN §2, §7); one-hot + cond_tokens unless L6a."""
    overrides: dict[str, Any] = {
        "task_rung": "c2",
        "data_source": "worldsnap",
        "cropset": "v3b",
        "encoding_mode": "onehot",
        "batch_size": 16,
        "patch_size": 16,
        "learning_rate": 1e-3,
        "eval_interval": 500,
        "c2_train_source": "fixed_crops",
        "c2_weight_alpha": 0.5,
        "c2_held_out": False,
        "c2_no_task_input": False,
        "balanced_ce": False,
        "diagnostics_enabled": True,
    }
    overrides.update(override)
    return {
        "train_steps": 2000,
        "force_restart": False,
        "overrides": overrides,
        "spatial_eval": {"max_batches": None, "train_snap_batches": 3},
    }


def c2_gate_ts2_singles_2k() -> dict:
    """§4 gate: four ts2 singles through cond_tokens; mean rule IoU ≈ one-hot L5b (~0.97)."""
    spec = _c2_v3b_2k(c2_task_set="ts2_singles")
    spec["id"] = "c2_gate_ts2_singles_2k"
    spec["notes"] = "C2 §4 gate: ts2 singles only, cond_tokens, full val grid"
    return spec


def _c3_step4_quieter_sidewalk_2k(c3_task_key: str) -> dict:
    """C3_PLAN §5 step 4: single relative rule on v4 noise7, compare to local ceiling."""
    return {
        "train_steps": 2000,
        "force_restart": False,
        "overrides": {
            "task_rung": "c3",
            "data_source": "worldsnap",
            "cropset": "v4",
            "encoding_mode": "onehot_v4",
            "c3_task_key": c3_task_key,
            "c3_weight_alpha": 0.5,
            "batch_size": 16,
            "patch_size": 16,
            "learning_rate": 1e-3,
            "eval_interval": 500,
            "balanced_ce": False,
            "diagnostics_enabled": True,
        },
        "spatial_eval": {"max_batches": None, "train_snap_batches": 3},
    }


def c3_quieter_sidewalk_median_strict_2k() -> dict:
    spec = _c3_step4_quieter_sidewalk_2k("median_strict")
    spec["id"] = "c3_quieter_sidewalk_median_strict_2k"
    spec["notes"] = "C3 step 4: median_strict rule, local ceiling ~0.762"
    return spec


def c3_quieter_sidewalk_quietest_tertile_2k() -> dict:
    spec = _c3_step4_quieter_sidewalk_2k("quietest_tertile")
    spec["id"] = "c3_quieter_sidewalk_quietest_tertile_2k"
    spec["notes"] = "C3 step 4: quietest_tertile rule, local ceiling ~0.588"
    return spec


def c3_within_20m_fixes() -> list[dict]:
    """within_20m collapsed to all-background (IoU 0.0): the batch is correct (probe 23 Sep), the
    single marker pixel is starved (c0 mechanism) and alpha=0.5 makes 'nothing' cheap (w ~ 7).
    Two independent fixes, run separately: full inverse-frequency weight; a 3 px marker disc."""
    a = c3_within_20m_2k(); a["id"] = "c3_within_20m_2k_alpha1"; a["notes"] = "within_20m, alpha=1.0 (w ~ 50)"
    a["overrides"] = {**a["overrides"], "c3_weight_alpha": 1.0}
    b = c3_within_20m_2k(); b["id"] = "c3_within_20m_2k_disc3"; b["notes"] = "within_20m, marker disc radius 3 px, alpha=0.5"
    b["overrides"] = {**b["overrides"], "marker_radius_px": 3}
    return [a, b]


def c3_within_20m_multimarker() -> list[dict]:
    """Marvin (23 Sep): K random markers per crop, >= 40 m apart, target = union of K 20 m discs.
    Hypothesis: K x the marker signal per sample and K x the positive share make the collapse
    less likely and training faster. K=1 random isolates 'fresh markers' from 'more markers'.
    Val stays the fixed single-marker set (comparable to c3_within_20m_2k)."""
    out = []
    for k, tag in ((1, "rand1"), (4, "rand4")):
        spec = c3_within_20m_2k(); spec["id"] = f"c3_within_20m_2k_{tag}"
        spec["notes"] = f"within_20m, {k} random marker(s) per batch, min dist 40 m, alpha 0.5"
        spec["overrides"] = {**spec["overrides"], "c3_n_markers": k, "c3_marker_min_dist_m": 40.0}
        out.append(spec)
    d = c3_within_20m_2k(); d["id"] = "c3_within_20m_2k_rand4_disc3"
    d["notes"] = "within_20m, 4 random markers + 3 px marker disc"
    d["overrides"] = {**d["overrides"], "c3_n_markers": 4, "c3_marker_min_dist_m": 40.0, "marker_radius_px": 3}
    out.append(d)
    return out


def c3_within_20m_2k() -> dict:
    """C3 §5 step 6: marker-relation disk from t0_point_v0 markers."""
    spec = _c3_step4_quieter_sidewalk_2k("within_20m")
    spec["id"] = "c3_within_20m_2k"
    spec["notes"] = "C3 step 6: within 20 m of marker (marker plane + cond token)"
    return spec


def _c3_within_multimarker(
    task_key: str,
    *,
    train_steps: int = 2000,
    run_id: str | None = None,
) -> dict:
    """N4: radius-specific cond token; K=4 @ 20 m, K=2 @ 50 m, K=1 @ 75/100 m on train.

    Default run_id is ``{task_key}_rand{K}`` so raising ``train_steps`` auto-resumes
    (``harness_auto_resume``). Pass ``run_id`` to continue an older id (e.g. 6k → 8k).
    """
    from spatial_data.c3_tasks import get_c3_task, within_train_markers

    spec_obj = get_c3_task(task_key)
    k, min_dist = within_train_markers(spec_obj.radius_m)
    spec = _c3_step4_quieter_sidewalk_2k(task_key)
    spec["train_steps"] = train_steps
    spec["id"] = run_id or f"{task_key}_rand{k}"
    spec["notes"] = (
        f"N4 {task_key}: cond REL id {spec_obj.relative_cond_id}, "
        f"train {k} markers min {min_dist}m, val single t0 marker, {train_steps} steps"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "c3_n_markers": k,
        "c3_marker_min_dist_m": min_dist,
    }
    spec["force_restart"] = False
    return spec


def _c3_within_8k_multimarker(task_key: str, *, run_id: str | None = None) -> dict:
    return _c3_within_multimarker(task_key, train_steps=8000, run_id=run_id)


C3_MARKER_DISC3_PX = 3


def _with_marker_disc3(spec: dict) -> dict:
    """3 px marker plane on input (C3_PLAN §5b); val still fixed t0 point, drawn as disc."""
    out = dict(spec)
    rid = out["id"]
    if not rid.endswith("_disc3"):
        out["id"] = f"{rid}_disc3"
    out["overrides"] = {
        **out.get("overrides", {}),
        "marker_radius_px": C3_MARKER_DISC3_PX,
        "diagnostics_enabled": False,
    }
    note = f"marker plane {C3_MARKER_DISC3_PX}px disc on input"
    out["notes"] = f"{out.get('notes', '').rstrip()}; {note}".lstrip("; ")
    out["force_restart"] = False
    return out


def _with_rel_pos_v2(spec: dict) -> dict:
    """E5b V2: learned 2-D rel bias in every encoder block."""
    out = dict(spec)
    rid = out["id"]
    if not rid.endswith("_relV2"):
        out["id"] = f"{rid}_relV2"
    out["overrides"] = {**out.get("overrides", {}), "rel_pos_bias": "all"}
    out["notes"] = f"{out.get('notes', '').rstrip()}; rel_pos_bias=all (E5b V2)".lstrip("; ")
    out["force_restart"] = False
    return out


def c3_within_50m_8k_disc3() -> dict:
    return _with_marker_disc3(_c3_within_multimarker("within_50m", train_steps=8000))


def c3_within_75m_8k_disc3() -> dict:
    return _with_marker_disc3(_c3_within_multimarker("within_75m", train_steps=8000))


def c3_within_100m_8k_disc3() -> dict:
    return _with_marker_disc3(_c3_within_multimarker("within_100m", train_steps=8000))


def c3_m1_food_drink_within_100m_8k_disc3() -> dict:
    return _with_marker_disc3(c3_m1_food_drink_within_100m_8k())


def c3_m1_food_drink_within_100m_L4_8k_disc3() -> dict:
    return _with_marker_disc3(c3_m1_food_drink_within_100m_L4_8k())


def c3_m1_food_drink_within_100m_L4_8k_disc3_relV2() -> dict:
    """E8 m1: L4 b32 8k, 3px marker disc + rel_pos_bias=all (best m0 stack on conjunctive task)."""
    return _with_rel_pos_v2(_with_marker_disc3(c3_m1_food_drink_within_100m_L4_8k()))


def c3_m1_food_drink_within_100m_L4_16k_disc3_relV2() -> dict:
    """Fresh 0→16k m1 L4 disc3+relV2 with full diagnostics (pred strips + eval.jsonl each CE eval)."""
    spec = c3_m1_food_drink_within_100m_L4_8k_disc3_relV2()
    spec["train_steps"] = 16000
    spec["id"] = "food_drink_within_100m_L4_16k_b32_disc3_relV2"
    spec["force_restart"] = True
    spec.pop("init_checkpoint", None)
    spec.pop("load_optimizer", None)
    spec.pop("reset_step_counter", None)
    spec["notes"] = (
        "m1 L4 disc3+relV2: fresh 16k, batch 32, eval/1k, diagnostics on (learning trace)"
    )
    spec["overrides"] = {**spec["overrides"], "diagnostics_enabled": True}
    return spec


# ── M1 report overnight batch (docs/reports/M1_FOOD_DRINK_WITHIN_100M_2026-09-24.md §6) ──

_M1_REPORT_LR_HORIZON_8K = 8000
_M1_REPORT_LR_FLAT_HORIZON = 50_000_000  # ~constant LR over 8k (cosine decay negligible)


def _m1_report_overrides(
    *,
    rel_pos_bias: str = "all",
    enc_n_layer: int = 4,
    batch_size: int = 32,
    c3_n_markers: int | None = None,
    c3_marker_min_dist_m: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    from spatial_data.c3_tasks import within_train_markers

    k, min_dist = within_train_markers(100)
    if c3_n_markers is not None:
        k = c3_n_markers
    if c3_marker_min_dist_m is not None:
        min_dist = c3_marker_min_dist_m
    return {
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": k,
        "c3_marker_min_dist_m": min_dist,
        "enc_n_layer": enc_n_layer,
        "batch_size": batch_size,
        "eval_interval": 1000,
        "rel_pos_bias": rel_pos_bias,
        "marker_radius_px": C3_MARKER_DISC3_PX,
        "diagnostics_enabled": True,
        **extra,
    }


def _m1_report_spec(
    *,
    run_id: str,
    notes: str,
    train_steps: int = 8000,
    lr_horizon_steps: int = _M1_REPORT_LR_HORIZON_8K,
    c3_task_key: str | None = None,
    seed: int | None = None,
    init_checkpoint: str | None = None,
    load_optimizer: bool = False,
    reset_step_counter: bool = True,
    force_restart: bool = True,
    override_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from spatial_data.c3_tasks import M1_TASK_KEY

    spec = _c3_step4_quieter_sidewalk_2k(c3_task_key or M1_TASK_KEY)
    spec["train_steps"] = train_steps
    spec["id"] = run_id
    spec["lr_horizon_steps"] = lr_horizon_steps
    spec["force_restart"] = force_restart
    spec["notes"] = notes
    if init_checkpoint:
        spec["init_checkpoint"] = init_checkpoint
        spec["load_optimizer"] = load_optimizer
        spec["reset_step_counter"] = reset_step_counter
    else:
        spec.pop("init_checkpoint", None)
        spec.pop("load_optimizer", None)
        spec.pop("reset_step_counter", None)
    ov = _m1_report_overrides(**(override_extra or {}))
    if seed is not None:
        ov["seed"] = seed
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def m1_report_S_seed(seed: int) -> dict:
    """§6.1: run-11 recipe, alternate seed (escape step + final IoU)."""
    return _m1_report_spec(
        run_id=f"food_drink_within_100m_L4_8k_b32_disc3_relV2_seed{seed}",
        seed=seed,
        notes=f"M1 report §6.1 S{seed}: L4 disc3+relV2 8k, lr_horizon=8k, diagnostics on, seed={seed}",
    )


def m1_report_A1_disc_only(seed: int) -> dict:
    """§6.2 A1: disc marker, no rel bias (2×2 missing corner)."""
    return _m1_report_spec(
        run_id=f"food_drink_within_100m_L4_8k_b32_disc3_seed{seed}",
        seed=seed,
        notes=f"M1 report §6.2 A1: L4 disc3 only (no rel bias), seed={seed}",
        override_extra={"rel_pos_bias": "none"},
    )


def m1_report_A2_disc_relV1() -> dict:
    """§6.2 A2 optional: disc + rel bias first block only."""
    return _m1_report_spec(
        run_id="food_drink_within_100m_L4_8k_b32_disc3_relV1",
        notes="M1 report §6.2 A2: L4 disc3 + rel_pos_bias=first (V1)",
        override_extra={"rel_pos_bias": "first"},
    )


def m1_report_L2a_seed(seed: int) -> dict:
    """§6.3 L2-a: depth control vs run 11 (L2, batch 32)."""
    return _m1_report_spec(
        run_id=f"food_drink_within_100m_L2_8k_b32_disc3_relV2_seed{seed}",
        seed=seed,
        notes=f"M1 report §6.3 L2-a: L2/H6 b32, disc3+relV2, seed={seed}",
        override_extra={"enc_n_layer": 2},
    )


def m1_report_L2b_k4() -> dict:
    """§6.3 L2-b: K=4 train markers, batch 16."""
    return _m1_report_spec(
        run_id="food_drink_within_100m_L2_8k_b16_disc3_relV2_k4",
        notes="M1 report §6.3 L2-b: L2 b16, K=4 markers min 40m, disc3+relV2",
        override_extra={
            "enc_n_layer": 2,
            "batch_size": 16,
            "c3_n_markers": 4,
            "c3_marker_min_dist_m": 40.0,
        },
    )


def m1_report_L2c_k4() -> dict:
    """§6.3 L2-c: K=4, batch 32."""
    return _m1_report_spec(
        run_id="food_drink_within_100m_L2_8k_b32_disc3_relV2_k4",
        notes="M1 report §6.3 L2-c: L2 b32, K=4 markers min 40m, disc3+relV2",
        override_extra={
            "enc_n_layer": 2,
            "c3_n_markers": 4,
            "c3_marker_min_dist_m": 40.0,
        },
    )


def m1_report_C1_lr_flat_1e3() -> dict:
    """§6.4 C1: flat ~1e-3 LR (no meaningful cosine decay over 8k)."""
    return _m1_report_spec(
        run_id="food_drink_within_100m_L4_8k_b32_disc3_relV2_lrflat1e3",
        lr_horizon_steps=_M1_REPORT_LR_FLAT_HORIZON,
        notes="M1 report §6.4 C1: L4 disc3+relV2, constant ~1e-3 LR, 8k",
    )


def m1_report_C2_lr_flat_3e4() -> dict:
    """§6.4 C2: flat ~3e-4 LR."""
    return _m1_report_spec(
        run_id="food_drink_within_100m_L4_8k_b32_disc3_relV2_lrflat3e4",
        lr_horizon_steps=_M1_REPORT_LR_FLAT_HORIZON,
        notes="M1 report §6.4 C2: L4 disc3+relV2, constant ~3e-4 LR, 8k",
        override_extra={"learning_rate": 3e-4},
    )


def _m1_report_mix_spec(
    *,
    run_id: str,
    train_steps: int,
    lr_horizon_steps: int,
    weights: tuple[float, float, float],
    notes: str,
    seed: int | None = None,
    init_checkpoint: str | None = None,
    mix_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    extra: dict[str, Any] = {
        "c3_task_mode": "multi",
        "c3_task_keys": list(N5_MIX_TASK_KEYS),
        "c3_eval_task_key": M1_TASK_KEY,
        "c3_train_task_weights": list(weights),
        "c3_n_markers": 0,
    }
    if mix_override:
        extra.update(mix_override)
    return _m1_report_spec(
        run_id=run_id,
        train_steps=train_steps,
        lr_horizon_steps=lr_horizon_steps,
        notes=notes,
        seed=seed,
        init_checkpoint=init_checkpoint,
        load_optimizer=False,
        reset_step_counter=True,
        override_extra=extra,
    )


def m1_report_M1_mix_uniform_16k() -> dict:
    """§6.5 M1: ⅓ mix, 16k, horizon=16k."""
    w = (1 / 3, 1 / 3, 1 / 3)
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix33_L4_16k_b32_disc3_relV2",
        train_steps=16000,
        lr_horizon_steps=16000,
        weights=w,
        notes="M1 report §6.5 M1: 3-task mix ⅓ each, L4 disc3+relV2, 16k, diagnostics on",
    )


def m1_report_M2_mix80_16k() -> dict:
    """§6.5 M2: 80/10/10 toward m1."""
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix80_L4_16k_b32_disc3_relV2",
        train_steps=16000,
        lr_horizon_steps=16000,
        weights=(0.1, 0.1, 0.8),
        notes="M1 report §6.5 M2: mix 80/10/10 m1, L4 disc3+relV2, 16k",
    )


def m1_report_M3_mix_uniform_24k() -> dict:
    """§6.5 M3: ⅓ mix, 24k."""
    w = (1 / 3, 1 / 3, 1 / 3)
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix33_L4_24k_b32_disc3_relV2",
        train_steps=24000,
        lr_horizon_steps=24000,
        weights=w,
        notes="M1 report §6.5 M3: 3-task mix ⅓, L4 disc3+relV2, 24k",
    )


def m1_report_M4_phase1_within_100m_3k() -> dict:
    """§6.5 M4 phase 1: within_100m only, 3k (build disc head before mix)."""
    return _m1_report_spec(
        run_id="m1_M4_within100m_L4_3k_disc3_relV2",
        c3_task_key="within_100m",
        train_steps=3000,
        lr_horizon_steps=8000,
        notes="M1 report §6.5 M4 phase1: within_100m L4 disc3+relV2, 3k → ckpt for M4 phase2",
        override_extra={"c3_task_mode": "single"},
    )


def m1_report_M4_phase2_mix_13k() -> dict:
    """§6.5 M4 phase 2: 3-task mix 13k, warm-start from phase1 weights (optimizer reset)."""
    ckpt = config.REPO_ROOT / "runs" / "checkpoints" / "m1_M4_within100m_L4_3k_disc3_relV2.pt"
    w = (1 / 3, 1 / 3, 1 / 3)
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix33_L4_13k_b32_disc3_relV2_M4",
        train_steps=13000,
        lr_horizon_steps=13000,
        weights=w,
        init_checkpoint=str(ckpt),
        notes="M1 report §6.5 M4 phase2: ⅓ mix 13k, init M4 phase1 ckpt, fresh step counter",
    )


_M1_MIX33_W = (1 / 3, 1 / 3, 1 / 3)
_M1_MIX80_W = (0.1, 0.1, 0.8)


def m1_report_79_mix33_8k_seed(seed: int) -> dict:
    """§7.9.1: uniform ⅓ mix, 8k steps, lr horizon 8k (mix reproducibility + vs 16k reference)."""
    return _m1_report_mix_spec(
        run_id=f"food_drink_within_100m_mix33_L4_8k_b32_disc3_relV2_seed{seed}",
        train_steps=8000,
        lr_horizon_steps=8000,
        weights=_M1_MIX33_W,
        seed=seed,
        notes=f"M1 report §7.9.1: ⅓ mix L4 8k, lr_horizon=8k, seed={seed}",
    )


def m1_report_79_mix80_8k_seed(seed: int) -> dict:
    """§7.9.1: 80/10/10 mix, 8k steps, lr horizon 8k."""
    return _m1_report_mix_spec(
        run_id=f"food_drink_within_100m_mix80_L4_8k_b32_disc3_relV2_seed{seed}",
        train_steps=8000,
        lr_horizon_steps=8000,
        weights=_M1_MIX80_W,
        seed=seed,
        notes=f"M1 report §7.9.1: mix 80/10/10 L4 8k, lr_horizon=8k, seed={seed}",
    )


def m1_report_79_mix33_8k_lrflat_1e3() -> dict:
    """§7.9.2: ⅓ mix, ~constant 1e-3 LR, 8k."""
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix33_L4_8k_b32_disc3_relV2_lrflat1e3",
        train_steps=8000,
        lr_horizon_steps=_M1_REPORT_LR_FLAT_HORIZON,
        weights=_M1_MIX33_W,
        notes="M1 report §7.9.2: ⅓ mix L4 8k, flat ~1e-3 LR",
    )


def m1_report_79_mix33_8k_lrflat_3e4() -> dict:
    """§7.9.2: ⅓ mix, ~constant 3e-4 LR, 8k."""
    return _m1_report_mix_spec(
        run_id="food_drink_within_100m_mix33_L4_8k_b32_disc3_relV2_lrflat3e4",
        train_steps=8000,
        lr_horizon_steps=_M1_REPORT_LR_FLAT_HORIZON,
        weights=_M1_MIX33_W,
        notes="M1 report §7.9.2: ⅓ mix L4 8k, flat ~3e-4 LR",
        mix_override={"learning_rate": 3e-4},
    )


def m1_report_79_mix33_L2_16k_seed(seed: int) -> dict:
    """§7.9.3: ⅓ mix, 2 encoder blocks, 16k, lr horizon 16k."""
    return _m1_report_mix_spec(
        run_id=f"food_drink_within_100m_mix33_L2_16k_b32_disc3_relV2_seed{seed}",
        train_steps=16000,
        lr_horizon_steps=16000,
        weights=_M1_MIX33_W,
        seed=seed,
        notes=f"M1 report §7.9.3: ⅓ mix L2/H6 b32 16k, seed={seed}",
        mix_override={"enc_n_layer": 2},
    )


def m1_report_79_run_queue() -> list[dict]:
    """M1 report §7.9 (24 Sep): mix seeds @ 8k, flat LR mix, L2 mix @ 16k. Seeds 4/5 avoid §7.1–7.3 single-task seeds 1–3."""
    return [
        m1_report_79_mix33_8k_seed(4),
        m1_report_79_mix33_8k_seed(5),
        m1_report_79_mix80_8k_seed(4),
        m1_report_79_mix80_8k_seed(5),
        m1_report_79_mix33_8k_lrflat_1e3(),
        m1_report_79_mix33_8k_lrflat_3e4(),
        m1_report_79_mix33_L2_16k_seed(4),
        m1_report_79_mix33_L2_16k_seed(5),
    ]


def m1_report_overnight_run_queue() -> list[dict]:
    """Full §6 queue (excludes §6.6 ceiling / marker-token / rare conj). Order: seeds → ablations → L2 → LR → mixes → M4."""
    return [
        m1_report_S_seed(1),
        m1_report_S_seed(2),
        m1_report_S_seed(3),
        m1_report_A1_disc_only(1),
        m1_report_A1_disc_only(2),
        m1_report_A2_disc_relV1(),
        m1_report_L2a_seed(1),
        m1_report_L2a_seed(2),
        m1_report_L2b_k4(),
        m1_report_L2c_k4(),
        m1_report_C1_lr_flat_1e3(),
        m1_report_C2_lr_flat_3e4(),
        m1_report_M1_mix_uniform_16k(),
        m1_report_M2_mix80_16k(),
        m1_report_M3_mix_uniform_24k(),
        m1_report_M4_phase1_within_100m_3k(),
        m1_report_M4_phase2_mix_13k(),
    ]


def c3_n5_m1_mix_80_20_8k() -> dict:
    """Mix 80/20 @ L2/H6, batch 16, 8k (N4-style recipe, weighted sampling)."""
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_mix80_8k"
    spec["notes"] = (
        "Mix 80/20: m1 80%, aux 20%, L2 b16, 8k, CE+Dice, eval/500"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "c3_task_mode": "multi",
        "c3_task_keys": list(N5_MIX_TASK_KEYS),
        "c3_eval_task_key": M1_TASK_KEY,
        "c3_train_task_weights": [0.1, 0.1, 0.8],
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": 0,
    }
    return spec


def c3_n5_m1_mix_80_20_8k_disc3() -> dict:
    return _with_marker_disc3(c3_n5_m1_mix_80_20_8k())


def c3_within_50m_8k_disc3_relV2() -> dict:
    return _with_rel_pos_v2(c3_within_50m_8k_disc3())


def c3_within_75m_8k_disc3_relV2() -> dict:
    return _with_rel_pos_v2(c3_within_75m_8k_disc3())


def c3_within_100m_8k_disc3_relV2() -> dict:
    return _with_rel_pos_v2(c3_within_100m_8k_disc3())


def c3_m1_food_drink_within_100m_8k_disc3_relV2() -> dict:
    return _with_rel_pos_v2(c3_m1_food_drink_within_100m_8k_disc3())


def c3_n5_m1_mix_80_20_8k_disc3_relV2() -> dict:
    return _with_rel_pos_v2(c3_n5_m1_mix_80_20_8k_disc3())


def c3_n4_within_smoke_runs() -> list[dict]:
    """Short N4 ladder: 50 m first (2k); 75/100 m only after 50 m passes."""
    return [_c3_within_multimarker("within_50m", train_steps=2000)]


def c3_n4_within_8k_runs() -> list[dict]:
    """N4 @ 8k: 50 m resumes ``within_50m_6k_rand2`` (train K=2); 75/100 m train K=1."""
    return [
        _c3_within_multimarker("within_50m", train_steps=8000, run_id="within_50m_6k_rand2"),
        _c3_within_multimarker("within_75m", train_steps=8000),
        _c3_within_multimarker("within_100m", train_steps=8000),
    ]


def c3_n4_within_radius_runs() -> list[dict]:
    return c3_n4_within_8k_runs()


def _c3_within_50m_4k_fast(*, tag: str, **config_overrides) -> dict:
    """50 m m0, 4k steps, no diagnostics, sparse CE + capped end IoU (arch sweeps)."""
    spec = _c3_within_multimarker("within_50m", train_steps=4000)
    spec["id"] = f"within_50m_4k_rand2_{tag}"
    spec["force_restart"] = False
    spec["notes"] = (
        f"50m arch sweep ({tag}): enc_n_layer={config_overrides.get('enc_n_layer', 2)}, "
        f"h_heads={config_overrides.get('h_heads', 6)}; 4k, diagnostics off, fast eval"
    )
    spec["spatial_eval"] = {"max_batches": 15, "train_snap_batches": 0}
    spec["overrides"] = {
        **spec["overrides"],
        "diagnostics_enabled": False,
        "eval_interval": 1000,
        "eval_iters": 5,
        "eval_iters_final": 10,
        **config_overrides,
    }
    return spec


def c3_within_50m_arch_sweep_4k() -> list[dict]:
    """Baseline depth: L2/H6. Variants: +1 layer, +2 heads (8), both."""
    return [
        _c3_within_50m_4k_fast(tag="L3", enc_n_layer=3),
        _c3_within_50m_4k_fast(tag="H8", h_heads=8),
        _c3_within_50m_4k_fast(tag="L3H8", enc_n_layer=3, h_heads=8),
    ]


def c3_within_50m_4k_lr3x() -> dict:
    """L2/H6 @ 50 m, same fast 4k recipe as arch sweep, learning_rate = 3e-3."""
    spec = _c3_within_50m_4k_fast(tag="lr3x", learning_rate=3e-3)
    spec["notes"] = "50m L2/H6, lr 3e-3 (3× default), 4k, diagnostics off, fast eval"
    return spec


def _c3_rel_pos_variant(
    task_key: str,
    *,
    rel_pos_bias: str,
    train_steps: int,
    run_id: str,
    notes: str,
    **extra_overrides,
) -> dict:
    spec = _c3_within_multimarker(task_key, train_steps=train_steps, run_id=run_id)
    spec["notes"] = notes
    spec["overrides"] = {**spec["overrides"], "rel_pos_bias": rel_pos_bias, **extra_overrides}
    spec["force_restart"] = False
    return spec


def c3_rel_pos_smoke() -> dict:
    """E5b: ~150 steps, within_75m K=1, rel bias in block 0 — plumbing + loss decrease."""
    spec = _c3_within_multimarker("within_75m", train_steps=150, run_id="within_75m_relV1_smoke")
    spec["notes"] = "E5b smoke: within_75m K=1, rel_pos_bias=first, 150 steps, light eval"
    spec["spatial_eval"] = {"max_batches": 5, "train_snap_batches": 0}
    spec["overrides"] = {
        **spec["overrides"],
        "rel_pos_bias": "first",
        "diagnostics_enabled": False,
        "eval_interval": 50,
        "eval_iters": 3,
        "eval_iters_final": 5,
    }
    return spec


def c3_rel_pos_e5b_v1_within_75m() -> dict:
    """E5b V1: slow IoU baseline vs rel bias in encoder block 0 only (C3_PLAN §8.4)."""
    return _c3_rel_pos_variant(
        "within_75m",
        rel_pos_bias="first",
        train_steps=8000,
        run_id="within_75m_rand1_relV1",
        notes="E5b V1: within_75m K=1 train, rel_pos_bias=first, 8k, full diagnostics",
    )


def c3_rel_pos_e5b_v1_within_20m() -> dict:
    """E5b control: fast 4-marker 20 m with rel bias block 0 (compare to c3_within_20m_8k_rand4)."""
    return _c3_rel_pos_variant(
        "within_20m",
        rel_pos_bias="first",
        train_steps=8000,
        run_id="within_20m_rand4_relV1",
        notes="E5b V1: within_20m K=4 train, rel_pos_bias=first, 8k",
    )


def c3_rel_pos_e5b_v2_within_20m() -> dict:
    """E5b V2 @ 20 m (K=4 train); compare to c3_within_20m_8k_rand4 ~0.94."""
    return _c3_rel_pos_variant(
        "within_20m",
        rel_pos_bias="all",
        train_steps=8000,
        run_id="within_20m_rand4_relV2",
        notes="E5b V2: within_20m K=4, rel_pos_bias=all, 8k",
    )


def c3_rel_pos_e5b_v2_within_20m_var_k() -> dict:
    """E5b §8.6c: relV2 + K ~ Uniform[1,4] per train sample (count-invariance probe)."""
    spec = _c3_step4_quieter_sidewalk_2k("within_20m")
    spec["train_steps"] = 8000
    spec["id"] = "within_20m_rand_varK_relV2"
    spec["force_restart"] = False
    spec["notes"] = (
        "E5b §8.6c: within_20m, rel_pos_bias=all, train K~Uniform[1,4]/sample, val K=1, 8k"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "rel_pos_bias": "all",
        "c3_n_markers": 0,
        "c3_n_markers_random_min": 1,
        "c3_n_markers_random_max": 4,
        "c3_marker_min_dist_m": 40.0,
    }
    return spec


def c3_rel_pos_e5b_v2_within_75m() -> dict:
    """E5b V2: rel bias in both encoder blocks (within_75m K=1)."""
    return _c3_rel_pos_variant(
        "within_75m",
        rel_pos_bias="all",
        train_steps=8000,
        run_id="within_75m_rand1_relV2",
        notes="E5b V2: within_75m K=1, rel_pos_bias=all, 8k; compare to rand1 + relV1",
    )


def c3_rel_pos_e5b_v1_within_50m() -> dict:
    """E5b V1 @ 50 m (K=2 train); baseline no-bias ~0.92 @ 8k on within_50m_6k_rand2."""
    return _c3_rel_pos_variant(
        "within_50m",
        rel_pos_bias="first",
        train_steps=8000,
        run_id="within_50m_rand2_relV1",
        notes="E5b V1: within_50m K=2, rel_pos_bias=first, 8k",
    )


def c3_rel_pos_e5b_v2_within_50m() -> dict:
    """E5b V2 @ 50 m (K=2 train)."""
    return _c3_rel_pos_variant(
        "within_50m",
        rel_pos_bias="all",
        train_steps=8000,
        run_id="within_50m_rand2_relV2",
        notes="E5b V2: within_50m K=2, rel_pos_bias=all, 8k",
    )


def c3_m1_food_drink_within_100m_smoke() -> dict:
    """N5 m1 smoke: conjunctive target, CE+Dice, K=1 train @ 100 m."""
    from spatial_data.c3_tasks import M1_TASK_KEY, within_train_markers

    k, min_dist = within_train_markers(100)
    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 2000
    spec["id"] = "food_drink_within_100m_smoke"
    spec["notes"] = (
        f"N5 m1 smoke: {M1_TASK_KEY}, estab_3 + within_100m, K={k} train, CE+Dice, 2k steps"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": k,
        "c3_marker_min_dist_m": min_dist,
        "balanced_ce": False,
    }
    return spec


def c3_m1_food_drink_within_100m_8k() -> dict:
    """N5 m1 reference run @ 8k."""
    from spatial_data.c3_tasks import M1_TASK_KEY, within_train_markers

    k, min_dist = within_train_markers(100)
    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_rand1"
    spec["force_restart"] = False
    spec["notes"] = f"N5 m1: {M1_TASK_KEY}, CE+Dice, train K={k}, val single t0 marker, 8k"
    spec["overrides"] = {
        **spec["overrides"],
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": k,
        "c3_marker_min_dist_m": min_dist,
    }
    return spec


def c3_n5_m1_mix_2k_smoke() -> dict:
    """N5 mix smoke: estab + within_100m + m1; warm-start within_100m checkpoint."""
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    ckpt = config.REPO_ROOT / "runs" / "checkpoints" / "within_100m_rand1.pt"
    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 2000
    spec["id"] = "food_drink_within_100m_mix_smoke"
    spec["init_checkpoint"] = str(ckpt)
    spec["load_optimizer"] = False
    spec["reset_step_counter"] = True
    spec["notes"] = (
        f"N5 mix smoke: {list(N5_MIX_TASK_KEYS)}, CE+Dice, init within_100m_rand1, 2k"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "c3_task_mode": "multi",
        "c3_task_keys": list(N5_MIX_TASK_KEYS),
        "c3_eval_task_key": M1_TASK_KEY,
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": 0,
    }
    return spec


def _c3_m1_e8_overrides(**extra) -> dict:
    """E8: L4 encoder, 8k @ batch 32 (~256k presentations ≈ 16k×16), no diagnostics."""
    from spatial_data.c3_tasks import within_train_markers

    k, min_dist = within_train_markers(100)
    return {
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": k,
        "c3_marker_min_dist_m": min_dist,
        "enc_n_layer": 4,
        "batch_size": 32,
        "eval_interval": 1000,
        "diagnostics_enabled": False,
        **extra,
    }


def c3_m1_food_drink_within_100m_L4_8k() -> dict:
    """E8: single-task m1, L4/H6, 8k×batch32, diagnostics off."""
    from spatial_data.c3_tasks import M1_TASK_KEY

    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_L4_8k_b32"
    spec["notes"] = "E8 m1 only: L4, 8k, batch 32 (~256k pres), eval/1k, CE+Dice, diagnostics off"
    spec["overrides"] = {
        **spec["overrides"],
        **_c3_m1_e8_overrides(c3_task_mode="single"),
    }
    return spec


def c3_n5_m1_mix_80_20_L4_8k() -> dict:
    """Mix: ~10% food_drink + ~10% within_100m + ~80% m1 (N5_MIX_TASK_KEYS order)."""
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_mix80_L4_8k_b32"
    spec["notes"] = (
        "E8 mix 80/20: m1 80%, aux 20%, L4, 8k, batch 32, eval/1k, CE+Dice, diagnostics off"
    )
    spec["overrides"] = {
        **spec["overrides"],
        **_c3_m1_e8_overrides(
            c3_task_mode="multi",
            c3_task_keys=list(N5_MIX_TASK_KEYS),
            c3_eval_task_key=M1_TASK_KEY,
            c3_train_task_weights=[0.1, 0.1, 0.8],
            c3_n_markers=0,
        ),
    }
    return spec


def c3_n5_m1_mix_80_20_L4_8k_relV2() -> dict:
    """E8 mix 80/20 + E5b V2 (rel_pos_bias=all in every encoder block)."""
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_mix80_L4_8k_b32_relV2"
    spec["notes"] = (
        "E8 mix 80/20 + relV2: rel_pos_bias=all, L4, 8k, batch 32, eval/1k, CE+Dice, diag off"
    )
    spec["overrides"] = {
        **spec["overrides"],
        **_c3_m1_e8_overrides(
            c3_task_mode="multi",
            c3_task_keys=list(N5_MIX_TASK_KEYS),
            c3_eval_task_key=M1_TASK_KEY,
            c3_train_task_weights=[0.1, 0.1, 0.8],
            c3_n_markers=0,
            rel_pos_bias="all",
        ),
    }
    return spec


def c3_m1_food_drink_within_100m_L4_8k_relV2() -> dict:
    """E8 m1-only + E5b V2 (rel_pos_bias=all)."""
    from spatial_data.c3_tasks import M1_TASK_KEY

    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_L4_8k_b32_relV2"
    spec["notes"] = (
        "E8 m1 only + relV2: rel_pos_bias=all, L4, 8k, batch 32, eval/1k, CE+Dice, diagnostics off"
    )
    spec["overrides"] = {
        **spec["overrides"],
        **_c3_m1_e8_overrides(c3_task_mode="single", rel_pos_bias="all"),
    }
    return spec


def c3_n5_m1_mix_8k() -> dict:
    """N5 reference: three-task mix + warm-start from m0 @ 100 m."""
    from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS

    ckpt = config.REPO_ROOT / "runs" / "checkpoints" / "within_100m_rand1.pt"
    spec = _c3_step4_quieter_sidewalk_2k(M1_TASK_KEY)
    spec["train_steps"] = 8000
    spec["id"] = "food_drink_within_100m_mix_8k"
    spec["force_restart"] = False
    spec["init_checkpoint"] = str(ckpt)
    spec["load_optimizer"] = False
    spec["reset_step_counter"] = True
    spec["notes"] = (
        f"N5 mix: {list(N5_MIX_TASK_KEYS)}, CE+Dice, init within_100m_rand1, K=1 t0 marker, 8k"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "c3_task_mode": "multi",
        "c3_task_keys": list(N5_MIX_TASK_KEYS),
        "c3_eval_task_key": M1_TASK_KEY,
        "c3_grid_loss": "ce_dice",
        "c3_n_markers": 0,
    }
    return spec


def c3_rel_pos_e5b_round2_runs() -> list[dict]:
    """Marvin break batch: 75 m V2, 50 m V1+V2, 20 m V1+V2 (§8.6)."""
    return [
        c3_rel_pos_e5b_v2_within_75m(),
        c3_rel_pos_e5b_v1_within_50m(),
        c3_rel_pos_e5b_v2_within_50m(),
        c3_rel_pos_e5b_v1_within_20m(),
        c3_rel_pos_e5b_v2_within_20m(),
    ]


def c2_gate_ts2_singles_2k_v4() -> dict:
    """C3 step 3: same gate on cropset v4 (merged noise/surface/estab identical to v3b)."""
    spec = c2_gate_ts2_singles_2k()
    spec["id"] = "c2_gate_ts2_singles_2k_v4"
    spec["overrides"] = {**spec["overrides"], "cropset": "v4"}
    spec["notes"] = "C3 regression: §4 gate on v4 RGB planes (expect ≈ v3b gate ~0.978)"
    return spec


def c2_l6a_quiet_sidewalk_2k() -> dict:
    """L6a: one conjunction, no task input (n_tasks=0); AND channels without description."""
    spec = _c2_v3b_2k(
        c2_task_set="l6a_quiet_sidewalk",
        c2_no_task_input=True,
        balanced_ce=True,
    )
    spec["id"] = "c2_L6a_quiet_sidewalk_2k"
    spec["notes"] = "C2 L6a: quiet sidewalk only, no cond_tokens (pass: rule IoU ≥ 0.8)"
    return spec


def c2_l6b_ts2_2k() -> dict:
    """L6b: four singles + five core conjunctions; regression vs gate per single."""
    spec = _c2_v3b_2k(c2_task_set="ts2")
    spec["id"] = "c2_L6b_ts2_2k"
    spec["notes"] = "C2 L6b: full ts2 mixture, cond_tokens, fixed 2k train crops"
    return spec


def c2_l6b_ts2_district_2k() -> dict:
    """L6b variant: on-the-fly district windows (unbounded train crops)."""
    spec = c2_l6b_ts2_2k()
    spec["id"] = "c2_L6b_ts2_district_2k"
    spec["overrides"] = {**spec["overrides"], "c2_train_source": "district_random"}
    spec["notes"] = "C2 L6b: ts2, district_random train source"
    return spec


def c2_l6d_ts2_2k() -> dict:
    """L6d: same ts2 mixture as L6b; plain CE + soft Dice (no per-task f); score primary @ 0.5."""
    spec = _c2_v3b_2k(c2_task_set="ts2")
    spec["id"] = "c2_L6d_ts2_2k"
    spec["notes"] = "C2 L6d: ts2, cond_tokens, CE+Dice (no dampened weights), eval cut-off 0.5"
    spec["overrides"] = {
        **spec["overrides"],
        "c2_grid_loss": "ce_dice",
        "diagnostics_enabled": False,
    }
    return spec


def c2_l6c_c2_full_2k() -> dict:
    """L6c: all usable conjunctions + atom singles; four conjunctions held out from train."""
    spec = _c2_v3b_2k(c2_task_set="c2_full", c2_held_out=True)
    spec["id"] = "c2_L6c_c2_full_2k"
    spec["train_steps"] = 4000
    spec["notes"] = "C2 L6c: c2_full, 4 held-out conjunctions, 4k steps"
    return spec


# ── c4 route tasks (C4_IMPLEMENTATION Phase B) ─────────────────────────────


def _c4_worldsnap_base(**override) -> dict:
    """Shared c4 worldsnap recipe: v4 district windows + c4_routes npz indices."""
    overrides: dict[str, Any] = {
        "task_rung": "c4",
        "data_source": "worldsnap",
        "cropset": "v4",
        "encoding_mode": "onehot_v4",
        "c4_weight_alpha": 0.5,
        "c4_grid_loss": "ce_dice",
        "batch_size": 16,
        "patch_size": 16,
        "learning_rate": 1e-3,
        "eval_interval": 500,
        "balanced_ce": False,
        "worldsnap_gpu_resident": True,
        "marker_radius_px": C3_MARKER_DISC3_PX,
        "rel_pos_bias": "all",
        "diagnostics_enabled": False,
    }
    overrides.update(override)
    return {
        "train_steps": 2000,
        "force_restart": False,
        "overrides": overrides,
        "spatial_eval": {"max_batches": 10, "train_snap_batches": 0},
    }


def c4_loader_smoke_checks(*, device: Any = None) -> dict:
    """Phase B gate: train + val batch from smoke ``c4_routes_*.npz`` (no training).

    Raises on missing store, bad shapes, or empty targets. Returns a small stats dict
    for logs / pytest."""
    import torch
    from spatial_data.c4_tasks import TASK_KEYS, get_c4_task
    from spatial_data.dataset_c4_gpu import (
        C4MultiTaskGpuTrainLoader,
        C4MultiTaskGpuValLoader,
        load_c4_route_index,
        load_district_planes_v4,
    )

    dev = device
    if dev is None:
        if not torch.cuda.is_available():
            raise RuntimeError("c4_loader_smoke_checks requires CUDA")
        dev = torch.device("cuda")
    crops = config.district_data_dir() / "crops" / "v4"
    routes = crops / "c4"
    train_npz = routes / "c4_routes_train.npz"
    val_npz = routes / "c4_routes_val.npz"
    if not train_npz.is_file() or not val_npz.is_file():
        raise FileNotFoundError(f"c4 smoke store missing under {routes}")
    pads = [get_c4_task(k).padded_cond_ids() for k in TASK_KEYS]
    if len(set(pads)) != len(TASK_KEYS):
        raise AssertionError(f"c4 cond ids not distinct: {pads}")
    district = load_district_planes_v4(crops, dev)
    train_routes = load_c4_route_index(routes, "train", dev, max_items=64)
    train_loader = C4MultiTaskGpuTrainLoader(
        list(TASK_KEYS), "onehot_v4", 64, 4, 42,
        district=district, route_index=train_routes,
    )
    img, tgt, task_id, cond_ids = train_loader.next_batch()
    if img.shape != (4, 16, 256, 256):
        raise AssertionError(f"bad img shape {tuple(img.shape)}")
    if tgt.shape != (4, 64, 64) or int(tgt.sum().item()) <= 0:
        raise AssertionError(f"bad or empty tgt shape {tuple(tgt.shape)}")
    if cond_ids.shape != (4, 2):
        raise AssertionError(f"bad cond_ids shape {tuple(cond_ids.shape)}")
    val_routes = load_c4_route_index(routes, "val", dev, max_items=8)
    val_loader = C4MultiTaskGpuValLoader(
        list(TASK_KEYS), "onehot_v4", 64, 4,
        district=district, route_index=val_routes,
    )
    if len(val_loader) != 8 * len(TASK_KEYS):
        raise AssertionError(f"val len {len(val_loader)} != 8 * n_tasks")
    vimg, vtgt, vtid, _ = val_loader.next_batch()
    if vimg.shape[0] != 4 or len({int(t) for t in vtid.tolist()}) < 1:
        raise AssertionError("val batch task mix failed")
    return {
        "n_train_routes": int(train_routes["origin"].shape[0]),
        "n_val_routes": int(val_routes["origin"].shape[0]),
        "train_fg_cells": int(tgt.sum().item()),
        "val_fg_cells": int(vtgt.sum().item()),
        "sample_cond": cond_ids[0].tolist(),
    }


def c4_phase_b_loader_smoke() -> dict:
    """~150 steps: segment+detour multi on capped smoke npz; ViT forward + loss decrease."""
    from spatial_data.c4_tasks import TASK_KEYS

    spec = _c4_worldsnap_base(
        c4_task_mode="multi",
        c4_task_keys=list(TASK_KEYS),
        worldsnap_max_train_items=64,
        worldsnap_max_val_items=16,
    )
    spec["train_steps"] = 150
    spec["id"] = "c4_loader_smoke"
    spec["notes"] = (
        "C4 Phase B smoke: segment+detour multi, 64 train routes, 3 px discs, "
        "rel_pos_bias=all, 150 steps (detour GT from current P1 store)"
    )
    spec["spatial_eval"] = {"max_batches": 5, "train_snap_batches": 0}
    spec["overrides"] = {
        **spec["overrides"],
        "eval_interval": 50,
        "eval_iters": 3,
        "eval_iters_final": 5,
    }
    return spec


C4_EXPLORE_BATCH_SIZE = 128  # profile_c4_batch_size.py on 3060: ~803 samp/s L2, ~4.4 GB peak
# 16k×b32 = 512k presentations → 512k/128 = 4k steps at b128 (same data exposure, ~4× fewer steps).
C4_EXPLORE_TRAIN_STEPS = 4000
C4_EXPLORE_EVAL_INTERVAL = 250  # b128: more steps between evals than b32@1k; ~16 CE points / 4k
C4_EXPLORE_FORCE_RESTART = False

def _c4_learning_recipe_overrides(**extra: Any) -> dict[str, Any]:
    """m1 report §7.10 style defaults on L2 (Phase D/E)."""
    return {
        "enc_n_layer": 2,
        "batch_size": C4_EXPLORE_BATCH_SIZE,
        "marker_radius_px": C3_MARKER_DISC3_PX,
        "rel_pos_bias": "all",
        "c4_grid_loss": "ce_dice",
        "eval_interval": C4_EXPLORE_EVAL_INTERVAL,
        "diagnostics_enabled": True,
        **extra,
    }


def _c4_c4a_learning_spec(
    *,
    run_id: str,
    notes: str,
    train_steps: int = C4_EXPLORE_TRAIN_STEPS,
    lr_horizon_steps: int | None = None,
    seed: int = 4,
    mix_within_20m: bool = False,
    force_restart: bool = False,
) -> dict:
    """c4a segment only on val; train optionally mixed with c3 within_20m helper."""
    spec = _c4_worldsnap_base(
        worldsnap_max_train_items=None,
        worldsnap_max_val_items=None,
    )
    spec["train_steps"] = train_steps
    spec["lr_horizon_steps"] = train_steps if lr_horizon_steps is None else lr_horizon_steps
    spec["id"] = run_id
    spec["force_restart"] = force_restart
    spec["notes"] = notes
    spec["spatial_eval"] = {"max_batches": None, "train_snap_batches": 0}
    ov = _c4_learning_recipe_overrides(
        c4_task_mode="single",
        c4_task_key="segment",
        c4_eval_task_key="segment",
        c4_task_keys=None,
        diagnostics_enabled=True,
        seed=seed,
    )
    if mix_within_20m:
        ov["c4_mix_c3_task_keys"] = ["within_20m"]
        ov["c3_n_markers"] = 2
        ov["c3_marker_min_dist_m"] = 40.0
        ov["c4_train_task_weights"] = [0.5, 0.5]
    else:
        ov["c4_mix_c3_task_keys"] = None
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def c4_c4a_explore_run_queue(
    *,
    seed: int = 4,
    train_steps: int = C4_EXPLORE_TRAIN_STEPS,
) -> list[dict]:
    """2×2: segment vs mix20m+segment × cosine vs flat LR (c4a only; val always segment)."""
    flat_h = _M1_REPORT_LR_FLAT_HORIZON
    out = []
    for mix, lr_flat in ((False, False), (False, True), (True, False), (True, True)):
        mix_tag = "mix20m_" if mix else ""
        lr_tag = "lrflat" if lr_flat else f"cosine{train_steps // 1000}k"
        run_id = (
            f"c4_{mix_tag}segment_L2_{train_steps // 1000}k_b{C4_EXPLORE_BATCH_SIZE}"
            f"_disc3_relV2_{lr_tag}_seed{seed}"
        )
        lr_note = f"lr flat (horizon={flat_h})" if lr_flat else f"cosine lr_horizon={train_steps}"
        train_note = (
            "train ½ within_20m K=2 (v4 crops) + ½ segment (routes)"
            if mix
            else "train segment only (c4 routes)"
        )
        out.append(
            _c4_c4a_learning_spec(
                run_id=run_id,
                train_steps=train_steps,
                lr_horizon_steps=flat_h if lr_flat else train_steps,
                seed=seed,
                mix_within_20m=mix,
                force_restart=C4_EXPLORE_FORCE_RESTART,
                notes=(
                    f"c4a explore: {train_note}; val=segment routes only; "
                    f"L2 b{C4_EXPLORE_BATCH_SIZE} disc3 relV2; {lr_note}; {train_steps} steps seed={seed}"
                ),
            )
        )
    return out


_C4_MIX_L8B_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]


def _c4_c4b_learning_spec(
    *,
    run_id: str,
    notes: str,
    train_steps: int = C4_EXPLORE_TRAIN_STEPS,
    lr_horizon_steps: int | None = None,
    seed: int = 4,
    mix_within_20m: bool = True,
    force_restart: bool = False,
) -> dict:
    """c4b: val detour; train L8b mix or ½ segment + ½ detour only."""
    from spatial_data.c4_tasks import TASK_KEYS

    spec = _c4_worldsnap_base(
        worldsnap_max_train_items=None,
        worldsnap_max_val_items=None,
    )
    spec["train_steps"] = train_steps
    spec["lr_horizon_steps"] = train_steps if lr_horizon_steps is None else lr_horizon_steps
    spec["id"] = run_id
    spec["force_restart"] = force_restart
    spec["notes"] = notes
    spec["spatial_eval"] = {"max_batches": None, "train_snap_batches": 0}
    ov = _c4_learning_recipe_overrides(
        c4_task_mode="multi",
        c4_task_keys=list(TASK_KEYS),
        c4_eval_task_key="detour",
        diagnostics_enabled=True,
        seed=seed,
    )
    if mix_within_20m:
        ov["c4_mix_c3_task_keys"] = ["within_20m"]
        ov["c3_n_markers"] = 2
        ov["c3_marker_min_dist_m"] = 40.0
        ov["c4_train_task_weights"] = _C4_MIX_L8B_WEIGHTS
    else:
        ov["c4_mix_c3_task_keys"] = None
        ov["c4_train_task_weights"] = [0.5, 0.5]
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def c4_c4b_run_queue(
    *,
    seeds: tuple[int, ...] = (4, 5),
    train_steps: int = C4_EXPLORE_TRAIN_STEPS,
    include_segdet_only_ablation: bool = True,
) -> list[dict]:
    """Phase E1: 4× L8b mix (2 seeds × LR); optional 1× segment+detour only (flat LR, seed 4)."""
    flat_h = _M1_REPORT_LR_FLAT_HORIZON
    out = []
    for seed in seeds:
        for lr_flat in (False, True):
            lr_tag = "lrflat" if lr_flat else f"cosine{train_steps // 1000}k"
            run_id = (
                f"c4_mix20m_segdet_L2_{train_steps // 1000}k_b{C4_EXPLORE_BATCH_SIZE}"
                f"_disc3_relV2_{lr_tag}_seed{seed}"
            )
            lr_note = f"lr flat (horizon={flat_h})" if lr_flat else f"cosine lr_horizon={train_steps}"
            out.append(
                _c4_c4b_learning_spec(
                    run_id=run_id,
                    train_steps=train_steps,
                    lr_horizon_steps=flat_h if lr_flat else train_steps,
                    seed=seed,
                    mix_within_20m=True,
                    force_restart=C4_EXPLORE_FORCE_RESTART,
                    notes=(
                        "c4b E1: train ⅓ within_20m K=2 + ⅓ segment + ⅓ detour; val=detour; "
                        f"L2 b{C4_EXPLORE_BATCH_SIZE} disc3 relV2; {lr_note}; {train_steps} steps seed={seed}"
                    ),
                )
            )
    if include_segdet_only_ablation:
        out.append(
            _c4_c4b_learning_spec(
                run_id=(
                    f"c4_segdet_L2_{train_steps // 1000}k_b{C4_EXPLORE_BATCH_SIZE}"
                    f"_disc3_relV2_lrflat_seed4"
                ),
                train_steps=train_steps,
                lr_horizon_steps=flat_h,
                seed=4,
                mix_within_20m=False,
                force_restart=C4_EXPLORE_FORCE_RESTART,
                notes=(
                    "c4b ablation: train ½ segment + ½ detour (no within_20m); val=detour; "
                    f"L2 b{C4_EXPLORE_BATCH_SIZE} disc3 relV2; lr flat; {train_steps} steps seed=4"
                ),
            )
        )
    return out


def c4_smoke_2k_diagnostics() -> dict:
    """2k c4a segment plumbing; tier1/2 diagnostics + c4 eval on segment val."""
    spec = _c4_c4a_learning_spec(
        run_id="c4_segment_smoke_2k_diag",
        train_steps=2000,
        lr_horizon_steps=2000,
        force_restart=False,
        notes="C4 smoke 2k: segment only, diagnostics, full segment-route val",
    )
    spec["spatial_eval"] = {"max_batches": None, "train_snap_batches": 0}
    spec["overrides"] = {
        **spec["overrides"],
        "eval_interval": 500,
        "eval_iters": 20,
        "eval_iters_final": 50,
    }
    return spec


def c4_phase_d_run_queue() -> list[dict]:
    """Alias: first-pass c4a 2×2 explore (one seed)."""
    return c4_c4a_explore_run_queue()


def c4_phase_e_run_queue() -> list[dict]:
    """Alias: c4b E1 (2 seeds × cosine/flat LR)."""
    return c4_c4b_run_queue()


def c1_l5b_v3b_sidewalk_sweep() -> list[dict]:
    """C2_PLAN P2: is sidewalk (0.85-0.87) limited by the input encoding or by the patch size?
    Same recipe as c1_l5b_core_four_2k_v3b (scalar, P=16 = the existing baseline run); only encoding / patch change."""
    variants = [("onehot_P16", {"encoding_mode": "onehot"}),
                ("scalar_P8", {"patch_size": 8}),
                ("onehot_P8", {"encoding_mode": "onehot", "patch_size": 8})]
    out = []
    for tag, extra in variants:
        spec = c1_l5b_core_four_2k_v3b()
        spec["id"] = f"c1_L5b_core_four_2k_v3b_{tag}"
        spec["notes"] = f"C1 L5b on v3b, {tag}: sidewalk comparison (C2_PLAN P2)"
        spec["overrides"] = {**spec["overrides"], **extra}
        out.append(spec)
    return out



# ---------------------------------------------------------------- c4 overnight batch (25 Sep 2026)
def _c4_night_spec(*, run_id: str, notes: str, keys: list[str], weights: list[float], train_steps: int,
                   seed: int = 4, **extra: Any) -> dict:
    """c4b on the fast loader: multi-task over c4 route tasks only (no c3 mix), val = detour, tolerant metrics."""
    spec = _c4_worldsnap_base(worldsnap_max_train_items=None, worldsnap_max_val_items=None)
    spec["train_steps"] = train_steps
    spec["lr_horizon_steps"] = train_steps
    spec["id"] = run_id
    spec["force_restart"] = False
    spec["notes"] = notes
    spec["spatial_eval"] = {"max_batches": None, "train_snap_batches": 0}
    ov = _c4_learning_recipe_overrides(
        c4_task_mode="multi", c4_task_keys=keys, c4_train_task_weights=weights,
        c4_eval_task_key="detour", c4_mix_c3_task_keys=None, diagnostics_enabled=True,
        eval_interval=1000, seed=seed, **extra,
    )
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def c4_overnight_run_queue() -> list[dict]:
    """C4_PLAN §10 next runs, on the vectorised loader. Order: cheap sanity first, big-store runs last
    (the 100k store builds on the CPU while the first runs train). Each spec's notes say what it tests."""
    SD = ["segment", "detour"]; BSD = ["building", "segment", "detour"]
    half = [0.5, 0.5]; third = [1 / 3, 1 / 3, 1 / 3]
    out = [
        # N0: same as the best 4k run but on the new loader; must reproduce IoU ~0.32 / dilated ~0.53 (sanity, ~9 min)
        _c4_night_spec(run_id="c4n_segdet_L2_4k_seed4_fastloader", keys=SD, weights=half, train_steps=4000, seed=4,
                       notes="N0 sanity: ½ segment + ½ detour, 4k, new loader; expect detour IoU ~0.32, dilated ~0.53"),
        # N1: the c4b reference: 16k, two seeds
        _c4_night_spec(run_id="c4n_segdet_L2_16k_seed4", keys=SD, weights=half, train_steps=16000, seed=4,
                       notes="N1 reference: ½ segment + ½ detour, 16k cosine; prediction dilated IoU > 0.8"),
        _c4_night_spec(run_id="c4n_segdet_L2_16k_seed5", keys=SD, weights=half, train_steps=16000, seed=5,
                       notes="N1 reference seed 5"),
        # N2: dense building helper (C4_PLAN §10 addendum a)
        _c4_night_spec(run_id="c4n_bld_segdet_L2_16k_seed4", keys=BSD, weights=third, train_steps=16000, seed=4,
                       notes="N2 helper: ⅓ building + ⅓ segment + ⅓ detour, 16k; does the dense building task lift the detour?"),
        _c4_night_spec(run_id="c4n_bld_segdet_L2_16k_seed5", keys=BSD, weights=third, train_steps=16000, seed=5,
                       notes="N2 helper seed 5"),
        # N3: gradient clip 3 (pre-clip norm sat above 1.0 from 1.5k on in the 4k runs)
        _c4_night_spec(run_id="c4n_segdet_L2_16k_seed4_clip3", keys=SD, weights=half, train_steps=16000, seed=4, grad_clip=3.0,
                       notes="N3 clip: as N1 seed 4 with grad_clip=3.0"),
        # N4: short routes only (is route length the limit?)
        _c4_night_spec(run_id="c4n_segdet_L2_16k_seed4_max120m", keys=SD, weights=half, train_steps=16000, seed=4, c4_max_straight_m=120.0,
                       notes="N4 length: as N1 seed 4, routes with straight distance <= 120 m (train and val)"),
        # N5: depth
        _c4_night_spec(run_id="c4n_segdet_L4_16k_seed4", keys=SD, weights=half, train_steps=16000, seed=4, enc_n_layer=4,
                       notes="N5 depth: as N1 seed 4 with 4 blocks"),
        # N6: the 100k store (needs crops/v4/c4_100k built first; ~3.5 h CPU, start it before the queue)
        _c4_night_spec(run_id="c4n_segdet_L2_16k_seed4_store100k", keys=SD, weights=half, train_steps=16000, seed=4, c4_routes_subdir="c4_100k",
                       notes="N6 data: as N1 seed 4 on the 100k route store (train/val CE gap test)"),
        _c4_night_spec(run_id="c4n_segdet_L4_16k_seed4_store100k", keys=SD, weights=half, train_steps=16000, seed=4,
                       enc_n_layer=4, c4_routes_subdir="c4_100k",
                       notes="N7 depth+data: as N5 (4 blocks) on the 100k route store"),
    ]
    return out


def c4_l4_store100k_run_queue() -> list[dict]:
    """Single follow-up: N5 architecture on c4_100k (after overnight N0–N6)."""
    return [_c4_night_spec(
        run_id="c4n_segdet_L4_16k_seed4_store100k",
        keys=["segment", "detour"],
        weights=[0.5, 0.5],
        train_steps=16000,
        seed=4,
        enc_n_layer=4,
        c4_routes_subdir="c4_100k",
        notes="N7 depth+data: as N5 (4 blocks) on the 100k route store",
    )]



def c4_mix_probe_run_queue() -> list[dict]:
    """25 Sep evening (Marvin): which task is the odd one out? 2x2 over {segment, building} with the
    detour always present; N7 (½ segment + ½ detour, 4 blocks, 100k store) is the fourth cell, already run."""
    base = dict(train_steps=16000, seed=4, enc_n_layer=4, c4_routes_subdir="c4_100k")
    return [
        _c4_night_spec(run_id="c4n_det_only_L4_16k_store100k", keys=["detour"], weights=[1.0],
                       notes="N8 mix probe: detour only; compare with N7 (½ segment + ½ detour): does the segment help or hurt?", **base),
        _c4_night_spec(run_id="c4n_bld_det_L4_16k_store100k", keys=["building", "detour"], weights=[0.5, 0.5],
                       notes="N9 mix probe: ½ building + ½ detour, no segment; compare with N8: does the building task compete on its own?", **base),
        _c4_night_spec(run_id="c4n_bld_seg_det_L4_16k_store100k", keys=["building", "segment", "detour"], weights=[1 / 3, 1 / 3, 1 / 3],
                       notes="N10 mix probe: all three at ⅓; the fourth cell of the 2x2", **base),
    ]


# ---------------------------------------------------------------- C5 rung 0 / rung 1 (C5_PLAN §5: R0, R1)
C5_N9_16CH_CKPT = config.REPO_ROOT / "runs" / "checkpoints" / "c4n_bld_det_L4_16k_store100k.pt"


def _c5_warmstart_n9_if_fresh(spec: dict) -> dict:
    """First launch only: 16→17ch expand from N9; resume uses this run's own checkpoints."""
    rid = spec["id"]
    ckpt_dir = config.harness_checkpoint_dir
    has_run_ckpt = (ckpt_dir / f"{rid}.pt").is_file() or (
        (ckpt_dir / rid).is_dir() and any((ckpt_dir / rid).glob("step*.pt"))
    )
    if not has_run_ckpt:
        spec["init_checkpoint"] = str(C5_N9_16CH_CKPT)
        spec["load_optimizer"] = False
        spec["reset_step_counter"] = True
    return spec


def c5_r0_regression_17ch() -> dict:
    """R0: the best one-shot detour recipe (N9: ½ building + ½ detour, 4 blocks, 100k store, 16k) with the
    17-channel encoding (scratchpad plane, always empty here). Pass: detour dilated IoU within 0.02 of 0.81."""
    spec = _c4_night_spec(run_id="c5_r0_bld_det_L4_16k_store100k_17ch", keys=["building", "detour"], weights=[0.5, 0.5],
                          train_steps=16000, seed=4, enc_n_layer=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          notes="C5 R0: N9 recipe with 17 input channels (empty scratchpad); regression against N9 (0.813 dilated)")
    return spec


def c5_r0a_grid128() -> dict:
    """R0 variant A (resolution test, 29 Sep 2026): R0 with a 128x128 output (2 m cells), same 256 m window and input.
    Compare with R0 (grid 64) on the metre-based metrics: dilated IoU and precision/recall use a 4 m tolerance at both sizes."""
    spec = c5_r0_regression_17ch()
    spec["id"] = "c5_r0a_bld_det_L4_16k_store100k_17ch_g128"
    spec["overrides"]["grid_out_size"] = 128
    spec["notes"] = "C5 R0 variant A: R0 with 128x128 output (2 m cells); resolution test against R0 at 64x64"
    return spec


def c5_r0b_window512() -> dict:
    """R0 variant B (resolution test, 29 Sep 2026): R0 on 512 m windows (512 px input at 1 m, 128x128 output =
    4 m cells as in R0, 1024 patch tokens at P=16), on the new Zuid + Centrum map (cropset v4ns: v4 without sun),
    store c4_512_20k (routes 40-500 m straight, new 4x4 split with 1.2 km blocks). Compare with R0 / variant A on
    the 40-250 m length bins; R0 and A are also evaluated on this map's held-out ground (256 m val store c4_256_val)."""
    spec = c5_r0_regression_17ch()
    spec["id"] = "c5_r0b_bld_det_L4_16k_zc512_17ch"
    ov = spec["overrides"]
    ov.update(img_size=512, grid_out_size=128, district_name="zuid_centrum", cropset="v4ns",
              c4_routes_subdir="c4_512_20k", diagnostics_pred_c4_reference_subdir=None,
              batch_size=64)   # batch 128 does not fit in 12 GB at 1024 tokens
    # first a short run (Marvin, 29 Sep): time, progress, problems; 8k x 64 = R0's samples at its step 4000.
    # The full-length run is 32k steps (= R0's 16k x 128 samples).
    spec["train_steps"] = 8000
    spec["lr_horizon_steps"] = 8000
    spec["id"] = "c5_r0b_bld_det_L4_8k_b64_zc512_17ch"
    spec["notes"] = "C5 R0 variant B: 512 m windows, 128x128 output (4 m cells), Zuid + Centrum map, routes 40-500 m"
    return spec


def c5_rerun_queue() -> list[dict]:
    """Reruns after the 29 Sep task changes (C5_PLAN §6), De Pijp 100k store, 2 blocks, 8k each (~15-20 min each):
    R1 with grazing rays only, at 64x64 and at 128x128 output (does the finer grid help the rays?);
    R2 and R3 with the token as the only target of length samples (mask cells ignored) - watch iou_segment."""
    r1 = c5_r1_rays_segment_mix()
    r1["id"] = "c5_r1b_seg_rays_grazing_L2_8k_store100k"
    r1["notes"] = "C5 R1b: as R1 with grazing rays only (fans rebuilt 29 Sep)"
    r1g = c5_r1_rays_segment_mix()
    r1g["id"] = "c5_r1b_seg_rays_grazing_L2_8k_store100k_g128"
    r1g["overrides"]["grid_out_size"] = 128
    r1g["notes"] = "C5 R1b at 128x128 output (2 m cells): resolution test A on the rays"
    r2 = c5_r2_length_line()
    r2["id"] = "c5_r2b_seg_lenline_tokonly_L2_8k_store100k"
    r2["notes"] = "C5 R2b: length_line trains the token only (no mask target); segment IoU must stay ~0.8"
    r3 = c5_r3_length_path()
    r3["id"] = "c5_r3b_seg_lenline_lenpath_tokonly_L2_8k_store100k"
    r3["notes"] = "C5 R3b: length_line + length_path, token only; segment IoU must stay ~0.8"
    return [r1, r1g, r2, r3]


def c5_traj_queue() -> list[dict]:
    """C5 rung 1, second definition (30 Sep 2026): key trajectories within 100 m walking from both markers, as the
    only task (no segment mix), 2 blocks, 8k, De Pijp 100k store; at 64x64 and at 128x128 output."""
    out = []
    for g in (64, 128):
        spec = _c4_night_spec(run_id="c5_r1c_traj_L2_8k_store100k" + ("" if g == 64 else "_g128"), keys=["trajectories"],
                              weights=[1.0], train_steps=8000, seed=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                              notes=f"C5 R1c: key trajectories only, {g}x{g} output, 17ch, 2 blocks, 8k")
        spec["overrides"]["c4_eval_task_key"] = "trajectories"
        if g != 64:
            spec["overrides"]["grid_out_size"] = g
        out.append(spec)
    return out


def c5_traj_one() -> dict:
    """Key trajectories from ONE marker (marker 0 drawn, target = its trajectories only), 2 blocks, 8k, De Pijp, 64x64."""
    spec = _c4_night_spec(run_id="c5_r1d_traj_one_L2_8k_store100k", keys=["trajectories_one"], weights=[1.0], train_steps=8000,
                          seed=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          notes="C5 R1d: key trajectories from one marker only, 64x64, 17ch, 2 blocks, 8k")
    spec["overrides"]["c4_eval_task_key"] = "trajectories_one"
    return spec


def c5_r1d_traj_one_L2_16k_continue() -> dict:
    """Resume the finished 8k R1d run to 16k (same run id — harness auto-resumes from step 7999)."""
    spec = c5_traj_one()
    spec["train_steps"] = 16000
    spec["lr_horizon_steps"] = 16000
    spec["notes"] = "C5 R1d: continue L2 g64 8k→16k; val loss still falling at 7999 (dilated IoU ~0.55)"
    return spec


def c5_r1d_traj_one_L4_16k() -> dict:
    """One-marker trajectories, 4 encoder blocks, 16k, g64 (capacity ablation vs R1d L2)."""
    spec = _c4_night_spec(
        run_id="c5_r1d_traj_one_L4_16k_store100k", keys=["trajectories_one"], weights=[1.0], train_steps=16000,
        seed=4, enc_n_layer=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
        notes="C5 R1d L4: one-marker trajectories, 64x64 output, 4 blocks, 16k",
    )
    spec["overrides"]["c4_eval_task_key"] = "trajectories_one"
    return spec


def c5_r1d_traj_one_L2_g128() -> dict:
    """One-marker trajectories at 128x128 output (2 m cells), 8k — resolution ablation vs R1d L2 g64."""
    spec = c5_traj_one()
    spec["id"] = "c5_r1d_traj_one_L2_8k_store100k_g128"
    spec["overrides"]["grid_out_size"] = 128
    spec["notes"] = "C5 R1d L2 g128: one-marker trajectories, 128x128 output, 2 blocks, 8k"
    return spec


def c5_r1d_traj_one_L4_g128() -> dict:
    """One-marker trajectories, L4 + 128x128 output, 16k."""
    spec = _c4_night_spec(
        run_id="c5_r1d_traj_one_L4_16k_store100k_g128", keys=["trajectories_one"], weights=[1.0], train_steps=16000,
        seed=4, enc_n_layer=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
        notes="C5 R1d L4 g128: one-marker trajectories, 128x128 output, 4 blocks, 16k",
    )
    spec["overrides"].update(c4_eval_task_key="trajectories_one", grid_out_size=128)
    return spec


def c5_r1d_traj_one_run_queue() -> list[dict]:
    """R1d sweeps after the first 8k L2 g64 run (~20–35 min each on 3060 except L4 16k ~40 min)."""
    return [
        c5_r1d_traj_one_L2_16k_continue(),
        c5_r1d_traj_one_L4_16k(),
        c5_r1d_traj_one_L2_g128(),
        c5_r1d_traj_one_L4_g128(),
    ]


def c5_traj_zc512() -> dict:
    """Key trajectories on the new map (Zuid + Centrum), 512 m windows, 128x128 output (4 m cells), batch 64 x 16k steps
    (= the samples of the 8k x 128 De Pijp runs). Needs c5_traj_train.npz in c4_512_20k (built 30 Sep night)."""
    spec = _c4_night_spec(run_id="c5_r1c_traj_L2_16k_b64_zc512", keys=["trajectories"], weights=[1.0], train_steps=16000,
                          seed=4, c4_routes_subdir="c4_512_20k", encoding_mode="onehot_v4_sp",
                          notes="C5 R1c on the new map: key trajectories only, 512 m windows, 128x128 output, batch 64, 16k")
    spec["overrides"].update(c4_eval_task_key="trajectories", img_size=512, grid_out_size=128, district_name="zuid_centrum",
                             cropset="v4ns", diagnostics_pred_c4_reference_subdir=None, batch_size=64)
    return spec


def c5_overnight_30sep_queue() -> list[dict]:
    """Night 30 Sep -> 1 Oct, in the order their data is ready (~4-5 h): trajectories on De Pijp at 64 and 128 (~20 min
    each), R0 variant B on the new map (512 m, 8k x 64, ~90 min), trajectories on the new map at 512 m (~1.5-2.5 h, 2 blocks at 1024 tokens).
    The 512 m runs need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment."""
    traj64, traj128 = c5_traj_queue()
    return [traj64, c5_traj_one(), traj128, c5_r0b_window512(), c5_traj_zc512()]


def c5_traj_r40_queue(enc_n_layer: int = 2, train_steps: int = 8000) -> list[dict]:
    """Key trajectories at a 40 m horizon, one marker, four steps (1 Oct 2026): step 1 marker -> round 1; steps 2-4
    marker + earlier rounds on the scratchpad -> that round. Each a task of its own (no mix), De Pijp store, first
    40k routes (the 40 m file covers those), 64x64 output."""
    out = []
    for key, tag in (("trajectories_one_r40", "step1"), ("trajectories_step2_r40", "step2"),
                     ("trajectories_step3_r40", "step3"), ("trajectories_step4_r40", "step4")):
        spec = _c4_night_spec(run_id=f"c5_r1e_traj40_{tag}_L{enc_n_layer}_{train_steps // 1000}k_store40k", keys=[key], weights=[1.0],
                              train_steps=train_steps, seed=4, enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k",
                              encoding_mode="onehot_v4_sp", worldsnap_max_train_items=40000,
                              notes=f"C5 R1e: 40 m key trajectories, {tag} (one marker), 17ch, {enc_n_layer} blocks, {train_steps // 1000}k, first 40k routes")
        spec["overrides"]["c4_eval_task_key"] = key
        out.append(spec)
    return out


def c5_wave_queue(enc_n_layer: int = 2, train_steps: int = 8000, grid: int = 128) -> list[dict]:
    """C5 R5 (1 Oct 2026, C5_PLAN §7): the four step types of the wave search, each alone, De Pijp first 40k routes,
    128x128 output (2 m cells), fronts every 40 m, state token in slot 0, token head with 40 classes.
    wave_first: start -> first front; wave_step: random middle pass; wave_reach: the pass reaching the goal (token
    BUILD_PATH); wave_path: fronts -> the way back (token DONE)."""
    out = []
    for key in ("wave_first", "wave_step", "wave_reach", "wave_back", "wave_origin", "wave_draw"):
        spec = _c4_night_spec(run_id=f"c5_r5_{key}_L{enc_n_layer}_{train_steps // 1000}k_g{grid}_store40k", keys=[key], weights=[1.0],
                              train_steps=train_steps, seed=4, enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k",
                              encoding_mode="onehot_v4_sp", worldsnap_max_train_items=40000, grid_out_size=grid,
                              c5_token_targets=True, c5_token_head_enabled=True, c5_n_token_classes=41,
                              notes=f"C5 R5 {key}: wave search step type alone, {enc_n_layer} blocks, {train_steps // 1000}k, {grid}x{grid}, first 40k routes")
        spec["overrides"]["c4_eval_task_key"] = key
        out.append(spec)
    return out


def c5_wave_back_counter_queue(key: str = "wave_back", enc_n_layer: int = 2, train_steps: int = 8000) -> list[dict]:
    """Counter test (2 Oct 2026): which ingredient of the CE + noise recipe makes the backward hop a blob?
    Reference (exists): c5_r5_wave_back_L2_8k_g128_store40k = this task alone, CE + Dice, no pad noise
    (new-cell recall 0.68, 12.5 false cells per sample, 95 % of the consumed front erased).
    Each run differs from the reference only in the named ingredient(s); same store (first 40k), seed, steps, lr.
      A  dice_noisep1   CE + Dice, pad noise on           -> the noise alone
      B  ce_nonoise     CE recipe (beta 0), no pad noise  -> dropping Dice alone (the lit-cell weight ~7.5 comes with it)
      C  ce_noisep1     CE recipe (beta 0), pad noise on  -> the failing recipe without the mix
    Read with harness/diagnostics/wave_back_ties.py RUN_ID (threshold sweep, false cells by cause)."""
    def base(tag: str, what: str) -> dict:
        spec = next(sp for sp in c5_wave_queue(enc_n_layer, train_steps) if sp["overrides"]["c4_eval_task_key"] == key)
        spec["id"] = f"c5_r5_{key}_{tag}_L{enc_n_layer}_{train_steps // 1000}k_g128_store40k"
        spec["notes"] = f"C5 R5 counter test, {key} alone, L{enc_n_layer} {train_steps // 1000}k, first 40k routes: {what}"
        spec["overrides"]["diagnostics_wave_rollout_start_pass"] = "first_back"
        return spec

    ce = {**_c5_wave_beta_overrides(beta=0.0), "c5_wave_change_ce_beta_schedule": ""}
    a = base("dice_noisep1", "CE + Dice, pad noise p1")
    a["overrides"]["c5_wave_pad_noise"] = dict(_C5_WAVE_FWD3_PAD_NOISE)
    b = base("ce_nonoise", "CE recipe beta 0, no pad noise")
    b["overrides"].update(ce)
    c = base("ce_noisep1", "CE recipe beta 0, pad noise p1")
    c["overrides"].update(ce); c["overrides"]["c5_wave_pad_noise"] = dict(_C5_WAVE_FWD3_PAD_NOISE)
    return [a, b, c]


def c5_wave_back_counter_queue2(enc_n_layer: int = 2, train_steps: int = 8000) -> list[dict]:
    """Counter test, round 2 (2 Oct 2026). Round 1: neither the pad noise nor dropping Dice reproduces the blob of the
    bwd2 runs on wave_back alone (C = CE + noise: 68 false cells per sample against 167-259). What C does not have and
    the bwd2 runs do: the mix with wave_origin (60/40) and the flat lr. One factor each, then both:
      ref  wave_back alone, CE + Dice, no noise, cosine   -> the reference again on today's code
      D    C + flat lr                                    -> the flat lr alone
      E    C + mix back/origin 60/40 (cosine)             -> the mix alone
      F    C + mix + flat lr                              -> the bwd2 plain recipe at L2 / 8k / first 40k (should fail)
    C = c5_r5_wave_back_ce_noisep1_L2_8k_g128_store40k (CE recipe beta 0, pad noise p1, cosine over the run)."""
    tag = f"L{enc_n_layer}_{train_steps // 1000}k_g128_store40k"
    ce_noise = {**_c5_wave_beta_overrides(beta=0.0), "c5_wave_change_ce_beta_schedule": "",
                "c5_wave_pad_noise": dict(_C5_WAVE_FWD3_PAD_NOISE)}

    def spec_for(run_id: str, keys: list[str], weights: list[float], what: str) -> dict:
        spec = _c4_night_spec(run_id=run_id, keys=keys, weights=weights, train_steps=train_steps, seed=4,
                              enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                              worldsnap_max_train_items=40000, grid_out_size=128, c5_token_targets=True,
                              c5_token_head_enabled=True, c5_n_token_classes=41,
                              notes=f"C5 R5 counter test 2, L{enc_n_layer} {train_steps // 1000}k, first 40k routes: {what}")
        spec["overrides"]["c4_eval_task_key"] = "wave_back"
        spec["overrides"]["diagnostics_wave_rollout_start_pass"] = "first_back"
        return spec

    ref = spec_for(f"c5_r5_wave_back_ref_rerun_{tag}", ["wave_back"], [1.0], "reference again: wave_back alone, CE + Dice, no noise, cosine")
    d = spec_for(f"c5_r5_wave_back_ce_noisep1_lrflat_{tag}", ["wave_back"], [1.0], "wave_back alone, CE beta 0 + noise p1, flat lr")
    d["overrides"].update(ce_noise); d["lr_horizon_steps"] = _M1_REPORT_LR_FLAT_HORIZON
    e = spec_for(f"c5_r5_wave_bwd2_ce_noisep1_cosine_{tag}", ["wave_back", "wave_origin"], [0.6, 0.4], "back/origin 60/40, CE beta 0 + noise p1, cosine")
    e["overrides"].update(ce_noise)
    f = spec_for(f"c5_r5_wave_bwd2_ce_noisep1_lrflat_{tag}", ["wave_back", "wave_origin"], [0.6, 0.4], "back/origin 60/40, CE beta 0 + noise p1, flat lr")
    f["overrides"].update(ce_noise); f["lr_horizon_steps"] = _M1_REPORT_LR_FLAT_HORIZON
    return [ref, d, e, f]


def c5_wave_back_counter_queue3(enc_n_layer: int = 2, train_steps: int = 8000) -> list[dict]:
    """Counter test, round 3 (2 Oct 2026). Round 2: the mix with wave_origin is what breaks wave_back (E: 180 false
    cells per sample against 68 alone; flat lr adds little; the loaders emit identical wave_back samples; the damage is
    the same for every number of fronts on the pad). What the mix has and a single task has not: the output token is no
    longer constant (BACK vs DRAW), so the token loss is a real classification through the shared trunk. In E the
    train loss ends at 0.128 against a grid loss of 0.036; in C (token constant) it ends at 0.022.
      G0  E with token loss weight 0     -> does the grid recover to the single-task level without the token loss?
      G1  E with token loss weight 0.1   -> the practical setting, if G0 recovers
    E = c5_r5_wave_bwd2_ce_noisep1_cosine_L2_8k_g128_store40k (back/origin 60/40, CE beta 0, noise p1, cosine)."""
    out = []
    for tag, w in (("tok0", 0.0), ("tok01", 0.1)):
        spec = next(sp for sp in c5_wave_back_counter_queue2(enc_n_layer, train_steps) if "bwd2_ce_noisep1_cosine" in sp["id"])
        spec["id"] = spec["id"].replace("bwd2_ce_noisep1_cosine", f"bwd2_ce_noisep1_cosine_{tag}")
        spec["notes"] = spec["notes"] + f", token loss weight {w:g}"
        spec["overrides"]["c5_token_loss_weight"] = w
        out.append(spec)
    return out


def c5_wave_mix(enc_n_layer: int = 4, train_steps: int = 16000, grid: int = 128) -> dict:
    """C5 R5: all six wave step types in one network (equal shares), tokens included."""
    keys = ["wave_first", "wave_step", "wave_reach", "wave_back", "wave_origin", "wave_draw"]
    spec = _c4_night_spec(run_id=f"c5_r5_wave_mix6_L{enc_n_layer}_{train_steps // 1000}k_g{grid}_store40k", keys=keys,
                          weights=[1 / 6] * 6, train_steps=train_steps, seed=4, enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k",
                          encoding_mode="onehot_v4_sp", worldsnap_max_train_items=40000, grid_out_size=grid,
                          c5_token_targets=True, c5_token_head_enabled=True, c5_n_token_classes=41,
                          notes=f"C5 R5 mix6: forward + backward wave passes, {enc_n_layer} blocks, {train_steps // 1000}k, {grid}x{grid}")
    spec["overrides"]["c4_eval_task_key"] = "wave_back"
    return spec


def _c5_wave_change_ce_overrides(**extra: Any) -> dict[str, Any]:
    """Plain CE + four-group scratchpad weighting (memo CHANGE_WEIGHTED_LOSS_2026-10-02)."""
    return {
        "c4_grid_loss": "ce",
        "c5_wave_change_group_ce": True,
        "c5_wave_change_ce_beta": 0.0,
        "c5_wave_change_pad_px": 0,
        **extra,
    }


def _c5_wave_long_run_compile_overrides(**extra: Any) -> dict[str, Any]:
    """CUDA compile defaults for C5 wave runs ≥ auto-compile threshold (probe B aligned in harness)."""
    return {"use_compile_dynamic": None, **extra}


_C5_FWD3_KEYS = ["wave_first", "wave_step", "wave_reach"]
_C5_FWD3_W_STEP50 = [0.25, 0.5, 0.25]  # emphasize wave_step in training mix
_C5_WAVE_FWD3_PAD_NOISE = {"p": 0.5, "thicken": 0.5, "shift": 0.5, "drop": 0.05, "add": 0.002}
_C5_WAVE_FWD3_NOISE_PROFILE_P1 = "p1"  # pad-noise recipe above; future profiles → p2, …
_C5_WAVE_TRAIN_ITEMS = 100_000  # full c5_wave_train.npz on c4_100k (Oct 2026; was 40k cap)
_C5_WAVE_STORE_TAG = "store100k"


def _c5_wave_fwd3_banneal_run_id(
    *,
    beta_high: float,
    enc_n_layer: int,
    grid: int,
    patch_size: int = 16,
    noise_profile: str | None = None,
) -> str:
    btag = int(beta_high) if beta_high == int(beta_high) else beta_high
    tag = f"banneal{btag}" + (f"_noise{noise_profile}" if noise_profile else "")
    mid = f"L{enc_n_layer}"
    if patch_size != 16:
        mid = f"{mid}_p{patch_size}"
    return f"c5_r5_wave_fwd3_{tag}_{mid}_g{grid}_{_C5_WAVE_STORE_TAG}"


def _c5_wave_fwd3_flat_spec(
    *,
    run_id: str,
    train_steps: int,
    notes: str,
    enc_n_layer: int = 4,
    grid: int = 128,
    patch_size: int = 16,
    force_restart: bool = False,
    extra_overrides: dict[str, Any] | None = None,
) -> dict:
    """Fwd3 mix, flat LR, step-less run id (resume by extending ``train_steps`` on same id)."""
    spec = _c5_wave_mix_ablation(
        run_id=run_id,
        keys=_C5_FWD3_KEYS,
        weights=_C5_FWD3_W_STEP50,
        train_steps=train_steps,
        eval_key="wave_step",
        notes=notes,
        enc_n_layer=enc_n_layer,
        grid=grid,
        patch_size=patch_size,
    )
    spec["lr_horizon_steps"] = _M1_REPORT_LR_FLAT_HORIZON
    spec["force_restart"] = force_restart
    ov: dict[str, Any] = {
        "t_train": train_steps,
        **_c5_wave_long_run_compile_overrides(),
        "c4_grid_loss": "ce",  # wave beta path; ablation inherits c4b ce_dice from _c4_learning_recipe
        "c5_wave_eval_decision_rule": True,
    }
    if extra_overrides:
        ov.update(extra_overrides)
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def c5_wave_fwd3_plain(train_steps: int = 16000, grid: int = 128) -> dict:
    """Plain CE fwd3 (β=0), flat lr, 50% wave_step mix."""
    return _c5_wave_fwd3_flat_spec(
        run_id=f"c5_r5_wave_fwd3_plain_L4_g{grid}_{_C5_WAVE_STORE_TAG}",
        train_steps=train_steps,
        notes=f"C5 fwd3 plain CE, step mix 50%, flat lr, L4 target {train_steps} steps",
        extra_overrides={
            "c5_wave_change_group_ce": False,
            "c5_wave_change_ce_beta": 0.0,
            "c5_wave_change_ce_beta_schedule": "",
        },
    )


def c5_wave_fwd3_plain_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Plain CE (β=0) + pad noise profile p1; isolates anneal vs static β under noise."""
    return _c5_wave_fwd3_flat_spec(
        run_id=f"c5_r5_wave_fwd3_plain_noisep1_L4_g{grid}_{_C5_WAVE_STORE_TAG}",
        train_steps=train_steps,
        notes=f"C5 fwd3 plain CE β=0, noisep1, step mix 50%, flat lr, L4, {train_steps} steps",
        extra_overrides={
            "c5_wave_change_group_ce": False,
            "c5_wave_change_ce_beta": 0.0,
            "c5_wave_change_ce_beta_schedule": "",
            "c5_wave_pad_noise": dict(_C5_WAVE_FWD3_PAD_NOISE),
        },
    )


def c5_wave_fwd3_beta_thirds_decay(
    train_steps: int = 16000,
    grid: int = 128,
    *,
    beta_high: float = 5.0,
    force_restart: bool = False,
    enc_n_layer: int = 4,
    patch_size: int = 16,
    pad_noise: dict | None = None,
    noise_profile: str = _C5_WAVE_FWD3_NOISE_PROFILE_P1,
) -> dict:
    """Change beta: first third high, middle linear→0, last third 0."""
    _patch_geometry(patch_size, grid_out=grid)
    run_id = _c5_wave_fwd3_banneal_run_id(
        beta_high=beta_high,
        enc_n_layer=enc_n_layer,
        grid=grid,
        patch_size=patch_size,
        noise_profile=noise_profile if pad_noise is not None else None,
    )
    extra: dict[str, Any] = {
        "c5_wave_change_group_ce": False,
        "c5_wave_change_ce_beta": beta_high,
        "c5_wave_change_ce_beta_high": beta_high,
        "c5_wave_change_ce_beta_schedule": "thirds_decay",
    }
    if pad_noise is not None:
        extra["c5_wave_pad_noise"] = pad_noise
    ps_note = f", P={patch_size}" if patch_size != 16 else ""
    noise_note = f", pad noise profile {noise_profile}" if pad_noise else ""
    return _c5_wave_fwd3_flat_spec(
        run_id=run_id,
        train_steps=train_steps,
        enc_n_layer=enc_n_layer,
        patch_size=patch_size,
        force_restart=force_restart,
        notes=(
            f"C5 fwd3 beta schedule thirds_decay peak={beta_high}, flat lr, "
            f"step mix 50%, L{enc_n_layer}{ps_note}{noise_note}, target {train_steps} steps"
        ),
        extra_overrides=extra,
    )


def c5_wave_fwd3_banneal5_L4(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5 fwd3, L4 P16, full train store (no pad noise)."""
    return c5_wave_fwd3_beta_thirds_decay(train_steps, grid, beta_high=5.0, enc_n_layer=4, patch_size=16)


def c5_wave_fwd3_L4_p16_100k_ablation_queue(train_steps: int = 16000) -> list[dict]:
    """L4 P16 on 100k: banneal β5 vs plain β=0, both with noisep1, same fwd3 mix."""
    return [
        c5_wave_fwd3_banneal5_noisep1(train_steps),
        c5_wave_fwd3_plain_noisep1(train_steps),
    ]


def c5_wave_fwd3_banneal5_L2(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5 fwd3 on L2 (same recipe as banneal L4; test whether shallow net catches up)."""
    return c5_wave_fwd3_beta_thirds_decay(train_steps, grid, beta_high=5.0, enc_n_layer=2)


def c5_wave_fwd3_banneal5_p32(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5 fwd3 with patch 32 (8×8 tokens on 128 grid) — prep for larger canvas."""
    return c5_wave_fwd3_beta_thirds_decay(train_steps, grid, beta_high=5.0, patch_size=32)


def c5_wave_fwd3_banneal5_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5 fwd3 + pad noise profile p1 (training pads only, clean targets)."""
    return c5_wave_fwd3_beta_thirds_decay(
        train_steps,
        grid,
        beta_high=5.0,
        pad_noise=dict(_C5_WAVE_FWD3_PAD_NOISE),
        noise_profile=_C5_WAVE_FWD3_NOISE_PROFILE_P1,
    )


def _c5_wave_fwd3_banneal5_noisep1_combo(
    train_steps: int,
    grid: int,
    *,
    enc_n_layer: int,
    patch_size: int,
) -> dict:
    return c5_wave_fwd3_beta_thirds_decay(
        train_steps,
        grid,
        beta_high=5.0,
        enc_n_layer=enc_n_layer,
        patch_size=patch_size,
        pad_noise=dict(_C5_WAVE_FWD3_PAD_NOISE),
        noise_profile=_C5_WAVE_FWD3_NOISE_PROFILE_P1,
    )


def c5_wave_fwd3_banneal5_p32_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5, P32, noise p1 — combine successful p32 geometry with rollout-robust pads."""
    return _c5_wave_fwd3_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=4, patch_size=32)


def c5_wave_fwd3_banneal5_L2_p32_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5, L2, P32, noise p1 — shallow net + coarser tokens + perturbed pads."""
    return _c5_wave_fwd3_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=2, patch_size=32)


def c5_wave_fwd3_banneal5_L2_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5, L2, P16, noise p1 — shallow net + standard tokens + perturbed pads."""
    return _c5_wave_fwd3_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=2, patch_size=16)


def c5_wave_fwd3_banneal5_p64_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Anneal β5, L4, P64 (4×4 tokens on 128 grid), noise p1 — coarsest patch in sweep."""
    return _c5_wave_fwd3_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=4, patch_size=64)


def c5_wave_fwd3_anneal_l2_p16_p64_noisep1_queue(train_steps: int = 16000) -> list[dict]:
    """L2+P16+noisep1 vs L4+P64+noisep1 (patch geometry bracket on noise recipe)."""
    return [
        c5_wave_fwd3_banneal5_L2_noisep1(train_steps),
        c5_wave_fwd3_banneal5_p64_noisep1(train_steps),
    ]


def c5_wave_fwd3_anneal_followup_queue(train_steps: int = 16000) -> list[dict]:
    """After plain L2 + banneal L4: p32 geometry, L2 anneal, L4 anneal + pad noise."""
    return [
        c5_wave_fwd3_banneal5_p32(train_steps),
        c5_wave_fwd3_banneal5_L2(train_steps),
        c5_wave_fwd3_banneal5_noisep1(train_steps),
    ]


def c5_wave_fwd3_anneal_p32_noise_queue(train_steps: int = 16000) -> list[dict]:
    """After p32 / L2 / noisep1 singles: stack p32 with noise; L2+p32+noise."""
    return [
        c5_wave_fwd3_banneal5_p32_noisep1(train_steps),
        c5_wave_fwd3_banneal5_L2_p32_noisep1(train_steps),
    ]


# Backward SFT: peel fronts (wave_back, token BACK) + terminal at start (wave_origin, token DRAW).
# No wave_draw — path is already on the pad; DONE pass adds little. Goal-side BACK is wave_reach (fwd mix).
_C5_BWD2_KEYS = ["wave_back", "wave_origin"]
_C5_BWD2_W_BACK60 = [0.6, 0.4]
_C5_BWD_ROLLOUT_START = "first_back"


def _c5_wave_bwd2_banneal_run_id(
    *,
    beta_high: float,
    enc_n_layer: int,
    grid: int,
    patch_size: int = 16,
    noise_profile: str | None = None,
) -> str:
    btag = int(beta_high) if beta_high == int(beta_high) else beta_high
    tag = f"banneal{btag}" + (f"_noise{noise_profile}" if noise_profile else "")
    mid = f"L{enc_n_layer}"
    if patch_size != 16:
        mid = f"{mid}_p{patch_size}"
    return f"c5_r5_wave_bwd2_{tag}_{mid}_g{grid}_{_C5_WAVE_STORE_TAG}"


def _c5_wave_bwd2_flat_spec(
    *,
    run_id: str,
    train_steps: int,
    notes: str,
    enc_n_layer: int = 4,
    grid: int = 128,
    patch_size: int = 16,
    force_restart: bool = False,
    extra_overrides: dict[str, Any] | None = None,
) -> dict:
    spec = _c5_wave_mix_ablation(
        run_id=run_id,
        keys=_C5_BWD2_KEYS,
        weights=_C5_BWD2_W_BACK60,
        train_steps=train_steps,
        eval_key="wave_back",
        notes=notes,
        enc_n_layer=enc_n_layer,
        grid=grid,
        patch_size=patch_size,
    )
    spec["lr_horizon_steps"] = _M1_REPORT_LR_FLAT_HORIZON
    spec["force_restart"] = force_restart
    ov: dict[str, Any] = {
        "t_train": train_steps,
        **_c5_wave_long_run_compile_overrides(),
        "c4_grid_loss": "ce",
        "c5_wave_eval_decision_rule": True,
        "diagnostics_wave_rollout_start_pass": _C5_BWD_ROLLOUT_START,
    }
    if extra_overrides:
        ov.update(extra_overrides)
    spec["overrides"] = {**spec["overrides"], **ov}
    return spec


def c5_wave_bwd2_beta_thirds_decay(
    train_steps: int = 16000,
    grid: int = 128,
    *,
    beta_high: float = 5.0,
    force_restart: bool = False,
    enc_n_layer: int = 4,
    patch_size: int = 16,
    pad_noise: dict | None = None,
    noise_profile: str = _C5_WAVE_FWD3_NOISE_PROFILE_P1,
) -> dict:
    _patch_geometry(patch_size, grid_out=grid)
    run_id = _c5_wave_bwd2_banneal_run_id(
        beta_high=beta_high,
        enc_n_layer=enc_n_layer,
        grid=grid,
        patch_size=patch_size,
        noise_profile=noise_profile if pad_noise is not None else None,
    )
    extra: dict[str, Any] = {
        "c5_wave_change_group_ce": False,
        "c5_wave_change_ce_beta": beta_high,
        "c5_wave_change_ce_beta_high": beta_high,
        "c5_wave_change_ce_beta_schedule": "thirds_decay",
    }
    if pad_noise is not None:
        extra["c5_wave_pad_noise"] = pad_noise
    ps_note = f", P={patch_size}" if patch_size != 16 else ""
    noise_note = f", pad noise profile {noise_profile}" if pad_noise else ""
    return _c5_wave_bwd2_flat_spec(
        run_id=run_id,
        train_steps=train_steps,
        enc_n_layer=enc_n_layer,
        patch_size=patch_size,
        force_restart=force_restart,
        notes=(
            f"C5 bwd2 beta schedule thirds_decay peak={beta_high}, flat lr, "
            f"60% wave_back / 40% origin, L{enc_n_layer}{ps_note}{noise_note}, rollout {_C5_BWD_ROLLOUT_START}, {train_steps} steps"
        ),
        extra_overrides=extra,
    )


def _c5_wave_bwd2_banneal5_noisep1_combo(
    train_steps: int, grid: int, *, enc_n_layer: int, patch_size: int,
) -> dict:
    return c5_wave_bwd2_beta_thirds_decay(
        train_steps,
        grid,
        beta_high=5.0,
        enc_n_layer=enc_n_layer,
        patch_size=patch_size,
        pad_noise=dict(_C5_WAVE_FWD3_PAD_NOISE),
        noise_profile=_C5_WAVE_FWD3_NOISE_PROFILE_P1,
    )


def c5_wave_bwd2_banneal5_L4_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Bwd2 L4 P16, banneal β5, noisep1, 100k store."""
    return _c5_wave_bwd2_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=4, patch_size=16)


def c5_wave_bwd2_plain_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    """Bwd2 L4 P16, plain CE β=0, noisep1 — pairs with banneal5_L4_noisep1."""
    return _c5_wave_bwd2_flat_spec(
        run_id=f"c5_r5_wave_bwd2_plain_noisep1_L4_g{grid}_{_C5_WAVE_STORE_TAG}",
        train_steps=train_steps,
        enc_n_layer=4,
        patch_size=16,
        notes=(
            f"C5 bwd2 plain CE β=0, noisep1, 60/40 back/origin, L4 P16, rollout {_C5_BWD_ROLLOUT_START}, {train_steps} steps"
        ),
        extra_overrides={
            "c5_wave_change_group_ce": False,
            "c5_wave_change_ce_beta": 0.0,
            "c5_wave_change_ce_beta_schedule": "",
            "c5_wave_pad_noise": dict(_C5_WAVE_FWD3_PAD_NOISE),
        },
    )


def c5_wave_bwd2_L4_p16_noisep1_ablation_queue(train_steps: int = 16000) -> list[dict]:
    """L4 P16 bwd2 on 100k: banneal β5 vs plain β=0, both noisep1 (same as fwd ablation axes)."""
    return [
        c5_wave_bwd2_banneal5_L4_noisep1(train_steps),
        c5_wave_bwd2_plain_noisep1(train_steps),
    ]


def c5_wave_bwd2_banneal5_L2_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    return _c5_wave_bwd2_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=2, patch_size=16)


def c5_wave_bwd2_banneal5_L4_p32_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    return _c5_wave_bwd2_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=4, patch_size=32)


def c5_wave_bwd2_banneal5_L2_p32_noisep1(train_steps: int = 16000, grid: int = 128) -> dict:
    return _c5_wave_bwd2_banneal5_noisep1_combo(train_steps, grid, enc_n_layer=2, patch_size=32)


def c5_wave_bwd2_anneal_noisep1_queue(train_steps: int = 16000) -> list[dict]:
    return [
        c5_wave_bwd2_banneal5_L2_noisep1(train_steps),
        c5_wave_bwd2_banneal5_L4_p32_noisep1(train_steps),
        c5_wave_bwd2_banneal5_L2_p32_noisep1(train_steps),
    ]


def c5_wave_fwd3_beta2_L2_staple(train_steps: int = 16000, grid: int = 128) -> dict:
    """Reference: L2, beta=2, flat lr, 50% step mix (extend steps on same run id)."""
    return _c5_wave_fwd3_flat_spec(
        run_id=f"c5_r5_wave_fwd3_beta2_L2_g{grid}_store40k",
        train_steps=train_steps,
        enc_n_layer=2,
        notes=f"C5 staple: L2 fwd3 beta=2 flat lr, step mix 50%, target {train_steps} steps",
        extra_overrides=_c5_wave_beta_overrides(beta=2.0),
    )


def c5_wave_oct2_queue() -> list[dict]:
    return [
        c5_wave_fwd3_plain(16000),
        c5_wave_fwd3_beta_thirds_decay(16000, beta_high=5.0),
        c5_wave_fwd3_beta2_L2_staple(16000),
    ]


def _c5_wave_beta_overrides(*, beta: float = 10.0, **extra: Any) -> dict[str, Any]:
    """Plain CE + per-task fg weights; changing cells scaled by (1 + beta)."""
    return {
        "c4_grid_loss": "ce",
        "c5_wave_change_group_ce": False,
        "c5_wave_change_ce_beta": beta,
        "c5_wave_change_pad_px": 0,
        "c5_wave_eval_decision_rule": True,
        **_c5_wave_long_run_compile_overrides(),
        **extra,
    }


def _c5_wave_mix_ablation(
    *,
    run_id: str,
    keys: list[str],
    weights: list[float],
    train_steps: int,
    eval_key: str,
    notes: str,
    enc_n_layer: int = 4,
    grid: int = 128,
    patch_size: int = 16,
) -> dict:
    spec = _c4_night_spec(
        run_id=run_id, keys=keys, weights=weights, train_steps=train_steps, seed=4,
        enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
        worldsnap_max_train_items=_C5_WAVE_TRAIN_ITEMS, grid_out_size=grid, patch_size=patch_size,
        c5_token_targets=True, c5_token_head_enabled=True, c5_n_token_classes=41,
        notes=notes,
    )
    spec["overrides"]["c4_eval_task_key"] = eval_key
    return spec


def c5_wave_mix_forward3(train_steps: int = 4000, grid: int = 128) -> dict:
    """(i) Forward only: first / step / reach — no backward interference."""
    keys = ["wave_first", "wave_step", "wave_reach"]
    return _c5_wave_mix_ablation(
        run_id=f"c5_r5_wave_mix_fwd3_L4_{train_steps // 1000}k_g{grid}_store40k",
        keys=keys, weights=[1 / 3] * 3, train_steps=train_steps, eval_key="wave_step",
        notes=f"C5 wave ablation (i): forward mix ⅓ each, L4 {train_steps // 1000}k; watch c5_wave_new_recall_wave_step",
    )


def c5_wave_mix_back2(train_steps: int = 4000, grid: int = 128) -> dict:
    """(ii) Backward only: back + origin (no draw)."""
    keys = ["wave_back", "wave_origin"]
    return _c5_wave_mix_ablation(
        run_id=f"c5_r5_wave_mix_bwd2_L4_{train_steps // 1000}k_g{grid}_store40k",
        keys=keys, weights=[0.5, 0.5], train_steps=train_steps, eval_key="wave_back",
        notes=f"C5 wave ablation (ii): backward mix ½ each, L4 {train_steps // 1000}k; no wave_draw",
    )


def c5_wave_mix_step_back(train_steps: int = 16000, grid: int = 128) -> dict:
    """(iii) Emit vs peel: only wave_step and wave_back (competition diagnostic)."""
    return _c5_wave_mix_ablation(
        run_id=f"c5_r5_wave_mix_step_back_L4_{train_steps // 1000}k_g{grid}_store40k",
        keys=["wave_step", "wave_back"], weights=[0.5, 0.5], train_steps=train_steps, eval_key="wave_step",
        notes=f"C5 wave ablation (iii): ½ step + ½ back, L4 {train_steps // 1000}k; primary metric new recall on both",
    )


def c5_wave_ablation_queue() -> list[dict]:
    """Legacy 4k probe + step↔back 16k (4k mixes underperformed singles; use half_duration_queue)."""
    return [
        c5_wave_mix_forward3(train_steps=4000),
        c5_wave_mix_back2(train_steps=4000),
        c5_wave_mix_step_back(),
    ]


def _c5_wave_resume_4k_to(spec: dict, *, train_steps: int, label: str) -> dict:
    """Same ``run_id`` / checkpoint as the 4k ablation; harness auto-resumes from step ~3999."""
    run_id = spec["id"]
    if not run_id.endswith("_4k_g128_store40k"):
        raise ValueError(f"expected 4k run_id to resume, got {run_id!r}")
    spec["train_steps"] = train_steps
    spec["lr_horizon_steps"] = train_steps
    spec["force_restart"] = False
    spec["notes"] = (
        f"{label}: continue {run_id} from 4k checkpoint → {train_steps // 1000}k total steps "
        f"(do not change run_id — harness auto_resume)"
    )
    return spec


def c5_wave_mix5_cgce(train_steps: int = 8000, enc_n_layer: int = 4, grid: int = 128) -> dict:
    """Five wave tasks (no draw), equal mix, change-group CE."""
    keys = ["wave_first", "wave_step", "wave_reach", "wave_back", "wave_origin"]
    spec = _c5_wave_mix_ablation(
        run_id=f"c5_r5_wave_mix5_cgce_L{enc_n_layer}_{train_steps // 1000}k_g{grid}_store40k",
        keys=keys,
        weights=[1 / 5] * 5,
        train_steps=train_steps,
        eval_key="wave_step",
        notes=f"C5 wave mix5 + change-group CE, L{enc_n_layer} {train_steps // 1000}k",
        enc_n_layer=enc_n_layer,
    )
    spec["overrides"] = {**spec["overrides"], **_c5_wave_change_ce_overrides()}
    return spec


def c5_wave_mix_forward3_cgce(train_steps: int = 8000, grid: int = 128) -> dict:
    spec = c5_wave_mix_forward3(train_steps=train_steps, grid=grid)
    spec["id"] = f"c5_r5_wave_mix_fwd3_cgce_L4_{train_steps // 1000}k_g{grid}_store40k"
    spec["lr_horizon_steps"] = train_steps
    spec["force_restart"] = True
    spec["notes"] = f"C5 fwd3 + change-group CE, L4 {train_steps // 1000}k"
    spec["overrides"] = {**spec["overrides"], **_c5_wave_change_ce_overrides()}
    return spec


def c5_wave_mix_back2_cgce(train_steps: int = 8000, grid: int = 128) -> dict:
    spec = c5_wave_mix_back2(train_steps=train_steps, grid=grid)
    spec["id"] = f"c5_r5_wave_mix_bwd2_cgce_L4_{train_steps // 1000}k_g{grid}_store40k"
    spec["lr_horizon_steps"] = train_steps
    spec["force_restart"] = True
    spec["notes"] = f"C5 bwd2 + change-group CE, L4 {train_steps // 1000}k"
    spec["overrides"] = {**spec["overrides"], **_c5_wave_change_ce_overrides()}
    return spec


def c5_wave_change_ce_smoke() -> dict:
    """~500 steps on fwd3 only; verify loss runs with group CE."""
    spec = c5_wave_mix_forward3_cgce(train_steps=500, grid=128)
    spec["id"] = "c5_r5_wave_change_ce_smoke_500"
    spec["force_restart"] = True
    spec["overrides"]["eval_interval"] = 250
    spec["overrides"]["diagnostics_enabled"] = True
    return spec


def c5_wave_compile_smoke(*, use_compile_dynamic: bool = False) -> dict:
    """Short fwd3 beta run: harness auto-compile (lowered min_steps), probe B=train B; A/B dynamic flag."""
    steps = 512
    spec = c5_wave_mix_forward3_beta(train_steps=steps, beta=2.0)
    tag = "dyn" if use_compile_dynamic else "static"
    spec["id"] = f"c5_r5_wave_compile_smoke_512_{tag}"
    spec["train_steps"] = steps
    spec["lr_horizon_steps"] = steps
    spec["force_restart"] = True
    spec["notes"] = (
        f"Compile smoke only: auto compile (min_steps=400), dynamic={use_compile_dynamic}; "
        "probe batch follows train batch_size when compile on; not a training-quality run"
    )
    spec["overrides"] = {
        **spec["overrides"],
        "use_compile_min_train_steps": 400,
        "use_compile_dynamic": use_compile_dynamic,
        "eval_interval": 256,
        "harness_save_checkpoints": False,
    }
    return spec


def c5_wave_change_ce_queue(*, fwd_bwd_steps: int = 4000, mix5_steps: int = 4000) -> list[dict]:
    """fwd3 + bwd2 + mix5 with change-group CE (fresh ``*_cgce_*`` run ids)."""
    return [
        c5_wave_mix_forward3_cgce(train_steps=fwd_bwd_steps),
        c5_wave_mix_back2_cgce(train_steps=fwd_bwd_steps),
        c5_wave_mix5_cgce(train_steps=mix5_steps),
    ]


def c5_wave_mix_forward3_beta(
    train_steps: int = 4000, grid: int = 128, *, beta: float = 10.0
) -> dict:
    """Forward mix + soft change upweight (1 + beta) on pad≠target cells; keeps fg table weights."""
    btag = int(beta) if beta == int(beta) else beta
    spec = c5_wave_mix_forward3(train_steps=train_steps, grid=grid)
    spec["id"] = f"c5_r5_wave_mix_fwd3_beta{btag}_L4_{train_steps // 1000}k_g{grid}_store40k"
    spec["lr_horizon_steps"] = train_steps
    spec["force_restart"] = True
    spec["notes"] = (
        f"C5 fwd3 + change beta={beta} (plain CE, fg weights), L4 {train_steps // 1000}k; "
        "IoU/false_fg gate, not new recall alone"
    )
    spec["overrides"] = {**spec["overrides"], **_c5_wave_beta_overrides(beta=beta)}
    return spec


def c5_wave_mix_forward3_cgce_pad(
    train_steps: int = 4000, grid: int = 128, *, pad_px: int = 10
) -> dict:
    """Forward mix + group CE with a dilated halo (fifth group) around core change cells."""
    spec = c5_wave_mix_forward3(train_steps=train_steps, grid=grid)
    spec["id"] = f"c5_r5_wave_mix_fwd3_cgce_pad{pad_px}_L4_{train_steps // 1000}k_g{grid}_store40k"
    spec["lr_horizon_steps"] = train_steps
    spec["force_restart"] = True
    spec["notes"] = (
        f"C5 fwd3 + change-group CE + {pad_px}px change halo, L4 {train_steps // 1000}k"
    )
    spec["overrides"] = {
        **spec["overrides"],
        **_c5_wave_change_ce_overrides(c5_wave_change_pad_px=pad_px),
    }
    return spec


def c5_wave_change_fwd4k_queue(*, beta: float = 10.0, pad_px: int = 10) -> list[dict]:
    """Two forward-only 4k probes: beta upweight vs group CE + padded change focus."""
    return [
        c5_wave_mix_forward3_beta(train_steps=4000, beta=beta),
        c5_wave_mix_forward3_cgce_pad(train_steps=4000, pad_px=pad_px),
    ]


def c5_wave_change_fwd_beta_queue(
    betas: tuple[float, ...] = (2.0, 5.0), *, train_steps: int = 4000, grid: int = 128
) -> list[dict]:
    """Forward mix @ 4k for each change beta (same recipe as beta10)."""
    return [c5_wave_mix_forward3_beta(train_steps=train_steps, grid=grid, beta=b) for b in betas]


def c5_wave_change_fwd_beta_resume_queue(
    betas: tuple[float, ...] = (2.0, 5.0, 10.0),
    *,
    train_steps: int = 12000,
    grid: int = 128,
) -> list[dict]:
    """Same ``*_4k_*`` run ids; cosine LR with ``lr_horizon_steps=train_steps``; harness auto-resumes."""
    out: list[dict] = []
    for b in betas:
        spec = c5_wave_mix_forward3_beta(train_steps=4000, grid=grid, beta=b)
        out.append(
            _c5_wave_resume_4k_to(spec, train_steps=train_steps, label=f"fwd3 beta={b} cosine→{train_steps // 1000}k")
        )
    return out


def c5_wave_change_fwd_beta_resume_8k_queue(
    betas: tuple[float, ...] = (2.0, 5.0, 10.0),
    *,
    train_steps: int = 8000,
    grid: int = 128,
) -> list[dict]:
    return c5_wave_change_fwd_beta_resume_queue(betas, train_steps=train_steps, grid=grid)


def c5_wave_mix_forward3_beta_lrflat(
    train_steps: int = 12000, grid: int = 128, *, beta: float = 10.0
) -> dict:
    """Fresh fwd3 + change beta, flat LR (``lr_horizon`` = 50M), own ``*_lrflat_*12k_*`` run id."""
    btag = int(beta) if beta == int(beta) else beta
    spec = c5_wave_mix_forward3_beta(train_steps=train_steps, grid=grid, beta=beta)
    spec["id"] = f"c5_r5_wave_mix_fwd3_beta{btag}_lrflat_L4_{train_steps // 1000}k_g{grid}_store40k"
    spec["lr_horizon_steps"] = _M1_REPORT_LR_FLAT_HORIZON
    spec["force_restart"] = True
    spec["notes"] = (
        f"C5 fwd3 change beta={beta}, flat lr 1e-3 (horizon={_M1_REPORT_LR_FLAT_HORIZON}), "
        f"L4 {train_steps // 1000}k fresh run_id"
    )
    return spec


def c5_wave_change_fwd_beta_explore_12k_queue(
    betas: tuple[float, ...] = (2.0, 5.0, 10.0),
    *,
    train_steps: int = 12000,
    grid: int = 128,
) -> list[dict]:
    """Per beta: resume cosine ``*_4k_*`` → 12k, then fresh flat-lr 12k sibling."""
    out = c5_wave_change_fwd_beta_resume_queue(betas, train_steps=train_steps, grid=grid)
    for b in betas:
        out.append(c5_wave_mix_forward3_beta_lrflat(train_steps=train_steps, grid=grid, beta=b))
    return out


def c5_wave_change_fwd_beta_lrflat_12k_queue(
    betas: tuple[float, ...] = (2.0, 5.0),
    *,
    train_steps: int = 12000,
    grid: int = 128,
) -> list[dict]:
    """Fresh flat-lr 12k only (separate ``*_lrflat_*`` run ids; compile-friendly)."""
    return [
        c5_wave_mix_forward3_beta_lrflat(train_steps=train_steps, grid=grid, beta=b) for b in betas
    ]


def c5_wave_half_duration_queue(train_steps: int = 8000) -> list[dict]:
    """Extend the 4k forward/back mixes to 8k on the **same** checkpoints (not new run_ids).

    Gate before step↔back mix: ``c5_wave_new_recall_wave_step`` / ``wave_back`` on val via
    ``python -m diagnostics.wave_eval_dashboard <run_id>`` (singles @8k: step new ~0.65, back ~0.64).
    """
    return [
        _c5_wave_resume_4k_to(c5_wave_mix_forward3(train_steps=4000), train_steps=train_steps, label="(i) fwd3"),
        _c5_wave_resume_4k_to(c5_wave_mix_back2(train_steps=4000), train_steps=train_steps, label="(ii) bwd2"),
    ]


def c5_wave_mix_natural(enc_n_layer: int = 4, train_steps: int = 24000, grid: int = 128, beta: float = 5.0,
                        pad_noise: dict | None = None, lr_flat: bool = False) -> dict:
    """C5 R5 (2 Oct): the wave search sampled the way a rollout produces it: wave_fwd (k uniform over the route's
    forward passes, token BACK on the last), wave_bwd (j uniform over the fronts, DRAW on the last), wave_draw; shares
    4 : 4 : 1 (passes per route). Optional pad noise in training (dict, see config.c5_wave_pad_noise)."""
    tag = f"nat3_beta{beta:g}" + ("_noise" if pad_noise else "") + ("_lrflat" if lr_flat else "")
    spec = _c4_night_spec(run_id=f"c5_r5_wave_{tag}_L{enc_n_layer}_{train_steps // 1000}k_g{grid}_{_C5_WAVE_STORE_TAG}",
                          keys=["wave_fwd", "wave_bwd", "wave_draw"], weights=[4 / 9, 4 / 9, 1 / 9], train_steps=train_steps,
                          seed=4, enc_n_layer=enc_n_layer, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          worldsnap_max_train_items=_C5_WAVE_TRAIN_ITEMS, grid_out_size=grid, c5_token_targets=True, c5_token_head_enabled=True,
                          c5_n_token_classes=41, c5_wave_pad_noise=pad_noise,
                          notes=f"C5 R5 natural mix: wave_fwd/wave_bwd/wave_draw 4:4:1, beta {beta:g}, {enc_n_layer} blocks, {train_steps // 1000}k, {grid}x{grid}"
                                + (f", pad noise {pad_noise}" if pad_noise else ""))
    spec["overrides"]["c4_eval_task_key"] = "wave_bwd"
    spec["overrides"].update(_c5_wave_beta_overrides(beta=beta))      # Marvin's loss recipe (plain CE, beta on changed cells)
    if lr_flat:
        spec["lr_horizon_steps"] = 10 ** 9                               # cosine horizon far beyond the run = flat lr (as m1 lrflat)
    return spec


_C5_TOKEN_LOSS_WEIGHT = 0.1   # 2 Oct counter tests: at 1.0 the token loss dominates any wave mix and blurs the grid


def c5_wave_night_queue_2oct(bwd_steps: int = 16000, long_steps: int = 48000) -> list[dict]:
    """Night queue 2/3 Oct 2026, with what the counter tests showed: token loss weight 0.1, cosine lr over the run
    (flat lr made the mix worse), pad noise p1, 100k store.
      1-3  bwd2 (back/origin 60/40), L2, 16k: annealed beta 5 / plain beta 0 / annealed beta 5 without the lit-cell
           weight (c4_weight_alpha 0 -> weight 1, the output is then calibrated at 0.5 once beta has decayed)
      4-5  the whole search in one network: wave_fwd / wave_bwd / wave_draw 4:4:1, annealed beta 5, 48k, L4 then L2
    Read with harness/diagnostics/wave_rollout_eval.py RUN_ID (rollouts over the val routes, no figures)."""
    noise = dict(_C5_WAVE_FWD3_PAD_NOISE)
    anneal = {"c5_wave_change_ce_beta": 5.0, "c5_wave_change_ce_beta_high": 5.0, "c5_wave_change_ce_beta_schedule": "thirds_decay"}
    tag = f"L2_{bwd_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"

    def bwd2(name: str, what: str, spec: dict, **extra) -> dict:
        spec["id"] = f"c5_r5_wave_bwd2_tok01_{name}_{tag}"
        spec["lr_horizon_steps"] = bwd_steps                              # cosine over the run
        spec["notes"] = f"C5 bwd2 60/40 back/origin, token loss weight {_C5_TOKEN_LOSS_WEIGHT}, cosine, noise p1, L2 {bwd_steps} steps: {what}"
        spec["overrides"].update({"c5_token_loss_weight": _C5_TOKEN_LOSS_WEIGHT, **extra})
        return spec

    out = [
        bwd2("banneal5_noisep1", "annealed beta 5",
             c5_wave_bwd2_beta_thirds_decay(bwd_steps, beta_high=5.0, enc_n_layer=2, pad_noise=noise)),
        bwd2("plain_noisep1", "plain beta 0",
             _c5_wave_bwd2_flat_spec(run_id="x", train_steps=bwd_steps, enc_n_layer=2, notes="",
                                     extra_overrides={"c5_wave_change_group_ce": False, "c5_wave_change_ce_beta": 0.0,
                                                      "c5_wave_change_ce_beta_schedule": "", "c5_wave_pad_noise": noise})),
        bwd2("banneal5_alpha0_noisep1", "annealed beta 5, no lit-cell weight",
             c5_wave_bwd2_beta_thirds_decay(bwd_steps, beta_high=5.0, enc_n_layer=2, pad_noise=noise), c4_weight_alpha=0.0),
    ]
    for L in (4, 2):
        spec = c5_wave_mix_natural(enc_n_layer=L, train_steps=long_steps, beta=5.0, pad_noise=noise)
        spec["id"] = f"c5_r5_wave_nat3_tok01_banneal5_noisep1_L{L}_{long_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"
        spec["lr_horizon_steps"] = long_steps
        spec["notes"] = (f"C5 whole search in one network: wave_fwd/wave_bwd/wave_draw 4:4:1, token loss weight {_C5_TOKEN_LOSS_WEIGHT}, "
                         f"annealed beta 5, cosine, noise p1, L{L} {long_steps} steps")
        spec["overrides"].update({"c5_token_loss_weight": _C5_TOKEN_LOSS_WEIGHT, "t_train": long_steps, **anneal,
                                  "diagnostics_wave_rollout_start_pass": 1})
        out.append(spec)
    return out


_C5_OWN_PADS_SOURCE_RUN = "c5_r5_wave_nat3_tok01_banneal5_noisep1_L4_48k_g128_store100k"   # the network whose pads are stored


def c5_wave_own_pads_queue(cont_steps: int = 16000, long_steps: int = 48000, p: float = 0.5, cont_lr: float = 3e-4) -> list[dict]:
    """C5 R5 own pads (3 Oct 2026): train on the network's own scratchpads, not only the teacher's. The store
    c5_own_pads_train.npz (spatial_data/c5_own_pads.py, built from the L4 48k checkpoint under the teacher's pass
    schedule, first 40k train routes) replaces the clean pad in with probability p by the network's output of the
    pass that produces that input; then the pad noise p1 as before; the target stays the teacher's clean next pad.
    The store covers routes 0..39,999 of the 100k the runs load, so the share of wave samples on own pads is
    p x 0.4 (x ~0.9: the first forward pass reads the start disc and is never replaced).
      a  continue the L4 48k network for cont_steps with own pads: fresh optimizer, step counter from 0, cosine
         from cont_lr (the 48k run ended at 1e-4; 1e-3 again would be a warm restart), beta 0 throughout (where the
         48k anneal ended), token weight 0.1, noise p1
      b  the L4 48k recipe of c5_wave_night_queue_2oct() from scratch, own pads from the start of the second third
         (step 16k of 48k, where the beta anneal starts to decay; docs/plans/NEXT_STEPS_2026-10-03.md §3), switched
         on at once, not ramped
    Read with harness/diagnostics/wave_rollout_eval.py RUN_ID (free-running success against the 75 % of the source)."""
    import copy
    own = {"path": "c5_own_pads_train.npz", "p": p}
    ptag = f"ownp{int(round(p * 100)):02d}"
    L4 = next(sp for sp in c5_wave_night_queue_2oct(long_steps=long_steps) if "_L4_" in sp["id"] and "nat3" in sp["id"])

    a = copy.deepcopy(L4)
    a["id"] = f"c5_r5_wave_nat3_tok01_{ptag}_from48k_L4_{cont_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"
    a["train_steps"] = cont_steps
    a["lr_horizon_steps"] = cont_steps
    a["init_checkpoint"] = str(config.REPO_ROOT / "runs" / "checkpoints" / f"{_C5_OWN_PADS_SOURCE_RUN}.pt")
    a["load_optimizer"] = False
    a["reset_step_counter"] = True
    a["force_restart"] = True          # with init_checkpoint the harness never auto-resumes: a stopped run starts over
    a["overrides"].update({"t_train": cont_steps, "learning_rate": cont_lr, "c5_wave_own_pads": own,
                           "c5_wave_change_ce_beta": 0.0, "c5_wave_change_ce_beta_schedule": ""})
    a["notes"] = (f"C5 own pads (a): {_C5_OWN_PADS_SOURCE_RUN} continued {cont_steps} steps, own pads p {p} (first 40k routes), "
                  f"noise p1, beta 0, cosine from {cont_lr:g}, fresh optimizer")

    b = copy.deepcopy(L4)
    b["id"] = f"c5_r5_wave_nat3_tok01_banneal5_noisep1_{ptag}_L4_{long_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"
    b["overrides"]["c5_wave_own_pads"] = {**own, "start_frac": 1 / 3}
    b["notes"] = L4["notes"] + f"; own pads p {p} from {_C5_OWN_PADS_SOURCE_RUN} (first 40k routes) from step {long_steps // 3} on, from scratch"
    return [a, b]


def c5_wave_natural_noise_round(source_run: str, round_no: int, cont_steps: int = 16000, p: float = 0.5,
                                cont_lr: float = 1e-4) -> dict:
    """Natural noise, next round (4 Oct 2026): continue `source_run` for cont_steps on the store of ITS OWN drawings
    (c5_own_pads_train_r{round_no}.npz, generated from that checkpoint over all 100k train routes with
    spatial_data/c5_own_pads.py --out). Round 1 was c5_wave_own_pads_queue()[0] (store from the 48k network, +14 points
    free-running on De Pijp val, +16 on the new map); each further round repairs the current network's own mistakes.
    Same settings as round 1: fresh optimizer, cosine from cont_lr, beta 0, token weight 0.1, noise p1."""
    spec = c5_wave_own_pads_queue(cont_steps=cont_steps, p=p, cont_lr=cont_lr)[0]
    spec["id"] = f"c5_r5_wave_nat3_tok01_ownp{int(round(p * 100)):02d}_r{round_no}_L4_{cont_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"
    spec["init_checkpoint"] = str(config.REPO_ROOT / "runs" / "checkpoints" / f"{source_run}.pt")
    spec["overrides"]["c5_wave_own_pads"] = {"path": f"c5_own_pads_train_r{round_no}.npz", "p": p}
    spec["notes"] = (f"C5 natural noise round {round_no}: {source_run} continued {cont_steps} steps on its own drawings "
                     f"(all 100k routes), p {p}, noise p1, beta 0, cosine from {cont_lr:g}, fresh optimizer")
    return spec


_C5_WAVE_BEST_256 = "c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k"   # 78 % free-running, De Pijp val 500
_C5_WAVE_512_STORE_TAG = "zc512"     # Zuid + Centrum, 512 m windows, store c4_512_20k (routes 40-500 m straight)


def _c5_wave_512(spec: dict, *, patch_size: int = 16, batch_size: int = 128) -> dict:
    """Move a 256 m wave spec to 512 m windows on the Zuid + Centrum store c4_512_20k: 512 px input at 1 m, output
    256 x 256 cells of 2 m (the cell size of the 256 m runs, so fronts, path width and pad noise mean the same in
    metres). P=16 gives 1024 patch tokens, P=32 gives 256 tokens of 16 x 16 cells each. The lit-cell weight comes from
    the store's own task stats at grid 256 (c4_task_stats_g256.json), as it did at 256 m."""
    _patch_geometry(patch_size, img_size=512, grid_out=256)
    spec["overrides"].update(img_size=512, grid_out_size=256, patch_size=patch_size, batch_size=batch_size,
                             district_name="zuid_centrum", cropset="v4ns", c4_routes_subdir="c4_512_20k",
                             worldsnap_max_train_items=None, diagnostics_pred_c4_reference_subdir=None)
    return spec


def c5_wave_512_queue(patch_size: int = 16, enc_n_layer: int = 4, batch_size: int | None = None, long_steps: int = 48000,
                      cont_steps: int = 16000, cont_lr: float = 1e-4) -> list[dict]:
    """C5 R5 at 512 m windows (4 Oct 2026), Zuid + Centrum store c4_512_20k (20k train routes, 40-500 m straight,
    walking up to ~1.1 km, up to 28 fronts at 40 m; wave data c5_wave_{split}.npz in uint16).
      a  continue the best 256 m network (_C5_WAVE_BEST_256) at 512 m for cont_steps: same weights (no learned weight
         depends on the window: sin-cos positions and the relative-bias table are rebuilt for 32 x 32 patches, the
         bias is per clipped offset, the grid head is per patch), fresh optimizer, cosine from cont_lr, beta 0,
         token weight 0.1, pad noise p1, no own pads (there is no 512 m own-pad store yet). P=16 only: a P=32 model
         cannot take the 256 m weights (patch embedding and grid head change shape).
      b  the whole-search recipe from scratch at 512 m, as the L4 48k run of c5_wave_night_queue_2oct(): wave_fwd /
         wave_bwd / wave_draw 4:4:1, annealed beta 5, cosine over long_steps, token weight 0.1, noise p1.
    Batch (4 Oct, RTX 3060 12 GB, measured): at P=16 (1024 tokens) batch 128 does not fit, 64 does (~10.6 GB peak
    compiled, model only); at P=32 (256 tokens) 128 fits. batch_size=None picks 64 at P=16 and 128 at P=32; the learning
    rate is not rescaled for batch 64 (an open choice; at 48k steps x 64 the run sees half the samples of the 256 m
    run). Cost: ~136 samples/s at L4 P=16 b64 compiled -> 48k x 64 in ~6.3 h model time on the 3060 (48k x 128 ~12.5 h);
    ~620 samples/s at L4 P=32 b128 -> ~2.8 h.
    Read with harness/diagnostics/wave_rollout_eval.py RUN_ID --crops ../data/amsterdam/zuid_centrum/crops/v4ns
    --store c4_512_20k --n 500 (and --split test once); the 256 m network's zero-shot number on the same routes is the
    baseline (wave_rollout_eval_zuid_centrum_c4_512_20k_val.json of _C5_WAVE_BEST_256: 2.6 % solved, 26 % on 2-3
    fronts; the first pass fails where the start lies beyond the trained 16 x 16 patch positions)."""
    import copy
    if batch_size is None:
        batch_size = 128 if patch_size >= 32 else 64
    L4 = next(sp for sp in c5_wave_night_queue_2oct(long_steps=long_steps) if "_L4_" in sp["id"] and "nat3" in sp["id"])
    ptag = "" if patch_size == 16 else f"_p{patch_size}"
    btag = "" if batch_size == 128 else f"_b{batch_size}"

    a = _c5_wave_512(copy.deepcopy(L4), patch_size=16, batch_size=min(batch_size, 64))
    a["id"] = f"c5_r5_wave_nat3_tok01_from_ownp50_L4_{cont_steps // 1000}k_g256_b{min(batch_size, 64)}_{_C5_WAVE_512_STORE_TAG}"
    a["train_steps"] = cont_steps
    a["lr_horizon_steps"] = cont_steps
    a["init_checkpoint"] = str(config.REPO_ROOT / "runs" / "checkpoints" / f"{_C5_WAVE_BEST_256}.pt")
    a["load_optimizer"] = False
    a["reset_step_counter"] = True
    a["force_restart"] = True          # with init_checkpoint the harness never auto-resumes: a stopped run starts over
    a["overrides"].update({"t_train": cont_steps, "learning_rate": cont_lr,
                           "c5_wave_change_ce_beta": 0.0, "c5_wave_change_ce_beta_schedule": ""})
    a["notes"] = (f"C5 512 m (a): {_C5_WAVE_BEST_256} continued {cont_steps} steps on Zuid + Centrum 512 m windows "
                  f"(grid 256 of 2 m, P=16, batch {min(batch_size, 64)}), noise p1, beta 0, cosine from {cont_lr:g}, fresh optimizer")

    b = _c5_wave_512(copy.deepcopy(L4), patch_size=patch_size, batch_size=batch_size)
    b["overrides"]["enc_n_layer"] = enc_n_layer
    b["id"] = (f"c5_r5_wave_nat3_tok01_banneal5_noisep1_L{enc_n_layer}{ptag}_{long_steps // 1000}k_g256{btag}"
               f"_{_C5_WAVE_512_STORE_TAG}")
    b["notes"] = (L4["notes"].replace("L4", f"L{enc_n_layer}") + f"; 512 m windows on Zuid + Centrum (grid 256 of 2 m, "
                  f"P={patch_size}, batch {batch_size}), from scratch")
    return [a, b]


def c5_wave_512_cost_variants() -> list[dict]:
    """The three shapes timed for the 512 m move (smokes only, 4 Oct): L4 P=16 (1024 tokens), L4 P=32 (256 tokens,
    16 x 16 cells per token), L6 P=16. Each is c5_wave_512_queue()'s spec b with that shape."""
    return [c5_wave_512_queue(patch_size=16, enc_n_layer=4)[1], c5_wave_512_queue(patch_size=32, enc_n_layer=4)[1],
            c5_wave_512_queue(patch_size=16, enc_n_layer=6, batch_size=48)[1]]


def c5_wave_512_natural_noise_round(source_run: str, round_no: int, cont_steps: int = 16000, p: float = 0.5) -> dict:
    """512 m, natural noise (4 Oct 2026): continue `source_run` (a network trained at 512 m) on the store of its own
    drawings over the 20k Zuid + Centrum 512 m train routes (c5_own_pads_train_r{round_no}.npz in c4_512_20k, built with
    spatial_data/c5_own_pads.py SOURCE --limit 20000 --batch 32 --out ...). Same settings as c5_wave_512_queue()[0].
    At 256 m this step was worth +14 points free-running, most of it on long routes."""
    spec = c5_wave_512_queue(cont_steps=cont_steps)[0]
    spec["id"] = f"c5_r5_wave_nat3_tok01_ownp{int(round(p * 100)):02d}_r{round_no}_L4_{cont_steps // 1000}k_g256_b64_zc512"
    spec["init_checkpoint"] = str(config.REPO_ROOT / "runs" / "checkpoints" / f"{source_run}.pt")
    spec["overrides"]["c5_wave_own_pads"] = {"path": f"c5_own_pads_train_r{round_no}.npz", "p": p}
    spec["notes"] = (f"C5 512 m natural noise round {round_no}: {source_run} continued {cont_steps} steps on its own drawings "
                     f"(20k routes), p {p}, noise p1, beta 0, cosine from 1e-4, batch 64")
    return spec


_C5_WAVE_BEST_512 = "c5_r5_wave_nat3_tok01_ownp50_r1_L4_16k_g256_b64_zc512"   # natural noise round 1 at 512 m (5 Oct)


def c5_wave_cleanup_queue(cont_steps: int = 16000, cont_lr: float = 1e-4, p_own: float = 0.5, p_cleanup: float = 1.0,
                          draw_fg_weight: float = 1.0, round_512: int = 1) -> list[dict]:
    """C5 cleanup pass (5 Oct 2026): train the final pass to turn the network's own messy drawing into one sharp line
    along the shortest route it found (spatial_data/c5_cleanup.py). Both specs continue a natural-noise network as
    its natural-noise round did (fresh optimizer, cosine from cont_lr, beta 0, token weight 0.1, pad noise p1), with:
      - draw share raised to one third: wave_fwd / wave_bwd / wave_draw 1 : 1 : 1 (was 4 : 4 : 1)
      - wave_fwd / wave_bwd: own drawings with probability p_own (c5_wave_own_pads, the teacher's target)
      - wave_draw: with probability p_cleanup the network's own drawing in and the cleanup target out
        (c5_wave_cleanup: the 3-wide shortest chain through that drawing when it is a success by the rollout eval's
        rule, else the teacher's path); the hand-made pad noise hits that input too
      - wave_draw lit-cell weight draw_fg_weight (c5_wave_fg_weight_override; 1 = calibrated at 0.5, against the
        over-painting the weight of ~7.5 / 11 rewards; config.c4_fg_weight)
      a  256 m: _C5_WAVE_BEST_256 continued on c4_100k with its own store c5_own_pads_train_r2.npz and
         c5_cleanup_train_r2.npz built from it
      b  512 m: _C5_WAVE_BEST_512 continued on c4_512_20k (batch 64) with c5_own_pads_train_r1.npz and
         c5_cleanup_train_r1.npz (the r1 store holds the drawings of the network round 1 started from, not of
         _C5_WAVE_BEST_512 itself: the only 512 m store there is)
    Read with harness/diagnostics/wave_rollout_eval.py RUN_ID --mode free (cleanliness numbers: final_excess_ratio,
    final_clean, solved_and_clean; success must not drop)."""
    ptag = f"ownp{int(round(p_own * 100)):02d}"
    ctag = f"cleanup{int(round(p_cleanup * 100)):03d}_dw{draw_fg_weight:g}".replace(".", "p")

    def cleanup(spec: dict, round_no: int, what: str) -> dict:
        spec["overrides"]["c4_train_task_weights"] = [1 / 3, 1 / 3, 1 / 3]
        spec["overrides"]["c5_wave_cleanup"] = {"path": f"c5_cleanup_train_r{round_no}.npz",
                                                "own_pads": f"c5_own_pads_train_r{round_no}.npz", "p": p_cleanup}
        spec["overrides"]["c5_wave_fg_weight_override"] = {"wave_draw": draw_fg_weight}
        spec["notes"] = (f"C5 cleanup pass {what}: fwd/bwd/draw 1:1:1, own drawings p {p_own} on fwd/bwd (r{round_no}), "
                         f"cleanup p {p_cleanup} on draw (c5_cleanup_train_r{round_no}.npz), draw lit-cell weight {draw_fg_weight:g}, "
                         f"noise p1, beta 0, token weight 0.1, cosine from {cont_lr:g}, fresh optimizer, {cont_steps} steps")
        return spec

    a = c5_wave_natural_noise_round(_C5_WAVE_BEST_256, 2, cont_steps=cont_steps, p=p_own, cont_lr=cont_lr)
    a = cleanup(a, 2, f"256 m, {_C5_WAVE_BEST_256} continued")
    a["id"] = f"c5_r5_wave_nat3_tok01_{ctag}_{ptag}_r2_L4_{cont_steps // 1000}k_g128_{_C5_WAVE_STORE_TAG}"
    b = c5_wave_512_natural_noise_round(_C5_WAVE_BEST_512, round_512, cont_steps=cont_steps, p=p_own)
    b["overrides"]["learning_rate"] = cont_lr
    b = cleanup(b, round_512, f"512 m, {_C5_WAVE_BEST_512} continued")   # round_512 = 2: its own drawings (6 Oct)
    b["id"] = f"c5_r5_wave_nat3_tok01_{ctag}_{ptag}_r{round_512}_L4_{cont_steps // 1000}k_g256_b64_{_C5_WAVE_512_STORE_TAG}"
    return [a, b]


def c5_wave_back_queue() -> list[dict]:
    """2 Oct 2026: the three backward step types alone (L2 8k, 128x128), then all six mixed at L4 16k (~2.5 h)."""
    q = {sp["overrides"]["c4_task_keys"][0]: sp for sp in c5_wave_queue()}
    return [q["wave_back"], q["wave_origin"], q["wave_draw"], c5_wave_mix()]


def c5_wave_night_1oct_queue() -> list[dict]:
    """Night 1 -> 2 Oct 2026 (~5 h): each step type alone at L2 8k 128x128; wave_first also at 64x64 (grid question);
    wave_first and wave_path at L4 16k; then all four mixed at L4 16k."""
    # ran 1-2 Oct (ids c5_r5_wave_{first,step,reach,path}_..., c5_r5_wave_mix4_...); wave_path was replaced by the
    # backward passes on 2 Oct, so this queue is history and returns nothing
    return []


def c5_r0_smoke() -> dict:
    """R0 gate: load N9 16ch weights into 17ch model, 500 steps; end spatial eval on detour val."""
    ckpt = config.REPO_ROOT / "runs" / "checkpoints" / "c4n_bld_det_L4_16k_store100k.pt"
    spec = c5_r0_regression_17ch()
    spec["id"] = "c5_r0_smoke_500_17ch"
    spec["train_steps"] = 500
    spec["lr_horizon_steps"] = 500
    spec["force_restart"] = True
    spec["init_checkpoint"] = str(ckpt)
    spec["load_optimizer"] = False
    spec["reset_step_counter"] = True
    spec["notes"] = (
        "C5 R0 smoke: init N9 (16ch→17ch expand), 500 steps; pass if detour dilated IoU within 0.02 of ~0.813"
    )
    spec["spatial_eval"] = {"max_batches": 20, "train_snap_batches": 0}
    spec["overrides"] = {
        **spec["overrides"],
        "eval_interval": 250,
        "eval_iters": 10,
        "diagnostics_enabled": False,
    }
    return spec


def c5_r1_rays_segment_mix() -> dict:
    """R1: cast rays (marker -> visible building corners within 80 m), trained with the bare segment task,
    2 blocks, batch 128, 8k. Needs ``c5_fans_{split}.npz`` next to the route store (D2). Val spatial eval
    uses ``c4_eval_task_key=rays``; segment metrics still in diagnostics / per-task breakdown."""
    spec = _c4_night_spec(run_id="c5_r1_seg_rays_L2_8k_store100k", keys=["segment", "rays"], weights=[0.5, 0.5],
                          train_steps=8000, seed=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          notes="C5 R1: ½ segment + ½ rays, 17ch, 2 blocks, 8k; pass: rays dilated IoU >= 0.85, < 5 % ray cells on buildings")
    spec["overrides"]["c4_eval_task_key"] = "rays"
    return spec



def c5_r2_length_line() -> dict:
    """R2: length of a straight line on the scratchpad as a token (10 m bins), with the bare segment task.
    Needs the token head (C5_PLAN M2) and c5_token_targets=True; mask target of length samples = scratchpad copy."""
    spec = _c4_night_spec(run_id="c5_r2_seg_lenline_L2_8k_store100k", keys=["segment", "length_line"], weights=[0.5, 0.5],
                          train_steps=8000, seed=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          c5_token_targets=True, c5_token_head_enabled=True,
                          notes="C5 R2: ½ segment + ½ length_line, token head; pass: bin accuracy >= 0.8, within one bin >= 0.98, mean error <= 10 m")
    spec["overrides"]["c4_eval_task_key"] = "length_line"
    return spec


def c5_r3_length_path() -> dict:
    """R3: length of a polyline on the scratchpad (same token as R2), mixed with the segment and the straight-line length."""
    spec = _c4_night_spec(run_id="c5_r3_seg_lenline_lenpath_L2_8k_store100k", keys=["segment", "length_line", "length_path"],
                          weights=[1 / 3, 1 / 3, 1 / 3], train_steps=8000, seed=4, c4_routes_subdir="c4_100k",
                          encoding_mode="onehot_v4_sp", c5_token_targets=True, c5_token_head_enabled=True,
                          notes="C5 R3: ⅓ segment + ⅓ length_line + ⅓ length_path; pass as R2 on length_path")
    spec["overrides"]["c4_eval_task_key"] = "length_path"
    return spec


def c5_rung23_run_queue() -> list[dict]:
    return [c5_r2_length_line(), c5_r3_length_path()]


def c5_r4_tighten() -> dict:
    """R4: pull the loose path on the scratchpad tight around the buildings (target = stored tight path), with the bare
    segment task; 2 blocks, 8k. Pass: output length / taut length <= 1.05 on >= 90 % of samples, no building cells."""
    spec = _c4_night_spec(run_id="c5_r4_seg_tighten_L2_8k_store100k", keys=["segment", "tighten"], weights=[0.5, 0.5],
                          train_steps=8000, seed=4, c4_routes_subdir="c4_100k", encoding_mode="onehot_v4_sp",
                          notes="C5 R4: ½ segment + ½ tighten, 17 channels, 2 blocks, 8k")
    spec["overrides"]["c4_eval_task_key"] = "tighten"
    return spec

def c5_rung1_run_queue() -> list[dict]:
    return [c5_r1_rays_segment_mix(), c5_r0_regression_17ch()]


def c5_rung01_run_queue() -> list[dict]:
    """C5_PLAN §6: R0 regression (17ch, N9 warm-start) then R1 rays+segment."""
    return [
        _c5_warmstart_n9_if_fresh(c5_r0_regression_17ch()),
        c5_r1_rays_segment_mix(),
    ]


def c5_resume_after_r0_run_queue() -> list[dict]:
    """R0 complete: R1 → R2 → R3 → R4 (harness auto-resumes partial runs)."""
    return [c5_r1_rays_segment_mix(), *c5_rung23_run_queue(), c5_r4_tighten()]


def c5_harness_run_queue() -> list[dict]:
    """Ordered C5 work left on the harness (finished ids skipped automatically).

    Done 29 Sep 2026: R0, R0a, R1b (g64 + g128), R2b, R3b on De Pijp 100k.
    Next: R0b 512 m window on Zuid+Centrum (8k smoke, ~90 min @ 3060), then R4 tighten."""
    return [
        c5_r0b_window512(),
        c5_r4_tighten(),
    ]


_C5_ONESHOT_256_ID = "c5_os_bld_det_L4_64k_g128_store100k"


def c5_oneshot_matched_queue(steps_256: int = 64000, steps_512: int = 32000, cont_lr_512: float = 1e-4,
                             loss: str = "ce_dice") -> list[dict]:
    """One-shot path finding with the search's budget (5 Oct 2026), so the comparison is method against method and
    not a 16k one-shot run against a 64k search. Scored with diagnostics/oneshot_route_eval.py, which applies the
    search's success rule (route_score) to the one-shot mask; pictures with diagnostics/showcase_compare.py.
      a  256 m: the best one-shot recipe (C4 report §7.2, N9: ½ building + ½ detour, 4 blocks, CE + Dice, 100k store)
         in the C5 setup of R0a (17 channels with an empty scratchpad, patch 16, output 128 x 128 = 2 m cells, the
         search's cell size), De Pijp c4_100k with all 100k train routes, batch 128, cosine over steps_256 from scratch.
         64k = the search network's 48k + 16k. Why this loss: CE + Dice is what every one-shot detour run used, and N9
         was the best of the mix probe (0.813 dilated against 0.799 for the detour alone; the building helper is neutral
         to slightly positive without the segment). The search's own loss (plain CE with the lit-cell weight, beta on
         changed cells) was tuned for a scratchpad that changes a little per pass; no report compares it with CE + Dice
         on a one-shot path, so it is not queued; loss="ce" builds that variant if wanted.
      b  512 m: (a) continued at 512 m windows on Zuid + Centrum c4_512_20k (512 px, output 256 x 256 = 2 m cells,
         P=16, 1024 tokens, batch 64), steps_512 steps, fresh optimizer, cosine from cont_lr_512, as the 512 m search
         was continued from the 256 m search (c5_wave_512_queue()[0], 16k, then 16k on its own drawings = 32k at
         512 m). The weights transfer the same way: no learned weight depends on the number of patches (sin-cos
         positions and the relative-bias table are rebuilt for 32 x 32 patches, the bias is per clipped offset, the
         grid head maps one patch to its 8 x 8 cells at both sizes). Needs (a)'s final checkpoint. cont_lr 1e-4 copies
         the search's continuation; a one-shot model has more to relearn at 512 m (routes up to ~1.1 km walking against
         ~300 m), so a higher rate is a fair alternative (an open choice).
    Matched budget, not matched presentations: half of the one-shot samples are the building helper, while every
    search sample is a search pass. Cost (L4, measured): 256 m ~700 samples/s -> 64k x 128 ~3.3 h model time (R0a's
    16k took 55 min with evals, i.e. ~3.7 h for 64k); 512 m b64 ~116 samples/s -> 32k x 64 ~4.9 h."""
    import copy
    a = c5_r0a_grid128()
    a["id"] = _C5_ONESHOT_256_ID + ("" if loss == "ce_dice" else f"_{loss}")
    a["train_steps"] = steps_256
    a["lr_horizon_steps"] = steps_256
    a["overrides"]["c4_grid_loss"] = loss
    a["notes"] = (f"C5 one-shot matched to the search's budget: N9 recipe (½ building + ½ detour, L4, {loss}), 17ch, grid 128 "
                  f"(2 m), De Pijp c4_100k, batch 128, cosine {steps_256} steps from scratch")

    b = _c5_wave_512(copy.deepcopy(a), patch_size=16, batch_size=64)
    b["id"] = (f"c5_os_bld_det_from{steps_256 // 1000}k_L4_{steps_512 // 1000}k_g256_b64_zc512"
               + ("" if loss == "ce_dice" else f"_{loss}"))
    b["train_steps"] = steps_512
    b["lr_horizon_steps"] = steps_512
    b["init_checkpoint"] = str(config.REPO_ROOT / "runs" / "checkpoints" / f"{a['id']}.pt")
    b["load_optimizer"] = False
    b["reset_step_counter"] = True
    b["force_restart"] = True          # with init_checkpoint the harness never auto-resumes: a stopped run starts over
    b["overrides"]["learning_rate"] = cont_lr_512
    b["notes"] = (f"C5 one-shot matched, 512 m: {a['id']} continued {steps_512} steps on Zuid + Centrum 512 m windows "
                  f"(grid 256 of 2 m, P=16, batch 64), cosine from {cont_lr_512:g}, fresh optimizer")
    return [a, b]


def _no_compile(spec: dict) -> dict:
    """torch.compile off for one spec (the 512 m one-shot model hit an Inductor assertion at its first step, 6 Oct)."""
    spec["overrides"]["use_compile"] = False
    return spec


RUN: list[dict] = [
    # Queue of 6 Oct (about 12 h; smoke-tested before launch): cleanup pass and budget-matched one-shot, 256 m then 512 m.
    # c5_wave_cleanup_queue()[0],        -- done 6 Oct # 1  cleanup pass at 256 m: continue the best 256 m network 16k (~1 h)
    # c5_oneshot_matched_queue()[0],     -- done 6 Oct # 2  one-shot detour, 256 m, 64k from scratch (~3.5 h)
    # 6 Oct, after run 1: draw weight 1.0 thins the line but breaks it (success 74 % before the draw pass, 56 % after);
    # weight 3 next, 256 m first (1 h) so the answer is early, 512 m on its own round-2 drawings last.
    # c5_wave_cleanup_queue(draw_fg_weight=3.0)[0],                   # cleanup 256 m, draw weight 3 (~1 h)  -- done 6 Oct 12:50
    _no_compile(c5_oneshot_matched_queue()[1]),                     # one-shot continued at 512 m, 32k (~5.5 h); compile off: Inductor assertion at step 0 (6 Oct)
    c5_wave_cleanup_queue(draw_fg_weight=3.0, round_512=2)[1],      # cleanup 512 m, draw weight 3, r2 drawings (~2.6 h)
    # c5_wave_cleanup_queue()[1],  -- not run: draw weight 1.0 (see run 1)
    # c5_wave_512_natural_noise_round("c5_r5_wave_nat3_tok01_from_ownp50_L4_16k_g256_b64_zc512", 1),  # 5 Oct: natural noise at 512 m (needs c5_own_pads_train_r1.npz in c4_512_20k)  -- done 5 Oct: 512 m val 32 -> 42 % (6-8 fronts 33 -> 53 %), maze 512 40 -> 39 %
    # c5_wave_512_queue()[0],  # done 4 Oct: 512 m val 32 % solved (91 / 65 / 33 / 18 / 2 % by fronts 2-3 / 4-5 / 6-8 / 9-12 / 13+)
    # c5_wave_natural_noise_round("c5_r5_wave_nat3_tok01_ownp50_from48k_L4_16k_g128_store100k", 2),  # 4 Oct: round 2 (needs c5_own_pads_train_r2.npz)  -- done: round 2 adds little (80 / 71 / 60 %)
    # *c5_wave_512_queue(),  # 4 Oct, prepared, not run: 512 m windows (Zuid + Centrum c4_512_20k, grid 256, P=16, batch 64): (a) best 256 m net continued 16k, (b) from scratch 48k (~6.3 h)
    # *c5_wave_own_pads_queue(cont_lr=1e-4),  # done 4 Oct: (a) 78 % / (b) 75 % on De Pijp val 500 (baseline 64 %)
    # *c5_wave_night_queue_2oct(),  # done: night 2/3 Oct: bwd2 x3 (token weight 0.1) then the whole search L4 / L2 48k
    # *c5_wave_back_counter_queue3(),  # done @7999: token loss weight 0 / 0.1 brings the mix back (180 -> 88 / 98 false cells)
    # *c5_wave_back_counter_queue2(),  # done @7999: reference reproduces; flat lr is not it; the mix with origin is
    # *c5_wave_back_counter_queue(),  # done @7999: A Dice+noise / B CE no noise / C CE+noise; none reproduces the blob
    # *c5_wave_bwd2_L4_p16_noisep1_ablation_queue(16000),  # done @15999 (store100k, banneal5 + plain)
    # *c5_wave_fwd3_L4_p16_100k_ablation_queue(16000),  # fwd — cancelled; use bwd2 ablation
    # *c5_wave_fwd3_anneal_l2_p16_p64_noisep1_queue(16000),  # done store40k
    # *c5_wave_bwd2_anneal_noisep1_queue(16000),  # done store40k (L2/L2p32/L4p32)
    # *c5_wave_fwd3_anneal_p32_noise_queue(16000),  # done @15999
    # *c5_wave_fwd3_anneal_followup_queue(16000),  # p32, L2 anneal done; noisep1 L4 finish separately
    # c5_wave_fwd3_banneal5_noisep1(16000),  # in flight — do not re-queue until @16k
    # c5_wave_fwd3_beta2_L2_staple(16000),  # done — plain L2; anneal is the key
    # c5_wave_fwd3_plain(16000),  # done @15999
    # c5_wave_fwd3_beta_thirds_decay(16000, beta_high=5.0),  # done @15999 (ce fix)
    # c5_wave_mix_forward3_beta(train_steps=12000, beta=5.0),
    # *c5_wave_change_fwd_beta_lrflat_12k_queue((2.0, 5.0)),
    # c5_wave_compile_smoke(use_compile_dynamic=False),  # passed 1 Oct ~131s
    # c5_wave_compile_smoke(use_compile_dynamic=True),   # passed 1 Oct ~236s (slower)
    # *c5_wave_change_fwd_beta_explore_12k_queue((2.0, 5.0, 10.0)),
    # *c5_wave_change_fwd_beta_queue((2.0, 5.0)),
    # *c5_wave_change_fwd4k_queue(),
    # *c5_wave_change_ce_queue(),
    # c5_wave_change_ce_smoke(),
    # *c5_wave_half_duration_queue(8000),
    # c5_wave_mix_step_back(),  # (iii) defer until half-duration gate passes
    # *c5_wave_ablation_queue(),  # 4k probe done: fwd step new ~0.03, bwd partial — not enough steps
    # c5_wave_mix(),  # mix6 done @16k — see wave_eval_dashboard.png
    # *c5_wave_back_queue(),  # back+origin done @8k; draw stopped early @1k (strip looked good)
    # *c5_r1d_traj_one_run_queue(),
    # *c5_wave_night_1oct_queue(),  # done 1-2 Oct (forward singles + mix4 aborted)
    # *c5_overnight_30sep_queue(),  # done 30 Sep night
    # *c5_harness_run_queue(),  # R0b is in the overnight queue; R4 tighten already done
    # *c5_rerun_queue(),  # R1b–R3b done 29 Sep 2026
    # c5_r0a_grid128(),  # resolution test A — done
    # *c5_resume_after_r0_run_queue(),  # old R1–R4 task recipe (29 Sep morning)
    # *c5_rung01_run_queue(),  # R0 done 29 Sep 2026
    # c5_r0_smoke(),
    # *m1_report_79_run_queue(),
    # *c4_overnight_run_queue(),
    # *c4_l4_store100k_run_queue(),
    # *c4_mix_probe_run_queue(),
    # done 25 Sep: c4a/c4b 4k explore — see docs/reports/C4_SEGMENT_DETOUR_2026-09-25.md
    # *c4_c4a_explore_run_queue(),
    # *c4_c4b_run_queue(),
    # c4_smoke_2k_diagnostics(),
    # c4_phase_b_loader_smoke(),
    # done 23 Sep §7: m1_report_overnight_run_queue(),
    # done 23 Sep: c3_m1_food_drink_within_100m_L4_16k_disc3_relV2(),
    # c3_m1_food_drink_within_100m_L4_8k_disc3_relV2(),
    # Disc3 batch (L2 b16) — done Sep 2026; see C3_PLAN §5g-vi
    # c3_within_50m_8k_disc3(),
    # c3_within_75m_8k_disc3(),
    # c3_within_100m_8k_disc3(),
    # c3_m1_food_drink_within_100m_8k_disc3(),
    # c3_n5_m1_mix_80_20_8k_disc3(),
    # c3_within_50m_8k_disc3_relV2(),
    # c3_within_75m_8k_disc3_relV2(),
    # c3_within_100m_8k_disc3_relV2(),
    # c3_m1_food_drink_within_100m_8k_disc3_relV2(),
    # c3_n5_m1_mix_80_20_8k_disc3_relV2(),
    # c3_m1_food_drink_within_100m_L4_8k_disc3(),
    # c3_n5_m1_mix_80_20_L4_8k_relV2(),
    # c3_m1_food_drink_within_100m_L4_8k_relV2(),
    # c3_n5_m1_mix_80_20_L4_8k(),
    # c3_m1_food_drink_within_100m_L4_8k(),
    # c3_n5_m1_mix_8k(),
    # c3_n5_m1_mix_2k_smoke(),
    # c3_m1_food_drink_within_100m_8k(),
    # c3_rel_pos_e5b_v2_within_20m_var_k(),
    # c3_m1_food_drink_within_100m_smoke(),
    # c3_m1_food_drink_within_100m_8k(),
    # *c3_rel_pos_e5b_round2_runs(),
    # c3_rel_pos_e5b_v1_within_75m(),  # done ~0.74 @ 8k (§8.6)
    # c3_rel_pos_smoke(),
    # c3_rel_pos_e5b_v1_within_20m(),
    # c3_within_50m_4k_lr3x(),
    # *c3_within_50m_arch_sweep_4k(),
    # *c3_n4_within_8k_runs(),
    # *c3_n4_within_smoke_runs(),
    
    # c2_gate_ts2_singles_2k(),  # done: mean rule IoU 0.978 @ 2k
    # c2_l6a_quiet_sidewalk_2k(),
    # c2_l6b_ts2_2k(),
    # c2_l6c_c2_full_2k(),
    # c2_l6b_ts2_district_2k(),  # after L6b fixed_crops if train scale matters
    # *c1_l5b_v3b_sidewalk_sweep(),
    # c1_l5c_factorised_2k_v3b(),
    # c1_l5b_core_four_2k_v3b(),
    # *c1_l5b_clamp_sweep((10.0, 30.0)),  # v2 disc baseline
    # c1_l5c_factorised_4k(),
    # c1_l5c_factorised_2k(),
    # c1_l5b_core_four_2k(),
    # c1_l5a_building_2k(),
    # c0_p16_2k_diagnostics_perf(),
    # *build_c0_worldsnap_p_sweep(C0_WORLDSNAP_P_SWEEP),
    # *as_single(C0_WORLDSNAP_BASE_2K, run_id="c0_worldsnap_full_2k", notes="P=16 baseline"),
]

TO_RUN = RUN  # harness.py reads this name
