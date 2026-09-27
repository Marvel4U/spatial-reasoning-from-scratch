"""Experiment specs + RUN queue. Builders: experiment_sweeps.py."""

from __future__ import annotations

from typing import Any

import config
from experiment_sweeps import as_single, build_c0_worldsnap_p_sweep

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
    """N5 m1 portfolio run @ 8k."""
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
    """N5 portfolio: three-task mix + warm-start from m0 @ 100 m."""
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

RUN: list[dict] = [
    # *m1_report_79_run_queue(),
    # *c4_overnight_run_queue(),
    # *c4_l4_store100k_run_queue(),
    *c4_mix_probe_run_queue(),
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
