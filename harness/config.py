"""Spatial local-model harness config (Track B grid ViT)."""
from pathlib import Path

# Repo root (parent of harness/), not cwd — data and runs stay out of harness/.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Bump when data pipeline, model semantics, or eval changes.
code_version = "1.5.0"
verbose = True  # stage logs in train.py / data.load_data (set False or train.py --quiet)


def log(msg: str, *, force: bool = False) -> None:
    if force or verbose:
        print(msg, flush=True)

batch_size = 32  # P=16 c0: 64–128 often fits 3060 with gpu loader; see PERF_BASELINE Step 6
t_train = 500
eval_interval = 500  # c3 full-grid + oracle + tier2 probe: keep CE evals sparse
eval_iters = 20
eval_iters_final = 50
eval_skip_val_in_ce_eval = False  # harness sets True when spatial_eval.train_snap_batches > 0
learning_rate = 1e-3
n_embd = 384
h_heads = 6
enc_n_layer = 2
dropout = 0.0
use_compile = False  # ~10–15% GPU step after Steps 1–4; 30s+ warm-up — enable for long GEX runs only
seed = 1337
use_gpu = True
# Intra-op CPU threads for PyTorch ops (batch building). None = PyTorch default.
# On 6-core rig with one core pinned (e.g. GNOME Resources), 4 avoids oversubscription.
cpu_num_threads: int | None = 4

in_chans = 4
img_size = 256
patch_size = 16
grid_out_size = 64
num_grid_classes = 2
encoding_mode = "scalar"  # scalar | onehot | onehot_v4 (c3 v4 noise7)
pos_temperature = 10000.0
rel_pos_bias = "none"  # none | first | all (C3_PLAN §8 E5b)
rel_pos_max_offset = 7  # K patches; (2K+1)² spatial buckets + one prefix bucket
use_sincos_pos = True  # False for V3 (relative bias without absolute sin-cos on patches)
task_rung = "c0"  # c0 | c1 | c2 | c3 | c4
# c2 (conjunctions across layers, C2_PLAN). Task sets and positive shares live in
# spatial_data/c2_tasks.py; f is measured from the train crops, never typed in here.
c2_task_set = "ts2"  # c2_core | ts2 | ts2_singles | l6a_quiet_sidewalk | c2_full
c2_weight_alpha = 0.5  # w = ((1-f)/f)**alpha (balancing memo §5); decision rule undoes it
c2_train_source = "district_random"  # district_random | fixed_crops
c2_held_out = False  # hold out C2_FULL_HELD_OUT (c2_full only)
c2_no_task_input = False  # L6a: no cond_tokens / task_id (single-task AND sanity)
c2_grid_loss = "ce"  # ce (dampened per-task weights) | ce_dice (L6d: plain CE + Dice, eval @ 0.5)
c2_ce_dice_weight = 1.0
c2_active_task_keys: list[str] | None = None  # set in data.load_data for c2
c3_task_key = "median_strict"  # median_strict | quietest_tertile | within_*m | food_drink_within_100m
c3_task_mode = "single"  # single | multi (N5 mix: food_drink + within_100m + m1)
c3_task_keys: list[str] | None = None  # multi mode: list of c3 task keys
c3_eval_task_key: str | None = None  # spatial eval primary task when multi (default: first key)
c3_train_task_weights: list[float] | None = None  # multi: per-key sample weights (same order as c3_task_keys)
c3_active_task_keys: list[str] | None = None  # set in data.load_data
c3_weight_alpha = 0.5
c3_grid_loss = "ce"  # ce (weighted) | ce_dice (N5 m1, §6b)
c3_ce_dice_weight = 1.0
marker_radius_px = 0  # marker plane: 0 = single pixel; c3 within_20m fix runs use 3
c3_n_markers = 0  # 0 = the fixed t0 marker per crop; K > 0 = K random markers per batch (union of discs)
c3_n_markers_random_min = 0  # with max > min: train draws K ~ Uniform[min, max] per sample (§8.6c)
c3_n_markers_random_max = 0
c3_marker_min_dist_m = 40.0  # pairwise minimum distance for random markers (2R keeps the discs disjoint)
c4_task_key = "segment"  # segment | detour
c4_task_mode = "single"  # single | multi
c4_task_keys: list[str] | None = None
c4_train_task_weights: list[float] | None = None
c4_active_task_keys: list[str] | None = None
c4_weight_alpha = 0.5
c4_grid_loss = "ce_dice"
c4_ce_dice_weight = 1.0
c4_mix_c3_task_keys: list[str] | None = None  # e.g. ["within_20m"] → train mix with c4 routes
c4_eval_task_key: str = "segment"  # val / spatial eval primary task (Phase D: segment on c4 routes)
c4_routes_subdir = "c4"  # folder under the cropset dir holding c4_routes_{split}.npz (e.g. "c4_100k")
c4_max_straight_m: float | None = None  # keep only routes whose straight distance <= this (train and val)
grad_clip = 1.0  # clip_grad_norm_ threshold; 1.0 was the fixed value before 25 Sep 2026
c1_task_key = "building"  # L5a single-task mode
c1_task_mode = "single"  # single | core_four (L5b) | factorised_twelve (L5c)
c1_held_out_task_keys: tuple[str, ...] = ("surface_0", "estab_2")
c1_val_pairs = 512  # fixed (crop, task) val samples when multi-task c1
c1_ce_weight_clamp = 100.0
c1_active_task_keys: list[str] | None = None  # set in data.load_data for c1 multi
balanced_ce = True  # c0 default; set False for c1 (use c1 per-task weights or plain CE)

data_root = REPO_ROOT / "data"
district_city = "amsterdam"
district_name = "de_pijp"
synthetic_train_samples = 4096
synthetic_val_samples = 512
val_same_as_train = False  # L1 overfit: eval on the same fixed pool as train
data_source = "synthetic"  # synthetic | worldsnap
cropset = "v2"  # v2 | v3/v3b/v4 footprints; v4 = v3b RGB + extra npz planes (noise7, sun) for c3/c4
taskset = "t0_point_v0"  # jsonl items supply crop_id + marker_256 for c0
worldsnap_max_train_items: int | None = None  # first N jsonl rows; None = full split
worldsnap_max_val_items: int | None = None
worldsnap_cache_layers = True  # keep unique crop npz layers in RAM (~1 GB for de_pijp v2)
worldsnap_gpu_resident = False  # CUDA: crop planes + batches on GPU (see dataset_c0_gpu.py)
worldsnap_materialize_items: int | None = None  # prebuild all (img,tgt); use if len(items) <= this

experiment_skip_enabled = True
harness_save_checkpoints = True
harness_checkpoint_dir = REPO_ROOT / "runs" / "checkpoints"
harness_auto_resume = True

training = False
log_experiments = True
run_id = None
CHECKPOINT_PATH = REPO_ROOT / "runs" / "checkpoint_spatial.pt"

RESULTS_PATH = REPO_ROOT / "runs" / "experiments.json"

# Analysis suite (docs/plans/ANALYSIS_SUITE_SPEC.md §5–§6); see analysis/README.md.
diagnostics_enabled = False
diagnostics_interval = 20
diagnostics_record_step0 = True
diagnostics_dir = REPO_ROOT / "runs" / "diagnostics"
# None = append eval.jsonl at every CE eval; else only when step % N == 0 (still at CE eval only).
diagnostics_eval_interval: int | None = None
diagnostics_probe_enabled = True
diagnostics_probe_batch_size = 64
diagnostics_pred_samples = 8
diagnostics_probe_seed = 1337
diagnostics_pred_enabled = True
# When c4_routes_subdir differs (e.g. c4_100k), save one extra pred PNG on val from this store at train end.
diagnostics_pred_c4_reference_subdir = "c4"
diagnostics_checkpoint_dir = REPO_ROOT / "runs" / "checkpoints"
diagnostics_checkpoint_steps = (
    0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
)


def c3_crops_dir():
    return district_data_dir() / "crops" / cropset


def c4_routes_dir():
    return c3_crops_dir() / c4_routes_subdir


def c4_is_multi_task() -> bool:
    return c4_task_mode == "multi" and bool(c4_task_keys)


def c4_keys_active() -> list[str]:
    if c4_active_task_keys is not None:
        return c4_active_task_keys
    if c4_is_multi_task() and c4_task_keys:
        return list(c4_task_keys)
    return [c4_task_key]


def c4_c4_task_keys_train() -> list[str]:
    """c4 route tasks in train (not c3 mix helpers)."""
    if c4_is_multi_task() and c4_task_keys:
        return list(c4_task_keys)
    return [c4_task_key]


def c4_train_task_keys() -> list[str]:
    """All train task ids (c3 mix + c4) for n_tasks / logging."""
    c3k = list(c4_mix_c3_task_keys or [])
    return c3k + c4_c4_task_keys_train()


def ensure_c4_encoding() -> None:
    global encoding_mode
    if encoding_mode != "onehot_v4":
        encoding_mode = "onehot_v4"
    sync_in_chans_from_encoding()


def c4_task_stats() -> dict:
    from spatial_data.c4_tasks import load_task_stats
    return load_task_stats(c4_routes_dir())


def c4_model_extra_config() -> dict:
    from spatial_data.c3_tasks import n_cond_emb_for_c3
    from spatial_data.c4_tasks import fg_ce_weights_for_key, n_cond_emb_for_c4
    stats = c4_task_stats()
    use_dice = c4_grid_loss == "ce_dice"
    keys = c4_train_task_keys()
    c4_only = [k for k in keys if k in ("segment", "detour", "building")]
    if use_dice or len(keys) == 1:
        fg_w = None if use_dice else (
            fg_ce_weights_for_key(c4_only[0], stats, alpha=c4_weight_alpha) if c4_only else None
        )
    else:
        fg_w = tuple(
            fg_ce_weights_for_key(k, stats, alpha=c4_weight_alpha)[0]
            for k in c4_only
        ) if c4_only else None
    return {
        "n_tasks": len(keys),
        "n_cond_emb": max(n_cond_emb_for_c3(), n_cond_emb_for_c4()),
        "c1_task_conditioning": "cond_tokens",
        "c1_task_layer_ids": (),
        "c1_task_class_ids": (),
        "c1_per_task_fg_weights": fg_w,
        "balanced_ce": False,
        "grid_loss": c4_grid_loss,
        "ce_dice_weight": c4_ce_dice_weight,
    }


def c3_task_stats() -> dict:
    from spatial_data.c3_tasks import load_task_stats
    return load_task_stats(c3_crops_dir(), district_dir=district_data_dir())


def ensure_c3_encoding() -> None:
    global encoding_mode
    if encoding_mode != "onehot_v4":
        encoding_mode = "onehot_v4"
    sync_in_chans_from_encoding()


def c3_is_multi_task() -> bool:
    return c3_task_mode == "multi" and bool(c3_task_keys)


def c3_keys_active() -> list[str]:
    if c3_active_task_keys is not None:
        return c3_active_task_keys
    if c3_is_multi_task() and c3_task_keys:
        return list(c3_task_keys)
    return [c3_task_key]


def c3_eval_key() -> str:
    if c3_eval_task_key:
        return c3_eval_task_key
    keys = c3_keys_active()
    return keys[0]


def c3_model_extra_config() -> dict:
    from spatial_data.c3_tasks import fg_ce_weights_for_key, n_cond_emb_for_c3
    stats = c3_task_stats()
    use_dice = c3_grid_loss == "ce_dice"
    keys = c3_keys_active()
    if use_dice or len(keys) == 1:
        fg_w = None if use_dice else fg_ce_weights_for_key(keys[0], stats, alpha=c3_weight_alpha)
    else:
        fg_w = tuple(
            fg_ce_weights_for_key(k, stats, alpha=c3_weight_alpha)[0] for k in keys
        )
    return {
        "n_tasks": len(keys),
        "n_cond_emb": n_cond_emb_for_c3(),
        "c1_task_conditioning": "cond_tokens",
        "c1_task_layer_ids": (),
        "c1_task_class_ids": (),
        "c1_per_task_fg_weights": fg_w,
        "balanced_ce": False,
        "grid_loss": c3_grid_loss,
        "ce_dice_weight": c3_ce_dice_weight,
    }


def c2_crops_dir():
    return district_data_dir() / "crops" / cropset


def c2_task_stats() -> dict:
    """Cached per-task positive shares; measured once per cropset (c2_tasks.load_task_stats)."""
    from spatial_data.c2_tasks import load_task_stats
    return load_task_stats(c2_crops_dir())


def c2_task_keys_active() -> list[str]:
    from spatial_data.c2_tasks import task_keys_for_set
    return task_keys_for_set(c2_task_set, c2_task_stats())


def c2_held_out_keys_active() -> tuple[str, ...]:
    from spatial_data.c2_tasks import held_out_keys_for_set
    return held_out_keys_for_set(c2_task_set) if c2_held_out else ()


def c2_train_task_indices_active() -> list[int] | None:
    from spatial_data.c2_tasks import train_task_indices
    keys = c2_active_task_keys or c2_task_keys_active()
    return train_task_indices(keys, c2_held_out_keys_active())


def c2_fg_weights_active() -> tuple[float, ...]:
    from spatial_data.c2_tasks import fg_ce_weights_for_keys
    keys = c2_active_task_keys or c2_task_keys_active()
    return fg_ce_weights_for_keys(keys, c2_task_stats(), alpha=c2_weight_alpha)


def ensure_c2_encoding() -> None:
    """c2 GPU loaders emit one-hot planes (C2_PLAN §2); align in_chans before build_model."""
    global encoding_mode
    if encoding_mode != "onehot":
        encoding_mode = "onehot"
    sync_in_chans_from_encoding()


def c2_model_extra_config() -> dict:
    """Per-task CE weights + condition-token task prefix (C2_PLAN §4)."""
    if c2_no_task_input:
        return {
            "n_tasks": 0,
            "c1_task_conditioning": "flat",
            "c1_task_layer_ids": (),
            "c1_task_class_ids": (),
            "c1_per_task_fg_weights": None,
            "balanced_ce": True,
        }
    keys = c2_active_task_keys or c2_task_keys_active()
    use_dice = c2_grid_loss == "ce_dice"
    return {
        "n_tasks": len(keys),
        "c1_task_conditioning": "cond_tokens",
        "c1_task_layer_ids": (),
        "c1_task_class_ids": (),
        "c1_per_task_fg_weights": None if use_dice else c2_fg_weights_active(),
        "balanced_ce": False,
        "grid_loss": c2_grid_loss,
        "ce_dice_weight": c2_ce_dice_weight,
    }


def c1_is_multi_task() -> bool:
    return c1_task_mode in ("core_four", "factorised_twelve")


def c1_task_keys_active() -> list[str]:
    from spatial_data.c1_tasks import task_keys_for_mode
    if c1_is_multi_task():
        return task_keys_for_mode(c1_task_mode)
    return [c1_task_key]


def c1_train_task_indices_active() -> list[int] | None:
    from spatial_data.c1_tasks import train_task_indices
    keys = c1_task_keys_active()
    return train_task_indices(c1_task_mode, c1_held_out_task_keys, keys)


def c1_model_extra_config() -> dict:
    from spatial_data.c1_tasks import factorised_ids_for_keys, fg_ce_weights_for_keys
    keys = c1_task_keys_active()
    multi = c1_is_multi_task()
    n_tasks = len(keys) if multi else 0
    weights = None
    conditioning = "flat"
    layer_ids: tuple[int, ...] = ()
    class_ids: tuple[int, ...] = ()
    if multi:
        weights = fg_ce_weights_for_keys(keys, cropset=cropset, clamp=c1_ce_weight_clamp)
        balanced = False
        if c1_task_mode == "factorised_twelve":
            conditioning = "factorised"
            layer_ids, class_ids = factorised_ids_for_keys(keys, cropset=cropset)
    else:
        balanced = balanced_ce
    return {
        "n_tasks": n_tasks,
        "c1_task_conditioning": conditioning,
        "c1_task_layer_ids": layer_ids,
        "c1_task_class_ids": class_ids,
        "c1_per_task_fg_weights": weights,
        "balanced_ce": balanced,
    }


def local_grid_vit_config_dict() -> dict:
    if task_rung == "c2":
        ensure_c2_encoding()
    elif task_rung == "c3":
        ensure_c3_encoding()
    elif task_rung == "c4":
        ensure_c4_encoding()
    extra = (
        c2_model_extra_config()
        if task_rung == "c2"
        else c3_model_extra_config()
        if task_rung == "c3"
        else c4_model_extra_config()
        if task_rung == "c4"
        else c1_model_extra_config()
    )
    return {
        "in_chans": in_chans,
        "img_size": img_size,
        "patch_size": patch_size,
        "grid_out_size": grid_out_size,
        "num_classes": num_grid_classes,
        "n_embd": n_embd,
        "n_head": h_heads,
        "enc_n_layer": enc_n_layer,
        "dropout": dropout,
        "pos_temperature": pos_temperature,
        "rel_pos_bias": rel_pos_bias,
        "rel_pos_max_offset": rel_pos_max_offset,
        "use_sincos_pos": use_sincos_pos,
        **extra,
    }


def district_data_dir(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else data_root
    return root / district_city / district_name


def sync_in_chans_from_encoding() -> None:
    """Keep config.in_chans aligned with encoding_mode (scalar 4, onehot 13, onehot_v4 16)."""
    global in_chans
    from spatial_data.channels import in_chans_for_encoding
    in_chans = in_chans_for_encoding(encoding_mode)
