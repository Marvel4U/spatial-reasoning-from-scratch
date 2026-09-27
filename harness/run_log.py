"""Experiment logging to runs/experiments.json."""
import json
from datetime import datetime, timezone
from pathlib import Path

import config
import experiments
from eval_spatial import spatial_summary_for_run_log

CONFIG_KEYS = (
    "batch_size", "t_train", "eval_interval", "eval_iters", "eval_iters_final",
    "learning_rate", "n_embd", "h_heads", "enc_n_layer", "dropout", "seed", "use_gpu",
    "use_compile", "in_chans", "img_size", "patch_size", "grid_out_size",
    "num_grid_classes", "encoding_mode", "pos_temperature", "task_rung",
    "data_source", "cropset", "taskset", "encoding_mode",
    "synthetic_train_samples", "synthetic_val_samples", "val_same_as_train",
    "worldsnap_max_train_items", "worldsnap_max_val_items",
    "worldsnap_cache_layers", "worldsnap_materialize_items",
    "district_city", "district_name",
    "diagnostics_enabled", "diagnostics_interval", "diagnostics_record_step0",
    "diagnostics_eval_interval",
    "diagnostics_probe_enabled", "diagnostics_probe_batch_size", "diagnostics_pred_samples",
    "diagnostics_pred_enabled", "diagnostics_checkpoint_steps",
)
RESUME_SKIP_CONFIG_KEYS = frozenset({
    "t_train",
    "eval_interval",
    # diagnostics / logging — must not invalidate resume after JSON round-trip or toggles
    "diagnostics_enabled",
    "diagnostics_interval",
    "diagnostics_record_step0",
    "diagnostics_eval_interval",
    "diagnostics_probe_enabled",
    "diagnostics_probe_batch_size",
    "diagnostics_pred_samples",
    "diagnostics_pred_enabled",
    "diagnostics_checkpoint_steps",
})
# Harness spec overrides that do not change the training problem.
RESUME_SKIP_OVERRIDE_KEYS = frozenset({"diagnostics_enabled", "eval_interval"})
_OVERRIDE_PRIOR_MISSING = object()

DEFAULT_ARCHITECTURE = "local_grid_vit"
DEFAULT_ARCH_VERSION = 1


def config_snapshot() -> dict:
    snap = {k: getattr(config, k) for k in CONFIG_KEYS}
    snap["code_version"] = config.code_version
    snap["model"] = config.local_grid_vit_config_dict()
    snap["data_root"] = str(config.data_root)
    snap["results_path"] = str(config.RESULTS_PATH)
    snap["checkpoint_dir"] = str(config.harness_checkpoint_dir)
    return snap


def config_snapshot_for_resume() -> dict:
    return {k: v for k, v in config_snapshot().items() if k not in RESUME_SKIP_CONFIG_KEYS}


def overrides_for_resume(overrides: dict | None) -> dict:
    if not overrides:
        return {}
    return {k: v for k, v in overrides.items() if k not in RESUME_SKIP_OVERRIDE_KEYS}


def overrides_equal_for_resume(a: dict | None, b: dict | None) -> bool:
    """Current spec (b) is canonical; prior (a) may omit keys added after an older run."""
    fb = overrides_for_resume(b or {})
    fa = overrides_for_resume(a or {})
    for k, vb in fb.items():
        if k in RESUME_SKIP_OVERRIDE_KEYS:
            continue
        va = fa.get(k, _OVERRIDE_PRIOR_MISSING)
        if va is _OVERRIDE_PRIOR_MISSING:
            continue
        if va != vb:
            return False
    return True


def find_run_by_id(path: Path, run_id: str) -> dict | None:
    matches = [r for r in load_results(path).get("runs", []) if r["id"] == run_id]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    def _rank(r: dict) -> tuple[bool, int]:
        summary = r.get("summary") or {}
        steps = summary.get("steps")
        return (not r.get("aborted", True), int(steps) if steps is not None else -1)

    return max(matches, key=_rank)


def training_setup_fingerprint(
    *,
    architecture: str,
    arch_version: int,
    overrides: dict,
    train_samples: int,
    val_samples: int,
    init_checkpoint: str | None,
) -> dict:
    return {
        "architecture": architecture,
        "arch_version": arch_version,
        "overrides": dict(overrides),
        "config": config_snapshot_for_resume(),
        "data": {"train_samples": train_samples, "val_samples": val_samples},
        "init_checkpoint": init_checkpoint,
    }


def _resume_value_equal(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_resume_value_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_resume_value_equal(a[k], b[k]) for k in a)
    return a == b


def _resume_config_equal(prior_cfg: dict, fingerprint_cfg: dict) -> bool:
    skip = RESUME_SKIP_CONFIG_KEYS
    pk = {k: v for k, v in prior_cfg.items() if k not in skip}
    fk = {k: v for k, v in fingerprint_cfg.items() if k not in skip}
    if set(pk) != set(fk):
        return False
    return all(_resume_value_equal(pk[k], fk[k]) for k in pk)


def training_goal_met(steps: int | None, train_steps: int) -> bool:
    return steps is not None and steps + 1 >= train_steps


def should_skip_finished_run(
    *,
    force_restart: bool,
    train_steps: int,
    overrides: dict,
    prior: dict | None,
    ckpt_steps: int | None,
    ckpt_path: Path | None = None,
) -> bool:
    if force_restart:
        return False
    steps = ckpt_steps
    if ckpt_path is not None:
        from grid_vit_adapter import max_checkpoint_steps

        steps = max_checkpoint_steps(ckpt_path)
    if steps is None or not training_goal_met(steps, train_steps):
        return False
    if prior is not None and not overrides_equal_for_resume(prior.get("overrides"), overrides):
        return False
    return True


def training_setup_matches(prior: dict, fingerprint: dict) -> bool:
    if prior.get("architecture") != fingerprint["architecture"]:
        return False
    if prior.get("arch_version") != fingerprint["arch_version"]:
        return False
    if not overrides_equal_for_resume(prior.get("overrides"), fingerprint["overrides"]):
        return False
    if prior.get("init_checkpoint") != fingerprint["init_checkpoint"]:
        return False
    prior_data = prior.get("data") or {}
    if prior_data.get("train_samples") != fingerprint["data"]["train_samples"]:
        return False
    if prior_data.get("val_samples") != fingerprint["data"]["val_samples"]:
        return False
    prior_cfg = prior.get("config") or {}
    return _resume_config_equal(prior_cfg, fingerprint["config"])


def record_for_resume(prior: dict, *, train_steps: int, started: str) -> dict:
    record = {
        "id": prior["id"],
        "source": prior.get("source", "harness"),
        "code_version": prior.get("code_version", config.code_version),
        "architecture": prior["architecture"],
        "arch_version": prior["arch_version"],
        "aborted": True,
        "started_at": prior.get("started_at", started),
        "resumed_at": started,
        "finished_at": None,
        "updated_at": started,
        "notes": prior.get("notes", ""),
        "overrides": dict(prior.get("overrides") or {}),
        "config": dict(prior.get("config") or config_snapshot()),
        "data": dict(prior.get("data") or {}),
        "metrics": list(prior.get("metrics") or []),
        "summary": dict(prior.get("summary") or {}),
    }
    if prior.get("spatial_evals"):
        record["spatial_evals"] = list(prior["spatial_evals"])
    if prior.get("init_checkpoint"):
        record["init_checkpoint"] = prior["init_checkpoint"]
    if prior.get("checkpoint"):
        record["checkpoint"] = prior["checkpoint"]
    record["summary"]["target_steps"] = train_steps
    return record


def load_results(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {"schema_version": experiments.SCHEMA_VERSION, "runs": []}


def save_results(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def upsert_run(doc: dict, record: dict):
    for i, r in enumerate(doc["runs"]):
        if r["id"] == record["id"]:
            doc["runs"][i] = record
            return
    doc["runs"].append(record)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def partial_summary(metrics: list, training_time_s: float, n_params: int, target_steps: int) -> dict:
    if not metrics:
        return {
            "steps": None,
            "train_loss": None,
            "val_loss": None,
            "training_time_s": round(training_time_s, 2),
            "n_params": n_params,
            "target_steps": target_steps,
        }
    last = metrics[-1]
    return {
        "steps": last["step"],
        "train_loss": last["train_loss"],
        "val_loss": last["val_loss"],
        "training_time_s": round(training_time_s, 2),
        "n_params": n_params,
        "target_steps": target_steps,
    }


class RunRecorder:
    def __init__(self, path: Path, record: dict):
        self.path = path
        self.record = record

    def flush(self, label: str = ""):
        doc = load_results(self.path)
        doc["schema_version"] = experiments.SCHEMA_VERSION
        self.record["updated_at"] = now_iso()
        upsert_run(doc, self.record)
        save_results(self.path, doc)
        if label:
            print(f"  saved → {self.path} ({label})")

    def on_eval(self, step: int, losses: dict, secs_since_last_eval: float, training_time_s: float):
        self.record["metrics"].append({
            "step": step,
            "train_loss": round(losses["train"], 6),
            "val_loss": round(losses["val"], 6),
            "secs_since_last_eval": round(secs_since_last_eval, 3),
        })
        self.record["summary"] = partial_summary(
            self.record["metrics"], training_time_s,
            self.record["summary"]["n_params"], self.record["summary"]["target_steps"],
        )
        self.flush()

    def on_spatial_eval(self, step: int, result: dict):
        entry = {"step": step, "at": now_iso(), **result}
        self.record.setdefault("spatial_evals", []).append(entry)
        val = result.get("val") or {}
        self.record["summary"].update(spatial_summary_for_run_log(val))
        self.record["summary"]["spatial_eval_step"] = step
        self.flush(f"spatial eval @ step {step}")

    def set_sample(self, sample: dict):
        self.record["sample"] = sample

    def complete(self, stats: dict):
        self.record["aborted"] = False
        self.record["finished_at"] = now_iso()
        self.record["summary"] = {
            **self.record["summary"],
            "steps": stats["steps"],
            "train_loss": round(stats["train_loss"], 6),
            "val_loss": round(stats["val_loss"], 6),
            "training_time_s": round(stats["training_time_s"], 2),
            "target_steps": self.record["summary"]["target_steps"],
            "n_params": self.record["summary"]["n_params"],
        }
        self.flush("complete")

    def abort(self, reason: str = "interrupted"):
        self.record["aborted"] = True
        self.record["abort_reason"] = reason
        self.record["finished_at"] = now_iso()
        self.flush("aborted")

    def skip(self, stats: dict):
        self.record["aborted"] = True
        self.record["skipped"] = True
        self.record["abort_reason"] = "skipped by user"
        self.record["finished_at"] = now_iso()
        self.record["summary"].update({
            "steps": stats["steps"],
            "train_loss": round(stats["train_loss"], 6) if stats["train_loss"] == stats["train_loss"] else None,
            "val_loss": round(stats["val_loss"], 6) if stats["val_loss"] == stats["val_loss"] else None,
            "training_time_s": round(stats["training_time_s"], 2),
        })
        self.flush("skipped")


def initial_record(
    run_id: str,
    *,
    source: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    arch_version: int = DEFAULT_ARCH_VERSION,
    train_steps: int,
    overrides: dict | None = None,
    notes: str = "",
    started: str,
    n_params: int | None = None,
    train_samples: int,
    val_samples: int,
    init_checkpoint: str | None = None,
) -> dict:
    return {
        "id": run_id,
        "source": source,
        "code_version": config.code_version,
        "architecture": architecture,
        "arch_version": arch_version,
        "aborted": True,
        "started_at": started,
        "finished_at": None,
        "updated_at": started,
        "notes": notes,
        "overrides": dict(overrides or {}),
        "config": config_snapshot(),
        "data": {"train_samples": train_samples, "val_samples": val_samples},
        "metrics": [],
        "summary": {
            "steps": None,
            "train_loss": None,
            "val_loss": None,
            "training_time_s": 0.0,
            "n_params": n_params,
            "target_steps": train_steps,
        },
        **({"init_checkpoint": init_checkpoint} if init_checkpoint else {}),
    }


def manual_run_id() -> str:
    if config.run_id:
        return config.run_id
    return f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
