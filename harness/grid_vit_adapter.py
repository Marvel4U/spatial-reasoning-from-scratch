"""Bridge harness config/data to plain_gpt_module LocalGridViT."""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

import config
import data
from spatial_batch import forward as model_forward, next_batch as loader_next_batch
from checkpoints import checkpoint_extra, checkpoint_model_config

if str(config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT))

from plain_gpt_module import get_lr, load_checkpoint as _load_checkpoint, save_checkpoint as _save_checkpoint, unwrap_compiled
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT

ARCHITECTURE = "local_grid_vit"
ARCH_VERSION = 1
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 10
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.1


def throughput_between_evals(train_steps: int, seconds: float) -> dict[str, float]:
    """Rates since last eval (GPT-style: report positions/s, not only batches/s).

    - samples/s: batch items (depends on batch size).
    - grid_cells/s: supervised CE targets = samples × grid_out² (model-depth invariant).
    - mpix/s: input luminance-plane pixels = samples × img_size² (channel-count invariant).
    """
    if seconds <= 0 or train_steps <= 0:
        return {"samples_s": 0.0, "grid_cells_s": 0.0, "mpix_s": 0.0}
    samples = train_steps * config.batch_size
    g, h = config.grid_out_size, config.img_size
    inv = 1.0 / seconds
    return {
        "samples_s": samples * inv,
        "grid_cells_s": samples * g * g * inv,
        "mpix_s": samples * h * h * inv / 1e6,
    }


def _format_throughput(rates: dict[str, float]) -> str:
    cs = rates["grid_cells_s"]
    cells = f"{cs / 1e6:.2f}M cells/s" if cs >= 1e6 else f"{cs:,.0f} cells/s"
    return f"{rates['samples_s']:,.0f} samples/s, {cells}, {rates['mpix_s']:.1f} Mpix/s"


def model_config() -> LocalGridViTConfig:
    return LocalGridViTConfig(**config.local_grid_vit_config_dict())


def build_model(device=None, use_compile=None):
    device = device or data.device
    if use_compile is None:
        use_compile = config.use_compile
    cfg = model_config()
    config.log(
        f"grid_vit_adapter.build_model: enc={cfg.enc_n_layer} n_embd={cfg.n_embd} "
        f"compile={use_compile} device={device}"
    )
    t0 = time.perf_counter()
    model = LocalGridViT(cfg)
    config.log(f"  model constructed ({time.perf_counter() - t0:.2f}s)")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    config.log(f"  on {device}, params={n_params:,}")
    if use_compile and str(device).startswith("cuda"):
        config.log("  torch.compile … (first forward can take minutes)")
        t1 = time.perf_counter()
        model = torch.compile(model, dynamic=True)
        config.log(f"  compile wrapper ready ({time.perf_counter() - t1:.2f}s)")
    return model


def configure_optimizer(model, device=None):
    device = device or data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return unwrap_compiled(model).configure_optimizers(
        WEIGHT_DECAY, config.learning_rate, device_type,
    )


def _checkpoint_extra() -> dict:
    return {
        "code_version": config.code_version,
        "dtype": str(data.dtype),
        "architecture": ARCHITECTURE,
        "arch_version": ARCH_VERSION,
        "encoding_mode": config.encoding_mode,
        "task_rung": config.task_rung,
        "data_source": config.data_source,
    }


def save_checkpoint(path, model, optimizer=None, training_stats=None):
    _save_checkpoint(
        path, model,
        optimizer=optimizer,
        training_stats=training_stats,
        model_config=model_config(),
        extra=_checkpoint_extra(),
    )


def checkpoint_training_steps(path) -> int | None:
    if not Path(path).is_file():
        return None
    ckpt = read_checkpoint(path, map_location="cpu")
    tr = ckpt.get("training")
    if not tr:
        return None
    steps = tr.get("steps")
    return int(steps) if steps is not None else None


def checkpoint_saved_step(path) -> int | None:
    """Step index for main ckpt (training block) or diagnostics series (extra / filename)."""
    path = Path(path)
    if not path.is_file():
        return None
    s = checkpoint_training_steps(path)
    if s is not None:
        return s
    ckpt = read_checkpoint(path, map_location="cpu")
    extra = ckpt.get("extra") or {}
    if extra.get("step") is not None:
        return int(extra["step"])
    stem = path.stem
    if stem.startswith("step") and stem[4:].isdigit():
        return int(stem[4:])
    return None


def iter_run_checkpoint_paths(main_path: Path):
    """Main ``{run_id}.pt`` plus diagnostics ``{run_id}/step*.pt`` if present."""
    main_path = Path(main_path)
    if main_path.is_file():
        yield main_path
    series_dir = main_path.parent / main_path.stem
    if series_dir.is_dir():
        yield from sorted(series_dir.glob("step*.pt"))


def max_checkpoint_steps(main_path: Path) -> int | None:
    best: int | None = None
    for p in iter_run_checkpoint_paths(main_path):
        s = checkpoint_saved_step(p)
        if s is not None:
            best = s if best is None else max(best, s)
    return best


def best_checkpoint_path(main_path: Path) -> Path | None:
    best_p: Path | None = None
    best_s: int | None = None
    for p in iter_run_checkpoint_paths(main_path):
        s = checkpoint_saved_step(p)
        if s is None:
            continue
        if best_s is None or s > best_s:
            best_s, best_p = s, p
    return best_p


def load_checkpoint(path, model=None, optimizer=None):
    ckpt = read_checkpoint(path, map_location=data.device)
    if model is None:
        cfg_dict = checkpoint_model_config(ckpt)
        model = LocalGridViT(LocalGridViTConfig(**cfg_dict))
        if config.use_compile and str(data.device).startswith("cuda"):
            model = torch.compile(model, dynamic=True)
    _load_checkpoint(path, model, optimizer=optimizer, map_location=data.device)
    model.to(data.device)
    data.m = model
    data.training_stats = ckpt.get("training")
    if data.training_stats is None:
        step = checkpoint_saved_step(path)
        if step is not None:
            data.training_stats = {
                "steps": step,
                "training_time_s": 0.0,
                "train_loss": float("nan"),
                "val_loss": float("nan"),
            }
    return model, ckpt


@torch.no_grad()
def estimate_loss(model, eval_iters=None, *, include_val: bool | None = None):
    iters = config.eval_iters if eval_iters is None else eval_iters
    if include_val is None:
        include_val = not config.eval_skip_val_in_ce_eval
    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    splits = ("train", "val") if include_val else ("train",)
    loaders = {"train": data.train_loader, "val": data.val_loader}
    out: dict[str, float] = {}
    model.eval()
    for split in splits:
        loader = loaders[split]
        losses = torch.zeros(iters)
        for k in range(iters):
            img, tgt, task_id, cond_ids = loader_next_batch(loader)
            img, tgt = img.to(device), tgt.to(device)
            if task_id is not None:
                task_id = task_id.to(device)
            if cond_ids is not None:
                cond_ids = cond_ids.to(device)
            with torch.autocast(device_type=device_type, dtype=data.dtype):
                _, loss = model_forward(model, img, tgt, task_id, cond_ids)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if not include_val:
        out["val"] = float("nan")
    model.train()
    return out


def _sync_device(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train(
    num_steps,
    model=None,
    opt=None,
    checkpoint_path=config.CHECKPOINT_PATH,
    save=True,
    on_eval=None,
    should_stop=None,
    lr_horizon_steps: int | None = None,
    tier1=None,
    ckpt_series=None,
):
    if model is None:
        model = data.m
    if opt is None:
        opt = data.optimizer
    if model is None or opt is None:
        raise RuntimeError("pass model= and opt=, or set data.m / data.optimizer")
    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    loader = data.train_loader
    max_lr = config.learning_rate
    min_lr = max_lr * MIN_LR_RATIO
    if data.training_stats is None:
        start_step = 0
        accumulated_time = 0.0
    else:
        start_step = data.training_stats["steps"] + 1
        accumulated_time = data.training_stats["training_time_s"]
    planned_end = start_step + num_steps - 1
    lr_horizon = lr_horizon_steps if lr_horizon_steps is not None else planned_end + 1
    samples_per_step = config.batch_size
    print(f"Training steps {start_step}..{planned_end} ({num_steps} updates, {samples_per_step} samples/step)...")
    model.train()
    if tier1 is not None:
        tier1.prepare(model)
    if ckpt_series is not None and start_step == 0 and ckpt_series.should_save(0):
        ckpt_series.save(0, model)
    t0 = time.perf_counter()
    last_eval_end = t0
    last_eval_step = start_step
    acc_train_s = 0.0
    acc_other_s = 0.0
    final_losses = None
    skipped = False
    last_completed = start_step - 1
    for step in range(start_step, start_step + num_steps):
        if should_stop is not None and should_stop(step):
            print(f"  skip requested — stopping at step {step}")
            skipped = True
            break
        if step % config.eval_interval == 0 or step == planned_end:
            wall_s = time.perf_counter() - last_eval_end
            train_steps_since = step - last_eval_step
            iters = config.eval_iters_final if step == planned_end else config.eval_iters
            t_ce = time.perf_counter()
            final_losses = estimate_loss(model, eval_iters=iters)
            ce_s = time.perf_counter() - t_ce
            snap_s = log_s = 0.0
            if on_eval is not None:
                ret = on_eval(step, final_losses, wall_s, eval_iters=iters)
                if isinstance(ret, tuple) and len(ret) == 2:
                    patched, extra = ret
                    snap_s = float(extra.get("snap_s", 0.0))
                    log_s = float(extra.get("log_s", 0.0))
                else:
                    patched = ret
                if patched is not None:
                    final_losses = patched
            eval_s = ce_s + snap_s + log_s
            other_s = max(0.0, wall_s - acc_train_s - eval_s)
            rates = throughput_between_evals(train_steps_since, acc_train_s)
            val_s = (
                f"{final_losses['val']:.4f}"
                if final_losses["val"] == final_losses["val"]
                else "n/a"
            )
            thr = _format_throughput(rates) if rates["samples_s"] else "—"
            print(
                f"step {step}: train {final_losses['train']:.4f} val {val_s} | {thr} | "
                f"{wall_s:.1f}s (train {acc_train_s:.1f}s ce {ce_s:.1f}s "
                f"snap {snap_s:.1f}s log {log_s:.1f}s other {other_s + acc_other_s:.1f}s)"
            )
            last_eval_end = time.perf_counter()
            last_eval_step = step
            acc_train_s = 0.0
            acc_other_s = 0.0
        opt.zero_grad(set_to_none=True)
        rec = tier1 is not None and tier1.should_record(step)
        if rec:
            _sync_device(device)
        t_step = time.perf_counter()
        img, tgt, task_id, cond_ids = loader_next_batch(loader)
        t_after_batch = time.perf_counter()
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            _, loss = model_forward(model, img, tgt, task_id, cond_ids)
        loss.backward()
        pre_clip = tier1.collect_pre_clip(model) if rec else None
        grad_norm_total = torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(config, "grad_clip", GRAD_CLIP)))
        weight_before = tier1.snapshot_weights(model) if rec else None
        lr = get_lr(step, max_lr, min_lr, WARMUP_STEPS, lr_horizon)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        opt.step()
        if rec:
            _sync_device(device)
            tier1.record_step(
                step=step,
                loss=loss.item(),
                lr=lr,
                t_data=t_after_batch - t_step,
                t_step=time.perf_counter() - t_after_batch,
                batch_size=samples_per_step,
                grad_norm_total=float(grad_norm_total),
                pre_clip=pre_clip,
                update_ratios=tier1.update_ratios(model, weight_before),
            )
        ckpt_dt = 0.0
        if ckpt_series is not None and ckpt_series.should_save(step) and step != 0:
            t_ck = time.perf_counter()
            ckpt_series.save(step, model)
            ckpt_dt = time.perf_counter() - t_ck
            acc_other_s += ckpt_dt
        acc_train_s += time.perf_counter() - t_step - ckpt_dt
        last_completed = step
    segment_time = time.perf_counter() - t0
    end_step = last_completed if skipped else planned_end
    if final_losses is None:
        final_losses = {"train": float("nan"), "val": float("nan")}
    stats = {
        "steps": end_step,
        "train_loss": final_losses["train"],
        "val_loss": final_losses["val"],
        "training_time_s": accumulated_time + segment_time,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "dtype": str(data.dtype),
        "skipped": skipped,
    }
    data.training_stats = stats
    label = " (skipped early)" if skipped else ""
    print(f"Segment done in {segment_time:.2f}s (total {stats['training_time_s']:.2f}s){label}")
    if save:
        save_checkpoint(path=checkpoint_path, model=model, optimizer=opt, training_stats=stats)
    return stats


def config_dict_for_log() -> dict:
    return asdict(model_config())
