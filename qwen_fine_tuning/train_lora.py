#!/usr/bin/env python3
"""LoRA SFT on track_a route_v1 examples (default: 100 train / 20 dev)."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from qwen_fine_tuning.examples import DEFAULT_EXPORT
from qwen_fine_tuning.sft_batch import RouteSFTDataset, collate_sft_batch, train_dev_split
from qwen_reference.infer import infer_route
from qwen_reference.load_model import resolve_model
from spatial_data.track_a_export import load_predictions, score, summary_line

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/qwen_sft/lora100_v1"
STATE_JSON = "training_state.json"
STATE_PT = "training_state.pt"


def _save_checkpoint(
    out_dir: Path,
    opt_step: int,
    micro_step: int,
    epoch: int,
    total_opt_steps: int,
    model,
    proc,
    opt,
    sched,
) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"step_{opt_step:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    proc.save_pretrained(ckpt_dir)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, out_dir / STATE_PT)
    state = {
        "opt_step": opt_step,
        "micro_step": micro_step,
        "epoch": epoch,
        "total_opt_steps": total_opt_steps,
        "checkpoint_dir": str(ckpt_dir.relative_to(out_dir)),
    }
    (out_dir / STATE_JSON).write_text(json.dumps(state, indent=2))


def _try_resume(
    out_dir: Path,
    model_path: Path,
    lora_cfg: LoraConfig,
    device_map: str,
    *,
    grad_checkpoint: bool,
) -> tuple | None:
    state_path = out_dir / STATE_JSON
    pt_path = out_dir / STATE_PT
    if not state_path.is_file() or not pt_path.is_file():
        return None
    state = json.loads(state_path.read_text())
    ckpt_dir = out_dir / state["checkpoint_dir"]
    if not ckpt_dir.is_dir():
        return None
    proc = AutoProcessor.from_pretrained(ckpt_dir)
    base = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device_map
    )
    if grad_checkpoint:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=True)
    print(f"resume from {ckpt_dir}  opt_step={state['opt_step']}/{state['total_opt_steps']}")
    return (
        proc,
        model,
        state["opt_step"],
        state["micro_step"],
        state["epoch"],
        state["total_opt_steps"],
        ckpt_dir,
    )


@torch.inference_mode()
def eval_dev_solved(
    proc,
    model,
    export: Path,
    dev_records: list[dict],
    *,
    max_new_tokens: int = 384,
) -> dict:
    preds: dict[int, list] = {}
    from PIL import Image

    n = len(dev_records)
    for i, rec in enumerate(dev_records):
        img = Image.open(export / rec["image"]).convert("RGB")
        _, points, _ = infer_route(
            proc, model, img, mode="v1", max_new_tokens=max_new_tokens, verbose=False, record=rec
        )
        preds[rec["route"]] = points
        print(f"  dev {i + 1}/{n} route {rec['route']}", flush=True)
    pred_path = export / "_dev_pred_tmp.jsonl"
    with pred_path.open("w") as f:
        for rec in dev_records:
            f.write(json.dumps({"route": rec["route"], "points": preds.get(rec["route"])}))
            f.write("\n")
    rep = score(dev_records, load_predictions(pred_path))
    pred_path.unlink(missing_ok=True)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA SFT: route_v1 + teacher polyline (0–1000 JSON)")
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--train-limit", type=int, default=100)
    p.add_argument("--dev-limit", type=int, default=20)
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch_size * grad_accum")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke); default: full epochs")
    p.add_argument("--eval-every", type=int, default=0, help="dev generate+score every N optimizer steps; 0=end only")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save adapter+optimizer every N optimizer steps (0=final only)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from out_dir/training_state.json if present",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    export = Path(args.export)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs, dev_recs = train_dev_split(export, args.train_limit, args.dev_limit)
    model_path = resolve_model(args.model)
    proc = AutoProcessor.from_pretrained(model_path)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    grad_ckpt = not args.no_grad_checkpoint
    resumed = None
    if args.resume:
        resumed = _try_resume(out_dir, model_path, lora_cfg, args.device_map, grad_checkpoint=grad_ckpt)
    if resumed:
        proc, model, opt_step, micro_step, start_epoch, total_opt_steps, _ = resumed
        model.print_trainable_parameters()
    else:
        opt_step = 0
        micro_step = 0
        start_epoch = 0
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=args.device_map
        )
        if grad_ckpt:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    ds = RouteSFTDataset(proc, export, train_recs)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, pad_id),
    )
    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    if not resumed:
        total_opt_steps = math.ceil(steps_per_epoch * args.epochs / args.grad_accum)
        if args.max_steps is not None:
            total_opt_steps = min(total_opt_steps, args.max_steps)
    warmup_steps = max(1, int(total_opt_steps * args.warmup_ratio))

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_opt_steps)
    if resumed:
        blob = torch.load(out_dir / STATE_PT, map_location="cpu", weights_only=True)
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])

    run_cfg = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "export": str(export),
        "prompt": "route_v1",
        "train_routes": [r["route"] for r in train_recs],
        "dev_routes": [r["route"] for r in dev_recs],
        "hyperparams": vars(args),
        "total_optimizer_steps": total_opt_steps,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    log_path = out_dir / "train_log.jsonl"

    if not resumed:
        micro_step = 0
        opt_step = 0
    accum_loss = 0.0
    accum_n = 0
    t0 = time.perf_counter()
    model.train()
    skip_micro = micro_step if resumed else 0

    def _run_eval(tag: int | str) -> None:
        model.eval()
        rep = eval_dev_solved(proc, model, export, dev_recs)
        line = summary_line(rep)
        print(f"dev eval ({tag}): {line}")
        ev = {"opt_step": tag, "dev": rep, "summary": line}
        with (out_dir / "dev_eval.jsonl").open("a") as f:
            f.write(json.dumps(ev) + "\n")
        model.train()

    stop = opt_step >= total_opt_steps
    for epoch in range(args.epochs):
        if stop:
            break
        for batch in loader:
            micro_step += 1
            if micro_step <= skip_micro:
                continue
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            batch.pop("routes", None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch, labels=labels)
            (out.loss / args.grad_accum).backward()
            accum_loss += out.loss.item()
            accum_n += 1
            if micro_step % args.grad_accum != 0:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            sched.step()
            avg = accum_loss / accum_n
            accum_loss = accum_n = 0
            lr = sched.get_last_lr()[0]
            row = {"opt_step": opt_step, "micro_step": micro_step, "loss": round(avg, 4), "lr": lr}
            print(f"step {opt_step}/{total_opt_steps}  loss={avg:.4f}  lr={lr:.2e}")
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if args.checkpoint_every and opt_step % args.checkpoint_every == 0:
                _save_checkpoint(
                    out_dir, opt_step, micro_step, epoch, total_opt_steps, model, proc, opt, sched
                )
                print(f"checkpoint step {opt_step} → {out_dir / 'checkpoints'}", flush=True)
            if args.eval_every and opt_step % args.eval_every == 0:
                _run_eval(opt_step)
            if opt_step >= total_opt_steps:
                stop = True
                break

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    proc.save_pretrained(adapter_dir)
    _run_eval("final")
    elapsed = time.perf_counter() - t0
    print(f"saved adapter to {adapter_dir}  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
