import math
import time

import torch

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(
    model,
    optimizer,
    train_loader,
    device,
    max_steps,
    grad_accum_steps,
    max_lr=6e-4,
    min_lr=None,
    warmup_steps=10,
    grad_clip=1.0,
):
    """Run the GPT training loop. train_loader must expose .B, .T, and .next_batch()."""
    if min_lr is None:
        min_lr = max_lr * 0.1
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    history = []
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        record = {
            "step": step,
            "loss": loss_accum.item(),
            "lr": lr,
            "norm": norm.item(),
            "dt": dt,
            "t_s": tokens_per_sec,
        }
        history.append(record)
        print(f"step {step}, loss: {record['loss']}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}s, t/s: {tokens_per_sec:.2f}")
    return history
