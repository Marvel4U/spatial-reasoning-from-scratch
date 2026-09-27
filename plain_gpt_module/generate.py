import torch
from torch.nn import functional as F

from .build import unwrap_compiled


def generate_tokens(model, idx, max_new_tokens):
    """Append max_new_tokens via full-vocab multinomial sampling. Returns extended idx (B, T+K)."""
    model.eval()
    core = unwrap_compiled(model)
    block_size = core.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def model_generate(model, idx, max_new_tokens):
    """Translator qual API: model.generate(...) if present, else generate_tokens."""
    gen = getattr(model, "generate", None)
    if callable(gen):
        return gen(idx, max_new_tokens=max_new_tokens)
    return generate_tokens(model, idx, max_new_tokens)


def generate(model, x, enc, max_length=30, num_return_sequences=None, top_k=50, seed=42, verbose=True):
    """Sample continuations from a prompt tensor x (B, T). Returns x and decoded strings."""
    if num_return_sequences is None:
        num_return_sequences = x.size(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model.eval()
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    decoded = []
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        text = enc.decode(tokens)
        decoded.append(text)
        if verbose:
            print(">", text)
    return x, decoded
