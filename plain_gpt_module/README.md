# plain_gpt_module

Optimized GPT-2 backbone extracted from the GPT2 124M reproduction (flash SDPA, compile, fused AdamW). Import from sibling projects under `my_work/`.

## Setup

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # my_work/
```

## Train from scratch

```python
import tiktoken
from plain_gpt_module import (
    GPTConfig, build_model, unwrap_compiled,
    TokenBatchLoader, grad_accum_steps, train, generate,
)

enc = tiktoken.get_encoding("gpt2")
config = GPTConfig(n_layer=4, n_head=4, n_embd=256, vocab_size=50304)
model = build_model(config, device="cuda", use_compile=True, matmul_precision="high")

loader = TokenBatchLoader.from_text("input.txt", B=8, T=1024, encode_fn=enc.encode)
accum = grad_accum_steps(524288, loader.B, loader.T)

optimizer = unwrap_compiled(model).configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type="cuda", verbose=True,
)

history = train(
    model, optimizer, loader, device="cuda",
    max_steps=50, grad_accum_steps=accum,
    max_lr=6e-4, warmup_steps=10,
)
```

## Generate

```python
import torch

x = torch.tensor([enc.encode("Hello")], device="cuda")
_, texts = generate(model, x, enc, max_length=50)
```

## Load HuggingFace GPT-2 (optional, separate module)

```python
from plain_gpt_module.use_pretrained_GPT import load_pretrained

model = load_pretrained("gpt2", verbose=True)
model.eval()
model.to("cuda")
```

## Data loader variants

```python
from plain_gpt_module import TokenBatchLoader, load_tokens_npy

# token tensor
loader = TokenBatchLoader(tokens, B=8, T=1024)

# text file + custom encode (char vocab, BPE, tiktoken, …)
loader = TokenBatchLoader.from_text("data.txt", B=8, T=1024, encode_fn=my_encode)

# numpy shard (FineWeb-style)
loader = TokenBatchLoader.from_npy("fineweb_train_000000.npy", B=8, T=1024)
```

## Exports

| Symbol | Module |
|--------|--------|
| `GPT`, `GPTConfig` | `neural_network` |
| `build_model`, `unwrap_compiled` | `build` |
| `TokenBatchLoader`, `grad_accum_steps`, `load_tokens_npy` | `data_loader` |
| `train`, `get_lr` | `train` |
| `generate` | `generate` |
| `load_pretrained` | `use_pretrained_GPT` (not in package root) |
