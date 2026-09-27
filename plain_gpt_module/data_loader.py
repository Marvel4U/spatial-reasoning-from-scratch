"""Token batch loaders for plain_gpt_module.train().

Both loaders expose the same interface expected by train():
  - attributes: .B (batch size), .T (context length)
  - method: next_batch() -> (x, y)  each shape (B, T), y is x shifted by one token

train() pulls batches in a loop; you choose the loader when building the pipeline:

  tokens on disk  ->  encode once (your code)  ->  loader  ->  train(model, optimizer, loader, ...)

Usage contrast
--------------
TokenBatchLoader — sequential stream (GPT-2 / FineWeb style)
  Walks the corpus in order: batch 0 = tokens[0:B*T], batch 1 = tokens[B*T:2*B*T], …
  Wraps to the start when the stream runs out. Good when the corpus is one long document
  or shuffled shard and you want near-full coverage over many steps.

  loader = TokenBatchLoader.from_text("input.txt", B=8, T=1024, encode_fn=enc.encode)
  # or: TokenBatchLoader(train_tokens, B=8, T=1024)
  # or: TokenBatchLoader.from_npy("shard.npy", B=8, T=1024)

RandomWindowLoader — random windows (translator / nanoGPT get_batch style)
  Each next_batch() picks B independent start indices in the token tensor and slices
  length-T windows. Same positions are often resampled; some tokens may never appear.
  Good for a fixed pre-tokenized train/val pool (e.g. Europarl after load_data()).

  train_tokens = ...  # torch.long 1-D, built by your project (BPE, char, etc.)
  loader = RandomWindowLoader(train_tokens, B=128, T=256)
  val_loader = RandomWindowLoader(val_tokens, B=128, T=256)  # separate instance for eval

Minimal training wiring (either loader):

  from plain_gpt_module import build_model, unwrap_compiled, train, grad_accum_steps

  model = build_model(config, device="cuda")
  optimizer = unwrap_compiled(model).configure_optimizers(0.1, 6e-4, "cuda")
  accum = grad_accum_steps(total_batch_size=524288, B=loader.B, T=loader.T)
  history = train(model, optimizer, loader, device="cuda", max_steps=1000, grad_accum_steps=accum)

Loss masks and split-specific sampling stay in the caller (e.g. translator utils.get_batch);
these loaders only return (x, y).
"""
from pathlib import Path

import numpy as np
import torch

def load_tokens_npy(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class TokenBatchLoader:
    """Sequential (x, y) batches: contiguous chunks of the token stream in order.

    Example:
        loader = TokenBatchLoader.from_text("input.txt", B=8, T=1024, encode_fn=enc.encode)
        x, y = loader.next_batch()  # x[b,t]=tokens[pos+t], y[b,t]=tokens[pos+t+1]
    """

    def __init__(self, tokens, B, T, verbose=True):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.to(dtype=torch.long)
        self.tokens = tokens
        self.B = B
        self.T = T
        self.current_position = 0
        if verbose:
            n_batches = len(self.tokens) // (B * T)
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {n_batches} batches")

    @classmethod
    def from_text(cls, path, B, T, encode_fn, verbose=True):
        text = Path(path).read_text(encoding="utf-8")
        tokens = encode_fn(text)
        return cls(tokens, B, T, verbose=verbose)

    @classmethod
    def from_npy(cls, path, B, T, verbose=True):
        return cls(load_tokens_npy(path), B, T, verbose=verbose)

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


class RandomWindowLoader:
    """Random (x, y) windows from a pre-tokenized 1-D tensor.

    Same sampling idea as translator utils.get_batch (random start indices), without
    loss masks or train/val split logic — pass train_tokens or val_tokens explicitly.

    Example (translator-shaped pipeline):
        utils.load_data()  # fills utils.train_data, utils.val_data
        train_loader = RandomWindowLoader(utils.train_data, B=config.batch_size, T=config.block_size)
        val_loader = RandomWindowLoader(utils.val_data, B=config.batch_size, T=config.block_size)
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
    """

    def __init__(self, tokens, B, T, verbose=True):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.to(dtype=torch.long)
        if len(tokens) <= T:
            raise ValueError(f"need len(tokens) > T, got {len(tokens)} tokens and T={T}")
        self.tokens = tokens
        self.B = B
        self.T = T
        if verbose:
            print(f"loaded {len(self.tokens):,} tokens (random windows B={B}, T={T})")

    def next_batch(self):
        B, T = self.B, self.T
        ix = torch.randint(len(self.tokens) - T, (B,))
        x = torch.stack([self.tokens[i:i + T] for i in ix])
        y = torch.stack([self.tokens[i + 1:i + T + 1] for i in ix])
        return x, y

    def reset(self):
        pass


class PairBatchLoader:
    """Padded encoder-decoder batches from pre-tokenized (src, tgt) pairs.

    Each next_batch() returns:
      src_ids, tgt_in, targets, src_key_padding_mask, tgt_key_padding_mask
    Shapes: src [B, max_src_len]; tgt_in/targets/masks [B, max_tgt_len - 1].
    Teacher forcing: tgt_in = tgt[:, :-1], targets = tgt[:, 1:] after padding full targets.
    """

    def __init__(
        self,
        pairs: list[tuple[list[int], list[int]]],
        B: int,
        max_src_len: int,
        max_tgt_len: int,
        pad_id: int = 0,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        if not pairs:
            raise ValueError("PairBatchLoader requires at least one pair")
        self.pairs = pairs
        self.B = B
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        self.shuffle = shuffle
        self._cursor = 0
        if verbose:
            print(f"PairBatchLoader: {len(pairs):,} pairs, B={B}, "
                  f"src≤{max_src_len}, tgt≤{max_tgt_len}, pad={pad_id}, shuffle={shuffle}")

    def __len__(self) -> int:
        return (len(self.pairs) + self.B - 1) // self.B

    def reset(self):
        self._cursor = 0

    @staticmethod
    def _pad_batch(seqs: list[list[int]], max_len: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(seqs)
        out = torch.full((B, max_len), pad_id, dtype=torch.long)
        pad = torch.ones(B, max_len, dtype=torch.bool)
        for i, seq in enumerate(seqs):
            n = min(len(seq), max_len)
            if n:
                out[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
                pad[i, :n] = False
        return out, pad

    def next_batch(self):
        B = self.B
        n = len(self.pairs)
        if self.shuffle:
            ix = torch.randint(n, (B,))
            batch = [self.pairs[i] for i in ix.tolist()]
        else:
            if self._cursor >= n:
                self._cursor = 0
            end = min(self._cursor + B, n)
            batch = self.pairs[self._cursor:end]
            if len(batch) < B:
                batch = batch + self.pairs[: B - len(batch)]
            self._cursor = (self._cursor + B) % n
        src_seqs = [p[0] for p in batch]
        tgt_seqs = [p[1] for p in batch]
        src_ids, src_pad = self._pad_batch(src_seqs, self.max_src_len, self.pad_id)
        tgt_full, _ = self._pad_batch(tgt_seqs, self.max_tgt_len, self.pad_id)
        tgt_in = tgt_full[:, :-1].contiguous()
        targets = tgt_full[:, 1:].contiguous()
        tgt_pad = tgt_in == self.pad_id
        return src_ids, tgt_in, targets, src_pad, tgt_pad


def grad_accum_steps(total_batch_size, B, T, world_size=1):
    """Micro-batch size is B*T; return how many next_batch() calls per optimizer step."""
    micro = B * T * world_size
    assert total_batch_size % micro == 0, f"total_batch_size {total_batch_size} not divisible by B*T*world_size ({micro})"
    return total_batch_size // micro
