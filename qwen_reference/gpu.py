#!/usr/bin/env python3
"""GPU memory on the local machine: who holds VRAM, and torch cache clear (current process only).

Usage:
  python -m qwen_reference.gpu status
  python -m qwen_reference.gpu empty-cache
"""
from __future__ import annotations

import gc
import subprocess
import sys


def status() -> None:
    subprocess.run(["nvidia-smi"], check=False)
    print(
        "\nNote: Qwen3-VL-2B wants ~5 GB contiguous for inference. "
        "If another service (often ollama ~9 GB) fills the 3060, "
        "transformers offloads weights to CPU ('meta device' warning)."
    )


def empty_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("torch.cuda.empty_cache() done (frees unused blocks in *this* Python process only).")
        else:
            print("CUDA not available in this process.")
    except ImportError:
        print("torch not installed.", file=sys.stderr)
        sys.exit(1)
    status()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "empty-cache":
        empty_cache()
    else:
        raise SystemExit(f"unknown command {cmd!r}; use status | empty-cache")


if __name__ == "__main__":
    main()
