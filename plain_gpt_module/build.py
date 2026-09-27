import torch

from .neural_network import GPT

def build_model(
    config,
    device="cuda",
    use_compile=True,
    matmul_precision="high",
    mode="train",
):
    """
    Create GPT, move to device, optionally compile and set matmul precision.
    mode: "train" | "eval"
    matmul_precision: "highest" | "high" | "medium" | None to skip
    """
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    model = GPT(config)
    if mode == "train":
        model.train()
    else:
        model.eval()
    model.to(device)
    if use_compile:
        model = torch.compile(model, dynamic=True)
    return model

def unwrap_compiled(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model
