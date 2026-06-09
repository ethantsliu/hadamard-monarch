"""Support utilities: synthetic outlier weights, real GPT-2 weight loading, parameter counting."""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def outlier_weight(
    out_features: int = 512,
    in_features: int = 512,
    n_outlier_channels: int = 8,
    outlier_scale: float = 25.0,
    base_std: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """Gaussian weights with a few input channels (columns) scaled up.

    Returns a (out_features, in_features) tensor.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_features, in_features, generator=g) * base_std
    cols = torch.randperm(in_features, generator=g)[:n_outlier_channels]
    W[:, cols] *= outlier_scale
    return W


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "gpt2_pytorch_model.bin",
)


def available(path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(path)


def load_gpt2_weight(key: str = "h.0.mlp.c_fc.weight", path: str = DEFAULT_PATH) -> torch.Tensor:
    """Return a real GPT-2 weight matrix as (out_features, in_features), float32.

    GPT-2 stores its linear layers as ``Conv1D`` with weights shaped
    ``(in, out)``, so we transpose to the conventional ``(out, in)``.

    Raises FileNotFoundError if the checkpoint hasn't been downloaded (see module
    docstring for the one-line curl command).
    """
    if not available(path):
        raise FileNotFoundError(
            f"GPT-2 checkpoint not found at {path}. Download it with:\n"
            f"  mkdir -p models && curl -fsSL "
            f"https://huggingface.co/gpt2/resolve/main/pytorch_model.bin "
            f"-o {path}"
        )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if key not in sd:
        raise KeyError(f"{key!r} not in checkpoint. Example keys: "
                       f"{[k for k in list(sd)[:6]]}")
    return sd[key].t().contiguous().float()


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
