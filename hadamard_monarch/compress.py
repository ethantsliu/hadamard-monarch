"""Replace GPT-2's dense MLPs with Monarch layers, fit them (data-aware), fine-tune, and quantize the factors."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import MonarchLinear
from .model import GPT2, MLP, Conv1D, gelu_new


class MonarchMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, nblocks: int):
        super().__init__()
        self.c_fc = MonarchLinear(d_model, d_ff, nblocks=nblocks)
        self.c_proj = MonarchLinear(d_ff, d_model, nblocks=nblocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(x)))


def fit_monarch_to_linear(monarch: MonarchLinear, Wc: torch.Tensor, bias: torch.Tensor,
                          steps: int = 200, lr: float = 5e-3, device="cpu") -> None:
    """Initialize a MonarchLinear by least-squares-fitting it to a dense map y = xWc + b."""
    in_f = Wc.shape[0]
    monarch.to(device)
    opt = torch.optim.Adam(monarch.parameters(), lr=lr)
    for _ in range(steps):
        x = torch.randn(256, in_f, device=device)
        target = x @ Wc.to(device) + bias.to(device)
        opt.zero_grad()
        F.mse_loss(monarch(x), target).backward()
        opt.step()


def swap_mlps_with_monarch(model: GPT2, nblocks: int, fit_init_steps: int = 200,
                           device: str = "cpu") -> GPT2:
    """Replace every Block.mlp with a MonarchMLP, optionally fit-initialized to the dense MLP."""
    d_model = model.cfg.n_embd
    d_ff = 4 * d_model
    for block in model.h:
        old = block.mlp
        mm = MonarchMLP(d_model, d_ff, nblocks)
        if fit_init_steps > 0:
            fit_monarch_to_linear(mm.c_fc, old.c_fc.weight.data, old.c_fc.bias.data,
                                  steps=fit_init_steps, device=device)
            fit_monarch_to_linear(mm.c_proj, old.c_proj.weight.data, old.c_proj.bias.data,
                                  steps=fit_init_steps, device=device)
        block.mlp = mm.to(device)
    return model


def finetune_with_eval(model: GPT2, train_ids: torch.Tensor, val_ids: torch.Tensor,
                       total_steps: int, eval_every: int, ctx: int = 256, bs: int = 8,
                       peak_lr: float = 2.5e-4, warmup: int = 40, device: str = "cpu",
                       seed: int = 0, eval_ctx: int = 512, optimizer: str = "adamw") -> list:
    """End-to-end fine-tune with a single optimizer + cosine LR + grad clipping.

    Evaluates held-out perplexity every ``eval_every`` steps. Returns a list of
    (step, val_ppl) including the pre-training point at step 0. ``optimizer`` is
    "adamw" (default) or "sgd" — SGD+momentum keeps only one state buffer (~half
    AdamW's memory), letting a full-size model fine-tune on a tight-RAM laptop.
    """
    torch.manual_seed(seed)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=peak_lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    g = torch.Generator().manual_seed(seed)
    n = train_ids.numel()
    traj = [(0, model.perplexity(val_ids, ctx=eval_ctx))]
    for step in range(1, total_steps + 1):
        if step <= warmup:
            lr = peak_lr * step / warmup
        else:
            prog = (step - warmup) / max(1, total_steps - warmup)
            lr = 0.5 * peak_lr * (1 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        ix = torch.randint(0, n - ctx - 1, (bs,), generator=g)
        xb = torch.stack([train_ids[i:i + ctx] for i in ix]).to(device)
        yb = torch.stack([train_ids[i + 1:i + 1 + ctx] for i in ix]).to(device)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0:
            traj.append((step, model.perplexity(val_ids, ctx=eval_ctx)))
    return traj


@torch.no_grad()
def collect_activations(
    model: GPT2,
    ids: torch.Tensor,
    layer_module: nn.Module,
    max_samples: int = 4096,
    ctx: int = 256,
) -> torch.Tensor:
    """Capture the real input activations that feed ``layer_module``.

    Runs ``model`` forward over non-overlapping windows of the 1-D token tensor
    ``ids`` with a ``forward_pre_hook`` on ``layer_module`` that records its input
    tensor. Batch and time dims are flattened, so the result is a 2-D tensor of
    individual token activations.

    Args:
        model: a GPT2 (or compatible) module to run forward.
        ids: 1-D LongTensor of token ids to feed the model.
        layer_module: the submodule whose *input* we want (e.g. ``block.mlp``).
        max_samples: stop once this many (token) activation rows are collected.
        ctx: context window length per forward pass.

    Returns:
        A ``(N, in_features)`` float tensor of real activations, ``N <= max_samples``.
    """
    device = model.wte.weight.device
    chunks: list[torch.Tensor] = []
    # The hook stashes the latest window's input here; the loop drains it.
    latest: list[torch.Tensor] = []

    def _pre_hook(_module, args):
        # forward_pre_hook receives positional ``args``; the layer input is args[0].
        x = args[0]
        latest.append(x.reshape(-1, x.size(-1)).detach().to("cpu").float())

    handle = layer_module.register_forward_pre_hook(_pre_hook)
    model.eval()
    total = 0
    try:
        n = ids.numel()
        for begin in range(0, n - 1, ctx):
            if total >= max_samples:
                break
            end = min(begin + ctx, n)
            if end - begin < 2:
                break
            window = ids[begin:end].to(device)[None, :]
            latest.clear()
            model(window)
            x = latest[-1]
            chunks.append(x)
            total += x.size(0)
    finally:
        handle.remove()

    return torch.cat(chunks, dim=0)[:max_samples].contiguous()


def _dense_target(sublayer: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """The dense sublayer's output on ``X`` (the fitting target). No grad."""
    with torch.no_grad():
        return sublayer(X)


def _new_monarch_for(sublayer: nn.Module, nblocks: int, device: str) -> nn.Module:
    """Build an untrained Monarch module matching the shape of ``sublayer``.

    - ``Conv1D`` (y = x @ W + b, W is (in, out))  -> a single ``MonarchLinear``.
    - ``MLP`` (c_fc -> gelu_new -> c_proj)         -> a ``MonarchMLP``.
    """
    if isinstance(sublayer, Conv1D):
        in_f, out_f = sublayer.weight.shape  # Conv1D stores (in, out)
        return MonarchLinear(in_f, out_f, nblocks=nblocks, bias=True).to(device)
    if isinstance(sublayer, MLP):
        d_model = sublayer.c_fc.weight.shape[0]   # (in=d_model, out=d_ff)
        d_ff = sublayer.c_fc.weight.shape[1]
        return MonarchMLP(d_model, d_ff, nblocks=nblocks).to(device)
    raise TypeError(
        f"fit target must be a Conv1D or MLP, got {type(sublayer).__name__}"
    )


def fit_monarch(
    target: nn.Module,
    X: torch.Tensor,
    steps: int = 600,
    lr: float = 3e-3,
    nblocks: int = 16,
    batch: int = 512,
    device: str = "cpu",
    seed: int = 0,
) -> nn.Module:
    """Fit a Monarch approximation to reproduce ``target``'s output on inputs ``X``.

    Supports fitting EITHER a single :class:`MonarchLinear` to a :class:`Conv1D`
    (``y = x @ W + b``) OR a :class:`MonarchMLP` to a dense :class:`MLP` block
    (``c_fc -> gelu_new -> c_proj``). The objective is the MSE between the Monarch
    output and the dense ``target`` output, both evaluated on the SAME inputs
    ``X`` — so passing real activations gives a data-aware (activation-aware) fit,
    while passing Gaussian probes gives the naive weight-error fit.

    Args:
        target: the dense sublayer to approximate (``Conv1D`` or ``MLP``).
        X: ``(N, in_features)`` inputs the Monarch is fit on.
        steps: number of Adam steps.
        lr: Adam learning rate.
        nblocks: Monarch block count (parameter budget knob).
        batch: minibatch size drawn (with replacement) from ``X`` each step.
        device: torch device.
        seed: RNG seed for reproducible minibatch sampling / init.

    Returns:
        The fitted Monarch module (``MonarchLinear`` or ``MonarchMLP``).
    """
    torch.manual_seed(seed)
    X = X.to(device)
    target_y = _dense_target(target, X).to(device)

    mono = _new_monarch_for(target, nblocks=nblocks, device=device)
    opt = torch.optim.Adam(mono.parameters(), lr=lr)
    n = X.size(0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        xb, yb = X[idx], target_y[idx]
        opt.zero_grad()
        F.mse_loss(mono(xb), yb).backward()
        opt.step()
    return mono


@torch.no_grad()
def output_rel_error(mono: nn.Module, target: nn.Module, X: torch.Tensor) -> float:
    """Relative output error ``||mono(X) - target(X)|| / ||target(X)||`` on ``X``."""
    y = target(X)
    return (mono(X) - y).norm().item() / y.norm().item()


@torch.no_grad()
def quantize_monarch_factors(mono: nn.Module, bits: int = 4, hadamard: bool = False) -> nn.Module:
    """Return a deep copy of ``mono`` with all MonarchLinear factors quantized.

    Works for a single :class:`MonarchLinear` or a :class:`MonarchMLP` (both of
    its factor-bearing submodules). Uses ``monarch_quant.quantize_factor`` per
    factor. With ``hadamard=False`` (the default — C1 showed the rotation does not
    help Monarch factors) plain symmetric per-block quantization is used.
    """
    from .quant import quantize_factor  # local import to avoid cycles

    m = copy.deepcopy(mono)
    for sub in m.modules():
        if isinstance(sub, MonarchLinear):
            sub.w1.data.copy_(quantize_factor(sub.w1.data, bits=bits, hadamard=hadamard))
            sub.w2.data.copy_(quantize_factor(sub.w2.data, bits=bits, hadamard=hadamard))
    return m
