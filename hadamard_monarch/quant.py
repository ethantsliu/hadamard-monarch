"""Quantization: dense round-to-nearest, the Hadamard incoherence trick, and Monarch-factor quantization."""
from __future__ import annotations

import copy

import torch

from .transforms import fwht, next_pow2, pad_to_pow2, MonarchLinear


def incoherence(W: torch.Tensor) -> float:
    """Outlier metric: max |entry| relative to RMS entry. Higher = more outlier-dominated."""
    rms = W.pow(2).mean().sqrt()
    return (W.abs().max() / rms).item()


def quantize(
    W: torch.Tensor, bits: int = 4, per_row: bool = True, group_size: int | None = None
) -> torch.Tensor:
    """Symmetric uniform quantization to ``bits`` bits, then dequantize.

    Args:
        W: weight matrix (out, in).
        bits: bit-width (e.g. 4 -> levels in [-8, 7]).
        per_row: one scale per output channel (row) if True, else one global scale
            (per-tensor). Ignored when ``group_size`` is set.
        group_size: if given, use one scale per contiguous group of ``group_size``
            input columns within each row (group-wise quantization, as in GPTQ/AWQ).
            ``in`` must be divisible by ``group_size``. This is the finest, most
            outlier-robust granularity.

    Returns:
        The dequantized approximation of ``W`` (same shape/dtype).
    """
    qmax = 2 ** (bits - 1) - 1
    if group_size is not None:
        out, inn = W.shape
        if inn % group_size != 0:
            raise ValueError(f"in ({inn}) must be divisible by group_size ({group_size})")
        Wg = W.view(out, inn // group_size, group_size)
        scale = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
        q = (Wg / scale).round().clamp(-qmax - 1, qmax)
        return (q * scale).view(out, inn)
    if per_row:
        scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    else:
        scale = W.abs().max().clamp_min(1e-12) / qmax
    q = (W / scale).round().clamp(-qmax - 1, qmax)
    return q * scale


def _largest_pow2_divisor(n: int) -> int:
    b = 1
    while n % (b << 1) == 0:
        b <<= 1
    return b


def block_fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """Apply the normalized FWHT within contiguous blocks of the last dim."""
    *lead, d = x.shape
    if d % block != 0:
        raise ValueError(f"last dim {d} not divisible by block {block}")
    return fwht(x.reshape(*lead, d // block, block)).reshape(*lead, d)


def hadamard_rotate(W: torch.Tensor) -> torch.Tensor:
    """Two-sided block-wise Hadamard rotation of W (rows and columns).

    Each dimension is rotated by a *block-diagonal* Hadamard whose block size is
    the largest power of two dividing that dimension — so no zero-padding is
    needed (padding would corrupt coarse, e.g. per-tensor, quantization) and the
    transform is exactly orthogonal. This is the block-Hadamard approach used by
    QuaRot. The op is its own inverse, so ``hadamard_unrotate`` just reapplies it.
    """
    out, inn = W.shape
    W = block_fwht(W, _largest_pow2_divisor(inn))            # rotate columns
    W = block_fwht(W.t().contiguous(), _largest_pow2_divisor(out)).t()  # rotate rows
    return W.contiguous()


def hadamard_unrotate(Wr: torch.Tensor) -> torch.Tensor:
    """Inverse of ``hadamard_rotate`` (which is an involution)."""
    return hadamard_rotate(Wr)


def quantize_with_hadamard(
    W: torch.Tensor, bits: int = 4, per_row: bool = True, group_size: int | None = None
) -> torch.Tensor:
    """Quantize ``W`` in the Hadamard-rotated (incoherent) basis, then rotate back.

    Granularity options match ``quantize``. The rotation is block-wise with no
    padding (see ``hadamard_rotate``), so ``group_size`` just needs to divide the
    matrix's own ``in`` dimension (powers of two like 128 do for GPT-2's dims).
    """
    Wr = hadamard_rotate(W)
    Wr_q = quantize(Wr, bits=bits, per_row=per_row, group_size=group_size)
    return hadamard_unrotate(Wr_q)


def rel_error(approx: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Frobenius error ||approx - target|| / ||target||."""
    return (approx - target).norm().item() / target.norm().item()


@torch.no_grad()
def quantize_factor(W: torch.Tensor, bits: int = 4, hadamard: bool = False) -> torch.Tensor:
    """Quantize a 3D Monarch factor block-by-block to ``bits`` bits.

    ``W`` has shape ``(num_blocks, rows, cols)``; each ``(rows, cols)`` slice is a
    2D block quantized independently. With ``hadamard=True`` the per-block
    block-Hadamard rotation of ``quantize_with_hadamard`` is applied before
    quantizing (and undone after); otherwise plain symmetric quantization is used.

    Returns a dequantized copy of ``W`` with the same shape/dtype.
    """
    if W.dim() != 3:
        raise ValueError(f"expected a 3D Monarch factor (num_blocks, rows, cols), got shape {tuple(W.shape)}")
    q = quantize_with_hadamard if hadamard else quantize
    out = torch.empty_like(W)
    for i in range(W.shape[0]):
        out[i] = q(W[i], bits=bits)
    return out


@torch.no_grad()
def quantized_dense(mono: MonarchLinear, bits: int = 4, hadamard: bool = False) -> torch.Tensor:
    """Dense matrix realized by ``mono`` after quantizing its factors to ``bits`` bits.

    Temporarily swaps the quantized factors into a *copy* of ``mono`` (the input
    module is left untouched), materializes ``to_dense()``, and returns it. With
    ``hadamard=True`` each factor block is rotated into the incoherent basis
    before quantizing. Pair this with the un-quantized ``mono.to_dense()`` to
    measure the factor-quantization error.
    """
    m = copy.deepcopy(mono)
    m.w1.data.copy_(quantize_factor(mono.w1.data, bits=bits, hadamard=hadamard))
    m.w2.data.copy_(quantize_factor(mono.w2.data, bits=bits, hadamard=hadamard))
    return m.to_dense()


@torch.no_grad()
def factor_incoherence(mono: MonarchLinear) -> float:
    """Incoherence (max/RMS entry) of the Monarch factors: max over ``w1`` and ``w2``.

    A single number summarizing how outlier-dominated the *factors* are, directly
    comparable to ``incoherence(W)`` on the dense weight.
    """
    return max(incoherence(mono.w1.data), incoherence(mono.w2.data))
