"""Structured-matrix primitives: the fast Walsh–Hadamard transform and the Monarch layer."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    return 1 << (max(1, n) - 1).bit_length()


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform along the last dimension.

    The last dimension must be a power of two. The transform is normalized by
    1/sqrt(n) so that it is *orthogonal*: ``fwht`` is its own inverse
    (an involution) and preserves the L2 norm.

    Args:
        x: tensor of shape (..., n) with n a power of two.

    Returns:
        Tensor of the same shape, Hadamard-transformed along the last axis.
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Last dim must be a power of two, got {n}")

    orig_shape = x.shape
    y = x.reshape(-1, n).clone()

    # Iterative butterfly: at each stage, combine pairs of half-blocks.
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        y = y.view(-1, n)
        h *= 2

    y = y * (n ** -0.5)
    return y.view(orig_shape)


def pad_to_pow2(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad the last dimension up to the next power of two (no-op if already)."""
    n = x.shape[-1]
    target = next_pow2(n)
    if target == n:
        return x
    pad = target - n
    return torch.nn.functional.pad(x, (0, pad))


class HadamardMix(nn.Module):
    """Apply a (optionally randomized, optionally scaled) Hadamard transform.

    By default this is a *parameter-free* orthogonal mixing of the input's last
    dimension — it adds no learnable weights and costs O(n log n).

    Options:
        randomize: multiply by a fixed random {+1, -1} sign vector before the
            transform. This is the "randomized Hadamard transform" used by QuIP#
            to make the mixing data-independent yet decorrelated across runs.
        learn_scale: add a learnable per-channel diagonal scale applied *after*
            the transform (init = ones). This is a cheap nod to SpinQuant's
            observation that *learned* rotations can beat a fixed Hadamard.
    """

    def __init__(self, dim: int, randomize: bool = True, learn_scale: bool = False):
        super().__init__()
        self.dim = dim
        self.padded_dim = next_pow2(dim)

        if randomize:
            # Fixed (non-learnable) +-1 signs, stored as a buffer so it moves with
            # the module across devices and is saved in the state dict.
            signs = torch.randint(0, 2, (self.padded_dim,)).float() * 2 - 1
            self.register_buffer("signs", signs)
        else:
            self.signs = None

        if learn_scale:
            self.scale = nn.Parameter(torch.ones(self.padded_dim))
        else:
            self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_to_pow2(x)
        if self.signs is not None:
            x = x * self.signs
        x = fwht(x)
        if self.scale is not None:
            x = x * self.scale
        # Trim back to the requested dim (padding contributes nothing meaningful).
        return x[..., : self.dim]


class MonarchLinear(nn.Module):
    """Structured linear layer ``y = Monarch(x) + bias``.

    Args:
        in_features: input dimension (must be divisible by ``nblocks``).
        out_features: output dimension (must be divisible by ``nblocks``).
        nblocks: number of diagonal blocks. ``None`` picks ~sqrt(in_features),
            the choice that minimizes parameters for a square layer.
        bias: whether to add a learnable bias.
    """

    def __init__(self, in_features, out_features, nblocks=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if nblocks is None:
            nblocks = self._default_nblocks(in_features)
        if in_features % nblocks != 0 or out_features % nblocks != 0:
            raise ValueError(
                f"in_features ({in_features}) and out_features ({out_features}) "
                f"must both be divisible by nblocks ({nblocks})"
            )
        self.nblocks = nblocks
        self.in_per = in_features // nblocks
        self.out_per = out_features // nblocks

        # Factor 1: nblocks blocks, each maps in_per -> out_per.
        self.w1 = nn.Parameter(torch.empty(nblocks, self.out_per, self.in_per))
        # Factor 2: out_per blocks, each maps nblocks -> nblocks.
        self.w2 = nn.Parameter(torch.empty(self.out_per, nblocks, nblocks))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @staticmethod
    def _default_nblocks(n: int) -> int:
        """Largest divisor of n that is <= sqrt(n) (keeps blocks roughly square)."""
        root = int(math.isqrt(n))
        for b in range(root, 0, -1):
            if n % b == 0:
                return b
        return 1

    def reset_parameters(self) -> None:
        # Kaiming-style init scaled per factor so the composed map has a sane variance.
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, n = x.shape
        if n != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {n}")
        x = x.reshape(-1, self.nblocks, self.in_per)            # (N, b, in_per)

        # Factor 1: block-diagonal mix within each of the `nblocks` blocks.
        y = torch.einsum("boi,nbi->nbo", self.w1, x)            # (N, b, out_per)

        # Permutation: transpose block <-> within-block axes.
        y = y.transpose(1, 2)                                   # (N, out_per, b)

        # Factor 2: block-diagonal mix across the (now-leading) block axis.
        y = torch.einsum("pqr,npr->npq", self.w2, y)            # (N, out_per, b)

        y = y.reshape(*batch, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def param_count(self) -> int:
        n = self.w1.numel() + self.w2.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    @torch.no_grad()
    def to_dense(self) -> torch.Tensor:
        """Materialize the equivalent dense weight matrix (out_features, in_features).

        Pushes the identity basis through ``forward`` (bias excluded), so the
        result is exactly the linear map this layer realizes. Useful for tests
        and for measuring how well the structure approximates a target matrix.
        """
        eye = torch.eye(self.in_features, device=self.w1.device, dtype=self.w1.dtype)
        cols = self.forward(eye)            # (in_features, out_features), incl. bias
        if self.bias is not None:
            cols = cols - self.bias         # broadcast-subtract the bias row
        return cols.t().contiguous()
