"""Experiment 0: the idea that DIDN'T work (kept for honesty + understanding).

The project started from an appealing intuition: a Monarch layer is built from
block-diagonal factors, which only mix coordinates *within* a block, so maybe
pre-mixing the input with a cheap global Hadamard transform would let it
approximate a dense matrix better — buying a "global receptive field" for free.

This experiment tests that directly: fit Monarch alone vs Hadamard -> Monarch to
a random dense target, across parameter budgets. The result is a flat no:
Hadamard barely moves the error.

Two reasons, both worth understanding:
  1. A Monarch matrix already mixes across blocks through its built-in
     permutation — it is not as receptive-field-starved as a pure block-diagonal
     matrix.
  2. A *fixed orthogonal* pre-rotation cannot enlarge the set of matrices a
     structured family can represent when the target is generic (no outliers,
     no special basis). Rotating the input just rotates the target; a random
     dense target stays exactly as hard.

That second point is the same reason the Hadamard quantization trick (exp1/exp2)
needs *outliers* to help. Run this, then run exp1 to see where Hadamard really
earns its keep.

Writes figures/exp0_receptive_field.png.
"""

from __future__ import annotations
from experiments._common import plt, FIG_DIR

import os


import torch
import torch.nn.functional as F


from hadamard_monarch.transforms import HadamardMix
from hadamard_monarch.transforms import MonarchLinear



def divisors_up_to_sqrt(n):
    return [b for b in range(2, int(n ** 0.5) + 1) if n % b == 0]


def fit(W, nblocks, use_hadamard, steps=800, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    n = W.shape[0]
    monarch = MonarchLinear(n, n, nblocks=nblocks, bias=False)
    hadamard = HadamardMix(n, randomize=True, learn_scale=False) if use_hadamard else None
    opt = torch.optim.Adam(monarch.parameters(), lr=lr)
    X = torch.randn(256, n)
    target = X @ W.t()
    for _ in range(steps):
        opt.zero_grad()
        h = hadamard(X) if hadamard is not None else X
        loss = F.mse_loss(monarch(h), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = hadamard(X) if hadamard is not None else X
        err = (monarch(h) - target).norm() / target.norm()
    return err.item(), monarch.param_count()


def main():
    torch.manual_seed(0)
    n = 256
    W = torch.randn(n, n) / n ** 0.5
    rows = []
    for b in divisors_up_to_sqrt(n):
        em, p = fit(W, b, use_hadamard=False)
        eh, _ = fit(W, b, use_hadamard=True)
        rows.append((p / (n * n) * 100, em, eh))
        print(f"params={p:>6,} ({p/(n*n):5.1%})  monarch={em:.4f}  hadamard+monarch={eh:.4f}")
    rows.sort()

    os.makedirs(FIG_DIR, exist_ok=True)
    xs = [r[0] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [r[1] for r in rows], "o-", label="Monarch alone")
    plt.plot(xs, [r[2] for r in rows], "s-", label="Hadamard → Monarch")
    plt.xlabel("parameter budget (% of dense)")
    plt.ylabel("relative error ||$\\hat{W}$ - W|| / ||W||")
    plt.title("The idea that didn't work:\nHadamard does NOT help approximate a random dense matrix")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp0_receptive_field.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
