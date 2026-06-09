"""E2 (headline): end-to-end perplexity of a quantized GPT-2, with/without Hadamard.

Quantizes GPT-2's attention/MLP matmuls post-training and measures the
perplexity of the *actual model* on real text. We use the QuaRot mechanism: a
block-Hadamard rotation folded into each matmul's contraction dimension, so the
rotations cancel `(xHᵀ)(HW)=xW` while both the weights and the activations are
quantized in the incoherent (outlier-free) basis.

Several precision regimes are compared (Wn = n-bit weights, Am = m-bit
activations; weight-only keeps activations full precision). The story: Hadamard's
payoff is largest when **activations** are quantized — that's where GPT-2's
outliers live — and marginal for weight-only.

Embeddings/LayerNorms stay full precision (standard). Writes
figures/exp2_perplexity.png and results/exp2_perplexity.json.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/exp2_perplexity.py
"""

from __future__ import annotations
from experiments._common import plt, ROOT, FIG_DIR, RES_DIR, WEIGHTS

import json
import os


import numpy as np
import torch


from hadamard_monarch.model import GPT2Tokenizer
from hadamard_monarch.model import load_eval_ids
from hadamard_monarch.model import load_gpt2
from hadamard_monarch.model import configure_quant_, reset_quant_


# (label, weight bits, activation bits)
CONFIGS = [
    ("W4 (weight-only)", 4, None),
    ("W8A8", 8, 8),
    ("W4A8", 4, 8),
    ("W8A4", 8, 4),
    ("W4A4", 4, 4),
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ctx = int(os.environ.get("CTX", "512"))

    tok = GPT2Tokenizer.from_dir(os.path.join(ROOT, "models"))
    model = load_gpt2(WEIGHTS, device=device)
    ids = load_eval_ids(tok, max_chars=int(os.environ.get("MAX_CHARS", "200000")))
    print(f"device={device}  eval_tokens={ids.numel()}  ctx={ctx}")

    reset_quant_(model)
    fp_ppl = model.perplexity(ids, ctx=ctx)
    print(f"full-precision GPT-2 perplexity = {fp_ppl:.2f}\n")

    results = {"fp": fp_ppl, "configs": {}}
    for label, wbits, abits in CONFIGS:
        row = {}
        for had in (False, True):
            configure_quant_(model, wbits=wbits, abits=abits, hadamard=had)
            row["hadamard" if had else "baseline"] = model.perplexity(ids, ctx=ctx)
        reset_quant_(model)
        results["configs"][label] = row
        ratio = row["baseline"] / row["hadamard"]
        print(f"{label:18s} baseline PPL={row['baseline']:9.2f}   "
              f"+Hadamard PPL={row['hadamard']:9.2f}   ({ratio:.2f}x better)")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp2_perplexity.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Grouped bar chart, log-y (PPL spans orders of magnitude across regimes).
    os.makedirs(FIG_DIR, exist_ok=True)
    labels = [c[0] for c in CONFIGS]
    base = [results["configs"][l]["baseline"] for l in labels]
    had = [results["configs"][l]["hadamard"] for l in labels]
    x = np.arange(len(labels))
    w = 0.38
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - w / 2, base, w, label="baseline (no rotation)")
    plt.bar(x + w / 2, had, w, label="+ Hadamard")
    plt.axhline(fp_ppl, color="gray", ls="--", lw=1, label=f"full precision ({fp_ppl:.1f})")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("perplexity (log scale, lower = better)")
    plt.title("Quantized GPT-2 perplexity: Hadamard helps most when activations are quantized")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp2_perplexity.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out} and results/exp2_perplexity.json")


if __name__ == "__main__":
    main()
