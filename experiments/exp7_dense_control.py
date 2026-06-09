"""E7 (control): how much of the Monarch recovery is just in-domain adaptation?

The E3/E6 Monarch models recover perplexity by fine-tuning on a handful of
19th-century books, then evaluating on held-out Alice — all same-era English. A
skeptic rightly asks: how much of that recovery is Monarch *fitting* vs. simply
adapting the model to the training domain? Any model fine-tuned on those books
would improve on Alice.

This control answers it: fine-tune the FULL DENSE GPT-2 the exact same way (same
books, steps, schedule) and watch held-out Alice perplexity. That curve is the
"in-domain adaptation floor". The Monarch model can't beat it (fewer params); the
gap between the two is the genuine compression cost.

Only ONE GPT-2 is loaded at a time (the Monarch curve is read from the saved
results/exp6_doubly_compressed.json, not recomputed). The optimizer defaults to
memory-light SGD+momentum: fine-tuning the full 124M model with AdamW needs ~2×
the parameters in optimizer state and swap-thrashes a tight-RAM laptop, whereas
SGD (one momentum buffer) fits. SGD yields a conservative (slightly higher)
in-domain floor — a fair *lower bound* on the structural cost. Set OPT=adamw on a
machine with more RAM for a tighter floor.

Writes figures/exp7_dense_control.png + results/exp7_dense_control.json.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/exp7_dense_control.py
"""

from __future__ import annotations
from experiments._common import plt, ROOT, FIG_DIR, RES_DIR, WEIGHTS

import glob
import json
import os


import torch


from hadamard_monarch.model import GPT2Tokenizer
from hadamard_monarch.model import load_eval_ids
from hadamard_monarch.model import load_gpt2
from hadamard_monarch.compress import finetune_with_eval



def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    steps = int(os.environ.get("STEPS", "700"))
    eval_every = max(100, steps // 6)
    print(f"device={device}  steps={steps}")

    tok = GPT2Tokenizer.from_dir(os.path.join(ROOT, "models"))
    train_ids = torch.cat([load_eval_ids(tok, path=p, max_chars=2000000)
                           for p in sorted(glob.glob(os.path.join(ROOT, "models", "book_*.txt")))])
    val_ids = load_eval_ids(tok, path=os.path.join(ROOT, "models", "alice.txt"), max_chars=200000)

    # ONE model only.
    # Fine-tuning the FULL 124M model with AdamW (2x params in state) swap-thrashes
    # a tight-RAM laptop, so default to memory-light SGD+momentum and small batch.
    bs = int(os.environ.get("BS", "4"))
    ft_ctx = int(os.environ.get("FTCTX", "256"))
    opt = os.environ.get("OPT", "sgd")
    peak_lr = float(os.environ.get("PEAK_LR", "5e-3" if opt == "sgd" else "2.5e-4"))
    model = load_gpt2(WEIGHTS, device=device)
    print(f"dense fp val PPL = {model.perplexity(val_ids, ctx=512):.2f}  "
          f"(optimizer={opt}, lr={peak_lr})", flush=True)
    ctrl = finetune_with_eval(model, train_ids, val_ids, total_steps=steps,
                              eval_every=eval_every, ctx=ft_ctx, bs=bs, peak_lr=peak_lr,
                              device=device, seed=0, optimizer=opt)
    for s, p in ctrl:
        print(f"  dense ft step {s:>4}: val PPL={p:.2f}")
    dense_best = min(p for _, p in ctrl)

    # Monarch curve from the saved capstone run (not recomputed).
    monarch = None
    e6_path = os.path.join(RES_DIR, "exp6_doubly_compressed.json")
    if os.path.exists(e6_path):
        e6 = json.load(open(e6_path))
        monarch = {"trajectory": e6["trajectory"], "fp": e6["ppl_monarch_fp"],
                   "params_frac": e6["params_monarch"] / e6["params_fp"]}
        print(f"\ndense control best PPL = {dense_best:.2f}   "
              f"Monarch best PPL = {min(p for _, p in monarch['trajectory']):.2f}   "
              f"compression cost ≈ {min(p for _,p in monarch['trajectory']) - dense_best:.1f} PPL")

    os.makedirs(RES_DIR, exist_ok=True)
    json.dump({"dense_control": ctrl, "dense_best": dense_best}, open(
        os.path.join(RES_DIR, "exp7_dense_control.json"), "w"), indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot([s for s, _ in ctrl], [p for _, p in ctrl], "^-", color="tab:green",
             label="dense control (100% params)")
    if monarch:
        m = monarch["trajectory"]
        plt.plot([s for s, _ in m], [p for _, p in m], "o-", color="tab:purple",
                 label=f"Monarch MLP ({monarch['params_frac']:.0%} params)")
    plt.yscale("log")
    plt.xlabel("fine-tuning steps (same books, same schedule)")
    plt.ylabel("held-out perplexity (log)")
    plt.title("In-domain adaptation vs. compression cost\n"
              "(gap between the curves = what Monarch's structure actually costs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp7_dense_control.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out} and results/exp7_dense_control.json")


if __name__ == "__main__":
    main()
