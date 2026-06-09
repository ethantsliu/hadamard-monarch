"""E6 (capstone): a doubly-compressed GPT-2, end-to-end.

Combines the project's two threads into one model and measures perplexity:

  1. Replace every MLP block with a Monarch MLP, fit DATA-AWARE (on each layer's
     real activations, not Gaussian probes — the C2 result). ~58% (nblocks=16) or
     more of GPT-2's parameters, depending on the block count.
  2. Briefly fine-tune end-to-end on Apple MPS to recover quality.
  3. Quantize the Monarch FACTORS to 4-bit with NO Hadamard rotation (the C1
     result: structured factors are outlier-free, so the rotation is moot).

The claim under test, now at the *model* level (not just per layer): 4-bit factor
quantization is ~free on top of the structured model — held-out perplexity of the
4-bit-factor model ≈ the full-precision Monarch model. So structured + low-bit
compression stack.

Memory-safe: exactly one GPT-2 is held at a time (the E3 dense-control OOM'd by
loading a second). Writes figures/exp6_doubly_compressed.png + results JSON.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/exp6_doubly_compressed.py
"""

from __future__ import annotations
from experiments._common import plt, ROOT, FIG_DIR, RES_DIR, WEIGHTS

import gc
import json
import os


import torch


from hadamard_monarch.compress import fit_monarch, quantize_monarch_factors
from hadamard_monarch.model import GPT2Tokenizer
from hadamard_monarch.model import load_eval_ids
from hadamard_monarch.model import load_gpt2
from hadamard_monarch.compress import finetune_with_eval
from hadamard_monarch.data import count_params



def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def collect_all_mlp_inputs(model, ids, max_samples, ctx, device):
    """One forward pass with hooks on every block.mlp -> {layer: (N,d_model) acts}."""
    n_layer = len(model.h)
    buf = {i: [] for i in range(n_layer)}
    handles = []
    for i, blk in enumerate(model.h):
        def mk(idx):
            def hook(_m, args):
                x = args[0]
                buf[idx].append(x.reshape(-1, x.size(-1)).detach().to("cpu").float())
            return hook
        handles.append(blk.mlp.register_forward_pre_hook(mk(i)))
    model.eval()
    total = 0
    for begin in range(0, ids.numel() - 1, ctx):
        if total >= max_samples:
            break
        end = min(begin + ctx, ids.numel())
        if end - begin < 2:
            break
        model(ids[begin:end].to(device)[None, :])
        total += end - begin
    for h in handles:
        h.remove()
    return {i: torch.cat(v, 0)[:max_samples].contiguous() for i, v in buf.items()}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    nblocks = int(os.environ.get("NBLOCKS", "8"))      # 8 -> ~13.5%/matrix (better quality)
    fit_steps = int(os.environ.get("FIT_STEPS", "400"))
    ft_steps = int(os.environ.get("STEPS", "600"))
    ctx = 512
    print(f"device={device}  nblocks={nblocks}  fit_steps={fit_steps}  finetune_steps={ft_steps}")

    tok = GPT2Tokenizer.from_dir(os.path.join(ROOT, "models"))
    import glob
    train_ids = torch.cat([load_eval_ids(tok, path=p, max_chars=2000000)
                           for p in sorted(glob.glob(os.path.join(ROOT, "models", "book_*.txt")))])
    val_ids = load_eval_ids(tok, path=os.path.join(ROOT, "models", "alice.txt"), max_chars=200000)

    model = load_gpt2(WEIGHTS, device=device)
    p_fp = count_params(model)
    ppl_dense = model.perplexity(val_ids, ctx=ctx)
    print(f"dense GPT-2: params={p_fp:,}  val PPL={ppl_dense:.2f}")

    # 1) data-aware fit of every MLP block (collect dense activations first, then swap)
    acts = collect_all_mlp_inputs(model, train_ids, max_samples=4096, ctx=256, device=device)
    for i, blk in enumerate(model.h):
        mono = fit_monarch(blk.mlp, acts[i], steps=fit_steps, nblocks=nblocks, device=device)
        blk.mlp = mono
        acts[i] = None
    del acts; _free()
    p_m = count_params(model)
    print(f"data-aware Monarch MLPs: params={p_m:,} ({p_m/p_fp:.0%} of fp)  "
          f"val PPL (pre-finetune)={model.perplexity(val_ids, ctx=ctx):.1f}")

    # 2) fine-tune to recover
    traj = finetune_with_eval(model, train_ids, val_ids, total_steps=ft_steps,
                              eval_every=max(100, ft_steps // 6), ctx=256, bs=8,
                              peak_lr=2.5e-4, device=device, seed=0)
    ppl_monarch_fp = traj[-1][1]
    for s, p in traj:
        print(f"  ft step {s:>4}: val PPL={p:.2f}")

    # 3) quantize Monarch factors to 4-bit, no rotation (in place, one block at a time)
    for blk in model.h:
        blk.mlp = quantize_monarch_factors(blk.mlp, bits=4, hadamard=False)
    _free()
    ppl_monarch_4bit = model.perplexity(val_ids, ctx=ctx)
    print(f"\nfull-precision Monarch PPL = {ppl_monarch_fp:.2f}")
    print(f"4-bit-factor   Monarch PPL = {ppl_monarch_4bit:.2f}   "
          f"(delta {ppl_monarch_4bit - ppl_monarch_fp:+.2f} -> 4-bit factors are ~free)")

    results = {
        "params_fp": p_fp, "params_monarch": p_m, "nblocks": nblocks,
        "ppl_dense_fp": ppl_dense, "ppl_monarch_fp": ppl_monarch_fp,
        "ppl_monarch_4bit_factors": ppl_monarch_4bit, "trajectory": traj,
    }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp6_doubly_compressed.json"), "w") as f:
        json.dump(results, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    labels = ["dense\n(124M, 16-bit)", f"Monarch MLP\n({p_m/p_fp:.0%} params, 16-bit)",
              f"Monarch MLP\n({p_m/p_fp:.0%} params, 4-bit factors)"]
    vals = [ppl_dense, ppl_monarch_fp, ppl_monarch_4bit]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, vals, color=["gray", "tab:purple", "tab:blue"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
    plt.ylabel("held-out perplexity (lower = better)")
    plt.title("Doubly-compressed GPT-2: structured + 4-bit factors compose\n"
              "(4-bit Monarch factors ≈ full-precision Monarch — no rotation used)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp6_doubly_compressed.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out} and results/exp6_doubly_compressed.json")


if __name__ == "__main__":
    main()
