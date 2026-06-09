"""E8: activation outliers and the Hadamard rotation, on a real GPT-2 layer.

Run a forward pass, find the MLP-input activation row with the worst outlier
(highest incoherence = max / RMS), and show what a block-Hadamard rotation does
to it: the spike spreads across all coordinates and the incoherence collapses.

This is the activation-side counterpart to the weight analysis in exp1, and the
mechanism behind why rotating activations rescues 4-bit quantization (exp2).

Writes figures/exp8_activation_outliers.png and results/exp8_activation_outliers.json.
"""
import json
import os

import torch

from experiments._common import plt, MODELS, FIG_DIR, RES_DIR, WEIGHTS
from hadamard_monarch.model import GPT2Tokenizer, load_eval_ids, load_gpt2
from hadamard_monarch.quant import block_fwht, incoherence, _largest_pow2_divisor


def main():
    if not os.path.exists(WEIGHTS):
        print("GPT-2 checkpoint missing; see README for the fetch command.")
        return

    model = load_gpt2(WEIGHTS, device="cpu")
    tok = GPT2Tokenizer.from_dir(MODELS)
    ids = load_eval_ids(tok, max_chars=int(os.environ.get("MAX_CHARS", "8000")))
    ids = ids[: int(os.environ.get("CTX", "256"))].unsqueeze(0)

    # Capture the input activations to every MLP down-projection (the post-GELU
    # FFN hidden state, where GPT-2's activation outliers live).
    captured = {}
    handles = []
    for i, blk in enumerate(model.h):
        def hook(mod, inp, i=i):
            captured[i] = inp[0].detach()[0]            # (tokens, d)
        handles.append(blk.mlp.c_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    # The single (layer, token) activation row with the worst outlier.
    best = None
    for layer, act in captured.items():
        for t in range(act.size(0)):
            ic = incoherence(act[t])
            if best is None or ic > best[0]:
                best = (ic, layer, t)
    ic_before, layer, t = best
    row = captured[layer][t]
    block = _largest_pow2_divisor(row.numel())
    rot = block_fwht(row, block)
    ic_after = incoherence(rot)

    print(f"worst outlier: layer {layer}, token {t}, dim {row.numel()}, block {block}")
    print(f"incoherence (max/RMS): {ic_before:.1f} -> {ic_after:.1f} after Hadamard")
    print(f"max |x|: {row.abs().max():.2f} -> {rot.abs().max():.2f}")

    before, after = row.abs().numpy(), rot.abs().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    ax[0].fill_between(range(len(before)), before, lw=0.4)
    ax[0].set_title(f"layer {layer} MLP activations — one channel spikes\nincoherence (max/RMS) = {ic_before:.0f}")
    ax[1].fill_between(range(len(after)), after, lw=0.4)
    ax[1].set_title(f"after the Hadamard rotation — energy spread out\nincoherence (max/RMS) = {ic_after:.1f}")
    for a in ax:
        a.set_xlabel("channel")
    ax[0].set_ylabel("|activation|")
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "exp8_activation_outliers.png")
    plt.savefig(out, dpi=130)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp8_activation_outliers.json"), "w") as f:
        json.dump(
            {
                "layer": layer, "token": t, "dim": row.numel(), "block": block,
                "incoherence_before": ic_before, "incoherence_after": ic_after,
                "max_before": float(row.abs().max()), "max_after": float(rot.abs().max()),
            },
            f, indent=2,
        )
    print(f"\nsaved {out} and results/exp8_activation_outliers.json")


if __name__ == "__main__":
    main()
