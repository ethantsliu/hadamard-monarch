"""E1: quantizing real GPT-2 weight matrices — incoherence, bit-width, granularity.

Operates directly on GPT-2's actual weight matrices (loaded from the published
checkpoint) and measures weight-reconstruction error under quantization, with
and without a block-Hadamard rotation. Four panels:

  (a) entry-magnitude histogram of a real layer, raw vs Hadamard-rotated
      (the mechanism: outliers spread, incoherence collapses)
  (b) relative quant error vs bit-width, baseline vs Hadamard
  (c) per-layer 4-bit error reduction vs that layer's incoherence (all 48 matrices)
  (d) granularity ablation (E5): per-tensor / per-channel / group-128, ±Hadamard
      — showing the rotation and finer granularity are two ways to fight outliers

Writes figures/exp1_weight_quant.png. Needs the GPT-2 checkpoint (see README).
"""

from __future__ import annotations
from experiments._common import plt, FIG_DIR, RES_DIR, WEIGHTS

import json
import os


import numpy as np
import torch


from hadamard_monarch.quant import incoherence, quantize, quantize_with_hadamard, hadamard_rotate, rel_error



def load_all_weights():
    """Return {name: (out,in) weight} for every attn/MLP projection across layers."""
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    mats = {}
    for i in range(12):
        for sub in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
            key = f"h.{i}.{sub}.weight"
            mats[f"h{i}.{sub}"] = sd[key].t().contiguous().float()  # (in,out)->(out,in)
    return mats


def main():
    if not os.path.exists(WEIGHTS):
        print("GPT-2 checkpoint missing; see README for the curl command.")
        return
    mats = load_all_weights()
    bits_list = [8, 6, 4, 3]

    # Representative layer for panels (a)/(b): the heaviest-outlier MLP output proj.
    rep_name = "h0.mlp.c_proj"
    Wrep = mats[rep_name]
    Wrot = hadamard_rotate(Wrep)
    print(f"{rep_name}: incoherence {incoherence(Wrep):.1f} -> {incoherence(Wrot):.1f} after Hadamard")

    err_base = [rel_error(quantize(Wrep, b), Wrep) for b in bits_list]
    err_had = [rel_error(quantize_with_hadamard(Wrep, b), Wrep) for b in bits_list]

    # (c) per-layer 4-bit improvement vs incoherence, over all 48 matrices.
    incs, imps = [], []
    for W in mats.values():
        eb = rel_error(quantize(W, 4), W)
        eh = rel_error(quantize_with_hadamard(W, 4), W)
        incs.append(incoherence(W))
        imps.append(eb / eh)
    print(f"4-bit error reduction across {len(mats)} matrices: "
          f"min={min(imps):.2f}x median={np.median(imps):.2f}x max={max(imps):.2f}x")

    # (d) granularity ablation at 4-bit on the representative layer.
    gran = [
        ("per-tensor", dict(per_row=False)),
        ("per-channel", dict(per_row=True)),
        ("group-128", dict(group_size=128)),
    ]
    g_base, g_had = [], []
    for _, kw in gran:
        g_base.append(rel_error(quantize(Wrep, 4, **kw), Wrep))
        g_had.append(rel_error(quantize_with_hadamard(Wrep, 4, **kw), Wrep))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) histogram
    ax[0, 0].hist(Wrep.abs().flatten().numpy(), bins=80, alpha=0.6, label="raw", log=True)
    ax[0, 0].hist(Wrot.abs().flatten().numpy(), bins=80, alpha=0.6, label="Hadamard-rotated", log=True)
    ax[0, 0].set_title(f"(a) {rep_name} entry magnitudes\nincoherence {incoherence(Wrep):.0f} → {incoherence(Wrot):.0f}")
    ax[0, 0].set_xlabel("|weight|"); ax[0, 0].set_ylabel("count (log)"); ax[0, 0].legend()

    # (b) error vs bits
    ax[0, 1].plot(bits_list, err_base, "o-", label="baseline")
    ax[0, 1].plot(bits_list, err_had, "s-", label="Hadamard")
    ax[0, 1].invert_xaxis()
    ax[0, 1].set_title(f"(b) {rep_name}: quant error vs bits")
    ax[0, 1].set_xlabel("bits"); ax[0, 1].set_ylabel("relative error"); ax[0, 1].grid(alpha=0.3); ax[0, 1].legend()

    # (c) scatter
    ax[1, 0].scatter(incs, imps, s=40, alpha=0.7, color="tab:green")
    ax[1, 0].axhline(1.0, color="gray", ls="--", lw=1)
    ax[1, 0].set_title("(c) 4-bit error reduction vs incoherence\n(all 48 GPT-2 matrices)")
    ax[1, 0].set_xlabel("layer incoherence (max/RMS)"); ax[1, 0].set_ylabel("error reduction (×)"); ax[1, 0].grid(alpha=0.3)

    # (d) granularity bars
    names = [g[0] for g in gran]
    x = np.arange(len(names)); w = 0.38
    ax[1, 1].bar(x - w / 2, g_base, w, label="baseline")
    ax[1, 1].bar(x + w / 2, g_had, w, label="Hadamard")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(names)
    ax[1, 1].set_title(f"(d) {rep_name}: granularity vs Hadamard (4-bit)")
    ax[1, 1].set_ylabel("relative error"); ax[1, 1].legend(); ax[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp1_weight_quant.png")
    plt.savefig(out, dpi=130)

    os.makedirs(RES_DIR, exist_ok=True)
    results = {
        "rep_layer": rep_name,
        "incoherence_raw": incoherence(Wrep),
        "incoherence_rotated": incoherence(Wrot),
        "bits": bits_list,
        "rep_error_baseline": err_base,
        "rep_error_hadamard": err_had,
        "per_layer_incoherence": incs,
        "per_layer_4bit_reduction": imps,
        "granularity_4bit": {g[0]: {"baseline": b, "hadamard": h}
                             for g, b, h in zip(gran, g_base, g_had)},
    }
    with open(os.path.join(RES_DIR, "exp1_weight_quant.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out} and results/exp1_weight_quant.json")


if __name__ == "__main__":
    main()
