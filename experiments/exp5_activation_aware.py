"""E5: data-aware (activation-aware) fitting of Monarch FFNs, and it stays quantizable.

Honest framing / prior art
--------------------------
Data-aware (activation-aware) low-rank / structured compression of LLM weights is
well established — DRONE (NeurIPS 2021), ASVD, SVD-LLM, FWSVD (ICLR 2022) — and
ProcrustesGPT (ACL Findings 2025) already does data-aware fitting of a structured
class that GENERALIZES Monarch. We are not claiming the principle. Our narrow
slice: data-aware fitting of an EXACT Monarch factorization to a real GPT-2 FFN,
plus a demonstration that the data-aware Monarch then COMPOSES with 4-bit factor
quantization (tying to E4 / monarch_quant.py: Monarch factors are low-incoherence
and quantize well WITHOUT a Hadamard rotation).

What this experiment shows
--------------------------
For ~3 GPT-2 layers (blocks 0, 5, 11) we fit a ``MonarchMLP`` (d_model=768,
d_ff=3072, nblocks=16) to the dense MLP block (c_fc -> gelu_new -> c_proj) TWO
ways at the SAME parameter budget and SAME steps/lr:

  (a) naive     -- fit on Gaussian probes whose per-coordinate RMS matches the
                   real activation scale (weight-error fitting; every input
                   direction weighted equally).
  (b) data-aware-- fit on the layer's REAL training activations (input to
                   block.mlp), captured by a forward hook over real text.

Train and eval activations come from DISJOINT halves of the token stream (no
leakage). We then report, on HELD-OUT real activations, the relative output error
``||M(x) - MLP(x)|| / ||MLP(x)||`` for naive vs data-aware. Data-aware should win
clearly. Finally we quantize the data-aware Monarch's factors to 4-bit (NO
Hadamard) and report the (small) extra held-out output error: structured +
quantized composes.

CPU-only (a GPU job may own MPS). Needs the GPT-2 checkpoint (see README).
Run: python experiments/exp5_activation_aware.py
"""

from __future__ import annotations
from experiments._common import plt, ROOT, FIG_DIR, RES_DIR, WEIGHTS

import json
import os


import numpy as np
import torch


from hadamard_monarch.compress import (
    collect_activations,
    fit_monarch,
    output_rel_error,
    quantize_monarch_factors,
)
from hadamard_monarch.model import GPT2Tokenizer
from hadamard_monarch.model import load_eval_ids
from hadamard_monarch.model import load_gpt2
from hadamard_monarch.quant import factor_incoherence
from hadamard_monarch.transforms import MonarchLinear


DEVICE = "cpu"  # a GPU job may own MPS; stay on CPU per the task.
LAYERS = [0, 5, 11]
NBLOCKS = 16
D_MODEL = 768
D_FF = 3072
BITS = 4
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "4096"))  # per split
CTX = int(os.environ.get("CTX", "256"))
FIT_STEPS = int(os.environ.get("FIT_STEPS", "600"))
FIT_LR = float(os.environ.get("FIT_LR", "3e-3"))
SEED = 0


def mean_factor_incoherence(mono) -> float:
    """Max factor incoherence over every MonarchLinear inside ``mono`` (a MonarchMLP)."""
    incs = [factor_incoherence(s) for s in mono.modules() if isinstance(s, MonarchLinear)]
    return float(max(incs)) if incs else float("nan")


def gaussian_probes(n: int, in_f: int, rms: float, seed: int) -> torch.Tensor:
    """N(0, rms^2) probes — Gaussian inputs scaled to match the real activation RMS.

    This is the "naive" weight-error fit: it reproduces the dense map on isotropic
    Gaussian inputs (every direction weighted equally), at the same input energy as
    the real activations so the two fits see the same target magnitudes.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, in_f, generator=g) * rms


def run_layer(model, layer_idx: int, train_ids: torch.Tensor, eval_ids: torch.Tensor) -> dict:
    """Fit naive + data-aware MonarchMLP to one block's MLP; eval on held-out acts."""
    mlp = model.h[layer_idx].mlp

    # --- collect DISJOINT train / held-out real MLP-input activations -------------
    X_train = collect_activations(model, train_ids, mlp, max_samples=MAX_SAMPLES, ctx=CTX)
    X_eval = collect_activations(model, eval_ids, mlp, max_samples=MAX_SAMPLES, ctx=CTX)
    act_rms = X_train.pow(2).mean().sqrt().item()

    # --- (a) naive: fit on Gaussian probes matched to the activation RMS ----------
    X_gauss = gaussian_probes(X_train.size(0), D_MODEL, act_rms, seed=SEED)
    mono_naive = fit_monarch(mlp, X_gauss, steps=FIT_STEPS, lr=FIT_LR,
                             nblocks=NBLOCKS, device=DEVICE, seed=SEED)

    # --- (b) data-aware: fit on real training activations -------------------------
    mono_data = fit_monarch(mlp, X_train, steps=FIT_STEPS, lr=FIT_LR,
                            nblocks=NBLOCKS, device=DEVICE, seed=SEED)

    # --- held-out real-activation output errors -----------------------------------
    err_naive = output_rel_error(mono_naive, mlp, X_eval)
    err_data = output_rel_error(mono_data, mlp, X_eval)

    # --- quantize the DATA-AWARE Monarch's factors to 4-bit (no Hadamard) ---------
    mono_data_q = quantize_monarch_factors(mono_data, bits=BITS, hadamard=False)
    err_data_q = output_rel_error(mono_data_q, mlp, X_eval)

    fac_inc = mean_factor_incoherence(mono_data)

    row = {
        "layer": layer_idx,
        "train_samples": int(X_train.size(0)),
        "eval_samples": int(X_eval.size(0)),
        "activation_rms": act_rms,
        "held_out_err_naive_fp": err_naive,
        "held_out_err_data_aware_fp": err_data,
        "held_out_err_data_aware_4bit": err_data_q,
        "data_aware_improvement_x": err_naive / err_data,
        "quant_extra_err_abs": err_data_q - err_data,
        "quant_err_ratio": err_data_q / err_data,
        "factor_incoherence": fac_inc,
    }
    print(f"  block {layer_idx:>2}: act_rms={act_rms:.3f}  "
          f"held-out err  naive={err_naive:.4f}  data-aware={err_data:.4f}  "
          f"({row['data_aware_improvement_x']:.2f}x better)  "
          f"data-aware+4bit={err_data_q:.4f}  fac_inc={fac_inc:.2f}")
    return row


def main():
    if not os.path.exists(WEIGHTS):
        print("GPT-2 checkpoint missing; see README for the download command.")
        return
    torch.manual_seed(SEED)
    print(f"device={DEVICE}  layers={LAYERS}  nblocks={NBLOCKS}  bits={BITS}  "
          f"max_samples/split={MAX_SAMPLES}  ctx={CTX}  fit_steps={FIT_STEPS}  lr={FIT_LR}")

    tok = GPT2Tokenizer.from_dir(os.path.join(ROOT, "models"))
    model = load_gpt2(WEIGHTS, device=DEVICE)

    # Split the eval text into DISJOINT train / eval token halves (no leakage):
    # fitting activations come from the first half, held-out from the second.
    ids = load_eval_ids(tok, path=os.path.join(ROOT, "models", "alice.txt"),
                        max_chars=int(os.environ.get("MAX_CHARS", "120000")))
    half = ids.numel() // 2
    train_ids, eval_ids = ids[:half], ids[half:]
    print(f"tokens: total={ids.numel():,}  train_split={train_ids.numel():,}  "
          f"eval_split={eval_ids.numel():,}  (disjoint)\n")

    rows = [run_layer(model, li, train_ids, eval_ids) for li in LAYERS]

    # ---- summary -----------------------------------------------------------------
    naive = np.array([r["held_out_err_naive_fp"] for r in rows])
    data = np.array([r["held_out_err_data_aware_fp"] for r in rows])
    dataq = np.array([r["held_out_err_data_aware_4bit"] for r in rows])
    impr = np.array([r["data_aware_improvement_x"] for r in rows])
    fincs = np.array([r["factor_incoherence"] for r in rows])

    print("\n" + "=" * 88)
    print("HELD-OUT real-activation output error  ||M(x)-MLP(x)|| / ||MLP(x)||")
    print(f"{'block':>5} {'naive(fp)':>11} {'data-aware(fp)':>15} {'data-aware(4b)':>15} "
          f"{'gain':>7} {'4b_extra':>9} {'fac_inc':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['layer']:>5} {r['held_out_err_naive_fp']:>11.4f} "
              f"{r['held_out_err_data_aware_fp']:>15.4f} "
              f"{r['held_out_err_data_aware_4bit']:>15.4f} "
              f"{r['data_aware_improvement_x']:>6.2f}x "
              f"{r['quant_extra_err_abs']:>+9.4f} {r['factor_incoherence']:>8.2f}")
    print("-" * 88)
    print(f"mean held-out err:  naive={naive.mean():.4f}  data-aware={data.mean():.4f}  "
          f"data-aware+4bit={dataq.mean():.4f}")
    print(f"data-aware improvement over naive: median={np.median(impr):.2f}x  "
          f"(>1 => data-aware wins)")
    print(f"4-bit factors extra error: mean={np.mean(dataq - data):.4f} abs  "
          f"(median ratio {np.median(dataq / data):.3f}x of fp error)")
    print(f"data-aware factor incoherence: median={np.median(fincs):.2f}  "
          f"(low => quantizes well without Hadamard, cf. E4)")
    print("=" * 88)

    summary = {
        "config": {"layers": LAYERS, "nblocks": NBLOCKS, "d_model": D_MODEL, "d_ff": D_FF,
                   "bits": BITS, "max_samples_per_split": MAX_SAMPLES, "ctx": CTX,
                   "fit_steps": FIT_STEPS, "fit_lr": FIT_LR, "device": DEVICE,
                   "splits": "disjoint train/eval halves of alice.txt"},
        "prior_art": ("Data-aware low-rank/structured compression is established "
                      "(DRONE NeurIPS'21, ASVD, SVD-LLM, FWSVD ICLR'22; "
                      "ProcrustesGPT ACL-Findings'25 fits a Monarch-generalizing "
                      "class data-aware). Our slice: data-aware fitting of EXACT "
                      "Monarch + showing it composes with 4-bit factor quantization."),
        "per_layer": rows,
        "summary": {
            "mean_err_naive_fp": float(naive.mean()),
            "mean_err_data_aware_fp": float(data.mean()),
            "mean_err_data_aware_4bit": float(dataq.mean()),
            "data_aware_improvement_median_x": float(np.median(impr)),
            "quant_extra_err_mean_abs": float(np.mean(dataq - data)),
            "quant_err_ratio_median": float(np.median(dataq / data)),
            "factor_incoherence_median": float(np.median(fincs)),
        },
    }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp5_activation_aware.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: per-layer grouped bars ------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    labels = [f"block {r['layer']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w, naive, w, color="tab:red", label="naive (Gaussian-probe fit), fp")
    b2 = ax.bar(x, data, w, color="tab:green", label="data-aware (real acts), fp")
    b3 = ax.bar(x + w, dataq, w, color="tab:blue", label="data-aware, 4-bit factors")

    for bars in (b1, b2, b3):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("held-out relative output error  ||M(x) − MLP(x)|| / ||MLP(x)||")
    ax.set_title("Data-aware fitting of an exact Monarch FFN beats Gaussian-probe fitting\n"
                 "on HELD-OUT real activations — and the factors still quantize to 4-bit cleanly")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(naive.max(), data.max(), dataq.max()) * 1.18)

    note = (f"4-bit factors add ≈{np.mean(dataq - data):+.4f} abs error "
            f"(×{np.median(dataq / data):.2f} of fp)\n"
            f"data-aware factor incoherence ≈ {np.median(fincs):.1f} "
            f"(low → no Hadamard needed, cf. E4)")
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp5_activation_aware.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out} and results/exp5_activation_aware.json")


if __name__ == "__main__":
    main()
