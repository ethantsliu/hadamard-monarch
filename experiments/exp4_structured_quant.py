"""E4: incoherence explains who needs the Hadamard rotation — dense vs Monarch.

A single thread ties together the project's two halves (Hadamard quantization and
Monarch structure): **incoherence** (max |entry| / RMS entry, an outlier metric)
predicts whether a Hadamard rotation helps low-bit quantization.

  - GPT-2's dense FFN weights are outlier-heavy: incoherence ~10-160. Rotating
    them into the Hadamard (incoherent) basis before 4-bit quantization spreads
    the outliers and cuts the error by ~1.1-1.6x.
  - Fit a Monarch layer to one of those weights and its *factors* (w1, w2) come
    out low-incoherence (~8) regardless of how outlier-heavy the dense matrix
    was. With no outliers to spread, a per-block Hadamard rotation buys nothing
    when quantizing the factors (benefit ~1.0x).

So the rotation is not magic — it is an outlier remedy, and structured factors
simply don't have the disease. We fit Monarch by naive weight-error matching on
Gaussian probes (it can only approximate these matrices), but the *incoherence*
of the resulting factors is the quantity of interest and is robust.

Three panels (figures/exp4_structured_quant.png):
  (a) incoherence(dense)  vs  incoherence(Monarch factors), one point/matrix
      — factors hug a low flat line (~8) no matter the dense incoherence.
  (b) 4-bit "rotation benefit" (err_baseline / err_hadamard) for DENSE (>1) vs
      MONARCH FACTORS (~1), per matrix — the rotation helps one, not the other.
  (c) rotation benefit vs dense incoherence for both — the dense benefit grows
      with incoherence; the factor benefit stays flat at ~1.

CPU-only (a GPU job owns MPS). Needs the GPT-2 checkpoint (see README).
Run: python experiments/exp4_structured_quant.py
"""

from __future__ import annotations
from experiments._common import plt, FIG_DIR, RES_DIR, WEIGHTS

import json
import os


import numpy as np
import torch
import torch.nn.functional as F


from hadamard_monarch.quant import factor_incoherence, quantized_dense
from hadamard_monarch.transforms import MonarchLinear
from hadamard_monarch.quant import incoherence, quantize, quantize_with_hadamard, rel_error


DEVICE = "cpu"  # a GPU job owns MPS; stay on CPU.
NBLOCKS = 16
BITS = 4
N_PROBES = 1536   # must exceed in_features (768) for a well-determined fit
FIT_STEPS = int(os.environ.get("FIT_STEPS", "600"))  # factor incoherence plateaus ~here
FIT_LR = 3e-3

# Representative MLP matrices spanning a range of incoherence. Both c_fc (768->3072)
# and c_proj (3072->768) per layer; together they cover incoherence ~10-160.
LAYERS = [0, 2, 4, 6, 8, 11]
SUBS = ["mlp.c_fc", "mlp.c_proj"]


def fit_monarch(W: torch.Tensor) -> MonarchLinear:
    """Naive weight-error fit of a MonarchLinear to ``W`` (out, in) on Gaussian probes."""
    out_f, in_f = W.shape
    torch.manual_seed(0)
    mono = MonarchLinear(in_f, out_f, nblocks=NBLOCKS, bias=False).to(DEVICE)
    X = torch.randn(N_PROBES, in_f, device=DEVICE)
    target = X @ W.t()
    opt = torch.optim.Adam(mono.parameters(), lr=FIT_LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        F.mse_loss(mono(X), target).backward()
        opt.step()
    return mono


def main():
    if not os.path.exists(WEIGHTS):
        print("GPT-2 checkpoint missing; see README for the download command.")
        return
    print(f"device={DEVICE}  nblocks={NBLOCKS}  bits={BITS}  probes={N_PROBES}  fit_steps={FIT_STEPS}")
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)

    rows = []
    for i in LAYERS:
        for sub in SUBS:
            name = f"h{i}.{sub}"
            W = sd[f"h.{i}.{sub}.weight"].t().contiguous().float()  # (in,out) -> (out,in)

            mono = fit_monarch(W)
            with torch.no_grad():
                dense_inc = incoherence(W)
                fac_inc = factor_incoherence(mono)
                fit_err = rel_error(mono.to_dense(), W)

                # 4-bit quant of the DENSE weight, with / without Hadamard.
                dense_base = rel_error(quantize(W, BITS), W)
                dense_had = rel_error(quantize_with_hadamard(W, BITS), W)

                # 4-bit quant of the MONARCH FACTORS, with / without per-block Hadamard.
                # Error is measured against the (unquantized) Monarch dense-equivalent,
                # isolating the cost of quantizing the factors from the fit error.
                mono_dense = mono.to_dense()
                fac_base = rel_error(quantized_dense(mono, BITS, hadamard=False), mono_dense)
                fac_had = rel_error(quantized_dense(mono, BITS, hadamard=True), mono_dense)

            row = {
                "name": name,
                "shape": list(W.shape),
                "dense_incoherence": dense_inc,
                "factor_incoherence": fac_inc,
                "monarch_fit_rel_error": fit_err,
                "dense_q_baseline": dense_base,
                "dense_q_hadamard": dense_had,
                "dense_rotation_benefit": dense_base / dense_had,
                "factor_q_baseline": fac_base,
                "factor_q_hadamard": fac_had,
                "factor_rotation_benefit": fac_base / fac_had,
            }
            rows.append(row)
            print(f"  {name:16s} dense_inc={dense_inc:5.1f} fac_inc={fac_inc:5.2f} "
                  f"| dense benefit={row['dense_rotation_benefit']:.2f}x "
                  f"factor benefit={row['factor_rotation_benefit']:.2f}x")

    # ---- summary table -------------------------------------------------------
    d_inc = np.array([r["dense_incoherence"] for r in rows])
    f_inc = np.array([r["factor_incoherence"] for r in rows])
    d_ben = np.array([r["dense_rotation_benefit"] for r in rows])
    f_ben = np.array([r["factor_rotation_benefit"] for r in rows])

    print("\n" + "=" * 78)
    print(f"{'matrix':16s} {'shape':>13s} {'dense_inc':>9s} {'fac_inc':>8s} "
          f"{'dense_ben':>9s} {'fac_ben':>8s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:16s} {str(tuple(r['shape'])):>13s} "
              f"{r['dense_incoherence']:9.1f} {r['factor_incoherence']:8.2f} "
              f"{r['dense_rotation_benefit']:8.2f}x {r['factor_rotation_benefit']:7.2f}x")
    print("-" * 78)
    print(f"dense incoherence:   min={d_inc.min():.1f}  median={np.median(d_inc):.1f}  max={d_inc.max():.1f}")
    print(f"factor incoherence:  min={f_inc.min():.1f}  median={np.median(f_inc):.1f}  max={f_inc.max():.1f}")
    print(f"DENSE  rotation benefit:  median={np.median(d_ben):.2f}x  (>1 => Hadamard helps)")
    print(f"FACTOR rotation benefit:  median={np.median(f_ben):.2f}x  (~1 => Hadamard does nothing)")
    print("=" * 78)

    summary = {
        "config": {"nblocks": NBLOCKS, "bits": BITS, "n_probes": N_PROBES,
                   "fit_steps": FIT_STEPS, "fit_lr": FIT_LR, "device": DEVICE},
        "per_matrix": rows,
        "summary": {
            "dense_incoherence": {"min": float(d_inc.min()), "median": float(np.median(d_inc)),
                                   "max": float(d_inc.max())},
            "factor_incoherence": {"min": float(f_inc.min()), "median": float(np.median(f_inc)),
                                    "max": float(f_inc.max())},
            "dense_rotation_benefit_median": float(np.median(d_ben)),
            "factor_rotation_benefit_median": float(np.median(f_ben)),
        },
    }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp4_structured_quant.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure --------------------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (a) dense incoherence vs factor incoherence.
    ax[0].scatter(d_inc, f_inc, s=55, alpha=0.8, color="tab:blue", zorder=3)
    fmed = float(np.median(f_inc))
    ax[0].axhline(fmed, color="gray", ls="--", lw=1.2,
                  label=f"factor median ≈ {fmed:.1f}")
    # diagonal y=x for reference (factors fall far below it).
    lo, hi = 0, max(d_inc.max(), f_inc.max()) * 1.05
    ax[0].plot([lo, hi], [lo, hi], color="lightgray", ls=":", lw=1, label="y = x")
    ax[0].set_xlim(0, d_inc.max() * 1.08)
    ax[0].set_ylim(0, max(d_inc.max() * 1.08, fmed * 2))
    ax[0].set_xlabel("incoherence of DENSE weight (max/RMS)")
    ax[0].set_ylabel("incoherence of MONARCH factors")
    ax[0].set_title("(a) Monarch factors stay low-incoherence\nregardless of the dense weight's outliers")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper left")

    # (b) per-matrix rotation benefit, dense vs factors.
    order = np.argsort(d_inc)
    x = np.arange(len(rows))
    w = 0.4
    ax[1].bar(x - w / 2, d_ben[order], w, color="tab:red", label="DENSE weight")
    ax[1].bar(x + w / 2, f_ben[order], w, color="tab:green", label="Monarch factors")
    ax[1].axhline(1.0, color="gray", ls="--", lw=1.2, label="no benefit (1.0×)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([rows[i]["name"] for i in order], rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("4-bit rotation benefit  (err_base / err_Hadamard)")
    ax[1].set_title("(b) Hadamard helps the dense weight, not the factors\n(sorted by dense incoherence)")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend()

    # (c) rotation benefit vs dense incoherence, both series.
    ax[2].scatter(d_inc, d_ben, s=55, alpha=0.85, color="tab:red", label="DENSE weight", zorder=3)
    ax[2].scatter(d_inc, f_ben, s=55, alpha=0.85, color="tab:green", label="Monarch factors", zorder=3)
    # trend line for the dense series (benefit grows with incoherence).
    if len(d_inc) >= 2:
        m, b = np.polyfit(d_inc, d_ben, 1)
        xs = np.linspace(d_inc.min(), d_inc.max(), 50)
        ax[2].plot(xs, m * xs + b, color="tab:red", ls="-", lw=1, alpha=0.5)
    ax[2].axhline(1.0, color="gray", ls="--", lw=1.2)
    ax[2].set_xlabel("incoherence of DENSE weight (max/RMS)")
    ax[2].set_ylabel("4-bit rotation benefit (×)")
    ax[2].set_title("(c) Rotation benefit tracks incoherence\nhigh-incoherence dense gains; flat factors don't")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    fig.suptitle("Incoherence explains who needs the Hadamard rotation: "
                 "dense GPT-2 FFN weights do, their Monarch factors don't",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp4_structured_quant.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out} and results/exp4_structured_quant.json")


if __name__ == "__main__":
    main()
