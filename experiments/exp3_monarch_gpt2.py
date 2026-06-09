"""E3: Monarch structured matrices in GPT-2 — approximation + compression.

Two parts, both on real GPT-2:

(A) Does a Hadamard pre-mix help Monarch *approximate* a real GPT-2 weight matrix?
    We fit Monarch and Hadamard->Monarch to GPT-2's c_fc weight across parameter
    budgets. (Spoiler, consistent with the synthetic exp0: it does not. A fixed
    rotation doesn't expand what a structured family can reach.)

(B) Monarch as compression: replace all 12 MLP blocks with MonarchLinear
    (~halving the model), then briefly fine-tune end-to-end on Apple-Silicon MPS
    and watch held-out perplexity recover. Train text and eval text are
    different books (no leakage).

Writes figures/exp3_monarch_gpt2.png and results/exp3_monarch.json.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/exp3_monarch_gpt2.py
"""

from __future__ import annotations
from experiments._common import plt, ROOT, FIG_DIR, RES_DIR, WEIGHTS

import json
import os


import torch
import torch.nn.functional as F


from hadamard_monarch.model import GPT2Tokenizer
from hadamard_monarch.model import load_eval_ids
from hadamard_monarch.model import load_gpt2
from hadamard_monarch.compress import finetune_with_eval, swap_mlps_with_monarch
from hadamard_monarch.transforms import HadamardMix
from hadamard_monarch.transforms import MonarchLinear
from hadamard_monarch.data import count_params
from hadamard_monarch.data import load_gpt2_weight



def approximation_curve():
    """Part A: Monarch vs Hadamard->Monarch fitting a real GPT-2 c_fc weight."""
    torch.manual_seed(0)
    W = load_gpt2_weight("h.0.mlp.c_fc.weight")  # (out, in)
    out_f, in_f = W.shape
    X = torch.randn(512, in_f)
    target = X @ W.t()
    dense = out_f * in_f

    def fit(nblocks, had, steps=400):
        torch.manual_seed(0)
        mono = MonarchLinear(in_f, out_f, nblocks=nblocks, bias=False)
        hmix = HadamardMix(in_f, randomize=True) if had else None
        opt = torch.optim.Adam(mono.parameters(), lr=5e-3)
        for _ in range(steps):
            opt.zero_grad()
            F.mse_loss(mono(hmix(X) if hmix else X), target).backward()
            opt.step()
        with torch.no_grad():
            err = (mono(hmix(X) if hmix else X) - target).norm() / target.norm()
        return err.item(), mono.param_count()

    rows = []
    for nb in [8, 16, 32, 64]:
        em, p = fit(nb, False)
        eh, _ = fit(nb, True)
        rows.append((p / dense * 100, em, eh))
        print(f"  nblocks={nb:>3} params={p/dense:5.1%}  monarch={em:.4f}  hadamard+monarch={eh:.4f}")
    rows.sort()
    return rows


def compression_demo(device, nblocks, total_steps, eval_every):
    """Part B: swap MLPs for Monarch, fine-tune on MPS, track held-out perplexity."""
    import glob
    tok = GPT2Tokenizer.from_dir(os.path.join(ROOT, "models"))
    # Train on a handful of public-domain books; evaluate on held-out Alice.
    book_paths = sorted(glob.glob(os.path.join(ROOT, "models", "book_*.txt")))
    train_ids = torch.cat([load_eval_ids(tok, path=p, max_chars=2000000) for p in book_paths])
    val_ids = load_eval_ids(tok, path=os.path.join(ROOT, "models", "alice.txt"), max_chars=200000)
    print(f"  train tokens={train_ids.numel():,} (from {len(book_paths)} books)  val tokens={val_ids.numel():,}")

    model = load_gpt2(WEIGHTS, device=device)
    p_fp = count_params(model)
    ppl_fp = model.perplexity(val_ids, ctx=512)
    print(f"  fp: params={p_fp:,}  val PPL={ppl_fp:.2f}")

    swap_mlps_with_monarch(model, nblocks=nblocks, fit_init_steps=150, device=device)
    p_m = count_params(model)
    print(f"  monarch(nb={nblocks}): params={p_m:,} ({p_m/p_fp:.0%} of fp)  — fine-tuning")
    traj = finetune_with_eval(model, train_ids, val_ids, total_steps=total_steps,
                              eval_every=eval_every, ctx=256, bs=8, peak_lr=2.5e-4,
                              device=device, seed=0)
    for step, ppl in traj:
        print(f"    step {step:>4}: val PPL={ppl:.2f}")

    return {"params_fp": p_fp, "params_monarch": p_m, "ppl_fp": ppl_fp,
            "nblocks": nblocks, "trajectory": traj, "best_ppl": min(p for _, p in traj)}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    steps = int(os.environ.get("STEPS", "700"))
    eval_every = int(os.environ.get("EVAL_EVERY", "150"))
    print(f"device={device}  finetune steps={steps}")

    print("Part A — Monarch approximation of a real GPT-2 weight (does Hadamard help?):")
    approx = approximation_curve()

    print("Part B — Monarch MLP compression + local fine-tune:")
    comp = compression_demo(device, nblocks=16, total_steps=steps, eval_every=eval_every)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp3_monarch.json"), "w") as f:
        json.dump({"approx": approx, "compression": comp}, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    xs = [r[0] for r in approx]
    ax[0].plot(xs, [r[1] for r in approx], "o-", label="Monarch alone")
    ax[0].plot(xs, [r[2] for r in approx], "s-", label="Hadamard → Monarch")
    ax[0].set_xlabel("parameter budget (% of dense)")
    ax[0].set_ylabel("relative approximation error")
    ax[0].set_title("(A) Approximating GPT-2's c_fc weight\nHadamard does NOT help (as on random matrices)")
    ax[0].grid(alpha=0.3); ax[0].legend()

    steps_x = [t[0] for t in comp["trajectory"]]
    ppl_y = [t[1] for t in comp["trajectory"]]
    frac = comp["params_monarch"] / comp["params_fp"]
    ax[1].plot(steps_x, ppl_y, "o-", color="tab:purple", label=f"Monarch MLP ({frac:.0%} of params)")
    ax[1].axhline(comp["ppl_fp"], color="gray", ls="--", lw=1, label=f"full GPT-2 ({comp['ppl_fp']:.0f})")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("fine-tuning steps (Apple MPS)")
    ax[1].set_ylabel("held-out perplexity (log)")
    ax[1].set_title("(B) Monarch MLP compression: fine-tune recovers quality")
    ax[1].grid(alpha=0.3); ax[1].legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp3_monarch_gpt2.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out} and results/exp3_monarch.json")


if __name__ == "__main__":
    main()
