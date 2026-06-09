#!/usr/bin/env python3
"""Run the experiments behind the blog post.

    python run.py list            # show the experiments
    python run.py exp2            # run one
    python run.py all             # run all, in order (exp7 reads exp6's output)

Everything runs on CPU or Apple Silicon (MPS); no GPU needed. All but exp0 need
the GPT-2 assets from ``bash scripts/fetch_assets.sh`` first.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# (key, module, description, needs_mps_fallback)
EXPERIMENTS = [
    ("exp0", "exp0_receptive_field",   "synthetic; no checkpoint",              False),
    ("exp1", "exp1_weight_quant",      "weight quant; needs checkpoint",        False),
    ("exp2", "exp2_perplexity",        "end-to-end perplexity (minutes)",       True),
    ("exp3", "exp3_monarch_gpt2",      "Monarch swap + fine-tune (~10 min)",    True),
    ("exp4", "exp4_structured_quant",  "Monarch-factor incoherence (minutes)",  False),
    ("exp5", "exp5_activation_aware",  "data-aware Monarch fit (minutes)",      False),
    ("exp6", "exp6_doubly_compressed", "doubly-compressed GPT-2 (minutes)",     True),
    ("exp7", "exp7_dense_control",     "dense control; run after exp6",         True),
    ("exp8", "exp8_activation_outliers","activation outliers ±Hadamard (one layer)", False),
]
BY = {k: (mod, desc, mps) for k, mod, desc, mps in EXPERIMENTS}


def run(key):
    mod, desc, mps = BY[key]
    if mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"\n=== {key}: {desc} ===", flush=True)
    importlib.import_module(f"experiments.{mod}").main()


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "list"):
        print("experiments:")
        for k, mod, desc, mps in EXPERIMENTS:
            print(f"  {k}  -  {desc}")
        print("\nusage: python run.py [exp0..exp8 | all | list]")
        return
    cmd = args[0]
    if cmd == "all":
        for k, *_ in EXPERIMENTS:
            run(k)
    elif cmd in BY:
        run(cmd)
    else:
        print(f"unknown experiment: {cmd!r} (try `python run.py list`)")
        sys.exit(1)


if __name__ == "__main__":
    main()
