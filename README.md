# Hadamard and Monarch: Compressing GPT-2 Small

A laptop-sized study of two ways to shrink a transformer's linear
layers: rotating activations with the Walsh-Hadamard transform so they survive
4-bit quantization, and replacing dense MLPs with structured Monarch matrices. Everything runs on CPU or Apple Silicon (MPS) in pure PyTorch.

Write-up: [ethantsliu.github.io/hadamard-monarch.html](https://ethantsliu.github.io/hadamard-monarch.html)

## Key results

GPT-2 small, perplexity on held-out *Alice in Wonderland*.

| quantization      | no rotation | + Hadamard |
| ----------------- | ----------: | ---------: |
| W8A8              |        29.4 |       29.2 |
| W4 (weight-only)  |        48.6 |       37.6 |
| W8A4              |        1842 |     **51** |
| W4A4              |        2152 |     **94** |


## Quick Setup (macOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_assets.sh        # GPT-2 weights, tokenizer, texts (~0.5 GB, curl)
python run.py list                  # then e.g. python run.py exp2
```

## Experiments

Each writes a figure to `figures/` and metrics to `results/`.

| command         | experiment                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| `run.py exp0`   | fits a Monarch matrix to a random dense target, with and without a Hadamard pre-rotation, across parameter budgets (synthetically) |
| `run.py exp1`   | quantizes each of GPT-2's 48 weight matrices to 4-bit, with and without a Hadamard rotation, and measures their incoherence |
| `run.py exp2`   | quantizes the full model at several weight/activation bit-widths, with and without Hadamard, and evaluates perplexity on Alice in Wonderland |
| `run.py exp3`   | replaces the MLP blocks with naively-fit Monarch matrices and fine-tunes on six Project Gutenberg books |
| `run.py exp4`   | fits Monarch to each MLP weight and compares the incoherence and 4-bit rotation benefit of the dense weight versus its factors |
| `run.py exp5`   | fits Monarch to single layers two ways, on Gaussian probes versus on the layer's real activations |
| `run.py exp6`   | replaces all MLPs with data-aware Monarch (60% of params), fine-tunes, then quantizes the factors to 4-bit |
| `run.py exp7`   | fine-tunes the full dense GPT-2 on Alice in Wonderland as a control |
| `run.py exp8`   | finds the worst activation outlier in a single layer and applies a Hadamard rotation to it |

`python run.py all` runs everything, in order.

## Structure

```
hadamard_monarch/      # library
  transforms.py        # fast Walsh-Hadamard transform and Monarch matrices
  quant.py             # round-to-nearest quantization, incoherence, rotation
  model.py             # GPT-2 small in pure PyTorch, with BPE and perplexity eval
  compress.py          # Monarch fitting and local fine-tuning
  data.py              # Project Gutenberg loaders
experiments/           # exp0 to exp8, one file each
run.py                 # dispatcher
tests/                 # pytest: transform orthogonality, Monarch shapes, quantization
```
