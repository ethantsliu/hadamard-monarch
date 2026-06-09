#!/usr/bin/env bash
# Download the GPT-2 weights, BPE tokenizer files, and eval/train texts used by
# the experiments. All public, no auth needed. ~0.5 GB total (mostly the weights).
# Files land in models/ (gitignored). Pure curl — no transformers/datasets needed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

dl() { [ -f "$2" ] || curl -fsSL "$1" -o "$2"; echo "  $2 ($(du -h "$2" | cut -f1))"; }

echo "Downloading GPT-2 assets into models/ ..."
dl "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin" models/gpt2_pytorch_model.bin
dl "https://huggingface.co/gpt2/resolve/main/vocab.json"        models/vocab.json
dl "https://huggingface.co/gpt2/resolve/main/merges.txt"        models/merges.txt
dl "https://www.gutenberg.org/files/11/11-0.txt"               models/alice.txt   # eval text (held out)
# Training corpus for the E3 fine-tune: several public-domain books.
for spec in 1342:pride 161:sense 158:emma 98:tale 84:franken 1661:holmes; do
  dl "https://www.gutenberg.org/cache/epub/${spec%%:*}/pg${spec%%:*}.txt" "models/book_${spec##*:}.txt"
done
echo "Done."
