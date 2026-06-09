"""GPT-2 in pure PyTorch: byte-level BPE tokenizer, model, eval text, and QuaRot-style quantization."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import regex as re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import block_fwht, quantize, _largest_pow2_divisor


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's tanh-approximation GELU."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


@dataclass
class GPT2Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    n_positions: int = 1024
    layer_norm_eps: float = 1e-5


class Conv1D(nn.Module):
    """HF-style Conv1D: weight is (in_features, out_features); forward is x @ W + b.

    Supports optional QuaRot-style quantization: a block-Hadamard rotation of the
    contraction (input) dimension is applied to the activation online and folded
    into the weight, so the rotations cancel in the matmul `(xHᵀ)(HW) = xW` while
    *both* the rotated weight and rotated activation are quantized in the
    incoherent basis. Set via ``set_quant``; cleared via ``reset_quant``.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._quant_on = False

    @torch.no_grad()
    def set_quant(self, wbits=None, abits=None, hadamard=False, w_group=None):
        """Configure quantization. wbits/abits=None means that side stays full precision."""
        self._abits = abits
        self._hadamard = hadamard
        self._blk = _largest_pow2_divisor(self.weight.shape[0]) if hadamard else None
        # Rotate the input dim of W (fold in the Hadamard), then quantize per output channel.
        Wr = block_fwht(self.weight.data.t().contiguous(), self._blk).t() if hadamard else self.weight.data
        if wbits is not None:
            Wq = quantize(Wr.t().contiguous(), wbits, per_row=(w_group is None), group_size=w_group).t()
        else:
            Wq = Wr
        self._Wq = Wq.contiguous()
        self._quant_on = True

    def reset_quant(self):
        self._quant_on = False
        self._Wq = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quant_on:
            x2 = x.reshape(-1, x.size(-1))
            if self._hadamard:
                x2 = block_fwht(x2, self._blk)            # rotate activation feature dim
            if self._abits is not None:
                x2 = quantize(x2, self._abits, per_row=True)  # per-token activation quant
            y = torch.addmm(self.bias, x2, self._Wq)
            return y.view(*x.shape[:-1], self._Wq.size(1))
        return torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight).view(
            *x.shape[:-1], self.weight.size(1)
        )


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = Conv1D(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        dh = C // self.n_head
        q = q.view(B, T, self.n_head, dh).transpose(1, 2)
        k = k.view(B, T, self.n_head, dh).transpose(1, 2)
        v = v.view(B, T, self.n_head, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = Conv1D(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = Conv1D(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config = GPT2Config()):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        # LM head is tied to wte; no separate parameter.

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)  # weight-tied head, no bias
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def perplexity(self, ids: torch.Tensor, ctx: int = 512, stride: int | None = None) -> float:
        """Perplexity over a 1-D token tensor via a sliding window.

        Uses a strided window so each token is scored with real left-context.
        Returns exp(mean token NLL).
        """
        self.eval()
        device = self.wte.weight.device
        stride = stride or ctx
        nll_sum, n_tokens = 0.0, 0
        prev_end = 0
        for begin in range(0, ids.size(0), stride):
            end = min(begin + ctx, ids.size(0))
            if end - begin < 2:
                break
            window = ids[begin:end].to(device)[None, :]
            target = window.clone()
            # Only score tokens not already scored in the previous window.
            n_new = end - prev_end
            target[:, :-n_new] = -100
            logits, loss = self(window[:, :-1], target[:, 1:])
            # loss is mean over scored tokens; recover the sum.
            scored = (target[:, 1:] != -100).sum().item()
            nll_sum += loss.item() * scored
            n_tokens += scored
            prev_end = end
            if end == ids.size(0):
                break
        return math.exp(nll_sum / max(1, n_tokens))


def load_gpt2(weights_path: str, device: str = "cpu", cfg: GPT2Config = GPT2Config()) -> GPT2:
    """Build GPT-2 and load the published state_dict (import torch BEFORE this).

    The checkpoint has flat keys (wte/wpe/h.N.../ln_f) and per-block causal-mask
    buffers `h.N.attn.bias` that we don't use (SDPA handles masking), so we load
    with strict=False and assert nothing important is missing.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"GPT-2 weights not found at {weights_path}. Download with:\n"
            f"  mkdir -p models && curl -fsSL "
            f"https://huggingface.co/gpt2/resolve/main/pytorch_model.bin -o {weights_path}"
        )
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = GPT2(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `missing` should be empty; `unexpected` should be only the attn.bias mask buffers.
    real_missing = [k for k in missing if "attn.bias" not in k]
    assert not real_missing, f"missing weights: {real_missing[:8]}"
    assert all("attn.bias" in k for k in unexpected), f"unexpected: {unexpected[:8]}"
    return model.to(device).eval()


def configure_quant_(
    model: GPT2,
    wbits: int | None = 4,
    abits: int | None = None,
    hadamard: bool = False,
    w_group: int | None = None,
) -> int:
    """Apply quantization config to every Conv1D in place. Returns #matrices touched."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, Conv1D):
            mod.set_quant(wbits=wbits, abits=abits, hadamard=hadamard, w_group=w_group)
            n += 1
    return n


def reset_quant_(model: GPT2) -> None:
    """Restore full-precision forward (clear all quantization)."""
    for mod in model.modules():
        if isinstance(mod, Conv1D):
            mod.reset_quant()


# GPT-2 pre-tokenization pattern: contractions, words, numbers, punctuation, runs of space.
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache()
def bytes_to_unicode():
    """Reversible map from the 256 byte values to printable unicode chars.

    Avoids control/whitespace bytes that would break the regex or round-trip.
    Byte 32 (space) maps to 'Ġ' (U+0120), the familiar GPT-2 space marker.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def get_pairs(word):
    """Set of adjacent symbol pairs in a tuple-of-symbols `word`."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GPT2Tokenizer:
    """Encode text <-> GPT-2 token ids. `decode(encode(s)) == s` for any string."""

    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, encoding="utf-8") as f:
            merges = f.read().split("\n")[1:-1]  # drop "#version" header + trailing blank
        merges = [tuple(m.split()) for m in merges]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache = {}

    @classmethod
    def from_dir(cls, dirpath: str) -> "GPT2Tokenizer":
        return cls(os.path.join(dirpath, "vocab.json"), os.path.join(dirpath, "merges.txt"))

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if word[j] == first and j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self._cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in re.findall(_PAT, text):
            piece = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[bpe_tok] for bpe_tok in self._bpe(piece).split(" "))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


DEFAULT_TEXT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "alice.txt",
)


def load_eval_ids(tokenizer, path: str = DEFAULT_TEXT, max_chars: int = 60000) -> torch.Tensor:
    """Strip the Gutenberg header/footer, tokenize the body, return a 1-D LongTensor."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eval text not found at {path}. Download with:\n"
            f"  curl -fsSL https://www.gutenberg.org/files/11/11-0.txt -o {path}"
        )
    raw = open(path, encoding="utf-8").read()
    m = re.search(r"\*\*\* START OF.*?\*\*\*(.*)\*\*\* END OF", raw, re.S)
    body = (m.group(1) if m else raw).strip()
    body = re.sub(r"\n{2,}", "\n\n", body)
    return torch.tensor(tokenizer.encode(body[:max_chars]), dtype=torch.long)
