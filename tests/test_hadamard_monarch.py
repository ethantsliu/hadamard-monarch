"""Test suite: core layers, quantization, and GPT-2 smoke tests.

The GPT-2 tests skip cleanly (rather than fail) when the optional, gitignored
assets haven't been downloaded, so the suite stays green on a fresh clone.
"""
import os

import pytest
import torch

from hadamard_monarch.transforms import fwht, next_pow2, HadamardMix, MonarchLinear
from hadamard_monarch.quant import (
    incoherence,
    quantize,
    quantize_with_hadamard,
    hadamard_rotate,
    hadamard_unrotate,
    block_fwht,
    rel_error,
)
from hadamard_monarch.data import outlier_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
WEIGHTS = os.path.join(MODELS, "gpt2_pytorch_model.bin")
HAS_TOK = os.path.exists(os.path.join(MODELS, "vocab.json")) and os.path.exists(os.path.join(MODELS, "merges.txt"))
HAS_W = os.path.exists(WEIGHTS)


# ---------------- Hadamard transform ----------------

def test_fwht_involution():
    x = torch.randn(4, 64)
    assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)


def test_fwht_norm_preserving():
    x = torch.randn(8, 128)
    assert torch.allclose(x.norm(dim=-1), fwht(x).norm(dim=-1), atol=1e-4)


def test_fwht_is_orthogonal_matrix():
    n = 32
    H = fwht(torch.eye(n))            # columns = transform of basis vectors
    assert torch.allclose(H @ H.t(), torch.eye(n), atol=1e-5)


def test_fwht_rejects_non_power_of_two():
    try:
        fwht(torch.randn(3, 12))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-power-of-two last dim")


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(5) == 8
    assert next_pow2(16) == 16
    assert next_pow2(17) == 32


def test_hadamard_mix_shape_and_param_free():
    mix = HadamardMix(48, randomize=True, learn_scale=False)
    assert sum(p.numel() for p in mix.parameters()) == 0   # no learnable params
    out = mix(torch.randn(5, 48))
    assert out.shape == (5, 48)


# ---------------- Monarch layer ----------------

def test_monarch_param_count_subquadratic():
    n = 256
    m = MonarchLinear(n, n, bias=False)
    assert m.param_count() < n * n            # cheaper than dense
    # square case: 2 * n * sqrt(n) for nblocks = sqrt(n)
    assert m.nblocks == 16
    assert m.param_count() == 2 * (16 * 16 * 16)


def test_monarch_shapes_square_and_rectangular():
    for in_f, out_f, b in [(64, 64, 8), (256, 1024, 16), (128, 64, 8)]:
        m = MonarchLinear(in_f, out_f, nblocks=b)
        assert m(torch.randn(7, in_f)).shape == (7, out_f)


def test_monarch_to_dense_matches_forward():
    m = MonarchLinear(64, 96, nblocks=8)
    x = torch.randn(10, 64)
    W = m.to_dense()
    assert W.shape == (96, 64)
    ref = x @ W.t() + m.bias
    assert torch.allclose(ref, m(x), atol=1e-4)


def test_monarch_rejects_bad_nblocks():
    try:
        MonarchLinear(10, 20, nblocks=3)     # 10 % 3 != 0
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible nblocks")


# ---------------- Quantization + Hadamard incoherence ----------------

def test_rotate_unrotate_roundtrip_nonpow2():
    # Non-power-of-two dims: block-wise Hadamard must still round-trip exactly.
    W = torch.randn(96, 80)
    back = hadamard_unrotate(hadamard_rotate(W))
    assert torch.allclose(back, W, atol=1e-4)


def test_rotation_lowers_incoherence_on_outliers():
    W = outlier_weight(256, 256, seed=0)
    assert incoherence(hadamard_rotate(W)) < incoherence(W)


def test_quantize_more_bits_lower_error():
    W = torch.randn(128, 128)
    e8 = rel_error(quantize(W, 8), W)
    e4 = rel_error(quantize(W, 4), W)
    assert e8 < e4


def test_hadamard_helps_quantize_outlier_weights():
    W = outlier_weight(256, 256, seed=0)
    base = rel_error(quantize(W, 4), W)
    had = rel_error(quantize_with_hadamard(W, 4), W)
    assert had < base * 0.7        # a real, sizeable improvement at 4-bit


def test_hadamard_neutral_on_gaussian_weights():
    # No outliers -> already incoherent -> rotation should not help (within noise).
    W = torch.randn(256, 256, generator=torch.Generator().manual_seed(3)) * 0.05
    base = rel_error(quantize(W, 4), W)
    had = rel_error(quantize_with_hadamard(W, 4), W)
    assert abs(had - base) / base < 0.1


def test_block_fwht_involution_nonpow2():
    # block-wise FWHT over a non-power-of-two dim (96 = 32*3, block 32) round-trips.
    x = torch.randn(4, 96)
    assert torch.allclose(block_fwht(block_fwht(x, 32), 32), x, atol=1e-5)


def test_group_quant_beats_per_tensor_on_outliers():
    W = outlier_weight(256, 256, seed=0)
    e_tensor = rel_error(quantize(W, 4, per_row=False), W)
    e_group = rel_error(quantize(W, 4, group_size=64), W)
    assert e_group < e_tensor        # finer granularity fights outliers


def test_real_gpt2_weights_if_available():
    """If the GPT-2 checkpoint is present, the win should hold on real weights too."""
    from hadamard_monarch.data import available, load_gpt2_weight

    if not available():
        pytest.skip("GPT-2 checkpoint not downloaded (optional)")
    W = load_gpt2_weight("h.0.mlp.c_proj.weight")
    base = rel_error(quantize(W, 4), W)
    had = rel_error(quantize_with_hadamard(W, 4), W)
    assert had < base        # Hadamard helps on real weights


# ---------------- GPT-2 smoke tests (asset-gated) ----------------

@pytest.mark.skipif(not HAS_TOK, reason="tokenizer files not downloaded (optional)")
def test_tokenizer_known_ids_and_roundtrip():
    from hadamard_monarch.model import GPT2Tokenizer

    tok = GPT2Tokenizer.from_dir(MODELS)
    assert tok.encode("hello world") == [31373, 995]
    assert tok.encode(" Hello") == [18435]
    for s in ["hello world", "The quick brown fox.", " GPT-2 rocks!"]:
        assert tok.decode(tok.encode(s)) == s


@pytest.mark.skipif(not (HAS_TOK and HAS_W), reason="GPT-2 weights/tokenizer not downloaded (optional)")
def test_baseline_perplexity_is_sane():
    from hadamard_monarch.model import GPT2Tokenizer, load_eval_ids, load_gpt2

    tok = GPT2Tokenizer.from_dir(MODELS)
    model = load_gpt2(WEIGHTS, device="cpu")
    ids = load_eval_ids(tok, max_chars=10000)
    ppl = model.perplexity(ids, ctx=256)
    assert 20 < ppl < 60, f"GPT-2 perplexity {ppl:.1f} outside sane range (impl bug?)"


@pytest.mark.skipif(not (HAS_TOK and HAS_W), reason="GPT-2 weights/tokenizer not downloaded (optional)")
def test_hadamard_helps_4bit_activation_quant():
    # The headline: with 4-bit activations, the Hadamard rotation should clearly help.
    from hadamard_monarch.model import GPT2Tokenizer, load_eval_ids, load_gpt2, configure_quant_, reset_quant_

    tok = GPT2Tokenizer.from_dir(MODELS)
    model = load_gpt2(WEIGHTS, device="cpu")
    ids = load_eval_ids(tok, max_chars=8000)
    configure_quant_(model, wbits=8, abits=4, hadamard=False)
    ppl_base = model.perplexity(ids, ctx=256)
    configure_quant_(model, wbits=8, abits=4, hadamard=True)
    ppl_had = model.perplexity(ids, ctx=256)
    reset_quant_(model)
    assert ppl_had < ppl_base * 0.5, f"Hadamard should help a lot at A4: {ppl_base:.0f} -> {ppl_had:.0f}"
