"""Unit tests for the hand-written encoder in ``src/transformer.py``."""
from __future__ import annotations

import math

import pytest
import torch

from src.transformer import (
    MultiHeadSelfAttention,
    SinusoidalPositionalEncoding,
    TransformerEncoder,
    TransformerEncoderBlock,
    scaled_dot_product_attention,
)

B, T, D, H, FF = 3, 7, 16, 4, 32
ATOL = 1e-5  # float32 tolerance


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


def _x() -> torch.Tensor:
    return torch.randn(B, T, D)


# ------------------------------------------------------------------ shapes
def test_block_output_shape_equals_input_shape():
    block = TransformerEncoderBlock(D, H, FF).eval()
    out, attn = block(_x())
    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)


def test_encoder_output_shape_equals_input_shape():
    enc = TransformerEncoder(D, H, FF, num_layers=3).eval()
    out, attns = enc(_x())
    assert out.shape == (B, T, D)
    assert len(attns) == 3 and all(a.shape == (B, H, T, T) for a in attns)


def test_mha_output_shape():
    out, attn = MultiHeadSelfAttention(D, H)(_x())
    assert out.shape == (B, T, D) and attn.shape == (B, H, T, T)


# --------------------------------------------------------------- attention
def test_attention_weights_sum_to_one_over_keys():
    _, w = scaled_dot_product_attention(
        torch.randn(B, H, T, 4), torch.randn(B, H, T, 4), torch.randn(B, H, T, 4)
    )
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=ATOL)
    assert (w >= 0).all()

    _, w2 = MultiHeadSelfAttention(D, H)(_x())
    assert torch.allclose(w2.sum(-1), torch.ones(B, H, T), atol=ATOL)


def test_attention_matches_manual_computation():
    q, k, v = torch.randn(2, 5, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 3)
    out, w = scaled_dot_product_attention(q, k, v)
    expected_w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(4), dim=-1)
    assert torch.allclose(w, expected_w, atol=ATOL)
    assert torch.allclose(out, expected_w @ v, atol=ATOL)


def test_mask_blocks_attention_and_rows_still_normalised():
    q = k = v = torch.randn(1, 1, 5, 4)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()  # causal
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.all(w.triu(diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 1, 5), atol=ATOL)
    # first query can only see key 0 -> output equals v[0]
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=ATOL)


def test_fully_masked_query_row_has_zero_weights_and_zero_output():
    q = torch.randn(2, 1, 3, 4, requires_grad=True)
    k = torch.randn(2, 1, 3, 4)
    v = torch.randn(2, 1, 3, 4)
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[0] = False  # query 0 may attend to nothing
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.all(w[:, :, 0] == 0)
    assert torch.all(out[:, :, 0] == 0)
    # other rows are untouched and still normalised
    assert torch.allclose(w[:, :, 1:].sum(-1), torch.ones(2, 1, 2), atol=ATOL)
    # no NaNs, including in gradients
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()


def test_fully_masked_row_via_broadcast_key_padding_mask():
    # Batch item 1 has every key padded out -> all its rows are zero.
    q = k = v = torch.randn(2, 2, 4, 5)
    mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
    mask[1] = False
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.all(w[1] == 0) and torch.all(out[1] == 0)
    assert torch.allclose(w[0].sum(-1), torch.ones(2, 4), atol=ATOL)


def test_non_bool_mask_rejected():
    q = torch.randn(1, 2, 4)
    with pytest.raises(ValueError, match="bool"):
        scaled_dot_product_attention(q, q, q, torch.ones(2, 2))


def test_padding_mask_keys_get_zero_weight_in_block():
    block = TransformerEncoderBlock(D, H, FF).eval()
    key_ok = torch.ones(B, 1, 1, T, dtype=torch.bool)
    key_ok[..., -2:] = False
    _, attn = block(_x(), key_ok)
    assert torch.all(attn[..., -2:] == 0)


# --------------------------------------------------------------- positional
def test_positional_encoding_sin_even_cos_odd():
    pe = SinusoidalPositionalEncoding(D, max_len=20).pe
    pos = torch.arange(20, dtype=torch.float64)
    for i in range(D // 2):
        angle = pos / (10000.0 ** (2 * i / D))
        assert torch.allclose(pe[:, 2 * i].double(), torch.sin(angle), atol=1e-5)
        assert torch.allclose(pe[:, 2 * i + 1].double(), torch.cos(angle), atol=1e-5)
    assert torch.allclose(pe[0, 0::2], torch.zeros(D // 2))  # sin(0)
    assert torch.allclose(pe[0, 1::2], torch.ones(D // 2))   # cos(0)


def test_positional_encoding_odd_d_model_and_length_limit():
    pe = SinusoidalPositionalEncoding(7, max_len=5)
    assert pe.pe.shape == (5, 7) and torch.isfinite(pe.pe).all()
    with pytest.raises(ValueError, match="max_len"):
        pe(torch.zeros(1, 6, 7))


# ----------------------------------------------------------------- gradients
def test_end_to_end_gradients_finite_and_nonzero():
    enc = TransformerEncoder(D, H, FF, num_layers=2)
    x = _x().requires_grad_(True)
    out, _ = enc(x)
    out.pow(2).mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if name.endswith("k_proj.bias"):
            # Mathematically zero: softmax is invariant to a per-query constant
            # shift of the scores, which is exactly what a key bias adds.
            assert p.grad.abs().max() < 1e-6, name
        else:
            assert p.grad.abs().sum() > 0, f"zero grad for {name}"


# --------------------------------------------------------------- permutation
def _perm() -> torch.Tensor:
    return torch.tensor([3, 0, 6, 1, 5, 2, 4])


def test_permutation_equivariant_without_positional_encoding():
    enc = TransformerEncoder(
        D, H, FF, num_layers=2, dropout=0.0, use_positional_encoding=False
    ).eval()
    x, perm = _x(), _perm()
    with torch.no_grad():
        out, _ = enc(x)
        out_perm, _ = enc(x[:, perm])
    assert torch.allclose(out[:, perm], out_perm, atol=1e-5, rtol=1e-4)


def test_permutation_equivariance_broken_with_positional_encoding():
    enc = TransformerEncoder(
        D, H, FF, num_layers=2, dropout=0.0, use_positional_encoding=True
    ).eval()
    x, perm = _x(), _perm()  # distinct random tokens -> a nontrivial reorder
    with torch.no_grad():
        out, _ = enc(x)
        out_perm, _ = enc(x[:, perm])
    gap = (out[:, perm] - out_perm).abs().max().item()
    assert gap > 1e-2, f"positional encoding should break equivariance, gap={gap}"


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("d_model,heads", [(10, 4), (16, 3), (7, 2)])
def test_indivisible_heads_raise_clear_error(d_model, heads):
    with pytest.raises(ValueError, match=r"divisible by num_heads"):
        MultiHeadSelfAttention(d_model, heads)
    with pytest.raises(ValueError, match=r"divisible by num_heads"):
        TransformerEncoderBlock(d_model, heads, 8)


@pytest.mark.parametrize("d_model,heads", [(16, 0), (16, -2), (0, 4), (16, 2.0)])
def test_nonpositive_or_nonint_config_raises(d_model, heads):
    with pytest.raises(ValueError, match="positive integer"):
        MultiHeadSelfAttention(d_model, heads)


def test_wrong_input_width_raises():
    with pytest.raises(ValueError, match="expected input of shape"):
        MultiHeadSelfAttention(D, H)(torch.randn(B, T, D + 1))
