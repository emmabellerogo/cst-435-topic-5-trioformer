"""Tests for the benchmark setup: weight mapping and forward/backward equivalence
between our TransformerEncoderBlock and torch.nn.TransformerEncoderLayer."""
from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_encoder import (
    build_pair,
    randomize_layernorms,
    copy_custom_to_torch,
    equivalence_report,
    make_batches,
    make_custom_block,
    make_torch_layer,
)

D, H, FF = 32, 4, 64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _pair_blocks():
    custom = make_custom_block(D, H, FF)
    randomize_layernorms(custom)  # default LayerNorm (1, 0) would make copying unobservable
    ref = make_torch_layer(D, H, FF)
    return custom, ref


def test_reference_layer_is_configured_like_our_block():
    ref = make_torch_layer(D, H, FF)
    assert ref.norm_first is True
    assert ref.self_attn.batch_first is True
    assert ref.dropout.p == 0 and ref.dropout1.p == 0 and ref.dropout2.p == 0
    assert ref.norm1.eps == ref.norm2.eps == 1e-5
    assert ref.linear1.out_features == FF
    assert ref.self_attn.num_heads == H
    assert ref.activation is torch.nn.functional.gelu


def test_every_torch_parameter_is_mapped_and_overwritten():
    custom, ref = _pair_blocks()
    before = {n: p.detach().clone() for n, p in ref.named_parameters()}
    written = copy_custom_to_torch(custom, ref)
    assert set(written) == set(before)  # nothing missed, nothing invented
    # Starting from an independent random init, every tensor must have changed.
    for n, p in ref.named_parameters():
        assert not torch.equal(p, before[n]), f"{n} was not overwritten"


def test_fused_qkv_layout_matches_separate_projections():
    custom, ref = _pair_blocks()
    copy_custom_to_torch(custom, ref)
    a = custom.attn
    w = ref.self_attn.in_proj_weight
    b = ref.self_attn.in_proj_bias
    assert w.shape == (3 * D, D) and b.shape == (3 * D,)
    # rows [0:D] = Q, [D:2D] = K, [2D:3D] = V
    assert torch.equal(w[:D], a.q_proj.weight) and torch.equal(b[:D], a.q_proj.bias)
    assert torch.equal(w[D : 2 * D], a.k_proj.weight) and torch.equal(b[D : 2 * D], a.k_proj.bias)
    assert torch.equal(w[2 * D :], a.v_proj.weight) and torch.equal(b[2 * D :], a.v_proj.bias)
    assert torch.equal(ref.self_attn.out_proj.weight, a.out_proj.weight)
    assert torch.equal(ref.linear1.weight, custom.mlp[0].weight)
    assert torch.equal(ref.linear2.bias, custom.mlp[2].bias)
    assert torch.equal(ref.norm1.weight, custom.norm1.weight)
    assert torch.equal(ref.norm2.bias, custom.norm2.bias)


def test_forward_outputs_match_train_and_eval_float32():
    custom, ref = _pair_blocks()
    copy_custom_to_torch(custom, ref)
    x = torch.randn(4, 9, D)
    custom.train(), ref.train()
    assert torch.allclose(custom(x)[0], ref(x), atol=1e-5, rtol=1e-5)
    custom.eval(), ref.eval()  # torch may use its fused inference path here
    with torch.no_grad():
        assert torch.allclose(custom(x)[0], ref(x), atol=1e-5, rtol=1e-5)


def test_forward_outputs_match_tightly_in_float64():
    custom, ref = _pair_blocks()
    copy_custom_to_torch(custom, ref)
    custom, ref = custom.double(), ref.double()
    x = torch.randn(4, 9, D, dtype=torch.float64)
    assert torch.allclose(custom(x)[0], ref(x), atol=1e-12, rtol=0)


def test_wrong_qkv_order_is_detected():
    """Guard against the mapping test being vacuous: a swapped Q/K order must differ."""
    custom, ref = _pair_blocks()
    copy_custom_to_torch(custom, ref)
    with torch.no_grad():
        w = ref.self_attn.in_proj_weight
        w.copy_(torch.cat([w[D : 2 * D], w[:D], w[2 * D :]], dim=0))  # swap Q and K
    x = torch.randn(4, 9, D)
    assert not torch.allclose(custom(x)[0], ref(x), atol=1e-3)


def test_full_models_start_equivalent_in_output_and_gradients():
    custom, ref = build_pair(D, H, FF, seed=3)
    tokens = torch.randint(1, 6, (8, 12))
    rep = equivalence_report(custom, ref, tokens)
    assert rep["block_output_max_abs_diff"] < 1e-5
    assert rep["block_output_eval_fastpath_max_abs_diff"] < 1e-5
    assert rep["logits_max_abs_diff"] < 1e-5
    assert rep["grad_max_abs_diff"] < 1e-5  # includes fused QKV gradient vs concat of q/k/v grads


def test_layernorm_params_are_nontrivial_and_copied():
    custom, ref = build_pair(D, H, FF, seed=3)
    assert not torch.allclose(custom.block.norm1.weight, torch.ones(D))
    assert not torch.allclose(custom.block.norm2.bias, torch.zeros(D))
    for n in ("norm1", "norm2"):
        assert torch.equal(getattr(ref.block, n).weight, getattr(custom.block, n).weight)
        assert torch.equal(getattr(ref.block, n).bias, getattr(custom.block, n).bias)


def test_batches_are_deterministic():
    x1, y1, b1 = make_batches(256, 8, 32, 10, seed=7)
    x2, y2, b2 = make_batches(256, 8, 32, 10, seed=7)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert all(torch.equal(a, b) for a, b in zip(b1, b2)) and len(b1) == 10
