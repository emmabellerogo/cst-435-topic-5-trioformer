"""api/onnx_encoder.py (torch-free) must reproduce api/encoder_demo.py (PyTorch reference)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from api import onnx_encoder as oe  # noqa: E402

pytestmark = pytest.mark.skipif(not oe.available(), reason="models/edu/*.onnx not present")

TEXTS = ["the cat sat on the mat", "to be or not to be", "a", "a b c d e f g h i j k l m n o p"]


def test_attention_matches_the_pytorch_encoder_for_every_configuration():
    pytest.importorskip("torch")
    from api import encoder_demo as ref

    worst = 0.0
    for text in TEXTS:
        for heads in (1, 2, 4):
            for use_pe in (True, False):
                a = oe.compute_attention(text, heads, use_pe)
                b = ref.compute_attention(text, heads, use_pe)
                assert a["tokens"] == b["tokens"] and a["attention_shape"] == b["attention_shape"]
                worst = max(worst, float(np.abs(np.array(a["attention"]) - np.array(b["attention"])).max()),
                            float(np.abs(np.array(a["mean_attention"]) - np.array(b["mean_attention"])).max()))
                for k in ("num_heads", "d_model", "head_dim", "use_positional_encoding", "num_tokens", "disclaimer"):
                    assert a[k] == b[k], k
    assert worst < 1e-5, f"ONNX educational encoder differs from PyTorch by {worst:.2e}"


def test_helpers_and_constants_match_the_reference():
    pytest.importorskip("torch")
    from api import encoder_demo as ref
    from src.transformer import SinusoidalPositionalEncoding

    for c in ("D_MODEL", "D_FF", "VOCAB_BUCKETS", "MAX_TOKENS", "SEED", "DISCLAIMER"):
        assert getattr(oe, c) == getattr(ref, c), c
    for w in ("hello", "Cat", "zebra", "the"):
        assert oe.token_bucket(w) == ref.token_bucket(w)
    for n in (1, 6, 32):
        pe_ref = SinusoidalPositionalEncoding(oe.D_MODEL, 32).pe[:n].numpy()
        assert np.allclose(oe.sinusoidal_positions(n), pe_ref, atol=1e-6)
    ref_emb = ref._model(4)[0].weight.detach().numpy()
    assert np.array_equal(np.load(oe.edu_dir() / "embedding.npy"), ref_emb)   # exported weights are the seeded ones


def test_shapes_normalisation_and_determinism():
    a = oe.compute_attention("the cat sat on the mat", 4)
    assert a["attention_shape"] == [4, 6, 6] and a["head_dim"] == 8
    arr = np.array(a["attention"])
    assert np.allclose(arr.sum(-1), 1.0, atol=1e-5) and (arr >= 0).all()
    assert np.allclose(np.array(a["mean_attention"]).sum(-1), 1.0, atol=1e-5)
    assert a["max_row_sum_error"] < 1e-5
    assert a == oe.compute_attention("the cat sat on the mat", 4)
    assert "untrained" in a["disclaimer"] and "NOT" in a["disclaimer"]


def test_positional_encoding_changes_attention_and_identical_words_match_without_it():
    base = oe.compute_attention("to be or not to be", 2, True)["attention"]
    no_pe = oe.compute_attention("to be or not to be", 2, False)["attention"]
    assert base != no_pe
    assert np.allclose(no_pe[0][0], no_pe[0][4], atol=1e-6)      # identical word, no position info -> same row


@pytest.mark.parametrize("text,heads", [("", 4), ("   ", 4), ("a b", 3), (" ".join(["w"] * 33), 4)])
def test_invalid_inputs_raise_value_error(text, heads):
    with pytest.raises(ValueError):
        oe.compute_attention(text, heads)
