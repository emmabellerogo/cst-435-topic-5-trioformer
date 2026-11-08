"""Optional smoke test: ONE real inference per pretrained model.

Skipped automatically unless the weights are already in the local Hugging Face
cache, so ordinary runs and CI never download hundreds of MB. To run it:

    # 1) weights already cached (e.g. from a previous download):
    pytest tests/test_model_pipelines_smoke.py -v -s

    # 2) allow it to download them (~1.2 GB total, once):
    RUN_HF_SMOKE_DOWNLOAD=1 pytest tests/test_model_pipelines_smoke.py -v -s

Set HF_HOME to relocate the cache. Without the DOWNLOAD flag the loader runs
with TRIOFORMER_HF_LOCAL_ONLY=1 so it can never reach the network.
"""
from __future__ import annotations

import os

import pytest

from api import model_pipelines as mp


def _cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False
    weights = ("model.safetensors", "pytorch_model.bin")
    has_cfg = isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    has_w = any(isinstance(try_to_load_from_cache(repo_id, w), str) for w in weights)
    return has_cfg and has_w


def _require(repo_id: str, monkeypatch) -> None:
    if os.environ.get("RUN_HF_SMOKE_DOWNLOAD") == "1":
        monkeypatch.delenv("TRIOFORMER_HF_LOCAL_ONLY", raising=False)
        return
    if not _cached(repo_id):
        pytest.skip(f"{repo_id} weights not in local cache (set RUN_HF_SMOKE_DOWNLOAD=1 to download)")
    monkeypatch.setenv("TRIOFORMER_HF_LOCAL_ONLY", "1")


@pytest.fixture(autouse=True)
def _fresh_cache():
    mp.clear_model_cache()
    yield
    mp.clear_model_cache()


def test_real_bert_fill_mask(monkeypatch):
    _require(mp.BERT_MODEL, monkeypatch)
    r = mp.bert_fill_mask("The capital of France is [MASK].", top_k=5)
    print("\nBERT:", r.output_text, [(p.token, round(p.score, 3)) for p in r.predictions])
    assert r.predictions[0].token == "paris"
    assert r.predictions == mp.bert_fill_mask("The capital of France is [MASK].", top_k=5).predictions


def test_real_gpt2_continuation(monkeypatch):
    _require(mp.GPT2_MODEL, monkeypatch)
    prompt = "The meaning of life is"
    r = mp.gpt2_continue(prompt, max_new_tokens=20)
    print("\nGPT-2:", repr(r.output_text), r.prompt_tokens, r.generated_tokens)
    assert r.output_text.strip() and prompt not in r.output_text
    assert 0 < r.generated_tokens <= 20
    assert mp.gpt2_continue(prompt, max_new_tokens=20) == r  # greedy => reproducible


def test_real_t5_summarization(monkeypatch):
    _require(mp.T5_MODEL, monkeypatch)
    article = (
        "The city council voted on Tuesday to approve a new bike lane network that will "
        "stretch across 40 miles of downtown streets. Supporters say the plan will reduce "
        "traffic and improve safety, while some business owners worry about losing parking. "
        "Construction is expected to begin next spring and finish within two years."
    )
    r = mp.t5_summarize(article, max_new_tokens=40)
    print("\nT5:", repr(r.output_text), r.input_tokens, r.output_tokens)
    assert r.output_text.strip() and len(r.output_text) < len(article)
    assert mp.t5_summarize(article, max_new_tokens=40) == r
