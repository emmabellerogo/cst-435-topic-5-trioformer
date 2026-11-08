"""Unit tests for api/model_pipelines.py using lightweight fakes.

No model weights are downloaded and no network is used: ``_load_bundle`` is
replaced by a counting loader that returns tiny fake tokenizers/models.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from api import model_pipelines as mp

_REAL_LOAD_BUNDLE = mp._load_bundle  # captured before the autouse fixture replaces it

EOS = 50256
MASK_ID = 103


# ------------------------------------------------------------------- fakes
class FakeTokenizer:
    """Whitespace tokenizer with a growing vocabulary; [MASK] -> MASK_ID."""

    mask_token_id = MASK_ID
    eos_token_id = EOS

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.inv: Dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word == "[MASK]":
            return MASK_ID
        if word not in self.vocab:
            self.vocab[word] = 1000 + len(self.vocab)
            self.inv[self.vocab[word]] = word
        return self.vocab[word]

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = torch.tensor([[self._id(w) for w in text.split()]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special = {0, EOS}
        words: List[str] = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            words.append(self.inv.get(i, {MASK_ID: "[MASK]", 7: "paris", 8: "##ing"}.get(i, f"<{i}>")))
        return " ".join(words)


class FakeBert:
    """Returns logits that rank vocab ids 7 > 8 > ... at every position."""

    def __init__(self) -> None:
        self.saw_inference_mode = False

    def __call__(self, input_ids, attention_mask=None):
        self.saw_inference_mode = torch.is_inference_mode_enabled()
        logits = torch.zeros(1, input_ids.shape[1], 30)
        logits[..., 7] = 5.0
        logits[..., 8] = 3.0
        logits[..., 9] = 1.0
        return SimpleNamespace(logits=logits)


class FakeGenerator:
    """Records generate() kwargs; echoes the prompt (as GPT-2 does) then appends ids."""

    def __init__(self, new_ids: List[int], seq2seq: bool) -> None:
        self.new_ids, self.seq2seq = new_ids, seq2seq
        self.kwargs: dict = {}
        self.saw_inference_mode = False

    def generate(self, **kwargs):
        self.kwargs = kwargs
        self.saw_inference_mode = torch.is_inference_mode_enabled()
        new = torch.tensor([self.new_ids])
        if self.seq2seq:  # T5: [decoder_start=0, generated...]
            return torch.cat([torch.zeros(1, 1, dtype=torch.long), new], dim=1)
        return torch.cat([kwargs["input_ids"], new], dim=1)  # GPT-2 echoes the prompt


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch):
    mp.clear_model_cache()
    calls: List[str] = []
    models: Dict[str, object] = {}

    def loader(name: str, kind: str) -> mp._Bundle:
        calls.append(name)
        model = {
            mp.BERT_MODEL: FakeBert(),
            mp.GPT2_MODEL: FakeGenerator([2001, 2002, EOS], seq2seq=False),
            mp.T5_MODEL: FakeGenerator([3001, 3002, 3003, EOS], seq2seq=True),
        }[name]
        models[name] = model
        return mp._Bundle(tokenizer=FakeTokenizer(), model=model)

    monkeypatch.setattr(mp, "_load_bundle", loader)
    yield SimpleNamespace(calls=calls, models=models)
    mp.clear_model_cache()


# -------------------------------------------------------------------- BERT
def test_bert_returns_ranked_predictions_and_filled_sentence(fake_loader):
    r = mp.bert_fill_mask("the capital of france is [MASK] .", top_k=3)
    assert r.model == "bert-base-uncased"
    assert [p.token for p in r.predictions] == ["paris", "ing", "<9>"]
    assert r.predictions[0].sequence == "the capital of france is paris ."
    assert r.output_text == r.predictions[0].sequence
    scores = [p.score for p in r.predictions]
    assert scores == sorted(scores, reverse=True) and 0 < sum(scores) <= 1.0 + 1e-6
    assert r.input_tokens == 7
    assert fake_loader.models[mp.BERT_MODEL].saw_inference_mode


def test_bert_wordpiece_continuation_is_glued_to_previous_word(fake_loader):
    r = mp.bert_fill_mask("walk [MASK]", top_k=2)
    assert r.predictions[1].token == "ing"           # "##ing" -> "ing"
    assert r.predictions[1].sequence == "walking"    # no stray space


@pytest.mark.parametrize(
    "text",
    ["no mask here", "two [MASK] and [MASK] masks", "lowercase [mask] is not the token", "", "   "],
)
def test_bert_requires_exactly_one_mask(text, fake_loader):
    with pytest.raises(mp.InvalidInputError):
        mp.bert_fill_mask(text)
    assert fake_loader.calls == []  # validation happens before any model load


@pytest.mark.parametrize("top_k", [0, 11, -1, 2.5, "3", True])
def test_bert_top_k_validation(top_k, fake_loader):
    with pytest.raises(mp.InvalidInputError, match="top_k"):
        mp.bert_fill_mask("a [MASK] b", top_k=top_k)
    assert fake_loader.calls == []


def test_bert_rejects_over_512_tokens(fake_loader):
    text = "[MASK] " + "a " * 600  # ~1200 chars (under the char cap) but 601 tokens
    with pytest.raises(mp.InvalidInputError, match="too long for BERT"):
        mp.bert_fill_mask(text)


def test_non_string_and_oversized_text_rejected(fake_loader):
    with pytest.raises(mp.InvalidInputError, match="must be a string"):
        mp.bert_fill_mask(None)  # type: ignore[arg-type]
    with pytest.raises(mp.InvalidInputError, match="too long"):
        mp.bert_fill_mask("x" * (mp.MAX_INPUT_CHARS + 1) + " [MASK]")
    assert fake_loader.calls == []


# ------------------------------------------------------------------- GPT-2
def test_gpt2_returns_only_continuation_not_prompt(fake_loader):
    r = mp.gpt2_continue("once upon a time", max_new_tokens=10)
    assert r.model == "gpt2"
    assert "once" not in r.output_text and "upon" not in r.output_text
    assert r.prompt_tokens == 4
    assert r.generated_tokens == 3          # 2 words + EOS (EOS is skipped in the text)
    assert r.output_text == "<2001> <2002>"


def test_gpt2_generation_is_deterministic_and_pads_explicitly(fake_loader):
    mp.gpt2_continue("hello world", max_new_tokens=7)
    kw = fake_loader.models[mp.GPT2_MODEL].kwargs
    assert kw["do_sample"] is False and kw["num_beams"] == 1
    assert kw["max_new_tokens"] == 7
    assert kw["pad_token_id"] == EOS        # set explicitly: GPT-2 has no pad token
    assert "temperature" not in kw and "top_p" not in kw
    assert "attention_mask" in kw
    assert fake_loader.models[mp.GPT2_MODEL].saw_inference_mode


@pytest.mark.parametrize("n", [0, -5, mp.MAX_NEW_TOKENS_LIMIT + 1, 1.5, "10", None, True])
def test_gpt2_max_new_tokens_validation(n, fake_loader):
    with pytest.raises(mp.InvalidInputError, match="max_new_tokens"):
        mp.gpt2_continue("hi", max_new_tokens=n)
    assert fake_loader.calls == []


def test_gpt2_prompt_plus_new_tokens_must_fit_context(fake_loader):
    prompt = "a " * 1000                    # 1000 tokens, 2000 chars
    with pytest.raises(mp.InvalidInputError, match="context window"):
        mp.gpt2_continue(prompt, max_new_tokens=40)
    assert mp.gpt2_continue(prompt, max_new_tokens=24).prompt_tokens == 1000  # exactly fits


# ---------------------------------------------------------------------- T5
def test_t5_uses_summarize_prefix_and_beam_search_settings(fake_loader):
    r = mp.t5_summarize("the quick brown fox jumps over the lazy dog", max_new_tokens=30)
    kw = fake_loader.models[mp.T5_MODEL].kwargs
    assert r.input_tokens == 9 + 1          # 9 words + "summarize:" (prefix is one fake token)
    assert kw["do_sample"] is False and kw["num_beams"] == 4
    assert kw["max_new_tokens"] == 30 and kw["min_new_tokens"] == 0
    assert kw["no_repeat_ngram_size"] == 3 and kw["early_stopping"] is True
    assert r.model == "t5-small"
    assert r.output_tokens == 4             # 3 words + EOS, decoder-start excluded
    assert r.output_text == "<3001> <3002> <3003>"
    assert fake_loader.models[mp.T5_MODEL].saw_inference_mode


def test_t5_prefix_is_prepended_to_the_text(fake_loader, monkeypatch):
    seen = {}
    real_call = FakeTokenizer.__call__

    def spy(self, text, return_tensors="pt"):
        seen["text"] = text
        return real_call(self, text, return_tensors)

    monkeypatch.setattr(FakeTokenizer, "__call__", spy)
    mp.t5_summarize("  some article text  ")
    assert seen["text"] == "summarize: some article text"


def test_t5_limit_validation(fake_loader):
    with pytest.raises(mp.InvalidInputError, match="must not exceed"):
        mp.t5_summarize("text", max_new_tokens=10, min_new_tokens=20)
    with pytest.raises(mp.InvalidInputError, match="max_new_tokens"):
        mp.t5_summarize("text", max_new_tokens=0)
    with pytest.raises(mp.InvalidInputError, match="min_new_tokens"):
        mp.t5_summarize("text", min_new_tokens=-1)
    assert fake_loader.calls == []


def test_t5_rejects_over_512_input_tokens(fake_loader):
    with pytest.raises(mp.InvalidInputError, match="too long for t5-small"):
        mp.t5_summarize("a " * 600)


def test_outputs_are_reproducible(fake_loader):
    assert mp.t5_summarize("same input text") == mp.t5_summarize("same input text")
    assert mp.gpt2_continue("same prompt") == mp.gpt2_continue("same prompt")
    assert mp.bert_fill_mask("same [MASK] input") == mp.bert_fill_mask("same [MASK] input")


# -------------------------------------------------------- caching / loading
def test_models_load_lazily_and_only_once(fake_loader):
    assert fake_loader.calls == [] and mp.loaded_models() == []   # nothing at import time
    for _ in range(3):
        mp.bert_fill_mask("a [MASK] b")
        mp.gpt2_continue("hello")
        mp.t5_summarize("hello world")
    assert sorted(fake_loader.calls) == sorted([mp.BERT_MODEL, mp.GPT2_MODEL, mp.T5_MODEL])
    assert mp.loaded_models() == sorted([mp.BERT_MODEL, mp.GPT2_MODEL, mp.T5_MODEL])


def test_concurrent_first_requests_load_once(monkeypatch, fake_loader):
    slow_calls: List[str] = []

    def slow_loader(name, kind):
        slow_calls.append(name)
        time.sleep(0.15)  # widen the race window
        return mp._Bundle(FakeTokenizer(), FakeGenerator([2001, EOS], seq2seq=False))

    monkeypatch.setattr(mp, "_load_bundle", slow_loader)
    errors: List[BaseException] = []

    def work():
        try:
            mp.gpt2_continue("hi there")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert slow_calls == [mp.GPT2_MODEL]


def test_load_failure_is_wrapped_not_cached_and_retryable(monkeypatch, fake_loader):
    attempts = {"n": 0}

    def flaky(name, kind):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("network down")
        return mp._Bundle(FakeTokenizer(), FakeGenerator([2001, EOS], seq2seq=False))

    monkeypatch.setattr(mp, "_load_bundle", flaky)
    with pytest.raises(mp.ModelLoadError, match="gpt2.*OSError.*network down") as ei:
        mp.gpt2_continue("hello")
    assert isinstance(ei.value.__cause__, OSError)
    assert mp.loaded_models() == []                      # failure not cached
    assert mp.gpt2_continue("hello").generated_tokens == 2   # retry works


def test_model_load_error_is_not_an_input_error():
    assert not issubclass(mp.ModelLoadError, mp.InvalidInputError)
    assert issubclass(mp.InvalidInputError, ValueError)


def test_clear_model_cache_forces_reload(fake_loader):
    mp.gpt2_continue("hello")
    mp.clear_model_cache()
    mp.gpt2_continue("hello")
    assert fake_loader.calls.count(mp.GPT2_MODEL) == 2


# -------------------- real loader function, with transformers stubbed out
def test_real_loader_sets_eval_cpu_and_respects_local_only(monkeypatch):
    import transformers

    seen = {}

    class FakeModel:
        def __init__(self):
            self.eval_called = self.cpu = False

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.cpu = device == "cpu"
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name, **kw):
            seen["model_kw"] = kw
            return FakeModel()

    class FakeAutoTok:
        @classmethod
        def from_pretrained(cls, name, **kw):
            seen["tok_kw"] = kw
            return object()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTok)
    for cls in ("AutoModelForMaskedLM", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"):
        monkeypatch.setattr(transformers, cls, FakeAutoModel)

    monkeypatch.delenv("TRIOFORMER_HF_LOCAL_ONLY", raising=False)
    bundle = _REAL_LOAD_BUNDLE("gpt2", "causal")
    assert bundle.model.eval_called and bundle.model.cpu
    assert seen["model_kw"] == {"local_files_only": False}

    monkeypatch.setenv("TRIOFORMER_HF_LOCAL_ONLY", "1")
    _REAL_LOAD_BUNDLE("gpt2", "causal")
    assert seen["tok_kw"] == {"local_files_only": True}
