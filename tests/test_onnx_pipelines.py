"""Unit tests for api/onnx_pipelines.py with scripted fake ONNX sessions.

No model files, no onnxruntime and no network are needed: the real algorithms (validation, KV-cache
plumbing, n-gram banning, beam search, one-model-at-a-time eviction) run against tiny fakes.
Output *quality* against the PyTorch reference is covered by tests/test_onnx_parity.py (opt-in).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

from api import onnx_pipelines as op

GPT_EOS, T5_EOS, T5_START, MASK_ID = 50256, 1, 0, 103


# ------------------------------------------------------------------ fakes
class FakeTokenizer:
    """Whitespace tokenizer with a growing vocabulary (the subset of tokenizers.Tokenizer we use)."""

    def __init__(self, wrap=None, special=()):
        self.wrap, self.special = wrap, set(special)       # wrap=(prefix_ids, suffix_ids) for add_special_tokens
        self.vocab: Dict[str, int] = {}
        self.inv: Dict[int, str] = {7: "paris", 8: "##ing", 9: "lyon", 101: "[CLS]", 102: "[SEP]", MASK_ID: "[MASK]"}
        self.seen: List[str] = []

    def _id(self, word):
        if word == "[MASK]":
            return MASK_ID
        if word not in self.vocab:
            self.vocab[word] = 1000 + len(self.vocab)
            self.inv[self.vocab[word]] = word
        return self.vocab[word]

    def encode(self, text, add_special_tokens=True):
        self.seen.append(text)
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens and self.wrap:
            ids = list(self.wrap[0]) + ids + list(self.wrap[1])
        return SimpleNamespace(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids))

    def token_to_id(self, token):
        return MASK_ID if token == "[MASK]" else None

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for i in ids:
            if skip_special_tokens and i in self.special:
                continue
            words.append(self.inv.get(i, f"<{i}>"))
        return " ".join(words)


class BertSession:
    def __init__(self):
        self.feeds = None

    def run(self, _names, feeds):
        self.feeds = feeds
        logits = np.zeros((1, 40), np.float32)
        logits[0, 7], logits[0, 8], logits[0, 9] = 5.0, 3.0, 1.0
        return [logits]


class GptSession:
    """Emits ``program[step]`` as the argmax of each step; records what the cache looked like."""

    def __init__(self, program, vocab=GPT_EOS + 1, prefer=None):
        self.program, self.vocab, self.prefer, self.step, self.calls = list(program), vocab, prefer, 0, []
        self._inputs = [SimpleNamespace(name="input_ids")] + [
            SimpleNamespace(name=f"past_{i}_{kv}") for i in range(12) for kv in ("k", "v")]

    def get_inputs(self):
        return self._inputs

    def run(self, _names, feeds):
        past_len = feeds["past_0_k"].shape[2]
        n_in = feeds["input_ids"].shape[1]
        if past_len == 0:                                 # empty cache == a new generation: restart the script
            self.step = 0
        self.calls.append((n_in, past_len, feeds["input_ids"].dtype, feeds["past_0_k"].dtype))
        logits = np.full((1, self.vocab), -10.0, np.float32)
        if self.prefer:                                   # optional ranked preferences: first allowed wins
            for rank, tok in enumerate(self.prefer(self.step)):
                logits[0, tok] = 10.0 - rank
        else:
            logits[0, self.program[self.step]] = 10.0
        self.step += 1
        present = [np.zeros((1, 12, past_len + n_in, 64), np.float32) for _ in range(24)]
        return [logits, *present]


class T5Sessions:
    """Bigram model over a tiny vocabulary: ``table[last_token] = {next_token: logit}``."""

    V = 12

    def __init__(self, table, default_logit=-20.0):
        self.table, self.default = table, default_logit
        self.step_calls: List[int] = []
        names = ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"] + [
            f"past_{i}_{n}" for i in range(6) for n in ("sk", "sv", "ck", "cv")]
        step_inputs = [SimpleNamespace(name=n) for n in names]
        t5 = self

        class Encoder:
            def run(self, _n, feeds):
                return [np.zeros((1, feeds["input_ids"].shape[1], 8), np.float32)]

        class Cross:
            def run(self, _n, feeds):
                return [np.zeros((1, 8, feeds["hidden"].shape[1], 1), np.float32) for _ in range(12)]

        class Step:
            def get_inputs(self):
                return step_inputs

            def run(self, _n, feeds):
                last = feeds["decoder_input_ids"][:, 0]
                past_len = feeds["past_0_sk"].shape[2]
                t5.step_calls.append(past_len)
                logits = np.full((len(last), t5.V), t5.default, np.float32)
                for b, tok in enumerate(last.tolist()):
                    for nxt, val in t5.table.get(tok, {}).items():
                        logits[b, nxt] = val
                present = [np.zeros((len(last), 8, past_len + 1, 1), np.float32) for _ in range(12)]
                return [logits, *present]

        self.sessions = {"encoder": Encoder(), "cross": Cross(), "step": Step()}


def t5_bundle(table, tok=None):
    tok = tok or FakeTokenizer(wrap=((), (T5_EOS,)), special={T5_START, T5_EOS})
    tok.inv.update({5: "w5", 6: "w6", 7: "w7", 10: "w10", 11: "w11"})
    return op._Bundle(tokenizer=tok, sessions=T5Sessions(table).sessions), tok


@pytest.fixture(autouse=True)
def clean_cache():
    op.clear_model_cache()
    yield
    op.clear_model_cache()


@pytest.fixture
def loader(monkeypatch):
    """Replace the disk loader with scripted fakes; count and order the loads."""
    state = SimpleNamespace(calls=[], bert=BertSession(), gpt=GptSession([2001, 2002, GPT_EOS]),
                            tok={}, fail=False)

    def fake(name, kind):
        state.calls.append(name)
        if state.fail:
            raise OSError("disk exploded: secret-path-/etc/x")
        if name == op.BERT_MODEL:
            tok = FakeTokenizer(wrap=((101,), (102,)), special={101, 102})
            state.tok[name] = tok
            return op._Bundle(tok, {"model": state.bert})
        if name == op.GPT2_MODEL:
            tok = FakeTokenizer(special={GPT_EOS})
            state.tok[name] = tok
            return op._Bundle(tok, {"model": state.gpt})
        bundle, tok = t5_bundle({0: {5: 3.0}, 5: {6: 3.0}, 6: {T5_EOS: 3.0}})
        state.tok[name] = tok
        return bundle

    monkeypatch.setattr(op, "_load_bundle", fake)
    return state


# ------------------------------------------------------------------ BERT
def test_bert_ranks_predictions_and_fills_the_sentence(loader):
    r = op.bert_fill_mask("the capital of france is [MASK] .", top_k=3)
    assert r.model == "bert-base-uncased"
    assert [p.token for p in r.predictions] == ["paris", "ing", "lyon"]        # "##ing" -> "ing"
    assert r.predictions[0].sequence == "the capital of france is paris ."
    assert r.output_text == r.predictions[0].sequence
    scores = [p.score for p in r.predictions]
    assert scores == sorted(scores, reverse=True) and 0 < sum(scores) <= 1.0 + 1e-9
    assert r.input_tokens == 9                                                  # 7 words + [CLS] + [SEP]


def test_bert_feeds_int64_inputs_and_the_mask_position(loader):
    op.bert_fill_mask("a b [MASK] c")
    f = loader.bert.feeds
    assert {k: v.dtype for k, v in f.items()} == {k: np.dtype("int64") for k in
                                                  ("input_ids", "attention_mask", "token_type_ids", "mask_pos")}
    assert f["input_ids"][0, f["mask_pos"][0]] == MASK_ID                      # points at the [MASK] token
    assert f["input_ids"][0, 0] == 101 and f["input_ids"][0, -1] == 102


def test_bert_wordpiece_continuation_glues_to_previous_word(loader):
    r = op.bert_fill_mask("walk [MASK]", top_k=2)
    assert r.predictions[1].token == "ing" and r.predictions[1].sequence == "walking"


@pytest.mark.parametrize("text", ["no mask", "two [MASK] and [MASK]", "lower [mask] case", "", "   ", None, 5])
def test_bert_input_validation_before_any_load(text, loader):
    with pytest.raises(op.InvalidInputError):
        op.bert_fill_mask(text)
    assert loader.calls == []


@pytest.mark.parametrize("k", [0, 11, -1, 2.5, "3", True])
def test_bert_top_k_validation(k, loader):
    with pytest.raises(op.InvalidInputError, match="top_k"):
        op.bert_fill_mask("a [MASK]", top_k=k)
    assert loader.calls == []


def test_bert_rejects_over_512_tokens(loader):
    with pytest.raises(op.InvalidInputError, match="too long for BERT"):
        op.bert_fill_mask("[MASK] " + "a " * 600)


def test_oversized_text_rejected_before_loading(loader):
    with pytest.raises(op.InvalidInputError, match="too long"):
        op.bert_fill_mask("x" * (op.MAX_INPUT_CHARS + 1) + " [MASK]")
    assert loader.calls == []


# ------------------------------------------------------------------ GPT-2
def test_gpt2_returns_only_the_continuation_and_counts_tokens(loader):
    r = op.gpt2_continue("once upon a time", max_new_tokens=10)
    assert r.model == "gpt2" and r.prompt_tokens == 4
    assert r.generated_tokens == 3                       # two words + EOS
    assert r.output_text == "<2001> <2002>" and "once" not in r.output_text


def test_gpt2_kv_cache_plumbing(loader):
    op.gpt2_continue("a b c d e", max_new_tokens=10)
    calls = loader.gpt.calls
    assert [c[:2] for c in calls] == [(5, 0), (1, 5), (1, 6)]    # prefill whole prompt, then 1 token per step
    assert all(c[2] == np.dtype("int64") and c[3] == np.dtype("float32") for c in calls)


def test_gpt2_stops_at_eos_and_at_max_new_tokens(loader):
    assert op.gpt2_continue("a b", max_new_tokens=50).generated_tokens == 3
    op.clear_model_cache()
    loader.gpt = GptSession([2001] * 100)                # never emits EOS
    r = op.gpt2_continue("a b", max_new_tokens=7)
    assert r.generated_tokens == 7


def test_gpt2_never_repeats_a_trigram(monkeypatch):
    # prompt "a b c a b": last two tokens (a, b) already continued with c -> c must be banned.
    tok = FakeTokenizer(special={GPT_EOS})
    sess = GptSession([], prefer=lambda step: [tok._id("c"), tok._id("z")])      # model "wants" c, then z
    monkeypatch.setattr(op, "_load_bundle", lambda n, k: op._Bundle(tok, {"model": sess}))
    r = op.gpt2_continue("a b c a b", max_new_tokens=1)
    assert r.output_text == "z"                                                  # c banned -> second choice


@pytest.mark.parametrize("n", [0, -1, op.MAX_NEW_TOKENS_LIMIT + 1, 1.5, "10", None, True])
def test_gpt2_max_new_tokens_validation(n, loader):
    with pytest.raises(op.InvalidInputError, match="max_new_tokens"):
        op.gpt2_continue("hi", max_new_tokens=n)
    assert loader.calls == []


def test_gpt2_context_window_check(loader):
    with pytest.raises(op.InvalidInputError, match="context window"):
        op.gpt2_continue("a " * 1000, max_new_tokens=40)
    assert op.gpt2_continue("a " * 1000, max_new_tokens=24).prompt_tokens == 1000    # exactly 1024


# ------------------------------------------------------------------ T5
def test_t5_summarize_uses_prefix_and_reports_token_counts(loader):
    r = op.t5_summarize("the council approved a bike lane", max_new_tokens=20)
    assert loader.tok[op.T5_MODEL].seen == ["summarize: the council approved a bike lane"]
    assert r.model == "t5-small" and r.output_text == "w5 w6"
    assert r.input_tokens == 7 + 1                        # "summarize:" + 6 words... + </s>
    assert r.output_tokens == 3                           # w5, w6, EOS (decoder start excluded)


def test_t5_beam_search_finds_the_best_path_and_ends_with_eos():
    bundle, _ = t5_bundle({0: {5: 3.0, 6: 2.0}, 5: {6: 3.0, 7: 1.0}, 6: {7: 1.0, T5_EOS: 0.5}, 7: {T5_EOS: 3.0}})
    ids = op._t5_beam_search(bundle, [10, 11, T5_EOS], max_new_tokens=10, min_new_tokens=0)
    assert ids[0] == T5_START and ids[-1] == T5_EOS and ids[1:3] == [5, 6]


def test_t5_min_new_tokens_suppresses_early_eos():
    table = {0: {5: 3.0}, 5: {6: 3.0}, 6: {T5_EOS: 3.0, 7: 1.0}, 7: {T5_EOS: 3.0}}
    bundle, _ = t5_bundle(table)
    early = op._t5_beam_search(bundle, [10, T5_EOS], 10, 0)
    forced = op._t5_beam_search(t5_bundle(table)[0], [10, T5_EOS], 10, 3)
    assert early == [0, 5, 6, 1]
    assert forced == [0, 5, 6, 7, 1]                      # EOS blocked until 3 tokens exist


def test_t5_max_new_tokens_truncates_without_eos_and_no_trigram_repeats():
    table = {t: {5 + (t + 1) % 3: 3.0, 5 + (t + 2) % 3: 2.5} for t in (0, 5, 6, 7)}
    bundle, _ = t5_bundle(table)
    ids = op._t5_beam_search(bundle, [10, T5_EOS], max_new_tokens=4, min_new_tokens=0)
    assert len(ids) == 5 and ids[-1] != T5_EOS            # start + 4 generated, cut off at the limit
    trigrams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    assert len(trigrams) == len(set(trigrams))


def test_t5_decoder_cache_grows_one_step_at_a_time():
    sessions = T5Sessions({0: {5: 3.0}, 5: {6: 3.0}, 6: {T5_EOS: 3.0}})
    bundle = op._Bundle(FakeTokenizer(wrap=((), (T5_EOS,)), special={0, 1}), sessions.sessions)
    op._t5_beam_search(bundle, [10, T5_EOS], 10, 0)
    assert sessions.step_calls == list(range(len(sessions.step_calls))) and len(sessions.step_calls) >= 3


def test_t5_validation(loader):
    for kw, match in [({"max_new_tokens": 0}, "max_new_tokens"), ({"min_new_tokens": -1}, "min_new_tokens"),
                      ({"max_new_tokens": 5, "min_new_tokens": 9}, "must not exceed")]:
        with pytest.raises(op.InvalidInputError, match=match):
            op.t5_summarize("text", **kw)
    with pytest.raises(op.InvalidInputError, match="too long for t5-small"):
        op.t5_summarize("a " * 600)
    assert loader.calls == [op.T5_MODEL]                  # only the token-length check needed the tokenizer


# ------------------------------------------------------------------ one-model-at-a-time cache
def test_only_one_model_is_resident_and_switching_reloads(loader):
    op.bert_fill_mask("a [MASK]")
    assert op.loaded_models() == [op.BERT_MODEL]
    op.gpt2_continue("hello")
    assert op.loaded_models() == [op.GPT2_MODEL]          # BERT was evicted
    op.t5_summarize("hello world")
    assert op.loaded_models() == [op.T5_MODEL]
    op.gpt2_continue("hello")
    assert loader.calls == [op.BERT_MODEL, op.GPT2_MODEL, op.T5_MODEL, op.GPT2_MODEL]
    op.gpt2_continue("again")                              # same model: no reload
    assert loader.calls.count(op.GPT2_MODEL) == 2


def test_evicted_bundle_is_actually_released(loader):
    import gc
    import weakref

    op.bert_fill_mask("a [MASK]")
    ref = weakref.ref(op._CACHE[op.BERT_MODEL])
    op.gpt2_continue("hello")
    gc.collect()
    assert ref() is None                                   # nothing else keeps the sessions alive


def test_load_failure_is_wrapped_not_cached_and_retryable(loader):
    loader.fail = True
    with pytest.raises(op.ModelLoadError, match="gpt2.*OSError") as ei:
        op.gpt2_continue("hello")
    assert isinstance(ei.value.__cause__, OSError) and op.loaded_models() == []
    loader.fail = False
    assert op.gpt2_continue("hello").generated_tokens == 3


def test_failed_load_does_not_leave_a_half_evicted_state(loader):
    op.bert_fill_mask("a [MASK]")
    loader.fail = True
    with pytest.raises(op.ModelLoadError):
        op.gpt2_continue("hello")
    assert op.loaded_models() == []                        # old model freed first; nothing half-loaded
    loader.fail = False
    assert op.bert_fill_mask("a [MASK]").predictions


def test_concurrent_first_requests_load_once(monkeypatch):
    calls: List[str] = []

    def slow(name, kind):
        calls.append(name)
        time.sleep(0.15)
        return op._Bundle(FakeTokenizer(special={GPT_EOS}), {"model": GptSession([2001, GPT_EOS])})

    monkeypatch.setattr(op, "_load_bundle", slow)
    errs = []

    def work():
        try:
            op.gpt2_continue("hi there")
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and calls == [op.GPT2_MODEL]


def test_a_running_request_cannot_be_evicted_by_another_thread(monkeypatch):
    """The module lock serialises inference, so a model switch waits for the in-flight request."""
    events: List[str] = []
    gate = threading.Event()

    class SlowGpt(GptSession):
        def run(self, names, feeds):
            events.append("gpt-step")
            gate.wait(1.0)
            return super().run(names, feeds)

    def fake(name, kind):
        events.append(f"load-{name}")
        if name == op.GPT2_MODEL:
            return op._Bundle(FakeTokenizer(special={GPT_EOS}), {"model": SlowGpt([2001, GPT_EOS])})
        return op._Bundle(FakeTokenizer(wrap=((101,), (102,)), special={101, 102}), {"model": BertSession()})

    monkeypatch.setattr(op, "_load_bundle", fake)
    t1 = threading.Thread(target=lambda: op.gpt2_continue("hi"))
    t1.start()
    while "gpt-step" not in events:
        time.sleep(0.01)
    t2 = threading.Thread(target=lambda: op.bert_fill_mask("a [MASK]"))
    t2.start()
    time.sleep(0.2)
    assert "load-bert-base-uncased" not in events          # blocked while GPT-2 is mid-generation
    gate.set()
    t1.join()
    t2.join()
    assert events.index("load-bert-base-uncased") > events.index("gpt-step")


# ------------------------------------------------------------------ status / files / auto-fetch
def _touch_models(base):
    for rel in sum(op._FILES.values(), ()):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        if rel.endswith(".onnx"):
            (base / (rel + ".data")).write_bytes(b"x")


def test_backend_status_reports_missing_files_without_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", str(tmp_path))
    s = op.backend_status()
    assert s["ok"] is False and "model file" in s["detail"] and "bert/model.onnx" in s["missing"]
    assert op.loaded_models() == []
    _touch_models(tmp_path)
    s = op.backend_status()
    assert s["ok"] is True and s["missing"] == [] and s["detail"] is None


def test_missing_files_raise_a_clear_model_load_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("TRIOFORMER_AUTO_FETCH", raising=False)
    with pytest.raises(op.ModelLoadError, match="missing model files.*bert/model.onnx"):
        op.bert_fill_mask("a [MASK]")


def test_auto_fetch_downloads_only_the_needed_group_and_failures_are_reported(tmp_path, monkeypatch):
    from scripts import fetch_models

    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TRIOFORMER_AUTO_FETCH", "1")
    seen = {}

    def fake_fetch(manifest, dest, base_url=None, only=None, verify_only=False, retries=3, log=print):
        seen.update(only=only, dest=dest)
        return 1

    monkeypatch.setattr(fetch_models, "fetch", fake_fetch)
    with pytest.raises(op.ModelLoadError, match="automatic download"):
        op.gpt2_continue("hi")
    assert seen["only"] == ["gpt2"] and seen["dest"] == tmp_path


# ------------------------------------------------------------------ the module must stay torch-free
def test_module_imports_without_torch_or_transformers():
    code = ("import sys, api.onnx_pipelines, api.onnx_encoder; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','transformers','huggingface_hub')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ parity with the PyTorch reference API
def test_public_api_matches_the_pytorch_reference():
    pytest.importorskip("torch")
    import dataclasses
    import inspect

    from api import model_pipelines as ref

    for fn in ("bert_fill_mask", "gpt2_continue", "t5_summarize", "clear_model_cache", "loaded_models", "preload"):
        a, b = inspect.signature(getattr(ref, fn)), inspect.signature(getattr(op, fn))
        assert list(a.parameters) == list(b.parameters), fn
        assert [p.default for p in a.parameters.values() if fn != "preload"] == \
               [p.default for p in b.parameters.values() if fn != "preload"], fn
    for cls in ("MaskPrediction", "FillMaskResult", "GenerationResult", "SummaryResult"):
        assert [f.name for f in dataclasses.fields(getattr(ref, cls))] == \
               [f.name for f in dataclasses.fields(getattr(op, cls))], cls
    for const in ("BERT_MODEL", "GPT2_MODEL", "T5_MODEL", "MASK_TOKEN", "T5_PREFIX", "MAX_INPUT_CHARS",
                  "BERT_MAX_TOKENS", "GPT2_CONTEXT", "T5_MAX_INPUT_TOKENS", "MAX_NEW_TOKENS_LIMIT", "BERT_MAX_TOP_K"):
        assert getattr(ref, const) == getattr(op, const), const
    assert issubclass(op.InvalidInputError, ValueError) and issubclass(op.ModelLoadError, RuntimeError)
    assert not issubclass(op.ModelLoadError, op.InvalidInputError)


def test_validation_messages_match_the_pytorch_reference():
    pytest.importorskip("torch")
    from api import model_pipelines as ref

    cases = [(lambda m: m.bert_fill_mask("no mask"),), (lambda m: m.bert_fill_mask("[MASK]", top_k=0),),
             (lambda m: m.gpt2_continue("  "),), (lambda m: m.gpt2_continue("hi", max_new_tokens=999),),
             (lambda m: m.t5_summarize("x", max_new_tokens=3, min_new_tokens=9),), (lambda m: m.bert_fill_mask(None),)]
    for (call,) in cases:
        msgs = []
        for mod in (ref, op):
            with pytest.raises(mod.InvalidInputError) as ei:
                call(mod)
            msgs.append(str(ei.value))
        assert msgs[0] == msgs[1]
