"""Tests for the TrioFormer endpoints.

Everything external is faked: the Supabase layer is an in-memory stub and the
pretrained models are the tiny fakes from test_model_pipelines.py, driven through
the REAL pipeline code. No weights are downloaded and no network is used.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import db
from api import main as main_mod
from api import model_pipelines as mp
from tests.fakes import SECRET, FakeDb, iso, sha  # noqa: F401
from tests.test_model_pipelines import EOS, FakeBert, FakeGenerator, FakeTokenizer

@pytest.fixture
def fdb(monkeypatch) -> FakeDb:
    f = FakeDb()
    for name in ("ping", "insert_inference", "get_benchmarks", "get_benchmark_status", "refresh_benchmarks"):
        monkeypatch.setattr(db, name, getattr(f, name))
    return f


@pytest.fixture
def loader(monkeypatch):
    """Fake model loader; counts calls and can be told to fail."""
    mp.clear_model_cache()
    state = SimpleNamespace(calls=[], fail=False, delay=0.0)

    def fake_load(name, kind):
        state.calls.append(name)
        if state.delay:
            time.sleep(state.delay)
        if state.fail:
            raise OSError(f"cannot reach hub, token={SECRET}")
        model = {
            mp.BERT_MODEL: FakeBert(),
            mp.GPT2_MODEL: FakeGenerator([2001, 2002, EOS], seq2seq=False),
            mp.T5_MODEL: FakeGenerator([3001, 3002, EOS], seq2seq=True),
        }[name]
        return mp._Bundle(FakeTokenizer(), model)

    monkeypatch.setattr(mp, "_load_bundle", fake_load)
    yield state
    mp.clear_model_cache()


@pytest.fixture
def api(fdb, loader, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", SECRET)
    return TestClient(main_mod.app)


# ------------------------------------------------------------ happy paths
BERT_BODY = {"text": "the capital of france is [MASK] .", "top_k": 3}
GPT_BODY = {"prompt": "once upon a time", "max_new_tokens": 10}
T5_BODY = {"text": "the council approved a new bike lane network downtown", "max_new_tokens": 20}


@pytest.mark.parametrize(
    "path,body,text_key,model",
    [
        ("/bert_fillmask", BERT_BODY, "text", "prajjwal1/bert-small"),
        ("/gpt2_continue", GPT_BODY, "prompt", "erwanf/gpt2-mini"),
        ("/t5_summarize", T5_BODY, "text", "t5-small"),
    ],
)
def test_inference_is_logged_before_success_and_matches_response(api, fdb, path, body, text_key, model):
    r = api.post(path, json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == model and j["cold_start"] is True
    assert j["input_sha256"] == sha(body[text_key])
    assert isinstance(j["latency_ms"], int) and j["latency_ms"] >= 0

    assert len(fdb.inferences) == 1
    row = fdb.inferences[0]
    assert row["id"] == j["inference_id"]
    assert row["input_sha256"] == sha(body[text_key]) and row["input_sha256"] in fdb.inputs
    assert row["model_name"] == model
    assert row["latency_ms"] == j["latency_ms"]
    assert row["output"].strip()
    # privacy: the raw prompt is stored nowhere
    stored = " ".join(str(v) for v in row.values()) + " ".join(fdb.inputs)
    assert body[text_key] not in stored


def test_bert_response_shape_and_persisted_output_excludes_prompt(api, fdb):
    j = api.post("/bert_fillmask", json=BERT_BODY).json()
    assert [p["token"] for p in j["predictions"]] == ["paris", "ing", "<9>"]
    assert j["output_text"] == "the capital of france is paris ."
    assert j["input_tokens"] == 7
    out = fdb.inferences[0]["output"]
    assert out.startswith("paris (") and "capital" not in out  # tokens+scores only


def test_gpt2_returns_only_continuation(api):
    j = api.post("/gpt2_continue", json=GPT_BODY).json()
    assert "once" not in j["output_text"] and j["prompt_tokens"] == 4 and j["generated_tokens"] == 3


def test_t5_response_counts(api):
    j = api.post("/t5_summarize", json=T5_BODY).json()
    assert j["input_tokens"] == 10 and j["output_tokens"] == 3  # 9 words + prefix; 2 words + EOS


def test_cold_start_flag_and_load_time_excluded_from_latency(api, loader):
    loader.delay = 0.4  # model load takes 400 ms
    first = api.post("/gpt2_continue", json=GPT_BODY).json()
    second = api.post("/gpt2_continue", json=GPT_BODY).json()
    assert first["cold_start"] is True and second["cold_start"] is False
    assert first["latency_ms"] < 300  # would be >= 400 if loading were timed
    assert loader.calls == ["erwanf/gpt2-mini"]   # loaded once


def test_latency_is_measured_with_perf_counter(api, fdb, monkeypatch):
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(main_mod, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    j = api.post("/gpt2_continue", json=GPT_BODY).json()
    assert j["latency_ms"] == 250 and fdb.inferences[0]["latency_ms"] == 250


# ----------------------------------------------------------------- hashing
def test_sha256_is_consistent_exact_and_shared_across_models(api, fdb):
    text = "Héllo wörld"
    assert main_mod.prompt_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert main_mod.prompt_sha256(text) != main_mod.prompt_sha256(text + " ")  # no normalisation
    a = api.post("/gpt2_continue", json={"prompt": text}).json()["input_sha256"]
    b = api.post("/t5_summarize", json={"text": text}).json()["input_sha256"]
    c = api.post("/gpt2_continue", json={"prompt": text}).json()["input_sha256"]
    assert a == b == c == sha(text) and len(a) == 64
    assert len(fdb.inputs) == 1 and len(fdb.inferences) == 3  # one input row, three inferences


# -------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "path,body",
    [
        ("/bert_fillmask", {"text": "no mask here"}),
        ("/bert_fillmask", {"text": "two [MASK] and [MASK]"}),
        ("/bert_fillmask", {"text": "a [MASK]", "top_k": 0}),
        ("/bert_fillmask", {"text": "a [MASK]", "top_k": 11}),
        ("/bert_fillmask", {"text": ""}),
        ("/bert_fillmask", {}),
        ("/gpt2_continue", {"prompt": "   "}),
        ("/gpt2_continue", {"prompt": "hi", "max_new_tokens": 0}),
        ("/gpt2_continue", {"prompt": "hi", "max_new_tokens": 129}),
        ("/gpt2_continue", {"prompt": "x" * 4001}),
        ("/t5_summarize", {"text": ""}),
        ("/t5_summarize", {"text": "hi", "max_new_tokens": "many"}),
    ],
)
def test_schema_validation_returns_422_without_loading_or_logging(api, fdb, loader, path, body):
    r = api.post(path, json=body)
    assert r.status_code == 422
    assert loader.calls == [] and fdb.inferences == []


def test_pipeline_level_validation_returns_422_and_logs_nothing(api, fdb):
    r = api.post("/gpt2_continue", json={"prompt": "a " * 1000, "max_new_tokens": 40})
    assert r.status_code == 422 and "context window" in r.json()["detail"]
    r = api.post("/t5_summarize", json={"text": "a b c", "max_new_tokens": 5, "min_new_tokens": 9})
    assert r.status_code == 422 and "must not exceed" in r.json()["detail"]
    r = api.post("/t5_summarize", json={"text": "a " * 600})
    assert r.status_code == 422 and "too long" in r.json()["detail"]
    assert fdb.inferences == []


# ------------------------------------------------------------ failure modes
def test_database_write_failure_returns_503_and_withholds_result(api, fdb):
    fdb.fail_insert = True
    r = api.post("/gpt2_continue", json=GPT_BODY)
    assert r.status_code == 503
    body = r.text
    assert "could not be recorded" in r.json()["detail"]
    assert SECRET not in body and "2001" not in body and "output_text" not in body
    fdb.fail_insert = False
    assert api.post("/gpt2_continue", json=GPT_BODY).status_code == 200  # recovers


def test_model_load_failure_returns_sanitized_503_and_logs_nothing(api, fdb, loader):
    loader.fail = True
    r = api.post("/t5_summarize", json=T5_BODY)
    assert r.status_code == 503
    assert r.json()["detail"] == "Model 't5-small' is currently unavailable (failed to load)."
    assert SECRET not in r.text and "OSError" not in r.text and "hub" not in r.text
    assert fdb.inferences == []
    loader.fail = False  # failure was not cached -> retry works
    assert api.post("/t5_summarize", json=T5_BODY).status_code == 200


def test_busy_server_returns_503_with_retry_after(api, monkeypatch):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()  # someone else holds the only slot
    monkeypatch.setattr(main_mod, "_inference_slots", sem)
    monkeypatch.setattr(main_mod, "_QUEUE_TIMEOUT_SECONDS", 0.05)
    r = api.post("/gpt2_continue", json=GPT_BODY)
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_db_layer_sanitizes_errors():
    def boom():
        raise RuntimeError(f"connection failed: Authorization: Bearer {SECRET}")

    with pytest.raises(db.DatabaseError) as ei:
        db._call("insert_inference", boom)
    assert SECRET not in str(ei.value) and "insert_inference" in str(ei.value)

    def missing():
        raise KeyError("SUPABASE_URL")

    with pytest.raises(db.DatabaseError, match="not configured"):
        db._call("x", missing)


# ------------------------------------------------------- encoder attention
def test_encoder_attention_shapes_normalisation_and_disclaimer(api, fdb):
    r = api.post("/encoder_attention", json={"text": "the cat sat on the mat", "num_heads": 4})
    assert r.status_code == 200
    j = r.json()
    assert j["tokens"] == ["the", "cat", "sat", "on", "the", "mat"]
    assert j["attention_shape"] == [4, 6, 6] and j["num_heads"] == 4 and j["head_dim"] == 8
    assert len(j["attention"]) == 4 and all(len(h) == 6 and all(len(row) == 6 for row in h) for h in j["attention"])
    for head in j["attention"]:
        for row in head:
            assert sum(row) == pytest.approx(1.0, abs=1e-5) and min(row) >= 0
    for row in j["mean_attention"]:
        assert sum(row) == pytest.approx(1.0, abs=1e-5)
    assert j["max_row_sum_error"] < 1e-5
    assert "untrained" in j["disclaimer"] and "NOT" in j["disclaimer"]
    assert fdb.inferences == []  # not persisted


@pytest.mark.parametrize("heads", [1, 2])
def test_encoder_attention_head_counts(api, heads):
    j = api.post("/encoder_attention", json={"text": "a b c", "num_heads": heads}).json()
    assert j["attention_shape"] == [heads, 3, 3]


def test_encoder_attention_is_deterministic_and_positional_encoding_matters(api):
    body = {"text": "to be or not to be", "num_heads": 2}
    a = api.post("/encoder_attention", json=body).json()["attention"]
    assert a == api.post("/encoder_attention", json=body).json()["attention"]
    no_pe = api.post("/encoder_attention", json={**body, "use_positional_encoding": False}).json()["attention"]
    assert a != no_pe
    # without positional encoding, identical words ("to", "to") attend identically
    to0, to4 = no_pe[0][0], no_pe[0][4]
    assert to0 == pytest.approx(to4, abs=1e-6)


@pytest.mark.parametrize(
    "body",
    [{"text": "a b", "num_heads": 3}, {"text": "   "}, {"text": ""}, {"text": " ".join(["w"] * 33)}],
)
def test_encoder_attention_validation(api, body):
    assert api.post("/encoder_attention", json=body).status_code == 422


# ----------------------------------------------------------------- health
def test_healthz_reports_components_separately_without_loading_models(api, loader):
    j = api.get("/healthz").json()
    assert j == {"status": "ok", "model_loader": True, "supabase": True,
                 "models_loaded": [], "model_loader_detail": None, "backend": "torch"}
    assert loader.calls == []  # no download forced
    api.post("/gpt2_continue", json=GPT_BODY)
    assert api.get("/healthz").json()["models_loaded"] == ["erwanf/gpt2-mini"]


def test_healthz_database_down_is_degraded_but_models_ok(api, fdb):
    fdb.fail_ping = True
    j = api.get("/healthz").json()
    assert api.get("/healthz").status_code == 200
    assert (j["status"], j["supabase"], j["model_loader"]) == ("degraded", False, True)


def test_healthz_missing_transformers_is_reported(api, monkeypatch):
    import importlib.util as iu

    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None if name == "transformers" else real(name, *a, **k))
    j = api.get("/healthz").json()
    assert (j["status"], j["supabase"], j["model_loader"]) == ("degraded", True, False)
    assert "transformers" in j["model_loader_detail"]


def test_version_endpoint(api):
    j = api.get("/version").json()
    assert j["torch_version"] and j["git_sha"] and j["app_version"] == "2.0.0"
    assert j["checkpoints"] == {"/bert_fillmask": "prajjwal1/bert-small", "/gpt2_continue": "erwanf/gpt2-mini",
                                "/t5_summarize": "t5-small"}


# -------------------------------------------------------------- benchmarks
def test_benchmarks_returns_rows_and_refresh_metadata(api, fdb):
    r = api.get("/benchmarks")
    assert r.status_code == 200
    j = r.json()
    assert [b["model_name"] for b in j["benchmarks"]] == ["erwanf/gpt2-mini", "t5-small"]
    g = j["benchmarks"][0]
    assert g["p95_latency_ms"] == 150.0 and g["n_inferences"] == 4
    ref = j["refresh"]
    assert ref["last_mode"] == "concurrent" and ref["refreshed_now"] is False and ref["stale"] is False
    assert ref["ttl_seconds"] == 300 and 0 <= ref["age_seconds"] < 60
    assert "perf_counter" in j["latency_definition"] and "excludes model loading" in j["latency_definition"].lower()
    assert fdb.refresh_calls == 0  # fresh data -> no refresh


def test_benchmarks_refreshes_when_never_refreshed(api, fdb):
    fdb.status = {"last_refreshed_at": None, "last_duration_ms": None, "last_row_count": None, "last_mode": None}
    j = api.get("/benchmarks").json()
    assert fdb.refresh_calls == 1 and j["refresh"]["refreshed_now"] is True and j["refresh"]["stale"] is False


def test_benchmarks_refreshes_when_older_than_ttl(api, fdb):
    fdb.status["last_refreshed_at"] = iso(1000)
    j = api.get("/benchmarks").json()
    assert fdb.refresh_calls == 1 and j["refresh"]["refreshed_now"] is True


def test_forced_refresh_is_rate_limited(api, fdb):
    api.get("/benchmarks?refresh=true")           # status is 5 s old -> too recent, ignored
    assert fdb.refresh_calls == 0
    fdb.status["last_refreshed_at"] = iso(60)     # 60 s old -> allowed
    assert api.get("/benchmarks?refresh=true").json()["refresh"]["refreshed_now"] is True
    assert fdb.refresh_calls == 1


def test_inference_does_not_refresh_the_view(api, fdb):
    for _ in range(3):
        api.post("/gpt2_continue", json=GPT_BODY)
    assert fdb.refresh_calls == 0


def test_refresh_failure_serves_stale_data_with_note(api, fdb):
    fdb.status["last_refreshed_at"] = iso(1000)
    fdb.fail_refresh = True
    r = api.get("/benchmarks")
    assert r.status_code == 200
    ref = r.json()["refresh"]
    assert ref["stale"] is True and ref["refreshed_now"] is False and "Refresh failed" in ref["note"]
    assert len(r.json()["benchmarks"]) == 2


def test_refresh_already_running_is_reported(api, fdb):
    fdb.status["last_refreshed_at"] = iso(1000)
    fdb.refresh_result = {"refreshed": False, "reason": "refresh_in_progress"}
    ref = api.get("/benchmarks").json()["refresh"]
    assert ref["refreshed_now"] is False and "already running" in ref["note"]


def test_benchmarks_database_outage_is_503(api, fdb):
    fdb.fail_read = True
    r = api.get("/benchmarks")
    assert r.status_code == 503 and SECRET not in r.text


# ----------------------------------------------------------------- openapi
def test_all_seven_endpoints_documented_with_examples(api):
    spec = api.get("/openapi.json").json()
    for path, method in [("/bert_fillmask", "post"), ("/gpt2_continue", "post"), ("/t5_summarize", "post"),
                         ("/encoder_attention", "post"), ("/benchmarks", "get"), ("/healthz", "get"),
                         ("/version", "get")]:
        op = spec["paths"][path][method]
        assert op.get("summary") or op.get("description")
    schemas = spec["components"]["schemas"]
    for name in ("BertFillMaskRequest", "Gpt2ContinueRequest", "T5SummarizeRequest", "EncoderAttentionRequest"):
        assert schemas[name]["examples"], name
        assert all(p.get("description") for p in schemas[name]["properties"].values()), name
    assert "503" in spec["paths"]["/gpt2_continue"]["post"]["responses"]


def test_switching_models_through_the_api_keeps_one_resident_and_flags_cold_start(api):
    seen = []
    for path, body in (("/bert_fillmask", BERT_BODY), ("/gpt2_continue", GPT_BODY), ("/t5_summarize", T5_BODY),
                       ("/t5_summarize", T5_BODY), ("/bert_fillmask", BERT_BODY)):
        j = api.post(path, json=body).json()
        seen.append((j["model"], j["cold_start"]))
        assert len(api.get("/healthz").json()["models_loaded"]) == 1
    assert seen == [("prajjwal1/bert-small", True), ("erwanf/gpt2-mini", True), ("t5-small", True),
                    ("t5-small", False), ("prajjwal1/bert-small", True)]
