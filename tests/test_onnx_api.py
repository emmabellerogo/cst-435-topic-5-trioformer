"""The seven endpoints running on the ONNX backend (TRIOFORMER_BACKEND=onnx), using scripted fakes
for the model sessions and an in-memory stand-in for Supabase. Nothing here touches real weights,
the network or a database; real-model behaviour is covered by tests/test_onnx_parity.py (opt-in)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import db
from api import main as main_mod
from api import onnx_encoder as oe
from api import onnx_pipelines as op
from tests.test_onnx_pipelines import BertSession, FakeTokenizer, GptSession, GPT_EOS, t5_bundle
from tests.fakes import SECRET, FakeDb, sha


@pytest.fixture
def fdb(monkeypatch):
    f = FakeDb()
    for name in ("ping", "insert_inference", "get_benchmarks", "get_benchmark_status", "refresh_benchmarks"):
        monkeypatch.setattr(db, name, getattr(f, name))
    return f


@pytest.fixture
def onnx_api(fdb, monkeypatch):
    op.clear_model_cache()
    state = type("S", (), {"calls": [], "fail": False})()

    def fake(name, kind):
        state.calls.append(name)
        if state.fail:
            raise OSError(f"cannot read /secret/models/dir token={SECRET}")
        if name == op.BERT_MODEL:
            return op._Bundle(FakeTokenizer(wrap=((101,), (102,)), special={101, 102}), {"model": BertSession()})
        if name == op.GPT2_MODEL:
            return op._Bundle(FakeTokenizer(special={GPT_EOS}), {"model": GptSession([2001, 2002, GPT_EOS])})
        return t5_bundle({0: {5: 3.0}, 5: {6: 3.0}, 6: {1: 3.0}})[0]

    monkeypatch.setattr(op, "_load_bundle", fake)
    monkeypatch.setattr(main_mod, "BACKEND", "onnx")
    monkeypatch.setattr(main_mod, "mp", op)
    monkeypatch.setattr(main_mod, "encoder_backend", oe)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", SECRET)
    yield TestClient(main_mod.app), state, fdb
    op.clear_model_cache()


def test_backend_selection_function():
    pytest.importorskip("torch")
    mp_onnx, enc_onnx = main_mod._load_backend("onnx")
    assert mp_onnx is op and enc_onnx is oe
    mp_torch, enc_torch = main_mod._load_backend("torch")
    assert mp_torch.__name__ == "api.model_pipelines" and enc_torch.__name__ == "api.encoder_demo"
    with pytest.raises(RuntimeError, match="must be 'onnx' or 'torch'"):
        main_mod._load_backend("tensorflow")


def test_environment_variable_selects_the_backend_at_import():
    code = "import api.main as m; print(m.BACKEND, m.mp.__name__)"
    for value, expect in (("onnx", "onnx api.onnx_pipelines"), ("ONNX ", "onnx api.onnx_pipelines")):
        env = {**__import__("os").environ, "TRIOFORMER_BACKEND": value}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == expect, r.stderr[-400:]
    bad = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "TRIOFORMER_BACKEND": "nope"})
    assert bad.returncode != 0 and "must be 'onnx' or 'torch'" in bad.stderr


@pytest.mark.parametrize("path,body,key,model", [
    ("/bert_fillmask", {"text": "the capital of france is [MASK] .", "top_k": 3}, "text", "bert-base-uncased"),
    ("/gpt2_continue", {"prompt": "once upon a time", "max_new_tokens": 10}, "prompt", "gpt2"),
    ("/t5_summarize", {"text": "the council approved a bike lane network"}, "text", "t5-small"),
])
def test_model_endpoints_work_and_log_on_the_onnx_backend(onnx_api, path, body, key, model):
    api, state, fdb = onnx_api
    r = api.post(path, json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == model and j["cold_start"] is True and j["input_sha256"] == sha(body[key])
    row = fdb.inferences[-1]
    assert row["model_name"] == model and row["latency_ms"] == j["latency_ms"] and row["input_sha256"] == sha(body[key])
    assert body[key] not in " ".join(str(v) for v in row.values())          # raw prompt never stored
    assert api.post(path, json=body).json()["cold_start"] is False


def test_switching_models_keeps_only_one_resident_and_health_shows_it(onnx_api):
    api, state, _ = onnx_api
    assert api.get("/healthz").json()["models_loaded"] == []
    api.post("/bert_fillmask", json={"text": "a [MASK]"})
    assert api.get("/healthz").json()["models_loaded"] == ["bert-base-uncased"]
    api.post("/gpt2_continue", json={"prompt": "hi"})
    assert api.get("/healthz").json()["models_loaded"] == ["gpt2"]
    api.post("/t5_summarize", json={"text": "hi there"})
    assert api.get("/healthz").json()["models_loaded"] == ["t5-small"]
    api.post("/gpt2_continue", json={"prompt": "hi"})
    assert api.post("/gpt2_continue", json={"prompt": "hi"}).json()["cold_start"] is False
    assert state.calls == ["bert-base-uncased", "gpt2", "t5-small", "gpt2"]


def test_model_load_failure_is_a_sanitized_503_and_nothing_is_logged(onnx_api):
    api, state, fdb = onnx_api
    state.fail = True
    r = api.post("/gpt2_continue", json={"prompt": "hi"})
    assert r.status_code == 503 and r.json()["detail"] == "Model 'gpt2' is currently unavailable (failed to load)."
    assert SECRET not in r.text and "/secret" not in r.text and "OSError" not in r.text
    assert fdb.inferences == []
    state.fail = False
    assert api.post("/gpt2_continue", json={"prompt": "hi"}).status_code == 200


def test_validation_errors_are_422_without_loading_a_model(onnx_api):
    api, state, fdb = onnx_api
    for path, body in [("/bert_fillmask", {"text": "no mask"}), ("/gpt2_continue", {"prompt": "hi", "max_new_tokens": 0})]:
        assert api.post(path, json=body).status_code == 422
    assert api.post("/gpt2_continue", json={"prompt": "a " * 1000, "max_new_tokens": 40}).status_code == 422
    assert state.calls == [op.GPT2_MODEL] and fdb.inferences == []     # only the token-length check loaded GPT-2


def test_database_outage_withholds_the_result(onnx_api):
    api, _, fdb = onnx_api
    fdb.fail_insert = True
    r = api.post("/gpt2_continue", json={"prompt": "hi"})
    assert r.status_code == 503 and "output_text" not in r.text


@pytest.mark.skipif(not oe.available(), reason="models/edu not present")
def test_encoder_attention_endpoint_runs_without_torch_backend(onnx_api):
    api, _, fdb = onnx_api
    j = api.post("/encoder_attention", json={"text": "the cat sat on the mat", "num_heads": 4}).json()
    arr = np.array(j["attention"])
    assert j["attention_shape"] == [4, 6, 6] and np.allclose(arr.sum(-1), 1.0, atol=1e-5)
    assert "untrained" in j["disclaimer"] and fdb.inferences == []
    assert api.post("/encoder_attention", json={"text": "a b", "num_heads": 3}).status_code == 422


def test_encoder_files_missing_is_a_503(onnx_api, monkeypatch, tmp_path):
    api, _, _ = onnx_api
    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(oe, "_SESSIONS", {})
    monkeypatch.setattr(oe, "_EMBEDDING", [])
    r = api.post("/encoder_attention", json={"text": "hello world"})
    assert r.status_code == 503 and str(tmp_path) not in r.text


def test_healthz_reports_missing_model_files_separately_from_supabase(onnx_api, monkeypatch, tmp_path):
    api, state, fdb = onnx_api
    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", str(tmp_path))
    j = api.get("/healthz").json()
    assert (j["status"], j["supabase"], j["model_loader"], j["backend"]) == ("degraded", True, False, "onnx")
    assert "model file" in j["model_loader_detail"] and state.calls == []         # checked without loading anything
    from tests.test_onnx_pipelines import _touch_models

    _touch_models(tmp_path)
    j = api.get("/healthz").json()
    assert (j["status"], j["model_loader"], j["model_loader_detail"]) == ("ok", True, None)
    fdb.fail_ping = True
    j = api.get("/healthz").json()
    assert (j["status"], j["supabase"], j["model_loader"]) == ("degraded", False, True)


def test_version_reports_backend_and_runtime(onnx_api):
    j = onnx_api[0].get("/version").json()
    assert j["backend"] == "onnx" and j["onnxruntime_version"] and j["torch_version"]
    assert j["checkpoints"]["/gpt2_continue"] == "gpt2"


def test_legacy_training_routes_degrade_cleanly_without_torch(monkeypatch):
    import api as api_pkg
    from fastapi import HTTPException

    monkeypatch.delattr(api_pkg, "training", raising=False)
    monkeypatch.setitem(sys.modules, "api.training", None)       # simulates "torch not installed"
    with pytest.raises(HTTPException) as ei:
        main_mod._training()
    assert ei.value.status_code == 503 and "PyTorch" in ei.value.detail
