"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the Transformer classifier behind HTTP so the Streamlit UI can call it
    over HTTPS,
  * read dataset parameters from Supabase before training,
  * write run rows, model artifacts, and sequence metadata (hash only) back.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Former-It": sequence classification
with a from-scratch tiny Transformer encoder that returns per-head self-attention
matrices so a client can see *which positions attend to which*.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Iterator, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import db
from shared.data import CLASS_NAMES, N_CLASSES, VOCAB_SIZE, generate_one
from shared.schemas import (
    BenchmarkRefreshInfo,
    BenchmarkRow,
    BenchmarksResponse,
    BertFillMaskRequest,
    BertFillMaskResponse,
    ClassesResponse,
    ClassMetrics,
    Dataset,
    DatasetCreate,
    EncoderAttentionRequest,
    EncoderAttentionResponse,
    Gpt2ContinueRequest,
    Gpt2ContinueResponse,
    Health,
    MaskPredictionOut,
    PredictRequest,
    PredictResponse,
    PredictSampleRequest,
    Run,
    T5SummarizeRequest,
    T5SummarizeResponse,
    TrainRequest,
    TrainResponse,
    Version,
)

logger = logging.getLogger("trioformer")

# ---------------------------------------------------------------------------
# Inference backend  (TRIOFORMER_BACKEND = "onnx" | "torch")
#   onnx  - api/onnx_pipelines.py + api/onnx_encoder.py: int8 ONNX Runtime, memory-mapped weights,
#           one model resident at a time, NO torch/transformers (what runs on Render Free).
#   torch - api/model_pipelines.py + api/encoder_demo.py: the PyTorch reference implementation.
# Both expose the same function names, result types and errors. If the variable is unset the
# reference ("torch") is used when torch AND transformers are installed (local development),
# otherwise the light "onnx" backend. Production sets it explicitly (render.yaml).
# ---------------------------------------------------------------------------
BACKEND = (
    os.environ.get("TRIOFORMER_BACKEND", "").strip().lower()
    or ("torch" if all(importlib.util.find_spec(m) for m in ("torch", "transformers")) else "onnx")
)


def _load_backend(name: str):
    if name == "onnx":
        from api import onnx_encoder, onnx_pipelines

        return onnx_pipelines, onnx_encoder
    if name == "torch":
        try:
            from api import encoder_demo, model_pipelines
        except ImportError as exc:  # torch / transformers missing
            raise RuntimeError(
                "TRIOFORMER_BACKEND=torch needs PyTorch and Transformers "
                "(pip install -r requirements-export.txt). Use TRIOFORMER_BACKEND=onnx for the light runtime."
            ) from exc
        return model_pipelines, encoder_demo
    raise RuntimeError(f"TRIOFORMER_BACKEND must be 'onnx' or 'torch', got {name!r}")


mp, encoder_backend = _load_backend(BACKEND)

app = FastAPI(
    title="TrioFormer API",
    description=(
        "Side-by-side Transformer inference: pretrained BERT (fill-mask), GPT-2 (continuation) "
        "and T5 (summarization), plus educational attention heatmaps from our own encoder, with "
        "server-side latency benchmarks persisted in Supabase.\n\n"
        "The original Former-It endpoints (/datasets, /train, /runs, /predict, /predict_sample, "
        "/classes) are kept under the `legacy` tag.\n\n"
        "**Errors**: 422 = invalid input (`{\"detail\": ...}`); 503 = a model could not be loaded, "
        "the database is unavailable, or the server is busy (see `Retry-After`)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _training():
    """Former-It training/prediction needs PyTorch; the light ONNX deployment does not ship it."""
    try:
        from api import training
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="This endpoint needs PyTorch, which is not installed in this deployment "
                   f"(backend: {BACKEND}).",
        ) from exc
    return training


def _artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


def _seq_sha256(sequence: list[int]) -> str:
    """Stable hash of the symbol sequence (no raw symbols are ever stored)."""
    return hashlib.sha256(json.dumps(sequence).encode()).hexdigest()


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["legacy"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Register a synthetic sequence dataset (parameters only; symbols come from seed)."""
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        seq_len=req.seq_len,
        vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES,
        seed=req.seed,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["legacy"])
def train(req: TrainRequest) -> TrainResponse:
    """Read dataset params from Supabase, train the model, persist run + artifact."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    metrics, model_b64, _loss = _training().train_transformer(
        n_rows=dataset["n_rows"],
        seq_len=dataset["seq_len"],
        seed=dataset["seed"],
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        test_size=req.test_size,
    )
    run = db.insert_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        metrics=metrics,
        model_b64=model_b64,
    )
    return TrainResponse(run_id=run["id"], metrics=ClassMetrics(**metrics))


@app.get("/runs/{run_id}", response_model=Run, tags=["legacy"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**{k: row[k] for k in Run.model_fields})


@app.get("/runs", response_model=list[Run], tags=["legacy"])
def list_runs() -> list[Run]:
    rows = db.latest_runs(limit=50)
    return [Run(**{k: r[k] for k in Run.model_fields}) for r in rows]


# ---------------------------------------------------------------------------
# prediction  (per-head self-attention explainability)
# ---------------------------------------------------------------------------
def _predict_sequence(
    run_id: int, sequence: list[int], true_class: str | None
) -> PredictResponse:
    model_b64 = _artifact_or_404(run_id)
    result = _training().predict_seq(model_b64, sequence)
    sha256 = _seq_sha256(sequence)
    db.insert_sequence_metadata(
        run_id=run_id,
        sha256=sha256,
        length=len(sequence),
        label=result["label"],
        class_name=result["class_name"],
        confidence=result["confidence"],
    )
    return PredictResponse(
        run_id=run_id,
        sequence=sequence,
        sequence_sha256=sha256,
        true_class=true_class,
        **result,
    )


@app.post("/predict", response_model=PredictResponse, tags=["legacy"])
def predict(req: PredictRequest) -> PredictResponse:
    """Classify a symbol sequence and return per-layer, per-head attention.

    Only the sequence's sha256 hash + length + prediction are logged -- never
    the raw symbols.
    """
    if any(t < 0 or t >= VOCAB_SIZE for t in req.sequence):
        raise HTTPException(
            status_code=422, detail=f"symbol ids must be in [0, {VOCAB_SIZE})"
        )
    return _predict_sequence(req.run_id, req.sequence, true_class=None)


@app.post("/predict_sample", response_model=PredictResponse, tags=["legacy"])
def predict_sample(req: PredictSampleRequest) -> PredictResponse:
    """Generate a synthetic sequence, classify it, and return attention matrices.

    Lets the UI demonstrate the model without the user typing a sequence.
    """
    kind = (
        req.true_class
        if (req.true_class is None or req.true_class < N_CLASSES)
        else None
    )
    seq, label = generate_one(seed=req.seed, kind=kind)
    return _predict_sequence(
        req.run_id, [int(t) for t in seq], true_class=CLASS_NAMES[label]
    )


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["legacy"])
def classes() -> ClassesResponse:
    """Return the class labels the model predicts over."""
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


# ===========================================================================
# TrioFormer: pretrained-model endpoints
# ===========================================================================
# Latency definition (persisted and shown by /benchmarks):
#   INCLUDED : tokenization + model forward/generation + decoding, measured with
#              time.perf_counter() around the pipeline call.
#   EXCLUDED : model loading (done just before the timer; reported as
#              `cold_start`), waiting for a free inference slot, the Supabase
#              write, JSON serialisation and network time.
LATENCY_DEFINITION = (
    "latency_ms = server-side time.perf_counter() around the pipeline call: tokenization + model "
    "forward/generation + decoding. Excludes model loading (first request per model is flagged "
    "cold_start), queueing for the inference slot, the database write and network transfer."
)

BENCHMARK_TTL_SECONDS = int(os.environ.get("BENCHMARK_REFRESH_TTL_SECONDS", "300"))
BENCHMARK_MIN_FORCE_SECONDS = 10  # `?refresh=true` is ignored if refreshed more recently than this
_QUEUE_TIMEOUT_SECONDS = float(os.environ.get("TRIOFORMER_QUEUE_TIMEOUT_SECONDS", "25"))
# Free-tier hosts have little RAM/CPU: run model inference one request at a time.
_inference_slots = threading.BoundedSemaphore(
    max(1, int(os.environ.get("TRIOFORMER_MAX_CONCURRENT_INFERENCES", "1")))
)

_RESP_ERRORS = {
    422: {"description": "Invalid input. Body: {\"detail\": \"...\"}"},
    503: {"description": "Model failed to load, database unavailable (inference NOT recorded), or server busy."},
}


def prompt_sha256(text: str) -> str:
    """SHA-256 hex digest of the exact submitted text (UTF-8, no normalisation).

    Used as the primary key of ``inputs``; the raw text is never stored. The same
    text yields the same hash for every endpoint and every model.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _inference_slot() -> Iterator[None]:
    if not _inference_slots.acquire(timeout=_QUEUE_TIMEOUT_SECONDS):
        raise HTTPException(
            status_code=503,
            detail="Server is busy with another inference. Please retry shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        _inference_slots.release()


def _serve_pretrained(
    model_name: str, text: str, run: Callable[[], object], db_output: Callable[[object], str]
) -> Tuple[object, dict]:
    """Run a pipeline, persist the inference, and only then return.

    Order: take an inference slot -> load the model (untimed) -> time the pipeline
    call -> write ``inputs``/``inferences`` to Supabase -> return. If the write
    fails the result is withheld (503) so every successful response is on record.
    """
    sha = prompt_sha256(text)
    with _inference_slot():
        cold = model_name not in mp.loaded_models()
        try:
            mp.preload((model_name,))
        except mp.ModelLoadError as exc:
            logger.error("model load failed for %s: %s", model_name, exc)
            raise HTTPException(
                status_code=503, detail=f"Model '{model_name}' is currently unavailable (failed to load)."
            ) from exc
        t0 = time.perf_counter()
        try:
            result = run()
        except mp.InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except mp.ModelLoadError as exc:
            raise HTTPException(
                status_code=503, detail=f"Model '{model_name}' is currently unavailable (failed to load)."
            ) from exc
        latency_ms = max(0, int(round((time.perf_counter() - t0) * 1000)))
    try:
        row = db.insert_inference(
            input_sha256=sha, model_name=model_name, output=db_output(result), latency_ms=latency_ms
        )
    except db.DatabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail="The inference could not be recorded in the database, so the result was withheld. Please retry.",
        ) from exc
    meta = {
        "model": model_name,
        "input_sha256": sha,
        "latency_ms": latency_ms,
        "inference_id": row["id"],
        "served_at": row.get("served_at"),
        "cold_start": cold,
    }
    return result, meta


@app.post(
    "/bert_fillmask",
    response_model=BertFillMaskResponse,
    tags=["trioformer"],
    summary="BERT masked-token prediction",
    responses=_RESP_ERRORS,
)
def bert_fillmask(req: BertFillMaskRequest) -> BertFillMaskResponse:
    """Predict the token behind the single `[MASK]` using `bert-base-uncased`.

    Persists: SHA-256 of `text`, model name, and the top-k predicted tokens with
    scores (NOT the filled sentence, which would contain the prompt).
    """
    result, meta = _serve_pretrained(
        mp.BERT_MODEL,
        req.text,
        lambda: mp.bert_fill_mask(req.text, top_k=req.top_k),
        lambda r: ", ".join(f"{p.token} ({p.score:.3f})" for p in r.predictions),
    )
    return BertFillMaskResponse(
        **meta,
        output_text=result.output_text,
        predictions=[MaskPredictionOut(**asdict(p)) for p in result.predictions],
        input_tokens=result.input_tokens,
    )


@app.post(
    "/gpt2_continue",
    response_model=Gpt2ContinueResponse,
    tags=["trioformer"],
    summary="GPT-2 text continuation",
    responses=_RESP_ERRORS,
)
def gpt2_continue(req: Gpt2ContinueRequest) -> Gpt2ContinueResponse:
    """Continue `prompt` with greedy (deterministic) decoding using `gpt2`.

    `output_text` is only the new continuation. Persists the continuation, not the prompt.
    """
    result, meta = _serve_pretrained(
        mp.GPT2_MODEL,
        req.prompt,
        lambda: mp.gpt2_continue(req.prompt, max_new_tokens=req.max_new_tokens),
        lambda r: r.output_text,
    )
    return Gpt2ContinueResponse(
        **meta,
        output_text=result.output_text,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
    )


@app.post(
    "/t5_summarize",
    response_model=T5SummarizeResponse,
    tags=["trioformer"],
    summary="T5 summarization",
    responses=_RESP_ERRORS,
)
def t5_summarize(req: T5SummarizeRequest) -> T5SummarizeResponse:
    """Summarize `text` with `t5-small` (`summarize:` prefix, 4-beam search).

    t5-small is small: summaries are often extractive and may be imperfect.
    """
    result, meta = _serve_pretrained(
        mp.T5_MODEL,
        req.text,
        lambda: mp.t5_summarize(
            req.text, max_new_tokens=req.max_new_tokens, min_new_tokens=req.min_new_tokens
        ),
        lambda r: r.output_text,
    )
    return T5SummarizeResponse(
        **meta,
        output_text=result.output_text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ===========================================================================
# TrioFormer: custom encoder attention (educational; not persisted)
# ===========================================================================
@app.post(
    "/encoder_attention",
    response_model=EncoderAttentionResponse,
    tags=["trioformer"],
    summary="Educational attention heatmap from our custom encoder",
    responses={422: _RESP_ERRORS[422]},
)
def encoder_attention(req: EncoderAttentionRequest) -> EncoderAttentionResponse:
    """Per-head self-attention from `src/transformer.py` for the words of `text`.

    **Educational only**: an UNTRAINED encoder with fixed random weights. It shows
    the attention mechanism (and the effect of positional encoding), not how a
    pretrained language model reads text. Nothing is logged to the database.
    """
    try:
        out = encoder_backend.compute_attention(
            req.text, num_heads=req.num_heads, use_positional_encoding=req.use_positional_encoding
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, ImportError) as exc:  # ONNX files missing / onnxruntime unavailable
        logger.error("encoder backend unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="The educational encoder is currently unavailable.") from exc
    return EncoderAttentionResponse(**out)


# ===========================================================================
# TrioFormer: benchmarks
# ===========================================================================
def _parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@app.get(
    "/benchmarks",
    response_model=BenchmarksResponse,
    tags=["trioformer"],
    summary="Persisted per-model latency statistics",
    responses={503: _RESP_ERRORS[503]},
)
def benchmarks(
    refresh: bool = Query(
        False,
        description=(
            "Ask for a refresh of the materialized view now. Ignored if it was refreshed in the last "
            f"{BENCHMARK_MIN_FORCE_SECONDS}s. Without it, the view is refreshed automatically at most once per TTL."
        ),
    ),
) -> BenchmarksResponse:
    """Latency statistics (mean, min, p50, p95, p99, max) per model, from the `benchmarks` view.

    **Refresh strategy**: the view is NOT refreshed per inference. On read, if it
    was never refreshed or is older than the TTL (default 300 s), this endpoint
    calls the SQL function `refresh_benchmarks()`, which is advisory-locked (one
    refresh at a time) and uses `REFRESH ... CONCURRENTLY`. If the refresh fails,
    the last materialized data is returned with `refresh.stale = true`. New
    inferences therefore appear after the next refresh.
    """
    try:
        status = db.get_benchmark_status() or {}
        last = _parse_ts(status.get("last_refreshed_at"))
        now = datetime.now(timezone.utc)
        age = (now - last).total_seconds() if last else None
        due = age is None or age > BENCHMARK_TTL_SECONDS or (refresh and age > BENCHMARK_MIN_FORCE_SECONDS)

        refreshed_now, note = False, None
        if due:
            try:
                res = db.refresh_benchmarks() or {}
                if res.get("refreshed"):
                    refreshed_now = True
                    status = db.get_benchmark_status() or status
                elif res.get("reason") == "refresh_in_progress":
                    note = "A refresh is already running; serving the latest materialized data."
            except db.DatabaseError:
                note = "Refresh failed; serving the last materialized data."

        rows = db.get_benchmarks()
    except db.DatabaseError as exc:
        raise HTTPException(status_code=503, detail="Benchmark data is temporarily unavailable.") from exc

    last = _parse_ts(status.get("last_refreshed_at"))
    age = (datetime.now(timezone.utc) - last).total_seconds() if last else None
    return BenchmarksResponse(
        benchmarks=[BenchmarkRow(**{k: r.get(k) for k in BenchmarkRow.model_fields}) for r in rows],
        refresh=BenchmarkRefreshInfo(
            last_refreshed_at=last,
            last_duration_ms=status.get("last_duration_ms"),
            last_row_count=status.get("last_row_count"),
            last_mode=status.get("last_mode"),
            age_seconds=age,
            stale=age is None or age > BENCHMARK_TTL_SECONDS,
            refreshed_now=refreshed_now,
            ttl_seconds=BENCHMARK_TTL_SECONDS,
            note=note,
        ),
        latency_definition=LATENCY_DEFINITION,
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
def _dist_version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    """Report model-loader and Supabase health separately.

    Cheap by design: it never loads a model or downloads weights (the free tier
    health-check must stay fast and low-memory). `model_loader` means the selected
    backend is ready: for `onnx`, onnxruntime + tokenizers are installed and every model
    file is present on disk; for `torch`, `transformers` is installed. `models_loaded`
    lists what is resident in memory (at most one with the ONNX backend).
    """
    supabase_ok = db.ping()
    if BACKEND == "onnx":
        status_info = mp.backend_status()
        model_ok, detail = status_info["ok"], status_info["detail"]
    else:
        model_ok = importlib.util.find_spec("transformers") is not None
        detail = None if model_ok else "the `transformers` package is not installed"
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(
        status=status,
        model_loader=model_ok,
        supabase=supabase_ok,
        models_loaded=mp.loaded_models(),
        model_loader_detail=detail,
        backend=BACKEND,
    )


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=_dist_version("torch") or "not installed",
        supabase_project_ref=_project_ref(),
        transformers_version=_dist_version("transformers"),
        backend=BACKEND,
        onnxruntime_version=_dist_version("onnxruntime"),
        checkpoints={
            "/bert_fillmask": mp.BERT_MODEL,
            "/gpt2_continue": mp.GPT2_MODEL,
            "/t5_summarize": mp.T5_MODEL,
        },
    )
