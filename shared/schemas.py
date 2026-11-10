"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. "Former-It" is sequence classification
with a from-scratch tiny **Transformer encoder** (positional encoding +
multi-head self-attention); the per-head attention matrices are returned so a
client can see *which positions attend to which*. The *shape* of the contract is
identical to the base template; only the fields change.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(2000, ge=200, le=20_000)
    seq_len: int = Field(8, ge=4, le=32, description="Symbols per sequence (even).")
    seed: int = Field(42, description="Seed for the synthetic sequence generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    seq_len: int
    vocab_size: int
    n_classes: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train."""

    dataset_id: int
    lr: float = Field(0.005, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(20, ge=1, le=500)
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out fraction.")


class ClassMetrics(BaseModel):
    accuracy: float
    macro_f1: float


class Run(BaseModel):
    """A training-run row as stored in Supabase (metrics only)."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    accuracy: float
    macro_f1: float
    created_at: datetime


class TrainResponse(BaseModel):
    run_id: int
    metrics: ClassMetrics


# ---------------------------------------------------------------------------
# Prediction  (per-head self-attention explainability)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    run_id: int
    sequence: List[int] = Field(..., min_length=2, description="Symbol ids to classify.")


class PredictResponse(BaseModel):
    run_id: int
    label: int
    class_name: str
    confidence: float = Field(..., description="Softmax probability of the top class.")
    probs: List[float]
    sequence: List[int] = Field(..., description="The symbols the model scored.")
    n_layers: int
    n_heads: int
    attention: List[List[List[List[float]]]] = Field(
        ...,
        description="Per-layer, per-head [n_layers][n_heads][T][T] attention; each row sums to 1.",
    )
    sequence_sha256: str = Field(..., description="Hash of the input sequence (no raw symbols stored).")
    true_class: Optional[str] = Field(
        None, description="Ground-truth class when predicting a generated sample."
    )


class PredictSampleRequest(BaseModel):
    """Request body for POST /predict_sample (no sequence needed)."""

    run_id: int
    seed: int = Field(0, description="Seed for the generated sample.")
    true_class: Optional[int] = Field(
        None, ge=0, description="Force a class index; None picks one at random."
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    """GET /healthz. Never loads a model or downloads weights."""

    status: str = Field(..., description='"ok" if everything is healthy, else "degraded".')
    model_loader: bool = Field(
        ..., description="True if the Hugging Face `transformers` library is installed and the pipelines are importable."
    )
    supabase: bool = Field(..., description="True if a trivial Supabase query succeeded (checked separately from models).")
    models_loaded: List[str] = Field(
        default_factory=list,
        description="Checkpoint currently resident in memory (at most one; a different model evicts it). Empty on a cold start.",
    )
    model_loader_detail: Optional[str] = Field(None, description="Why model_loader is false, if it is.")
    backend: Optional[str] = Field(None, description='Inference backend in use (always "torch": plain PyTorch + Transformers).')

    model_config = ConfigDict(
        protected_namespaces=(),  # field names start with "model_"
        json_schema_extra={
            "examples": [
                {"status": "ok", "model_loader": True, "supabase": True,
                 "models_loaded": ["erwanf/gpt2-mini"], "model_loader_detail": None}
            ]
        }
    )


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
    app_version: str = Field("2.0.0", description="TrioFormer API version.")
    transformers_version: Optional[str] = Field(None, description="Installed `transformers` version, if any.")
    backend: Optional[str] = Field(None, description='Inference backend in use (always "torch").')
    checkpoints: Dict[str, str] = Field(
        default_factory=dict, description="Endpoint -> Hugging Face checkpoint used."
    )


# ===========================================================================
# TrioFormer endpoints
# ===========================================================================
class InferenceMeta(BaseModel):
    """Fields shared by every pretrained-model response."""

    model: str = Field(..., description="Hugging Face checkpoint that produced the result.")
    input_sha256: str = Field(
        ..., description="SHA-256 (hex) of the exact submitted text, UTF-8, no normalisation. The raw text is never stored."
    )
    latency_ms: int = Field(
        ...,
        description=(
            "Server-side time from `time.perf_counter()`, in whole ms, covering tokenization + model "
            "forward/generation + decoding. EXCLUDES model loading (see `cold_start`), queueing, the "
            "database write and network time. This is the value persisted for /benchmarks."
        ),
    )
    inference_id: int = Field(..., description="Primary key of the persisted row in `inferences`.")
    served_at: Optional[datetime] = Field(None, description="Database timestamp of the persisted inference.")
    cold_start: bool = Field(
        ..., description="True if this request had to load the model first (that load time is NOT in latency_ms). Models are evicted when another is requested, so switching models is a cold start."
    )


# ---- BERT -----------------------------------------------------------------
class BertFillMaskRequest(BaseModel):
    """POST /bert_fillmask body."""

    text: str = Field(..., min_length=1, max_length=4000, description="Sentence containing exactly one `[MASK]` (uppercase).")
    top_k: int = Field(5, ge=1, le=10, description="How many candidate tokens to return.")

    @field_validator("text")
    @classmethod
    def _exactly_one_mask(cls, v: str) -> str:
        n = v.count("[MASK]")
        if not v.strip():
            raise ValueError("text must not be blank")
        if n != 1:
            raise ValueError(f"text must contain exactly one [MASK], found {n}")
        return v

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": "The capital of France is [MASK].", "top_k": 5}]}
    )


class MaskPredictionOut(BaseModel):
    token: str
    score: float = Field(..., description="Softmax probability of this token at the masked position.")
    sequence: str = Field(..., description="Input with [MASK] replaced by this token.")


class BertFillMaskResponse(InferenceMeta):
    output_text: str = Field(..., description="Input with [MASK] replaced by the top prediction.")
    predictions: List[MaskPredictionOut] = Field(..., description="Candidates, best first.")
    input_tokens: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "prajjwal1/bert-small",
                    "input_sha256": "9f2c...64 hex chars...",
                    "latency_ms": 41,
                    "inference_id": 17,
                    "served_at": "2026-11-08T01:23:45.678+00:00",
                    "cold_start": False,
                    "output_text": "The capital of France is paris.",
                    "predictions": [
                        {"token": "paris", "score": 0.417, "sequence": "The capital of France is paris."},
                        {"token": "lille", "score": 0.071, "sequence": "The capital of France is lille."},
                    ],
                    "input_tokens": 9,
                }
            ]
        }
    )


# ---- GPT-2 ----------------------------------------------------------------
class Gpt2ContinueRequest(BaseModel):
    """POST /gpt2_continue body."""

    prompt: str = Field(..., min_length=1, max_length=4000, description="Text to continue.")
    max_new_tokens: int = Field(40, ge=1, le=128, description="Maximum tokens to generate (greedy, deterministic).")

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"prompt": "The meaning of life is", "max_new_tokens": 30}]}
    )


class Gpt2ContinueResponse(InferenceMeta):
    output_text: str = Field(..., description="ONLY the newly generated continuation (the prompt is not echoed).")
    prompt_tokens: int
    generated_tokens: int


# ---- T5 -------------------------------------------------------------------
class T5SummarizeRequest(BaseModel):
    """POST /t5_summarize body."""

    text: str = Field(..., min_length=1, max_length=4000, description="Article-like text to summarize.")
    max_new_tokens: int = Field(60, ge=1, le=128, description="Maximum summary length in tokens (beam search, deterministic).")
    min_new_tokens: int = Field(0, ge=0, le=128, description="Minimum summary length in tokens; must not exceed max_new_tokens.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "The city council voted on Tuesday to approve a new bike lane network "
                    "across 40 miles of downtown streets. Construction begins next spring.",
                    "max_new_tokens": 40,
                }
            ]
        }
    )


class T5SummarizeResponse(InferenceMeta):
    output_text: str = Field(
        ..., description="Summary. t5-small is a small model: expect extractive, sometimes imperfect output."
    )
    input_tokens: int = Field(..., description="Includes the 'summarize: ' task prefix.")
    output_tokens: int


# ---- custom encoder attention --------------------------------------------
class EncoderAttentionRequest(BaseModel):
    """POST /encoder_attention body."""

    text: str = Field(..., min_length=1, max_length=500, description="Words to visualise (split on whitespace, max 32 words).")
    num_heads: int = Field(4, description="Attention heads (1, 2 or 4).")
    use_positional_encoding: bool = Field(
        True, description="Add sinusoidal positional encoding. Off => attention ignores word order."
    )

    @field_validator("num_heads")
    @classmethod
    def _heads(cls, v: int) -> int:
        if v not in (1, 2, 4):
            raise ValueError("num_heads must be 1, 2 or 4")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "the cat sat on the mat", "num_heads": 4, "use_positional_encoding": True}]
        }
    )


class EncoderAttentionResponse(BaseModel):
    tokens: List[str] = Field(..., description="Lower-cased whitespace tokens, in order. Axis labels for the heatmap.")
    num_tokens: int
    num_heads: int
    d_model: int
    head_dim: int
    use_positional_encoding: bool
    attention_shape: List[int] = Field(..., description="[num_heads, T, T]; attention[h][i][j] = weight of query i on key j.")
    attention: List[List[List[float]]] = Field(..., description="Per-head attention; every row sums to 1.")
    mean_attention: List[List[float]] = Field(..., description="Average over heads, [T][T]. Convenient for a single heatmap.")
    max_row_sum_error: float = Field(..., description="max |row sum - 1| across all rows (numerical check).")
    disclaimer: str = Field(
        ...,
        description="Always present: these are EDUCATIONAL weights from an UNTRAINED custom encoder, not a pretrained language explanation.",
    )


# ---- benchmarks -----------------------------------------------------------
class BenchmarkRow(BaseModel):
    """One row of the `benchmarks` materialized view. All latencies are server-side milliseconds."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    n_inferences: int
    mean_latency_ms: float
    min_latency_ms: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: int
    last_served_at: Optional[datetime] = None


class BenchmarkRefreshInfo(BaseModel):
    last_refreshed_at: Optional[datetime] = Field(None, description="When the materialized view was actually last refreshed (null = never).")
    last_duration_ms: Optional[int] = None
    last_row_count: Optional[int] = None
    last_mode: Optional[str] = Field(None, description='"initial" or "concurrent".')
    age_seconds: Optional[float] = Field(None, description="Seconds since last_refreshed_at, at response time.")
    stale: bool = Field(..., description="True if the data is older than the refresh TTL (e.g. a refresh failed).")
    refreshed_now: bool = Field(..., description="True if THIS request triggered a refresh.")
    ttl_seconds: int = Field(..., description="Automatic refresh happens at most once per TTL, on read.")
    note: Optional[str] = Field(None, description="Human-readable note, e.g. a refresh failure.")


class BenchmarksResponse(BaseModel):
    benchmarks: List[BenchmarkRow] = Field(..., description="One row per model, as of the last refresh. New inferences appear after the next refresh.")
    refresh: BenchmarkRefreshInfo
    latency_definition: str
