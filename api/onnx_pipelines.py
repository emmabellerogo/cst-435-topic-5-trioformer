"""Lightweight ONNX Runtime inference for BERT, GPT-2 and T5 (no PyTorch, no Transformers).

Drop-in counterpart of ``api/model_pipelines.py`` (the PyTorch *reference* implementation):
same function names and signatures, same result dataclasses, same validation limits and
error types, so ``api/main.py`` can select either with ``TRIOFORMER_BACKEND=onnx|torch``.

Why this exists: Render Free has 512 MB of RAM and 0.1 CPU. The PyTorch pipelines need
~0.6-0.8 GB per model. Here every model is an int8-quantized ONNX graph whose weights are
**memory-mapped** from disk by onnxruntime (stored as one external-data file), only **one
model is kept in memory at a time**, and tokenization uses the small Rust ``tokenizers``
wheel. Measured peak process memory when switching among all three: see the report in
``scripts/measure_memory.py``.

Same checkpoints, different numerics: the artifacts are bert-base-uncased, gpt2 and t5-small
exported with ``scripts/export_onnx.py`` and quantized to int8, so outputs are *close to, not
bit-identical with*, the PyTorch pipelines. (With ``--precision fp32`` the same code reproduces
the PyTorch outputs exactly -- that is how the generation loops below are tested.)

Generation is implemented here in numpy (no ``model.generate``): GPT-2 uses greedy decoding with
a KV cache and ``no_repeat_ngram_size=3``; T5 uses 4-beam search with ``length_penalty=2.0``,
``no_repeat_ngram_size=3`` and ``early_stopping`` following Hugging Face's beam-search rules.
All decoding is deterministic.

Environment:
    TRIOFORMER_MODEL_DIR   directory holding the exported files (default: <repo>/models)
    TRIOFORMER_ORT_THREADS onnxruntime intra-op threads (default 1; Render Free has 0.1 CPU)
"""
from __future__ import annotations

import gc
import importlib.metadata
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Checkpoints and limits (identical to api/model_pipelines.py -- parity-tested)
# --------------------------------------------------------------------------- #
BERT_MODEL = "bert-base-uncased"
GPT2_MODEL = "gpt2"
T5_MODEL = "t5-small"

MASK_TOKEN = "[MASK]"
T5_PREFIX = "summarize: "

MAX_INPUT_CHARS = 4000
BERT_MAX_TOKENS = 512
GPT2_CONTEXT = 1024
T5_MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS_LIMIT = 128
BERT_MAX_TOP_K = 10

_NO_REPEAT_NGRAM = 3
_T5_BEAMS = 4
_T5_LENGTH_PENALTY = 2.0
_GPT2_EOS = 50256
_T5_START, _T5_EOS = 0, 1
_T5_LAYERS, _T5_HEADS, _T5_HEAD_DIM = 6, 8, 64
_GPT2_LAYERS, _GPT2_HEADS, _GPT2_HEAD_DIM = 12, 12, 64

# Files that must exist (relative to the model dir). Each .onnx also needs its sibling .onnx.data.
_FILES: Dict[str, Tuple[str, ...]] = {
    BERT_MODEL: ("bert/model.onnx", "bert/tokenizer.json"),
    GPT2_MODEL: ("gpt2/model.onnx", "gpt2/tokenizer.json"),
    T5_MODEL: ("t5-small/encoder.onnx", "t5-small/cross.onnx", "t5-small/decoder_step.onnx",
               "t5-small/tokenizer.json"),
}
_KINDS = {BERT_MODEL: "mlm", GPT2_MODEL: "causal", T5_MODEL: "seq2seq"}


# --------------------------------------------------------------------------- #
# Errors and results (same shapes as api/model_pipelines.py)
# --------------------------------------------------------------------------- #
class InvalidInputError(ValueError):
    """The caller's input is unacceptable (maps to HTTP 422)."""


class ModelLoadError(RuntimeError):
    """A model could not be loaded (missing/corrupt files, onnxruntime problem, ...)."""


@dataclass(frozen=True)
class MaskPrediction:
    token: str
    score: float
    sequence: str


@dataclass(frozen=True)
class FillMaskResult:
    model: str
    output_text: str
    predictions: Tuple[MaskPrediction, ...]
    input_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    model: str
    output_text: str
    prompt_tokens: int
    generated_tokens: int


@dataclass(frozen=True)
class SummaryResult:
    model: str
    output_text: str
    input_tokens: int
    output_tokens: int


# --------------------------------------------------------------------------- #
# Model directory, status, loading, and the one-model-at-a-time cache
# --------------------------------------------------------------------------- #
def model_dir() -> Path:
    return Path(os.environ.get("TRIOFORMER_MODEL_DIR") or Path(__file__).resolve().parents[1] / "models")


def _required_paths(name: str) -> List[Path]:
    base = model_dir()
    paths: List[Path] = []
    for rel in _FILES[name]:
        p = base / rel
        paths.append(p)
        if rel.endswith(".onnx"):
            paths.append(Path(str(p) + ".data"))
    return paths


def backend_status() -> Dict[str, Any]:
    """Cheap readiness check for /healthz: imports and file presence only; loads nothing."""
    problems: List[str] = []
    for mod in ("onnxruntime", "tokenizers"):
        try:
            importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"python package '{mod}' is not installed")
    missing = sorted({str(p.relative_to(model_dir())) for n in _FILES for p in _required_paths(n) if not p.is_file()})
    if missing:
        problems.append(f"{len(missing)} model file(s) missing from the model directory "
                        "(run `python scripts/fetch_models.py`)")
    return {"ok": not problems, "detail": "; ".join(problems) or None, "missing": missing,
            "model_dir": str(model_dir())}


def runtime_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for mod in ("onnxruntime", "tokenizers"):
        try:
            out[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            out[mod] = None
    return out


@dataclass(frozen=True)
class _Bundle:
    tokenizer: Any                 # tokenizers.Tokenizer
    sessions: Dict[str, Any]       # name -> onnxruntime.InferenceSession (or a test double)


def _make_session(path: Path):
    """One onnxruntime session. Weights in the external-data file are memory-mapped, not copied."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    threads = int(os.environ.get("TRIOFORMER_ORT_THREADS", "1"))
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = False   # don't keep a growing scratch arena
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


_FETCH_GROUP = {BERT_MODEL: "bert", GPT2_MODEL: "gpt2", T5_MODEL: "t5"}


def _missing_files(name: str) -> List[str]:
    return [str(p.relative_to(model_dir())) for p in _required_paths(name) if not p.is_file()]


def _auto_fetch(name: str) -> None:
    """Safety net (TRIOFORMER_AUTO_FETCH=1): download this model's files if the build didn't leave them.

    Uses the same checksummed manifest/release as ``scripts/fetch_models.py``. The disk is
    ephemeral on Render Free, so this costs a download on each cold start -- it only exists so a
    build-time download that did not persist degrades to a slow first request, not a dead service.
    """
    if os.environ.get("TRIOFORMER_AUTO_FETCH", "").lower() not in {"1", "true", "yes"}:
        return
    from scripts import fetch_models

    manifest = Path(__file__).resolve().parents[1] / "models" / "manifest.json"
    rc = fetch_models.fetch(manifest, model_dir(), only=[_FETCH_GROUP[name]], log=lambda *_: None)
    if rc != 0:
        raise FileNotFoundError("automatic download of the model files failed")


def _load_bundle(name: str, kind: str) -> _Bundle:
    """Read a model's tokenizer and sessions from disk. Isolated so tests can replace it."""
    from tokenizers import Tokenizer

    if _missing_files(name):
        _auto_fetch(name)
    missing = _missing_files(name)
    if missing:
        raise FileNotFoundError(f"missing model files: {', '.join(missing)}")
    base = model_dir()
    if name == BERT_MODEL:
        sessions = {"model": _make_session(base / "bert/model.onnx")}
        tok = Tokenizer.from_file(str(base / "bert/tokenizer.json"))
    elif name == GPT2_MODEL:
        sessions = {"model": _make_session(base / "gpt2/model.onnx")}
        tok = Tokenizer.from_file(str(base / "gpt2/tokenizer.json"))
    else:
        sessions = {"encoder": _make_session(base / "t5-small/encoder.onnx"),
                    "cross": _make_session(base / "t5-small/cross.onnx"),
                    "step": _make_session(base / "t5-small/decoder_step.onnx")}
        tok = Tokenizer.from_file(str(base / "t5-small/tokenizer.json"))
    return _Bundle(tokenizer=tok, sessions=sessions)


# The lock guards the cache AND serialises inference: evicting a model while a request is
# still running on it would crash, and the host has far too little CPU/RAM for parallelism.
_CACHE: Dict[str, _Bundle] = {}
_LOCK = threading.RLock()


def _trim_memory() -> None:
    """Hand freed heap back to the OS (glibc only; a no-op elsewhere)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _evict_all() -> None:
    _CACHE.clear()
    gc.collect()
    _trim_memory()


def _get_bundle(name: str, kind: str) -> _Bundle:
    """Return the resident bundle for ``name``; otherwise evict everything and load it."""
    with _LOCK:
        bundle = _CACHE.get(name)
        if bundle is not None:
            return bundle
        _evict_all()  # one model at a time keeps the process well under 512 MB
        try:
            bundle = _load_bundle(name, kind)
        except Exception as exc:  # noqa: BLE001 - normalise every loader failure
            raise ModelLoadError(f"Could not load model '{name}': {type(exc).__name__}: {exc}") from exc
        _CACHE[name] = bundle
        return bundle


def clear_model_cache() -> None:
    with _LOCK:
        _evict_all()


def loaded_models() -> List[str]:
    return sorted(_CACHE)


def preload(names: Tuple[str, ...] = ()) -> None:
    """Load the given model(s). Only ONE model stays resident: the last name wins."""
    for name in names:
        _get_bundle(name, _KINDS[name])


# --------------------------------------------------------------------------- #
# Validation (copied from api/model_pipelines.py; parity-tested)
# --------------------------------------------------------------------------- #
def _check_text(text: Any, field: str) -> str:
    if not isinstance(text, str):
        raise InvalidInputError(f"{field} must be a string, got {type(text).__name__}")
    if not text.strip():
        raise InvalidInputError(f"{field} must not be empty")
    if len(text) > MAX_INPUT_CHARS:
        raise InvalidInputError(f"{field} is too long ({len(text)} characters; maximum {MAX_INPUT_CHARS})")
    return text


def _check_int(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(f"{field} must be an integer, got {value!r}")
    if not low <= value <= high:
        raise InvalidInputError(f"{field} must be between {low} and {high}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# numpy helpers
# --------------------------------------------------------------------------- #
_CLEANUPS = ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
             (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"))


def _clean(text: str) -> str:
    """transformers' ``clean_up_tokenization`` (applied by AutoTokenizer.decode for these models)."""
    for old, new in _CLEANUPS:
        text = text.replace(old, new)
    return text


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _banned_tokens(seq: List[int], n: int = _NO_REPEAT_NGRAM) -> List[int]:
    """Token ids that would complete an n-gram already present in ``seq`` (HF NoRepeatNGram)."""
    if len(seq) < n:
        return []
    prefix = tuple(seq[-(n - 1):])
    return [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == prefix]


def _decode(tokenizer, ids: List[int], skip_special: bool) -> str:
    return _clean(tokenizer.decode(ids, skip_special_tokens=skip_special))


# --------------------------------------------------------------------------- #
# BERT: masked-token prediction
# --------------------------------------------------------------------------- #
def bert_fill_mask(text: str, top_k: int = 5) -> FillMaskResult:
    """Predict the token hidden behind a single ``[MASK]`` (see api/model_pipelines.py)."""
    text = _check_text(text, "text")
    top_k = _check_int(top_k, "top_k", 1, BERT_MAX_TOP_K)
    n_masks = text.count(MASK_TOKEN)
    if n_masks != 1:
        raise InvalidInputError(f"text must contain exactly one {MASK_TOKEN}, found {n_masks}")

    with _LOCK:
        bundle = _get_bundle(BERT_MODEL, "mlm")
        tok = bundle.tokenizer
        enc = tok.encode(text)
        n_tokens = len(enc.ids)
        if n_tokens > BERT_MAX_TOKENS:
            raise InvalidInputError(f"text is too long for BERT ({n_tokens} tokens; maximum {BERT_MAX_TOKENS})")
        mask_id = tok.token_to_id(MASK_TOKEN)
        positions = [i for i, t in enumerate(enc.ids) if t == mask_id]
        if len(positions) != 1:
            raise InvalidInputError(f"could not locate a single {MASK_TOKEN} after tokenization")
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
            "mask_pos": np.array([positions[0]], dtype=np.int64),
        }
        logits = np.asarray(bundle.sessions["model"].run(None, feeds)[0])[0]
        probs = np.exp(_log_softmax(logits.astype(np.float64)))
        k = min(top_k, probs.shape[0])
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx], kind="stable")]

        predictions: List[MaskPrediction] = []
        for token_id in idx.tolist():
            token = _decode(tok, [token_id], skip_special=False).strip()
            if token.startswith("##"):  # WordPiece continuation: glue onto the preceding word
                token = token[2:]
                sequence = text.replace(" " + MASK_TOKEN, token).replace(MASK_TOKEN, token)
            else:
                sequence = text.replace(MASK_TOKEN, token)
            predictions.append(MaskPrediction(token=token, score=float(probs[token_id]), sequence=sequence))
        return FillMaskResult(model=BERT_MODEL, output_text=predictions[0].sequence,
                              predictions=tuple(predictions), input_tokens=n_tokens)


# --------------------------------------------------------------------------- #
# GPT-2: text continuation (greedy, KV cache)
# --------------------------------------------------------------------------- #
def gpt2_continue(prompt: str, max_new_tokens: int = 40) -> GenerationResult:
    """Continue ``prompt`` greedily; returns only the new text (see api/model_pipelines.py)."""
    prompt = _check_text(prompt, "prompt")
    max_new_tokens = _check_int(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS_LIMIT)

    with _LOCK:
        bundle = _get_bundle(GPT2_MODEL, "causal")
        tok, session = bundle.tokenizer, bundle.sessions["model"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
        n_prompt = len(prompt_ids)
        if n_prompt + max_new_tokens > GPT2_CONTEXT:
            raise InvalidInputError(
                f"prompt ({n_prompt} tokens) + max_new_tokens ({max_new_tokens}) exceeds "
                f"GPT-2's {GPT2_CONTEXT}-token context window")

        past_names = [i.name for i in session.get_inputs()][1:]
        past = {n: np.zeros((1, _GPT2_HEADS, 0, _GPT2_HEAD_DIM), np.float32) for n in past_names}
        step_ids = np.array([prompt_ids], dtype=np.int64)
        seq = list(prompt_ids)
        new: List[int] = []
        for _ in range(max_new_tokens):
            out = session.run(None, {"input_ids": step_ids, **past})
            logits = np.asarray(out[0])[0].astype(np.float64)
            banned = _banned_tokens(seq)
            if banned:
                logits[banned] = -np.inf
            token = int(np.argmax(logits))
            new.append(token)
            seq.append(token)
            if token == _GPT2_EOS:
                break
            past = dict(zip(past_names, out[1:]))
            step_ids = np.array([[token]], dtype=np.int64)
        text = _decode(tok, [t for t in new if t != _GPT2_EOS], skip_special=True)
        return GenerationResult(model=GPT2_MODEL, output_text=text, prompt_tokens=n_prompt,
                                generated_tokens=len(new))


# --------------------------------------------------------------------------- #
# T5: summarization (4-beam search, KV cache)
# --------------------------------------------------------------------------- #
def t5_summarize(text: str, max_new_tokens: int = 60, min_new_tokens: int = 0) -> SummaryResult:
    """Summarize ``text`` with t5-small (``summarize:`` prefix; see api/model_pipelines.py).

    t5-small is a small model and this build is int8-quantized: summaries may be extractive
    or imperfect.
    """
    text = _check_text(text, "text")
    max_new_tokens = _check_int(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS_LIMIT)
    min_new_tokens = _check_int(min_new_tokens, "min_new_tokens", 0, MAX_NEW_TOKENS_LIMIT)
    if min_new_tokens > max_new_tokens:
        raise InvalidInputError(
            f"min_new_tokens ({min_new_tokens}) must not exceed max_new_tokens ({max_new_tokens})")

    with _LOCK:
        bundle = _get_bundle(T5_MODEL, "seq2seq")
        tok = bundle.tokenizer
        enc = tok.encode(T5_PREFIX + text.strip())
        n_in = len(enc.ids)
        if n_in > T5_MAX_INPUT_TOKENS:
            raise InvalidInputError(f"text is too long for t5-small ({n_in} tokens; maximum {T5_MAX_INPUT_TOKENS})")
        out_ids = _t5_beam_search(bundle, enc.ids, max_new_tokens, min_new_tokens)
        return SummaryResult(model=T5_MODEL,
                             output_text=_decode(tok, out_ids, skip_special=True).strip(),
                             input_tokens=n_in, output_tokens=len(out_ids) - 1)


def _t5_beam_search(bundle: _Bundle, input_ids: List[int], max_new_tokens: int, min_new_tokens: int) -> List[int]:
    """Beam search with Hugging Face's rules (early_stopping=True, length_penalty applied on EOS)."""
    num_beams, length_penalty = _T5_BEAMS, _T5_LENGTH_PENALTY
    enc_s, cross_s, step_s = (bundle.sessions[k] for k in ("encoder", "cross", "step"))
    ids = np.array([input_ids], dtype=np.int64)
    mask = np.ones_like(ids)
    hidden = np.asarray(enc_s.run(None, {"input_ids": ids, "attention_mask": mask})[0])
    cross = [np.repeat(np.asarray(c), num_beams, 0) for c in cross_s.run(None, {"hidden": hidden})]
    hidden = np.repeat(hidden, num_beams, 0)
    mask_b = np.repeat(mask, num_beams, 0)
    pnames = [i.name for i in step_s.get_inputs()][3:]
    self_kv = [np.zeros((num_beams, _T5_HEADS, 0, _T5_HEAD_DIM), np.float32) for _ in range(2 * _T5_LAYERS)]

    seqs: List[List[int]] = [[_T5_START] for _ in range(num_beams)]
    scores = np.array([0.0] + [-1e9] * (num_beams - 1))   # only beam 0 is live at step 1
    hyps: List[Tuple[float, List[int]]] = []
    max_len = max_new_tokens + 1                           # +1 for the decoder start token
    done = False
    while True:
        feeds: Dict[str, Any] = {}
        for i in range(_T5_LAYERS):
            feeds[pnames[4 * i]], feeds[pnames[4 * i + 1]] = self_kv[2 * i], self_kv[2 * i + 1]
            feeds[pnames[4 * i + 2]], feeds[pnames[4 * i + 3]] = cross[2 * i], cross[2 * i + 1]
        out = step_s.run(None, {"decoder_input_ids": np.array([[s[-1]] for s in seqs], dtype=np.int64),
                                "encoder_hidden_states": hidden, "encoder_attention_mask": mask_b, **feeds})
        logp = _log_softmax(np.asarray(out[0]).astype(np.float64))
        for i, s in enumerate(seqs):
            if len(s) - 1 < min_new_tokens:
                logp[i, _T5_EOS] = -np.inf
            banned = _banned_tokens(s)
            if banned:
                logp[i, banned] = -np.inf
        total = (logp + scores[:, None]).reshape(-1)
        vocab = logp.shape[1]
        top = np.argpartition(-total, 2 * num_beams)[:2 * num_beams]   # 2*beams candidates, best first
        top = top[np.argsort(-total[top], kind="stable")]

        new_seqs, new_scores, src = [], [], []
        for rank, flat in enumerate(top.tolist()):
            beam, token, score = flat // vocab, flat % vocab, float(total[flat])
            if token == _T5_EOS:
                if rank >= num_beams:
                    continue                                   # EOS ranked below the beam width is ignored
                hyps.append((score / (len(seqs[beam]) ** length_penalty), seqs[beam][:]))
            else:
                new_seqs.append(seqs[beam] + [token])
                new_scores.append(score)
                src.append(beam)
            if len(new_seqs) == num_beams:
                break
        hyps = sorted(hyps, key=lambda h: -h[0])[:num_beams]
        done = len(hyps) >= num_beams                          # early_stopping=True
        self_kv = [np.take(np.asarray(x), src, axis=0) for x in out[1:]]
        seqs, scores = new_seqs, np.array(new_scores)
        if done or len(seqs[0]) >= max_len:
            break
    if not done:                                               # ran out of length: unfinished beams compete too
        for s, sc in zip(seqs, scores):
            hyps.append((sc / ((len(s) - 1) ** length_penalty), s))
    best = max(hyps, key=lambda h: h[0])[1]
    return best + [_T5_EOS] if len(best) < max_len else best
