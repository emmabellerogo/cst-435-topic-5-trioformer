"""Pretrained-model inference pipelines: BERT fill-mask, GPT-2 continuation, T5 summarization.

Standalone (no FastAPI, no database): every public function takes plain Python
values and returns a frozen dataclass, so it can be unit-tested with mocks and
wrapped by HTTP routes later.

Checkpoints (Hugging Face Hub), weights loaded in float32 on CPU with plain PyTorch:

==================  ======================  ======  ========  ===========================
task                checkpoint              params  fp32 size  context window
==================  ======================  ======  ========  ===========================
masked-token        prajjwal1/bert-small    28.8 M  115 MB     512 tokens
continuation        erwanf/gpt2-mini        38.6 M  154 MB     512 tokens (prompt + output)
summarization       t5-small                60.5 M  242 MB     512 input tokens
==================  ======================  ======  ========  ===========================

These are deliberately small so the whole service fits Render Free (512 MB RAM). Measured on
Linux in a 512 MB cgroup (see README "Memory"): one model at a time plus FastAPI, PyTorch and
Supabase uses ~320 MB of non-reclaimable memory; the OOM threshold is ~360 MB. Loading all three
at once also survived at 512 MB but leaves far less margin, so only ONE model is kept resident.

Design notes
------------
* **Lazy, cached, ONE at a time**: a model/tokenizer pair is loaded on first use and kept in
  a process-wide cache guarded by a lock, so repeated (and concurrent) requests for the same
  model never reload it. Asking for a DIFFERENT model first evicts the resident one
  (``gc.collect()`` + ``malloc_trim``), which keeps peak memory under Render Free's limit.
  A failed load is NOT cached, so a later call can retry.
* **Validate first, load second**: cheap checks (empty text, ``[MASK]`` count,
  length, generation limits) run BEFORE any model is loaded, so bad requests
  never trigger a multi-hundred-MB download.
* **Deterministic**: no sampling anywhere. GPT-2 uses greedy decoding and T5
  uses beam search, so identical inputs give identical outputs on CPU.
* **CPU inference**: ``model.eval()`` and ``torch.inference_mode()``.

Quality caveats
---------------
* ``t5-small`` (60 M parameters) is a small model. Its summaries are often
  extractive (copying sentences), can drop facts, and can occasionally be
  ungrammatical. Treat them as a demo of the pipeline, not as production-quality
  summaries. It was trained mainly on news-style text (CNN/DailyMail among its
  multi-task data) and handles short, article-like inputs best.
* ``erwanf/gpt2-mini`` (38.6 M parameters, GPT-2 architecture, trained on OpenWebText) continues
  text fluently for a sentence or two, then drifts and may repeat itself; it has no factual
  grounding and no safety tuning. ``no_repeat_ngram_size`` only partly mitigates repetition.
* ``prajjwal1/bert-small`` (28.8 M, 4 layers) is a genuinely pretrained masked-LM but weaker than
  bert-base: it ranks sensible candidates (e.g. "paris" first for the capital of France) with
  lower confidence. It lowercases its input, so predictions are lowercase.

Environment:
    ``TRIOFORMER_HF_LOCAL_ONLY=1`` makes loading fail instead of downloading
    when the weights are not already in the local Hugging Face cache (used by
    the optional smoke test).
"""
from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch

# --------------------------------------------------------------------------- #
# Checkpoints and limits
# --------------------------------------------------------------------------- #
BERT_MODEL = "prajjwal1/bert-small"
GPT2_MODEL = "erwanf/gpt2-mini"
T5_MODEL = "t5-small"

MASK_TOKEN = "[MASK]"
T5_PREFIX = "summarize: "

MAX_INPUT_CHARS = 4000        # cheap pre-load guard against huge payloads
BERT_MAX_TOKENS = 512         # bert-small position embeddings
GPT2_CONTEXT = 512            # gpt2-mini context window (prompt + generated)
T5_MAX_INPUT_TOKENS = 512     # t5-small was trained on 512-token inputs
MAX_NEW_TOKENS_LIMIT = 128    # hard cap on generated tokens per request
BERT_MAX_TOP_K = 10

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class InvalidInputError(ValueError):
    """The caller's input is unacceptable (maps naturally to HTTP 422/400)."""


class ModelLoadError(RuntimeError):
    """A checkpoint could not be loaded (network, disk, corrupt cache, ...)."""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaskPrediction:
    """One candidate for the masked position."""

    token: str
    score: float      # softmax probability over the vocabulary
    sequence: str     # the input with [MASK] replaced by ``token``


@dataclass(frozen=True)
class FillMaskResult:
    model: str
    output_text: str                       # input filled with the top prediction
    predictions: Tuple[MaskPrediction, ...]
    input_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    model: str
    output_text: str                       # ONLY the new continuation
    prompt_tokens: int
    generated_tokens: int


@dataclass(frozen=True)
class SummaryResult:
    model: str
    output_text: str
    input_tokens: int                      # includes the "summarize: " prefix
    output_tokens: int                     # generated tokens (incl. EOS, excl. decoder start)


# --------------------------------------------------------------------------- #
# Lazy model cache
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Bundle:
    tokenizer: Any
    model: Any


_CACHE: Dict[str, _Bundle] = {}
_LOCK = threading.Lock()


def _local_only() -> bool:
    return os.environ.get("TRIOFORMER_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}


def _load_bundle(name: str, kind: str) -> _Bundle:
    """Download/read a checkpoint. ``kind`` is 'mlm', 'causal' or 'seq2seq'.

    Isolated in one function so tests can replace it and never touch the network.
    """
    from transformers import (  # imported lazily: keeps `import api.model_pipelines` fast
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    auto = {
        "mlm": AutoModelForMaskedLM,
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }[kind]
    local = _local_only()
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local)
    model = auto.from_pretrained(name, local_files_only=local)
    model.eval()
    model.to("cpu")
    return _Bundle(tokenizer=tokenizer, model=model)


def _release_memory() -> None:
    """Collect garbage and hand freed heap pages back to the OS (glibc only; best effort)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not Linux/glibc (e.g. macOS, Windows, musl)
        pass


def _get_bundle(name: str, kind: str) -> _Bundle:
    """Return the cached bundle for ``name``, loading it on first use.

    Only ONE checkpoint stays resident: loading ``name`` first evicts any other model, so the
    process never holds two sets of weights (Render Free has 512 MB of RAM).
    """
    bundle = _CACHE.get(name)
    if bundle is not None:
        return bundle
    with _LOCK:
        bundle = _CACHE.get(name)  # another thread may have loaded it while we waited
        if bundle is not None:
            return bundle
        if _CACHE:                  # evict the previous model BEFORE loading the next one
            _CACHE.clear()
            _release_memory()
        try:
            bundle = _load_bundle(name, kind)
        except Exception as exc:  # noqa: BLE001 - normalise every loader failure
            raise ModelLoadError(
                f"Could not load model '{name}': {type(exc).__name__}: {exc}"
            ) from exc
        _CACHE[name] = bundle
        return bundle


def clear_model_cache() -> None:
    """Drop all cached models (frees memory; next call reloads)."""
    with _LOCK:
        _CACHE.clear()
    _release_memory()


def loaded_models() -> List[str]:
    """Names of the checkpoints currently resident in memory."""
    return sorted(_CACHE)


def preload(names: Tuple[str, ...] = (T5_MODEL,)) -> None:
    """Load checkpoints ahead of time (e.g. at server start).

    Only one model stays resident, so with several names the LAST one wins.
    """
    kinds = {BERT_MODEL: "mlm", GPT2_MODEL: "causal", T5_MODEL: "seq2seq"}
    for name in names:
        _get_bundle(name, kinds[name])


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _check_text(text: Any, field: str) -> str:
    if not isinstance(text, str):
        raise InvalidInputError(f"{field} must be a string, got {type(text).__name__}")
    if not text.strip():
        raise InvalidInputError(f"{field} must not be empty")
    if len(text) > MAX_INPUT_CHARS:
        raise InvalidInputError(
            f"{field} is too long ({len(text)} characters; maximum {MAX_INPUT_CHARS})"
        )
    return text


def _check_int(value: Any, field: str, low: int, high: int) -> int:
    # bool is a subclass of int; True/False are never a meaningful count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(f"{field} must be an integer, got {value!r}")
    if not low <= value <= high:
        raise InvalidInputError(f"{field} must be between {low} and {high}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# BERT: masked-token prediction
# --------------------------------------------------------------------------- #


def bert_fill_mask(text: str, top_k: int = 5) -> FillMaskResult:
    """Predict the token hidden behind a single ``[MASK]``.

    Args:
        text: A sentence containing **exactly one** ``[MASK]`` (case-sensitive).
        top_k: Number of candidates to return (1..10).

    Raises:
        InvalidInputError: bad text, wrong number of ``[MASK]``, or too many tokens.
        ModelLoadError: the checkpoint could not be loaded.
    """
    text = _check_text(text, "text")
    top_k = _check_int(top_k, "top_k", 1, BERT_MAX_TOP_K)
    n_masks = text.count(MASK_TOKEN)
    if n_masks != 1:
        raise InvalidInputError(
            f"text must contain exactly one {MASK_TOKEN}, found {n_masks}"
        )

    bundle = _get_bundle(BERT_MODEL, "mlm")
    tok, model = bundle.tokenizer, bundle.model
    enc = tok(text, return_tensors="pt")
    n_tokens = int(enc["input_ids"].shape[1])
    if n_tokens > BERT_MAX_TOKENS:
        raise InvalidInputError(
            f"text is too long for BERT ({n_tokens} tokens; maximum {BERT_MAX_TOKENS})"
        )
    mask_positions = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_positions.numel() != 1:  # e.g. the tokenizer did not recognise the literal
        raise InvalidInputError(f"could not locate a single {MASK_TOKEN} after tokenization")

    with torch.inference_mode():
        logits = model(**enc).logits[0, int(mask_positions[0])]   # [vocab]
        probs = torch.softmax(logits.float(), dim=-1)
        scores, ids = torch.topk(probs, k=top_k)

    predictions: List[MaskPrediction] = []
    for score, token_id in zip(scores.tolist(), ids.tolist()):
        token = tok.decode([token_id]).strip()
        if token.startswith("##"):
            # WordPiece continuation: glue it onto the preceding word.
            token = token[2:]
            sequence = text.replace(" " + MASK_TOKEN, token).replace(MASK_TOKEN, token)
        else:
            sequence = text.replace(MASK_TOKEN, token)
        predictions.append(MaskPrediction(token=token, score=float(score), sequence=sequence))
    return FillMaskResult(
        model=BERT_MODEL,
        output_text=predictions[0].sequence,
        predictions=tuple(predictions),
        input_tokens=n_tokens,
    )


# --------------------------------------------------------------------------- #
# GPT-2: text continuation
# --------------------------------------------------------------------------- #


def gpt2_continue(prompt: str, max_new_tokens: int = 40) -> GenerationResult:
    """Continue ``prompt`` with up to ``max_new_tokens`` new tokens (greedy decoding).

    Returns only the newly generated text, never the prompt.

    Raises:
        InvalidInputError: bad prompt/limits, or prompt + max_new_tokens > 512.
        ModelLoadError: the checkpoint could not be loaded.
    """
    prompt = _check_text(prompt, "prompt")
    max_new_tokens = _check_int(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS_LIMIT)

    bundle = _get_bundle(GPT2_MODEL, "causal")
    tok, model = bundle.tokenizer, bundle.model
    enc = tok(prompt, return_tensors="pt")
    n_prompt = int(enc["input_ids"].shape[1])
    if n_prompt + max_new_tokens > GPT2_CONTEXT:
        raise InvalidInputError(
            f"prompt ({n_prompt} tokens) + max_new_tokens ({max_new_tokens}) exceeds "
            f"GPT-2's {GPT2_CONTEXT}-token context window"
        )

    with torch.inference_mode():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # greedy => deterministic
            num_beams=1,
            no_repeat_ngram_size=3,          # deterministic guard against loops
            # GPT-2 has no pad token; reuse EOS explicitly (silences the warning
            # and makes batching/attention masks well-defined).
            pad_token_id=tok.eos_token_id,
        )
    new_ids = out[0, n_prompt:]              # drop the echoed prompt
    return GenerationResult(
        model=GPT2_MODEL,
        output_text=tok.decode(new_ids, skip_special_tokens=True),
        prompt_tokens=n_prompt,
        generated_tokens=int(new_ids.shape[0]),
    )


# --------------------------------------------------------------------------- #
# T5: summarization
# --------------------------------------------------------------------------- #


def t5_summarize(text: str, max_new_tokens: int = 60, min_new_tokens: int = 0) -> SummaryResult:
    """Summarize ``text`` with t5-small (``"summarize: "`` task prefix, beam search).

    .. note:: ``t5-small`` is a 60 M-parameter model; summaries may be extractive,
       incomplete or occasionally wrong. See the module docstring.

    Raises:
        InvalidInputError: bad text/limits, or more than 512 input tokens.
        ModelLoadError: the checkpoint could not be loaded.
    """
    text = _check_text(text, "text")
    max_new_tokens = _check_int(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS_LIMIT)
    min_new_tokens = _check_int(min_new_tokens, "min_new_tokens", 0, MAX_NEW_TOKENS_LIMIT)
    if min_new_tokens > max_new_tokens:
        raise InvalidInputError(
            f"min_new_tokens ({min_new_tokens}) must not exceed max_new_tokens ({max_new_tokens})"
        )

    bundle = _get_bundle(T5_MODEL, "seq2seq")
    tok, model = bundle.tokenizer, bundle.model
    enc = tok(T5_PREFIX + text.strip(), return_tensors="pt")
    n_in = int(enc["input_ids"].shape[1])
    if n_in > T5_MAX_INPUT_TOKENS:
        raise InvalidInputError(
            f"text is too long for t5-small ({n_in} tokens; maximum {T5_MAX_INPUT_TOKENS})"
        )

    with torch.inference_mode():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,                     # t5's own summarization defaults
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,                 # beam search => deterministic
        )
    ids = out[0]                             # [decoder_start, generated..., (eos)]
    return SummaryResult(
        model=T5_MODEL,
        output_text=tok.decode(ids, skip_special_tokens=True).strip(),
        input_tokens=n_in,
        output_tokens=int(ids.shape[0]) - 1,  # generated tokens, excluding decoder start
    )
