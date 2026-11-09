"""Educational attention heatmaps from the custom encoder, WITHOUT PyTorch.

Behaviour-compatible counterpart of ``api/encoder_demo.py`` (which runs ``src/transformer.py``
in PyTorch and remains the reference). Here the *same* untrained block (fixed seed) has been
exported to ONNX by ``scripts/export_onnx.py``; embedding lookup and sinusoidal positional
encoding are done in numpy. ``tests/test_onnx_encoder.py`` checks both give the same attention.

IMPORTANT: the weights are an UNTRAINED, randomly initialised block -- the attention patterns
illustrate the *mechanism* of self-attention, not how any pretrained model reads text.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

D_MODEL = 32
D_FF = 64
VOCAB_BUCKETS = 512
MAX_TOKENS = 32
SEED = 0

DISCLAIMER = (
    "Educational only: these attention weights come from an untrained, randomly initialised "
    "custom encoder block (fixed seed). They illustrate how self-attention works and are NOT "
    "an explanation of what BERT, GPT-2, T5 or any pretrained language model does."
)

_LOCK = threading.Lock()
_SESSIONS: Dict[int, Any] = {}
_EMBEDDING: List[np.ndarray] = []


def edu_dir() -> Path:
    base = os.environ.get("TRIOFORMER_MODEL_DIR") or Path(__file__).resolve().parents[1] / "models"
    return Path(base) / "edu"


def available() -> bool:
    d = edu_dir()
    return (d / "embedding.npy").is_file() and all((d / f"block_h{h}.onnx").is_file() for h in (1, 2, 4))


def tokenize(text: str) -> List[str]:
    """Lower-cased whitespace words."""
    tokens = text.lower().split()
    if not tokens:
        raise ValueError("text must contain at least one word")
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"text has {len(tokens)} words; maximum is {MAX_TOKENS}")
    return tokens


def token_bucket(token: str) -> int:
    """Stable (process-independent) id in [0, VOCAB_BUCKETS)."""
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % VOCAB_BUCKETS


def sinusoidal_positions(length: int, d_model: int = D_MODEL) -> np.ndarray:
    """PE[pos, 2i] = sin(pos / 10000^(2i/D)), PE[pos, 2i+1] = cos(...); float32, like src/transformer.py."""
    position = np.arange(length, dtype=np.float32)[:, None]
    even = np.arange(0, d_model, 2, dtype=np.float32)
    inv_freq = np.exp(-math.log(10000.0) * even / d_model).astype(np.float32)
    pe = np.zeros((length, d_model), np.float32)
    pe[:, 0::2] = np.sin(position * inv_freq)
    pe[:, 1::2] = np.cos(position * inv_freq[: d_model // 2])
    return pe


def _session(num_heads: int):
    """Create (once) the onnxruntime session for ``num_heads`` and load the embedding table.

    Raises OSError if the exported files are missing or cannot be loaded (the API maps it to 503).
    """
    with _LOCK:
        if num_heads not in _SESSIONS or not _EMBEDDING:
            if not available():
                raise FileNotFoundError("educational encoder files not found (models/edu); run scripts/export_onnx.py")
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                if num_heads not in _SESSIONS:
                    _SESSIONS[num_heads] = ort.InferenceSession(
                        str(edu_dir() / f"block_h{num_heads}.onnx"), so, providers=["CPUExecutionProvider"])
                if not _EMBEDDING:
                    _EMBEDDING.append(np.load(edu_dir() / "embedding.npy"))
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own exception types
                raise OSError(f"could not load the educational encoder: {type(exc).__name__}") from exc
        return _SESSIONS[num_heads], _EMBEDDING[0]


def compute_attention(text: str, num_heads: int = 4, use_positional_encoding: bool = True) -> Dict[str, object]:
    """Return tokens and per-head attention for ``text`` (same dict as api/encoder_demo.py).

    Raises:
        ValueError: empty text, too many words, or an unsupported ``num_heads``.
    """
    if num_heads not in (1, 2, 4):
        raise ValueError("num_heads must be 1, 2 or 4")
    tokens = tokenize(text)
    session, embedding = _session(num_heads)
    x = embedding[[token_bucket(t) for t in tokens]]                      # [T, D]
    if use_positional_encoding:
        x = x + sinusoidal_positions(len(tokens))
    attn = np.asarray(session.run(None, {"x": x[None].astype(np.float32)})[1])[0]   # [H, T, T]
    return {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "num_heads": num_heads,
        "d_model": D_MODEL,
        "head_dim": D_MODEL // num_heads,
        "use_positional_encoding": use_positional_encoding,
        "attention_shape": list(attn.shape),
        "attention": attn.tolist(),
        "mean_attention": attn.mean(axis=0).tolist(),
        "max_row_sum_error": float(np.abs(attn.sum(axis=-1) - 1.0).max()),
        "disclaimer": DISCLAIMER,
    }
