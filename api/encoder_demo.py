"""Educational attention heatmaps from the project's own encoder (src/transformer.py).

IMPORTANT: this uses a *small, UNTRAINED* encoder block whose weights are drawn
from a fixed random seed. The attention patterns therefore illustrate the
MECHANISM (queries attending to keys, the effect of positional encoding); they
are NOT a pretrained language model's explanation of anything. Pretrained
BERT / GPT-2 / T5 are served by api/model_pipelines.py and do not use this code.

Pipeline: whitespace words -> stable hash -> fixed random embedding
(+ optional sinusoidal positional encoding) -> one pre-norm TransformerEncoderBlock
-> per-head attention weights ``[H, T, T]``.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, List

import torch
from torch import nn

from src.transformer import SinusoidalPositionalEncoding, TransformerEncoderBlock

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


def tokenize(text: str) -> List[str]:
    """Lower-cased whitespace words."""
    tokens = text.lower().split()
    if not tokens:
        raise ValueError("text must contain at least one word")
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"text has {len(tokens)} words; maximum is {MAX_TOKENS}")
    return tokens


def token_bucket(token: str) -> int:
    """Stable (process-independent) id in [0, VOCAB_BUCKETS) -- not Python's salted hash()."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % VOCAB_BUCKETS


@lru_cache(maxsize=4)
def _model(num_heads: int):
    """Deterministically initialised (embedding, positional encoding, block)."""
    # fork_rng keeps the global RNG untouched (the rest of the app may rely on it).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        embedding = nn.Embedding(VOCAB_BUCKETS, D_MODEL)
        block = TransformerEncoderBlock(D_MODEL, num_heads, D_FF, dropout=0.0)
    pos = SinusoidalPositionalEncoding(D_MODEL, max_len=MAX_TOKENS)
    return embedding.eval(), pos.eval(), block.eval()


def compute_attention(text: str, num_heads: int = 4, use_positional_encoding: bool = True) -> Dict[str, object]:
    """Return tokens and per-head attention for ``text``.

    Raises:
        ValueError: empty text, too many words, or an unsupported ``num_heads``.
    """
    if num_heads not in (1, 2, 4):
        raise ValueError("num_heads must be 1, 2 or 4")
    tokens = tokenize(text)
    embedding, pos, block = _model(num_heads)
    ids = torch.tensor([[token_bucket(t) for t in tokens]])  # [1, T]
    with torch.inference_mode():
        x = embedding(ids)                                   # [1, T, D]
        if use_positional_encoding:
            x = pos(x)
        _, weights = block(x)                                # [1, H, T, T]
    attn = weights[0]                                        # [H, T, T]
    row_err = float((attn.sum(dim=-1) - 1.0).abs().max())
    return {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "num_heads": num_heads,
        "d_model": D_MODEL,
        "head_dim": D_MODEL // num_heads,
        "use_positional_encoding": use_positional_encoding,
        "attention_shape": list(attn.shape),
        "attention": attn.tolist(),
        "mean_attention": attn.mean(dim=0).tolist(),
        "max_row_sum_error": row_err,
        "disclaimer": DISCLAIMER,
    }
