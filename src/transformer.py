"""Hand-written Transformer encoder building blocks (PyTorch, CPU-friendly).

Everything here is built from primitive tensor ops and ``nn.Linear`` /
``nn.LayerNorm`` -- no ``nn.MultiheadAttention`` or ``nn.TransformerEncoder`` --
so each step of the math is visible.

Shape notation used in docstrings::

    B = batch size, T = sequence length (queries), S = key/value length
    H = number of heads, D = d_model, Dh = D / H (per-head width)

Mask convention: a boolean tensor where ``True`` means "this query MAY attend to
this key" and ``False`` means "blocked". Masks must be broadcastable to the
score shape ``[B, H, T, S]`` (e.g. ``[B, 1, 1, S]`` for key padding or
``[T, S]`` for a causal mask). A query whose keys are *all* blocked gets
all-zero attention weights and a zero output vector (no NaNs).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``softmax(Q K^T / sqrt(Dh)) V``.

    Args:
        q: Queries, ``[..., T, Dh]``.
        k: Keys, ``[..., S, Dh]``.
        v: Values, ``[..., S, Dv]``.
        mask: Optional boolean mask broadcastable to ``[..., T, S]``;
            ``True`` = allowed, ``False`` = blocked.

    Returns:
        ``(output [..., T, Dv], weights [..., T, S])``. ``weights`` sums to 1
        over the last (key) dimension, except for a query whose keys are all
        masked: its weights are all zero, so its output row is zero too.
    """
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)  # [..., T, S]
    any_allowed: Optional[torch.Tensor] = None
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be a bool tensor, got {mask.dtype}")
        # A large finite negative (not -inf) keeps the softmax NaN-free ...
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        any_allowed = torch.broadcast_to(mask, scores.shape).any(dim=-1, keepdim=True)
    weights = torch.softmax(scores, dim=-1)
    if any_allowed is not None:
        # ... and rows with no allowed key (softmax would be uniform) are zeroed.
        weights = weights.masked_fill(~any_allowed, 0.0)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V and output projections.

    Args:
        d_model: Model width ``D``. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads ``H``.

    Raises:
        ValueError: if ``num_heads`` or ``d_model`` is not a positive integer,
            or ``d_model % num_heads != 0``.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, D] -> [B, H, T, Dh]``."""
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, H, T, Dh] -> [B, T, D]``."""
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention.

        Args:
            x: ``[B, T, D]``.
            mask: Optional boolean mask broadcastable to ``[B, H, T, T]``
                (``True`` = may attend).

        Returns:
            ``(output [B, T, D], attention weights [B, H, T, T])``.
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"expected input of shape [B, T, {self.d_model}], got {tuple(x.shape)}"
            )
        q = self._split_heads(self.q_proj(x))  # [B, H, T, Dh]
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        ctx, weights = scaled_dot_product_attention(q, k, v, mask)  # [B,H,T,Dh], [B,H,T,T]
        return self.out_proj(self._merge_heads(ctx)), weights


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    ``PE[pos, 2i]   = sin(pos / 10000^(2i/D))`` (even dimensions)
    ``PE[pos, 2i+1] = cos(pos / 10000^(2i/D))`` (odd dimensions)

    The table is a non-trainable buffer; it is added to the input, not learned.

    Args:
        d_model: Model width ``D`` (odd widths are supported).
        max_len: Maximum supported sequence length.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        even = torch.arange(0, d_model, 2, dtype=torch.float32)             # [ceil(D/2)]
        inv_freq = torch.exp(-math.log(10000.0) * even / d_model)           # [ceil(D/2)]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * inv_freq)
        pe[:, 1::2] = torch.cos(position * inv_freq[: d_model // 2])
        self.register_buffer("pe", pe, persistent=False)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings. ``x: [B, T, D] -> [B, T, D]``."""
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.max_len}")
        return x + self.pe[:t].unsqueeze(0).to(x.dtype)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm encoder block.

    ``x = x + MHA(LayerNorm(x))`` then ``x = x + MLP(LayerNorm(x))``.

    Args:
        d_model: Model width ``D``.
        num_heads: Attention heads (``d_model`` must be divisible by it).
        d_ff: Hidden width of the feed-forward MLP.
        dropout: Dropout probability on sublayer outputs (0 disables it).
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x: [B, T, D] -> (output [B, T, D], attention [B, H, T, T])``."""
        a, weights = self.attn(self.norm1(x), mask)
        x = x + self.dropout(a)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, weights


class TransformerEncoder(nn.Module):
    """Stack of pre-norm blocks with optional sinusoidal positional encoding.

    Operates on already-embedded inputs ``[B, T, D]``. With
    ``use_positional_encoding=False`` and no mask the encoder is
    permutation-equivariant over the sequence dimension.

    Returns ``(output [B, T, D], [attention [B, H, T, T] per layer])``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_len: int = 512,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers!r}")
        self.pos_enc: Optional[SinusoidalPositionalEncoding] = (
            SinusoidalPositionalEncoding(d_model, max_len)
            if use_positional_encoding
            else None
        )
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        attentions = []
        for layer in self.layers:
            x, w = layer(x, mask)
            attentions.append(w)
        return self.final_norm(x), attentions
