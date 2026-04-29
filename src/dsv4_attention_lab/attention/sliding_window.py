"""Local causal sliding-window attention baseline."""

from __future__ import annotations

import torch
from torch import nn

from ..rope import apply_rope
from ..utils import AttentionOutput, estimate_kv_cache_bytes
from .common import local_sliding_attention, merge_heads, reshape_heads


def build_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Construct [seq_len, seq_len] local causal visibility mask."""

    from ..masks import make_sliding_window_causal_mask

    return make_sliding_window_causal_mask(seq_len, window_size, device)


class SlidingWindowAttention(nn.Module):
    """Causal attention where each query sees only a recent local window."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        window_size: int = 128,
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is omitted")
            head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.window_size = window_size
        self.use_rope = use_rope

        self.q_proj = nn.Linear(hidden_size, self.inner_dim)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        return_kv_cache_estimate: bool = False,
    ) -> torch.Tensor | AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape

        # [B, T, C] -> [B, H, T, D]
        q = reshape_heads(self.q_proj(hidden_states), self.num_heads, self.head_dim)
        k = reshape_heads(self.k_proj(hidden_states), self.num_heads, self.head_dim)
        v = reshape_heads(self.v_proj(hidden_states), self.num_heads, self.head_dim)

        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        context, weights = local_sliding_attention(
            q=q,
            k=k,
            v=v,
            window_size=self.window_size,
            dropout=self.dropout,
            training=self.training,
        )
        output = self.out_proj(merge_heads(context))

        if output_attentions or return_kv_cache_estimate:
            cache_tokens = min(seq_len, self.window_size + 1)
            cache_bytes = None
            if return_kv_cache_estimate:
                cache_bytes = estimate_kv_cache_bytes(
                    batch_size, cache_tokens, self.num_heads, self.head_dim, hidden_states.dtype
                )
            return AttentionOutput(output=output, attention_weights=weights if output_attentions else None, kv_cache_bytes=cache_bytes)
        return output

