"""Standard causal multi-head attention baseline."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..masks import make_causal_mask
from ..rope import apply_rope
from ..utils import AttentionOutput, estimate_kv_cache_bytes, masked_softmax
from .common import merge_heads, reshape_heads


class DenseMHA(nn.Module):
    """Readable PyTorch implementation of dense multi-head self-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is omitted")
            head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.use_rope = use_rope
        self.causal = causal

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = None
        if self.causal:
            mask = make_causal_mask(seq_len, seq_len, hidden_states.device).view(1, 1, seq_len, seq_len)
        weights = masked_softmax(scores, mask, dim=-1)
        weights = self.dropout(weights) if self.training else weights

        # [B, H, T, D] -> [B, T, C]
        context = torch.matmul(weights, v)
        output = self.out_proj(merge_heads(context))

        if output_attentions or return_kv_cache_estimate:
            cache_bytes = None
            if return_kv_cache_estimate:
                cache_bytes = estimate_kv_cache_bytes(
                    batch_size, seq_len, self.num_heads, self.head_dim, hidden_states.dtype
                )
            return AttentionOutput(output=output, attention_weights=weights if output_attentions else None, kv_cache_bytes=cache_bytes)
        return output

