"""Internal helpers shared by reference attention modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ..masks import make_compressed_causal_mask, make_sliding_window_causal_mask
from ..utils import masked_softmax


def reshape_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Convert [batch, seq_len, heads * head_dim] to [batch, heads, seq_len, head_dim]."""

    batch_size, seq_len, _ = x.shape
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert [batch, heads, seq_len, head_dim] to [batch, seq_len, heads * head_dim]."""

    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def expand_kv_heads(kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Expand shared KV heads to query heads when needed."""

    if kv.size(1) == num_heads:
        return kv
    if kv.size(1) != 1:
        raise ValueError("KV tensor must have one head or match query heads")
    return kv.expand(kv.size(0), num_heads, kv.size(2), kv.size(3))


def local_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    dropout: nn.Dropout,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply local causal attention in head space."""

    seq_len = q.size(2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = make_sliding_window_causal_mask(seq_len, window_size, q.device).view(1, 1, seq_len, seq_len)
    weights = masked_softmax(scores, mask, dim=-1)
    weights = dropout(weights) if training else weights
    return torch.matmul(weights, v), weights


def dense_compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ends: torch.Tensor,
    causal: bool,
    dropout: nn.Dropout,
    training: bool,
    sink_key: Optional[torch.Tensor] = None,
    sink_value: Optional[torch.Tensor] = None,
    query_positions: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense attention from queries to compressed KV entries."""

    batch_size, num_heads, seq_len, head_dim = q.shape
    k = expand_kv_heads(k, num_heads)
    v = expand_kv_heads(v, num_heads)
    if sink_key is not None and sink_value is not None:
        sink_k = sink_key.view(1, num_heads, 1, head_dim).expand(batch_size, num_heads, 1, head_dim)
        sink_v = sink_value.view(1, num_heads, 1, head_dim).expand(batch_size, num_heads, 1, head_dim)
        k = torch.cat([sink_k, k], dim=2)
        v = torch.cat([sink_v, v], dim=2)
        sink_end = torch.full((1,), -1, device=block_ends.device, dtype=block_ends.dtype)
        block_ends = torch.cat([sink_end, block_ends], dim=0)

    compressed_len = k.size(2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        if query_positions is None:
            query_positions = torch.arange(seq_len, device=q.device)
        visible = make_compressed_causal_mask(query_positions, block_ends).view(1, 1, seq_len, compressed_len)
    else:
        visible = torch.ones(seq_len, compressed_len, device=q.device, dtype=torch.bool).view(
            1, 1, seq_len, compressed_len
        )
    weights = masked_softmax(scores, visible, dim=-1)
    weights = dropout(weights) if training else weights
    return torch.matmul(weights, v), weights


def resolve_rope_dim(head_dim: int, use_partial_rope: bool, rope_dim: Optional[int]) -> int:
    """Resolve a valid even RoPE dimension."""

    if rope_dim is not None:
        resolved = rope_dim
    elif use_partial_rope:
        resolved = max(2, (head_dim // 2) // 2 * 2)
    else:
        resolved = head_dim
    if resolved % 2 != 0:
        resolved -= 1
    if resolved <= 0:
        raise ValueError("Resolved rope_dim must be positive and even")
    if resolved > head_dim:
        raise ValueError("rope_dim cannot exceed head_dim")
    return resolved

