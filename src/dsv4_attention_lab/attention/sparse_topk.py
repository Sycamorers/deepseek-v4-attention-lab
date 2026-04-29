"""Reference top-k sparse attention over compressed KV entries."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ..masks import make_compressed_causal_mask
from ..utils import AttentionOutput, masked_softmax
from .common import expand_kv_heads


class SparseTopKAttention(nn.Module):
    """DSA-like top-k sparse attention over compressed key/value entries.

    This module intentionally favors readability over speed. It computes index
    scores, selects top-k visible compressed entries per query, gathers those
    entries, then applies normal attention over the selected subset.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        top_k: int,
        indexer_dim: Optional[int] = None,
        causal: bool = True,
        include_sliding_window: bool = False,
        attention_sink: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.indexer_dim = indexer_dim or head_dim
        self.causal = causal
        self.include_sliding_window = include_sliding_window
        self.attention_sink = attention_sink
        self.q_index = nn.Linear(head_dim, self.indexer_dim, bias=False) if self.indexer_dim != head_dim else nn.Identity()
        self.k_index = nn.Linear(head_dim, self.indexer_dim, bias=False) if self.indexer_dim != head_dim else nn.Identity()
        if attention_sink:
            self.sink_key = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.sink_value = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.register_parameter("sink_key", None)
            self.register_parameter("sink_value", None)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        compressed_k: torch.Tensor,
        compressed_v: torch.Tensor,
        *,
        block_ends: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_selected_indices: bool = False,
    ) -> torch.Tensor | AttentionOutput:
        if q.dim() != 4:
            raise ValueError("q must have shape [batch, heads, seq_len, head_dim]")
        if compressed_k.dim() == 3:
            compressed_k = compressed_k.unsqueeze(1)
            compressed_v = compressed_v.unsqueeze(1)
        if compressed_k.dim() != 4 or compressed_v.dim() != 4:
            raise ValueError("compressed_k and compressed_v must be [batch, heads, compressed_len, head_dim]")

        batch_size, num_heads, seq_len, head_dim = q.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("q shape does not match configured num_heads/head_dim")
        compressed_k = expand_kv_heads(compressed_k, num_heads)
        compressed_v = expand_kv_heads(compressed_v, num_heads)
        compressed_len = compressed_k.size(2)
        if block_ends is None:
            block_ends = torch.arange(compressed_len, device=q.device)
        else:
            block_ends = block_ends.to(device=q.device)

        if self.attention_sink:
            sink_k = self.sink_key.view(1, num_heads, 1, head_dim).expand(batch_size, num_heads, 1, head_dim)
            sink_v = self.sink_value.view(1, num_heads, 1, head_dim).expand(batch_size, num_heads, 1, head_dim)
            compressed_k = torch.cat([sink_k, compressed_k], dim=2)
            compressed_v = torch.cat([sink_v, compressed_v], dim=2)
            sink_end = torch.full((1,), -1, device=q.device, dtype=block_ends.dtype)
            block_ends = torch.cat([sink_end, block_ends], dim=0)
            compressed_len += 1

        if compressed_len == 0:
            output = torch.zeros_like(q)
            if output_attentions or return_selected_indices:
                return AttentionOutput(output=output)
            return output

        q_index = self.q_index(q)
        k_index = self.k_index(compressed_k)
        index_scores = torch.matmul(q_index, k_index.transpose(-2, -1)) / math.sqrt(self.indexer_dim)

        if self.causal:
            if query_positions is None:
                query_positions = torch.arange(seq_len, device=q.device)
            else:
                query_positions = query_positions.to(device=q.device)
            visible = make_compressed_causal_mask(query_positions, block_ends).view(1, 1, seq_len, compressed_len)
        else:
            visible = torch.ones(seq_len, compressed_len, device=q.device, dtype=torch.bool).view(
                1, 1, seq_len, compressed_len
            )

        k_eff = min(self.top_k, compressed_len)
        min_value = torch.finfo(index_scores.dtype).min
        masked_index_scores = index_scores.masked_fill(~visible, min_value)
        selected_indices = torch.topk(masked_index_scores, k=k_eff, dim=-1).indices
        selected_visible = torch.gather(
            visible.expand(batch_size, num_heads, seq_len, compressed_len),
            dim=-1,
            index=selected_indices,
        )

        gather_index = selected_indices.unsqueeze(-1).expand(batch_size, num_heads, seq_len, k_eff, head_dim)
        k_expanded = compressed_k.unsqueeze(2).expand(batch_size, num_heads, seq_len, compressed_len, head_dim)
        v_expanded = compressed_v.unsqueeze(2).expand(batch_size, num_heads, seq_len, compressed_len, head_dim)
        selected_k = torch.gather(k_expanded, dim=3, index=gather_index)
        selected_v = torch.gather(v_expanded, dim=3, index=gather_index)

        attention_scores = torch.sum(q.unsqueeze(3) * selected_k, dim=-1) / math.sqrt(head_dim)
        weights = masked_softmax(attention_scores, selected_visible, dim=-1)
        weights = self.dropout(weights) if self.training else weights
        output = torch.sum(weights.unsqueeze(-1) * selected_v, dim=3)

        if output_attentions or return_selected_indices:
            return AttentionOutput(
                output=output,
                attention_weights=weights if output_attentions else None,
                selected_indices=selected_indices if return_selected_indices else None,
            )
        return output

