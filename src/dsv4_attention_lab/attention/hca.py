"""Heavily Compressed Attention inspired by DeepSeek-V4."""

from __future__ import annotations

import torch
from torch import nn

from ..norms import RMSNorm
from ..rope import apply_rope
from ..utils import AttentionOutput, estimate_kv_cache_bytes
from .common import dense_compressed_attention, local_sliding_attention, merge_heads, reshape_heads, resolve_rope_dim
from .csa import GroupedOutputProjection
from .kv_compression import WeightedKVCompressor


class HeavilyCompressedAttention(nn.Module):
    """DeepSeek-V4-inspired HCA reference module.

    HCA uses a larger compression ratio than CSA and applies dense attention over
    compressed entries. It does not perform top-k sparse selection. Attention
    sink is approximated as a learnable dummy key/value slot.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        compression_ratio: int = 16,
        window_size: int = 128,
        query_compression_dim: int | None = None,
        use_qk_rmsnorm: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int | None = None,
        use_attention_sink: bool = False,
        dropout: float = 0.0,
        shared_compressed_kv: bool = True,
        num_output_groups: int = 1,
        group_intermediate_dim: int | None = None,
    ) -> None:
        super().__init__()
        del query_compression_dim
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is omitted")
            head_dim = hidden_size // num_heads
        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.compression_ratio = compression_ratio
        self.window_size = window_size
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.rope_dim = resolve_rope_dim(head_dim, use_partial_rope, rope_dim)
        self.use_attention_sink = use_attention_sink
        self.shared_compressed_kv = shared_compressed_kv
        self.num_kv_heads = 1 if shared_compressed_kv else num_heads

        self.q_proj = nn.Linear(hidden_size, self.inner_dim)
        self.kv_compressor = WeightedKVCompressor(hidden_size, compression_ratio, method="learned")
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim)
        self.local_k_proj = nn.Linear(hidden_size, self.inner_dim)
        self.local_v_proj = nn.Linear(hidden_size, self.inner_dim)
        self.q_norm = RMSNorm(head_dim) if use_qk_rmsnorm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if use_qk_rmsnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        if use_attention_sink:
            self.sink_key = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.sink_value = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.register_parameter("sink_key", None)
            self.register_parameter("sink_value", None)

        if num_output_groups > 1:
            self.out_proj = GroupedOutputProjection(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_output_groups=num_output_groups,
                group_intermediate_dim=group_intermediate_dim,
            )
            self.uses_grouped_output = True
        else:
            self.out_proj = nn.Linear(self.inner_dim, hidden_size)
            self.uses_grouped_output = False

    def _project_compressed_kv(self, compressed_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, compressed_len, _ = compressed_hidden.shape
        k = self.k_proj(compressed_hidden).view(batch_size, compressed_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(compressed_hidden).view(batch_size, compressed_len, self.num_kv_heads, self.head_dim)
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        return_kv_cache_estimate: bool = False,
    ) -> torch.Tensor | AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape
        positions = torch.arange(seq_len, device=hidden_states.device)

        # Query stream: [B, T, hidden] -> [B, H, T, D]
        q = reshape_heads(self.q_proj(hidden_states), self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(q, positions=positions, rope_dim=self.rope_dim)

        # Heavily compressed stream: [B, T, hidden] -> [B, Cc, hidden] -> [B, KVH, Cc, D]
        compressed = self.kv_compressor(hidden_states, return_info=True)
        compressed_k, compressed_v = self._project_compressed_kv(compressed.entries)
        compressed_k = self.k_norm(compressed_k)
        compressed_k = apply_rope(compressed_k, positions=compressed.block_ends, rope_dim=self.rope_dim)

        compressed_context, weights = dense_compressed_attention(
            q=q,
            k=compressed_k,
            v=compressed_v,
            block_ends=compressed.block_ends,
            causal=True,
            dropout=self.dropout,
            training=self.training,
            sink_key=self.sink_key,
            sink_value=self.sink_value,
            query_positions=positions,
        )

        branches = [compressed_context]
        if self.window_size >= 0:
            local_k = reshape_heads(self.local_k_proj(hidden_states), self.num_heads, self.head_dim)
            local_v = reshape_heads(self.local_v_proj(hidden_states), self.num_heads, self.head_dim)
            local_k = self.k_norm(local_k)
            local_k = apply_rope(local_k, positions=positions, rope_dim=self.rope_dim)
            local_context, _ = local_sliding_attention(
                q=q,
                k=local_k,
                v=local_v,
                window_size=self.window_size,
                dropout=self.dropout,
                training=self.training,
            )
            branches.append(local_context)

        merged_heads = torch.stack(branches, dim=0).mean(dim=0)
        output = self.out_proj(merged_heads) if self.uses_grouped_output else self.out_proj(merge_heads(merged_heads))

        if output_attentions or return_kv_cache_estimate:
            cache_bytes = None
            if return_kv_cache_estimate:
                cache_bytes = estimate_kv_cache_bytes(
                    batch_size,
                    seq_len,
                    self.num_kv_heads,
                    self.head_dim,
                    hidden_states.dtype,
                    compression_ratio=self.compression_ratio,
                )
                if self.window_size >= 0:
                    cache_bytes += estimate_kv_cache_bytes(
                        batch_size,
                        min(seq_len, self.window_size + 1),
                        self.num_heads,
                        self.head_dim,
                        hidden_states.dtype,
                    )
            return AttentionOutput(output=output, attention_weights=weights if output_attentions else None, kv_cache_bytes=cache_bytes)
        return output

