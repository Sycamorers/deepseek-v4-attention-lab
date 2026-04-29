"""Compressed Sparse Attention inspired by DeepSeek-V4."""

from __future__ import annotations

import torch
from torch import nn

from ..norms import RMSNorm
from ..rope import apply_rope
from ..utils import AttentionOutput, estimate_kv_cache_bytes, module_output_tensor
from .common import local_sliding_attention, merge_heads, reshape_heads, resolve_rope_dim
from .kv_compression import WeightedKVCompressor
from .sparse_topk import SparseTopKAttention


class GroupedOutputProjection(nn.Module):
    """Optional grouped output projection for attention head outputs."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_output_groups: int,
        group_intermediate_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_output_groups <= 0:
            raise ValueError("num_output_groups must be positive")
        if num_heads % num_output_groups != 0:
            raise ValueError("num_heads must be divisible by num_output_groups")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_output_groups = num_output_groups
        self.heads_per_group = num_heads // num_output_groups
        group_dim = self.heads_per_group * head_dim
        intermediate = group_intermediate_dim or max(group_dim, hidden_size // num_output_groups)
        self.groups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, intermediate),
                    nn.GELU(),
                    nn.Linear(intermediate, hidden_size),
                )
                for _ in range(num_output_groups)
            ]
        )

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        # [B, H, T, D] -> [B, T, H, D]
        batch_size, _, seq_len, _ = head_outputs.shape
        by_token = head_outputs.transpose(1, 2).contiguous()
        chunks = by_token.view(
            batch_size,
            seq_len,
            self.num_output_groups,
            self.heads_per_group * self.head_dim,
        )
        projected = [layer(chunks[:, :, group_idx, :]) for group_idx, layer in enumerate(self.groups)]
        return torch.stack(projected, dim=0).mean(dim=0)


class CompressedSparseAttention(nn.Module):
    """DeepSeek-V4-inspired CSA reference module.

    The implementation compresses hidden states into block-level KV entries,
    applies DSA-like top-k sparse attention over the compressed entries, and can
    add a local sliding-window branch over uncompressed tokens. Attention sink is
    approximated as a learnable dummy key/value slot.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        compression_ratio: int = 4,
        top_k: int = 8,
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
        self.top_k = top_k
        self.window_size = window_size
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.use_partial_rope = use_partial_rope
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
        self.sparse_attention = SparseTopKAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            indexer_dim=query_compression_dim,
            causal=True,
            include_sliding_window=window_size > 0,
            attention_sink=use_attention_sink,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
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
        # [B, Cc, hidden] -> [B, KVH, Cc, D]
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

        # Compressed stream: [B, T, hidden] -> [B, Cc, hidden] -> [B, KVH, Cc, D]
        compressed = self.kv_compressor(hidden_states, return_info=True)
        compressed_k, compressed_v = self._project_compressed_kv(compressed.entries)
        compressed_k = self.k_norm(compressed_k)
        compressed_k = apply_rope(compressed_k, positions=compressed.block_ends, rope_dim=self.rope_dim)

        sparse_result = self.sparse_attention(
            q,
            compressed_k,
            compressed_v,
            block_ends=compressed.block_ends,
            query_positions=positions,
            output_attentions=output_attentions,
            return_selected_indices=output_attentions,
        )
        compressed_context = module_output_tensor(sparse_result)

        branches = [compressed_context]
        attention_weights = sparse_result.attention_weights if isinstance(sparse_result, AttentionOutput) else None
        selected_indices = sparse_result.selected_indices if isinstance(sparse_result, AttentionOutput) else None

        if self.window_size >= 0:
            # Local branch: [B, T, hidden] -> [B, H, T, D], attending over uncompressed recent tokens.
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
            return AttentionOutput(
                output=output,
                attention_weights=attention_weights,
                kv_cache_bytes=cache_bytes,
                selected_indices=selected_indices,
            )
        return output

