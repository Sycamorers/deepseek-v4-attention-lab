"""KV-cache size estimation helpers."""

from __future__ import annotations

import math

import torch

from ..utils import estimate_kv_cache_bytes


def dense_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Estimate full dense attention KV-cache size."""

    return estimate_kv_cache_bytes(batch_size, seq_len, num_heads, head_dim, dtype)


def compressed_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    compression_ratio: int,
) -> int:
    """Estimate compressed KV-cache size."""

    return estimate_kv_cache_bytes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        compression_ratio=compression_ratio,
    )


def compressed_length(seq_len: int, compression_ratio: int) -> int:
    """Return ceil(seq_len / compression_ratio)."""

    return math.ceil(seq_len / compression_ratio)

