"""Rotary positional embeddings with optional partial application."""

from __future__ import annotations

from typing import Optional

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
    base: float = 10000.0,
    positions: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cosine and sine caches shaped [seq_len, dim]."""

    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    if positions is None:
        positions = torch.arange(seq_len, device=device)
    else:
        positions = positions.to(device=device)
    positions = positions.to(dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    rope_dim: Optional[int] = None,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to [batch, heads, seq_len, head_dim].

    If `rope_dim` is smaller than the head dimension, RoPE is applied only to
    the last `rope_dim` features and the leading features are passed through.
    """

    if x.dim() != 4:
        raise ValueError("apply_rope expects [batch, heads, seq_len, head_dim]")
    head_dim = x.size(-1)
    rope_dim = head_dim if rope_dim is None else rope_dim
    if rope_dim == 0:
        return x
    if rope_dim > head_dim:
        raise ValueError("rope_dim cannot exceed head_dim")
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")

    prefix_dim = head_dim - rope_dim
    prefix = x[..., :prefix_dim] if prefix_dim > 0 else None
    rotary = x[..., prefix_dim:]
    seq_len = rotary.size(-2)
    cos, sin = build_rope_cache(
        seq_len=seq_len,
        dim=rope_dim,
        device=x.device,
        dtype=x.dtype,
        base=base,
        positions=positions,
    )
    cos = cos.view(1, 1, seq_len, rope_dim)
    sin = sin.view(1, 1, seq_len, rope_dim)
    rotated = (rotary * cos) + (_rotate_half(rotary) * sin)
    if prefix is None:
        return rotated
    return torch.cat([prefix, rotated], dim=-1)

