"""Boolean attention masks used by the reference implementations."""

from __future__ import annotations

import torch


def make_causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device | str,
    q_positions: torch.Tensor | None = None,
    kv_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [q_len, kv_len] with True where a key is visible to a query."""

    if q_positions is None:
        q_positions = torch.arange(q_len, device=device)
    else:
        q_positions = q_positions.to(device=device)
    if kv_positions is None:
        kv_positions = torch.arange(kv_len, device=device)
    else:
        kv_positions = kv_positions.to(device=device)
    return kv_positions.view(1, kv_len) <= q_positions.view(q_len, 1)


def make_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a local causal mask.

    Each query can see itself and the previous `window_size` tokens, so the
    maximum visible span is `window_size + 1`.
    """

    positions = torch.arange(seq_len, device=device)
    q_pos = positions.view(seq_len, 1)
    kv_pos = positions.view(1, seq_len)
    return (kv_pos <= q_pos) & (kv_pos >= (q_pos - window_size))


def make_compressed_causal_mask(
    query_positions: torch.Tensor,
    block_ends: torch.Tensor,
) -> torch.Tensor:
    """Return [seq_len, compressed_len] visibility for compressed KV blocks."""

    return block_ends.view(1, -1).to(query_positions.device) <= query_positions.view(-1, 1)

