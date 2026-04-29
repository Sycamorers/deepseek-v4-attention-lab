"""Token-level KV compression utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CompressionOutput:
    """Compressed sequence plus block metadata for causal visibility."""

    entries: torch.Tensor
    block_starts: torch.Tensor
    block_ends: torch.Tensor
    block_lengths: torch.Tensor


def _block_metadata(
    seq_len: int,
    compression_ratio: int,
    device: torch.device | str,
    drop_incomplete_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if drop_incomplete_tail:
        compressed_len = seq_len // compression_ratio
        if compressed_len == 0:
            compressed_len = 1
    else:
        compressed_len = math.ceil(seq_len / compression_ratio)
    starts = torch.arange(compressed_len, device=device) * compression_ratio
    ends = torch.minimum(starts + compression_ratio - 1, torch.tensor(seq_len - 1, device=device))
    lengths = ends - starts + 1
    return starts, ends, lengths, compressed_len


def average_pool_kv(
    x: torch.Tensor,
    compression_ratio: int,
    *,
    return_info: bool = False,
    drop_incomplete_tail: bool = False,
) -> torch.Tensor | CompressionOutput:
    """Average-pool tokens within fixed sequence blocks.

    This is parameter-free compression. For non-divisible sequence lengths, the
    final block is shorter unless `drop_incomplete_tail=True`. Causal-safe use is
    achieved by pairing entries with `block_ends`: a query at position `t` can
    only attend to compressed blocks whose `block_end <= t`.
    """

    if x.dim() != 3:
        raise ValueError("average_pool_kv expects [batch, seq_len, dim]")
    batch_size, seq_len, dim = x.shape
    starts, ends, lengths, compressed_len = _block_metadata(
        seq_len, compression_ratio, x.device, drop_incomplete_tail
    )
    effective_len = int(ends[-1].item()) + 1
    trimmed = x[:, :effective_len, :]
    pad_len = compressed_len * compression_ratio - effective_len
    if pad_len > 0:
        trimmed = F.pad(trimmed, (0, 0, 0, pad_len))
    blocks = trimmed.view(batch_size, compressed_len, compression_ratio, dim)
    token_positions = torch.arange(compression_ratio, device=x.device).view(1, 1, compression_ratio, 1)
    valid = token_positions < lengths.view(1, compressed_len, 1, 1)
    entries = (blocks * valid.to(dtype=x.dtype)).sum(dim=2) / lengths.view(1, compressed_len, 1).to(dtype=x.dtype)
    output = CompressionOutput(entries=entries, block_starts=starts, block_ends=ends, block_lengths=lengths)
    return output if return_info else output.entries


class WeightedKVCompressor(nn.Module):
    """Compress sequence blocks using average or learned softmax weights.

    `method="average"` is deterministic mean pooling.
    `method="learned"` uses a trainable linear scorer to produce one softmax
    distribution per block, then forms a weighted sum of tokens in that block.

    The module keeps incomplete tail blocks by default and returns block-end
    metadata when `return_info=True`, allowing causal-safe attention to hide a
    compressed entry until every token represented by that entry is available.
    """

    def __init__(
        self,
        dim: int,
        compression_ratio: int,
        method: Literal["average", "learned"] = "learned",
        drop_incomplete_tail: bool = False,
    ) -> None:
        super().__init__()
        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if method not in {"average", "learned"}:
            raise ValueError("method must be 'average' or 'learned'")
        self.dim = dim
        self.compression_ratio = compression_ratio
        self.method = method
        self.drop_incomplete_tail = drop_incomplete_tail
        self.score_proj = nn.Linear(dim, 1) if method == "learned" else None

    def forward(self, x: torch.Tensor, return_info: bool = False) -> torch.Tensor | CompressionOutput:
        if self.method == "average":
            return average_pool_kv(
                x,
                self.compression_ratio,
                return_info=return_info,
                drop_incomplete_tail=self.drop_incomplete_tail,
            )
        if x.dim() != 3:
            raise ValueError("WeightedKVCompressor expects [batch, seq_len, dim]")
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")

        starts, ends, lengths, compressed_len = _block_metadata(
            seq_len, self.compression_ratio, x.device, self.drop_incomplete_tail
        )
        effective_len = int(ends[-1].item()) + 1
        trimmed = x[:, :effective_len, :]
        pad_len = compressed_len * self.compression_ratio - effective_len
        if pad_len > 0:
            trimmed = F.pad(trimmed, (0, 0, 0, pad_len))

        # [B, T, D] -> [B, C, M, D], where C is compressed length and M is block size.
        blocks = trimmed.view(batch_size, compressed_len, self.compression_ratio, dim)
        scores = self.score_proj(blocks).squeeze(-1)
        token_positions = torch.arange(self.compression_ratio, device=x.device).view(1, 1, self.compression_ratio)
        valid = token_positions < lengths.view(1, compressed_len, 1)
        min_value = torch.finfo(scores.dtype).min
        weights = torch.softmax(scores.masked_fill(~valid, min_value), dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entries = torch.sum(weights.unsqueeze(-1) * blocks, dim=2)

        output = CompressionOutput(entries=entries, block_starts=starts, block_ends=ends, block_lengths=lengths)
        return output if return_info else output.entries

