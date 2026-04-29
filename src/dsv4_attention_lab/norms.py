"""Normalization layers used by the reference attention modules."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        return x_norm * self.weight


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm for tensors shaped [batch, heads, seq_len, head_dim]."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("HeadwiseRMSNorm expects [batch, heads, seq_len, head_dim]")
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        return x_norm * self.weight.view(1, self.num_heads, 1, self.head_dim)

