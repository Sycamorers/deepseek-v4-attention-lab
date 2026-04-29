"""Helpers for interleaving CSA, HCA, and sliding-window layers."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..utils import module_output_tensor
from .csa import CompressedSparseAttention
from .hca import HeavilyCompressedAttention
from .sliding_window import SlidingWindowAttention


HybridPattern = Literal["alternating", "csa_only", "hca_only", "swa_then_alternating"]


def _layer_kind(layer_idx: int, pattern: HybridPattern, warmup_layers: int) -> str:
    if pattern == "csa_only":
        return "csa"
    if pattern == "hca_only":
        return "hca"
    if pattern == "alternating":
        return "csa" if layer_idx % 2 == 0 else "hca"
    if pattern == "swa_then_alternating":
        if layer_idx < warmup_layers:
            return "swa"
        return "csa" if (layer_idx - warmup_layers) % 2 == 0 else "hca"
    raise ValueError(f"Unsupported hybrid pattern: {pattern}")


def build_hybrid_attention_layers(
    *,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int | None = None,
    pattern: HybridPattern = "alternating",
    warmup_layers: int = 1,
    compression_ratio: int = 4,
    hca_compression_ratio: int = 16,
    top_k: int = 8,
    window_size: int = 128,
    dropout: float = 0.0,
    use_qk_rmsnorm: bool = True,
    use_partial_rope: bool = True,
    rope_dim: int | None = None,
    use_attention_sink: bool = False,
    shared_compressed_kv: bool = True,
) -> tuple[nn.ModuleList, list[str]]:
    """Build an interleaved stack of attention modules and return layer labels."""

    layers: list[nn.Module] = []
    layer_types: list[str] = []
    for layer_idx in range(num_layers):
        kind = _layer_kind(layer_idx, pattern, warmup_layers)
        layer_types.append(kind)
        if kind == "swa":
            layers.append(
                SlidingWindowAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    window_size=window_size,
                    dropout=dropout,
                    use_rope=True,
                )
            )
        elif kind == "csa":
            layers.append(
                CompressedSparseAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    compression_ratio=compression_ratio,
                    top_k=top_k,
                    window_size=window_size,
                    use_qk_rmsnorm=use_qk_rmsnorm,
                    use_partial_rope=use_partial_rope,
                    rope_dim=rope_dim,
                    use_attention_sink=use_attention_sink,
                    dropout=dropout,
                    shared_compressed_kv=shared_compressed_kv,
                )
            )
        else:
            layers.append(
                HeavilyCompressedAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    compression_ratio=hca_compression_ratio,
                    window_size=window_size,
                    use_qk_rmsnorm=use_qk_rmsnorm,
                    use_partial_rope=use_partial_rope,
                    rope_dim=rope_dim,
                    use_attention_sink=use_attention_sink,
                    dropout=dropout,
                    shared_compressed_kv=shared_compressed_kv,
                )
            )
    return nn.ModuleList(layers), layer_types


class HybridCSAHLayers(nn.Module):
    """A minimal residual stack of interleaved CSA/HCA/SWA attention layers."""

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        pattern: HybridPattern = "alternating",
        warmup_layers: int = 1,
        compression_ratio: int = 4,
        hca_compression_ratio: int = 16,
        top_k: int = 8,
        window_size: int = 128,
        dropout: float = 0.0,
        use_qk_rmsnorm: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int | None = None,
        use_attention_sink: bool = False,
        shared_compressed_kv: bool = True,
    ) -> None:
        super().__init__()
        self.layers, self.layer_types = build_hybrid_attention_layers(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            pattern=pattern,
            warmup_layers=warmup_layers,
            compression_ratio=compression_ratio,
            hca_compression_ratio=hca_compression_ratio,
            top_k=top_k,
            window_size=window_size,
            dropout=dropout,
            use_qk_rmsnorm=use_qk_rmsnorm,
            use_partial_rope=use_partial_rope,
            rope_dim=rope_dim,
            use_attention_sink=use_attention_sink,
            shared_compressed_kv=shared_compressed_kv,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for layer in self.layers:
            x = x + module_output_tensor(layer(x))
        return x

