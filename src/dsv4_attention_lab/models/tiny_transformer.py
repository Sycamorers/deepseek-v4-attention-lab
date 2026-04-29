"""A minimal decoder-only Transformer for toy language-modeling experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..attention.csa import CompressedSparseAttention
from ..attention.dense_mha import DenseMHA
from ..attention.hca import HeavilyCompressedAttention
from ..attention.hybrid import build_hybrid_attention_layers
from ..attention.sliding_window import SlidingWindowAttention
from ..configs import TinyTransformerConfig
from ..norms import RMSNorm
from ..utils import module_output_tensor


class FeedForward(nn.Module):
    """Small GELU MLP used in the toy Transformer blocks."""

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        intermediate = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, hidden_size: int, attention: nn.Module, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attention = attention
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + module_output_tensor(self.attention(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_attention(config: TinyTransformerConfig, layer_idx: int, hybrid_layers: Optional[nn.ModuleList]) -> nn.Module:
    head_dim = config.resolved_head_dim()
    if config.attention_type == "dense":
        return DenseMHA(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            dropout=config.dropout,
            use_rope=True,
            causal=True,
        )
    if config.attention_type == "sliding_window":
        return SlidingWindowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            window_size=config.window_size,
            dropout=config.dropout,
            use_rope=True,
        )
    if config.attention_type == "csa":
        return CompressedSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            compression_ratio=config.compression_ratio,
            top_k=config.top_k,
            window_size=config.window_size,
            use_qk_rmsnorm=config.use_qk_rmsnorm,
            use_partial_rope=config.use_partial_rope,
            rope_dim=config.rope_dim,
            use_attention_sink=config.use_attention_sink,
            dropout=config.dropout,
            shared_compressed_kv=config.shared_compressed_kv,
        )
    if config.attention_type == "hca":
        return HeavilyCompressedAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            compression_ratio=config.hca_compression_ratio,
            window_size=config.window_size,
            use_qk_rmsnorm=config.use_qk_rmsnorm,
            use_partial_rope=config.use_partial_rope,
            rope_dim=config.rope_dim,
            use_attention_sink=config.use_attention_sink,
            dropout=config.dropout,
            shared_compressed_kv=config.shared_compressed_kv,
        )
    if config.attention_type == "hybrid":
        if hybrid_layers is None:
            raise ValueError("hybrid_layers must be provided for hybrid attention")
        return hybrid_layers[layer_idx]
    raise ValueError(f"Unsupported attention_type: {config.attention_type}")


class TinyTransformer(nn.Module):
    """Tiny decoder-only Transformer for sanity checks and experiments."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        hybrid_layers = None
        if config.attention_type == "hybrid":
            hybrid_layers, self.hybrid_layer_types = build_hybrid_attention_layers(
                num_layers=config.num_layers,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                head_dim=config.resolved_head_dim(),
                pattern=config.hybrid_pattern,
                warmup_layers=config.hybrid_warmup_layers,
                compression_ratio=config.compression_ratio,
                hca_compression_ratio=config.hca_compression_ratio,
                top_k=config.top_k,
                window_size=config.window_size,
                dropout=config.dropout,
                use_qk_rmsnorm=config.use_qk_rmsnorm,
                use_partial_rope=config.use_partial_rope,
                rope_dim=config.rope_dim,
                use_attention_sink=config.use_attention_sink,
                shared_compressed_kv=config.shared_compressed_kv,
            )
        else:
            self.hybrid_layer_types = []

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    attention=_build_attention(config, layer_idx, hybrid_layers),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence exceeds config.max_seq_len")

        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            output["loss"] = loss
        return output

