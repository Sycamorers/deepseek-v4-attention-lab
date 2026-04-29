"""Configuration dataclasses for attention modules and toy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AttentionConfig:
    """Common attention configuration used by examples and experiments."""

    hidden_size: int = 128
    num_heads: int = 4
    head_dim: Optional[int] = None
    dropout: float = 0.0
    causal: bool = True
    use_rope: bool = True
    use_partial_rope: bool = False
    rope_dim: Optional[int] = None
    use_qk_rmsnorm: bool = False
    compression_ratio: int = 4
    hca_compression_ratio: int = 16
    top_k: int = 8
    window_size: int = 128
    query_compression_dim: Optional[int] = None
    use_attention_sink: bool = False
    shared_compressed_kv: bool = True
    num_output_groups: int = 1
    group_intermediate_dim: Optional[int] = None
    hybrid_pattern: Literal[
        "alternating", "csa_only", "hca_only", "swa_then_alternating"
    ] = "alternating"
    hybrid_warmup_layers: int = 1

    def resolved_head_dim(self) -> int:
        """Return the explicit head dimension, validating divisibility if omitted."""

        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is omitted")
        return self.hidden_size // self.num_heads


@dataclass
class TinyTransformerConfig:
    """Configuration for the toy decoder-only Transformer."""

    vocab_size: int = 128
    max_seq_len: int = 1024
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    head_dim: Optional[int] = None
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_type: Literal["dense", "sliding_window", "csa", "hca", "hybrid"] = "dense"
    compression_ratio: int = 4
    hca_compression_ratio: int = 16
    top_k: int = 8
    window_size: int = 128
    use_qk_rmsnorm: bool = True
    use_partial_rope: bool = True
    rope_dim: Optional[int] = None
    use_attention_sink: bool = False
    shared_compressed_kv: bool = True
    hybrid_pattern: Literal[
        "alternating", "csa_only", "hca_only", "swa_then_alternating"
    ] = "swa_then_alternating"
    hybrid_warmup_layers: int = 1

    def resolved_head_dim(self) -> int:
        """Return the explicit head dimension, validating divisibility if omitted."""

        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is omitted")
        return self.hidden_size // self.num_heads

