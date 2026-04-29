"""Reference PyTorch attention modules inspired by DeepSeek-V4.

This package is educational code. It is not an official DeepSeek implementation
and does not include production CUDA, Triton, TileLang, or C++ kernels.
"""

from .configs import AttentionConfig, TinyTransformerConfig
from .attention.csa import CompressedSparseAttention
from .attention.dense_mha import DenseMHA
from .attention.hca import HeavilyCompressedAttention
from .attention.hybrid import HybridCSAHLayers, build_hybrid_attention_layers
from .attention.sliding_window import SlidingWindowAttention
from .attention.sparse_topk import SparseTopKAttention

__all__ = [
    "AttentionConfig",
    "CompressedSparseAttention",
    "DenseMHA",
    "HeavilyCompressedAttention",
    "HybridCSAHLayers",
    "SlidingWindowAttention",
    "SparseTopKAttention",
    "TinyTransformerConfig",
    "build_hybrid_attention_layers",
]

