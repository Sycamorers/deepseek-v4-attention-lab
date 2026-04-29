"""Small utilities shared by modules, benchmarks, and experiments."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class AttentionOutput:
    """Optional rich output returned when callers request diagnostics."""

    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    kv_cache_bytes: Optional[int] = None
    selected_indices: Optional[torch.Tensor] = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_string(name: str) -> torch.dtype:
    """Map a CLI dtype string to a PyTorch dtype."""

    normalized = name.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes used by a tensor."""

    return tensor.numel() * tensor.element_size()


def estimate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    compression_ratio: int = 1,
) -> int:
    """Estimate K/V cache bytes for a full or compressed cache."""

    compressed_len = math.ceil(seq_len / compression_ratio)
    element_size = torch.empty((), dtype=dtype).element_size()
    return batch_size * compressed_len * num_kv_heads * head_dim * 2 * element_size


def masked_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = -1,
) -> torch.Tensor:
    """Softmax with a boolean visibility mask and safe all-masked handling.

    `mask=True` means the score is visible. Rows with no visible entries return
    all zeros instead of NaNs or a uniform distribution over invalid positions.
    """

    if mask is None:
        return torch.softmax(scores, dim=dim)
    mask = mask.to(device=scores.device, dtype=torch.bool)
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(0)
    mask = torch.broadcast_to(mask, scores.shape)
    min_value = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, min_value)
    probs = torch.softmax(masked_scores, dim=dim)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=dim, keepdim=True)
    return torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))


def module_output_tensor(value: torch.Tensor | AttentionOutput) -> torch.Tensor:
    """Extract the tensor from a module result that may include diagnostics."""

    if isinstance(value, AttentionOutput):
        return value.output
    return value


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def dataclass_to_json_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass to a JSON-friendly dictionary."""

    if not is_dataclass(value):
        raise TypeError("value must be a dataclass instance")
    return asdict(value)


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with a stable indentation style."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

