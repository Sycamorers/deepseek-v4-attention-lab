"""Synthetic next-token prediction tasks for tiny LM experiments."""

from __future__ import annotations

from typing import Literal

import torch


TaskName = Literal["local", "retrieval"]


def make_local_dependency_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a task where the next token mostly depends on recent context."""

    tokens = torch.empty(batch_size, seq_len + 1, device=device, dtype=torch.long)
    tokens[:, 0] = torch.randint(4, vocab_size, (batch_size,), device=device)
    increments = torch.randint(1, 4, (batch_size, seq_len), device=device)
    for idx in range(1, seq_len + 1):
        tokens[:, idx] = ((tokens[:, idx - 1] + increments[:, idx - 1] - 4) % (vocab_size - 4)) + 4
    return tokens[:, :-1], tokens[:, 1:]


def make_retrieval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a long-range marker retrieval task."""

    tokens = torch.randint(4, vocab_size, (batch_size, seq_len + 1), device=device)
    key = torch.randint(4, vocab_size, (batch_size,), device=device)
    tokens[:, 0] = 1
    tokens[:, 1] = key
    if seq_len > 16:
        tokens[:, seq_len // 2] = 2
        tokens[:, seq_len // 2 + 1] = key
        tokens[:, -2] = 3
        tokens[:, -1] = key
    return tokens[:, :-1], tokens[:, 1:]


def make_batch(
    task: TaskName,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch synthetic task batch creation."""

    if task == "local":
        return make_local_dependency_batch(batch_size, seq_len, vocab_size, device)
    if task == "retrieval":
        return make_retrieval_batch(batch_size, seq_len, vocab_size, device)
    raise ValueError(f"Unsupported task: {task}")

