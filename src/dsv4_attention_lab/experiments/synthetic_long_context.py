"""Synthetic next-token prediction tasks for tiny LM experiments."""

from __future__ import annotations

from typing import Literal

import torch


IGNORE_INDEX = -100
TaskName = Literal["local", "retrieval", "copy_first", "associative_recall", "multi_query_retrieval"]
SUPPORTED_TASKS: tuple[str, ...] = (
    "local",
    "retrieval",
    "copy_first",
    "associative_recall",
    "multi_query_retrieval",
)


def make_local_dependency_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a task where the next token mostly depends on recent context."""

    tokens = torch.empty(batch_size, seq_len + 1, device=device, dtype=torch.long)
    tokens[:, 0] = torch.randint(4, vocab_size, (batch_size,), device=device, generator=generator)
    increments = torch.randint(1, 4, (batch_size, seq_len), device=device, generator=generator)
    for idx in range(1, seq_len + 1):
        tokens[:, idx] = ((tokens[:, idx - 1] + increments[:, idx - 1] - 4) % (vocab_size - 4)) + 4
    return tokens[:, :-1], tokens[:, 1:]


def make_retrieval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a long-range marker retrieval task."""

    tokens = torch.randint(4, vocab_size, (batch_size, seq_len + 1), device=device, generator=generator)
    key = torch.randint(4, vocab_size, (batch_size,), device=device, generator=generator)
    tokens[:, 0] = 1
    tokens[:, 1] = key
    if seq_len > 16:
        tokens[:, seq_len // 2] = 2
        tokens[:, seq_len // 2 + 1] = key
        tokens[:, -2] = 3
        tokens[:, -1] = key
    return tokens[:, :-1], tokens[:, 1:]


def make_copy_first_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a sparse-label task that copies an early value at the end."""

    if seq_len < 4:
        raise ValueError("copy_first requires seq_len >= 4")
    if vocab_size < 8:
        raise ValueError("copy_first requires vocab_size >= 8")

    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = torch.full((batch_size, seq_len), IGNORE_INDEX, device=device, dtype=torch.long)
    values = torch.randint(4, vocab_size, (batch_size,), device=device, generator=generator)

    input_ids[:, 0] = 1
    input_ids[:, 1] = values
    input_ids[:, -1] = 2
    labels[:, -1] = values
    return input_ids, labels


def make_associative_recall_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a key-value recall task with one far query per sequence."""

    if seq_len < 16:
        raise ValueError("associative_recall requires seq_len >= 16")
    if vocab_size < 32:
        raise ValueError("associative_recall requires vocab_size >= 32")

    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = torch.full((batch_size, seq_len), IGNORE_INDEX, device=device, dtype=torch.long)

    key_low = 4
    key_high = max(key_low + 8, vocab_size // 2)
    value_low = key_high
    value_high = vocab_size
    num_pairs = min(8, max(2, (seq_len // 2) // 3), key_high - key_low)

    for batch_idx in range(batch_size):
        keys = torch.randperm(key_high - key_low, device=device, generator=generator)[:num_pairs] + key_low
        values = torch.randint(value_low, value_high, (num_pairs,), device=device, generator=generator)
        for pair_idx in range(num_pairs):
            pos = pair_idx * 3
            input_ids[batch_idx, pos] = 1
            input_ids[batch_idx, pos + 1] = keys[pair_idx]
            input_ids[batch_idx, pos + 2] = values[pair_idx]
        target_idx = int(torch.randint(0, num_pairs, (1,), device=device, generator=generator).item())
        input_ids[batch_idx, -2] = 2
        input_ids[batch_idx, -1] = keys[target_idx]
        labels[batch_idx, -1] = values[target_idx]
    return input_ids, labels


def make_multi_query_retrieval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate several far key-value queries to reduce sparse-label variance."""

    if seq_len < 32:
        raise ValueError("multi_query_retrieval requires seq_len >= 32")
    if vocab_size < 32:
        raise ValueError("multi_query_retrieval requires vocab_size >= 32")

    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = torch.full((batch_size, seq_len), IGNORE_INDEX, device=device, dtype=torch.long)

    key_low = 4
    key_high = max(key_low + 8, vocab_size // 2)
    value_low = key_high
    value_high = vocab_size
    num_pairs = min(12, max(4, (seq_len // 3) // 3), key_high - key_low)
    num_queries = min(6, max(2, (seq_len - (seq_len * 2 // 3)) // 2), num_pairs)
    query_start = seq_len - num_queries * 2

    for batch_idx in range(batch_size):
        keys = torch.randperm(key_high - key_low, device=device, generator=generator)[:num_pairs] + key_low
        values = torch.randint(value_low, value_high, (num_pairs,), device=device, generator=generator)
        for pair_idx in range(num_pairs):
            pos = pair_idx * 3
            input_ids[batch_idx, pos] = 1
            input_ids[batch_idx, pos + 1] = keys[pair_idx]
            input_ids[batch_idx, pos + 2] = values[pair_idx]

        query_indices = torch.randperm(num_pairs, device=device, generator=generator)[:num_queries]
        for query_offset, pair_idx_tensor in enumerate(query_indices):
            pair_idx = int(pair_idx_tensor.item())
            pos = query_start + query_offset * 2
            input_ids[batch_idx, pos] = 2
            input_ids[batch_idx, pos + 1] = keys[pair_idx]
            labels[batch_idx, pos + 1] = values[pair_idx]
    return input_ids, labels


def make_batch(
    task: TaskName,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch synthetic task batch creation."""

    if task == "local":
        return make_local_dependency_batch(batch_size, seq_len, vocab_size, device, generator)
    if task == "retrieval":
        return make_retrieval_batch(batch_size, seq_len, vocab_size, device, generator)
    if task == "copy_first":
        return make_copy_first_batch(batch_size, seq_len, vocab_size, device, generator)
    if task == "associative_recall":
        return make_associative_recall_batch(batch_size, seq_len, vocab_size, device, generator)
    if task == "multi_query_retrieval":
        return make_multi_query_retrieval_batch(batch_size, seq_len, vocab_size, device, generator)
    raise ValueError(f"Unsupported task: {task}")
