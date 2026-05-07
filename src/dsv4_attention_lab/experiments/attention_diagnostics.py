"""Controlled diagnostics for sparse selection and KV compression."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from ..attention.kv_compression import average_pool_kv
from ..attention.sparse_topk import SparseTopKAttention
from ..utils import dtype_from_string, set_seed


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_topk_recall(
    *,
    seq_len: int,
    compression_ratio: int,
    top_k: int,
    noise_std: float,
    distractor_strength: float,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Measure whether top-k sparse selection includes the intended block."""

    compressed_len = math.ceil(seq_len / compression_ratio)
    block_ends = torch.minimum(
        torch.arange(compressed_len, device=device) * compression_ratio + compression_ratio - 1,
        torch.tensor(seq_len - 1, device=device),
    )
    compressed_k = torch.randn(batch_size, num_heads, compressed_len, head_dim, device=device, dtype=dtype)
    compressed_k = F.normalize(compressed_k, dim=-1)
    compressed_v = torch.randn_like(compressed_k)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    target_indices = torch.full((batch_size, num_heads, seq_len), -1, device=device, dtype=torch.long)
    for position in range(seq_len):
        visible_count = int((block_ends <= position).sum().item())
        if visible_count == 0:
            continue
        target = torch.randint(0, visible_count, (batch_size, num_heads), device=device)
        target_indices[:, :, position] = target
        gather_index = target.view(batch_size, num_heads, 1, 1).expand(batch_size, num_heads, 1, head_dim)
        target_vectors = torch.gather(compressed_k, dim=2, index=gather_index).squeeze(2)
        query_vectors = target_vectors
        if distractor_strength > 0 and visible_count > 1:
            offset = torch.randint(1, visible_count, (batch_size, num_heads), device=device)
            distractor = (target + offset) % visible_count
            distractor_index = distractor.view(batch_size, num_heads, 1, 1).expand(
                batch_size,
                num_heads,
                1,
                head_dim,
            )
            distractor_vectors = torch.gather(compressed_k, dim=2, index=distractor_index).squeeze(2)
            query_vectors = query_vectors + distractor_strength * distractor_vectors
        q[:, :, position, :] = query_vectors + noise_std * torch.randn_like(target_vectors)
    q = F.normalize(q, dim=-1)

    attention = SparseTopKAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        top_k=top_k,
        causal=True,
    ).to(device=device, dtype=dtype)
    result = attention(
        q,
        compressed_k,
        compressed_v,
        block_ends=block_ends,
        query_positions=torch.arange(seq_len, device=device),
        return_selected_indices=True,
    )
    if result.selected_indices is None:
        raise RuntimeError("SparseTopKAttention did not return selected indices")

    valid = target_indices >= 0
    hits = (result.selected_indices == target_indices.unsqueeze(-1)).any(dim=-1)
    evaluated = int(valid.sum().item())
    recall = float(hits[valid].float().mean().item()) if evaluated else 0.0
    return {
        "diagnostic": "topk_recall",
        "seq_len": seq_len,
        "compression_ratio": compression_ratio,
        "top_k": top_k,
        "noise_std": noise_std,
        "distractor_strength": distractor_strength,
        "recall": recall,
        "evaluated_queries": evaluated,
    }


def run_compression_signal(
    *,
    seq_len: int,
    compression_ratio: int,
    signal_scale: float,
    noise_std: float,
    batch_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Measure signal dilution from fixed-block average KV compression."""

    if signal_scale <= 0:
        raise ValueError("signal_scale must be positive")

    compressed_len = math.ceil(seq_len / compression_ratio)
    x = noise_std * torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    signals = F.normalize(torch.randn(batch_size, compressed_len, head_dim, device=device, dtype=dtype), dim=-1)

    for block_idx in range(compressed_len):
        start = block_idx * compression_ratio
        end = min(start + compression_ratio, seq_len)
        offsets = torch.randint(0, end - start, (batch_size,), device=device)
        for batch_idx in range(batch_size):
            x[batch_idx, start + int(offsets[batch_idx].item()), :] += signal_scale * signals[batch_idx, block_idx, :]

    pooled = average_pool_kv(x, compression_ratio, return_info=True).entries.float()
    signals_f = signals.float()
    mean_cosine = float(F.cosine_similarity(pooled, signals_f, dim=-1).mean().item())
    mean_relative_signal = float(((pooled * signals_f).sum(dim=-1) / signal_scale).mean().item())
    return {
        "diagnostic": "compression_signal",
        "seq_len": seq_len,
        "compression_ratio": compression_ratio,
        "noise_std": noise_std,
        "signal_scale": signal_scale,
        "mean_cosine": mean_cosine,
        "mean_relative_signal": mean_relative_signal,
        "evaluated_blocks": batch_size * compressed_len,
    }


def write_rows(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write diagnostics as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attention_diagnostics.csv"
    json_path = output_dir / "attention_diagnostics.json"
    fieldnames = [
        "diagnostic",
        "seq_len",
        "compression_ratio",
        "top_k",
        "noise_std",
        "distractor_strength",
        "signal_scale",
        "recall",
        "mean_cosine",
        "mean_relative_signal",
        "evaluated_queries",
        "evaluated_blocks",
        "batch_size",
        "num_heads",
        "head_dim",
        "device",
        "dtype",
        "seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run controlled attention diagnostics.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seq-len", default="512,2048")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--compression-ratios", default="4,8,16,32")
    parser.add_argument("--top-k", default="4,8,16")
    parser.add_argument("--noise-std", default="0.0,0.1,0.2")
    parser.add_argument("--distractor-strength", default="0.0,0.75")
    parser.add_argument("--signal-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/attention_diagnostics")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    rows: list[dict[str, object]] = []
    for seq_len in parse_int_list(args.seq_len):
        for compression_ratio in parse_int_list(args.compression_ratios):
            for noise_std in parse_float_list(args.noise_std):
                rows.append(
                    run_compression_signal(
                        seq_len=seq_len,
                        compression_ratio=compression_ratio,
                        signal_scale=args.signal_scale,
                        noise_std=noise_std,
                        batch_size=args.batch_size,
                        head_dim=args.head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                for top_k in parse_int_list(args.top_k):
                    for distractor_strength in parse_float_list(args.distractor_strength):
                        rows.append(
                            run_topk_recall(
                                seq_len=seq_len,
                                compression_ratio=compression_ratio,
                                top_k=top_k,
                                noise_std=noise_std,
                                distractor_strength=distractor_strength,
                                batch_size=args.batch_size,
                                num_heads=args.num_heads,
                                head_dim=args.head_dim,
                                device=device,
                                dtype=dtype,
                            )
                        )

    for row in rows:
        row.update(
            {
                "batch_size": args.batch_size,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "seed": args.seed,
            }
        )
    write_rows(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
