"""Benchmark attention modules with small, reproducible defaults."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from ..attention.csa import CompressedSparseAttention
from ..attention.dense_mha import DenseMHA
from ..attention.hca import HeavilyCompressedAttention
from ..attention.hybrid import HybridCSAHLayers
from ..attention.sliding_window import SlidingWindowAttention
from ..utils import AttentionOutput, dtype_from_string, module_output_tensor, set_seed


def parse_seq_lens(value: str) -> list[int]:
    """Parse a comma-separated sequence length string."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_attention_module(name: str, args: argparse.Namespace) -> nn.Module:
    """Construct an attention module from benchmark CLI args."""

    common = dict(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.hidden_size // args.num_heads,
        dropout=0.0,
    )
    if name == "dense":
        return DenseMHA(**common, use_rope=True, causal=True)
    if name == "sliding_window":
        return SlidingWindowAttention(**common, window_size=args.window_size, use_rope=True)
    if name == "csa":
        return CompressedSparseAttention(
            **common,
            compression_ratio=args.compression_ratio,
            top_k=args.top_k,
            window_size=args.window_size,
            use_qk_rmsnorm=True,
            use_partial_rope=True,
            use_attention_sink=args.use_attention_sink,
        )
    if name == "hca":
        return HeavilyCompressedAttention(
            **common,
            compression_ratio=args.hca_compression_ratio,
            window_size=args.window_size,
            use_qk_rmsnorm=True,
            use_partial_rope=True,
            use_attention_sink=args.use_attention_sink,
        )
    if name == "hybrid":
        return HybridCSAHLayers(
            num_layers=2,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            head_dim=args.hidden_size // args.num_heads,
            pattern="alternating",
            compression_ratio=args.compression_ratio,
            hca_compression_ratio=args.hca_compression_ratio,
            top_k=args.top_k,
            window_size=args.window_size,
            dropout=0.0,
            use_attention_sink=args.use_attention_sink,
        )
    raise ValueError(f"Unsupported attention module: {name}")


def estimate_score_count(name: str, args: argparse.Namespace, seq_len: int) -> int:
    """Estimate the number of attention scores materialized conceptually."""

    batch = args.batch_size
    heads = args.num_heads
    if name == "dense":
        return batch * heads * seq_len * seq_len
    if name == "sliding_window":
        return batch * heads * seq_len * min(seq_len, args.window_size + 1)
    if name == "csa":
        local = batch * heads * seq_len * min(seq_len, args.window_size + 1)
        sparse = batch * heads * seq_len * min(args.top_k, math.ceil(seq_len / args.compression_ratio))
        return local + sparse
    if name == "hca":
        local = batch * heads * seq_len * min(seq_len, args.window_size + 1)
        compressed = batch * heads * seq_len * math.ceil(seq_len / args.hca_compression_ratio)
        return local + compressed
    if name == "hybrid":
        return estimate_score_count("csa", args, seq_len) + estimate_score_count("hca", args, seq_len)
    raise ValueError(f"Unsupported attention module: {name}")


def estimate_kv_cache_from_args(name: str, args: argparse.Namespace, seq_len: int, dtype: torch.dtype) -> int:
    """Fallback KV-cache estimate for modules without rich output."""

    element_size = torch.empty((), dtype=dtype).element_size()
    batch = args.batch_size
    heads = args.num_heads
    head_dim = args.hidden_size // args.num_heads
    dense_tokens = min(seq_len, args.window_size + 1)
    if name == "dense":
        return batch * seq_len * heads * head_dim * 2 * element_size
    if name == "sliding_window":
        return batch * dense_tokens * heads * head_dim * 2 * element_size
    if name == "csa":
        compressed = math.ceil(seq_len / args.compression_ratio)
        return batch * (compressed * head_dim * 2 + dense_tokens * heads * head_dim * 2) * element_size
    if name == "hca":
        compressed = math.ceil(seq_len / args.hca_compression_ratio)
        return batch * (compressed * head_dim * 2 + dense_tokens * heads * head_dim * 2) * element_size
    if name == "hybrid":
        return estimate_kv_cache_from_args("csa", args, seq_len, dtype) + estimate_kv_cache_from_args(
            "hca", args, seq_len, dtype
        )
    return 0


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA timers when needed."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one(module: nn.Module, x: torch.Tensor, device: torch.device, iters: int) -> tuple[float, int, torch.Tensor | AttentionOutput]:
    """Run one benchmark and return average milliseconds and peak bytes."""

    module.eval()
    with torch.no_grad():
        for _ in range(1):
            _ = module(x)
        synchronize_if_needed(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result: torch.Tensor | AttentionOutput = x
        for _ in range(iters):
            try:
                result = module(x, return_kv_cache_estimate=True)
            except TypeError:
                result = module(x)
        synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return elapsed_ms, peak_memory, result


def print_summary(rows: Iterable[dict[str, object]]) -> None:
    """Print a compact text summary."""

    header = f"{'attention':<16} {'seq':>6} {'ms':>10} {'peak_mb':>10} {'kv_mb':>10} {'scores':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['attention']:<16} {row['seq_len']:>6} {row['runtime_ms']:>10.2f} "
            f"{row['peak_memory_mb']:>10.2f} {row['kv_cache_mb']:>10.2f} {row['score_count']:>14}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark reference attention modules.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seq-len", default="512", help="Single length or comma-separated lengths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--hca-compression-ratio", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--attention", default="all", choices=["all", "dense", "sliding_window", "csa", "hca", "hybrid"])
    parser.add_argument("--use-attention-sink", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    names = ["dense", "sliding_window", "csa", "hca", "hybrid"] if args.attention == "all" else [args.attention]
    rows: list[dict[str, object]] = []
    for seq_len in parse_seq_lens(args.seq_len):
        x = torch.randn(args.batch_size, seq_len, args.hidden_size, device=device, dtype=dtype)
        for name in names:
            module = make_attention_module(name, args).to(device=device, dtype=dtype)
            runtime_ms, peak_memory, result = run_one(module, x, device, args.iters)
            output = module_output_tensor(result)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"{name} produced non-finite values")
            kv_cache_bytes = result.kv_cache_bytes if isinstance(result, AttentionOutput) and result.kv_cache_bytes else None
            if kv_cache_bytes is None:
                kv_cache_bytes = estimate_kv_cache_from_args(name, args, seq_len, dtype)
            rows.append(
                {
                    "attention": name,
                    "seq_len": seq_len,
                    "runtime_ms": runtime_ms,
                    "peak_memory_mb": peak_memory / (1024**2),
                    "kv_cache_mb": kv_cache_bytes / (1024**2),
                    "score_count": estimate_score_count(name, args, seq_len),
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attention_benchmark.csv"
    json_path = output_dir / "attention_benchmark.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print_summary(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

