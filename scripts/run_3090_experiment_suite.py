#!/usr/bin/env python
"""Run a 3090-friendly attention experiment suite."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROFILES: dict[str, dict[str, object]] = {
    "smoke": {
        "benchmark_seq": "256,512",
        "benchmark_repeats": "1",
        "benchmark_iters": "2",
        "diagnostic_seq": "256",
        "diagnostic_ratios": "4,16",
        "diagnostic_topk": "4,8",
        "diagnostic_noise": "0.0,0.2",
        "train_tasks": ["copy_first"],
        "train_steps": "5",
        "train_seeds": "1234",
        "train_seq": "64",
        "train_batch": "2",
        "train_hidden": "32",
        "train_layers": "1",
    },
    "standard": {
        "benchmark_seq": "512,1024,2048,4096,8192",
        "benchmark_repeats": "5",
        "benchmark_iters": "5",
        "diagnostic_seq": "512,2048,8192",
        "diagnostic_ratios": "4,8,16,32,64,128",
        "diagnostic_topk": "4,8,16,32",
        "diagnostic_noise": "0.0,0.1,0.2",
        "train_tasks": ["local", "copy_first", "associative_recall", "multi_query_retrieval"],
        "train_steps": "250",
        "train_seeds": "2026,2027,2028",
        "train_seq": "512",
        "train_batch": "4",
        "train_hidden": "96",
        "train_layers": "2",
    },
    "extended": {
        "benchmark_seq": "512,1024,2048,4096,8192,12288",
        "benchmark_repeats": "7",
        "benchmark_iters": "7",
        "diagnostic_seq": "512,2048,8192,16384",
        "diagnostic_ratios": "4,8,16,32,64,128",
        "diagnostic_topk": "4,8,16,32,64",
        "diagnostic_noise": "0.0,0.05,0.1,0.2,0.4",
        "train_tasks": ["local", "copy_first", "associative_recall", "multi_query_retrieval"],
        "train_steps": "800",
        "train_seeds": "2026,2027,2028,2029,2030",
        "train_seq": "1024",
        "train_batch": "4",
        "train_hidden": "128",
        "train_layers": "3",
    },
}


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Print and optionally execute a command."""

    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build suite commands from the selected profile."""

    profile = PROFILES[args.profile]
    output_root = Path(args.output_root)
    commands: list[list[str]] = []

    if not args.skip_benchmark:
        commands.append(
            [
                sys.executable,
                "scripts/run_attention_benchmark.py",
                "--device",
                args.device,
                "--dtype",
                args.benchmark_dtype,
                "--seq-len",
                str(profile["benchmark_seq"]),
                "--batch-size",
                args.benchmark_batch_size,
                "--hidden-size",
                args.benchmark_hidden_size,
                "--num-heads",
                args.benchmark_num_heads,
                "--compression-ratio",
                args.compression_ratio,
                "--hca-compression-ratio",
                args.hca_compression_ratio,
                "--top-k",
                args.top_k,
                "--window-size",
                args.window_size,
                "--iters",
                str(profile["benchmark_iters"]),
                "--warmup-iters",
                args.warmup_iters,
                "--repeats",
                str(profile["benchmark_repeats"]),
                "--seed",
                args.seed,
                "--output-dir",
                str(output_root / args.profile / "benchmark_all"),
            ]
        )

    if not args.skip_diagnostics:
        commands.append(
            [
                sys.executable,
                "scripts/run_attention_diagnostics.py",
                "--device",
                args.device,
                "--dtype",
                args.diagnostic_dtype,
                "--seq-len",
                str(profile["diagnostic_seq"]),
                "--batch-size",
                args.diagnostic_batch_size,
                "--num-heads",
                args.diagnostic_num_heads,
                "--head-dim",
                args.diagnostic_head_dim,
                "--compression-ratios",
                str(profile["diagnostic_ratios"]),
                "--top-k",
                str(profile["diagnostic_topk"]),
                "--noise-std",
                str(profile["diagnostic_noise"]),
                "--seed",
                args.seed,
                "--output-dir",
                str(output_root / args.profile / "diagnostics"),
            ]
        )

    if not args.skip_training:
        for task in profile["train_tasks"]:
            commands.append(
                [
                    sys.executable,
                    "scripts/run_tiny_lm_experiment.py",
                    "--device",
                    args.device,
                    "--dtype",
                    args.train_dtype,
                    "--attention",
                    "all",
                    "--task",
                    str(task),
                    "--seq-len",
                    str(profile["train_seq"]),
                    "--batch-size",
                    str(profile["train_batch"]),
                    "--hidden-size",
                    str(profile["train_hidden"]),
                    "--num-heads",
                    args.train_num_heads,
                    "--num-layers",
                    str(profile["train_layers"]),
                    "--compression-ratio",
                    args.compression_ratio,
                    "--hca-compression-ratio",
                    args.hca_compression_ratio,
                    "--top-k",
                    args.top_k,
                    "--window-size",
                    args.train_window_size,
                    "--steps",
                    str(profile["train_steps"]),
                    "--val-batches",
                    args.val_batches,
                    "--seeds",
                    str(profile["train_seeds"]),
                    "--output-dir",
                    str(output_root / args.profile / f"tiny_lm_{task}"),
                ]
            )

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the redesigned attention experiment suite.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/attention_suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seed", default="1234")
    parser.add_argument("--compression-ratio", default="4")
    parser.add_argument("--hca-compression-ratio", default="16")
    parser.add_argument("--top-k", default="8")
    parser.add_argument("--window-size", default="128")
    parser.add_argument("--warmup-iters", default="2")
    parser.add_argument("--benchmark-dtype", default="float16")
    parser.add_argument("--benchmark-batch-size", default="1")
    parser.add_argument("--benchmark-hidden-size", default="128")
    parser.add_argument("--benchmark-num-heads", default="4")
    parser.add_argument("--diagnostic-dtype", default="float32")
    parser.add_argument("--diagnostic-batch-size", default="2")
    parser.add_argument("--diagnostic-num-heads", default="4")
    parser.add_argument("--diagnostic-head-dim", default="32")
    parser.add_argument("--train-dtype", default="float32")
    parser.add_argument("--train-num-heads", default="4")
    parser.add_argument("--train-window-size", default="64")
    parser.add_argument("--val-batches", default="16")
    args = parser.parse_args()

    for command in build_commands(args):
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
