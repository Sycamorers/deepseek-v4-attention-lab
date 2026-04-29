#!/usr/bin/env python
"""Plot benchmark and experiment outputs with matplotlib."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_benchmark(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_benchmark(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    attentions = sorted({row["attention"] for row in rows})
    for metric, ylabel, filename in [
        ("runtime_ms", "Runtime (ms)", "runtime_vs_sequence_length.png"),
        ("peak_memory_mb", "Peak CUDA memory (MB)", "memory_vs_sequence_length.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for attention in attentions:
            selected = [row for row in rows if row["attention"] == attention]
            selected.sort(key=lambda row: int(row["seq_len"]))
            plt.plot(
                [int(row["seq_len"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=attention,
            )
        plt.xlabel("Sequence length")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def plot_validation_losses(metrics_path: Path, output_dir: Path) -> None:
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = metrics.get("results", [])
    if not results:
        return
    plt.figure(figsize=(6, 4))
    names = [result["attention"] for result in results]
    losses = [float(result["validation_loss"]) for result in results]
    plt.bar(names, losses)
    plt.ylabel("Validation loss")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_loss_comparison.png", dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark and tiny LM results.")
    parser.add_argument("--benchmark-csv", default="outputs/attention_benchmark.csv")
    parser.add_argument("--tiny-lm-json", default="outputs/tiny_lm/metrics.json")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark(read_benchmark(Path(args.benchmark_csv)), output_dir)
    plot_validation_losses(Path(args.tiny_lm_json), output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()

