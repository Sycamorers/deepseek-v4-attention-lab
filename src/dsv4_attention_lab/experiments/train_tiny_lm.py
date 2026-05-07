"""Train TinyTransformer models on synthetic next-token tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ..configs import TinyTransformerConfig
from ..experiments.synthetic_long_context import IGNORE_INDEX, SUPPORTED_TASKS, make_batch
from ..models.tiny_transformer import TinyTransformer
from ..utils import dataclass_to_json_dict, dtype_from_string, set_seed


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def supervised_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """Return accuracy over non-ignored labels."""

    supervised = labels != IGNORE_INDEX
    count = int(supervised.sum().item())
    if count == 0:
        return 0.0, 0
    predictions = logits.argmax(dim=-1)
    correct = int((predictions[supervised] == labels[supervised]).sum().item())
    return correct / count, count


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a device-local generator for comparable synthetic data streams."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def evaluate(
    model: TinyTransformer,
    *,
    task: str,
    batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float | int]:
    """Evaluate average validation loss and supervised-token accuracy."""

    model.eval()
    losses: list[float] = []
    correct = 0.0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = make_batch(task, batch_size, seq_len, vocab_size, device, generator=generator)
            output = model(input_ids, labels=labels)
            losses.append(float(output["loss"].item()))
            accuracy, count = supervised_accuracy(output["logits"], labels)
            correct += accuracy * count
            total += count
    model.train()
    return {
        "validation_loss": sum(losses) / len(losses),
        "validation_accuracy": correct / total if total else 0.0,
        "validation_tokens": total,
    }


def train_one_attention(
    args: argparse.Namespace,
    attention: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int,
) -> dict[str, object]:
    """Train one attention variant and return metrics."""

    set_seed(seed)
    config = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attention_type=attention,
        compression_ratio=args.compression_ratio,
        hca_compression_ratio=args.hca_compression_ratio,
        top_k=args.top_k,
        window_size=args.window_size,
        use_attention_sink=args.use_attention_sink,
    )
    model = TinyTransformer(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_generator = make_generator(device, seed + 1_000_003)
    val_generator = make_generator(device, seed + 2_000_003)
    losses: list[dict[str, float | int | str]] = []

    for step in range(1, args.steps + 1):
        input_ids, labels = make_batch(
            args.task,
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        loss = output["loss"]
        train_accuracy, train_tokens = supervised_accuracy(output["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(
            {
                "seed": seed,
                "step": step,
                "loss": float(loss.item()),
                "attention": attention,
                "train_accuracy": train_accuracy,
                "train_tokens": train_tokens,
            }
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"seed {seed:<6} {attention:<14} step {step:>4}/{args.steps} "
                f"loss {loss.item():.4f} acc {train_accuracy:.3f}"
            )

    eval_metrics = evaluate(
        model,
        task=args.task,
        batches=args.val_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
        generator=val_generator,
    )
    return {
        "seed": seed,
        "attention": attention,
        **eval_metrics,
        "loss_curve": losses,
        "config": dataclass_to_json_dict(config),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a tiny LM on synthetic tasks.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--hca-compression-ratio", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", default="", help="Optional comma-separated seeds. Overrides --seed when set.")
    parser.add_argument("--output-dir", default="outputs/tiny_lm")
    parser.add_argument("--attention", default="hybrid", choices=["all", "dense", "sliding_window", "csa", "hca", "hybrid"])
    parser.add_argument("--task", default="local", choices=SUPPORTED_TASKS)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--use-attention-sink", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    attentions = ["dense", "sliding_window", "csa", "hca", "hybrid"] if args.attention == "all" else [args.attention]
    seeds = parse_int_list(args.seeds) if args.seeds else [args.seed]
    results = [
        train_one_attention(args, attention, device, dtype, seed=seed)
        for seed in seeds
        for attention in attentions
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    curves_path = output_dir / "loss_curves.csv"
    config_path = output_dir / "config.json"

    metrics = {
        "task": args.task,
        "seq_len": args.seq_len,
        "seeds": seeds,
        "results": [{k: v for k, v in result.items() if k != "loss_curve"} for result in results],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "attention", "step", "loss", "train_accuracy", "train_tokens"])
        writer.writeheader()
        for result in results:
            writer.writerows(result["loss_curve"])

    print("Validation metrics:")
    for result in results:
        print(
            f"seed {result['seed']:<6} {result['attention']:<14} "
            f"loss {result['validation_loss']:.4f} acc {result['validation_accuracy']:.3f}"
        )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {curves_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
