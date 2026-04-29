"""Train TinyTransformer models on synthetic next-token tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ..configs import TinyTransformerConfig
from ..experiments.synthetic_long_context import make_batch
from ..models.tiny_transformer import TinyTransformer
from ..utils import dataclass_to_json_dict, dtype_from_string, set_seed


def evaluate(
    model: TinyTransformer,
    *,
    task: str,
    batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    """Evaluate average validation loss."""

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = make_batch(task, batch_size, seq_len, vocab_size, device)
            losses.append(float(model(input_ids, labels=labels)["loss"].item()))
    model.train()
    return sum(losses) / len(losses)


def train_one_attention(args: argparse.Namespace, attention: str, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    """Train one attention variant and return metrics."""

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
    losses: list[dict[str, float | int | str]] = []

    for step in range(1, args.steps + 1):
        input_ids, labels = make_batch(args.task, args.batch_size, args.seq_len, args.vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=labels)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append({"step": step, "loss": float(loss.item()), "attention": attention})
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"{attention:<14} step {step:>4}/{args.steps} loss {loss.item():.4f}")

    val_loss = evaluate(
        model,
        task=args.task,
        batches=args.val_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )
    return {
        "attention": attention,
        "validation_loss": val_loss,
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
    parser.add_argument("--output-dir", default="outputs/tiny_lm")
    parser.add_argument("--attention", default="hybrid", choices=["all", "dense", "sliding_window", "csa", "hca", "hybrid"])
    parser.add_argument("--task", default="local", choices=["local", "retrieval"])
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

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    attentions = ["dense", "sliding_window", "csa", "hca", "hybrid"] if args.attention == "all" else [args.attention]
    results = [train_one_attention(args, attention, device, dtype) for attention in attentions]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    curves_path = output_dir / "loss_curves.csv"
    config_path = output_dir / "config.json"

    metrics = {
        "task": args.task,
        "seq_len": args.seq_len,
        "results": [{k: v for k, v in result.items() if k != "loss_curve"} for result in results],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attention", "step", "loss"])
        writer.writeheader()
        for result in results:
            writer.writerows(result["loss_curve"])

    print("Validation losses:")
    for result in results:
        print(f"{result['attention']:<14} {result['validation_loss']:.4f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {curves_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()

