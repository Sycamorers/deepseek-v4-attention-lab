# DeepSeek-V4 Attention Lab

Educational PyTorch reference implementations for long-context attention mechanisms inspired by DeepSeek-V4.

## Overview

`deepseek-v4-attention-lab` provides educational and research-oriented PyTorch reference implementations of attention mechanisms inspired by DeepSeek-V4-style long-context model design. The repository focuses on clean, inspectable modules that make compressed and sparse attention easier to study, modify, test, and compare.

Implemented components include:

- Dense multi-head attention baseline
- Sliding-window attention
- KV compression utilities
- DSA-like top-k sparse attention
- Compressed Sparse Attention (CSA)
- Heavily Compressed Attention (HCA)
- Hybrid CSA/HCA attention blocks
- Tiny Transformer experiments
- Runtime, memory, KV-cache, and attention-score benchmarks

This repository is intended for educational and research purposes. It is not the official DeepSeek implementation, and it is not provided as a commercial product or service.

The implementations prioritize readability, modularity, and testability over peak kernel-level performance.

## Disclaimer

This project is not affiliated with, endorsed by, or maintained by DeepSeek.

This project is not an official implementation. It is an educational/reference implementation inspired by publicly available descriptions of DeepSeek-V4-style attention mechanisms.

The implementation does not reproduce official CUDA, Triton, TileLang, C++, or vendor-optimized kernels. All initial attention mechanisms are implemented in pure Python and PyTorch for clarity.

Results from this repository should not be interpreted as official DeepSeek model results, model-quality claims, or faithful measurements of production DeepSeek inference systems.

This repository is intended for learning, research prototyping, and comparative experimentation. This project is maintained as an educational and research-oriented resource and is not offered as a commercial product or service.

## Why This Repository Exists

Modern long-context language models need attention mechanisms that reduce the cost of storing KV cache entries and computing attention scores. Dense attention is simple and expressive, but its quadratic sequence cost becomes expensive as context lengths grow.

Compressed and sparse attention mechanisms offer practical ways to study this tradeoff:

- Compress historical tokens into fewer KV entries.
- Restrict each query to a selected subset of long-range context.
- Preserve local detail with a sliding-window branch.
- Compare dense, local, compressed, sparse, and hybrid attention under the same toy model and benchmark harness.

This repository gives researchers, students, and ML engineers a small PyTorch codebase where those ideas can be inspected directly instead of hidden inside production kernels.

## Implemented Mechanisms

### Dense Attention

Dense attention is the standard causal multi-head attention baseline. Each query attends to all previous tokens and itself.

```text
query t -> keys [0, 1, 2, ..., t]
```

Dense attention is useful as a correctness baseline, but its attention-score cost grows as `O(T^2)` for sequence length `T`.

### Sliding-Window Attention

Sliding-window attention restricts each query to a fixed local context window. Query `t` can attend only to the previous `window_size` tokens plus itself.

```text
query t -> keys [max(0, t - window_size), ..., t]
```

This reduces attention-score cost for long sequences and is useful for local dependencies. The tradeoff is that information outside the window is unavailable unless another global or compressed mechanism is added.

### KV Compression

KV compression groups tokens into fixed-size sequence blocks and produces one compressed KV entry per block.

```text
tokens:      x0 x1 x2 x3 | x4 x5 x6 x7 | x8 ...
compressed:     c0      |     c1      | c2 ...
```

The repository includes average-pooling compression, learned weighted block compression, and metadata for causal-safe visibility of compressed blocks. Causal-safe compression matters because a compressed block should not be visible to a query until all tokens represented by that block are available.

### DSA-Like Top-k Sparse Attention

The DSA-like sparse attention module performs top-k selection over compressed KV entries. Each query scores the compressed entries, selects the most relevant visible entries, and computes attention only over those selected entries.

```text
query -> score compressed KV entries -> select top-k -> attend to selected entries
```

This is a reference implementation of the sparse-selection idea. It is designed to be readable and testable, not a production sparse-attention kernel.

### Compressed Sparse Attention (CSA)

Compressed Sparse Attention combines KV compression with top-k sparse attention:

1. Compress KV entries along the sequence dimension.
2. Score compressed entries for each query.
3. Select top-k visible compressed entries.
4. Compute attention over the selected compressed entries.
5. Optionally combine this branch with sliding-window local attention.

CSA reduces long-context cost while preserving access to selected long-range information.

```text
hidden states
  -> compressed KV blocks
  -> top-k sparse attention over compressed blocks
  + optional sliding-window local branch
  -> merged attention output
```

### Heavily Compressed Attention (HCA)

Heavily Compressed Attention uses a larger compression ratio than CSA and then applies dense attention over the shorter compressed sequence. It does not use top-k sparse selection.

```text
hidden states
  -> heavily compressed KV blocks
  -> dense attention over compressed blocks
  + optional sliding-window local branch
  -> merged attention output
```

HCA is useful for studying the effect of broad but low-resolution long-context access.

### Hybrid CSA/HCA Attention

Hybrid attention blocks combine or interleave CSA and HCA layers. A typical pattern uses CSA for sparse long-range retrieval and HCA for cheaper broad compressed context.

Supported patterns include:

- `alternating`
- `csa_only`
- `hca_only`
- `swa_then_alternating`

The hybrid stack is intended for toy Transformer experiments and comparative studies rather than direct production deployment.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Sycamorers/deepseek-v4-attention-lab.git
cd deepseek-v4-attention-lab
pip install -e .
```

For development, tests, and plotting utilities:

```bash
pip install -e ".[dev]"
```

The project is designed to use Python and PyTorch only. CUDA is optional; CPU execution is supported for tests and small experiments.

## Quickstart

### Import CSA and HCA

```python
import torch

from dsv4_attention_lab import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
)

x = torch.randn(2, 256, 128)

csa = CompressedSparseAttention(
    hidden_size=128,
    num_heads=4,
    compression_ratio=4,
    top_k=8,
    window_size=64,
)

y = csa(x)
print(y.shape)

hca = HeavilyCompressedAttention(
    hidden_size=128,
    num_heads=4,
    compression_ratio=16,
    window_size=64,
)

z = hca(x)
print(z.shape)
```

### Use the Tiny Transformer

```python
import torch

from dsv4_attention_lab.configs import TinyTransformerConfig
from dsv4_attention_lab.models import TinyTransformer

config = TinyTransformerConfig(
    vocab_size=128,
    max_seq_len=256,
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    attention_type="hybrid",
    compression_ratio=4,
    hca_compression_ratio=16,
    top_k=8,
    window_size=64,
)

model = TinyTransformer(config)
input_ids = torch.randint(0, config.vocab_size, (2, 256))
labels = torch.randint(0, config.vocab_size, (2, 256))

out = model(input_ids, labels=labels)
print(out["loss"])
print(out["logits"].shape)
```

## Benchmarks

Run the attention benchmark on CPU:

```bash
python scripts/run_attention_benchmark.py --device cpu --seq-len 512
```

Run multiple sequence lengths:

```bash
python scripts/run_attention_benchmark.py \
  --device cpu \
  --seq-len 512,1024,2048 \
  --batch-size 1 \
  --hidden-size 128 \
  --num-heads 4
```

Run on CUDA if available:

```bash
python scripts/run_attention_benchmark.py \
  --device cuda \
  --seq-len 512,1024,2048,4096 \
  --batch-size 1
```

The benchmark compares dense attention, sliding-window attention, CSA, HCA, and hybrid CSA/HCA. Outputs are written to:

```text
outputs/attention_benchmark.csv
outputs/attention_benchmark.json
```

Measured or estimated quantities include forward runtime, peak CUDA memory when CUDA is available, estimated KV-cache size, and estimated attention-score count.

## Toy Experiments

Train a tiny decoder-only Transformer on a local-dependency synthetic task:

```bash
python scripts/run_tiny_lm_experiment.py \
  --device cpu \
  --attention hybrid \
  --seq-len 256 \
  --steps 20
```

Compare all attention types on the retrieval-style task:

```bash
python scripts/run_tiny_lm_experiment.py \
  --device cpu \
  --attention all \
  --task retrieval \
  --seq-len 256 \
  --steps 50
```

Use CUDA for larger experiments if available:

```bash
python scripts/run_tiny_lm_experiment.py \
  --device cuda \
  --attention all \
  --task retrieval \
  --seq-len 512 \
  --steps 100
```

Experiment outputs are written to:

```text
outputs/tiny_lm/metrics.json
outputs/tiny_lm/loss_curves.csv
outputs/tiny_lm/config.json
```

Generate plots:

```bash
python scripts/plot_results.py
```

Figures are written to:

```text
outputs/figures/
```

## Expected Results and Interpretation

Dense attention should provide the most direct baseline but has quadratic attention-score cost and full KV-cache growth.

Sliding-window attention should be efficient and competitive on local-dependency tasks, but it can struggle when the target depends on tokens outside the window.

CSA should reduce long-context attention cost by combining compression with sparse top-k selection over compressed entries. In this pure PyTorch reference implementation, CSA may not be faster than dense attention for small sequence lengths because the code is optimized for clarity rather than kernel efficiency.

HCA should reduce cost by applying dense attention over a much shorter compressed sequence. It may lose fine-grained token information when compression is aggressive.

Hybrid CSA/HCA attention is intended to study whether sparse long-range retrieval and broad compressed context can complement each other in small controlled settings.

## Final GPU Results

A regenerated RTX 3090 result snapshot with benchmark plots, toy-LM plots, and conclusions is available in [docs/results/final_results.md](docs/results/final_results.md).

## Project Structure

```text
deepseek-v4-attention-lab/
  README.md
  LICENSE
  pyproject.toml
  src/
    dsv4_attention_lab/
      __init__.py
      configs.py
      utils.py
      rope.py
      norms.py
      masks.py
      attention/
        dense_mha.py
        sliding_window.py
        sparse_topk.py
        kv_compression.py
        csa.py
        hca.py
        hybrid.py
      models/
        tiny_transformer.py
      benchmarks/
        benchmark_attention.py
        benchmark_kv_cache.py
      experiments/
        train_tiny_lm.py
        compare_attention.py
        synthetic_long_context.py
  tests/
  scripts/
  docs/
```

## Testing

Run the test suite:

```bash
pytest
```

The tests are intended to cover output shapes, NaN/Inf checks, causal masking behavior, sliding-window mask correctness, compression output lengths, non-divisible sequence lengths, small visible-context top-k behavior, CPU forward and backward passes, and optional CUDA smoke tests when CUDA is available.

## What This Repository Does Not Claim

This repository does not claim to:

- Reproduce official DeepSeek model quality.
- Reproduce official DeepSeek kernels.
- Provide production-ready inference kernels.
- Match vendor-optimized CUDA/Triton/TileLang performance.
- Provide a commercial hosted service or commercial product.

It is a reference codebase for learning and experimentation.

## Contributing

Contributions are welcome when they improve clarity, correctness, tests, or experiment reproducibility.

Good contribution areas include:

- Bug fixes with small regression tests
- Additional unit tests for edge cases
- Clearer documentation and diagrams
- New synthetic tasks for attention comparisons
- More benchmark metrics
- Refactors that improve readability without obscuring the algorithms

Please keep contributions aligned with the project goals:

- Use Python and PyTorch for the reference implementation.
- Keep comments and documentation in English.
- Prefer simple, explicit code over clever abstractions.
- Do not add CUDA, C++, Triton, or TileLang kernels to the initial reference path.
- Avoid wording or code comments that imply this is an official DeepSeek implementation.

## Citation and References

If you use this repository in research or teaching material, please cite the repository and the relevant DeepSeek technical reports or public materials that motivated your experiments.

References:

- DeepSeek-V4 technical report: [DeepSeek_V4.pdf on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- DeepSeek-V3 technical report: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-V3.2 / DSA reference: [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

Suggested repository citation:

```bibtex
@software{deepseek_v4_attention_lab,
  title = {DeepSeek-V4 Attention Lab},
  author = {deepseek-v4-attention-lab contributors},
  year = {2026},
  url = {https://github.com/Sycamorers/deepseek-v4-attention-lab}
}
```

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
