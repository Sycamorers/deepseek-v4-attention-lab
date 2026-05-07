# Redesigned Attention Experiments

This repository is a pure PyTorch reference lab. The experiments below are designed to test the implemented mechanisms in a less biased way on a single RTX 3090. They are not official DeepSeek quality or kernel-performance reproductions.

## Current Source Alignment

Recent public DeepSeek attention changes give three useful reference points:

- DeepSeek-V3 uses Multi-head Latent Attention (MLA) for efficient inference and cost-aware training.
- DeepSeek-V3.2 introduces DeepSeek Sparse Attention (DSA) to reduce long-context attention complexity while preserving long-context performance.
- DeepSeek-V4 replaces MLA with a hybrid local + long-range design: sliding-window attention, Compressed Sparse Attention (CSA), and Heavily Compressed Attention (HCA). The public Transformers documentation describes CSA as a low-compression pool plus a top-k Lightning Indexer, and HCA as a higher-compression pool with every pooled entry contributing to attention.

This codebase currently focuses on V4-style CSA/HCA, a DSA-like top-k sparse selector, sliding-window attention, and dense MHA baselines.

## Problems With The Old Experiment Set

- `--attention all` used one global seed, so each attention variant received a different initialization and a different training data stream.
- The old `retrieval` task supervised every next-token position. Most positions were random background tokens, so the long-range retrieval signal was diluted in the average loss.
- Runtime was reported from a small number of measurements without median/IQR style repeat statistics.
- One small model shape and one training budget made it too easy to over-interpret noisy toy-LM results.
- The benchmark mixed three different questions: theoretical score count, PyTorch reference runtime, and model quality.
- There was no direct diagnostic for the two core risks in sparse/compressed attention: top-k miss rate and compression signal dilution.

## New Experiment Layers

### 1. Controlled Mechanism Diagnostics

Run:

```bash
python scripts/run_attention_diagnostics.py \
  --device cuda \
  --dtype float32 \
  --seq-len 512,2048,8192 \
  --compression-ratios 4,8,16,32,64,128 \
  --top-k 4,8,16,32 \
  --noise-std 0.0,0.1,0.2 \
  --distractor-strength 0.0,0.75 \
  --output-dir outputs/attention_diagnostics
```

Metrics:

- `topk_recall`: whether DSA-like top-k selection includes the intended visible compressed block.
- `distractor_strength`: how strongly the query is mixed with another visible compressed block, making top-k selection harder.
- `mean_relative_signal`: how much a single salient token survives fixed-block average compression. A value near `1 / compression_ratio` is expected when one token is averaged with unrelated tokens.
- `mean_cosine`: directional retention of that salient signal after compression.

Interpretation:

- If `topk_recall` collapses when noise grows or `top_k` is small, CSA quality can fail even if theoretical score count is low.
- If `mean_relative_signal` is tiny at high compression ratios, HCA should be treated as broad global summarization, not precise token retrieval.

### 2. Robust Microbenchmarks

Run:

```bash
python scripts/run_attention_benchmark.py \
  --device cuda \
  --dtype float16 \
  --seq-len 512,1024,2048,4096,8192 \
  --batch-size 1 \
  --hidden-size 128 \
  --num-heads 4 \
  --compression-ratio 4 \
  --hca-compression-ratio 16 \
  --top-k 8 \
  --window-size 128 \
  --iters 5 \
  --warmup-iters 2 \
  --repeats 5 \
  --output-dir outputs/benchmark_all
```

Report `runtime_ms` as the median, keep `runtime_ms_p25`, `runtime_ms_p75`, min/max, max peak memory, KV estimate, and conceptual score count. Treat PyTorch runtime as reference-code behavior only; optimized kernels can change the ranking.

### 3. Less Diluted Tiny-LM Tasks

The training harness now resets the seed for every attention variant, so same-seed runs are initialized comparably and see the same generated data sequence.

New sparse-label tasks:

- `copy_first`: copy an early value at the final query position.
- `associative_recall`: read early key-value pairs and answer one far query.
- `multi_query_retrieval`: answer several far key-value queries per sequence to reduce sparse-label variance.

Run:

```bash
python scripts/run_tiny_lm_experiment.py \
  --device cuda \
  --dtype float32 \
  --attention all \
  --task associative_recall \
  --seq-len 512 \
  --batch-size 4 \
  --hidden-size 96 \
  --num-heads 4 \
  --num-layers 2 \
  --compression-ratio 4 \
  --hca-compression-ratio 16 \
  --top-k 8 \
  --window-size 64 \
  --steps 250 \
  --val-batches 16 \
  --seeds 2026,2027,2028 \
  --output-dir outputs/tiny_lm/associative_recall
```

Use validation accuracy alongside validation loss. Accuracy is computed only over non-ignored supervised positions.

### 4. 3090 Suite Entry Point

Smoke test:

```bash
python scripts/run_3090_experiment_suite.py --profile smoke --device cuda
```

Standard run:

```bash
python scripts/run_3090_experiment_suite.py --profile standard --device cuda
```

Longer run:

```bash
python scripts/run_3090_experiment_suite.py --profile extended --device cuda
```

Use `--dry-run` to print the exact commands without running them. Use `--skip-benchmark`, `--skip-diagnostics`, or `--skip-training` when iterating.

## Minimum Reporting Standard

For any claim about an attention variant, include:

- GPU, dtype, PyTorch version, and command line.
- Sequence lengths and model sizes.
- Median runtime with repeat count.
- KV-cache estimate and score count.
- At least three training seeds for toy-LM quality claims.
- Validation loss and supervised-token accuracy.
- Diagnostics showing top-k recall and compression signal retention for the same compression/top-k regime.

## Expected Readout

Dense MHA should remain the clean correctness baseline and can be faster in this reference implementation because dense matrix multiplication is highly optimized.

Sliding-window attention should do well on local tasks and fail on far retrieval unless the answer remains inside the window.

CSA should be judged by whether top-k sparse selection can keep the relevant compressed blocks while reducing score count. Its quality is sensitive to `top_k`, compression ratio, and indexer quality.

HCA should be judged as global summarization. It can be efficient and useful for broad context, but high compression is expected to lose fine-grained token identity.

Hybrid CSA/HCA should only be considered better if it improves retrieval or robustness across seeds enough to justify its extra runtime and cache footprint in this reference implementation.
