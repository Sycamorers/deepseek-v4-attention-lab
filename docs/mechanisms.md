# Mechanisms

This document describes the mechanisms implemented in this repository. The code is a pure PyTorch reference implementation inspired by DeepSeek-V4, not an official implementation.

## DenseMHA

`DenseMHA` is standard causal multi-head attention. For a sequence of length `T`, each query attends to all keys up to its own position. It is useful as a correctness baseline because the masking behavior is simple and well understood.

## SlidingWindowAttention

`SlidingWindowAttention` restricts visibility to `window_size` previous tokens plus the current token. It keeps local dependencies cheap and is also used as the optional local branch in CSA and HCA.

## KV Compression

`average_pool_kv` groups tokens into fixed blocks and averages each block. `WeightedKVCompressor(method="learned")` uses a trainable scorer to produce softmax weights inside each block.

For causal safety, compressed entries include `block_ends`. A query at token position `t` can attend only to compressed blocks where `block_end <= t`, which prevents a compressed block from leaking future tokens.

## SparseTopKAttention

`SparseTopKAttention` computes lightweight index scores between queries and compressed keys, selects top-k visible compressed entries, gathers their K/V vectors, and performs attention only over that subset. If fewer than top-k entries are visible, only visible entries contribute. If no entries are visible, the output is zero unless attention sink is enabled.

## CSA

`CompressedSparseAttention` does:

1. Project hidden states to query heads.
2. Compress hidden states along the sequence dimension.
3. Project compressed entries to K/V.
4. Apply top-k sparse attention over compressed entries.
5. Optionally add a sliding-window branch over uncompressed local K/V.
6. Merge branches and apply the output projection.

## HCA

`HeavilyCompressedAttention` uses a larger compression ratio and dense attention over compressed K/V entries. It does not use top-k selection.

## Hybrid Layers

`build_hybrid_attention_layers` and `HybridCSAHLayers` construct simple interleaved stacks. Patterns include:

- `alternating`
- `csa_only`
- `hca_only`
- `swa_then_alternating`

