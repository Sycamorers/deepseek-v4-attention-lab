# API Notes

## Importing Modules

```python
from dsv4_attention_lab import (
    DenseMHA,
    SlidingWindowAttention,
    SparseTopKAttention,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    HybridCSAHLayers,
)
```

## Common Tensor Shapes

Public attention modules accept hidden states shaped `[batch, seq_len, hidden_size]` and return `[batch, seq_len, hidden_size]` by default.

Internal head-space tensors use `[batch, heads, seq_len, head_dim]`.

Compression utilities accept `[batch, seq_len, dim]` and return `[batch, compressed_len, dim]`.

## Optional Diagnostics

Most public attention modules support:

```python
result = module(
    x,
    output_attentions=True,
    return_kv_cache_estimate=True,
)
```

When diagnostics are requested, the module returns `AttentionOutput` with:

- `output`
- `attention_weights`
- `kv_cache_bytes`
- `selected_indices` for sparse top-k attention when requested

## TinyTransformer

```python
from dsv4_attention_lab.configs import TinyTransformerConfig
from dsv4_attention_lab.models import TinyTransformer

config = TinyTransformerConfig(attention_type="hybrid")
model = TinyTransformer(config)
out = model(input_ids, labels=labels)
loss = out["loss"]
```

