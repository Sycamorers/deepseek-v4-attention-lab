import torch

from dsv4_attention_lab.attention.kv_compression import CompressionOutput, WeightedKVCompressor, average_pool_kv


def test_average_pool_compression_output_length() -> None:
    x = torch.arange(2 * 7 * 4, dtype=torch.float32).view(2, 7, 4)
    result = average_pool_kv(x, compression_ratio=3, return_info=True)
    assert isinstance(result, CompressionOutput)
    assert result.entries.shape == (2, 3, 4)
    assert result.block_ends.tolist() == [2, 5, 6]


def test_average_pool_handles_non_divisible_tail() -> None:
    x = torch.ones(1, 5, 3)
    result = average_pool_kv(x, compression_ratio=2, return_info=True)
    assert isinstance(result, CompressionOutput)
    assert result.entries.shape[1] == 3
    assert result.block_lengths.tolist() == [2, 2, 1]
    assert torch.allclose(result.entries, torch.ones_like(result.entries))


def test_learned_weighted_compression_backward() -> None:
    compressor = WeightedKVCompressor(dim=6, compression_ratio=4, method="learned")
    x = torch.randn(2, 9, 6, requires_grad=True)
    result = compressor(x, return_info=True)
    assert isinstance(result, CompressionOutput)
    assert result.entries.shape == (2, 3, 6)
    result.entries.sum().backward()
    assert x.grad is not None
    assert compressor.score_proj is not None
    assert compressor.score_proj.weight.grad is not None

