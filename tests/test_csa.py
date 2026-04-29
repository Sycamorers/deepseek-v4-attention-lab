import torch

from dsv4_attention_lab.attention.csa import CompressedSparseAttention
from dsv4_attention_lab.utils import AttentionOutput


def test_csa_forward_cpu_shape_no_nans() -> None:
    module = CompressedSparseAttention(
        hidden_size=32,
        num_heads=4,
        compression_ratio=3,
        top_k=4,
        window_size=4,
        dropout=0.0,
    )
    x = torch.randn(2, 9, 32)
    result = module(x, return_kv_cache_estimate=True)
    assert isinstance(result, AttentionOutput)
    assert result.output.shape == x.shape
    assert result.kv_cache_bytes is not None
    assert torch.isfinite(result.output).all()


def test_csa_backward_cpu() -> None:
    module = CompressedSparseAttention(hidden_size=24, num_heads=3, compression_ratio=2, top_k=2, window_size=3)
    x = torch.randn(2, 6, 24, requires_grad=True)
    loss = module(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_csa_cuda_optional() -> None:
    if not torch.cuda.is_available():
        return
    module = CompressedSparseAttention(hidden_size=32, num_heads=4, compression_ratio=4, top_k=2).cuda()
    x = torch.randn(1, 8, 32, device="cuda")
    y = module(x)
    assert y.is_cuda
    assert torch.isfinite(y).all()

