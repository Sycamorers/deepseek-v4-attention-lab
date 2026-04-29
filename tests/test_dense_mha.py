import torch

from dsv4_attention_lab.attention.dense_mha import DenseMHA
from dsv4_attention_lab.utils import AttentionOutput


def test_dense_mha_shape_and_no_nans() -> None:
    module = DenseMHA(hidden_size=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 32)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dense_mha_causal_weights() -> None:
    module = DenseMHA(hidden_size=16, num_heads=2, dropout=0.0, use_rope=False, causal=True)
    x = torch.randn(1, 6, 16)
    result = module(x, output_attentions=True)
    assert isinstance(result, AttentionOutput)
    weights = result.attention_weights
    assert weights is not None
    upper = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert torch.allclose(weights[0, 0][upper], torch.zeros_like(weights[0, 0][upper]))


def test_dense_mha_backward_has_gradients() -> None:
    module = DenseMHA(hidden_size=16, num_heads=2, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    loss = module(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert any(parameter.grad is not None for parameter in module.parameters())

