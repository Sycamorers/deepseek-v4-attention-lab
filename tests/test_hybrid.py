import torch

from dsv4_attention_lab.attention.hybrid import HybridCSAHLayers, build_hybrid_attention_layers


def test_build_hybrid_attention_layers_pattern() -> None:
    layers, layer_types = build_hybrid_attention_layers(
        num_layers=4,
        hidden_size=32,
        num_heads=4,
        pattern="swa_then_alternating",
        warmup_layers=1,
        compression_ratio=2,
        hca_compression_ratio=4,
        top_k=2,
        window_size=3,
    )
    assert len(layers) == 4
    assert layer_types == ["swa", "csa", "hca", "csa"]


def test_hybrid_forward_backward() -> None:
    module = HybridCSAHLayers(
        num_layers=2,
        hidden_size=24,
        num_heads=3,
        pattern="alternating",
        compression_ratio=2,
        hca_compression_ratio=4,
        top_k=2,
        window_size=3,
    )
    x = torch.randn(2, 6, 24, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert x.grad is not None

