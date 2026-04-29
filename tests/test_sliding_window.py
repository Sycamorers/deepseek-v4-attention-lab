import torch

from dsv4_attention_lab.attention.sliding_window import SlidingWindowAttention, build_sliding_window_causal_mask
from dsv4_attention_lab.utils import AttentionOutput


def test_sliding_window_mask_correctness() -> None:
    mask = build_sliding_window_causal_mask(seq_len=5, window_size=2, device="cpu")
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_sliding_window_attention_shape_and_weights() -> None:
    module = SlidingWindowAttention(hidden_size=24, num_heads=3, window_size=2, dropout=0.0, use_rope=False)
    x = torch.randn(2, 7, 24)
    result = module(x, output_attentions=True)
    assert isinstance(result, AttentionOutput)
    assert result.output.shape == x.shape
    assert result.attention_weights is not None
    assert torch.isfinite(result.output).all()
    forbidden = ~build_sliding_window_causal_mask(7, 2, "cpu")
    assert torch.allclose(
        result.attention_weights[0, 0][forbidden],
        torch.zeros_like(result.attention_weights[0, 0][forbidden]),
    )

