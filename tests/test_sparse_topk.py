import torch

from dsv4_attention_lab.attention.sparse_topk import SparseTopKAttention
from dsv4_attention_lab.utils import AttentionOutput


def test_sparse_topk_shape_and_no_nans() -> None:
    module = SparseTopKAttention(num_heads=2, head_dim=4, top_k=3, causal=True)
    q = torch.randn(1, 2, 5, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    out = module(q, k, v, block_ends=torch.tensor([0, 2, 4]))
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_sparse_topk_handles_small_visible_context() -> None:
    module = SparseTopKAttention(num_heads=1, head_dim=4, top_k=8, causal=True)
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    result = module(
        q,
        k,
        v,
        block_ends=torch.tensor([1, 2]),
        output_attentions=True,
        return_selected_indices=True,
    )
    assert isinstance(result, AttentionOutput)
    assert result.output.shape == q.shape
    assert torch.isfinite(result.output).all()
    assert result.attention_weights is not None
    assert result.selected_indices is not None
    assert torch.allclose(result.output[:, :, 0, :], torch.zeros_like(result.output[:, :, 0, :]))


def test_sparse_topk_backward() -> None:
    module = SparseTopKAttention(num_heads=2, head_dim=4, top_k=2, causal=False, indexer_dim=2)
    q = torch.randn(1, 2, 4, 4, requires_grad=True)
    k = torch.randn(1, 2, 4, 4, requires_grad=True)
    v = torch.randn(1, 2, 4, 4, requires_grad=True)
    loss = module(q, k, v).pow(2).mean()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

