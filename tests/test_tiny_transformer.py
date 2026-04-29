import torch

from dsv4_attention_lab.configs import TinyTransformerConfig
from dsv4_attention_lab.models.tiny_transformer import TinyTransformer


def test_tiny_transformer_forward_dense() -> None:
    config = TinyTransformerConfig(
        vocab_size=32,
        max_seq_len=16,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_type="dense",
    )
    model = TinyTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    labels = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids, labels=labels)
    assert output["logits"].shape == (2, 10, config.vocab_size)
    assert torch.isfinite(output["loss"])


def test_tiny_transformer_forward_hybrid_backward() -> None:
    config = TinyTransformerConfig(
        vocab_size=32,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_type="hybrid",
        compression_ratio=2,
        hca_compression_ratio=4,
        top_k=2,
        window_size=3,
    )
    model = TinyTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    labels = torch.randint(0, config.vocab_size, (2, 12))
    loss = model(input_ids, labels=labels)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

