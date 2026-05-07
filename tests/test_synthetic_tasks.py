import torch

from dsv4_attention_lab.experiments.synthetic_long_context import IGNORE_INDEX, SUPPORTED_TASKS, make_batch


def test_synthetic_tasks_shape_and_supervision() -> None:
    for task in SUPPORTED_TASKS:
        input_ids, labels = make_batch(task, batch_size=2, seq_len=64, vocab_size=128, device="cpu")
        assert input_ids.shape == (2, 64)
        assert labels.shape == (2, 64)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
        assert torch.any(labels != IGNORE_INDEX)


def test_sparse_label_tasks_do_not_supervise_background() -> None:
    for task in ["copy_first", "associative_recall", "multi_query_retrieval"]:
        _, labels = make_batch(task, batch_size=2, seq_len=64, vocab_size=128, device="cpu")
        ignored = int((labels == IGNORE_INDEX).sum().item())
        assert ignored > labels.numel() // 2
