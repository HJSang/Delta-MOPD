"""Validate accepted two-domain rollout batches without changing any loss."""

from collections import Counter

from .paper_diagnostic_hooks import save_panel


def check_and_save_panel(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    expected = args.rollout_batch_size // 2
    counts = Counter(s.metadata["domain"] for s in samples)
    if counts != {"math": expected, "code": expected}:
        raise ValueError(f"Accepted rollout is not the frozen 1:1 batch: {dict(counts)}")
    if any(s.remove_sample for s in samples):
        raise ValueError("A removed sample would change the two-domain training mixture")
    return save_panel(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
