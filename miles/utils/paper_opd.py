"""Equations 1, 2 and 7 of arXiv:2610.10460, independent of serving/training.

Vocabulary axis is last. Teacher selection happens BEFORE these functions;
neither target construction nor the sampled-token loss chooses a teacher.
"""

from collections.abc import Sequence

import torch


def compose_target_scores(
    anchor: torch.Tensor,
    teachers: Sequence[torch.Tensor],
    bases: Sequence[torch.Tensor],
    *,
    mode: str,
) -> torch.Tensor:
    """Compose unnormalized [positions, vocabulary-or-support] scores.

    Logits or full-model log-probabilities both work: model-specific additive
    constants cancel at the single FINAL target normalization. Do not normalize
    teacher/base differences separately or weight a loss over support tokens.
    """
    if mode not in ("endpoint", "shiftmopd") or not teachers:
        raise ValueError("Choose endpoint/shiftmopd and at least one selected teacher.")
    if mode == "shiftmopd" and len(teachers) != len(bases):
        raise ValueError("Every selected teacher needs its exact precursor.")
    references = [anchor] * len(teachers) if mode == "endpoint" else bases
    if any(value.shape != anchor.shape for value in [*teachers, *references]):
        raise ValueError("Scorers must use identical prefixes, positions and token IDs.")
    result = anchor.clone()
    for teacher, base in zip(teachers, references, strict=True):
        result = result + teacher - base
    return result


def target_log_probs(
    anchor: torch.Tensor,
    teachers: Sequence[torch.Tensor],
    bases: Sequence[torch.Tensor],
    sampled_ids: torch.Tensor,
    *,
    mode: str,
    effective_vocab_size: int,
    support_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense reference: exact numerator, full or restricted target partition.

    Inputs [N,V_padded], sampled_ids [N], optional support [N,K] with UNIQUE
    IDs. Sampled IDs need not belong to the partition support (Eq. 7).
    """
    if anchor.ndim != 2 or not 0 < effective_vocab_size <= anchor.shape[-1]:
        raise ValueError("Invalid effective vocabulary for [N,V] logits.")
    if any(value.shape != anchor.shape for value in [*teachers, *bases]):
        raise ValueError("Dense reference requires aligned model vocabularies.")
    if sampled_ids.shape != anchor.shape[:1]:
        raise ValueError("One sampled token is required per position.")
    for ids in [sampled_ids, *([] if support_ids is None else [support_ids])]:
        if ids.dtype != torch.long or ((ids < 0) | (ids >= effective_vocab_size)).any():
            raise ValueError("Token IDs must be int64 and inside the effective vocabulary.")
    scores = compose_target_scores(
        anchor[..., :effective_vocab_size],
        [value[..., :effective_vocab_size] for value in teachers],
        [value[..., :effective_vocab_size] for value in bases],
        mode=mode,
    )
    partition_scores = scores
    if support_ids is not None:
        if support_ids.ndim != 2 or support_ids.shape[0] != scores.shape[0] or support_ids.shape[1] == 0:
            raise ValueError("Support must be nonempty [N,K].")
        ordered = support_ids.sort(dim=-1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("Duplicate support IDs would overcount the partition.")
        partition_scores = scores.gather(-1, support_ids)
    return scores.gather(-1, sampled_ids[:, None]).squeeze(-1) - partition_scores.logsumexp(-1)


def center_batch_rewards(
    rewards: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Appendix B: ONE token-weighted baseline across an optimization batch.

    Run before DP splitting or microbatching. Padding and removed samples have
    zero mask. This is mean subtraction, never variance whitening.
    """
    if not rewards or len(rewards) != len(masks):
        raise ValueError("Expected matching nonempty reward/mask lists.")
    detached = [value.detach() for value in rewards]
    valid = [
        torch.as_tensor(mask, device=value.device, dtype=torch.bool)
        for value, mask in zip(detached, masks, strict=True)
    ]
    for value, mask in zip(detached, valid, strict=True):
        if value.ndim != 1 or value.shape != mask.shape or not torch.isfinite(value[mask]).all():
            raise ValueError("Expected finite valid rewards and matching 1D masks.")
    count = sum(int(mask.sum()) for mask in valid)
    if count == 0:
        raise ValueError("Optimization batch contains no valid sampled tokens.")
    baseline = sum(value[mask].double().sum() for value, mask in zip(detached, valid, strict=True)) / count
    centered = [
        torch.where(mask, value - baseline.to(value.dtype), 0.0) for value, mask in zip(detached, valid, strict=True)
    ]
    return centered, baseline


def sampled_score_function_terms(log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Unreduced Eq. 2 loss; reducer averages tokens per response, then responses."""
    if log_probs.shape != rewards.shape:
        raise ValueError("Sampled student log-probabilities and rewards must align.")
    return -rewards.detach() * log_probs
