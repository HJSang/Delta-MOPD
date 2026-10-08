"""Candidate-wise Open-MOPD PPO; do not reduce the K axis before clipping."""

from argparse import Namespace

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.opd_candidates import OPD_CANDIDATE_DTYPES


def selected_log_probs(logits: torch.Tensor, ids: torch.Tensor, temperature: float, model_precision: bool = False) -> torch.Tensor:
    """Full-vocabulary normalization, followed by candidate gathering (not top-K softmax)."""
    if temperature <= 0:
        raise ValueError("Candidate OPD requires a positive student scoring temperature.")
    if model_precision:
        if temperature != 1.0:
            raise ValueError("True-on-policy response logits are already temperature scaled.")
        return logits.log_softmax(-1).gather(-1, ids)
    scores = logits.float() / temperature
    return scores.gather(-1, ids) - scores.logsumexp(-1, keepdim=True)


def candidate_ppo_loss(
    current: torch.Tensor,
    old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-position loss/clip fraction, summing candidates only AFTER PPO."""
    if not (current.shape == old.shape == rewards.shape == mask.shape) or current.ndim != 2:
        raise ValueError("Candidate OPD expects matching [response, candidate] tensors.")
    # Sanitize padding before exp/multiply: masked NaNs must not poison backward.
    current = torch.where(mask, current, 0.0)
    old = torch.where(mask, old.detach(), 0.0)
    advantages = torch.where(mask, rewards.detach(), 0.0)
    if not (torch.isfinite(current).all() and torch.isfinite(old).all() and torch.isfinite(advantages).all()):
        raise ValueError("Non-finite active candidate OPD input.")
    losses, clipped = compute_policy_loss(old - current, advantages, eps_clip, eps_clip_high, eps_clip_c)
    return losses.sum(-1), (clipped * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def candidate_opd_loss(args: Namespace, batch: dict, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Packed/padded single-training-GPU baseline; unsupported layouts fail closed.

    Candidate old probabilities use SGLang's default temperature-scaled output
    log-probs, so recompute at rollout_temperature, independent of the teacher's
    prefill scoring temperature. Do not enable SGLANG_RETURN_ORIGINAL_LOGPROB.
    """
    state = get_parallel_state()
    if not getattr(args, "use_opd", False):
        raise ValueError("Candidate OPD batch requires --use-opd.")
    if state.tp.size != 1 or state.cp.size != 1:
        raise ValueError("Candidate OPD currently requires training TP=1 and CP=1.")
    if args.use_tis or args.use_opsm or getattr(args, "get_mismatch_metrics", False) or args.advantage_estimator == "gspo":
        raise ValueError("Candidate OPD does not support sampled-action TIS/OPSM/GSPO corrections.")
    if getattr(args, "normalize_advantages", False):
        raise ValueError("Open-MOPD candidate rewards must not be whitened.")

    chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    losses, metrics = (
        [],
        {
            key: []
            for key in (
                "opd_candidate_clipfrac",
                "opd_candidate_logprob_abs_diff",
                "opd_candidate_old_support_mass",
                "opd_candidate_reward_abs",
                "opd_candidate_count",
            )
        },
    )
    temperature = 1.0 if getattr(args, "true_on_policy_mode", False) else args.rollout_temperature
    chunk_size = getattr(args, "log_probs_chunk_size", 128) or 128
    chunk_size = chunk_size if chunk_size > 0 else 128
    for i, (response_logits, _) in enumerate(chunks):
        vocab_size = getattr(args, "vocab_size", None)
        if vocab_size is not None:
            response_logits = response_logits[..., :vocab_size]
        ids, old, rewards, mask = [torch.as_tensor(batch[key][i], device=logits.device).detach() for key in OPD_CANDIDATE_DTYPES]
        ids, mask = ids.long(), mask.bool()
        if ids.shape[0] != response_logits.shape[0]:
            raise ValueError("Candidate OPD response/logit alignment mismatch.")
        active = torch.as_tensor(batch["loss_masks"][i], device=logits.device).bool().unsqueeze(-1)
        mask = mask & active
        ids = torch.where(mask, ids, 0)
        current = []
        for start in range(0, ids.shape[0], chunk_size):
            current.append(
                checkpoint(
                    selected_log_probs,
                    response_logits[start : start + chunk_size],
                    ids[start : start + chunk_size],
                    temperature,
                    getattr(args, "true_on_policy_mode", False),
                    use_reentrant=False,
                )
            )
        current = torch.cat(current) if current else response_logits.new_zeros(ids.shape, dtype=torch.float32)
        position_loss, clipfrac = candidate_ppo_loss(
            current,
            old,
            rewards * args.opd_kl_coef,
            mask,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
            eps_clip_c=getattr(args, "eps_clip_c", None),
        )
        losses.append(position_loss)
        count = mask.sum(-1).clamp_min(1)
        metrics["opd_candidate_clipfrac"].append(clipfrac.detach())
        metrics["opd_candidate_logprob_abs_diff"].append(torch.where(mask, (current.detach() - old).abs(), 0.0).sum(-1) / count)
        metrics["opd_candidate_old_support_mass"].append(torch.where(mask, old.exp(), 0.0).sum(-1))
        metrics["opd_candidate_reward_abs"].append(torch.where(mask, rewards.abs(), 0.0).sum(-1))
        metrics["opd_candidate_count"].append(mask.sum(-1).float())
    return torch.cat(losses), {key: torch.cat(values) for key, values in metrics.items()}
