"""Final-draft sampled-token score-function loss, not candidate-wise PPO."""

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.paper_opd import sampled_score_function_terms


def _sampled_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    scores = logits.float()
    return scores.gather(-1, tokens.long()[:, None]).squeeze(-1) - scores.logsumexp(-1)


def paper_opd_loss(args, batch, logits, sum_of_sample_mean):
    """Eq. 2: sum of per-response token means, scaled by Miles' outer reducer.

    The rollout hook has already centered over the complete optimization batch.
    Never recenter here: a microbatch/rank is not that batch. Chunk checkpointing
    bounds FP32 softmax activations without changing the normalization domain.
    """
    state = get_parallel_state()
    if state.tp.size != 1 or state.cp.size != 1:
        raise ValueError("Paper OPD loss requires training TP=CP=1.")
    if "opd_candidate_ids" in batch or "opd_reverse_kl" not in batch:
        raise ValueError("Paper loss requires centered sampled-token rewards, not candidate rewards.")
    if getattr(args, "normalize_advantages", False) or args.calculate_per_token_loss:
        raise ValueError("Paper loss requires unwhitened rewards and sequence-mean reduction.")
    vocab_size = args.opd_effective_vocab_size
    if logits.shape[-1] < vocab_size:
        raise ValueError("Student logits are smaller than the effective vocabulary.")
    chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    log_probs, entropies = [], []
    chunk_size = getattr(args, "log_probs_chunk_size", 128)
    chunk_size = chunk_size if chunk_size > 0 else 128
    for response_logits, tokens in chunks:
        if ((tokens < 0) | (tokens >= vocab_size)).any():
            raise ValueError("Response tokens fall outside the effective vocabulary.")
        for start in range(0, tokens.numel(), chunk_size):
            scores = response_logits[start : start + chunk_size, :vocab_size]
            ids = tokens[start : start + chunk_size]
            log_probs.append(checkpoint(_sampled_log_probs, scores, ids, use_reentrant=False))
            if getattr(args, "observe_training_entropy", False):
                with torch.no_grad():
                    full_log_p = scores.float().log_softmax(-1)
                    entropies.append(-(full_log_p.exp() * full_log_p).sum(-1))
    if not log_probs:
        zero = logits.sum(dtype=torch.float32) * 0
        return zero, {"loss": zero.detach(), "paper_opd_loss": zero.detach()}
    log_p = torch.cat(log_probs)
    reward = -torch.cat(batch["opd_reverse_kl"]).detach().to(log_p.device)
    mask = torch.cat(batch["loss_masks"]).to(device=log_p.device, dtype=torch.bool)
    loss_terms = sampled_score_function_terms(torch.where(mask, log_p, 0.0), torch.where(mask, reward, 0.0))
    loss = sum_of_sample_mean(loss_terms)
    metrics = {"loss": loss.detach(), "pg_loss": loss.detach(), "paper_opd_loss": loss.detach()}
    if entropies:
        metrics["entropy_loss"] = sum_of_sample_mean(torch.cat(entropies)).detach()
    return loss, metrics
