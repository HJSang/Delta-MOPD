"""Paper sampled-token OPD hooks; legacy candidate/PPO hooks remain separate.

All frozen models score the same student token IDs. This adapter assumes
tokenizer-aligned models; Appendix F cross-tokenizer projection is not provided.
"""

import asyncio

import torch

from miles.rollout.on_policy_distillation import (
    _adapter_map,
    _composition_support_ids,
    _score_payload,
    _score_unique_routes,
    _student_top_logprobs,
    _teacher_route_for_sample,
    parse_teacher_urls,
)
from miles.rollout.shift_composition import _aligned_tile
from miles.utils.paper_opd import center_batch_rewards, compose_target_scores
from miles.utils.selected_logprobs import selected_logprob_rows


def _selected_routes(args, sample):
    teachers = parse_teacher_urls(args.opd_teacher_urls)
    names = list(teachers) if args.opd_teacher_selection == "all" else [_teacher_route_for_sample(args, sample)[0]]
    if not names or (args.opd_teacher_selection == "all" and "default" in names and len(names) > 1):
        raise ValueError("Composition needs explicit teachers, not an additional fallback 'default' teacher.")
    teacher_adapters = _adapter_map(getattr(args, "opd_teacher_adapters", None), "opd-teacher-adapters")
    base_adapters = _adapter_map(getattr(args, "opd_base_adapters", None), "opd-base-adapters")
    base_urls = parse_teacher_urls(getattr(args, "opd_base_urls", None))
    anchor_adapter = getattr(args, "opd_anchor_adapter", None)
    anchor = (args.opd_anchor_url, None if anchor_adapter == "none" else anchor_adapter)
    teacher_routes = [(teachers[name], teacher_adapters.get(name)) for name in names]
    if args.opd_teacher_selection == "all" and len(set(teacher_routes)) != len(teacher_routes):
        raise ValueError("All-teacher composition needs unique models; do not count a shared Science/IF teacher twice.")
    bases = (
        [(base_urls[name], base_adapters.get(name)) for name in names] if args.opd_target_mode == "shiftmopd" else []
    )
    return names, anchor, teacher_routes, bases


async def _scores_on_support(args, sample, routes, supports):
    """Gather compact per-position scores, deduplicating shared (URL, adapter)."""
    _, anchor, teachers, bases = routes
    prompt_length = len(sample.tokens) - sample.response_length
    payload = _score_payload(sample.tokens, token_ids_positions=[[]] * prompt_length + supports)
    responses = await _score_unique_routes(
        [anchor, *teachers, *bases],
        payload,
        getattr(args, "sglang_router_request_timeout_secs", None),
        (prompt_length, sample.response_length, args.opd_topk_fallback_chunk_size),
        require_native=getattr(args, "opd_require_native_selected_ids", False),
    )
    width = max(map(len, supports))
    scores = {}
    for route, response in responses.items():
        rows = selected_logprob_rows(response["meta_info"], sample.response_length)
        scores[route] = torch.from_numpy(_aligned_tile(rows, supports, 0, width))
    target = compose_target_scores(
        scores[anchor],
        [scores[route] for route in teachers],
        [scores[route] for route in bases],
        mode=args.opd_target_mode,
    )
    lengths = torch.tensor([len(ids) for ids in supports])
    return target.masked_fill(torch.arange(width)[None, :] >= lengths[:, None], -torch.inf)


async def _score_sample(args, sample):
    routes = _selected_routes(args, sample)
    n = sample.response_length
    if n == 0:
        return {"mode": "paper", "target_log_probs": [], "selected_teachers": routes[0]}
    sampled = sample.tokens[-n:]
    vocab_size = args.opd_effective_vocab_size
    if any(token < 0 or token >= vocab_size for token in sampled):
        raise ValueError("Student sampled a token outside the effective vocabulary.")
    if args.opd_paper_partition == "student-topk":
        supports = _composition_support_ids(sample, _student_top_logprobs(sample, n))
        if any(token < 0 or token >= vocab_size for ids in supports for token in ids):
            raise ValueError("Student support includes padded/out-of-vocabulary IDs.")
        scores = await _scores_on_support(args, sample, routes, supports)
        numerator = scores[torch.arange(n), torch.tensor([len(ids) - 1 for ids in supports])]
        log_z = scores.logsumexp(-1)
    else:
        # Exact reference backend: stream vocabulary tiles; never allocate [N,V]
        # for every frozen model. This repeats scorer requests and is expensive;
        # it is a correctness path, not a claim of optimized GPU throughput.
        numerator = (await _scores_on_support(args, sample, routes, [[token] for token in sampled])).squeeze(-1)
        log_z = torch.full((n,), -torch.inf, dtype=torch.float64)
        width = args.opd_topk_fallback_chunk_size
        for start in range(0, vocab_size, width):
            ids = list(range(start, min(start + width, vocab_size)))
            scores = await _scores_on_support(args, sample, routes, [ids] * n)
            log_z = torch.logaddexp(log_z, scores.logsumexp(-1))
    target = numerator - log_z
    if not torch.isfinite(target).all():
        raise ValueError("Non-finite paper target log-probability.")
    return {"mode": "paper", "target_log_probs": target.float().tolist(), "selected_teachers": routes[0]}


async def reward_func(args, sample, **kwargs):
    """Score both arms through the identical sampled-token pipeline."""
    if isinstance(sample, list):
        return await asyncio.gather(*(_score_sample(args, item) for item in sample))
    return await _score_sample(args, sample)


def post_process_rewards(args, samples, **kwargs):
    """Center the COMPLETE one-update batch before any DP/CP/microbatch split."""
    if len(samples) != args.global_batch_size:
        raise ValueError(
            "Paper OPD requires exactly one complete optimization batch per rollout; refusing local centering."
        )
    rewards, masks = [], []
    for sample in samples:
        payload = sample.get_reward_value(args)
        if not isinstance(payload, dict) or payload.get("mode") != "paper":
            raise ValueError("Paper OPD requires the paper scorer hook, not legacy candidate rewards.")
        if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
            raise ValueError("Paper OPD needs aligned student rollout log-probabilities.")
        target = torch.tensor(payload["target_log_probs"], dtype=torch.float32)
        student = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
        if target.shape != student.shape:
            raise ValueError("Teacher target and student response lengths differ.")
        mask = torch.tensor(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length)
        if sample.remove_sample:
            mask.zero_()
        rewards.append(target - student)
        masks.append(mask.bool())
    centered, baseline = center_batch_rewards(rewards, masks)
    for sample, raw, reward, mask in zip(samples, rewards, centered, masks, strict=True):
        # Existing train-data transport and CP slicing already carry this field.
        sample.opd_reverse_kl = -reward
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata.pop("opd_candidates", None)
        sample.metadata["paper_opd"] = {
            "baseline": float(baseline),
            "partition": args.opd_paper_partition,
            "selected_teachers": sample.get_reward_value(args)["selected_teachers"],
            "raw_reward_sum": float(raw[mask].sum()),
            "valid_tokens": int(mask.sum()),
            "centered_reward_sum": float(reward[mask].sum()),
        }
    return [0.0] * len(samples), [0.0] * len(samples)
