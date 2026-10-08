"""Live transport parity for a shared base with math/code/IF adapters.

Pass --tokenizer with the local anchor tokenizer directory. This tests the
historical reward-conversion path, not the new paper objective's GPU readiness.
"""
import argparse
import asyncio
import json
from argparse import Namespace

import torch
from transformers import AutoTokenizer

from miles.rollout import on_policy_distillation as opd
from miles.utils.types import Sample


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer matching all scorer token IDs")
    parser.add_argument("--anchor", default="http://127.0.0.1:14147/generate")
    parser.add_argument("--teacher", default="http://127.0.0.1:14141/generate")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--response-tokens", type=int, default=129)
    cli = parser.parse_args()
    if cli.require_native:
        async def reject_fallback(*args, **kwargs):
            raise AssertionError("Native selected-ID scoring fell back to repeated flat requests")
        opd._bounded_flat_score = reject_fallback
    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain how to sum the first n positive integers."}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer.encode(
        "Pair the first and last terms. Each pair has the same sum, n plus one. "
        "There are n over two pairs, so the total is n times n plus one divided by two. "
        "For example, adding one through ten gives fifty five. " * 4,
        add_special_tokens=False)
    answer = (answer * ((cli.response_tokens + len(answer) - 1) // len(answer)))[:cli.response_tokens]
    tokens = prompt + answer
    prompt_len = len(prompt)
    response_len = len(tokens) - prompt_len
    student_scores = await opd._post_json(cli.anchor or cli.teacher,
                                          opd._score_payload(tokens, top_k=16), timeout_secs=120)
    top = opd._trim_input_field(student_scores["meta_info"], "input_top_logprobs", response_len)
    sampled_logps = opd._teacher_sampled_log_probs(student_scores, response_len).tolist()
    sample = Sample(tokens=tokens, response_length=response_len,
                    rollout_log_probs=sampled_logps,
                    metadata={"opd_student_top_logprobs": top, "opd_teacher": "math"})
    support = opd._composition_support_ids(sample, top)
    positions = [[] for _ in range(prompt_len)] + support
    routes = [(cli.teacher, adapter) for adapter in (None, "math", "code", "if")]
    if cli.anchor:
        routes.append((cli.anchor, None))
    dense_maps, sparse_maps = {}, {}
    for url, adapter in routes:
        dense = await opd._post_json(
            url, opd._with_adapter(opd._score_payload(tokens, token_ids=opd._ordered_unique(
                token for row in support for token in row)), adapter), timeout_secs=120)
        sparse = await opd._post_json_per_position_compatible(
            url, opd._with_adapter(opd._score_payload(tokens, token_ids_positions=positions), adapter),
            prompt_len=prompt_len, response_length=response_len, chunk_size=64, timeout_secs=120)
        a = opd._input_logprob_maps(dense, "input_token_ids_logprobs", response_len)
        b = opd._input_logprob_maps(sparse, "input_token_ids_logprobs", response_len)
        dense_maps[(url, adapter)], sparse_maps[(url, adapter)] = a, b
        assert len(b) == response_len
        assert all(set(row) == set(ids) for row, ids in zip(b, support, strict=True))
        delta = max(abs(a[i][token] - b[i][token]) for i, ids in enumerate(support) for token in ids)
        # BF16 prefill shape changes may shift logits; no support/alignment tolerance.
        worst = sorted([(abs(a[i][token] - b[i][token]), i, token, a[i][token], b[i][token])
                        for i, ids in enumerate(support) for token in ids], reverse=True)[:5]
        print(json.dumps({"worst": worst}), flush=True)
        errors = []
        for i, ids in enumerate(support):
            weights = torch.tensor([dict((entry[1], entry[0]) for entry in top[i]).get(
                token, sampled_logps[i]) for token in ids]).softmax(0)
            errors.append(sum(float(w) * abs(a[i][token] - b[i][token])
                              for w, token in zip(weights, ids, strict=True)))
        print(json.dumps({"weighted_mean_abs_logprob_delta": sum(errors) / len(errors),
                          "sampled_mean_abs_logprob_delta": sum(abs(a[i][tokens[prompt_len+i]]-
                              b[i][tokens[prompt_len+i]]) for i in range(response_len))/response_len}), flush=True)
        if not cli.diagnose:
            assert delta < 1e-5, (url, adapter, delta)
        print(json.dumps({"route": url, "adapter": adapter, "max_abs_logprob_delta": delta,
                          "retained_entries": sum(map(len, b)), "dense_entries": sum(map(len, a))}), flush=True)

    if cli.anchor:
        def composition(maps):
            return opd._compute_shiftmopd_reverse_kl(
                maps[(cli.anchor, None)],
                [maps[(cli.teacher, name)] for name in ("math", "code", "if")],
                [maps[(cli.teacher, None)]] * 3, torch.tensor(sampled_logps), support)[0]
        reward_delta = (composition(dense_maps) - composition(sparse_maps)).abs()
        print(json.dumps({"centered_shift_reward_mean_abs_delta": reward_delta.mean().item(),
                          "centered_shift_reward_max_abs_delta": reward_delta.max().item()}), flush=True)
        if not cli.diagnose:
            assert reward_delta.max() < 1e-5, reward_delta

    for mode in (["endpoint", "shiftmopd"] if cli.anchor else ["endpoint"]):
        args = Namespace(use_opd=True, opd_type="sglang", opd_target_mode=mode,
                         opd_log_prob_top_k=16, opd_topk_per_position=True,
                         opd_top_k_strategy="only-student", opd_reward_weight_mode="student_p",
                         opd_teacher_urls=[f"{name}={cli.teacher}" for name in ("math", "code", "if")],
                         opd_teacher_adapters=[f"{name}={name}" for name in ("math", "code", "if")],
                         opd_base_urls=[f"{name}={cli.teacher}" for name in ("math", "code", "if")],
                         opd_anchor_url=cli.anchor, reward_key=None,
                         sglang_router_request_timeout_secs=120)
        sample.reward = (await opd.reward_func(args, [sample]))[0]
        opd.post_process_rewards(args, [sample])
        values = torch.as_tensor(sample.opd_reverse_kl)
        assert values.shape == (response_len,) and torch.isfinite(values).all()
        print(json.dumps({"mode": mode, "reward_conversion": "PASS", "mean_kl": values.mean().item()}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
