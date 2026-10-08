import asyncio
from argparse import Namespace

import torch
import pytest

import miles.rollout.on_policy_distillation as opd
from miles.rollout.on_policy_distillation import (
    _composition_support_ids,
    _compute_shiftmopd_reverse_kl,
)
from miles.utils.types import Sample


@pytest.mark.parametrize("batched", [False, True])
def test_sparse_shift_support_includes_out_of_topk_sample_and_postprocesses(monkeypatch, batched):
    sample = Sample(
        tokens=[99, 4, 2], response_length=2,
        rollout_log_probs=[-1.1, -1.2],
        metadata={"opd_student_top_logprobs": [[[-0.1, 7]], [[-0.2, 8]]]},
    )
    args = Namespace(
        opd_target_mode="shiftmopd", opd_log_prob_top_k=1,
        opd_topk_per_position=True, opd_teacher_urls=["math=http://teacher"],
        opd_base_urls=["math=http://base"], opd_anchor_url="http://anchor",
        reward_key=None,
    )
    seen = []

    async def fake_post(url, payload, timeout_secs=None):
        seen.append(payload)
        if "token_ids_logprob_positions" in payload:
            assert payload["token_ids_logprob_positions"] == [[], [7, 4], [8, 2]]
            return {"meta_info": {}}
        rows = [None] + [
            [[-0.01 * (token_id + position), token_id, None]
             for token_id in payload["token_ids_logprob"]]
            for position in range(payload["logprob_start_len"] + 1, len(payload["input_ids"]))
        ]
        return {"meta_info": {"input_token_ids_logprobs": rows}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    result = asyncio.run(opd.reward_func(args, [sample] if batched else sample))
    sample.reward = result[0] if batched else result
    opd.post_process_rewards(args, [sample])
    assert len(sample.opd_reverse_kl) == 2
    assert torch.isfinite(torch.tensor(sample.opd_reverse_kl)).all()
    assert abs(sum(sample.opd_reverse_kl)) < 1e-6
    assert len(seen) == 6


def test_shiftmopd_support_places_sampled_token_last_and_deduplicates():
    sample = Sample(tokens=[9, 4, 2], response_length=2)
    student_top = [[[0.0, 4], [0.0, 7]], [[0.0, 2], [0.0, 8]]]

    assert _composition_support_ids(sample, student_top) == [[7, 4], [8, 2]]


def test_shiftmopd_composes_teacher_base_deltas_and_centers_reward():
    anchor = [{1: -0.2, 2: -1.1}, {1: -0.4, 2: -0.8}]
    teachers = [[{1: -0.1, 2: -1.4}, {1: -0.2, 2: -1.0}]]
    bases = [[{1: -0.3, 2: -0.9}, {1: -0.6, 2: -0.5}]]
    student = torch.tensor([-0.5, -0.7])
    support = [[1, 2], [1, 2]]

    centered, raw = _compute_shiftmopd_reverse_kl(anchor, teachers, bases, student, support)

    expected_raw = []
    for position in range(2):
        scores = torch.tensor(
            [
                anchor[position][1] + teachers[0][position][1] - bases[0][position][1],
                anchor[position][2] + teachers[0][position][2] - bases[0][position][2],
            ]
        )
        target_log_prob = scores[1] - torch.logsumexp(scores, dim=0)
        expected_raw.append(student[position] - target_log_prob)
    expected_raw = torch.stack(expected_raw)

    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(centered, expected_raw - expected_raw.mean())
    torch.testing.assert_close(centered.mean(), torch.tensor(0.0))


def test_shiftmopd_requires_matching_teacher_and_base_counts():
    try:
        _compute_shiftmopd_reverse_kl(
            anchor_log_probs=[{1: 0.0}],
            teacher_log_probs=[[{1: 0.0}]],
            base_log_probs=[],
            student_sampled_log_probs=torch.tensor([0.0]),
            support_ids=[[1]],
        )
    except ValueError as exc:
        assert "teacher/base count mismatch" in str(exc)
    else:
        raise AssertionError("expected teacher/base mismatch to fail")


def test_shiftmopd_global_support_includes_sampled_tokens(monkeypatch):
    sample = Sample(
        tokens=[99, 4, 2],
        response_length=2,
        metadata={"opd_student_top_logprobs": [[[-0.1, 7]], [[-0.2, 8]]]},
    )
    args = Namespace(
        opd_target_mode="shiftmopd",
        opd_log_prob_top_k=1,
        opd_topk_per_position=False,
        opd_teacher_urls=["math=http://teacher"],
        opd_base_urls=["math=http://base"],
        opd_anchor_url="http://anchor",
        sglang_router_request_timeout_secs=None,
    )
    seen = []

    async def fake_post(url, payload, timeout_secs=None):
        del url, timeout_secs
        seen.append(payload["token_ids_logprob"])
        return {"meta_info": {"input_token_ids_logprobs": [None, [], []]}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    try:
        asyncio.run(opd.reward_func(args, sample))
    except (KeyError, ValueError):
        # The intentionally empty fake response is enough to inspect the request;
        # reward extraction is covered by the existing scorer tests.
        pass
    assert seen
    assert set(seen[0]) == {2, 4, 7, 8}


def test_shiftmopd_deduplicates_shared_base_url(monkeypatch):
    sample = Sample(
        tokens=[99, 4, 2],
        response_length=2,
        metadata={"opd_student_top_logprobs": [[[-0.1, 7]], [[-0.2, 8]]]},
    )
    args = Namespace(
        opd_target_mode="shiftmopd",
        opd_log_prob_top_k=1,
        opd_topk_per_position=False,
        opd_teacher_urls=[
            "math=http://math",
            "code=http://code",
            "if=http://if",
        ],
        opd_base_urls=[
            "math=http://base",
            "code=http://base",
            "if=http://base",
        ],
        opd_anchor_url="http://anchor",
        sglang_router_request_timeout_secs=None,
    )
    seen = []

    async def fake_post(url, payload, timeout_secs=None):
        del payload, timeout_secs
        seen.append(url)
        return {"meta_info": {}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    reward = asyncio.run(opd.reward_func(args, sample))

    assert reward["mode"] == "shiftmopd"
    assert seen == ["http://anchor", "http://math", "http://code", "http://if", "http://base"]


def test_shiftmopd_reward_func_and_post_process_wire_all_scorers(monkeypatch):
    sample = Sample(
        tokens=[99, 4, 2],
        response_length=2,
        rollout_log_probs=[-0.5, -0.7],
        metadata={
            "opd_student_top_logprobs": [
                [[-0.1, 4], [-0.5, 7]],
                [[-0.2, 2], [-0.6, 8]],
            ]
        },
    )
    args = Namespace(
        opd_target_mode="shiftmopd",
        opd_log_prob_top_k=2,
        opd_topk_per_position=True,
        opd_teacher_urls=["math=http://teacher"],
        opd_base_urls=["math=http://base"],
        opd_anchor_url="http://anchor",
        opd_teacher_key="opd_teacher",
        reward_key=None,
        sglang_router_request_timeout_secs=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
    )

    scorer_maps = {
        "anchor": [{7: -0.2, 4: -1.0}, {8: -0.3, 2: -0.9}],
        "teacher": [{7: -0.1, 4: -1.1}, {8: -0.2, 2: -1.0}],
        "base": [{7: -0.4, 4: -0.8}, {8: -0.5, 2: -0.7}],
    }

    async def fake_post(url, payload, timeout_secs=None):
        del payload, timeout_secs
        kind = url.rsplit("://", 1)[-1]
        maps = scorer_maps[kind]
        return {"meta_info": {"input_token_ids_logprobs": [None, [[v, k] for k, v in maps[0].items()], [[v, k] for k, v in maps[1].items()]]}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    reward = asyncio.run(opd.reward_func(args, sample))
    sample.reward = reward
    raw_rewards, rewards = opd.post_process_rewards(args, [sample])

    assert len(raw_rewards) == len(rewards) == 1
    assert sample.opd_reverse_kl is not None
    assert len(sample.opd_reverse_kl) == 2
    assert abs(sum(sample.opd_reverse_kl)) < 1e-6
    assert "shiftmopd_raw_reward_mean" in sample.metadata
