from argparse import Namespace

import pytest
import torch

import miles.rollout.shift_composition as fast
from miles.rollout.on_policy_distillation import _compute_shiftmopd_reverse_kl_reference as reference
from miles.rollout.on_policy_distillation import post_process_rewards
from miles.utils.types import Sample


def make_case(n, k, seed=8):
    generator = torch.Generator().manual_seed(seed)
    supports = [list(range(k-(p % 3))) for p in range(n)]
    matrices = [-torch.rand((n, k), generator=generator, dtype=torch.float64)*20 for _ in range(5)]
    maps = [[dict(enumerate(row)) for row in matrix.tolist()] for matrix in matrices]
    return maps, -torch.rand(n, generator=generator), supports


@pytest.mark.parametrize("n,k,tile", [(1,8,1), (17,16,4), (257,128,64), (4097,128,2048)])
@pytest.mark.parametrize("center", [False, True])
def test_matches_unchanged_scalar_reward_and_gradient(n, k, tile, center):
    maps, student, supports = make_case(n,k)
    mask = torch.arange(n) % 4 != 1
    args = (maps[0], maps[1:4], [maps[4]]*3, student, supports)
    expected, raw_expected = reference(*args, loss_mask=mask, center=center)
    actual, raw_actual = fast.compose_reverse_kl(*args, loss_mask=mask, center=center, tile_size=tile)
    torch.testing.assert_close(raw_actual, raw_expected, atol=4e-6, rtol=1e-6)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-6)
    # Detached reward coefficients through a real softmax and one optimizer step.
    logits = torch.zeros(n, 3, requires_grad=True)
    old = logits.detach().log_softmax(-1)[:,0]
    for reward in (expected, actual):
        loss = ((logits.log_softmax(-1)[:,0]-old).exp()*reward.detach()).mean()
        grad = torch.autograd.grad(loss, logits)[0]
        if reward is expected:
            expected_grad = grad
        else:
            torch.testing.assert_close(grad, expected_grad, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(logits.detach()-1e-4*grad, logits.detach()-1e-4*expected_grad, atol=1e-9, rtol=1e-5)


def test_native_rows_fallback_order_shared_base_and_response_mean(monkeypatch):
    maps, student, supports = make_case(7,8)
    def response(rows):
        return {"meta_info": {"input_token_ids_logprobs": [None, *[
            [[v,k,None] for k,v in reversed(list(row.items()))] for row in rows
        ]]}, "_opd_transport": {"response_bytes": 100}}
    responses = [response(m) for m in maps]
    reward = dict(anchor=responses[0], teachers=dict(zip("abc",responses[1:4])),
                  bases={k:responses[4] for k in "abc"}, support_ids=supports, mode="shiftmopd")
    calls = []
    align = fast._aligned_tile
    def observed(*args):
        calls.append(id(args[0]))
        return align(*args)
    monkeypatch.setattr(fast, "_aligned_tile", observed)
    centered, raw, metrics = fast.compose_scorer_responses(reward, student, tile_size=4)
    expected, expected_raw = reference(maps[0],maps[1:4],[maps[4]]*3,student,supports)
    torch.testing.assert_close(raw, expected_raw, atol=4e-6, rtol=1e-6)
    torch.testing.assert_close(centered, expected, atol=4e-6, rtol=1e-6)
    assert len(calls) == 10  # five distinct scorers, two tiles
    assert metrics["response_bytes"] == 500  # not 700 for the shared base
    sample = Sample(tokens=[99, *[ids[-1] for ids in supports]], response_length=7,
                    rollout_log_probs=student.tolist(), reward=reward, metadata={})
    args = Namespace(opd_target_mode="shiftmopd", reward_key=None, opd_composition_tile_size=4)
    post_process_rewards(args, [sample])
    assert sample.metadata["opd_perf"]["distinct_scorers"] == 5


def test_compact_raw_value_id_transport_matches_nested_triples():
    maps, student, supports = make_case(9, 8)

    def nested(rows):
        return {"meta_info": {"input_token_ids_logprobs": [None, *[
            [[v, k, None] for k, v in row.items()] for row in rows
        ]]}}

    def compact(rows):
        return {"meta_info": {
            "input_token_ids_logprobs_val": [[v for v in row.values()] for row in rows],
            "input_token_ids_logprobs_idx": [[k for k in row.keys()] for row in rows],
        }}

    def make_reward(factory):
        responses = [factory(m) for m in maps]
        return dict(anchor=responses[0], teachers=dict(zip("abc", responses[1:4])),
                    bases={key: responses[4] for key in "abc"},
                    support_ids=supports, mode="shiftmopd")

    nested_centered, nested_raw, _ = fast.compose_scorer_responses(
        make_reward(nested), student, tile_size=4
    )
    compact_centered, compact_raw, _ = fast.compose_scorer_responses(
        make_reward(compact), student, tile_size=4
    )
    torch.testing.assert_close(compact_raw, nested_raw, atol=4e-6, rtol=1e-6)
    torch.testing.assert_close(compact_centered, nested_centered, atol=4e-6, rtol=1e-6)


@pytest.mark.parametrize("bad", ["duplicate", "empty", "missing", "nonfinite", "length", "tile"])
def test_fail_closed(bad):
    maps, student, support = make_case(4,8)
    kwargs = {}
    if bad == "duplicate": support[0] = [0,0]
    if bad == "empty": support[0] = []
    if bad == "missing": del maps[0][0][0]
    if bad == "nonfinite": maps[0][0][0] = float("nan")
    if bad == "length": student = student[:-1]
    if bad == "tile": kwargs["tile_size"] = 0
    with pytest.raises(ValueError):
        fast.compose_reverse_kl(maps[0],maps[1:4],[maps[4]]*3,student,support,**kwargs)


def test_empty_response_and_all_masked():
    centered, raw = fast.compose_reverse_kl([], [[]], [[]], torch.empty(0), [])
    assert centered.numel() == raw.numel() == 0
    maps, student, supports = make_case(4,8)
    centered, _ = fast.compose_reverse_kl(maps[0],maps[1:4],[maps[4]]*3,student,supports,loss_mask=[0]*4)
    assert not centered.any()
