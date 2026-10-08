"""Wire exactness, malformed-buffer rejection, and both OPD reward contracts."""

import asyncio
import base64
import copy
import json
import pickle
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiohttp import web

from miles.rollout import on_policy_distillation as opd
from miles.rollout.shift_composition import compose_scorer_responses
from miles.utils import selected_logprob_transport as transport
from miles.utils.selected_logprob_transport import DECODED_FIELD, FIELD, REQUEST_FIELD, pack_selected_logprobs
from miles.utils.selected_logprobs import decode_score_response, selected_logprob_rows
from miles.utils.types import Sample


def test_lossless_ragged_rows_and_read_only_views():
    values = [None, [], [-0., -1.125, -float("inf")], [-123.5]]
    ids = [None, [], [7, 42, 151935], [42]]
    wire = json.loads(json.dumps(pack_selected_logprobs(values, ids)))
    assert wire["value_dtype"] == "<f4"
    rows = selected_logprob_rows({FIELD: wire}, 2)
    assert [list(row) for row in rows] == [list(zip(v, i)) for v, i in zip(values[-2:], ids[-2:])]
    assert np.signbit(rows[0].values[0])
    assert not rows[0].values.flags.writeable
    assert selected_logprob_rows({FIELD: wire}, 0) == []


def test_fp64_is_never_rounded_to_fp32():
    values = [[-0.1, -1.0000000000000002, -1e100, -1e-100]]
    wire = pack_selected_logprobs(values, [[1, 2, 3, 4]])
    assert wire["value_dtype"] == "<f8"
    row = selected_logprob_rows({FIELD: wire}, 1)[0]
    assert np.array_equal(row.values.view("<u8"), np.array(values[0], dtype="<f8").view("<u8"))


@pytest.mark.parametrize("values,ids", [([], []), ([None], [None]), ([[], []], [[], []])])
def test_empty_and_null_rows(values, ids):
    meta = {FIELD: pack_selected_logprobs(values, ids)}
    assert all(not row for row in selected_logprob_rows(meta, len(values)))


@pytest.mark.parametrize("values,ids", [
    ([[-1]], []), ([[1, 2]], [[7]]), ([None], [[]]),
    ([[-1]], [[-1]]), ([[-1]], [[2**32]]), ([[-1]], [[1.5]]),
    ([[float("nan")]], [[1]]), ([[float("inf")]], [[1]]),
])
def test_encoder_rejects_corrupt_rows(values, ids):
    with pytest.raises(ValueError):
        pack_selected_logprobs(values, ids)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("value_dtype", "<f2"), ("values_b64", "!"),
    ("values_b64", base64.b64encode(b"x").decode()), ("ids_b64", ""),
    ("offsets_b64", base64.b64encode(np.array([1, 1], dtype="<i8").tobytes()).decode()),
    ("offsets_b64", base64.b64encode(np.array([0, 2, 1], dtype="<i8").tobytes()).decode()),
])
def test_decoder_fails_closed(field, value):
    payload = pack_selected_logprobs([[-1.]], [[7]])
    payload[field] = value
    with pytest.raises(ValueError):
        selected_logprob_rows({FIELD: payload}, 1)


def test_too_short_response_and_empty_schema_fail_closed():
    for meta, length in [({FIELD: {}}, 1), ({FIELD: pack_selected_logprobs([], [])}, 1)]:
        with pytest.raises(ValueError):
            selected_logprob_rows(meta, length)


def response(values, ids, packed):
    if packed:
        return {"meta_info": {FIELD: pack_selected_logprobs(values, ids)}}
    return {"meta_info": {"input_token_ids_logprobs": [None, *[
        [[v, i, None] for v, i in zip(vs, ts, strict=True)]
        for vs, ts in zip(values, ids, strict=True)
    ]]}}


@pytest.mark.parametrize("center", ["none", "response"])
@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
def test_request_reward_and_gradient_bitwise_parity(monkeypatch, mode, center):
    args = SimpleNamespace(
        use_opd=True, opd_type="sglang", opd_target_mode=mode,
        opd_log_prob_top_k=3, opd_top_k_strategy="only-student",
        opd_reward_weight_mode="student_p", opd_reward_centering=center,
        opd_topk_per_position=True, opd_require_native_selected_ids=True,
        opd_teacher_urls=[f"{name}=http://teacher" for name in ("math", "code")],
        opd_teacher_adapters=[f"{name}={name}" for name in ("math", "code")],
        opd_base_urls=[f"{name}=http://teacher" for name in ("math", "code")],
        opd_anchor_url="http://anchor", reward_key=None,
    )
    sample = Sample(tokens=[99, 10, 12, 11], response_length=3, rollout_log_probs=[-2., -1.25, -.75],
                    loss_mask=[1, 0, 1], metadata={"opd_teacher": "math", "opd_student_top_logprobs": [
                        [[-.5, 10], [-1., 11], [-2., 12]],
                        [[-.75, 11], [-1.25, 12]], [[-.75, 11], [-1.5, 12], [-2.5, 13]],
                    ]})
    outputs = []
    for packed in (False, True, "decoded"):
        calls = []

        async def post(url, payload, timeout_secs=None):
            assert payload[REQUEST_FIELD]
            assert payload["logprob_start_len"] == 0
            calls.append((url, payload.get("lora_path")))
            ids = payload["token_ids_logprob_positions"]
            shift = {None: 0., "math": .125, "code": .25}[payload.get("lora_path")]
            shift += .375 if url == "http://anchor" else 0.
            values = [[-float(t)/8 - shift - p/4 for t in row] for p, row in enumerate(ids)]
            result = response(values, ids, packed)
            if packed == "decoded":
                result = decode_score_response(json.dumps(result).encode())
            result["_opd_transport"] = {"packed_response": float(bool(packed)), "response_bytes": 100}
            return result

        monkeypatch.setattr(opd, "_post_json", post)
        item = copy.deepcopy(sample)
        item.reward = asyncio.run(opd.reward_func(args, item))
        opd.post_process_rewards(args, [item])
        assert len(calls) == (4 if mode == "shiftmopd" else 1)
        assert item.metadata["opd_perf"]["packed_response"] == float(bool(packed)) * len(calls)
        assert item.metadata["opd_perf"]["response_bytes"] == 100 * len(calls)
        outputs.append(item)
    for other in outputs[1:]:
        assert torch.equal(torch.as_tensor(outputs[0].opd_reverse_kl), torch.as_tensor(other.opd_reverse_kl))
        if mode == "endpoint":
            for field in outputs[0].metadata["opd_candidates"]:
                assert torch.equal(outputs[0].metadata["opd_candidates"][field], other.metadata["opd_candidates"][field])
    coefficients = ([s.metadata["opd_candidates"]["opd_candidate_rewards"] for s in outputs]
                    if mode == "endpoint" else [torch.as_tensor(s.opd_reverse_kl) for s in outputs])
    logits = torch.zeros_like(coefficients[0], requires_grad=True)
    grads = [torch.autograd.grad((logits.exp() * c).sum(), logits)[0] for c in coefficients]
    assert all(torch.equal(grads[0], other) for other in grads[1:])


def test_shift_reordered_ids_fp64_and_shared_base_are_unchanged():
    rng = np.random.default_rng(8)
    ids = [[7, 4, 9], [8, 6]]
    matrices = [rng.normal(-5, 2, (2, 3)).tolist() for _ in range(4)]
    supports = [[9, 7, 4], [6, 8]]
    outputs = []
    for packed in (False, True):
        res = [response([m[0], m[1][:2]], ids, packed) for m in matrices]
        reward = dict(anchor=res[0], teachers=dict(math=res[1], code=res[2]),
                      bases=dict(math=res[3], code=res[3]), support_ids=supports)
        outputs.append(compose_scorer_responses(reward, torch.tensor([-1., -2.])))
    for a, b in zip(outputs[0][:2], outputs[1][:2]):
        assert torch.equal(a, b)
    assert outputs[1][2]["distinct_scorers"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("packed", [False, True])
async def test_http_request_response_and_transport_metrics(packed, monkeypatch):
    main_thread = threading.get_ident()
    decoder_threads = []

    def tracked_decode(*args, **kwargs):
        decoder_threads.append(threading.get_ident())
        return decode_score_response(*args, **kwargs)

    monkeypatch.setattr(opd, "decode_score_response", tracked_decode)
    async def score(request):
        payload = await request.json()
        assert payload[REQUEST_FIELD] is True
        assert payload["sampling_params"]["max_new_tokens"] == 0
        return web.json_response(response([[], [-1., -2.]], [[], [42, 43]], packed))

    app = web.Application()
    app.router.add_post("/generate", score)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = runner.addresses[0][1]
        result = await opd._post_json_per_position_compatible(
            f"http://127.0.0.1:{port}/generate",
            opd._score_payload([7, 42], token_ids_positions=[[], [42, 43]]),
            prompt_len=1, response_length=1, require_native=True,
        )
        assert opd._input_logprob_maps(result, "input_token_ids_logprobs", 1) == [{42: -1., 43: -2.}]
        assert result["_opd_transport"]["packed_response"] == float(packed)
        assert result["_opd_transport"]["response_bytes"] > 0
        assert result["_opd_transport"]["packed_decode_seconds"] >= 0
        assert FIELD not in result["meta_info"]
        assert (DECODED_FIELD in result["meta_info"]) == packed
        assert decoder_threads and all(thread != main_thread for thread in decoder_threads)
    finally:
        await runner.cleanup()


def test_batch_payload_propagates_transport_flag_without_changing_support():
    payload = opd._score_payload([7, 42], token_ids_positions=[[], [42, 43]])
    batch = opd._batch_payload([payload, payload], ["math", "code"])
    assert batch[REQUEST_FIELD] is True
    assert batch["token_ids_logprob_positions"] == [payload["token_ids_logprob_positions"]] * 2
    assert batch["lora_path"] == ["math", "code"]
    with pytest.raises(ValueError, match="transport format"):
        opd._batch_payload([payload, {**payload, REQUEST_FIELD: False}])


def test_randomized_bit_patterns_round_trip():
    rng = np.random.default_rng(28)
    for dtype in ("<f4", "<f8"):
        values = rng.normal(-10, 15, 8192).astype(dtype)
        ids = rng.integers(0, 151936, len(values)).tolist()
        wire = pack_selected_logprobs([values.tolist()], [ids])
        row = selected_logprob_rows({FIELD: wire}, 1)[0]
        assert np.array_equal(row.values.astype("<f8").view("<u8"), values.astype("<f8").view("<u8"))
        assert row.token_ids.tolist() == ids


def test_simd_wire_bytes_match_stdlib(monkeypatch):
    values = [[], [-0., -1.125, -float("inf")], [-0.1]]
    ids = [[], [1, 7, 42], [9]]
    simd = pack_selected_logprobs(values, ids)
    with monkeypatch.context() as context:
        context.setattr(transport, "pybase64", base64)
        assert simd == pack_selected_logprobs(values, ids)
        legacy_rows = selected_logprob_rows({FIELD: simd}, len(values))
    rows = selected_logprob_rows({FIELD: simd}, len(values))
    for a, b in zip(rows, legacy_rows, strict=True):
        assert a.values.tobytes() == b.values.tobytes()
        assert a.token_ids.tobytes() == b.token_ids.tobytes()


def test_decoded_buffers_survive_process_serialization_and_reuse_views():
    result = decode_score_response(json.dumps(response([[], [-0., -0.1]], [[], [42, 43]], True)).encode())
    assert FIELD not in result["meta_info"]
    for item in (result, pickle.loads(pickle.dumps(result, protocol=5))):
        buffer = item["meta_info"][DECODED_FIELD]
        row = selected_logprob_rows(item["meta_info"], 1)[0]
        assert np.shares_memory(row.values, buffer.values)
        assert np.shares_memory(row.token_ids, buffer.token_ids)
        assert not row.values.flags.writeable
        assert not row.token_ids.flags.writeable
        assert row.values.tobytes() == np.array([-0., -0.1], dtype="<f8").tobytes()
        assert row.token_ids.tolist() == [42, 43]


@pytest.mark.parametrize("bad_body", [b"null", b"[1]", b'{"meta_info": []}',
                                     b'{"meta_info": {"input_token_ids_logprobs_packed": {}}}'])
def test_http_decoder_rejects_invalid_responses(bad_body):
    with pytest.raises(ValueError):
        decode_score_response(bad_body)


def test_wire_cannot_spoof_validated_buffer():
    result = response([[-1.]], [[42]], True)
    result["meta_info"][DECODED_FIELD] = {"values": [123.]}
    decoded = decode_score_response(json.dumps(result).encode())
    assert list(selected_logprob_rows(decoded["meta_info"], 1)[0]) == [(-1., 42)]
    with pytest.raises(ValueError, match="locally decoded"):
        selected_logprob_rows({DECODED_FIELD: {}}, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_batch_http_decodes_in_worker_and_accounts_shared_cost_once(wrapped, monkeypatch):
    items = [response([[], [-1., -2.]], [[], [42, 43]], packed) for packed in (False, True)]
    body = json.dumps({"responses": items} if wrapped else items).encode()
    main_thread = threading.get_ident()
    decoder_threads = []

    def tracked_decode(*args, **kwargs):
        decoder_threads.append(threading.get_ident())
        return decode_score_response(*args, **kwargs)

    monkeypatch.setattr(opd, "decode_score_response", tracked_decode)

    async def score(request):
        payload = await request.json()
        assert len(payload["input_ids"]) == 2
        return web.Response(body=body, content_type="application/json")

    app = web.Application()
    app.router.add_post("/generate", score)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        payload = opd._score_payload([7, 42], token_ids_positions=[[], [42, 43]])
        result = await opd._post_json_batch(f"http://127.0.0.1:{runner.addresses[0][1]}/generate",
                                          [payload, payload], ["math", "code"])
        assert len(result) == 2
        assert sum(item["_opd_transport"]["response_bytes"] for item in result) == len(body)
        assert sum(item["_opd_transport"]["packed_response"] for item in result) == 1
        for item in result:
            assert opd._input_logprob_maps(item, "input_token_ids_logprobs", 1) == [{42: -1., 43: -2.}]
        assert FIELD not in result[1]["meta_info"]
        assert decoder_threads and all(thread != main_thread for thread in decoder_threads)
    finally:
        await runner.cleanup()


def test_packed_decode_metric_reaches_rollout_aggregation():
    from miles.ray.rollout.metrics import _compute_shiftmopd_metrics

    sample = Sample(metadata={"opd_perf": {"packed_decode_seconds": .125, "packed_response": 1.}})
    metrics = _compute_shiftmopd_metrics([sample, sample])
    assert metrics["opd_perf/packed_decode_seconds_sum"] == .25
    assert metrics["opd_perf/packed_response_sum"] == 2.
