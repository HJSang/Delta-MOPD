"""CPU routing/TITO contracts; HTTP transport is mocked, scorer logic is real."""

import ast
import copy
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from miles.rollout import on_policy_distillation as opd
from miles.utils.types import Sample


def _args(*, topology, mode, top_k, per_position):
    domains = ("math", "code", "if")
    urls = {d: f"http://{d}/generate" for d in domains}
    if topology == "multi_lora":
        urls = dict.fromkeys(domains, "http://shared/generate")
    return Namespace(
        use_opd=True, opd_type="sglang", opd_target_mode=mode,
        opd_teacher_urls=[f"{d}={url}" for d, url in urls.items()],
        opd_teacher_adapters=(
            [f"{d}={d}-v1" for d in domains] if topology == "multi_lora" else None
        ),
        opd_base_urls=[f"{d}=http://shared/generate" for d in domains],
        opd_anchor_url="http://anchor/generate",
        opd_log_prob_top_k=top_k, opd_top_k_strategy="only-student",
        opd_topk_per_position=per_position,
        opd_require_native_selected_ids=per_position,
    )


def _samples():
    return [
        Sample(
            index=i, tokens=[151644, 42 + i, 151645, 77, 151645], response_length=2,
            response="DELIBERATELY NOT THE TEXT OF THE TOKEN IDS",
            metadata={
                "opd_teacher": d,
                "opd_student_top_logprobs": [[[-0.1, 77], [-2.0, 80]], [[-0.2, 81], [-2.0, 82]]],
            },
        )
        for i, d in enumerate(("math", "code", "if"))
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("topology", ["standalone", "multi_lora"])
@pytest.mark.parametrize("mode,top_k", [("endpoint", 0), ("endpoint", 2), ("shiftmopd", 2)])
@pytest.mark.parametrize("per_position", [False, True])
@pytest.mark.parametrize("batched", [False, True])
async def test_scoring_preserves_token_ids_and_teacher_identity(
    monkeypatch, topology, mode, top_k, per_position, batched,
):
    args = _args(topology=topology, mode=mode, top_k=top_k, per_position=per_position)
    samples = _samples()
    originals = [copy.deepcopy(s.tokens) for s in samples]
    calls = []

    async def post(url, payload, timeout_secs=None):
        calls.append((url, copy.deepcopy(payload)))
        # Presence of the native field proves no compatibility fallback is used.
        rows = payload.get("token_ids_logprob_positions", [[] for _ in payload["input_ids"]])
        return {"meta_info": {"input_token_ids_logprobs": [
            [[-1.0, token] for token in row] for row in rows
        ]}}

    async def post_batch(url, payloads, adapters=None, timeout_secs=None):
        return [await post(url, opd._with_adapter(p, a), timeout_secs)
                for p, a in zip(payloads, adapters, strict=True)]

    monkeypatch.setattr(opd, "_post_json", post)
    monkeypatch.setattr(opd, "_post_json_batch", post_batch)
    if batched:
        await opd.reward_func(args, samples)
    else:
        for sample in samples:
            await opd.reward_func(args, sample)

    assert [s.tokens for s in samples] == originals
    for sample, tokens in zip(samples, originals, strict=True):
        sample_calls = [(u, p) for u, p in calls if p["input_ids"] == tokens]
        assert len(sample_calls) == (1 if mode == "endpoint" else 5)
        domain = sample.metadata["opd_teacher"]
        teachers = [domain] if mode == "endpoint" else ["math", "code", "if"]
        for teacher in teachers:
            expected = (("http://shared/generate", f"{teacher}-v1")
                        if topology == "multi_lora" else (f"http://{teacher}/generate", None))
            assert expected in [(u, p.get("lora_path")) for u, p in sample_calls]
        for _, payload in sample_calls:
            assert "text" not in payload and "prompt" not in payload
            assert payload["logprob_start_len"] == 0
            assert payload["sampling_params"]["max_new_tokens"] == 0
            assert payload["sampling_params"]["skip_special_tokens"] is False
            if top_k and per_position:
                positions = payload["token_ids_logprob_positions"]
                assert len(positions) == len(tokens)
                assert positions[:3] == [[], [], []]
                for token, support in zip(tokens[-2:], positions[-2:], strict=True):
                    assert token in support


def test_train_conversion_uses_original_tokens_not_response_text():
    # Load the production function unchanged, stubbing distributed infrastructure
    # only. No Ray/Megatron installation is needed for this conversion contract.
    path = Path(__file__).resolve().parents[1] / "miles/ray/rollout/train_data_conversion.py"
    tree = ast.parse(path.read_text())
    names = {"convert_samples_to_train_data", "_compute_rollout_mask_sums"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {
        "Any": Any, "Sample": Sample, "domain_gradients_enabled": lambda: False,
        "_post_process_rewards": lambda args, samples, **kwargs: ([0] * len(samples), [0] * len(samples)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    samples = _samples()
    original = [list(s.tokens) for s in samples]
    result = namespace["convert_samples_to_train_data"](
        Namespace(use_dynamic_global_batch_size=False), samples, {}, None, None,
    )
    assert result["tokens"] == original
    assert result["response_lengths"] == [2, 2, 2]
    assert result["loss_masks"] == [[1, 1], [1, 1], [1, 1]]
    for tokens, sample in zip(result["tokens"], samples, strict=True):
        assert tokens is sample.tokens
