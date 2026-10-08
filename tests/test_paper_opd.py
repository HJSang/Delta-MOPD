"""Final-draft equations and real hook/loss boundaries; CPU, no model downloads."""

import asyncio
import copy
import json
from argparse import Namespace

import pytest
import torch
import numpy as np
from transformers import Qwen2Config, Qwen2ForCausalLM

import miles.backends.training_utils.loss_hub.paper_opd as loss_module
import miles.rollout.on_policy_distillation as legacy
from miles.rollout.paper_opd import post_process_rewards
from miles.utils.paper_opd import center_batch_rewards, compose_target_scores, target_log_probs
from miles.utils.paper_opd_config import effective_vocab_bias, validate_paper_opd_args
from miles.utils.types import Sample
from miles.utils.paper_eval import PAPER_SUITES, paper_eval_configs, summarize_paper_eval
from miles.utils.paper_opd_preflight import validate_paper_model_manifest, validate_vocabulary_maps
from miles.backends.training_utils.paper_diagnostics import full_vocab_geometry_metrics


def config(**updates):
    values = dict(
        opd_objective="paper",
        custom_rm_path="miles.rollout.on_policy_distillation.reward_func",
        custom_reward_post_process_path="miles.rollout.on_policy_distillation.post_process_rewards",
        use_opd=True,
        opd_type="sglang",
        loss_type="policy_loss",
        advantage_estimator="grpo",
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_min_p=0.0,
        rollout_top_k=-1,
        opd_kl_coef=1.0,
        opd_teacher_selection="all",
        opd_target_mode="shiftmopd",
        opd_anchor_url="anchor",
        opd_teacher_urls=["math=teacher-math", "science=teacher-science"],
        opd_base_urls=["math=base", "science=base"],
        opd_effective_vocab_size=5,
        opd_topk_fallback_chunk_size=2,
        opd_paper_partition="full",
        opd_top_k_strategy="only-student",
        opd_log_prob_top_k=0,
        opd_teacher_key="opd_teacher",
        global_batch_size=2,
        reward_key=None,
        calculate_per_token_loss=False,
        true_on_policy_mode=False,
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=1,
        observe_training_entropy=True,
    )
    return Namespace(**(values | updates))


@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
def test_dense_equation_with_effective_vocabulary_and_offset_invariance(mode):
    torch.manual_seed(9)
    anchor, t1, t2, b1, b2 = torch.randn(5, 3, 9, dtype=torch.float64)
    ids = torch.tensor([0, 1, 4])
    references = [anchor, anchor] if mode == "endpoint" else [b1, b2]
    expected = (anchor + t1 - references[0] + t2 - references[1])[:, :5].log_softmax(-1)
    actual = target_log_probs(anchor, [t1, t2], [b1, b2], ids, mode=mode, effective_vocab_size=5)
    torch.testing.assert_close(actual, expected.gather(-1, ids[:, None]).squeeze(-1))
    anchor[:, 5:] = 1e5  # Padded logits cannot affect the partition.
    changed = target_log_probs(anchor + 5, [t1 - 7, t2 + 3], [b1 - 2, b2 + 4], ids, mode=mode, effective_vocab_size=5)
    torch.testing.assert_close(actual, changed)


def test_same_origin_and_single_endpoint_invariants():
    anchor, t1, t2 = torch.randn(3, 4, 7, dtype=torch.float64)
    shift = compose_target_scores(anchor, [t1, t2], [anchor, anchor], mode="shiftmopd")
    endpoint = compose_target_scores(anchor, [t1, t2], [], mode="endpoint")
    torch.testing.assert_close(shift, endpoint)
    single = compose_target_scores(anchor, [t1], [], mode="endpoint")
    torch.testing.assert_close(single, t1)


def test_restricted_partition_bias_is_exact_missing_mass_not_candidate_loss():
    anchor, teacher, base = torch.randn(3, 2, 5, dtype=torch.float64)
    ids, support = torch.tensor([4, 3]), torch.tensor([[0, 1], [0, 2]])
    full_target = (anchor + teacher - base).log_softmax(-1)
    actual = target_log_probs(
        anchor, [teacher], [base], ids, mode="shiftmopd", effective_vocab_size=5, support_ids=support
    )
    expected = full_target.gather(-1, ids[:, None]).squeeze(-1) - full_target.exp().gather(-1, support).sum(-1).log()
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="Duplicate"):
        target_log_probs(
            anchor,
            [teacher],
            [base],
            ids,
            mode="shiftmopd",
            effective_vocab_size=5,
            support_ids=torch.zeros(2, 2, dtype=torch.long),
        )


def test_one_token_weighted_baseline_not_mean_of_response_means():
    rewards = [torch.tensor([1.0, 3.0, float("nan")], requires_grad=True), torch.tensor([8.0], requires_grad=True)]
    centered, baseline = center_batch_rewards(rewards, [torch.tensor([1, 1, 0]), torch.tensor([1])])
    assert baseline == 4.0
    torch.testing.assert_close(centered[0], torch.tensor([-3.0, -1.0, 0.0]))
    torch.testing.assert_close(centered[1], torch.tensor([4.0]))
    assert not baseline.requires_grad and not any(x.requires_grad for x in centered)
    assert sum(x.sum() for x in centered) == 0
    with pytest.raises(ValueError, match="no valid"):
        center_batch_rewards([torch.tensor([1.0])], [torch.tensor([0])])


@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
@pytest.mark.parametrize("selection", ["all", "routed"])
@pytest.mark.parametrize("partition", ["full", "student-topk"])
def test_scoring_transport_matches_dense_reference(monkeypatch, mode, selection, partition):
    args = config(
        opd_target_mode=mode,
        opd_teacher_selection=selection,
        opd_paper_partition=partition,
        opd_log_prob_top_k=2 if partition == "student-topk" else 0,
    )
    torch.manual_seed(11)
    models = {
        name: torch.randn(2, 7, dtype=torch.float64) for name in ["anchor", "teacher-math", "teacher-science", "base"]
    }
    # Score against the server's larger vocabulary: constants must cancel.
    model_logp = {name: value.log_softmax(-1) for name, value in models.items()}
    calls = []

    async def fake_score(url, payload, **kwargs):
        calls.append(url)
        assert payload["input_ids"] == [6, 3, 4]
        assert payload["logprob_start_len"] == 0
        rows = payload["token_ids_logprob_positions"][-2:]
        return {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    *[
                        [[float(model_logp[url][row, token]), token, None] for token in ids]
                        for row, ids in enumerate(rows)
                    ],
                ]
            }
        }

    monkeypatch.setattr(legacy, "_post_json_per_position_compatible", fake_score)
    sample = Sample(
        tokens=[6, 3, 4],
        response_length=2,
        rollout_log_probs=[-2.0, -3.0],
        metadata={
            "opd_teacher": "math",
            "opd_student_top_logprobs": [[[-1.0, 0, None], [-2.0, 1, None]]] * 2,
        },
    )
    result = asyncio.run(legacy.reward_func(args, sample))
    names = ["math", "science"] if selection == "all" else ["math"]
    supports = torch.tensor([[0, 1, 3], [0, 1, 4]]) if partition == "student-topk" else None
    expected = target_log_probs(
        models["anchor"],
        [models[f"teacher-{name}"] for name in names],
        [models["base"]] * len(names),
        torch.tensor([3, 4]),
        mode=mode,
        effective_vocab_size=5,
        support_ids=supports,
    )
    torch.testing.assert_close(torch.tensor(result["target_log_probs"]), expected.float())
    assert result["selected_teachers"] == names
    assert ("teacher-science" in calls) == (selection == "all")
    blocks = 4 if partition == "full" else 1
    assert calls.count("anchor") == blocks
    assert calls.count("base") == (blocks if mode == "shiftmopd" else 0)


def test_postprocessing_preserves_global_batch_baseline_and_excludes_removed_samples():
    samples = [
        Sample(
            tokens=[0, 1, 2],
            response_length=2,
            rollout_log_probs=[-5.0, -5.0],
            reward={"mode": "paper", "target_log_probs": [-4.0, -2.0], "selected_teachers": ["math"]},
        ),
        Sample(
            tokens=[0, 1],
            response_length=1,
            rollout_log_probs=[-5.0],
            reward={"mode": "paper", "target_log_probs": [-1.0], "selected_teachers": ["science"]},
        ),
    ]
    # Use the production dispatch, not just the new module's direct entrypoint.
    legacy.post_process_rewards(config(), samples)
    expected = torch.tensor([1.0, 3.0, 4.0]) - 8 / 3
    torch.testing.assert_close(-torch.cat([sample.opd_reverse_kl for sample in samples]), expected)
    assert all("opd_candidates" not in sample.metadata for sample in samples)
    samples[1].remove_sample = True
    post_process_rewards(config(), samples)
    torch.testing.assert_close(samples[0].opd_reverse_kl, torch.tensor([1.0, -1.0]))
    assert samples[1].opd_reverse_kl == 0.0
    with pytest.raises(ValueError, match="complete optimization"):
        post_process_rewards(config(), samples[:1])


def test_loss_is_detached_sampled_score_function_and_padding_has_zero_gradient(monkeypatch):
    monkeypatch.setattr(
        loss_module, "get_parallel_state", lambda: Namespace(tp=Namespace(size=1), cp=Namespace(size=1))
    )
    # get_responses consults its own parallel-state binding.
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_parallel_state",
        lambda: Namespace(tp=Namespace(size=1), cp=Namespace(size=1)),
    )
    logits = torch.randn(1, 5, 7, requires_grad=True)
    rewards = [torch.tensor([-1.0, 0.0], requires_grad=True), torch.tensor([1.0], requires_grad=True)]
    batch = {
        "unconcat_tokens": [torch.tensor([0, 1, 2]), torch.tensor([0, 3])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 0]), torch.tensor([1])],
        "opd_reverse_kl": [-value for value in rewards],
    }

    def reducer(terms):
        return terms[0] + terms[2]  # Sum of masked response means.

    loss, _ = loss_module.paper_opd_loss(config(), batch, logits, reducer)
    log_p = logits[..., :5].log_softmax(-1)
    reference = log_p[0, 0, 1] - log_p[0, 3, 3]
    torch.testing.assert_close(loss, reference)
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert all(value is None for value in torch.autograd.grad(loss, rewards, allow_unused=True))
    expected_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert not actual_grad[..., 5:].any()
    assert not actual_grad[0, 1:3].any()
    assert all(value.grad is None for value in rewards)


@pytest.mark.parametrize(
    "changes",
    [
        {"normalize_advantages": True},
        {"opd_reward_centering": "response"},
        {"opd_teacher_selection": None},
        {"rollout_temperature": 0.7},
        {"rollout_top_p": 0.95},
        {"opd_log_prob_top_k": 16},
        {"num_steps_per_rollout": 2},
        {"tensor_model_parallel_size": 2},
        {"fully_async": True},
        {"opd_paper_partition": "student-topk"},
        {"use_opd": False},
        {"calculate_per_token_loss": True},
    ],
)
def test_paper_profile_rejects_silent_legacy_semantics(changes):
    validate_paper_opd_args(config())
    with pytest.raises(ValueError):
        validate_paper_opd_args(config(**changes))


def test_effective_vocabulary_rollout_bias():
    assert effective_vocab_bias(5, 7) == {"5": -1e30, "6": -1e30}
    assert effective_vocab_bias(5, 5) == {}
    with pytest.raises(ValueError):
        effective_vocab_bias(8, 7)


def test_same_size_tokenizers_with_permuted_ids_are_rejected():
    vocab = {"a": 0, "b": 1, "c": 2}
    validate_vocabulary_maps({"student": vocab, "teacher": vocab}, 3)
    with pytest.raises(ValueError, match="tokenizer IDs differ"):
        validate_vocabulary_maps({"student": vocab, "teacher": {"a": 1, "b": 0, "c": 2}}, 3)
    with pytest.raises(ValueError, match="complete"):
        validate_vocabulary_maps({"student": {"a": 0, "b": 2}}, 3)


def test_manifest_checks_init_anchor_identity_and_named_routes(tmp_path, monkeypatch):
    path = str(tmp_path / "model")
    record = {"path": path, "repo_id": "example/pinned-model", "revision": "a" * 40}
    manifest = {
        "student": record,
        "anchor": record,
        "teachers": {name: record for name in ("math", "science")},
        "bases": {name: record for name in ("math", "science")},
    }
    manifest_path = tmp_path / "models.json"
    monkeypatch.setattr(
        "miles.utils.paper_opd_preflight.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Namespace(get_vocab=lambda: {str(i): i for i in range(5)}),
    )
    args = config(hf_checkpoint=path, opd_paper_model_manifest=str(manifest_path))
    manifest_path.write_text(json.dumps(manifest))
    validate_paper_model_manifest(args)
    manifest["anchor"] = record | {"revision": "b" * 40}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="anchor identity"):
        validate_paper_model_manifest(args)
    manifest["anchor"] = record
    del manifest["teachers"]["math"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="teacher names"):
        validate_paper_model_manifest(args)


def test_geometry_centering_and_conflict_overlap_convention():
    anchor = torch.zeros(2, 5)
    teacher = torch.tensor([[3.0, 2.0, -1.0, -2.0, -2.0]] * 2)
    kwargs = dict(
        anchor_logits=anchor,
        teacher_logits={"a": teacher, "b": -teacher},
        base_logits={"a": anchor, "b": anchor},
        vocab_size=5,
        top_k=2,
    )
    metrics = full_vocab_geometry_metrics(**kwargs)
    assert metrics["paper/geometry/shiftmopd/cancellation"] == pytest.approx(1.0)
    assert metrics["paper/geometry/shiftmopd/a_vs_b/cos"] == pytest.approx(-1.0)
    assert metrics["paper/geometry/shiftmopd/sign_conflict"] == 1.0
    shifted = full_vocab_geometry_metrics(**(kwargs | {"teacher_logits": {"a": teacher + 12, "b": -teacher - 7}}))
    for key in metrics:
        assert shifted[key] == pytest.approx(metrics[key], abs=1e-8)


def test_paper_eval_protocol_and_avg_not_pass_at_k():
    paths = {suite.name: f"/frozen/{suite.name}.jsonl" for suite in PAPER_SUITES}
    rms = dict.fromkeys(paths, "official-verifier-placeholder")
    configs = {config.name: config for config in paper_eval_configs(paths, rms, protocol="avg")}
    assert sum(suite.items for suite in PAPER_SUITES) == 1309
    assert configs["aime2025"].n_samples_per_eval_prompt == 16
    assert configs["math500"].temperature == 0.0
    assert configs["ifeval"].max_response_len == 2048
    assert configs["amc2023"].top_p == 0.95
    scores = {suite.name: np.zeros((suite.items, suite.samples)) for suite in PAPER_SUITES}
    scores["amc2023"][:, 0] = 1  # Every item has a success, but Avg@16 is 1/16, not 1.
    metrics = summarize_paper_eval(scores, protocol="avg")
    assert metrics["amc2023"] == 1 / 16
    assert metrics["five_suite_macro"] == 1 / 80
    with pytest.raises(ValueError, match="complete"):
        summarize_paper_eval(scores | {"amc2023": scores["amc2023"][:-1]}, protocol="avg")


@pytest.mark.parametrize("mode", ["endpoint", "shiftmopd"])
def test_tiny_qwen2_student_rollout_scoring_and_update(monkeypatch, mode):
    """Real CPU model forwards/backward; fake only the HTTP transport boundary."""
    torch.manual_seed(17)
    model_config = Qwen2Config(
        vocab_size=7,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    student = Qwen2ForCausalLM(model_config).eval()
    frozen = {name: Qwen2ForCausalLM(model_config).eval() for name in ["teacher-math", "teacher-science", "base"]}
    frozen["anchor"] = copy.deepcopy(student).requires_grad_(False)
    args = config(opd_target_mode=mode)
    samples = []
    with torch.no_grad():
        for first_token in [0, 1]:
            tokens, old = [first_token], []
            for _ in range(2):
                log_p = student(torch.tensor([tokens])).logits[0, -1, :5].log_softmax(-1)
                token = int(torch.multinomial(log_p.exp(), 1))
                tokens.append(token)
                old.append(float(log_p[token]))
            samples.append(Sample(tokens=tokens, response_length=2, rollout_log_probs=old, metadata={}))

    async def fake_score(url, payload, **kwargs):
        with torch.no_grad():
            values = frozen[url](torch.tensor([payload["input_ids"]])).logits[0, :-1].log_softmax(-1)
        ids = payload["token_ids_logprob_positions"][1:]
        return {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    *[
                        [[float(values[position, token]), token, None] for token in support]
                        for position, support in enumerate(ids)
                    ],
                ]
            }
        }

    monkeypatch.setattr(legacy, "_post_json_per_position_compatible", fake_score)
    for sample in samples:
        sample.reward = asyncio.run(legacy.reward_func(args, sample))
    legacy.post_process_rewards(args, samples)
    state = Namespace(tp=Namespace(size=1), cp=Namespace(size=1))
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr("miles.backends.training_utils.loss_hub.logit_processors.get_parallel_state", lambda: state)
    logits = torch.cat([student(torch.tensor([sample.tokens])).logits for sample in samples], dim=1)
    batch = dict(
        unconcat_tokens=[torch.tensor(sample.tokens) for sample in samples],
        total_lengths=[3, 3],
        response_lengths=[2, 2],
        loss_masks=[torch.ones(2), torch.ones(2)],
        opd_reverse_kl=[sample.opd_reverse_kl for sample in samples],
    )
    loss, metrics = loss_module.paper_opd_loss(args, batch, logits, lambda terms: terms.mean())
    before = student.get_input_embeddings().weight.detach().clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3)
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None
    )
    assert all(p.grad is None for model in frozen.values() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, student.get_input_embeddings().weight)
    assert torch.isfinite(metrics["entropy_loss"])
