"""Paper-equation, gradient isolation, causal alignment, and logger-hook tests."""

import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from experiments.measure_shiftmopd_checkpoint import _selected_logits, main
from miles.backends.training_utils.domain_gradients import domain_gradient_metrics
from miles.backends.training_utils.paper_diagnostics import component_gradient_metrics, full_vocab_kl_metrics
from miles.rollout.paper_diagnostic_hooks import diagnostic_panel, save_eval_traces, save_panel


def _case():
    torch.manual_seed(7)
    return {
        "student_logits": torch.randn(4, 7),
        "anchor_logits": torch.randn(4, 7),
        "teacher_logits": {name: torch.randn(4, 7) for name in ("math", "code", "if")},
        "base_logits": {name: torch.randn(4, 7) for name in ("math", "code", "if")},
        "vocab_size": 7,
    }


def test_full_vocab_kl_directions_match_explicit_probabilities():
    case = _case()
    result = full_vocab_kl_metrics(**case)
    target = case["anchor_logits"] + sum(case["teacher_logits"][k] - case["base_logits"][k] for k in case["teacher_logits"])
    q, p, a = [x.double().softmax(-1) for x in (target, case["student_logits"], case["anchor_logits"])]
    prefix = "paper/full_vocab/shiftmopd/"
    assert result[prefix + "target_to_student_kl"] == pytest.approx(float((q * (q.log() - p.log())).sum(-1).mean()), abs=1e-6)
    assert result[prefix + "student_to_target_kl"] == pytest.approx(float((p * (p.log() - q.log())).sum(-1).mean()), abs=1e-6)
    assert result[prefix + "target_to_anchor_kl"] == pytest.approx(float((q * (q.log() - a.log())).sum(-1).mean()), abs=1e-6)
    assert result[prefix + "target_to_student_kl"] != pytest.approx(result[prefix + "student_to_target_kl"])
    assert result["paper/full_vocab/positions"] == 4


def test_additive_logit_offsets_cancel_and_equal_origin_reduction():
    case = _case()
    reference = full_vocab_kl_metrics(**case)
    offset = torch.arange(4).unsqueeze(-1) * 10
    moved = {**case, "teacher_logits": {k: v + offset for k, v in case["teacher_logits"].items()}}
    result = full_vocab_kl_metrics(**moved)
    for key, value in reference.items():
        assert result[key] == pytest.approx(value, abs=3e-6)
    case["base_logits"] = {k: case["anchor_logits"] for k in case["teacher_logits"]}
    result = full_vocab_kl_metrics(**case)
    for metric in ("target_to_student_kl", "target_to_anchor_kl", "target_entropy"):
        assert result[f"paper/full_vocab/shiftmopd/{metric}"] == pytest.approx(result[f"paper/full_vocab/endpoint_composite/{metric}"])


def test_identity_kl_and_no_frozen_gradients():
    x = torch.randn(3, 5, requires_grad=True)
    result = full_vocab_kl_metrics(student_logits=x, anchor_logits=x, teacher_logits={"x": x}, base_logits={"x": x}, vocab_size=5)
    for name, value in result.items():
        if name.endswith("_kl"):
            assert abs(value) < 1e-7
    assert x.grad is None


@pytest.mark.parametrize("failure", ["topk", "nonfinite", "positions", "empty", "base_missing"])
def test_full_vocab_rejects_bad_inputs(failure):
    case = _case()
    if failure == "topk":
        case["vocab_size"] = 151936
    elif failure == "nonfinite":
        case["student_logits"][0, 0] = float("nan")
    elif failure == "positions":
        case["teacher_logits"]["math"] = torch.zeros(1, 7)
    elif failure == "empty":
        case["student_logits"] = torch.zeros(0, 7)
    else:
        del case["base_logits"]["math"]
    with pytest.raises(ValueError):
        full_vocab_kl_metrics(**case)


def test_component_gradients_match_paper_and_preserve_optimizer_update():
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 7)
    unused = torch.nn.Parameter(torch.ones(2))
    reference = copy.deepcopy(model)
    inputs = torch.randn(4, 3)
    logits = model(inputs)
    case = _case()
    sampled = torch.tensor([0, 2, 4, 6])
    params = [*model.parameters(), unused]
    for p in params:
        p.grad = torch.ones_like(p)
        p.main_grad = torch.full_like(p, 2)
    result = component_gradient_metrics(
        student_logits=logits,
        teacher_logits=case["teacher_logits"],
        base_logits=case["base_logits"],
        sampled_ids=sampled,
        parameters=params + params,
        vocab_size=7,
    )
    student = logits.log_softmax(-1).gather(1, sampled[:, None]).squeeze(-1)
    for name in case["teacher_logits"]:
        teacher = case["teacher_logits"][name].log_softmax(-1).gather(1, sampled[:, None]).squeeze(-1)
        base = case["base_logits"][name].log_softmax(-1).gather(1, sampled[:, None]).squeeze(-1)
        gradients = []
        for reward in (teacher - base, base - student, teacher - student):
            g = torch.autograd.grad(-(reward.detach() * student).mean(), list(model.parameters()), retain_graph=True)
            gradients.append(torch.cat([v.reshape(-1) for v in g]).double())
        shift, inherited, endpoint = gradients
        prefix = f"paper/gradient/{name}/"
        assert result[prefix + "shift_norm"] == pytest.approx(float(shift.norm()), abs=1e-7)
        assert result[prefix + "base_norm"] == pytest.approx(float(inherited.norm()), abs=1e-7)
        assert result[prefix + "endpoint_norm"] == pytest.approx(float(endpoint.norm()), abs=2e-7)
        assert result[prefix + "gamma_base"] == pytest.approx(float(inherited.norm() / (shift.norm() + 1e-8)))
    for p in params:
        torch.testing.assert_close(p.grad, torch.ones_like(p))
        torch.testing.assert_close(p.main_grad, torch.full_like(p, 2))
    # The original graph is still usable, and the optimizer sees the same update.
    for m in (model, reference):
        m.zero_grad()
    logits.square().mean().backward()
    reference(inputs).square().mean().backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for m in (model, reference):
        torch.optim.AdamW(m.parameters(), lr=0.01).step()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_zero_component_has_defined_gamma_but_no_cosine():
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    result = component_gradient_metrics(student_logits=parameter, teacher_logits={"x": parameter.detach()}, base_logits={"x": parameter.detach()}, sampled_ids=torch.tensor([0, 1]), parameters=[parameter], vocab_size=3)
    assert result["paper/gradient/x/shift_norm"] == 0
    assert result["paper/gradient/x/gamma_base"] == 0
    assert result["paper/gradient/x/base_shift_cos_defined"] == 0
    assert "paper/gradient/x/base_shift_cos" not in result


def test_domain_geometry_is_not_confused_with_teacher_components():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[1.0, 1.0]])
    result = domain_gradient_metrics([model], {"math": [torch.tensor([[1.0, 0.0]])], "if": [torch.tensor([[0.0, 1.0]])]})
    assert result["grad/domain_geometry/cancellation"] == pytest.approx(1 - 2**0.5 / 2)
    assert result["grad/domain_geometry/imbalance"] == 1


def _sample(domain="math", **kwargs):
    return Namespace(
        **{
            "metadata": {"domain": domain},
            "remove_sample": False,
            "index": 1,
            "tokens": [0, 1, 2, 3, 4, 5],
            "response_length": 4,
            "loss_mask": [0, 1, 1, 0],
            "prompt": "test",
            "response": "answer",
            "label": "label",
            "reward": {"acc": 1.0},
            **kwargs,
        }
    )


def test_panels_exclude_padding_removed_rows_and_do_not_mutate_samples(tmp_path):
    samples = [_sample(remove_sample=True), _sample(), _sample("if", index=2)]
    original = copy.deepcopy(samples)
    panel = diagnostic_panel(samples, max_positions=1)
    assert [r["domain"] for r in panel] == ["math", "if"]
    assert panel[0]["response_positions"] == [3]
    assert panel[0]["tokens"] == [0, 1, 2, 3]
    args = Namespace(save=str(tmp_path / "checkpoints"), rollout_temperature=0.7, opd_target_mode="shiftmopd", opd_log_prob_top_k=128)
    assert save_panel(1, args, samples, None, 1) is False
    assert not (tmp_path / "paper_diagnostics").exists()
    assert save_panel(19, args, samples, None, 1) is False
    saved = json.loads((tmp_path / "paper_diagnostics/panel_19.json").read_text())
    assert saved["support_k"] == 128
    assert "per_domain" in saved["sampling_policy"]
    assert save_eval_traces(19, args, {"dev": {"samples": samples}}) is False
    assert len(json.loads((tmp_path / "eval_traces/eval_19.json").read_text())["records"]) == 3
    assert samples == original


def test_selected_logits_predict_tokens_at_p_minus_one():
    embeddings = torch.nn.Embedding(7, 3)
    head = torch.nn.Linear(3, 7)
    model = Namespace(
        model=lambda input_ids, use_cache: Namespace(last_hidden_state=embeddings(input_ids).cumsum(1)),
        get_output_embeddings=lambda: head,
    )
    record = {"tokens": [1, 2, 3, 4, 5], "response_positions": [2, 4]}
    actual = _selected_logits(model, record, 7, "cpu")
    expected = head(embeddings(torch.tensor([1, 2, 3, 4])).cumsum(0)[[1, 3]])
    torch.testing.assert_close(actual, expected)
    changed = {**record, "tokens": [1, 2, 3, 4, 6]}
    torch.testing.assert_close(_selected_logits(model, changed, 7, "cpu"), actual)


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_real_qwen_checkpoint_runner_cpu(tmp_path, monkeypatch, family):
    torch.manual_seed(42)
    config_class, model_class = (Qwen2Config, Qwen2ForCausalLM) if family == "qwen2" else (Qwen3Config, Qwen3ForCausalLM)
    config = config_class(vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"v{i}": i for i in range(7)}, unk_token="v0")), unk_token="v0", bos_token="v0", eos_token="v0", pad_token="v0")
    paths = {}
    for role in ("student", "anchor", "base", "teacher"):
        path = tmp_path / role
        model = model_class(config).eval()
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        paths[role] = str(path)
    record = {"tokens": [1, 2, 3, 4, 5], "response_positions": [2, 4], "domain": "if", "sample_index": 0}
    full = model(torch.tensor([record["tokens"]]), use_cache=False).logits[0, [1, 3]]
    selected = _selected_logits(model, record, 7, "cpu")
    torch.testing.assert_close(selected, full, atol=1e-7, rtol=1e-6)
    panel_path, spec_path, output = (tmp_path / name for name in ("panel.json", "spec.json", "result.json"))
    panel_path.write_text(json.dumps({"student_temperature": 0.7, "rollout_id": 19, "sampling_policy": "test", "records": [record]}))
    spec_path.write_text(json.dumps({"student": paths["student"], "anchor": paths["anchor"], "teachers": {"if": paths["teacher"]}, "bases": {"if": paths["base"]}, "panel": str(panel_path), "checkpoint_step": 20, "source_run_id": "unit-test"}))
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--spec", str(spec_path), "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert result["effective_vocab_size"] == 7
    assert result["student_temperature"] == 0.7
    metrics = result["results"][0]["metrics"]
    assert metrics["paper/full_vocab/shiftmopd/target_to_student_kl"] >= 0
    assert metrics["paper/gradient/if/shift_norm"] > 0
    assert metrics["paper/gradient/positions"] == 2
