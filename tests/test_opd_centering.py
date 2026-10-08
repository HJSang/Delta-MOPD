"""Center sampled-token baselines, not already probability-weighted candidates."""
from argparse import Namespace

import pytest
import torch

from miles.utils.opd_centering import center_sampled_rewards, opd_centering_mode
from miles.utils.types import Sample
from miles.rollout.on_policy_distillation import _compute_topk_candidates, _compute_shiftmopd_reverse_kl, post_process_rewards
from miles.backends.training_utils.loss_hub.candidate_opd import candidate_ppo_loss
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages


def case(mask=None):
    student = torch.tensor([[.6, .3, .1], [.2, .5, .3], [.4, .1, .5]]).log()
    teacher = torch.tensor([[.1, .3, .6], [.5, .4, .1], [.2, .6, .2]]).log()
    sampled_ids = [0, 1, 2]
    top = [[[float(v), t] for t, v in enumerate(row)] for row in student]
    teacher_rows = [[[float(v), t] for t, v in enumerate(row)] for row in teacher]
    sample = Sample(tokens=[99, *sampled_ids], response_length=3, loss_mask=mask,
                    rollout_log_probs=[float(student[p,t]) for p,t in enumerate(sampled_ids)],
                    metadata={"opd_student_top_logprobs": top},
                    reward={"teacher": {"meta_info": {"input_token_ids_logprobs": [None, *teacher_rows]}}})
    args = Namespace(opd_target_mode="endpoint", opd_log_prob_top_k=3, opd_top_k_strategy="only-student",
                     opd_reward_weight_mode="student_p", opd_reward_centering="response", reward_key=None)
    return args, sample, student, teacher


def test_candidate_baseline_is_sampled_mean_before_weighting():
    args, sample, student, teacher = case([1, 0, 1])
    centered = _compute_topk_candidates(args, sample, sample.reward)
    baseline = ((teacher-student).diag()[[0, 2]]).mean()
    expected = student.exp() * (teacher-student-baseline)
    torch.testing.assert_close(centered["opd_candidate_rewards"], expected)
    assert sample.metadata["opd_centering_valid_tokens"] == 2
    assert abs(sample.metadata["opd_sampled_centered_reward_mean"]) < 1e-6
    args.opd_reward_centering = "none"
    raw = _compute_topk_candidates(args, sample, sample.reward)
    torch.testing.assert_close(raw["opd_candidate_rewards"]-centered["opd_candidate_rewards"], student.exp()*baseline)
    for key in ("opd_candidate_ids", "opd_candidate_old_log_probs", "opd_candidate_mask"):
        torch.testing.assert_close(raw[key], centered[key])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_full_support_centering_preserves_exact_reverse_kl_gradient(device):
    args, sample, student, teacher = case()
    candidates = _compute_topk_candidates(args, sample, sample.reward)
    logits = student.to(device).detach().requires_grad_(True)
    current = logits.log_softmax(-1)
    loss, _ = candidate_ppo_loss(current, candidates["opd_candidate_old_log_probs"].to(device),
                                 candidates["opd_candidate_rewards"].to(device), candidates["opd_candidate_mask"].to(device),
                                 eps_clip=.2, eps_clip_high=.2)
    actual = torch.autograd.grad(loss.mean(), logits, retain_graph=True)[0]
    expected = torch.autograd.grad((current.exp() * (current-teacher.to(device))).sum(-1).mean(), logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.norm() > 0


@pytest.mark.parametrize("mask", [None, [1,0,1], [0,0,0]])
def test_masked_centering_is_detached_and_never_whitened(mask):
    raw = torch.tensor([-4., 2., 8.], requires_grad=True)
    result, baseline = center_sampled_rewards(raw, mask)
    active = torch.ones(3, dtype=torch.bool) if mask is None else torch.tensor(mask).bool()
    expected_base = raw.detach()[active].mean() if active.any() else torch.tensor(0.)
    torch.testing.assert_close(baseline, expected_base)
    torch.testing.assert_close(result, torch.where(active, raw.detach()-expected_base, 0.))
    assert not result.requires_grad and not baseline.requires_grad


def test_padding_nan_empty_and_invalid_inputs():
    result, _ = center_sampled_rewards(torch.tensor([2., float("nan"), 4.]), [1,0,1])
    torch.testing.assert_close(result, torch.tensor([-1.,0.,1.]))
    result, baseline = center_sampled_rewards(torch.empty(0))
    assert result.numel() == 0 and baseline == 0
    with pytest.raises(ValueError, match="shape mismatch"):
        center_sampled_rewards(torch.ones(2), [1])
    with pytest.raises(ValueError, match="Non-finite"):
        center_sampled_rewards(torch.tensor([float("nan")]))


def test_shift_masks_and_explicit_off():
    anchor = [{0:-.2, 1:-1.5}, {0:-1.4, 1:-.3}, {0:-.8, 1:-.7}]
    student = torch.tensor([-.1,-.6,-1.2])
    # teacher=base isolates anchor; sampled ID is final support entry (1).
    centered, raw = _compute_shiftmopd_reverse_kl(anchor, [anchor], [anchor], student,
                                                 [[0,1]]*3, loss_mask=[1,0,1])
    torch.testing.assert_close(centered, torch.tensor([raw[0]-raw[[0,2]].mean(), 0., raw[2]-raw[[0,2]].mean()]))
    off, _ = _compute_shiftmopd_reverse_kl(anchor, [anchor], [anchor], student, [[0,1]]*3, center=False)
    torch.testing.assert_close(off, raw)


def test_candidate_postprocess_does_not_double_apply_scalar_opd():
    args, sample, _, _ = case()
    post_process_rewards(args, [sample])
    assert "opd_candidates" in sample.metadata
    advantages = [torch.zeros(3)]
    apply_opd_kl_to_advantages(args, {"opd_candidate_ids": [torch.arange(3)]}, advantages, [torch.zeros(3)])
    assert not advantages[0].any()


def test_shift_postprocess_matches_endpoint_sampled_baseline_and_logs():
    from miles.ray.rollout.metrics import _compute_shiftmopd_metrics

    args, endpoint_sample, student, teacher = case([1,0,1])
    post_process_rewards(args, [endpoint_sample])
    args, shift_sample, _, _ = case([1,0,1])
    teacher_response = shift_sample.reward["teacher"]
    args.opd_target_mode = "shiftmopd"
    # Anchor=base cancels, so the composite target is exactly this teacher.
    shift_sample.reward = dict(mode="shiftmopd", anchor=teacher_response,
                               teachers={"one": teacher_response}, bases={"one": teacher_response},
                               support_ids=[[1,2,0], [0,2,1], [0,1,2]])
    post_process_rewards(args, [shift_sample])
    expected, _ = center_sampled_rewards((teacher-student).diag(), [1,0,1])
    torch.testing.assert_close(-torch.tensor(shift_sample.opd_reverse_kl), expected)
    assert shift_sample.metadata["opd_centering_baseline"] == pytest.approx(endpoint_sample.metadata["opd_centering_baseline"])
    metrics = _compute_shiftmopd_metrics([endpoint_sample, shift_sample])
    assert metrics["opd/centering_enabled"] == 1
    assert metrics["opd/centering_valid_tokens"] == 4
    assert abs(metrics["opd/sampled_centered_reward_mean"]) < 1e-6


def test_sampled_only_endpoint_centers_once_with_mask():
    args = Namespace(opd_reward_centering="response", opd_target_mode="endpoint", opd_kl_coef=2.)
    data = {"teacher_log_probs": [torch.tensor([-2.,-4.,-8.])], "loss_masks": [[1,0,1]]}
    advantages = [torch.zeros(3)]
    apply_opd_kl_to_advantages(args, data, advantages, [torch.tensor([-1.,-1.,-1.])])
    torch.testing.assert_close(advantages[0], torch.tensor([6.,0.,-6.]))
    # Precomputed values must not be centered again at the consumer boundary.
    before = data["opd_reverse_kl"][0].clone()
    apply_opd_kl_to_advantages(args, data, [torch.zeros(3)], [torch.zeros(3)])
    torch.testing.assert_close(data["opd_reverse_kl"][0], before)


def test_postprocess_metrics_reach_tracking(monkeypatch):
    import miles.ray.rollout.metrics as metrics
    args, sample, _, _ = case()
    args.use_opd = True
    captured = []
    monkeypatch.setattr(metrics, "compute_rollout_step", lambda args, step: step)
    monkeypatch.setattr(metrics.tracking, "log", lambda args, data, **kw: captured.append(data))
    post_process_rewards(args, [sample])
    metrics.log_opd_reward_metrics(9, args, [sample])
    assert captured[0]["rollout/step"] == 9
    assert captured[0]["rollout/opd/centering_enabled"] == 1
    assert abs(captured[0]["rollout/opd/sampled_centered_reward_mean"]) < 1e-6


def test_legacy_centering_defaults():
    assert opd_centering_mode(Namespace(opd_target_mode="shiftmopd")) == "response"
    assert opd_centering_mode(Namespace()) == "none"
    with pytest.raises(ValueError, match="Unknown"):
        opd_centering_mode(Namespace(opd_reward_centering="whiten"))
