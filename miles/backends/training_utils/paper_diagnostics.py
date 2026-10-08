"""ShiftMOPD draft, Diagnostic Definitions: full-vocabulary KL and g_B/g_delta.

These are checkpoint measurements, not losses. Inputs must be complete, aligned
effective-vocabulary logits on identical prefixes, NOT top-K scoring responses.
The sampled score-function components deliberately use detached, uncentered
rewards, as defined by eq:gradient-components, independently of the training loss.
"""

import math
from collections.abc import Mapping, Sequence

import torch

from miles.utils.paper_opd import compose_target_scores


def _log_probs(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != vocab_size or logits.shape[0] == 0:
        raise ValueError("Expected nonempty [positions, effective_vocab_size] full-vocabulary logits.")
    if not torch.isfinite(logits).all():
        raise ValueError("Diagnostic logits must be finite on the effective vocabulary.")
    return logits.float().log_softmax(dim=-1)


def _aligned_log_probs(
    student_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    teacher_logits: Mapping[str, torch.Tensor],
    base_logits: Mapping[str, torch.Tensor],
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not teacher_logits or teacher_logits.keys() != base_logits.keys():
        raise ValueError("Every named teacher needs its exact base.")
    tensors = [anchor_logits, *teacher_logits.values(), *base_logits.values()]
    if any(t.shape != student_logits.shape or t.device != student_logits.device for t in tensors):
        raise ValueError("All models must score the same positions and vocabulary on the same device.")
    return (
        _log_probs(student_logits, vocab_size),
        _log_probs(anchor_logits, vocab_size),
        {name: _log_probs(value, vocab_size) for name, value in teacher_logits.items()},
        {name: _log_probs(value, vocab_size) for name, value in base_logits.items()},
    )


def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    # FP64 accumulation avoids cancellation near identical distributions.
    return float((log_p.double().exp() * (log_p.double() - log_q.double())).sum(-1).mean())


@torch.no_grad()
def full_vocab_kl_metrics(
    *,
    student_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    teacher_logits: Mapping[str, torch.Tensor],
    base_logits: Mapping[str, torch.Tensor],
    vocab_size: int,
) -> dict[str, float]:
    """Evaluate Shift, same-state endpoint composition, and each routed endpoint.

    All variants use the SAME supplied prefixes. An endpoint/<teacher> result is
    the routed-MOPD reference only on that teacher's domain. No averaging of
    individual teacher KLs is used as a substitute for the composite KL.
    """
    student, anchor, teachers, bases = _aligned_log_probs(student_logits, anchor_logits, teacher_logits, base_logits, vocab_size)
    selected_teachers = list(teachers.values())
    selected_bases = [bases[name] for name in teachers]
    shift = compose_target_scores(anchor, selected_teachers, selected_bases, mode="shiftmopd")
    endpoint = compose_target_scores(anchor, selected_teachers, [], mode="endpoint")
    targets = {
        "shiftmopd": shift.log_softmax(-1),
        "endpoint_composite": endpoint.log_softmax(-1),
        **{f"endpoint/{name}": teacher for name, teacher in teachers.items()},
    }
    metrics = {"paper/full_vocab/positions": float(student.shape[0]), "paper/full_vocab/vocab_size": float(vocab_size)}
    for name, target in targets.items():
        prefix = f"paper/full_vocab/{name}"
        metrics[f"{prefix}/target_to_student_kl"] = _kl(target, student)
        metrics[f"{prefix}/target_to_anchor_kl"] = _kl(target, anchor)
        metrics[f"{prefix}/student_to_target_kl"] = _kl(student, target)
        metrics[f"{prefix}/target_entropy"] = float(-(target.double().exp() * target.double()).sum(-1).mean())
    return metrics


@torch.no_grad()
def full_vocab_geometry_metrics(*, anchor_logits, teacher_logits, base_logits, vocab_size, top_k=16):
    """Appendix B concatenated-position geometry, NOT per-response averaging.

    Inputs contain only the selected VALID diagnostic positions. Sign conflict
    counts each teacher's top-|shift| coordinates, retaining overlapping IDs
    twice as in the draft. `top_k` here is diagnostic, not training support K.
    """
    _aligned_log_probs(anchor_logits, anchor_logits, teacher_logits, base_logits, vocab_size)
    if top_k <= 0:
        raise ValueError("Diagnostic top-k must be positive.")
    metrics = {"paper/geometry/positions": float(anchor_logits.shape[0]),
               "paper/geometry/vocab_size": float(vocab_size)}
    for mode in ("endpoint", "shiftmopd"):
        terms = {}
        for name, teacher in teacher_logits.items():
            reference = anchor_logits if mode == "endpoint" else base_logits[name]
            difference = teacher.double() - reference.double()
            terms[name] = difference - difference.mean(-1, keepdim=True)
        prefix = f"paper/geometry/{mode}"
        norms = {name: value.norm() for name, value in terms.items()}
        aggregate = sum(terms.values()).norm()
        metrics[f"{prefix}/aggregate_norm"] = float(aggregate)
        metrics[f"{prefix}/cancellation"] = float(1 - aggregate / (sum(norms.values()) + 1e-8))
        metrics.update({f"{prefix}/{name}/norm": float(norm) for name, norm in norms.items()})
        names = list(terms)
        conflicts = count = 0
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                a, b = terms[name], terms[other]
                product = norms[name] * norms[other]
                pair = f"{prefix}/{name}_vs_{other}"
                metrics[f"{pair}/cos_defined"] = float(product > 0)
                if product > 0:
                    metrics[f"{pair}/cos"] = float((a * b).sum() / product)
                k = min(top_k, vocab_size)
                ids = torch.cat([a.abs().topk(k, -1).indices, b.abs().topk(k, -1).indices], -1)
                conflicts += int((a.gather(-1, ids) * b.gather(-1, ids) < 0).sum())
                count += ids.numel()
        metrics[f"{prefix}/sign_conflict_defined"] = float(count > 0)
        if count:
            metrics[f"{prefix}/sign_conflict"] = conflicts / count
    return metrics


def _gradient_snapshot(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    # autograd.grad does not write .grad, main_grad, or optimizer state.
    return [None if grad is None else grad.detach().to(device="cpu", dtype=torch.float32, copy=True) for grad in torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)]


def _component_geometry(shift: Sequence[torch.Tensor | None], base: Sequence[torch.Tensor | None]) -> dict[str, float]:
    shift_sq = base_sq = dot = 0.0
    for x, y in zip(shift, base, strict=True):
        if x is None and y is None:
            continue
        count = x.numel() if x is not None else y.numel()
        if x is not None and y is not None and x.shape != y.shape:
            raise ValueError("Gradient shapes differ.")
        for offset in range(0, count, 262144):
            a = x.reshape(-1)[offset : offset + 262144].double() if x is not None else None
            b = y.reshape(-1)[offset : offset + 262144].double() if y is not None else None
            shift_sq += float(a.square().sum()) if a is not None else 0.0
            base_sq += float(b.square().sum()) if b is not None else 0.0
            dot += float((a * b).sum()) if a is not None and b is not None else 0.0
    if not all(math.isfinite(v) for v in (shift_sq, base_sq, dot)):
        raise ValueError("Non-finite component gradient.")
    shift_norm, base_norm = math.sqrt(shift_sq), math.sqrt(base_sq)
    metrics = {
        "shift_norm": shift_norm,
        "base_norm": base_norm,
        "endpoint_norm": math.sqrt(max(0.0, shift_sq + base_sq + 2 * dot)),
        "gamma_base": base_norm / (shift_norm + 1e-8),
        "base_shift_cos_defined": float(shift_norm > 0 and base_norm > 0),
    }
    if shift_norm > 0 and base_norm > 0:
        metrics["base_shift_cos"] = max(-1.0, min(1.0, dot / (shift_norm * base_norm)))
    return metrics


def component_gradient_metrics(
    *,
    student_logits: torch.Tensor,
    teacher_logits: Mapping[str, torch.Tensor],
    base_logits: Mapping[str, torch.Tensor],
    sampled_ids: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    vocab_size: int,
) -> dict[str, float]:
    """All-parameter, pre-clipping score-function gradients on N valid positions.

    g_delta = grad(-mean(stopgrad(log T[y]-log B[y]) * log student[y])).
    g_B     = grad(-mean(stopgrad(log B[y]-log student[y]) * log student[y])).
    g_endpoint = g_delta + g_B. These are NOT PPO or domain gradients.
    The caller supplies already-selected valid positions (paper default N<=128).
    """
    student, _, teachers, bases = _aligned_log_probs(student_logits, student_logits.detach(), teacher_logits, base_logits, vocab_size)
    if sampled_ids.shape != (student.shape[0],) or sampled_ids.dtype != torch.long:
        raise ValueError("Sampled IDs must be one int64 token per valid response position.")
    if (sampled_ids < 0).any() or (sampled_ids >= vocab_size).any():
        raise ValueError("Sampled IDs are outside the effective vocabulary.")
    parameters = list(dict.fromkeys(p for p in parameters if p.requires_grad))
    if not parameters or not student.requires_grad:
        raise ValueError("A live student graph and trainable parameters are required.")
    sampled_ids = sampled_ids.to(student.device).unsqueeze(-1)
    log_student = student.gather(-1, sampled_ids).squeeze(-1)
    metrics = {"paper/gradient/positions": float(student.shape[0])}
    for name, teacher in teachers.items():
        log_base = bases[name].gather(-1, sampled_ids).squeeze(-1)
        delta = (teacher.gather(-1, sampled_ids).squeeze(-1) - log_base).detach()
        inherited = (log_base - log_student).detach()
        shift_grad = _gradient_snapshot(-(delta * log_student).mean(), parameters)
        base_grad = _gradient_snapshot(-(inherited * log_student).mean(), parameters)
        metrics.update({f"paper/gradient/{name}/{key}": value for key, value in _component_geometry(shift_grad, base_grad).items()})
        del shift_grad, base_grad
    return metrics
