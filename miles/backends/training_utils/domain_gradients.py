"""Exact single-training-rank domain diagnostics; never used as optimizer gradients.

Enable with MILES_DOMAIN_GRAD_METRICS=1 in both rollout and actor workers.
Exact gradients are measured on evaluation-scheduled training rollouts only
(including final and epoch-boundary evaluations). No replay runs if eval is off.
Domain backward replays retain all batch denominators, masks and advantages.
Snapshots live on CPU, costing four bytes per trainable parameter per domain.
"""

import math
import os
import re
from collections.abc import Callable, Sequence
from time import perf_counter

import torch


def domain_gradients_enabled() -> bool:
    return os.environ.get("MILES_DOMAIN_GRAD_METRICS", "0") == "1"


def sample_domain(metadata: dict | None, teacher_key: str = "opd_teacher") -> str:
    metadata = metadata or {}
    value = metadata.get("domain", metadata.get(teacher_key))
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Domain gradients require metadata.domain or a nonempty teacher-domain label.")
    return value


def domain_reducer(
    reducer: Callable[[torch.Tensor], torch.Tensor],
    domains: Sequence[str],
    response_lengths: Sequence[int],
    domain: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gate numerators only. The original reducer owns every denominator."""
    if len(domains) != len(response_lengths):
        raise ValueError("Domain labels must align with response lengths.")

    def reduce_selected(values: torch.Tensor) -> torch.Tensor:
        weights = values.new_tensor([label == domain for label in domains])
        lengths = torch.tensor(response_lengths, device=values.device)
        gate = torch.repeat_interleave(weights, lengths)
        if values.ndim != 1 or values.numel() != gate.numel():
            raise ValueError("Domain reducer expects the full response-token layout (CP=1).")
        return reducer(values * gate)

    return reduce_selected


def validate_domain_diagnostics(args, parallel_state, model, optimizer) -> None:
    """Fail closed for paths with cross-sample losses or untracked replay state."""
    groups = ("tp", "pp", "cp", "ep", "etp", "intra_dp", "indep_dp")
    if len(model) != 1 or any(getattr(parallel_state, name).size != 1 for name in groups):
        raise ValueError("Domain gradient diagnostics currently require a single training rank (TP/PP/CP/DP/EP=1).")
    if getattr(args, "loss_type", None) != "policy_loss":
        raise ValueError("Domain gradient diagnostics require policy_loss.")
    forbidden = (
        "fp16",
        "fp8",
        "num_experts",
        "enable_mtp_training",
        "enable_witness",
        "multi_lora_n_adapters",
        "use_tis",
        "use_opsm",
        "get_mismatch_metrics",
        "custom_pg_loss_reducer_function_path",
        "use_rollout_routing_replay",
        "custom_megatron_before_train_step_hook_path",
        "dump_details",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "use_kl_loss",
        "dumper_enable",
    )
    active = [name for name in forbidden if getattr(args, name, None)]
    if active:
        raise ValueError(f"Domain gradient replay is not validated with: {', '.join(active)}.")
    if getattr(args, "attention_dropout", 0) or getattr(args, "hidden_dropout", 0):
        raise ValueError("Domain gradient replay requires zero dropout.")
    if optimizer is not None and float(optimizer.get_loss_scale()) != 1.0:
        raise ValueError("Domain gradient replay requires unit optimizer loss scale.")


def trainable_parameters(model: Sequence[torch.nn.Module]) -> list[torch.nn.Parameter]:
    # A tied embedding appears only once in the flattened parameter vector.
    return list(dict.fromkeys(p for module in model for p in module.parameters() if p.requires_grad))


def _cpu_gradient(parameter: torch.nn.Parameter) -> torch.Tensor:
    gradient = getattr(parameter, "main_grad", None)
    if gradient is None:
        gradient = parameter.grad
    if gradient is None:
        return torch.zeros(parameter.shape, dtype=torch.float32)
    return gradient.detach().to(device="cpu", dtype=torch.float32, copy=True)


def collect_domain_gradients(
    *,
    iterator,
    num_microbatches: int,
    model: Sequence[torch.nn.Module],
    run_backward: Callable[[str], object],
    zero_grad: Callable[[], None],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, float]]:
    """Replay each present domain, restoring the iterator, RNG and model buffers.

    The caller subsequently executes the original full-batch backward pass.
    No diagnostic gradients reach optimizer.step, even on a failed replay.
    """
    start = perf_counter()
    offset = iterator.offset
    data = iterator.rollout_data
    if data.get("loss_fn") is not None:
        raise ValueError("Domain gradients do not support custom/Tinker batch losses.")
    rows = []
    try:
        for _ in range(num_microbatches):
            rows.append(iterator.get_next(["gradient_domains", "gradient_prompt_ids", "loss_masks"]))
    finally:
        iterator.offset = offset
    if any(row["gradient_domains"] is None or row["gradient_prompt_ids"] is None for row in rows):
        raise ValueError("Domain labels missing: enable MILES_DOMAIN_GRAD_METRICS in rollout workers too.")
    labels = [label for row in rows for label in row["gradient_domains"]]
    parameters = trainable_parameters(model)
    devices = sorted({p.device.index for p in parameters if p.device.type == "cuda"})
    # Megatron DDP shadows nn.Module.buffers with its gradient-buffer list.
    # Use the base-class traversal for registered model buffers, not DDP's
    # communication/gradient storage (which zero_grad owns).
    buffers = [(buffer, buffer.detach().clone()) for module in model for buffer in torch.nn.Module.buffers(module)]
    metrics = {}
    snapshots = {}
    try:
        for domain in sorted(set(labels) | {"math", "code", "if"}):
            prompts = set()
            count = tokens = 0
            for row in rows:
                for label, prompt, mask in zip(row["gradient_domains"], row["gradient_prompt_ids"], row["loss_masks"], strict=True):
                    if label == domain:
                        count += 1
                        prompts.add(prompt)
                        tokens += float(torch.as_tensor(mask).sum())
            prefix = f"grad/domain/{domain}"
            metrics.update({f"{prefix}/sample_count": count, f"{prefix}/prompt_count": len(prompts), f"{prefix}/valid_tokens": tokens, f"{prefix}/present": int(count > 0)})
            if not count:
                metrics[f"{prefix}/cos_with_batch_defined"] = 0
                metrics[f"{prefix}/signed_share_defined"] = 0
                continue
            iterator.offset = offset
            zero_grad()
            with torch.no_grad():
                for buffer, original in buffers:
                    buffer.copy_(original)
            # Reset torch RNG after each replay; zero dropout is required for Megatron's separate tracker.
            with torch.random.fork_rng(devices=devices):
                run_backward(domain)
            snapshots[domain] = [_cpu_gradient(parameter) for parameter in parameters]
    finally:
        iterator.offset = offset
        zero_grad()
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
    metrics["grad/diagnostics/replay_seconds"] = perf_counter() - start
    metrics["grad/diagnostics/extra_backward_passes"] = len(snapshots)
    metrics["grad/diagnostics/snapshot_bytes"] = sum(x.numel() * x.element_size() for g in snapshots.values() for x in g)
    return snapshots, metrics


def domain_gradient_metrics(
    model: Sequence[torch.nn.Module],
    snapshots: dict[str, list[torch.Tensor]],
    *,
    chunk_size: int = 262144,
) -> dict[str, float]:
    """Compare exact full parameter vectors before clipping, with bounded scratch.

    Values are FP32 snapshots; CPU FP64 chunk reductions avoid norm/dot overflow.
    Undefined angles/shares are omitted and accompanied by *_defined=0.
    """
    start = perf_counter()
    parameters = trainable_parameters(model)
    domains = sorted(snapshots)
    if any(len(snapshots[d]) != len(parameters) for d in domains):
        raise ValueError("Gradient snapshots do not match the trainable parameter list.")
    gram = torch.zeros((len(domains) + 1, len(domains) + 1), dtype=torch.float64)
    residual_sq = max_error = 0.0
    for index, parameter in enumerate(parameters):
        batch = _cpu_gradient(parameter).reshape(-1)
        vectors = [batch, *(snapshots[d][index].reshape(-1) for d in domains)]
        if any(vector.numel() != batch.numel() for vector in vectors):
            raise ValueError("Gradient snapshot shape mismatch.")
        for offset in range(0, batch.numel(), chunk_size):
            values = torch.stack([v[offset : offset + chunk_size] for v in vectors]).double()
            gram += values @ values.T
            error = values[0] - values[1:].sum(dim=0)
            residual_sq += float(error.square().sum())
            max_error = max(max_error, float(error.abs().max()))
    finite = bool(torch.isfinite(gram).all()) and math.isfinite(residual_sq)
    metrics = {"grad/diagnostics/finite": int(finite)}
    if not finite:
        metrics["grad/diagnostics/measurement_seconds"] = perf_counter() - start
        return metrics
    norms = gram.diag().clamp_min(0).sqrt().tolist()
    metrics["grad/batch/norm"] = norms[0]
    domain_norm_sum = sum(norms[1:])
    metrics["grad/domain_geometry/cancellation_defined"] = int(domain_norm_sum > 0)
    if domain_norm_sum > 0:
        aggregate_norm = math.sqrt(max(0.0, float(gram[1:, 1:].sum())))
        metrics["grad/domain_geometry/aggregate_norm"] = aggregate_norm
        metrics["grad/domain_geometry/cancellation"] = 1 - aggregate_norm / (domain_norm_sum + 1e-8)
    metrics["grad/domain_geometry/imbalance_defined"] = int(bool(domains) and min(norms[1:]) > 0)
    if domains and min(norms[1:]) > 0:
        metrics["grad/domain_geometry/imbalance"] = max(norms[1:]) / min(norms[1:])
    metrics["grad/diagnostics/reconstruction_abs_error"] = math.sqrt(residual_sq)
    metrics["grad/diagnostics/reconstruction_max_abs_error"] = max_error
    metrics["grad/diagnostics/reconstruction_relative_error_defined"] = int(norms[0] > 0)
    if norms[0] > 0:
        metrics["grad/diagnostics/reconstruction_relative_error"] = math.sqrt(residual_sq) / norms[0]
    for i, domain in enumerate(domains, start=1):
        prefix = f"grad/domain/{domain}"
        metrics[f"{prefix}/norm"] = norms[i]
        metrics[f"{prefix}/cos_with_batch_defined"] = int(norms[0] > 0 and norms[i] > 0)
        metrics[f"{prefix}/signed_share_defined"] = int(norms[0] > 0)
        if norms[0] > 0:
            metrics[f"{prefix}/signed_share"] = float(gram[0, i]) / (norms[0] ** 2)
            if norms[i] > 0:
                metrics[f"{prefix}/cos_with_batch"] = max(-1.0, min(1.0, float(gram[0, i]) / (norms[0] * norms[i])))
        for j in range(i + 1, len(domains) + 1):
            pair = f"grad/domain_pair/{domain}_{domains[j - 1]}"
            metrics[f"{pair}/cos_defined"] = int(norms[i] > 0 and norms[j] > 0)
            if norms[i] > 0 and norms[j] > 0:
                metrics[f"{pair}/cos"] = max(-1.0, min(1.0, float(gram[i, j]) / (norms[i] * norms[j])))
    metrics["grad/diagnostics/measurement_seconds"] = perf_counter() - start
    return metrics
