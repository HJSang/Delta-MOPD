"""Compact deterministic prefix panels and eval traces for checkpoint diagnostics.

Logging hooks return False so Miles' ordinary W&B metrics still run. No scoring
requests, GPU allocation or model updates happen here. Panels intentionally use
the first valid sample PER DOMAIN, not the paper's first training micro-batch;
this sampling policy is recorded so the results cannot be conflated.
"""

import json
from pathlib import Path


def diagnostic_panel(samples, *, max_positions: int = 128) -> list[dict]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive.")
    records = {}
    for sample in samples:
        metadata = sample.metadata or {}
        domain = metadata.get("domain", metadata.get("opd_teacher"))
        if not domain or domain in records or sample.remove_sample or not sample.response_length:
            continue
        prompt_length = len(sample.tokens) - sample.response_length
        mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
        if prompt_length < 1 or len(mask) != sample.response_length:
            raise ValueError("Diagnostic samples require a prompt and a response-aligned loss mask.")
        positions = [prompt_length + i for i, valid in enumerate(mask) if valid][:max_positions]
        if not positions:
            continue
        records[domain] = {
            "domain": domain,
            "sample_index": sample.index,
            "weight_versions": [call.to_dicts() for call in getattr(sample, "weight_versions", [])],
            "tokens": list(sample.tokens[: positions[-1] + 1]),
            "response_positions": positions,
        }
    return list(records.values())


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence if an output directory is accidentally reused.
    with path.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False)


def save_panel(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if rollout_id != 0 and (rollout_id + 1) % (getattr(args, "eval_interval", None) or 20) != 0 and (rollout_id + 1) != getattr(args, "num_rollout", None):
        return False
    _write_new(
        Path(args.save).parent / "paper_diagnostics" / f"panel_{rollout_id}.json",
        {
            "rollout_id": rollout_id,
            "sampling_policy": "first_valid_sample_per_domain_first_128_valid_response_positions",
            "student_temperature": args.rollout_temperature,
            "target_mode": args.opd_target_mode,
            "support_k": args.opd_log_prob_top_k,
            "student_initialization": getattr(args, "hf_checkpoint", None),
            "records": diagnostic_panel(samples),
        },
    )
    return False


def save_eval_traces(rollout_id, args, data, extra_metrics=None):
    records = []
    for dataset, info in data.items():
        for sample in info.get("samples", []):
            records.append(
                {
                    "dataset": dataset,
                    "sample_index": sample.index,
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "tokens": list(sample.tokens),
                    "response_length": sample.response_length,
                    "label": sample.label,
                    "metadata": sample.metadata,
                    "reward": sample.reward,
                }
            )
    _write_new(
        Path(args.save).parent / "eval_traces" / f"eval_{rollout_id}.json",
        {"rollout_id": rollout_id, "records": records},
    )
    return False
