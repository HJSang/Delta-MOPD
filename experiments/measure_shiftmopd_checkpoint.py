"""Offline paper diagnostics for exported Qwen2/Qwen3 HF checkpoints and saved panels.

Run on an explicitly available GPU, NOT beside the live trainer by default:
  PYTHONPATH=. python experiments/measure_shiftmopd_checkpoint.py \
    --spec checkpoint_diagnostic.json --output diagnostic.json --device cuda:0

Spec: student, anchor, teachers {name: HF directory}, bases {name: HF directory},
panel (JSON from save_panel), checkpoint_step, source_run_id, and optionally
student_temperature (otherwise panel rollout temperature). Paths must be local,
merged HF checkpoints; Megatron distributed checkpoints and LoRA-only directories
must be exported/merged first. This tool never updates the checkpoint or trainer.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from miles.backends.training_utils.paper_diagnostics import (
    component_gradient_metrics, full_vocab_geometry_metrics, full_vocab_kl_metrics,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = None

try:
    import wandb
except ImportError:
    wandb = None


def _validate_record(record, vocab_size):
    tokens, positions = record["tokens"], record["response_positions"]
    if not positions or len(positions) > 128 or positions != sorted(set(positions)):
        raise ValueError("Panel must contain 1..128 unique, increasing response positions.")
    if any(not isinstance(p, int) or p < 1 or p >= len(tokens) for p in positions):
        raise ValueError("Response positions must have a preceding causal token.")
    if any(not isinstance(token, int) or token < 0 or token >= vocab_size for token in tokens):
        raise ValueError("Panel token outside the common effective vocabulary.")


def _selected_logits(model, record, vocab_size, device):
    """Apply the output head only at selected prefix states, not the whole prompt.

    Response token at index p is predicted by hidden state p-1. The sampled
    token itself and future suffix are never fed into that prediction.
    """
    _validate_record(record, vocab_size)
    last = record["response_positions"][-1]
    ids = torch.tensor([record["tokens"][:last]], device=device, dtype=torch.long)
    positions = torch.tensor(record["response_positions"], device=device) - 1
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
    return model.get_output_embeddings()(hidden)[..., :vocab_size].float()


def _load_model(path, device, dtype):
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    if model.config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("This checkpoint runner supports Qwen2/Qwen3; use the tensor API for other architectures.")
    return model


def _frozen_scores(paths, records, vocab_size, device, dtype):
    scores = {}
    # Shared teacher bases are scored once per unique checkpoint, not per teacher.
    for path in sorted(set(paths)):
        model = _load_model(path, device, dtype)
        with torch.no_grad():
            scores[path] = [_selected_logits(model, row, vocab_size, device).cpu() for row in records]
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return scores


def _measure_student(spec, records, frozen, vocab_size, args, temperature):
    model = _load_model(spec["student"], args.device, args.dtype)
    results = []
    for index, record in enumerate(records):
        started = perf_counter()
        student = _selected_logits(model, record, vocab_size, args.device) / temperature
        teachers = {name: frozen[path][index] for name, path in spec["teachers"].items()}
        bases = {name: frozen[path][index] for name, path in spec["bases"].items()}
        metrics = full_vocab_kl_metrics(
            student_logits=student.detach().cpu(),
            anchor_logits=frozen[spec["anchor"]][index],
            teacher_logits=teachers,
            base_logits=bases,
            vocab_size=vocab_size,
        )
        metrics.update(
            full_vocab_geometry_metrics(
                anchor_logits=frozen[spec["anchor"]][index], teacher_logits=teachers,
                base_logits=bases, vocab_size=vocab_size,
            )
        )
        metrics.update(
            component_gradient_metrics(
                student_logits=student,
                teacher_logits={name: value.to(args.device) for name, value in teachers.items()},
                base_logits={name: value.to(args.device) for name, value in bases.items()},
                sampled_ids=torch.tensor([record["tokens"][p] for p in record["response_positions"]], dtype=torch.long),
                parameters=list(model.parameters()),
                vocab_size=vocab_size,
            )
        )
        metrics["paper/measurement_seconds"] = perf_counter() - started
        assert all(p.grad is None for p in model.parameters())
        results.append({"domain": record["domain"], "sample_index": record["sample_index"], "metrics": metrics})
        del student
    return results


def _validate_spec(spec, panel):
    if not spec["teachers"] or spec["teachers"].keys() != spec["bases"].keys():
        raise ValueError("Teacher and base names must match.")
    if "checkpoint_step" not in spec or not spec.get("source_run_id"):
        raise ValueError("Record checkpoint_step and source_run_id to avoid ambiguous comparisons.")
    if not panel["records"]:
        raise ValueError("No valid diagnostic prefix records.")
    # All IDs must mean the same tokens, including added special tokens.
    paths = {spec["student"], spec["anchor"], *spec["teachers"].values(), *spec["bases"].values()}
    vocabulary = None
    for path in sorted(paths):
        current = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False).get_vocab()
        if vocabulary is None:
            vocabulary = current
        elif current != vocabulary:
            raise ValueError(f"Tokenizer mapping mismatch for {path}.")
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("Effective vocabulary must be contiguous and uniquely mapped; explicit projection is required.")
    return len(vocabulary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--wandb-project", help="Optional dedicated paper-<source_run_id> diagnostic run")
    args = parser.parse_args()
    if AutoTokenizer is None:
        parser.error("Install transformers for the checkpoint runner; tensor diagnostic tests only require torch.")
    if args.wandb_project and wandb is None:
        parser.error("Install wandb to publish the diagnostic results.")
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve earlier evidence.")
    spec = json.loads(args.spec.read_text())
    panel = json.loads(Path(spec["panel"]).read_text())
    vocab_size = _validate_spec(spec, panel)
    temperature = float(spec.get("student_temperature", panel["student_temperature"]))
    if not 0 < temperature < float("inf"):
        parser.error("Student temperature must be finite and positive.")
    args.dtype = getattr(torch, args.dtype)
    started = perf_counter()
    frozen = _frozen_scores(
        [spec["anchor"], *spec["teachers"].values(), *spec["bases"].values()],
        panel["records"],
        vocab_size,
        args.device,
        args.dtype,
    )
    results = _measure_student(spec, panel["records"], frozen, vocab_size, args, temperature)
    output = {
        "spec": spec,
        "sampling_policy": panel["sampling_policy"],
        "panel_rollout_id": panel["rollout_id"],
        "student_temperature": temperature,
        "frozen_temperature": 1.0,
        "dtype": str(args.dtype),
        "effective_vocab_size": vocab_size,
        "total_seconds": perf_counter() - started,
        "results": results,
        "scope": "offline HF checkpoint; exact full effective vocabulary; not live Megatron PPO gradients",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    if args.wandb_project:
        with wandb.init(
            project=args.wandb_project,
            id=f"paper-{spec['source_run_id']}",
            resume="allow",
            group=spec["source_run_id"],
            job_type="paper-diagnostics",
            config=spec,
            allow_val_change=True,
        ) as run:
            run.define_metric("paper/checkpoint_step")
            run.define_metric("paper/*", step_metric="paper/checkpoint_step")
            run.log(
                {
                    "paper/checkpoint_step": spec["checkpoint_step"],
                    **{f"{key}/domain/{row['domain']}": value for row in results for key, value in row["metrics"].items()},
                }
            )
    print(f"Saved {len(results)} domain panels to {args.output}; trainer and checkpoints unchanged.")


if __name__ == "__main__":
    main()
