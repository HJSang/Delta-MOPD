# Gradient and distribution diagnostics

Reference: [arXiv:2610.10460](https://arxiv.org/abs/2610.10460), especially the
diagnostic definitions. Training-domain gradients, teacher-component gradients,
and logit geometry are distinct quantities and must not be conflated.

## Teacher-component gradients

For the same sampled response tokens and student prefixes:

```text
R_delta = log T(y) - log B(y)
R_base  = log B(y) - log student(y)
g_delta = grad[-mean(stopgrad(R_delta) * log student(y))]
g_base  = grad[-mean(stopgrad(R_base)  * log student(y))]
Gamma_base = norm(g_base) / (norm(g_delta) + epsilon)
```

These diagnostic rewards are uncentered and unwhitened, as in the diagnostic
definition; that does not change training's centered objective. Norms cover all
trainable student parameters before clipping, counting tied parameters once.
`autograd.grad` leaves optimizer gradients/state untouched. Metrics include
shift/base/endpoint norms, Gamma_base, base-shift cosine, and definedness flags.

## Distribution distances and logit geometry

`full_vocab_kl_metrics` measures full effective-vocabulary target-to-student and
target-to-anchor **forward KL**, and student-to-target reverse KL separately.
It never substitutes top-K conditional KL or average per-teacher KL for full
composite KL. Targets include the shift composite, endpoint composite, and
individual endpoint teachers on the same supplied prefixes.

`full_vocab_geometry_metrics` vocabulary-centers contributions, then reports
norms, concatenated-position pairwise cosines, aggregate cancellation, and sign
conflict on each teacher's top-16 absolute coordinates. Overlapping selected
coordinates are counted separately for each teacher. Diagnostic top-16 is not
the training partition's support size or the generation sampling filter.

## Offline checkpoint runner

```bash
PYTHONPATH=. python experiments/measure_shiftmopd_checkpoint.py \
  --spec checkpoint_diagnostic.json --output diagnostic.json \
  --device cuda:0 --dtype bfloat16
```

The JSON spec contains `student`, `anchor`, `teachers` and `bases` maps of local
HF checkpoint directories, plus `panel`, `checkpoint_step`, and `source_run_id`.
An optional `student_temperature` overrides the saved panel temperature.
Export Megatron checkpoints and merge LoRA-only checkpoints first. Supported
checkpoint families are Qwen2 and Qwen3 with exactly aligned tokenizer maps.

The panel hook in `miles.rollout.paper_diagnostic_hooks` records exact token IDs
and up to 128 valid response positions per domain, with sampling/weight-version
metadata. Its evaluation hook saves responses, tokens, labels, and verifier
outputs. Configure hooks explicitly; merely importing them does not schedule
evaluation or save traces. Keep traces out of Git.

The runner loads frozen models sequentially, reuses shared bases, and applies
the output head only to selected causal predictor states. Measurements record
precision, temperature, vocabulary, positions, checkpoint provenance, and time.
Use an available GPU; this utility does not allocate hardware or stop training.

**Scope limitation:** offline per-domain panels are not the paper's live
first-microbatch/first-128-valid-position/every-20-update protocol. Matched
comparisons require the same panel, precision, temperature, and checkpoint
state. Report the measurement protocol alongside results.

## Domain gradients

Inherited domain-gradient replay can report each domain's norm, cosine with the
batch gradient, signed share, pairwise cosine, cancellation, and reconstruction
error. It adds backward passes and has single-rank/replay constraints. These
measure the configured training loss, not the paper's teacher/base decomposition.
Do not interpret a large domain norm as a large inherited-base component.

CPU tests cover KL direction, target identities, causal alignment, gradient
decomposition, unchanged optimizer updates, and tiny Qwen2/Qwen3 checkpoint
save/load/measurement. GPU and distributed integration remain release gates.
