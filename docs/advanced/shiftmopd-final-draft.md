# Final-draft Δ-MOPD contract

Source: [arXiv:2610.10460v1](https://arxiv.org/pdf/2610.10460), October 7, 2026,
especially Eq. 1–2 and Appendix A–B, D, F. This guide describes the explicit
`--opd-objective paper` path. It does **not** relabel the historical Qwen3
math/code/IF results as reproductions of this draft.

## One comparison, two independent choices

Set `--opd-teacher-selection all` for common-domain composition, or `routed`
for one metadata-selected teacher per prompt. Use the **same selection rule**
in both arms. Change only `--opd-target-mode endpoint|shiftmopd`:

```
endpoint:  z = z_A + sum_selected(z_T - z_A)
Δ-MOPD:    z = z_A + sum_selected(z_T - z_B)
```

A is the frozen initial student, not its moving training weights. B is each
teacher's exact pre-posttraining precursor. A single endpoint teacher reduces
to that teacher. If every B=A, the targets are identical. Frozen models score
the same student-generated token prefixes; no teacher response generation or
decode/retokenize round trip is introduced. Standalone and multi-LoRA scorer
routes are deduplicated by **(URL, adapter)**, not URL alone.

## Shared loss and baseline

Both arms use `R = log q(y) - log p_old_student(y)` at the **sampled token**.
Subtract one valid-token-weighted mean across the complete optimization batch,
before DP splitting or microbatch accumulation. Removed samples and masked
tokens contribute neither to the sum nor count. Empty batches fail closed.
The minimization loss is `-stopgrad(R - batch_mean) * log p_student(y)`, averaged
over valid tokens per response, then over responses (Eq. 2).

There is no candidate-probability weighting, candidate PPO clipping, reward
whitening, or added task reward in this path. A sum of candidate log-ratios is
not a replacement. `opd_reverse_kl` is retained only as the existing transport
field: it carries the **negative centered sampled reward**. The paper loss
consumes it directly and does not use a PPO ratio.

## Vocabulary and partition

The draft uses 151,665 effective tokens. Padding is masked in student rollout
sampling and excluded from the training softmax and composite partition.
`--opd-paper-model-manifest` is mandatory; local tokenizer maps must agree
exactly and the frozen anchor must identify the student initialization.
Scorer deployment must separately verify the manifest's weight revisions and
registered adapter identities; tokenizer validation cannot prove remote weights.

- `--opd-paper-partition full --opd-log-prob-top-k 0`: exact effective-vocabulary
  normalization, using vocabulary-streamed selected-ID scoring. This is a
  **slow correctness reference**, not an optimized full-run serving backend.
- `--opd-paper-partition student-topk --opd-log-prob-top-k K`: student top-K plus
  the sampled ID, deduplicated; only the target partition is approximated
  (Appendix B Eq. 7). The sampled numerator is exact. Logit/model-logprob
  additive constants cancel at the final normalization. K is an explicit
  experiment choice: this draft does not specify a universal training K=16,
  64 or 128. Diagnostic top-16 is a different parameter.

Appendix D interleaved routing requires the **full** partition. Do not label a
restricted-support run an exact reproduction of that experiment. Changing K
changes the partition approximation and must be paired across arms.

## Wiring and constraints

Use the existing reward entrypoints; they dispatch on `opd_objective`:

```
MILES_USE_LEGACY_ROLLOUT_V1=1
--use-opd --opd-type sglang --opd-objective paper
--opd-target-mode endpoint                 # paired arm: shiftmopd
--opd-teacher-selection all                # or routed, identical in both arms
--opd-paper-partition full --opd-log-prob-top-k 0
--opd-effective-vocab-size 151665
--opd-paper-model-manifest /absolute/path/to/resolved-models.json
--opd-reward-centering batch --opd-kl-coef 1
--opd-top-k-strategy only-student
--custom-rm-path miles.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
--loss-type policy_loss --advantage-estimator grpo
--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-min-p 0
--entropy-coef 0 --kl-coef 0
--update-weights-interval 1 --num-steps-per-rollout 1
```

These are **argument fragments**, not a provisioned launcher. Supply the actual
model URLs/adapters, model/training flags, data, batch size, and output paths.
Disable advantage normalization, extra KL loss, TIS, OPSM and token-sum loss
reduction. Use synchronous `train.py`; stale-policy `train_async.py` is rejected.
One rollout must equal exactly one global optimizer batch. Current paper loss
supports TP=CP=1 with DP/microbatch accumulation. Custom rollout generators,
dynamic global batch sizes, and partial/fully asynchronous rollouts are rejected.

Model-manifest schema (all paths local; each revision a pinned source identity):

```
{
  "student": {"path": "...", "repo_id": "...", "revision": "..."},
  "anchor": {"path": "...", "repo_id": "...", "revision": "..."},
  "teachers": {"math": {"path": "...", "repo_id": "...", "revision": "..."}},
  "bases": {"math": {"path": "...", "repo_id": "...", "revision": "..."}}
}
```

The student path must match `--hf-checkpoint`; the anchor shares its repo and
revision. Each teacher/base key matches a scorer route. Paths can point to local
merged expert checkpoints for tokenizer/provenance validation even when those
weights are served as adapters. See the example manifest under `experiments/`.

## Draft experiment and evaluation protocols

The final draft's student/anchor is DeepSeek-R1-Distill-Qwen-1.5B. Its primary
same-origin teacher is Nemotron-Research-Reasoning-Qwen-1.5B **v1**, and its
cross-origin math teacher is Polaris-7B-Preview with R1-Distill-Qwen-7B as base.
JustRL-DeepSeek-1.5B is an additional same-origin math teacher. The Qwen3-8B
fourth-teacher extension needs Appendix F token-string projection; the Miles
adapter deliberately rejects such mismatches instead of silently using raw IDs.

Common-domain acquisition/composition uses BigMath; routed tests use
Math/Science/IF, including 50/25/25 interleaving. The draft uses LR 1e-6 and
10,000-token training responses. Preserve paired seeds, prompts, optimizer,
update budgets and evaluation ordering. Do not reuse a historical 1:1:1
math/code/IF manifest under a paper-reproduction label. Phased scheduling and
optimizer-state carryover are experiment orchestration, not inferred by the
target hook.

`miles.utils.paper_eval.paper_eval_configs` builds Miles evaluation configuration
objects from **caller-supplied frozen paths and official verifier types**:

| Suite | Items | Avg@K samples | Decoding |
|---|---:|---:|---|
| AMC2023 | 40 | 16 | temperature 1, top-p .95 |
| MATH500 | 500 | 1 | greedy |
| AIME2025 | 30 | 16 | temperature 1, top-p .95 |
| GPQA Diamond | 198 | 4 | temperature 1, top-p .95 |
| IFEval | 541 | 4 | temperature 1, top-p .95 |

Total: 1,309 frozen items. Acquisition/mechanism evaluation cap: 16,384, with
IFEval driver cap 2,048. The scaling/reference protocol is 10,000-token greedy
(IFEval retains its driver cap). `summarize_paper_eval` checks complete outcome
matrix sizes and reports average sampled accuracy, **not any-success pass@K**.
The headline macro is an unweighted mean of the five benchmark scores. Five
independent training seeds and sample standard deviations are a separate
aggregation layer, not 5 generation samples. Dataset preparation, immutable
item manifests, verifier provisioning and automatic evaluation scheduling are
not performed by the factory.

## Diagnostics and validation boundary

`paper_diagnostics.py` provides full-vocabulary forward KL from target to
student/anchor, reverse KL separately, Eq. 8 uncentered component gradients,
Gamma_base, and vocabulary-centered contribution norms/cosines/cancellation/
top-16 sign conflict. Logit geometry and parameter-gradient geometry are not
interchangeable. Domain gradients are not Eq. 8 teacher/base components.

The existing checkpoint runner now accepts Qwen2 and Qwen3 and reports this
geometry. It remains **offline**: its per-domain panels are not the draft's
live first-microbatch, first-128-valid-token, every-20-update measurement protocol.
Never compare different position-pooling protocols without labeling them.
See [diagnostic scope](shiftmopd-paper-diagnostics.md).

The new rollout telemetry logs valid-token-weighted `paper_opd/raw_reward_mean`,
`paper_opd/centered_reward_mean`, `paper_opd/batch_baseline`, and token count.
Global centered mean should approach zero; per-domain centered means need not.

CPU tests cover equations, offset/padding invariance, independent selection,
shared-base reuse, exact streamed normalization, restricted missing-mass bias,
global masked centering, detached sampled-token gradients, tokenizer rejection,
evaluation denominators and legacy regressions. Production SGLang numerical
parity, CUDA memory/throughput and multi-rank execution remain required before
launching full experiments. No prior runs, checkpoints or results are rewritten.
