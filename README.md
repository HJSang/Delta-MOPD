# ShiftMOPD

Research code for **Composing What Each Teacher Learned: Multi-Teacher On-Policy
Distillation through Teacher-Relative Shifts**.

[Paper (arXiv:2610.10460)](https://arxiv.org/abs/2610.10460) ·
[Implementation contract](docs/advanced/shiftmopd-final-draft.md) ·
[Scorer deployment](docs/scoring.md) ·
[Diagnostics](docs/advanced/shiftmopd-paper-diagnostics.md)

This repository contains our Miles-based MOPD and ShiftMOPD implementation.
It is **private during review**. CPU correctness tests are available; the new
paper-aligned execution path still needs production SGLang numerical-parity,
GPU throughput, and multi-rank validation. This is not yet a turnkey
reproduction of every experiment in the paper.

## Motivation

A post-trained teacher contains both its precursor's behavior and changes learned
during post-training. ShiftMOPD transfers the teacher-relative changes while
anchoring the target to the student's initial model. The endpoint control instead
composes teachers relative to that same student anchor.

Let `A` be the frozen initial student, `T_i` a frozen expert, `B_i` its exact
pre-post-training precursor, and `S` the selected teachers. On each
student-generated token prefix, the two target scores are:

```text
MOPD endpoint: z_endpoint = z_A + sum_{i in S}(z_Ti - z_A)
ShiftMOPD:     z_shift    = z_A + sum_{i in S}(z_Ti - z_Bi)
```

Normalize the target scores to obtain `q`. With one selected teacher, the endpoint
target reduces to that teacher. If every precursor equals the anchor, the two
targets coincide. Teacher selection (`all` or `routed`) is independent of target
construction and must be matched across a controlled comparison.

Both arms use the **same sampled-token score-function loss**:

```text
reward      = log q(sampled_token) - log p_old_student(sampled_token)
baseline    = mean reward over valid response tokens in the optimizer batch
token_loss  = -stop_gradient(reward - baseline) * log p_student(sampled_token)
loss        = mean over responses of their mean valid-token loss
```

The paper path uses batch centering, **not variance whitening**, candidate-weighted
top-K losses, or PPO clipping. Top-K, when explicitly enabled, approximates the
target partition only; it does not replace the sampled-token loss.

## Status and boundaries

| Component | Status |
|---|---|
| Target algebra, sampled loss, global masked centering | Implemented; CPU-tested |
| Independent `all` / `routed` teacher selection | Implemented |
| Standalone teachers and shared-base multi-LoRA scoring | Implemented; routing/TITO tests |
| Full effective-vocabulary target partition | Exact streamed reference; expensive |
| Student top-K plus sampled-token partition | Optional approximation; not the full-partition experiment |
| Identical-tokenizer and model-manifest preflight | Implemented; remote weight identity must be verified separately |
| Benchmark configuration and Avg@K aggregation | Implemented; frozen datasets/verifiers must be provisioned |
| Component gradients, KL, and logit geometry | Offline checkpoint diagnostics |
| Cross-tokenizer projection (Appendix F) | Not implemented; incompatible tokenizers are rejected |
| Final paper results reproduced by this snapshot | Not established |

Historical experiments used different objectives and configurations. They must
not be presented as validation of the new paper path. The inherited runtime
retains `legacy` as its default; **select `--opd-objective paper` explicitly**.

## Quick start: CPU correctness tests

Use Python 3.12. The test environment does not require SGLang, Megatron, downloaded
checkpoints, or GPUs. Tests construct tiny models locally.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q
```

With `uv`, the equivalent isolated command is:

```bash
uv run --no-project --python 3.12 --with-requirements requirements-test.txt python -m pytest -q
```

Coverage includes target identities, padding/offset invariance, sampled-token
gradients, full and restricted normalization, batch centering, packed transport,
teacher routing, token preservation, evaluation denominators, and tiny Qwen
checkpoint diagnostics. A CUDA-only parity test is skipped on CPU. GPU-serving
tests under `experiments/test_native_position_logprobs.py` are intentionally not
part of the default CPU suite. On October 8, 2026, the source-export validation
completed with **151 passed and 1 CUDA-only test skipped**.

## Training configuration

Training requires a compatible Linux CUDA environment with Miles' SGLang and
Megatron dependencies. Install this source into that provisioned environment
with `python -m pip install -e . --no-deps`; this command does **not** install or
validate those GPU dependencies. See [upstream Miles](https://github.com/radixark/miles)
for framework/environment setup and [provenance](PROVENANCE.md) for this snapshot.

1. Download and pin the student, frozen anchor, experts, and exact precursors.
   Fill local paths in the [example manifest](experiments/final_draft_model_manifest.example.json).
2. Provision SGLang scoring routes and verify their checkpoint/adapter identities.
   Both standalone and shared-base multi-LoRA layouts are supported; see
   [scoring.md](docs/scoring.md).
3. Provision frozen training/evaluation data and official verifiers. Data and
   weights are not bundled in this repository.
4. Use synchronous `train.py`, the paper argument fragments in the
   [implementation contract](docs/advanced/shiftmopd-final-draft.md), and the same
   seed, data order, teacher selection, sampling, batch size, update budget,
   optimizer, partition, and evaluation protocol for both arms.
5. Change only `--opd-target-mode endpoint` to `--opd-target-mode shiftmopd`
   for the controlled target comparison. Run GPU parity and a short smoke test
   before a full experiment.

The paper path currently requires training TP=CP=1, one optimizer batch per
rollout, temperature 1, top-p 1, no sampling top-K truncation, zero dropout,
and no additional task/KL/entropy objective. Set
`MILES_USE_LEGACY_ROLLOUT_V1=1` for its supported rollout integration.
`train_async.py` is retained for other Miles workflows but is **not supported**
by the paper objective. Do not bypass the preflight checks.

Full normalization uses 151,665 effective tokens for the draft's aligned
tokenizers. It streams vocabulary blocks through the scoring API and is a slow
correctness reference. Restricted support requires an explicit K and must be
labeled as an approximation; the paper does not prescribe one universal training
K. Diagnostic top-16 is separate from training support and decoding top-K.

Configure W&B through your environment (`WANDB_API_KEY`) and the normal Miles
tracking arguments. Never put credentials, private deployment addresses, run
logs, model weights, or evaluation traces into Git.

## Paper experiment protocol

The draft's primary student/anchor is **DeepSeek-R1-Distill-Qwen-1.5B**.
Primary experts include **Nemotron-Research-Reasoning-Qwen-1.5B v1** and
**Polaris-7B-Preview**; Polaris' precursor is DeepSeek-R1-Distill-Qwen-7B.
JustRL-DeepSeek-1.5B provides an additional same-origin math expert.
See the pinned example manifest and the contract for scope and restrictions.

The common-domain protocol uses BigMath; routed experiments use Math/Science/IF.
These are not the historical Qwen3 math/code/IF recipes. The draft uses a
1e-6 learning rate and a 10,000-token training response cap.

| Evaluation suite | Items | Samples per item |
|---|---:|---:|
| AMC2023 | 40 | 16 |
| MATH500 | 500 | 1, greedy |
| AIME2025 | 30 | 16 |
| GPQA Diamond | 198 | 4 |
| IFEval | 541 | 4 |

`miles.utils.paper_eval` constructs configurations from caller-provided frozen
dataset paths and verifier types. Avg@K means average sampled correctness,
**not any-success pass@K**. The headline macro gives each benchmark equal weight.
Independent training seeds are a separate aggregation dimension. Decoding and
length-cap details are in the implementation contract.

## Repository map

| Path | Purpose |
|---|---|
| `miles/utils/paper_opd.py` | Target composition, normalization, and centering |
| `miles/rollout/paper_opd.py` | Frozen scorer orchestration and batch reward preparation |
| `miles/backends/training_utils/loss_hub/paper_opd.py` | Sampled-token training loss |
| `miles/utils/paper_opd_config.py` | Fail-closed argument validation |
| `miles/utils/paper_opd_preflight.py` | Model identity and tokenizer-map checks |
| `miles/utils/paper_eval.py` | Evaluation protocols and aggregation |
| `miles/backends/training_utils/paper_diagnostics.py` | Gradient and distribution diagnostics |
| `experiments/measure_shiftmopd_checkpoint.py` | Offline diagnostic runner |
| `experiments/patch_sglang_*.py` | Selected-ID and packed-transport extensions |
| `tests/` | Focused CPU correctness and regression tests |

## Before making this repository public

- [ ] Verify deployed SGLang scores against dense reference outputs.
- [ ] Complete CUDA and multi-rank training smoke tests for both paper arms.
- [ ] Freeze a tested serving/training image and dataset/verifier manifests.
- [ ] Measure full-partition memory and throughput; document practical limits.
- [ ] Review reproduction claims, licenses, and included files once more.
- [ ] Change visibility only after explicit approval.

## Citation and acknowledgments

```bibtex
@article{sang2026shiftmopd,
  title={Composing What Each Teacher Learned: Multi-Teacher On-Policy
         Distillation through Teacher-Relative Shifts},
  author={Sang, Hejian and Zhou, Zhengze and Hamidi, Shayan Mohajer and
          Li, Xiaomin and Jain, Rohit and Geramifard, Alborz},
  journal={arXiv preprint arXiv:2610.10460},
  year={2026}
}
```

Built on [Miles](https://github.com/radixark/miles), with SGLang for rollout and
scoring. Existing upstream copyright notices and the
[Apache-2.0 license](LICENSE) are retained. See [PROVENANCE.md](PROVENANCE.md)
for local modifications and import scope. External models and datasets retain
their own licenses.
