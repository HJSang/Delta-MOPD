# Frozen teacher engine recipes

Both recipes use the same [launcher](serve.py) and Miles scoring implementation.
They provision **frozen scorers**, not student training or student rollout GPUs.
Commands run from the repository root with Python 3.12 in a compatible SGLang
environment. Use that environment's `python`, not an older system interpreter.

| Recipe | When to use it | Frozen scoring layout |
|---|---|---|
| [Standalone teachers](standalone/README.md) | Full expert checkpoints, including experts with different precursors | Separate expert, precursor, and anchor processes; shared identical models reused |
| [Multi-LoRA teachers](multi_lora/README.md) | Real adapters trained from one exact shared base | One base-plus-adapters engine and a separate frozen student-anchor engine |

The standalone example uses the paper's two primary experts. The multi-LoRA
example is a **Qwen3 deployment template**, not a replacement for the paper's
different-sized full checkpoints or a claim of reproduced paper results.

## Shared safety and configuration

- Commands print by default. Only `--execute` starts one selected server in the
  foreground. Stop it with Ctrl-C. There is no cluster allocation, automatic
  download, broad process kill, or background supervisor.
- Edit the JSON configuration to your actual **absolute local** model/adapter
  paths, ports, and allocated GPU indices. These are single-host recipes.
- Each engine has disjoint `devices`; tensor parallel size is their count. The
  launcher sets `CUDA_VISIBLE_DEVICES` explicitly, overriding an inherited mask.
  Use device indices valid inside your container/allocation. Leave other GPUs
  for the student trainer and rollout engines.
- Defaults: BF16, context length 32,768, static-memory fraction 0.7, and at most
  16 running requests. Override `context_length`, `mem_fraction_static`, and
  `max_running_requests` per engine in JSON after measuring memory and latency.
  These are conservative examples, not benchmarked optimal settings.
- Servers bind only to `127.0.0.1`. Do not expose unprotected scoring endpoints.
  Multi-node deployments need explicit secured network routing and different
  client URLs; the loopback route generator is not a multi-node launcher.
- `--execute` checks model files and LoRA declarations before starting SGLang.
  It does not establish that weights are numerically identical to a manifest or
  that a model fits GPU memory. Record hashes/revisions separately.

## Patches and compatibility

Install a compatible SGLang GPU environment first. Selected-ID serving and
packed transport may require the staged patches in [scoring.md](../docs/scoring.md).
Check your installed `python -m sglang.launch_server --help` against the generated
command. Do not patch or replace a live serving process.

The launcher only uses Python's standard library for dry runs. CPU tests check
command construction, route maps, invalid configurations, and adapter-base
declarations; **live GPU serving is not certified by those tests**.
Refer to [SGLang's LoRA documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/lora.mdx)
for `--lora-paths`, native `lora_path`, and base-only requests. Inspect the exact
SGLang version's supported architecture/adapter combinations before deployment.

## Connecting to training

`--routes` prints only the topology/objective-selection portion of Miles'
arguments. It is **not a complete training command**. Combine it with your
model, data, optimizer, allocation, and batch flags and all requirements in the
[paper contract](../docs/advanced/shiftmopd-final-draft.md), including:

```text
--custom-rm-path miles.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
--loss-type policy_loss --advantage-estimator grpo
--opd-reward-centering batch --opd-kl-coef 1
--opd-top-k-strategy only-student
--opd-paper-partition full --opd-log-prob-top-k 0
--opd-paper-model-manifest /absolute/path/to/resolved-models.json
--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-min-p 0
--entropy-coef 0 --kl-coef 0
--hidden-dropout 0 --attention-dropout 0
--update-weights-interval 1 --num-steps-per-rollout 1
```

Use `MILES_USE_LEGACY_ROLLOUT_V1=1` and synchronous `train.py`; disable advantage
normalization, extra KL/TIS/OPSM, and token-sum reduction as required by preflight.
One rollout equals one optimization batch. The full-partition implementation is
an expensive streamed reference; establish numerical parity on a small smoke
run before planning a full run. Set the correct effective vocabulary explicitly.

For paired arms, keep teacher selection, sampling, data order, model revisions,
loss, partition, and optimizer settings identical. Generate routes with
`--target-mode endpoint` for MOPD or `--target-mode shiftmopd` for Δ-MOPD.
Use `--selection routed` only when each prompt has `metadata.opd_teacher` naming
one of the recipe's teachers; `--selection all` composes every configured expert.
