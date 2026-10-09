# Recipe 1: standalone teacher engines

This example serves the paper's Polaris math expert and Nemotron v1 Science/IF
expert. It reuses the frozen initial student as Nemotron's precursor. Full
teacher checkpoints need no LoRA adapters. Both MOPD and Δ-MOPD use the same
teacher endpoints; Δ-MOPD additionally uses precursor scores.

## 1. Prepare checkpoints

Download the pinned [manifest](../../experiments/final_draft_model_manifest.example.json)
revisions. Choose a writable models directory, download there, then update
`model_path` entries in [config.json](config.json) and the model manifest to those
exact directories. For example (these commands download model weights):

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --local-dir /models/DeepSeek-R1-Distill-Qwen-1.5B
hf download POLARIS-Project/Polaris-7B-Preview --revision faea72cf7123681069668e8da45c1c074e378830 --local-dir /models/Polaris-7B-Preview
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --revision 916b56a44061fd5cd7d6a8fb632557ed4f724f60 --local-dir /models/DeepSeek-R1-Distill-Qwen-7B
hf download nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --revision b89048893f95246c6b5749b287f0049e6df42ee9 --local-dir /models/Nemotron-Research-Reasoning-Qwen-1.5B-v1
```

Never overwrite an existing directory containing a different revision. Verify
the manifest's tokenizer maps and the deployed weights before training.

| Engine | GPU in example | Port | Frozen checkpoint |
|---|---:|---:|---|
| `anchor` | 0 | 31000 | Initial 1.5B student; also Science/IF precursor |
| `math` | 1 | 31001 | Polaris-7B-Preview |
| `math_base` | 2 | 31002 | DeepSeek-R1-Distill-Qwen-7B |
| `science_if` | 3 | 31003 | Nemotron-Research-Reasoning-Qwen-1.5B v1 |

These are scoring GPUs only. The hardware must fit each model and scoring
context; assign additional disjoint devices to an engine for TP if needed.

## 2. Preview and launch

Inspect a command without importing SGLang or allocating a GPU:

```bash
python -m recipes.serve --config recipes/standalone/config.json --engine math
```

Then run each of these **in a separate terminal** on the same host:

```bash
python -m recipes.serve --config recipes/standalone/config.json --engine anchor --execute
python -m recipes.serve --config recipes/standalone/config.json --engine math --execute
python -m recipes.serve --config recipes/standalone/config.json --engine math_base --execute
python -m recipes.serve --config recipes/standalone/config.json --engine science_if --execute
```

Check health after model loading:

```bash
curl --fail http://127.0.0.1:31000/health
curl --fail http://127.0.0.1:31001/health
curl --fail http://127.0.0.1:31002/health
curl --fail http://127.0.0.1:31003/health
```

Health only proves the server answers, not weight identity or logprob parity.
Perform the [scoring validation](../../docs/scoring.md#validation-gates) before
training, including identical-token teacher/base/anchor comparisons.

## 3. Generate matching Miles routes

```bash
python -m recipes.serve --config recipes/standalone/config.json --routes --target-mode endpoint --selection all
python -m recipes.serve --config recipes/standalone/config.json --routes --target-mode shiftmopd --selection all
```

Teacher names are `math` and `science_if`. For domain routing, use
`--selection routed` and those exact metadata values. Science and IF prompts
share the one `science_if` expert; do not duplicate it in an `all` composition.

The paper's effective vocabulary is **151,665**. Use
`--opd-effective-vocab-size 151665` and the resolved paper manifest. The endpoint
argument fragment omits precursor routes. You may leave `math_base` stopped for
an endpoint-only run; the current paper preflight still requires an anchor route.

To add the paper's third expert, provision a separate JustRL-DeepSeek-1.5B engine,
add one teacher route and a matching base route to `anchor`, and update the
manifest. Do not call a full checkpoint a LoRA adapter.

See the [shared instructions](../README.md) for training flags, patches,
resource ownership, and current validation limits.
