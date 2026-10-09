# Recipe 2: multi-LoRA teacher engine

This deployment template serves **one Qwen3-4B base plus three genuine expert
adapters** in one SGLang engine, alongside a frozen Qwen3-1.7B student anchor.
It demonstrates math/code/IF routing. It is not the paper's primary checkpoint
combination: Polaris-7B and Nemotron-1.5B cannot be loaded as adapters of one base.

## 1. Prepare compatible assets

Edit [config.json](config.json) to point to your pinned base, student, and PEFT
adapter directories:

| Asset | Example local directory |
|---|---|
| Frozen student/anchor | `/models/Qwen3-1.7B` |
| Shared expert precursor | `/models/Qwen3-4B` |
| Math LoRA | `/adapters/qwen3-4b-math` |
| Code LoRA | `/adapters/qwen3-4b-code` |
| IF LoRA | `/adapters/qwen3-4b-if` |

Each adapter needs `adapter_config.json` and `adapter_model.safetensors` (or
`adapter_model.bin`), and must actually have been trained on the exact shared
base weights. The launcher checks `peft_type=LORA`, positive rank, and declared
`base_model_name_or_path` against the configured `model_id` or local base path.
This is a declaration check, **not proof of exact revision or numerical parity**.
Record the base and adapter revisions/hashes and compare adapter versus merged
checkpoint scores. Do not edit adapter metadata merely to force a mismatch to pass.

Adapter files are **not included**. If you only have merged experts, use the
standalone recipe instead; this launcher does not invent or recover adapters.
Prepare merged HF directories for each teacher's tokenizer/provenance entry in
the paper model manifest even when inference uses LoRA adapters.

## 2. Preview and launch

The default example reserves GPU 0 for the frozen anchor and GPU 1 for the shared
base/adapters. Student training and rollout use other allocated GPUs.

```bash
python -m recipes.serve --config recipes/multi_lora/config.json --engine experts
```

The preview includes `--enable-lora`, one named `--lora-paths` entry per adapter,
and `--max-loras-per-batch 4`: three adapters **plus one base-only slot**. The
example uses the Triton LoRA backend; validate your SGLang version and architecture.

Run these in separate terminals:

```bash
python -m recipes.serve --config recipes/multi_lora/config.json --engine anchor --execute
python -m recipes.serve --config recipes/multi_lora/config.json --engine experts --execute
```

Check both servers:

```bash
curl --fail http://127.0.0.1:32000/health
curl --fail http://127.0.0.1:32001/health
```

## 3. Route experts and the unadapted base

```bash
python -m recipes.serve --config recipes/multi_lora/config.json --routes --target-mode endpoint --selection routed
python -m recipes.serve --config recipes/multi_lora/config.json --routes --target-mode shiftmopd --selection routed
```

| Logical score | Endpoint | Native request adapter |
|---|---|---|
| Math expert | `http://127.0.0.1:32001/generate` | `lora_path="math"` |
| Code expert | `http://127.0.0.1:32001/generate` | `lora_path="code"` |
| IF expert | `http://127.0.0.1:32001/generate` | `lora_path="if"` |
| Shared precursor | `http://127.0.0.1:32001/generate` | Omit `lora_path` |
| Frozen student anchor | `http://127.0.0.1:32000/generate` | Omit `lora_path` |

Base-only scoring does not need a fabricated zero-valued adapter. Miles reuses
identical `(URL, adapter)` scores; it must never merge different adapters merely
because their URLs match. Both arms receive the exact student token IDs.

Set `--selection all` in **both** commands for common-state composition rather
than domain routing. Keep model/adapter revisions, tokenization, partition,
sampling, and optimizer settings fixed between the paired targets.

## 4. Validate before training

The Qwen3 example's tokenizer family has **151,669** effective IDs; do not copy
the paper's 151,665 setting. Use `--opd-effective-vocab-size 151669` only after
verifying identical complete token-ID maps for your actual assets. Build a new
model manifest for Qwen3, not the DeepSeek/Polaris example manifest. The student
and anchor must identify the same frozen initialization.

With the compatible selected-ID/packed patches installed, check live transport:

```bash
python -m experiments.verify_packed_score_transport --url http://127.0.0.1:32001/generate --adapters none math code if
```

Also verify anchor scores, full-vocabulary reference scores, long-prefix causal
alignment, and adapter-versus-merged parity. Transport parity alone is not
validation of the complete paper training path. The shared
[instructions](../README.md) describe remaining training flags and limitations.
