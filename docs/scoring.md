# Frozen-model scoring

Scorers consume the student's exact prompt and response token IDs. They do not
generate teacher responses, apply another chat template, or decode/re-tokenize
the sequence. Scoring requests use `input_ids`, `max_new_tokens=0`,
`return_logprob=true`, and `logprob_start_len=0` to preserve full-prefix alignment.

The examples below are argument fragments for already provisioned servers, not
launch commands. Combine them with the [paper contract](advanced/shiftmopd-final-draft.md).
Every model must pass the same-tokenizer preflight. URLs below are placeholders.

## Standalone experts

```text
--opd-teacher-urls math=http://math-host:30000/generate science_if=http://science-host:30000/generate
--opd-base-urls math=http://math-base-host:30000/generate science_if=http://student-anchor-host:30000/generate
--opd-anchor-url http://student-anchor-host:30000/generate
```

Omit `--opd-teacher-adapters`. The endpoint arm does not consume precursor
scores; the ShiftMOPD arm does. The current paper preflight requires an anchor
and a complete model manifest in both arms, even though an endpoint with only
one selected teacher algebraically reduces to that teacher.

With `--opd-teacher-selection routed`, metadata such as
`{"opd_teacher": "math"}` selects a teacher. With `all`, distinct configured
teacher identities are composed. Do not register the same expert twice under
different domain names to accidentally count its contribution twice.

## Shared precursor with multiple LoRA experts

When experts are actual adapters of the **same exact precursor**, register them
in one SGLang engine and configure logical routes:

```text
--opd-teacher-urls math=http://shared-host:30000/generate code=http://shared-host:30000/generate
--opd-teacher-adapters math=math-v1 code=code-v1
--opd-base-urls math=http://shared-host:30000/generate code=http://shared-host:30000/generate
--opd-anchor-url http://student-anchor-host:30000/generate
```

Requests choose one adapter using `lora_path`; they do not apply all adapters
together. Base requests use the unadapted model. Deduplication is by
`(URL, adapter)`, not URL alone. An adapter map must cover all configured teacher
routes; use `none` for an explicitly unadapted teacher route.

This topology is conditional on matching base weights and adapter provenance.
It cannot combine different model architectures or different precursor weights
into one base just because they share a tokenizer. In particular, the paper's
1.5B and 7B experts cannot share one multi-LoRA base. Verify merged-versus-adapter
numerical parity separately.

## Per-position selected IDs

The optional extension accepts `token_ids_logprob_positions[position]`, indexed
by the **target input-token position**, not the preceding predictor row. Prompt
rows can be empty; response rows contain the selected IDs. SGLang accounts for
the causal one-token shift internally. Scores remain normalized over the model's
full vocabulary; the client constructs and normalizes the composite target.

```bash
python experiments/patch_sglang_position_logprobs.py /path/to/staged/sglang/python/sglang
python experiments/patch_sglang_position_logprobs.py /path/to/staged/sglang/python/sglang --write
```

The first invocation previews changes. Apply only to a staged, compatible
SGLang tree: patchers check source anchors and reject unrecognized layouts.
Keep `sglang_position_logprobs.py` beside its patcher. Do not patch a live
training deployment. Restart and verify scorer processes before testing.

With `--opd-require-native-selected-ids`, missing native output is an error
rather than silent fallback. Without that requirement, the compatibility path
may issue more requests; a successful response alone does not prove native
per-position scoring is active.

## Packed transport

After the per-position patch, optionally apply:

```bash
python experiments/patch_sglang_packed_scores.py /path/to/staged/sglang/python/sglang
python experiments/patch_sglang_packed_scores.py /path/to/staged/sglang/python/sglang --write
```

Clients request `return_packed_token_ids_logprobs=true`; servers return
`meta_info.input_token_ids_logprobs_packed` with version 1, int64 row offsets,
int32 token IDs, and FP32 or FP64 scores in base64-encoded little-endian buffers.
FP32 is used only if the FP64-to-FP32-to-FP64 round trip is exact. Decoders reject
invalid schemas, sizes, IDs, NaN, and positive infinity. `pybase64` is required
on both sides. This changes transport, not supports, reward formulas, or token
alignment. Check telemetry because older servers may ignore request flags.

## Validation gates

The default CPU suite checks routing, token preservation, packed decoding, and
composition using synthetic/mocked scorer responses. For a compatible patched
GPU environment, also run:

```bash
python -m pytest -q experiments/test_native_position_logprobs.py
python -m experiments.verify_packed_score_transport --help
```

Validate deployed teacher, base, and anchor scores against dense references on
identical tokens and precision, including long responses and adapter changes.
Full-partition streaming and restricted-support composition require separate
checks. Do not infer paper-path GPU readiness from historical transport tests.

The inherited Open-MOPD verifier adapter requires an external Open-MOPD checkout;
set `OPEN_MOPD_REWARD_SCORE_ROOT` explicitly if using it. This source export does
not bundle that repository or provision benchmark verifiers.
