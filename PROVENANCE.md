# Source provenance

This is a standalone, source-only research repository, not an upstream Miles
release or an automatically synchronized fork.

- Upstream framework: https://github.com/radixark/miles
- Recorded upstream checkout: `feb4e7cc81e85a9e10b7c21795ce8d4e4c9ca3e0`.
- Source lineage: the local Miles working tree imported into
  `multi_teacher_opd/miles_mopd` on September 25, 2026, with subsequent local
  MOPD/Δ-MOPD changes through October 8, 2026.
- The upstream revision identifies the recorded base, **not** the entire current
  tree. This snapshot includes modified and locally added framework files.
- Retained: Miles runtime/plugins, training entrypoints, selected Qwen model
  definitions/converters, paper implementation, diagnostics, scorer patches,
  and focused tests.
- Excluded: historical machine-specific launchers, experiment schedulers,
  monitoring scripts, unrelated examples/docs/CI, caches, credentials,
  checkpoints, datasets, and generated experiment artifacts.
- Two test sections asserting contents of excluded historical shell launchers
  were removed. Numerical and routing checks were retained; legacy centering
  defaults are still tested.
- The live transport smoke tool now takes an explicit local `--tokenizer` path
  instead of a historical machine-specific directory. Upstream chat-template
  whitespace is preserved because it can change tokenization.

Local additions and modifications include teacher-relative target composition,
teacher routing and multi-LoRA scoring, per-position selected-ID/packed transport,
sampled-token paper loss, optimizer-batch centering, effective-vocabulary
preflight, evaluation aggregation, and gradient/KL diagnostics. Historical
objective paths remain for compatibility and regression tests; the README
identifies the explicit paper path.

No original experiment directory is modified by this source export. No results
are relabeled or invented. GPU tests from earlier historical paths do not prove
the new paper path's production readiness.

The original Apache-2.0 license and source copyright notices are preserved.
This repository does not grant rights to external checkpoints, datasets,
inference frameworks, or benchmark verifiers beyond their respective licenses.
