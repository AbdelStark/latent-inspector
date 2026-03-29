# Validation Fixtures

This directory stores the checked-in evidence used by the validation workflow.

- `manifest.json` declares the shared fixture set and the per-model artifact map.
- `*.contract.json` stores the approved preprocessing and tensor-semantic contract for a model.
- `*.reference.json` stores the approved aggregate and per-fixture parity signals for a model.
- `report-summary.json` and `report-summary.html` are representative snapshots of the
  user-facing validation summary, including artifact provenance and parity
  diagnostics.

The fixture images are generated in-process from the manifest patterns so default
tests stay offline-safe and deterministic. The standard fixture set currently
uses gradient, checkerboard, and centered-square images so parity evidence can
catch fixture-specific regressions instead of only aggregate drift.

The checked-in reference artifacts now record which execution backend produced
them. The ready `dinov2-vit-l14` artifact is captured from live ONNX Runtime
execution, while planned integrations still use the development stub backend so
fixture plumbing stays offline-safe. The product treats any stub-backed
validation evidence as development-only and will mark it stale or unverified
instead of presenting it as release-grade source-alignment proof.
