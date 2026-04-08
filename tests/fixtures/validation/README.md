# Validation Fixtures

This directory stores the checked-in evidence used by the validation workflow.

- `manifest.json` declares the shared fixture set and the per-model artifact map.
- `*.contract.json` stores the approved preprocessing and tensor-semantic contract for a model.
- `*.reference.json` stores the approved aggregate and per-fixture parity signals for a model,
  plus provenance fields for runtime backend and the reference source used to generate the artifact.
- `report-summary.json` and `report-summary.html` are representative snapshots of the
  user-facing validation summary, including artifact provenance and parity
  diagnostics.

The fixture images are generated in-process from the manifest patterns so default
tests stay offline-safe and deterministic. The standard fixture set currently
uses gradient, checkerboard, and centered-square images so parity evidence can
catch fixture-specific regressions instead of only aggregate drift.

Validation also includes an input-independence gate that compares model outputs
for all-zero versus deterministic random probe images. High cosine similarity on
that probe pair is treated as a hard failure because it indicates an
input-insensitive export.
