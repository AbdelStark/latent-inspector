# Validation Fixtures

This directory stores the checked-in evidence used by the validation workflow.

- `manifest.json` declares the shared fixture set and the per-model artifact map.
- `*.contract.json` stores the approved preprocessing and tensor-semantic contract for a model.
- `*.reference.json` stores the approved aggregate and per-fixture parity signals for a model.
- `report-summary.json` and `report-summary.html` are representative snapshots used by report tests.

The fixture images are generated in-process from the manifest patterns so default
tests stay offline-safe and deterministic. The standard fixture set currently
uses gradient, checkerboard, and centered-square images so parity evidence can
catch fixture-specific regressions instead of only aggregate drift.

The checked-in reference artifacts are generated from the development stub
backend so regression tests can stay offline-safe. The product now treats any
stub-backed validation run as `unverified` even when those fixture comparisons
match, because synthetic outputs are useful for plumbing checks but not for
release-grade source-alignment claims.
