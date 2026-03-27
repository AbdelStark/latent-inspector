# Validation Fixtures

This directory stores the checked-in evidence used by the validation workflow.

- `manifest.json` declares the shared fixture set and the per-model artifact map.
- `*.contract.json` stores the approved preprocessing and tensor-semantic contract for a model.
- `*.reference.json` stores the approved parity signals and tolerances for a model.
- `report-summary.json` and `report-summary.html` are representative snapshots used by report tests.

The fixture images are generated in-process from the manifest patterns so default
tests stay offline-safe and deterministic.
