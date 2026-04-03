# Architecture

## Identity

`latent-inspector` is a Rust CLI/TUI for inspecting and comparing learned image representations from self-supervised vision models using ONNX Runtime, checked-in validation evidence, and exportable reports.

Deployment context: local developer/operator tool. This repository does not expose a network service, database, or stable library API.

## System overview

```text
CLI / TUI
  -> model registry + cache inspection
  -> image preprocessing
  -> ONNX session loading / stub backend
  -> feature extraction
  -> analysis metrics
  -> validation summary
  -> terminal / JSON / HTML / PNG / artifact manifest output
```

## Repository map

### Runtime flow

- `src/cli/`
  Command handlers. Each command is responsible for argument validation, orchestrating model work, and selecting output formats.
- `src/models/`
  Registry metadata, artifact download/cache verification, preprocessing, session loading, and model inventory reporting.
- `src/extract/`
  Converts raw `ModelOutput` into normalized feature shapes used by analysis and reporting.
- `src/analysis/`
  Representation metrics and pairwise comparison logic.
- `src/validation/`
  Fixture loading, freshness checks, preprocessing/tensor checks, and parity against approved reference artifacts.
- `src/viz/`
  Human-readable terminal output, machine-readable JSON, shareable HTML, PNG charts, and `artifacts.json` bundle manifests.
- `src/tui/`
  Interactive terminal UI. No image means demo data. Providing an image runs live analysis.

### Non-code assets

- `tests/fixtures/validation/`
  Checked-in contracts and reference artifacts that define the validation trust surface.
- `docs/assets/`
  Screenshots, sample images, and example report images used in documentation.

## Core data flow

### Single-image analysis

1. CLI parses the target model(s) and image path.
2. `models::preprocess` resizes, crops, and normalizes the image according to the registry entry.
3. `models::loader` loads the verified ONNX artifact bundle or uses the explicit stub backend when requested.
4. `extract::features` derives CLS, patch-token, and optional attention views.
5. `analysis::*` computes per-model or pairwise metrics.
6. `validation::*` attaches the current trust state for the selected model(s).
7. `viz::*` renders terminal, JSON, HTML, or PNG output and writes `artifacts.json` for non-terminal formats.

### Dataset analysis

Dataset commands (`neighbors`, `similarity`, `profile`, `drift`) use `dataset::map_images_parallel` to scan supported images, skip corrupt files with explicit accounting, and preserve output ordering.

## Runtime modes

### Live ONNX mode

- Default mode
- Uses cached/downloaded model artifacts
- Validation can reach `validated`, `stale`, `failed`, or `partial` depending on evidence and observed behavior

### Stub mode

- Enabled with `LATENT_INSPECTOR_MODEL_BACKEND=stub`
- Produces deterministic synthetic outputs for development and tests
- Useful for planned-model report flows and fast local iteration
- Validation must be treated as `unverified` because outputs are synthetic

## Invariants and contracts

- Every live model load must come from a registry entry with explicit artifact metadata.
- Artifact bundles are only usable when all required files are present and checksum-valid where checksums are pinned.
- Validation reports must not claim source alignment when the stub backend is active.
- Report bundle artifact paths must stay relative to the output directory and must not contain parent-directory escapes.
- Analysis functions must surface invalid or incoherent numeric inputs as typed errors instead of silently fabricating metric values.

## Operational notes

### Cache location

- Default: platform cache directory under `latent-inspector`
- Override: `LATENT_INSPECTOR_CACHE_DIR`

### Other environment variables

- `LATENT_INSPECTOR_MODEL_BACKEND=stub`
  Forces the synthetic development backend.
- `LATENT_INSPECTOR_FORCE_ASCII=1`
  Forces ASCII-only terminal rendering.

### Debugging workflow

1. Run `cargo test`.
2. Run `cargo clippy --all-targets -- -D warnings`.
3. Reproduce the failing command with `RUST_LOG=info` or `RUST_LOG=debug`.
4. If the failure concerns trust/reporting, run `validate` before changing goldens.
5. If the failure concerns downloads, inspect `latent-inspector models --verbose` to confirm bundle completeness and checksum status.

### Validation-golden workflow

- Goldens live in `tests/fixtures/validation/`.
- Refresh only after a verified contract/artifact change.
- Never refresh while the stub backend is active.

## Failure modes

- Download failure: surfaced as a typed `ModelError::DownloadFailed` or checksum/invalid-artifact error.
- Transient download failures: retried with bounded backoff before surfacing a hard failure.
- Validation staleness: surfaced as `stale`, not silently treated as approved.
- Corrupt dataset images: counted and reported in dataset summaries.
- Missing checkpoint entries in drift runs: now treated as hard errors if directory enumeration fails.

## Release/readiness notes

As of 2026-04-03 the project is alpha. The CLI is the product surface; library consumers should expect API churn until `1.0`. The machine-readable report bundle contract for the ready-model commands is documented in `docs/REPORT-SCHEMA.md`. The highest remaining readiness gaps are interactive TUI test coverage and live integrations for the planned models.
