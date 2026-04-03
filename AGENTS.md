# AGENTS.md

## Project identity

`latent-inspector` is a Rust CLI and terminal UI for inspecting, comparing, validating, and exporting reports about learned image representations from self-supervised vision models.

Deployment context: this is a local developer/operator tool, not a network service and not yet a stable library API.

Current state as of 2026-04-03: alpha. Four models are ready for live ONNX-backed analysis: `dinov2-vit-l14`, `ijepa-vit-h14`, `vjepa2-vitl-fpc2-256`, and `eupe-vit-b16`. Planned models remain visible in the registry but are not ready for live inference.

## Architecture map

- `src/cli/`: command entry points for `compare`, `inspect`, `neighbors`, `similarity`, `profile`, `drift`, `models`, `validate`, and `tui`
- `src/models/`: registry metadata, cache/download management, preprocessing, ONNX session loading, and model inventory reporting
- `src/extract/`: converts raw model outputs into CLS, patch-token, and attention-derived features
- `src/analysis/`: PCA, rank, entropy, isotropy, uniformity, CKA, k-NN overlap, correspondence, and intrinsic dimensionality
- `src/validation/`: fixture loading, preprocessing/tensor contract checks, freshness checks, and parity against checked-in reference artifacts
- `src/viz/`: terminal, JSON, HTML, PNG, and artifact-manifest generation
- `src/tui/`: interactive terminal application and views
- `tests/`: CLI integration coverage plus validation-golden checks

## Tech stack

- Rust 2021, minimum Rust `1.75.0`
- ONNX Runtime via `ort = 2.0.0-rc.12`
- `ndarray`, `image`, `rayon`, `clap`, `ratatui`, `crossterm`, `reqwest`, `serde`, `tracing`

## Verified commands

Run these from the repository root:

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo llvm-cov --workspace --ignore-filename-regex '(^|/)src/tui/|(^|/)src/cli/tui.rs$' --fail-under-lines 85 --fail-under-functions 80 --summary-only
cargo build --release
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- models
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- compare docs/assets/img/samples/elephant_sample_image.jpg --models dinov2-vit-l14,ijepa-vit-h14
```

## Conventions

- Keep errors typed and contextual via `src/errors.rs`. Do not silently convert analysis failures into numeric metrics.
- Treat validation output as a trust surface. If a change affects preprocessing, tensor semantics, or reference parity, update the validation evidence intentionally and document why.
- Non-terminal report commands should continue to emit `artifacts.json` manifests with relative artifact paths only.
- Keep `docs/REPORT-SCHEMA.md` aligned with any user-visible changes to non-terminal report filenames or top-level JSON keys.
- Keep README, architecture docs, and agent context aligned with the actual code paths and commands that work today.

## Critical constraints

- Do not refresh validation goldens while `LATENT_INSPECTOR_MODEL_BACKEND=stub` is active.
- Planned models may appear in reports under stub-backed development flows, but they must not be presented as ONNX-ready.
- Model artifact downloads must retain SHA-256 verification and companion-file handling.
- Model artifact downloads must retain bounded retry/backoff for transient request and stream failures.
- Report bundle artifact paths must stay relative and must not escape the output directory.
- Never hide a failed metric computation behind a plausible default.

## Gotchas

- The TUI launches with demo data only when no image is provided. If an image is provided, it runs live analysis.
- `LATENT_INSPECTOR_MODEL_BACKEND=stub` forces synthetic outputs and intentionally downgrades validation to `unverified`.
- Some ONNX models require companion `.onnx_data` files; cache completeness is per-artifact-bundle, not per single file.
- The crate is reusable internally, but downstream callers should not assume the public Rust API is stable before `1.0`.

## Current state

- Strongest areas: analysis coverage, validation/reporting structure, CLI integration tests, artifact manifesting, and now CI-enforced non-TUI coverage plus bounded download retries
- Still missing: interactive TUI test coverage, stable library/API guarantees, and live integrations for the planned models
