# Contributing

## Scope

This repository is a Rust CLI/TUI application. Contributions are expected to preserve three things together:

1. Correct analysis behavior
2. Honest validation/reporting surfaces
3. Accurate repository documentation

## Prerequisites

- Rust `1.75.0` or newer
- Network access if you want to exercise live model downloads
- Enough disk space for cached ONNX artifacts

## Setup

```bash
cargo build --release
```

For development without downloading models:

```bash
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- models
```

## Verification commands

These are the baseline checks for every change:

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo llvm-cov --workspace --ignore-filename-regex '(^|/)src/tui/|(^|/)src/cli/tui.rs$' --fail-under-lines 85 --fail-under-functions 80 --summary-only
cargo build --release
```

Use `make all` if you want the repository-default aggregate target.
Use `make coverage-ci` if you want the same coverage gate CI enforces.

## Development workflow

- Use the stub backend for fast iteration or for exercising planned-model report flows:

```bash
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- compare docs/assets/img/samples/elephant_sample_image.jpg --models dinov2-vit-l14,ijepa-vit-h14
```

- Use the live path when changing preprocessing, loader behavior, validation semantics, or artifact management.
- Keep report outputs deterministic where tests depend on them.
- Add or update tests for every bug fix that affects observable behavior.

## Validation goldens

Validation artifacts under `tests/fixtures/validation/` are a release-trust surface, not scratch data.

- Only refresh goldens after a verified change to a model artifact, preprocessing contract, tensor contract, or parity tolerance.
- Never refresh goldens while `LATENT_INSPECTOR_MODEL_BACKEND=stub` is active.
- When goldens change, update the docs that explain why the reference moved.

Refresh flow:

```bash
cargo run -- validate --model dinov2-vit-l14 --refresh-goldens
```

Repeat for each affected model, then inspect the fixture diffs carefully.

## Documentation expectations

If you change behavior, also update the docs that describe it:

- `README.md` for user-facing workflows and status
- `docs/ARCHITECTURE.md` for system behavior, invariants, and operational notes
- `docs/REPORT-SCHEMA.md` for non-terminal report filenames and JSON/manifest compatibility
- `AGENTS.md` for autonomous contributors
- `CHANGELOG.md` for user-visible changes

## Pull request checklist

- The change has a clear behavioral reason, not just a mechanical cleanup
- Tests cover the changed behavior or failure mode
- Validation/report outputs remain truthful
- Commands in the docs still work as written
- `CHANGELOG.md` is updated for user-visible changes

## Reporting issues

Use the GitHub issue tracker: <https://github.com/AbdelStark/latent-inspector/issues>
