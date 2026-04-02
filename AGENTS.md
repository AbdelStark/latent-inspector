# latent-inspector — Agent Development Guide

Fast Rust CLI for inspecting and comparing learned representations across
self-supervised vision models (DINOv2, I-JEPA, MAE, CLIP, SigLIP) via ONNX Runtime.

## Quick reference

| Task             | Command                        |
|------------------|--------------------------------|
| Build            | `cargo build`                  |
| Build (ONNX)     | `cargo build --features onnx-inference` |
| Test             | `cargo test`                   |
| Lint             | `cargo clippy -- -D warnings`  |
| Format           | `cargo fmt`                    |

## Architecture

```
src/
  main.rs          CLI dispatch (clap)
  lib.rs           Library root — re-exports all modules
  errors.rs        Typed error enums (Error, ModelError, AnalysisError, etc.)
  cli/             Subcommand implementations
  models/          Model registry, ONNX loading, cache, preprocessing
  extract/         Feature extraction from model outputs
  analysis/        Metrics: PCA, CKA, k-NN, rank, variance, attention, entropy
  validation/      Contract and parity validation against golden fixtures
  viz/             Output: terminal, JSON, HTML, PNG
  dataset/         Image loading and batch iteration
  tui/             Interactive terminal UI (ratatui)
```

## Ready models

| Model | Status |
|-------|--------|
| dinov2-vit-l14 | Ready — real ONNX inference, approved evidence |
| ijepa-vit-h14 | Ready — ONNX artifacts verified, reference evidence uses stub backend (needs ONNX refresh) |
| dinov3-vit-l14, mae-vit-l16, clip-vit-l14, siglip-so400m, vjepa2-vitl-fpc2-256 | Planned |

## Conventions

- **Error handling**: `thiserror` enums per module, propagate with `?`, no `unwrap` in library code
- **Naming**: `snake_case` functions, `PascalCase` types, `SCREAMING_SNAKE` constants
- **Testing**: inline `#[cfg(test)]` for unit tests, `tests/` for integration tests
- **Commits**: `type(scope): description` — e.g. `fix(analysis): correct CKA normalization`
