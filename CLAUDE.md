# CLAUDE.md -- Development Context for latent-inspector

## Project

CLI tool for inspecting and comparing self-supervised vision model (SSL) representations. Rust codebase, ONNX Runtime inference, no Python dependency.

## Build & Test

```bash
cargo build --release
cargo test
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings

# Full CI pipeline
make all

# Development without downloading models (~1-2 GB each)
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo test
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- compare docs/assets/img/samples/elephant_sample_image.jpg --models dinov2-vit-l14,ijepa-vit-h14

# Coverage (excludes TUI surface)
cargo llvm-cov --workspace \
  --ignore-filename-regex '(^|/)src/tui/|(^|/)src/cli/tui.rs$' \
  --fail-under-lines 85 --fail-under-functions 80 --summary-only
```

## Architecture

```
src/
  models/       Model registry, ONNX loading, preprocessing, download caching
  extract/      Feature extraction (CLS token, patch tokens, attention maps)
  analysis/     Representation metrics (PCA, CKA, k-NN, entropy, isotropy, coherence, etc.)
  viz/          Output rendering: terminal, JSON, HTML, PNG
  cli/          Subcommands: compare, inspect, neighbors, profile, similarity, drift, models, validate, tui
  tui/          Interactive terminal UI (ratatui)
  validation/   Contract checks, reference parity, golden fixtures
  dataset/      Image dataset loading and iteration
  errors.rs     Error types
```

## Key Conventions

- **Stub backend**: Set `LATENT_INSPECTOR_MODEL_BACKEND=stub` to bypass real ONNX inference. Produces deterministic synthetic outputs. Validation summaries downgrade to `unverified`.
- **All analysis metrics** live in `src/analysis/` as separate modules. Each metric function takes `ndarray::Array2<f32>` patch tokens and returns `Result<T, AnalysisError>`.
- **ModelMetrics** struct in `src/analysis/mod.rs` aggregates all per-model metrics. Any new metric must be added there and wired into `model_metrics_from_spectrum()`.
- **4 output formats**: terminal, JSON, HTML, PNG. Every new metric must be displayed in all formats.
- **TUI views**: dashboard, inspector, compare, spectrum, help. Each in `src/tui/views/`.
- **Validation**: Golden fixtures in `tests/fixtures/validation/`. Use `--refresh-goldens` after verified ONNX changes.
- **Models**: 4 ready (DINOv2, I-JEPA, V-JEPA 2, EUPE), 4 planned (DINOv3, MAE, CLIP, SigLIP). Registry in `src/models/registry.rs`.

## Adding a New Analysis Metric

1. Create `src/analysis/<metric>.rs` with the computation function
2. Add `pub mod <metric>` and re-export in `src/analysis/mod.rs`
3. Add field to `ModelMetrics` struct (use `Option<T>` if not always available)
4. Wire into `model_metrics_from_spectrum()` in `src/analysis/mod.rs`
5. Update display in: `src/viz/terminal.rs`, `src/viz/html.rs`, `src/viz/json.rs`
6. Update `src/cli/inspect.rs` terminal output
7. Update TUI views: `src/tui/views/inspector.rs`, `src/tui/views/compare.rs`, `src/tui/views/dashboard.rs`
8. Add `spatial_coherence` field to all `ModelMetrics` literals in test fixtures
9. Write tests in the metric module
10. Run `cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt -- --check`

## Testing

- Unit tests: inline `#[cfg(test)]` modules in each source file
- Integration tests: `tests/` directory (CLI-level tests with stub backend)
- All tests run with `LATENT_INSPECTOR_MODEL_BACKEND=stub` in CI
- Coverage thresholds: 85% lines, 80% functions (excluding TUI)
