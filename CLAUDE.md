<identity>
latent-inspector — Fast Rust CLI for inspecting and comparing learned representations across self-supervised vision models (DINOv2, DINOv3, MAE, I-JEPA, CLIP, SigLIP) via ONNX Runtime.
</identity>

<status>
PROJECT STATUS: Rust CLI and library implementation are present. Phase 1 is the active runtime scope: `dinov2-vit-l14` is the only ready model, validation evidence is implemented, and compare/inspect/drift reporting are wired through terminal, JSON, HTML, and PNG surfaces. Planned models remain registered but intentionally unavailable until their integrations are validated.
</status>

<stack>
| Layer       | Technology       | Version  | Notes                                          |
|-------------|------------------|----------|-------------------------------------------------|
| Language    | Rust             | 2021 ed. | Stable toolchain                                |
| Runtime     | ONNX Runtime     | 2.x      | Via `ort` crate, CPU + GPU auto-detect          |
| Arrays      | ndarray          | 0.16     | N-dimensional arrays for analysis pipeline      |
| Image I/O   | image            | 0.25     | Loading, resizing, format conversion            |
| Parallelism | rayon            | 1.10     | Parallel model inference                        |
| CLI         | clap             | 4.x      | Derive-based argument parsing                   |
| Serialization | serde + serde_json | 1.x  | JSON output, config                             |
| Terminal UI | ratatui          | 0.29     | Rich terminal rendering                         |
| Terminal backend | crossterm   | 0.28     | Cross-platform terminal control                 |
| Progress    | indicatif        | 0.17     | Download/batch progress bars                    |
| HTTP        | reqwest          | 0.12     | Model downloads from HuggingFace, blocking mode |
| Errors      | thiserror        | 2.0      | Error type derivation                           |
| Logging     | tracing          | 0.1      | Structured logging                              |
| Directories | dirs             | 6        | Platform cache dir (`~/.cache/latent-inspector/`) |
| Package mgr | cargo            | —        | NEVER use npm/yarn/bun. This is a Rust project. |
</stack>

<structure>
Target architecture (to be created):

src/
├── main.rs           # Entry point, CLI dispatch
├── cli/              # Command definitions (clap) [agent: create/modify]
│   ├── mod.rs
│   ├── compare.rs    # `compare` subcommand
│   ├── inspect.rs    # `inspect` subcommand
│   ├── neighbors.rs  # `neighbors` subcommand
│   ├── similarity.rs # `similarity` subcommand
│   ├── drift.rs      # `drift` subcommand
│   └── models.rs     # `models` subcommand
├── models/           # Model registry, ONNX loading, caching [agent: create/modify]
│   ├── mod.rs
│   ├── registry.rs   # Model metadata, download URLs, SHA-256 hashes
│   ├── loader.rs     # ONNX session creation, intermediate output extraction
│   ├── cache.rs      # Download, verify, cache management
│   └── preprocess.rs # Model-specific image normalization (mean/std)
├── extract/          # Feature extraction [agent: create/modify]
│   ├── mod.rs
│   └── features.rs   # CLS token, patch tokens, attention weights extraction
├── analysis/         # Metric computation [agent: create/modify]
│   ├── mod.rs
│   ├── pca.rs        # PCA via power method (default) or full SVD (optional)
│   ├── rank.rs       # Representation rank (singular value thresholding)
│   ├── variance.rs   # Feature variance spectrum
│   ├── attention.rs  # Gini coefficient, attention concentration
│   ├── entropy.rs    # Patch entropy via k-means clustering
│   ├── cka.rs        # Centered Kernel Alignment (cross-model)
│   ├── knn.rs        # k-NN overlap, neighbor retrieval
│   └── correspondence.rs # Patch correspondence via Hungarian matching
├── viz/              # Output rendering [agent: create/modify]
│   ├── mod.rs
│   ├── terminal.rs   # Unicode blocks + ANSI color rendering
│   ├── png.rs        # Attention overlays, PCA RGB, heatmaps
│   ├── json.rs       # Structured JSON metrics
│   └── html.rs       # Interactive HTML report
└── dataset/          # Image loading, batching [agent: create/modify]
    ├── mod.rs
    └── loader.rs     # Directory scanning, batch iteration, caching

Cargo.toml            # [agent: create/modify]
rustfmt.toml          # [agent: create/modify]
clippy.toml           # [agent: create/modify]
.github/workflows/    # CI [agent: create with care]
tests/                # Integration tests [agent: create/modify]
</structure>

<commands>
| Task             | Command                        | Notes                                    |
|------------------|--------------------------------|------------------------------------------|
| Build            | `cargo build`                  | Debug build                              |
| Build (release)  | `cargo build --release`        | Optimized, use for benchmarks            |
| Run              | `cargo run -- <args>`          | Pass CLI args after `--`                 |
| Test (all)       | `cargo test`                   | Unit + integration tests                 |
| Test (specific)  | `cargo test <name>`            | Filter by test name                      |
| Lint             | `cargo clippy -- -D warnings`  | Treat warnings as errors                 |
| Format           | `cargo fmt`                    | Apply rustfmt                            |
| Format (check)   | `cargo fmt -- --check`         | CI check, no modifications               |
| Type check       | `cargo check`                  | Faster than full build                   |
| Doc              | `cargo doc --open`             | Generate and view docs                   |
| Clean            | `cargo clean`                  | Remove build artifacts                   |
</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables/modules, PascalCase for types/structs/enums, SCREAMING_SNAKE for constants.
  Files: snake_case.rs — one module per file, mod.rs for re-exports.
  Imports: Group — std → external crates → crate-internal. Use `use crate::` for internal imports.
  Error handling: Use thiserror for defining error types. Each module has its own Error enum. Propagate with `?`. Never panic in library code. `unwrap()` only in tests.
  Visibility: Minimize pub surface. Use `pub(crate)` for internal APIs. Only `pub` what's in the CLI interface.
  Documentation: `///` doc comments on all public items. Include examples for non-obvious functions.
</code_style>

<patterns>
  <do>
    — Use `Result<T, Error>` for all fallible operations with module-specific error types.
    — Use `rayon` for parallelizing independent model inference — never manual thread spawning.
    — Use `ndarray` for all numerical computation — no raw Vec<Vec<f32>> for matrices.
    — Use builder pattern for complex structs (ModelConfig, AnalysisOptions).
    — Use `tracing` for structured logging — `tracing::info!`, `tracing::debug!`, not `println!`.
    — Use `indicatif` ProgressBar for any operation >1s (downloads, batch processing).
    — Keep all model-specific logic (normalization params, layer names) in the registry, not scattered.
    — Write unit tests inline (`#[cfg(test)] mod tests`) in each module.
    — Use `assert_relative_eq!` from approx crate for floating-point comparisons in tests.
  </do>
  <dont>
    — Don't use `println!` for user output — use the viz module's renderers.
    — Don't use `unwrap()`/`expect()` in non-test code — propagate errors with `?`.
    — Don't hardcode model paths — always resolve via the cache/registry system.
    — Don't use `ndarray-linalg` by default — use power method for PCA. Full SVD is optional feature.
    — Don't block the main thread on downloads — show progress.
    — Don't store large tensors (attention weights) longer than needed — compute metrics and drop.
    — Don't add Python bindings or FFI — this is a pure Rust CLI tool.
  </dont>
</patterns>

<commit_conventions>
  Format: type(scope): description
  Types: feat, fix, refactor, test, docs, chore, perf
  Scopes: cli, models, extract, analysis, viz, dataset, ci
  Examples:
    feat(models): add DINOv2 ONNX model loading and caching
    fix(analysis): correct CKA normalization for unequal dimensions
    test(pca): add property-based tests for PCA eigenvalue ordering
</commit_conventions>
</conventions>

<workflows>
<new_module>
  1. Create module directory under src/ with mod.rs
  2. Add `pub mod <name>;` to parent module or main.rs
  3. Define module-specific Error enum with thiserror
  4. Implement core functionality with unit tests inline
  5. Run `cargo test` — all must pass
  6. Run `cargo clippy -- -D warnings` — zero warnings
  7. Run `cargo fmt` — apply formatting
  8. Commit: feat(scope): description
</new_module>

<new_feature>
  1. Check IMPLEMENTATION_PLAN.md for current phase and priorities
  2. Read SPECIFICATION.md for interface contracts and expected behavior
  3. Implement in the appropriate module
  4. Write tests: unit tests inline, integration tests in tests/
  5. Run full validation: `cargo test && cargo clippy -- -D warnings && cargo fmt -- --check`
  6. Self-review: no unwrap() in lib code, no hardcoded paths, errors propagated
  7. Commit with conventional format
</new_feature>

<add_model>
  1. Add model metadata to registry (name, URL, SHA-256, arch, params)
  2. Define preprocessing params (input size, mean, std normalization)
  3. Map ONNX output names to ModelOutput fields (CLS, patches, attention)
  4. Test: download, load, extract features from a test image
  5. Verify output shapes match specification
  6. Add model to CLI --models enum
</add_model>

<debug_inference>
  1. Check model is downloaded and cached: `~/.cache/latent-inspector/<model>.onnx`
  2. Verify ONNX model loads: test with `ort::Session::builder()`
  3. Print input/output tensor names and shapes from the ONNX graph
  4. Compare expected vs actual output dimensions
  5. Check preprocessing: correct resize, normalize with right mean/std
  6. If attention weights missing: inspect ONNX graph for intermediate nodes
</debug_inference>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify or create under any circumstances:
  — .env, .env.* (credentials, secrets)
  — Any file containing API keys or tokens
  — ~/.cache/latent-inspector/ content directly (use the cache module)
</forbidden>

<gated>
  Modify ONLY with explicit human approval:
  — Cargo.toml dependency changes (adding/removing/upgrading crates)
  — .github/workflows/ CI configuration
  — SPECIFICATION.md, IMPLEMENTATION_PLAN.md (project planning docs)
  — Any public API surface changes to the CLI interface
  — Build scripts (build.rs) if created
</gated>

<safety_checks>
  Before ANY destructive operation:
  1. State what you're about to do and what it affects
  2. State what could go wrong
  3. Wait for confirmation

  Specific cautions:
  — Model downloads are large (300MB-2.4GB). Confirm before triggering.
  — ONNX graph surgery (modifying models to expose intermediate layers) is irreversible on the file.
  — `cargo clean` removes all build artifacts including cached deps.
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>
  | Symptom                                     | Cause                          | Fix                                              |
  |---------------------------------------------|--------------------------------|--------------------------------------------------|
  | `ort` fails to link ONNX Runtime            | Missing system lib             | `ort` bundles ONNX RT by default; check features  |
  | ONNX model output names don't match         | Model export version mismatch  | Inspect graph with `ort` debug, update name map   |
  | PCA eigenvalues not sorted                  | Power method convergence       | Increase iterations or switch to full SVD feature  |
  | Attention weights not in ONNX graph         | Model wasn't exported with them| Need ONNX graph surgery to expose attention nodes  |
  | `ndarray` shape mismatch                    | Wrong reshape dimensions       | Print actual shapes, compare with ModelInfo params |
  | Download fails mid-file                     | Network interruption           | Partial `.download-part` files resume automatically when the host supports HTTP Range; otherwise the cache restarts cleanly |
  | Terminal rendering garbled                   | Terminal doesn't support Unicode| Detect capabilities, fall back to ASCII           |
</known_issues>

<recovery_patterns>
  When stuck:
  1. Read the full error message — Rust errors are precise and usually contain the fix
  2. Run `cargo check` first — faster feedback than full build
  3. Check that `Cargo.toml` dependencies and features are correct
  4. For ONNX issues: print model input/output metadata with `session.inputs()` / `session.outputs()`
  5. For ndarray issues: print `.shape()` and `.ndim()` at each transform step
  6. If still stuck, state the problem clearly and ask for help
</recovery_patterns>
</troubleshooting>

<key_references>
  — SPECIFICATION.md: Model interface (ModelOutput struct), analysis metrics, CLI commands
  — IMPLEMENTATION_PLAN.md: Phase ordering, technical decisions (ONNX RT vs Tract, power method PCA)
  — RESOURCES.md: Model papers, crate versions, competitive landscape
  — README.md: Expected CLI UX, example outputs, supported models table
</key_references>

<decisions>
  2026-03-26 Use ONNX Runtime (ort) over Tract — Faster, GPU support, wider ViT compatibility — Tract (pure Rust but slower, less model support)
  2026-03-26 Power method for PCA by default — No system deps (LAPACK/OpenBLAS), works everywhere — ndarray-linalg (optional feature for full SVD)
  2026-03-26 Pure Rust analysis pipeline — No Python deps, single binary distribution — PyO3 (adds complexity), C FFI (unnecessary)
  2026-03-26 Common ModelOutput interface — All 6 models normalized to same struct for uniform analysis — Per-model custom output (would duplicate analysis code)
  2026-03-26 rayon for parallelism — Proven, ergonomic data parallelism for multi-model inference — tokio (async not needed for CPU-bound work), manual threads (error-prone)
</decisions>

<skills>
  Modular skills in .codex/skills/ (symlinked at .claude/skills/ and .agents/skills/).
  Load a skill when entering its domain.

  Available skills:
  — _index.md: Skill registry and discovery metadata
  — rust-development.md: Rust project setup, module patterns, error handling, Cargo workflows
  — onnx-inference.md: ONNX Runtime integration, model loading, feature extraction, graph inspection
  — analysis-pipeline.md: PCA, CKA, k-NN, rank, variance, Gini — numerical analysis patterns
  — testing.md: Test strategy, unit/integration/property-based testing, floating-point assertions
  — visualization.md: Terminal rendering, PNG export, JSON output, HTML report generation
</skills>
