---
name: rust-development
description: Rust project setup, module organization, error handling, Cargo workflows, and idiomatic patterns for the latent-inspector CLI. Activate when creating new modules, setting up Cargo.toml, defining error types, or structuring the project.
prerequisites: Rust stable toolchain, cargo
---

# Rust Development

<purpose>
Covers Rust-specific development patterns for latent-inspector: project initialization, module hierarchy, error handling with thiserror, Cargo.toml management, and idiomatic Rust patterns for a CLI application backed by numerical computation.
</purpose>

<context>
— Binary crate with clap derive-based CLI
— Module tree: cli/, models/, extract/, analysis/, viz/, dataset/
— Error handling: thiserror-derived enums per module, `?` propagation
— No async runtime — all sync with rayon for data parallelism
— Target: single binary, no dynamic linking except ONNX Runtime (bundled by ort)
</context>

<procedure>
### Initialize the project (Phase 1, first step)
1. `cargo init --name latent-inspector`
2. Set up Cargo.toml with all dependencies from SPECIFICATION.md
3. Configure `[profile.release]` with `lto = true`, `codegen-units = 1` for optimized binary
4. Create `rustfmt.toml`: `max_width = 100`, `edition = "2021"`
5. Create module directories: `mkdir -p src/{cli,models,extract,analysis,viz,dataset}`
6. Create mod.rs for each with public re-exports
7. Wire modules in main.rs
8. Verify: `cargo check`

### Add a new module
1. Create `src/<module>/mod.rs` with module-level doc comment
2. Define the module's Error enum:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum Error {
       #[error("description: {0}")]
       VariantName(String),
       #[error(transparent)]
       Io(#[from] std::io::Error),
   }
   ```
3. Define public types and traits
4. Implement with `pub(crate)` visibility by default
5. Add `pub mod <name>;` to parent
6. Add inline tests: `#[cfg(test)] mod tests { ... }`

### Error handling cascade
1. Each module defines its own `Error` enum
2. Top-level `src/error.rs` aggregates with `#[from]` conversions
3. `main()` uses `anyhow::Result` or prints user-friendly error and exits with code 1
4. Never expose internal error details to CLI users — map to human-readable messages
</procedure>

<patterns>
<do>
  — Use `#[derive(Debug, Clone)]` on all data structs.
  — Use `impl Display` for types shown to users (model names, metrics).
  — Use `From` trait for error conversions between modules.
  — Use `cfg(feature = ...)` for optional heavy deps like ndarray-linalg.
  — Use workspace-level `[lints]` in Cargo.toml for clippy configuration.
</do>
<dont>
  — Don't use `Box<dyn Error>` — use concrete thiserror types for better error messages.
  — Don't use `lazy_static` — use `std::sync::OnceLock` (stable since Rust 1.70).
  — Don't put business logic in main.rs — it should only parse CLI and dispatch.
  — Don't use `String` where `&str` or an enum suffices (model names → enum).
</dont>
</patterns>

<examples>
Example: Cargo.toml structure
```toml
[package]
name = "latent-inspector"
version = "0.1.0"
edition = "2021"
description = "Fast CLI for inspecting SSL vision model representations"
license = "MIT OR Apache-2.0"

[dependencies]
ort = { version = "2", features = ["load-dynamic"] }
ndarray = "0.16"
image = "0.25"
rayon = "1.10"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ratatui = "0.29"
crossterm = "0.28"
indicatif = "0.17"
tracing = "0.1"
tracing-subscriber = "0.3"
thiserror = "2.0"
dirs = "6"
reqwest = { version = "0.12", features = ["blocking"] }

[dev-dependencies]
approx = "0.5"
tempfile = "3"

[features]
default = []
full-svd = ["ndarray-linalg", "openblas-src"]

[profile.release]
lto = true
codegen-units = 1
strip = true
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| `ort` link errors | ONNX Runtime lib not found | Check `ort` feature flags; `load-dynamic` bundles it |
| Circular module deps | Module A uses B, B uses A | Extract shared types to a common module |
| Slow debug builds | Default dev profile | Add `[profile.dev.package."*"] opt-level = 2` for deps |
</troubleshooting>

<references>
— Cargo.toml: Project dependencies and features
— SPECIFICATION.md: Architecture diagram, module responsibilities
— IMPLEMENTATION_PLAN.md: Phase 1 setup steps
</references>
