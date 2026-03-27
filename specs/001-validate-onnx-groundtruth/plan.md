# Implementation Plan: Model Validation Evidence

**Branch**: `001-validate-onnx-groundtruth` | **Date**: 2026-03-27 | **Spec**: [/Users/abdel/dev/me/world-models/latent-inspector/specs/001-validate-onnx-groundtruth/spec.md](/Users/abdel/dev/me/world-models/latent-inspector/specs/001-validate-onnx-groundtruth/spec.md)
**Input**: Feature specification from `/specs/001-validate-onnx-groundtruth/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add a first-class validation workflow for model integrations that verifies
preprocessing and exported tensor semantics against explicit source-model
contracts, compares selected outputs against trusted reference evidence, stores
golden regression artifacts, and surfaces trust explanations consistently in
terminal, JSON, and HTML reports.

## Technical Context

**Language/Version**: Rust 2021  
**Primary Dependencies**: clap 4, ort 2.0.0-rc.12, ndarray 0.16, image 0.25, serde/serde_json, tracing, approx, tempfile  
**Storage**: Checked-in filesystem fixtures and golden artifacts in the repo, plus cached ONNX models under the user cache directory  
**Testing**: `cargo test`, unit tests, integration tests, fixture-backed golden regression tests, report serialization/snapshot assertions  
**Target Platform**: macOS and Linux CLI environments using ONNX Runtime, CPU-first with optional accelerator support inherited from existing runtime configuration  
**Project Type**: Rust CLI + library crate  
**ML/Analysis Scope**: Supported ViT-family exports (DINOv2, MAE, I-JEPA, CLIP, SigLIP); preprocessing parity, tensor-semantic validation, reference-output parity, and trust summaries for compare/inspect flows  
**Performance Goals**: Validation of one model on the standard fixture set completes in under 60 seconds without fresh downloads on a developer machine; report rendering adds negligible overhead relative to inference  
**Constraints**: Default regression runs MUST not depend on network access or a live Python environment; validation evidence MUST be reproducible and versioned; existing compare/inspect usage remains backward-compatible; full reference refresh is opt-in maintainer work  
**Validation Strategy**: Registry-backed contract assertions, ONNX tensor name/shape verification, fixture-based parity comparisons against trusted reference outputs, golden regression tests, and structured validation summaries rendered in every supported report surface  
**User-Facing Outputs**: A dedicated `validate` CLI workflow, validation summaries embedded into terminal/JSON/HTML report outputs, and quickstart/docs updates for maintainers and reviewers  
**Scale/Scope**: Five currently supported model families, multiple export variants over time, and a shared fixture set sized for CI-safe regression coverage

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] ML Value: The feature directly improves trust in model inspection and
      comparison results by validating that exported integrations still match
      the underlying models they claim to represent.
- [x] Technical Soundness: Research and design define explicit model contracts,
      parity evidence, tolerance handling, and regression artifacts rather than
      relying on implicit tensor-shape heuristics.
- [x] Educational Surface: The feature includes structured trust summaries and
      quickstart guidance so users understand what was validated and what
      remaining caveats apply.
- [x] Production Readiness: The design includes offline-safe regression tests,
      golden artifacts, model/export traceability, and failure states suitable
      for release gating.
- [x] Helpful UX: The planned CLI command and report payloads provide concise
      pass/fail/confidence summaries across terminal, JSON, and HTML outputs.
- [x] Exceptions Tracked: No constitutional exceptions are required for this
      plan.

Post-design re-check: PASS. `research.md`, `data-model.md`, `contracts/`, and
`quickstart.md` cover the validation workflow, evidence model, and user-facing
trust surfaces required by the constitution.

## Project Structure

### Documentation (this feature)

```text
specs/001-validate-onnx-groundtruth/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── validate-cli.md
│   └── validation-report.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── analysis/
├── cli/
│   ├── compare.rs
│   ├── inspect.rs
│   ├── mod.rs
│   └── validate.rs
├── dataset/
├── extract/
├── models/
│   ├── cache.rs
│   ├── loader.rs
│   ├── mod.rs
│   ├── preprocess.rs
│   └── registry.rs
├── validation/
│   ├── fixtures.rs
│   ├── mod.rs
│   ├── parity.rs
│   ├── report.rs
│   └── semantics.rs
├── viz/
│   ├── html.rs
│   ├── json.rs
│   ├── mod.rs
│   ├── png.rs
│   └── terminal.rs
├── errors.rs
├── lib.rs
└── main.rs

tests/
├── integration_test.rs
├── validation_cli.rs
└── validation_golden.rs

tests/fixtures/
└── validation/
```

**Structure Decision**: Keep the project as a single Rust crate and add a
dedicated `validation` domain module plus a `validate` CLI entry point. Reuse
the existing `models` registry/loader path as the source of truth for model
contracts, and extend the current `viz` renderers with a shared validation
summary instead of building a separate reporting subsystem.

## Complexity Tracking

No constitutional violations identified.
