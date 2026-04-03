# Implementation Plan: EUPE Model Support

**Branch**: `002-eupe-support` | **Date**: 2026-04-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-eupe-support/spec.md`

## Summary

Add EUPE ViT-B/16 (86M params, 768-dim, 196 patches, CLS token) as a ready model. This involves exporting from `facebook/EUPE-ViT-B` to ONNX, uploading to HuggingFace Hub, adding a registry entry, generating validation fixtures, and updating documentation. EUPE is the first model with a CLS token alongside DINOv2, enabling the first cross-model CLS cosine similarity comparison.

## Technical Context

**Language/Version**: Rust 2021 edition (stable toolchain)
**Primary Dependencies**: ort 2.0.0-rc.12 (ONNX Runtime), ndarray 0.16, image 0.25, clap 4
**Storage**: File-based cache at `~/.cache/latent-inspector/` (macOS: `~/Library/Caches/`)
**Testing**: `cargo test` (unit + integration), validation golden fixtures
**Target Platform**: macOS, Linux (aarch64, x86_64)
**Project Type**: CLI tool + library
**ML/Analysis Scope**: EUPE ViT-B/16 — multi-teacher distilled encoder, 768-dim, 196 patches, CLS token
**Performance Goals**: < 1s inference on laptop CPU (86M params, smaller than existing models)
**Constraints**: ONNX must load with ort 2.0.0-rc.12 (opset 14, onnxsim simplified, external data format)
**Validation Strategy**: Golden reference fixtures from ONNX Runtime inference, parity checks
**User-Facing Outputs**: All existing formats (terminal, JSON, HTML, PNG) — no new formats needed
**Scale/Scope**: Single model addition, no architectural changes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] ML Value: EUPE is a multi-teacher distilled encoder — comparing it against single-objective models (DINOv2, I-JEPA) reveals how distillation shapes representation geometry. First CLS-to-CLS pair.
- [x] Technical Soundness: ONNX export verified against PyTorch reference. Validation fixtures will be generated from live ONNX Runtime inference. SHA-256 checksums pinned.
- [x] Educational Surface: README updated with provenance, paper citation, and distillation method explanation. Case study comparison with DINOv2 demonstrates CLS cosine similarity for the first time.
- [x] Production Readiness: Tests updated for new model counts, validation golden fixtures generated, CI passes.
- [x] Helpful UX: No UX changes needed — EUPE fits the existing `PatchAndClsSequence` tensor role. All commands work automatically.
- [x] Exceptions Tracked: None — this is a straightforward model addition following the established pattern.

## Project Structure

### Documentation (this feature)

```text
specs/002-eupe-support/
├── plan.md              # This file
├── research.md          # Phase 0: ONNX export research
├── data-model.md        # Phase 1: registry entry schema
├── quickstart.md        # Phase 1: how to use EUPE
└── checklists/
    └── requirements.md  # Quality checklist
```

### Source Code (repository root)

```text
src/models/registry.rs      # Add EUPE RegistryEntry + SSLMethod::EUPE
src/models/loader.rs         # No changes (standard image path, not video)
src/models/preprocess.rs     # No changes (224px, ImageNet norm — same as DINOv2)

tests/fixtures/validation/
  manifest.json              # Add vjepa2 entry
  eupe-vit-b16.contract.json # New: preprocessing + tensor contract
  eupe-vit-b16.reference.json # New: golden reference from ONNX Runtime

tests/integration_test.rs    # Update ready model count assertions
tests/models_cli.rs          # Update model count assertions

README.md                    # Add to models table, provenance table, "Why this exists"
AGENTS.md                    # Add to ready models list
CHANGELOG.md                 # Document addition
```

**Structure Decision**: No new modules or files in `src/`. EUPE is a standard image model with CLS — it slots directly into the existing registry/loader/analysis pipeline. Only registry metadata, test assertions, documentation, and validation fixtures change.

## Complexity Tracking

No constitutional violations. This is a model addition following the established V-JEPA 2 pattern.
