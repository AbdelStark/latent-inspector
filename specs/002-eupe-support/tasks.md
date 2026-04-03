# Tasks: EUPE Model Support

**Input**: Design documents from `/specs/002-eupe-support/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Validation**: Includes ONNX export verification, golden reference fixtures, test assertion updates, and documentation updates per constitution requirements.

**Organization**: Tasks follow the established V-JEPA 2 pattern. Most work is in Phase 2 (ONNX export + registry) since EUPE is a standard image model that requires no code changes to the analysis pipeline.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1=compare, US2=inspect, US3=validation)
- Exact file paths in descriptions

---

## Phase 1: Setup (ONNX Export & HuggingFace Upload)

**Purpose**: Export EUPE to ONNX and host on HuggingFace — this is the prerequisite for all other work.

- [ ] T001 Clone facebook/EUPE repo and install dependencies in a temporary Python venv
- [ ] T002 Export EUPE ViT-B/16 to ONNX via `forward_features()` wrapper that concatenates CLS + patches into `[1, 197, 768]` output, using TorchScript legacy exporter at opset 14
- [ ] T003 Simplify ONNX graph with onnxsim and convert to external data format (`model.onnx` + `model.onnx_data`)
- [ ] T004 Verify ONNX output against PyTorch reference (max diff < 0.01) and confirm ort 2.0.0-rc.12 can load the model
- [ ] T005 Upload ONNX artifacts to `abdelstark/eupe-vit-b16-onnx` on HuggingFace Hub
- [ ] T006 Compute SHA-256 hashes of uploaded artifacts

**Checkpoint**: ONNX model hosted, verified, and SHA-256 hashes known.

---

## Phase 2: Foundational (Registry & Validation Fixtures)

**Purpose**: Wire EUPE into the model registry and generate golden reference evidence.

- [ ] T007 Add `SSLMethod::EUPE` variant to enum and Display impl in `src/models/registry.rs`
- [ ] T008 Add EUPE `RegistryEntry` to `registry()` in `src/models/registry.rs` with provenance comments (paper citation, checkpoint source, ONNX export method)
- [ ] T009 Add `eupe-vit-b16` to fixture manifest in `tests/fixtures/validation/manifest.json`
- [ ] T010 Create `tests/fixtures/validation/eupe-vit-b16.contract.json` with preprocessing and tensor contracts
- [ ] T011 Run `validate --model eupe-vit-b16 --refresh-goldens` to generate `tests/fixtures/validation/eupe-vit-b16.reference.json` from ONNX Runtime
- [ ] T012 Update model count assertions in `src/models/registry.rs` tests (model_names count, ready_model_names list)
- [ ] T013 [P] Update model count assertions in `src/models/inventory.rs` tests (ready_models, artifacts.total, evidence.approved)
- [ ] T014 [P] Update model count assertions in `src/viz/html.rs` tests (evidence.unverified, evidence.approved)
- [ ] T015 [P] Update model count assertions in `tests/integration_test.rs` (model_names, ready list)
- [ ] T016 [P] Update model count assertions in `tests/models_cli.rs` (total_models, ready_models, artifacts.total, needs_download, planned)

**Checkpoint**: `cargo test` passes, `validate --model eupe-vit-b16` reports "validated".

---

## Phase 3: User Story 1 — Cross-model comparison with CLS pair (Priority: P1)

**Goal**: `compare photo.jpg --models dinov2-vit-l14,eupe-vit-b16` works end-to-end with CLS cosine similarity — the first fully computable CLS pair.

**Independent Test**: Run compare with DINOv2 + EUPE on elephant sample image; verify CLS cosine, CKA, and k-NN are all non-null in JSON output.

### Implementation

- [ ] T017 [US1] Run `compare docs/assets/img/samples/elephant_sample_image.jpg --models dinov2-vit-l14,eupe-vit-b16` and capture terminal output to verify CLS cosine is computed
- [ ] T018 [US1] Run `compare --format json --output tmp/eupe-compare/` and verify `cls_cosine_sim` is non-null for the DINOv2↔EUPE pair in `compare.json`
- [ ] T019 [US1] Run 4-model compare (all ready models) and verify all pairwise metrics are correct

**Checkpoint**: First CLS-to-CLS pair works. Compare output shows real CLS cosine similarity.

---

## Phase 4: User Story 2 — Single model inspection (Priority: P2)

**Goal**: `inspect photo.jpg --model eupe-vit-b16` produces full representation analysis.

**Independent Test**: Run inspect and verify 196 patches, 768 embed dim, CLS L2 norm present.

### Implementation

- [ ] T020 [US2] Run `inspect docs/assets/img/samples/elephant_sample_image.jpg --model eupe-vit-b16` and verify all metrics present (including CLS L2 norm)
- [ ] T021 [US2] Run `inspect --format html --output tmp/eupe-inspect/` and verify HTML report generation with PCA projection and variance chart

**Checkpoint**: Inspect works with full metrics including CLS.

---

## Phase 5: User Story 3 — Validation and provenance (Priority: P3)

**Goal**: Validation passes and documentation is complete.

**Independent Test**: `validate --model eupe-vit-b16` reports "validated".

### Implementation

- [ ] T022 [US3] Verify `validate --model eupe-vit-b16` reports "validated", backend=onnx-runtime, 0 drifted signals
- [ ] T023 [US3] Verify `models --verbose` shows EUPE with correct status, evidence, and cache details

**Checkpoint**: Validation fully operational.

---

## Phase 6: Polish & Documentation

**Purpose**: Update all documentation and cross-cutting concerns.

- [ ] T024 [P] Update README.md: add EUPE to supported models table (Ready status)
- [ ] T025 [P] Update README.md: add EUPE to model provenance table (paper, checkpoint, ONNX source)
- [ ] T026 [P] Update README.md "Why this exists" section: add EUPE bullet about multi-teacher distillation
- [ ] T027 [P] Update AGENTS.md: add eupe-vit-b16 to ready models list
- [ ] T028 [P] Update CHANGELOG.md: document EUPE addition
- [ ] T029 Run `cargo test && cargo clippy -- -D warnings && cargo fmt -- --check` — zero failures, zero warnings
- [ ] T030 Commit and push all changes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (ONNX Export)**: No dependencies — start immediately
- **Phase 2 (Registry)**: Depends on Phase 1 (needs SHA-256 hashes from uploaded artifacts)
- **Phase 3-5 (User Stories)**: Depend on Phase 2 (needs registry entry and validation fixtures)
- **Phase 6 (Polish)**: Depends on all user stories passing

### Within Each Phase

- T001-T006 are sequential (each depends on previous)
- T007-T011 are sequential (registry → fixtures → validate)
- T012-T016 are parallel ([P] — different files)
- T024-T028 are parallel ([P] — different files)

### Parallel Opportunities

```
Phase 2 parallel:
  T012 (registry tests) || T013 (inventory tests) || T014 (html tests) || T015 (integration tests) || T016 (models_cli tests)

Phase 6 parallel:
  T024 (README models) || T025 (README provenance) || T026 (README why) || T027 (AGENTS) || T028 (CHANGELOG)
```

---

## Implementation Strategy

### MVP (Phase 1-2 only)

1. Export ONNX, upload to HuggingFace, add registry entry
2. Generate validation fixtures
3. Update test assertions
4. **Result**: `cargo test` passes, model is registered and downloadable

### Full Delivery (Phase 1-6)

1. MVP above
2. Manual verification of compare, inspect, validate commands
3. Documentation updates
4. Final test pass and commit

### Estimated scope

This is a model addition following the established V-JEPA 2 pattern. The analysis pipeline, CLI commands, and output formats require no code changes — EUPE fits the existing `PatchAndClsSequence` tensor role.

---

## Notes

- EUPE is 86M params (~350 MB ONNX) — smallest model in the registry
- embed_dim=768 is the first sub-1024 model — verify no hardcoded assumptions
- 196 patches (14x14 at 224px) differs from DINOv2's 256 — CKA/k-NN use min count
- CLS token is present — enables first cross-model CLS cosine pair with DINOv2
- FAIR Research License (non-commercial) — document in README
