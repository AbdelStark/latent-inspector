---

description: "Task list for implementing model validation evidence"
---

# Tasks: Model Validation Evidence

**Input**: Design documents from `/specs/001-validate-onnx-groundtruth/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Validation**: Include the tests, benchmarks, fixtures, docs/examples, and UX
validation work needed to satisfy the constitution. Do not omit this work just
because the prompt did not explicitly request it.

**Organization**: Tasks are grouped by user story to enable independent
implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Validation fixtures live under `tests/fixtures/validation/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare repo-level scaffolding for the validation workflow.

- [X] T001 Create Rust ignore rules for validation fixtures and generated outputs in `.gitignore`
- [X] T002 [P] Create the validation module entrypoint in `src/validation/mod.rs` and export it from `src/lib.rs`
- [X] T003 [P] Seed validation fixture documentation in `tests/fixtures/validation/README.md` and `tests/fixtures/validation/manifest.json`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared validation infrastructure that MUST exist before user story work.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Extend validation contract metadata for all supported models in `src/models/registry.rs`
- [X] T005 [P] Add validation status, summary, and error types in `src/validation/report.rs` and `src/errors.rs`
- [X] T006 [P] Implement fixture and golden artifact loading in `src/validation/fixtures.rs`
- [X] T007 Update loader output metadata for explicit tensor names, shapes, and CLS semantics in `src/models/loader.rs`
- [X] T008 Wire shared validation exports in `src/models/mod.rs` and `src/lib.rs`

**Checkpoint**: Foundation ready for command-level contract validation.

---

## Phase 3: User Story 1 - Validate Model Contracts (Priority: P1) 🎯 MVP

**Goal**: Add a maintainer-facing validation workflow that proves preprocessing
and consumed tensor semantics still match the source-model contract.

**Independent Test**: Run `cargo run -- validate --model dinov2-vit-l14` and
confirm the command reports preprocessing status, tensor-semantic status, and
contract mismatches with a non-zero exit code on failure.

### Validation for User Story 1 ⚠️

> **NOTE: Define executable validation first and ensure it fails before
> implementation when it can be written upfront**

- [X] T009 [P] [US1] Add validate CLI contract coverage for argument parsing and exit codes in `tests/validation_cli.rs`
- [X] T010 [P] [US1] Add preprocessing and tensor-semantic integration coverage in `tests/integration_test.rs`
- [X] T011 [P] [US1] Add contract-validation fixture expectations in `tests/fixtures/validation/manifest.json` and `tests/fixtures/validation/dinov2-vit-l14.contract.json`

### Implementation for User Story 1

- [X] T012 [US1] Implement preprocessing contract evaluation in `src/validation/semantics.rs`
- [X] T013 [US1] Implement tensor semantic validation and summary assembly in `src/validation/semantics.rs` and `src/validation/report.rs`
- [X] T014 [US1] Add the `validate` command and wiring in `src/cli/validate.rs`, `src/cli/mod.rs`, and `src/main.rs`
- [X] T015 [US1] Connect default fixture selection and contract-validation execution in `src/validation/fixtures.rs` and `src/cli/validate.rs`
- [X] T016 [US1] Add maintainer-facing help text and terminal summary output in `src/cli/validate.rs` and `src/viz/terminal.rs`

**Checkpoint**: The `validate` command independently verifies preprocessing and
tensor semantics for at least one supported model.

---

## Phase 4: User Story 2 - Preserve Reference Parity (Priority: P2)

**Goal**: Compare exported ONNX behavior against trusted reference evidence and
store golden artifacts for regression detection.

**Independent Test**: Run `cargo run -- validate --model dinov2-vit-l14 --format json --output tmp/validation`
and confirm the output reports parity status, tolerance decisions, artifact
identity, and golden-regression failures when evidence drifts.

### Validation for User Story 2 ⚠️

- [X] T017 [P] [US2] Add golden parity regression coverage in `tests/validation_golden.rs`
- [X] T018 [P] [US2] Add CLI integration coverage for tolerance failure and `--refresh-goldens` in `tests/validation_cli.rs`
- [X] T019 [P] [US2] Add approved reference artifacts and tolerance fixtures in `tests/fixtures/validation/dinov2-vit-l14.reference.json`, `tests/fixtures/validation/mae-vit-l16.reference.json`, `tests/fixtures/validation/ijepa-vit-h14.reference.json`, `tests/fixtures/validation/clip-vit-l14.reference.json`, and `tests/fixtures/validation/siglip-so400m.reference.json`

### Implementation for User Story 2

- [X] T020 [US2] Implement reference-output comparison and tolerance evaluation in `src/validation/parity.rs`
- [X] T021 [US2] Implement golden refresh and artifact identity handling in `src/validation/fixtures.rs` and `src/cli/validate.rs`
- [X] T022 [US2] Integrate parity status, traceability, and exit-code decisions in `src/validation/report.rs`, `src/models/registry.rs`, and `src/cli/validate.rs`
- [X] T023 [US2] Surface machine-readable parity evidence in `src/viz/json.rs` and `src/cli/validate.rs`

**Checkpoint**: The validation workflow detects parity drift against approved
reference evidence and supports reviewed golden refreshes.

---

## Phase 5: User Story 3 - Explain Trust In Reports (Priority: P3)

**Goal**: Make compare and inspect outputs explain what was validated, what the
consumed outputs mean, and which caveats limit trust.

**Independent Test**: Run `cargo run -- compare image.jpg --models dinov2-vit-l14 --format html --output tmp/compare`
or `cargo run -- inspect image.jpg --model dinov2-vit-l14 --format terminal`
and confirm the generated report includes validation status, tensor semantics,
reference-parity explanation, and caveats when evidence is incomplete.

### Validation for User Story 3 ⚠️

- [X] T024 [P] [US3] Add validation-report payload contract tests in `tests/validation_cli.rs`
- [X] T025 [P] [US3] Add compare and inspect integration coverage for validation summaries in `tests/integration_test.rs`
- [X] T026 [P] [US3] Add report summary snapshots in `tests/fixtures/validation/report-summary.json` and `tests/fixtures/validation/report-summary.html`

### Implementation for User Story 3

- [X] T027 [US3] Render structured validation summaries in `src/viz/json.rs`, `src/viz/html.rs`, and `src/viz/terminal.rs`
- [X] T028 [US3] Inject validation summaries into compare and inspect flows in `src/cli/compare.rs` and `src/cli/inspect.rs`
- [X] T029 [US3] Align recommendation, caveat, and explanatory copy in `src/validation/report.rs` and `src/viz/html.rs`

**Checkpoint**: Normal compare and inspect outputs now communicate validation
status and trust caveats without requiring users to inspect code or raw logs.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finish documentation, edge-case coverage, and release validation.

- [X] T030 [P] Update public docs for the validate workflow in `README.md` and `specs/001-validate-onnx-groundtruth/quickstart.md`
- [X] T031 [P] Add edge-case unit coverage for preprocessing and loader contract parsing in `src/models/preprocess.rs` and `src/models/loader.rs`
- [X] T032 Run release-surface validation with `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` for `src/cli/validate.rs`, `src/validation/semantics.rs`, `src/validation/parity.rs`, `src/viz/json.rs`, `src/viz/html.rs`, `src/viz/terminal.rs`, `tests/validation_cli.rs`, and `tests/validation_golden.rs`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all story work.
- **User Story 1 (Phase 3)**: Depends on Foundational completion.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because parity extends
  the validation workflow and summary model introduced there.
- **User Story 3 (Phase 5)**: Depends on User Story 2 because reports must
  surface the final validation summary including parity and caveat state.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: First shippable increment; no dependency on later stories.
- **User Story 2 (P2)**: Requires the `validate` command and shared summary
  structures from User Story 1.
- **User Story 3 (P3)**: Requires the validation summary payload from User
  Stories 1 and 2 so compare/inspect reports can explain trust consistently.

### Within Each User Story

- Executable validation MUST be defined early, and tests/regression checks MUST
  fail before implementation whenever they can be written upfront.
- Shared data structures before command wiring.
- Command wiring before renderer integration.
- Story complete before moving to the next dependent story.

### Parallel Opportunities

- **Setup**: T002 and T003 can run in parallel after T001.
- **Foundational**: T004, T005, and T006 can run in parallel before T007/T008.
- **User Story 1**: T009, T010, and T011 can run in parallel.
- **User Story 2**: T017, T018, and T019 can run in parallel.
- **User Story 3**: T024, T025, and T026 can run in parallel.
- **Polish**: T030 and T031 can run in parallel before T032.

---

## Parallel Example: User Story 1

```bash
# Launch validation for User Story 1 together:
Task: "Add validate CLI contract coverage in tests/validation_cli.rs"
Task: "Add preprocessing and tensor-semantic integration coverage in tests/integration_test.rs"
Task: "Add contract-validation fixture expectations in tests/fixtures/validation/manifest.json and tests/fixtures/validation/dinov2-vit-l14.contract.json"
```

## Parallel Example: User Story 2

```bash
# Launch validation for User Story 2 together:
Task: "Add golden parity regression coverage in tests/validation_golden.rs"
Task: "Add CLI integration coverage for tolerance failure and --refresh-goldens in tests/validation_cli.rs"
Task: "Add approved reference artifacts and tolerance fixtures in tests/fixtures/validation/*.reference.json"
```

## Parallel Example: User Story 3

```bash
# Launch validation for User Story 3 together:
Task: "Add validation-report payload contract tests in tests/validation_cli.rs"
Task: "Add compare and inspect integration coverage in tests/integration_test.rs"
Task: "Add report summary snapshots in tests/fixtures/validation/report-summary.json and tests/fixtures/validation/report-summary.html"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: Run the `validate` command for one supported model and
   confirm preprocessing and tensor-semantic failures are surfaced correctly.
5. Demo the contract-validation workflow before adding parity logic.

### Incremental Delivery

1. Complete Setup + Foundational → shared validation infrastructure is ready.
2. Add User Story 1 → validate contract checks independently.
3. Add User Story 2 → validate parity and golden evidence independently.
4. Add User Story 3 → validate compare/inspect trust summaries independently.
5. Finish Polish → docs, edge cases, and release validation.

### Parallel Team Strategy

1. Team completes Setup + Foundational together.
2. Once Foundational is done:
   - Developer A: User Story 1 command and semantics flow
   - Developer B: Reference fixtures and parity tests for User Story 2
   - Developer C: Report payload tests and renderer prep for User Story 3
3. Merge User Story 2 after User Story 1, then finalize User Story 3 on the
   completed summary payload.

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps each task to a specific user story for traceability
- Every user story includes executable validation before implementation
- Suggested MVP scope: Phase 1 + Phase 2 + Phase 3 (User Story 1 only)
- Avoid dropping fixture, golden, or explanation work; they are required by the constitution
