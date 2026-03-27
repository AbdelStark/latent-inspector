# Data Model: Model Validation Evidence

## ModelValidationProfile

- **Purpose**: Canonical validation contract for one supported model/export
  combination.
- **Fields**:
  - `model_name`: Stable CLI identifier.
  - `export_variant`: Export or artifact variant being validated.
  - `artifact_identity`: Versioned ONNX artifact metadata used for traceability.
  - `preprocess_contract`: Input size, resize policy, color space, channel
    order, mean/std, and any model-specific input caveats.
  - `tensor_contracts`: Ordered list of consumed output tensor contracts.
  - `reference_source`: Human-readable reference provenance for the trusted
    comparison.
  - `tolerance_profile`: Allowed deviation windows per compared signal.
  - `validation_status`: `unverified`, `partial`, `validated`, `failed`, or
    `stale`.
  - `last_verified_at`: Timestamp or version marker for the most recent approved
    validation evidence.
- **Validation rules**:
  - `model_name` plus `export_variant` must uniquely identify one profile.
  - A profile cannot be marked `validated` unless required tensor contracts and
    reference comparison evidence exist.

## TensorContract

- **Purpose**: Explicit meaning and expected structure of one consumed tensor.
- **Fields**:
  - `tensor_name`: Runtime tensor name or alias.
  - `semantic_role`: For example CLS embedding stream, patch token grid, or
    optional attention weights.
  - `required`: Whether the tensor is mandatory for the product claim.
  - `expected_rank`: Number of dimensions expected at runtime.
  - `shape_pattern`: Human-readable shape contract with variable dimensions.
  - `cls_policy`: Whether CLS is present, absent, or model-conditional.
  - `patch_layout_rule`: How patch count maps back to the model input grid.
  - `downstream_consumers`: Metrics or report features that rely on this tensor.
  - `notes`: Model-specific caveats that must appear in explanations when
    relevant.
- **Validation rules**:
  - Every required tensor contract must map to at least one downstream consumer.
  - `shape_pattern` and `cls_policy` must be sufficient to reject silent layout
    changes.

## ReferenceComparisonRecord

- **Purpose**: Evidence that one export still matches trusted reference
  behavior on the validation fixture set.
- **Fields**:
  - `profile_key`: Link back to `ModelValidationProfile`.
  - `fixture_set_id`: Shared validation input set used for comparison.
  - `signals_compared`: Named outputs or derived signals compared against the
    reference.
  - `measured_deviation`: Observed deviation for each compared signal.
  - `allowed_tolerance`: Approved tolerance for each compared signal.
  - `result`: `pass`, `partial`, or `fail`.
  - `evidence_ref`: Pointer to the stored golden artifact or reference bundle.
  - `review_notes`: Optional explanation for approved exceptions.
- **Validation rules**:
  - A `pass` record requires every required signal to be within tolerance.
  - A `partial` record requires a documented reason and may not be treated as
    release-ready without explicit approval.

## GoldenValidationArtifact

- **Purpose**: Versioned baseline artifact consumed by tests and validation runs.
- **Fields**:
  - `artifact_id`: Stable identifier for the golden bundle.
  - `profile_key`: Associated `ModelValidationProfile`.
  - `fixture_inputs`: Manifest of validation inputs included in the artifact.
  - `reference_outputs`: Stored expected outputs or derived signals.
  - `summary_snapshot`: Expected validation summary state.
  - `created_from`: Provenance of the source-model run used to produce it.
  - `refresh_policy`: Whether the artifact is routine, deprecated, or awaiting
    refresh.
- **Validation rules**:
  - Golden artifacts must be deterministic and linked to a specific profile
    version.
  - Replacing an artifact must preserve auditability of what changed.

## ReportValidationSummary

- **Purpose**: Shared user-facing trust payload rendered in terminal, JSON, and
  HTML reports.
- **Fields**:
  - `profile_key`: Associated validation profile.
  - `status`: `validated`, `partial`, `unverified`, `failed`, or `stale`.
  - `preprocess_summary`: Plain-language summary of input validation.
  - `tensor_summary`: Plain-language summary of output semantic checks.
  - `parity_summary`: Plain-language summary of reference comparison results.
  - `caveats`: Ordered list of warnings or limitations.
  - `evidence_timestamp`: Timestamp or version of the evidence being cited.
  - `report_recommendation`: Recommended level of user trust or reviewer action.
- **Validation rules**:
  - `status` must be derivable from the underlying validation profile and
    comparison record.
  - `caveats` cannot be empty when the status is `partial`, `failed`, or
    `stale`.

## Relationships

- One `ModelValidationProfile` has many `TensorContract` entries.
- One `ModelValidationProfile` has many `ReferenceComparisonRecord` entries over
  time.
- One `ModelValidationProfile` has many `GoldenValidationArtifact` versions over
  time, but only one current approved artifact per export variant.
- One `ReportValidationSummary` is derived from one current
  `ModelValidationProfile` and its most relevant `ReferenceComparisonRecord`.

## State Transitions

- `unverified` -> `validated`: All required contract checks and reference
  comparisons pass with approved evidence.
- `unverified` -> `partial`: Some evidence exists, but required signals or
  fixtures are incomplete.
- `validated` -> `stale`: Export, contract, or fixture inputs change without a
  refreshed golden artifact.
- `validated` -> `failed`: Required regression or parity checks exceed approved
  tolerance.
- `partial` -> `validated`: Missing evidence is completed and approved.
- `failed` -> `validated`: A corrected export or updated approved artifact
  passes the validation workflow.
