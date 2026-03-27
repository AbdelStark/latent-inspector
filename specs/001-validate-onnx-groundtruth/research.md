# Research: Model Validation Evidence

## Decision 1: Extend the model registry into the validation source of truth

- **Decision**: Keep validation metadata alongside the existing registry entry
  model by expanding registry-backed model contracts with explicit
  preprocessing, tensor-semantic, and reference-evidence fields.
- **Rationale**: `src/models/registry.rs` already owns model names, tensor
  names, normalization values, and artifact metadata. Extending that single
  source of truth keeps preprocessing, loader expectations, and validation
  rules aligned and reduces drift between code paths.
- **Alternatives considered**:
  - Separate YAML or JSON manifests: Easier to author, but introduces a second
    metadata system that can diverge from runtime behavior.
  - Hardcoded checks in `loader.rs`: Keeps data close to inference logic, but
    scatters model semantics across multiple modules and makes reports harder to
    explain.

## Decision 2: Use offline golden evidence generated from trusted references

- **Decision**: Compare exported ONNX behavior against checked-in golden
  artifacts produced from trusted reference runs, and use those artifacts in the
  default regression suite and CLI validation summaries.
- **Rationale**: The constitution requires reproducible evidence. Checked-in
  fixtures keep CI and local validation deterministic without requiring network
  access, remote model downloads beyond the ONNX export, or a live Python stack
  during every test run.
- **Alternatives considered**:
  - Invoke Python or source-model libraries during every validation run: Closer
    to the reference implementation, but too slow and fragile for routine CI and
    local testing.
  - Manual comparison only: Fails the requirement for repeatable, release-grade
    evidence.

## Decision 3: Make tensor semantics explicit instead of inferring them from shape

- **Decision**: Model output meaning will be described by explicit tensor
  contracts including tensor name, expected rank, sequence layout, CLS policy,
  patch-count derivation, and downstream consumer notes.
- **Rationale**: The current loader infers CLS presence from sequence length and
  assumes one output tensor encodes all usable features. That is convenient but
  too implicit for trustworthy validation and rich user-facing explanations.
- **Alternatives considered**:
  - Keep the current sequence-length heuristic: Simpler short term, but it will
    miss silent semantic drift and produce weak trust messaging.
  - Validate only tensor names and raw shapes: Better than nothing, but still
    leaves downstream meaning ambiguous.

## Decision 4: Keep heavy reference refresh out of the default test loop

- **Decision**: Separate routine regression checks from the maintainer-only
  workflow that refreshes golden reference evidence when exports intentionally
  change.
- **Rationale**: This feature must be production-grade, but default checks also
  need to stay fast and reliable. Routine tests should consume versioned
  artifacts, while golden refresh can be an explicit operation with stronger
  prerequisites and review.
- **Alternatives considered**:
  - Refresh reference evidence during every test run: Too expensive and too
    dependent on external tooling.
  - Never refresh automatically: Leaves no controlled path for intentional model
    export updates.

## Decision 5: Surface trust summaries as structured data across all reports

- **Decision**: Introduce a structured validation summary model that can be
  rendered in terminal, JSON, and HTML outputs and reused by compare, inspect,
  and the dedicated validation workflow.
- **Rationale**: The feature is both technical and educational. Structured data
  keeps the explanation consistent, testable, and machine-readable while
  avoiding duplicated prose in each renderer.
- **Alternatives considered**:
  - Add free-form explanatory strings directly in each renderer: Faster for one
    output surface, but hard to keep aligned and harder to test.
  - Document the trust status only in README/quickstart: Insufficient because
    the constitution requires the product itself to explain what it shows.

## Decision 6: Add a dedicated `validate` CLI workflow

- **Decision**: Expose validation as a first-class CLI command while also
  embedding validation summaries into existing compare and inspect outputs.
- **Rationale**: Maintainers need a direct workflow for contract and parity
  checks, and users still need to see trust information in the normal reporting
  paths they already use.
- **Alternatives considered**:
  - Hide validation behind tests only: Leaves no user-facing or maintainer-facing
    workflow in the product.
  - Overload existing `inspect` or `models` commands: Makes the interface less
    discoverable and mixes diagnostic concerns with unrelated command purposes.
