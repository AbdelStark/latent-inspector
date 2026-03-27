<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. World-Class ML Tooling
- Template Principle 2 -> II. Technical Soundness Before Novelty
- Template Principle 3 -> III. Educational Value By Default
- Template Principle 4 -> IV. Production-Grade By Construction
- Template Principle 5 -> V. Helpful and Visually Clear Experience
Added sections:
- Engineering Standards
- Delivery Workflow & Quality Gates
Removed sections:
- None
Templates requiring updates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
- ✅ .specify/templates/agent-file-template.md (reviewed, no changes required)
- ✅ README.md (reviewed, no changes required)
- ⚠ pending: .specify/templates/commands/*.md (directory absent; nothing to update)
Follow-up TODOs:
- None
-->
# Latent Inspector Constitution

## Core Principles

### I. World-Class ML Tooling
Every feature MUST serve a concrete machine learning inspection, comparison, or
analysis job to be done. Public interfaces MUST state the supported models,
artifacts, assumptions, and limits of the analysis they expose. Features that
do not improve real model understanding, debugging, evaluation, or workflow
throughput are out of scope.

Rationale: this project exists to be serious ML tooling, not generic
visualization or novelty software.

### II. Technical Soundness Before Novelty
Metrics, visualizations, and model outputs MUST be numerically correct,
reproducible, and traceable to documented computation paths. Changes that affect
preprocessing, inference, alignment, or evaluation MUST include validation
evidence such as regression tests, reference comparisons, or benchmark data. If
a shortcut trades correctness for speed or convenience, the tradeoff MUST be
documented and explicit to users.

Rationale: users can only trust the tool if the analysis is defensible.

### III. Educational Value By Default
The product MUST teach users what it is showing, why it matters, and how to
interpret it. CLI help, reports, docs, and visual outputs MUST explain
non-obvious terms, model-specific caveats, and any ambiguity that could mislead
interpretation. Every shipped capability MUST leave behind an example,
quickstart step, or reference output that a technically literate newcomer can
follow.

Rationale: the tool is valuable when it increases understanding, not only when
it emits numbers.

### IV. Production-Grade By Construction
Code MUST be maintainable, testable, observable, and safe to run on real
workloads. Each feature MUST define the validation, diagnostics, failure modes,
resource expectations, and performance constraints needed for reliable use.
Stable interfaces, clear errors, and documented operational behavior take
precedence over experimental convenience.

Rationale: research tooling becomes genuinely helpful only when it survives
repeat use outside the author's machine.

### V. Helpful and Visually Clear Experience
Defaults, terminology, and output structure MUST help users answer the common
question quickly and without ambiguity. Terminal views, images, JSON, and HTML
reports MUST be legible, consistent, and organized around comparison and
interpretation rather than decoration. Visual ambition is encouraged only when
it improves comprehension, confidence, or action.

Rationale: visual appeal matters here because it directly affects insight and
decision speed.

## Engineering Standards

Rust is the primary implementation language for the product surface. Any
exception MUST be justified in the implementation plan.

ML-facing code MUST document expected inputs, preprocessing, output semantics,
and model or dataset dependencies at the module or user-interface boundary.

Claims about speed, quality, or model behavior MUST be backed by reproducible
benchmarks, fixtures, or documented evaluation steps before they appear in
README content, CLI help, or generated reports.

User-facing features MUST specify whether they serve machine consumption, human
interpretation, or both, and MUST implement the necessary output surfaces
accordingly.

## Delivery Workflow & Quality Gates

Every feature spec MUST define the user task, technical context, edge cases,
and measurable success criteria, plus any correctness, educational, and UX
obligations needed to satisfy the core principles.

Every implementation plan MUST pass a Constitution Check before research starts
and again before coding starts. The check MUST confirm ML relevance, validation
strategy, educational artifacts, production-readiness work, and output clarity.

Task lists MUST include the validation, documentation, benchmark, diagnostic,
and polish work required for compliance. These tasks cannot be dropped because a
prompt omitted them. Any intentional omission MUST be recorded in Complexity
Tracking or an equivalent review note.

Reviews and release decisions MUST verify that user-facing changes remain
technically sound, clearly explained, and visually legible across supported
output surfaces.

## Governance

This constitution supersedes conflicting local process notes and planning
templates. Amendments MUST update this file and any affected templates or
guidance in the same change.

Versioning policy for this constitution follows semantic versioning. MAJOR
versions remove or redefine principles in a backward-incompatible way. MINOR
versions add principles, sections, or materially stronger governance. PATCH
versions clarify wording without changing expected behavior.

Compliance review is mandatory for every implementation plan, pull request, and
release candidate. Reviewers MUST block work that violates a core principle
unless the exception is explicitly documented, time-bound, and approved by the
project maintainers.

**Version**: 1.0.0 | **Ratified**: 2026-03-27 | **Last Amended**: 2026-03-27
