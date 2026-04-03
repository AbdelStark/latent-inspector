# Changelog

All notable user-visible changes to this project will be documented in this file.

The format is loosely based on Keep a Changelog. Versions below `1.0` may still make breaking CLI, report-schema, and integration changes.

## [Unreleased]

### Added

- `AGENTS.md` with repository-specific context for autonomous contributors
- `CONTRIBUTING.md` with verified setup, test, and validation-golden workflows
- `docs/ARCHITECTURE.md` with module map, invariants, and operational guidance
- A release build step in CI
- `docs/REPORT-SCHEMA.md` documenting the stable non-terminal report filenames, top-level JSON keys, and `artifacts.json` contract for the ready-model commands

### Changed

- The TUI now uses the live analysis path whenever an image is provided; the misleading compile-time `onnx-inference` split was removed
- Per-image metrics no longer silently coerce isotropy or uniformity failures to `0.0`
- Drift checkpoint discovery now fails on unreadable directory entries instead of silently skipping them
- The README now declares the project's actual alpha status, limitations, roadmap, help path, and verified development commands
- CI now enforces a measured `cargo llvm-cov` gate over the tested non-TUI surface
- Model artifact downloads now retry bounded transient HTTP and stream-read failures before surfacing an error

### Fixed

- Mean attention Gini now returns an explicit error for zero-head attention tensors instead of fabricating a `0.0` result
