# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Interactive terminal UI with ratatui (dashboard, inspector, compare, spectrum views)
- I-JEPA ViT-H/14 model registration with verified SHA-256 hashes (reference evidence currently stub-backed, pending ONNX refresh)
- Validation workflow: preprocessing contract, tensor semantics, and reference parity checks
- HTML report bundles with embedded validation summaries and artifact manifests
- ASCII fallback for non-Unicode terminals (`LATENT_INSPECTOR_FORCE_ASCII=1`)
- Dataset-backed analysis: neighbors, similarity, drift commands with parallel workers
- GitHub Actions CI pipeline (check, test, clippy, fmt)
- LICENSE-MIT and LICENSE-APACHE files

### Fixed
- Silent CLS-token validation bug: model returning wrong token count now errors instead of proceeding with missing CLS data
- NaN propagation through CKA denominator now guarded with finite checks
- Correspondence module silent fallback on matrix construction replaced with explicit invariant check
- Neighbor cosine similarity now clamped to [-1, 1] to prevent floating-point overflow in rankings
- HTML title tag XSS vulnerability: image names are now escaped in `<title>` elements
- Added `sanitize_href()` to block javascript:/data:/vbscript: URIs in generated HTML links
- JSON error variant: `VizError::Json` added; JSON operations no longer misreport as HTML errors
- Preprocessing pipeline: replaced `resize_exact` (which distorted non-square images) with standard resize-short-edge + center-crop matching torchvision ViT preprocessing
- DefaultHasher replaced with FNV-1a in stub backend for cross-toolchain reproducibility
- Test fixture path for `terminal_ascii_cli` corrected (missing `samples/` subdirectory)
- Mutex poisoning cascade in cache tests prevented with `unwrap_or_else(|e| e.into_inner())`
- All stale model-count assertions updated for I-JEPA ready status

### Changed
- Duplicated `patch_grid_side` in CLI modules now delegates to `analysis::square_grid_side`
- Duplicated `dim_color`, `mini_bar`, `truncate` in TUI views extracted to `tui::theme`
- Duplicated `l2_norm` / `l2_norm_view` collapsed into single function
- Shared test helpers extracted to `tests/common/mod.rs`
- AGENTS.md rewritten with accurate architecture and model status

## [0.1.0] — 2026-03-26

### Added
- Initial release: DINOv2 ViT-L/14 model loading and ONNX inference
- CLI commands: compare, inspect, neighbors, similarity, drift, models, validate
- Analysis pipeline: PCA, CKA, k-NN overlap, representation rank, variance spectrum, attention concentration, patch entropy
- Output formats: terminal, JSON, HTML, PNG
- Model registry with 6 planned SSL models
- Automatic model download with SHA-256 verification and HTTP resume
