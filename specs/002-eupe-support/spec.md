# Feature Specification: EUPE Model Support

**Feature Branch**: `002-eupe-support`
**Created**: 2026-04-03
**Status**: Draft
**Input**: Add EUPE full support — ONNX conversion, HuggingFace hosting, registry integration, inference pipeline, documentation, and validation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Compare EUPE against existing models (Priority: P1)

A researcher wants to understand how EUPE's distilled multi-teacher representation differs from DINOv2's self-distillation and I-JEPA's latent prediction. They run `latent-inspector compare photo.jpg --models dinov2-vit-l14,eupe-vit-b16` and see side-by-side metrics — effective rank, isotropy, CKA, k-NN overlap — revealing whether distillation from multiple domain experts produces a qualitatively different representation geometry.

EUPE is particularly interesting because it has a CLS token. Combined with DINOv2, this is the first model pair in latent-inspector where CLS cosine similarity is fully computable — enabling a global-representation comparison that was unavailable with I-JEPA or V-JEPA 2.

**Why this priority**: Cross-model comparison is the core value proposition. EUPE also unlocks the first CLS-to-CLS metric pair.

**Independent Test**: Run `compare` with EUPE and DINOv2 on a sample image; verify CLS cosine similarity, CKA, and k-NN are all non-null.

**Acceptance Scenarios**:

1. **Given** EUPE ONNX artifacts are not cached, **When** user runs `compare photo.jpg --models dinov2-vit-l14,eupe-vit-b16`, **Then** the tool auto-downloads EUPE (~350 MB), runs inference, and produces a report with per-model metrics, pairwise CKA/k-NN, and CLS cosine similarity.
2. **Given** EUPE is cached, **When** user runs the same command, **Then** inference starts immediately and CLS cosine is computed for the DINOv2/EUPE pair.
3. **Given** user compares EUPE with I-JEPA, **When** CLS cosine is requested, **Then** it is reported as N/A (only EUPE has CLS) with an explanation.
4. **Given** user compares all four ready models, **When** patch correspondence is requested for DINOv2↔EUPE, **Then** it is reported as N/A because embedding dimensions differ (1024 vs 768).

---

### User Story 2 — Inspect EUPE representation in depth (Priority: P2)

A researcher runs `inspect` on EUPE to understand its representation quality. Since EUPE is distilled from multiple specialist teachers (DINOv2, depth estimators, segmenters), its representation should be distinctively "balanced" — strong on both global classification features (from the CLS teacher) and local dense features (from the segmentation teacher).

**Why this priority**: Single-model deep-dive is the second most common workflow.

**Independent Test**: Run `inspect photo.jpg --model eupe-vit-b16` and verify 196 patches, 768 embed dim, all metrics present including CLS L2 norm.

**Acceptance Scenarios**:

1. **Given** EUPE model is available, **When** user runs `inspect photo.jpg --model eupe-vit-b16`, **Then** output includes: 196 patches, 768 embed dim, effective rank, CLS L2 norm, patch entropy, variance spectrum, isotropy, uniformity.
2. **Given** user requests HTML output, **Then** a self-contained report is generated with PCA projection, variance chart, and validation summary.

---

### User Story 3 — Full validation and provenance (Priority: P3)

A researcher verifies EUPE's integration is correct and traceable: preprocessing matches the contract, tensor shapes are as declared, and outputs match golden references.

**Why this priority**: Validation is essential for scientific credibility but is a one-time setup.

**Independent Test**: Run `validate --model eupe-vit-b16` and confirm "validated" status.

**Acceptance Scenarios**:

1. **Given** contract and reference fixtures exist, **When** user runs `validate --model eupe-vit-b16`, **Then** status is "validated", backend=onnx-runtime, 0 drifted signals.
2. **Given** the README model provenance table, **When** a reader looks up EUPE, **Then** they see the original checkpoint, ONNX source, paper citation, and a note about the multi-teacher distillation method.

---

### Edge Cases

- EUPE ViT-B has 196 patches (16x16 grid at 224px) while DINOv2 has 256 patches (16x16 at 224px with patch_size=14). CKA and k-NN use the smaller patch count; patch correspondence is N/A due to dimension mismatch (768 vs 1024).
- EUPE's embed_dim=768 is smaller than DINOv2 (1024) and I-JEPA (1280). This is the first sub-1024 model in the registry — analysis code must handle this without assumptions about minimum dimension.
- The ONNX export uses `forward_features` which returns normalized tokens (`x_norm_clstoken`, `x_norm_patchtokens`). The output tensor name may differ from the standard `last_hidden_state` — the registry must map the correct ONNX output name.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: EUPE ViT-B/16 MUST be exported to ONNX from `facebook/EUPE-ViT-B` using the model's `forward_features` method.
- **FR-002**: The ONNX artifact MUST be uploaded to `abdelstark/eupe-vit-b16-onnx` on HuggingFace Hub with SHA-256 checksums pinned.
- **FR-003**: The model registry MUST include EUPE with metadata: ViT-B/16, 86M params, 768 embed dim, 12 layers, 12 heads, 224px input, patch_size=16, 196 patches, CLS expected=true.
- **FR-004**: Preprocessing MUST use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) with resize+center-crop at 224px.
- **FR-005**: Inference MUST extract both CLS token (index 0) and 196 patch tokens from the ONNX output into `ModelOutput`.
- **FR-006**: All analysis commands MUST work with EUPE without special-case code — it fits the existing `PatchAndClsSequence` tensor role.
- **FR-007**: Validation fixtures (contract + golden reference) MUST be generated from ONNX Runtime inference.
- **FR-008**: README MUST be updated: models table, provenance table, "Why this exists" section (add EUPE bullet), case study mention.
- **FR-009**: `SSLMethod` enum MUST include an EUPE variant.
- **FR-010**: The CHANGELOG MUST document the addition.

### Quality Requirements

- **QR-001**: ONNX export MUST be verified against PyTorch reference with max numerical diff < 0.01.
- **QR-002**: All existing tests MUST pass after adding EUPE (model count assertions updated).
- **QR-003**: CLS cosine similarity between EUPE and DINOv2 MUST be computable — this is the first valid CLS pair.
- **QR-004**: EUPE provenance MUST include paper citation ([Zhu et al. 2026](https://arxiv.org/abs/2603.22387)), original checkpoint link, and export methodology.

### Key Entities

- **EUPE ViT-B/16**: 86M-parameter ViT-B backbone distilled from multiple domain-expert teachers into a single efficient encoder. Produces 1 CLS + 196 patch tokens of dimension 768. Trained on LVD-1689M.
- **ONNX artifact**: Exported from `facebook/EUPE-ViT-B`, hosted at `abdelstark/eupe-vit-b16-onnx`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `compare photo.jpg --models dinov2-vit-l14,eupe-vit-b16` produces a report with CLS cosine similarity computed (the first cross-model CLS pair in the tool).
- **SC-002**: `validate --model eupe-vit-b16` reports "validated" with 0 drifted parity signals.
- **SC-003**: All existing tests pass with zero regressions.
- **SC-004**: Model auto-downloads on first use and cache is SHA-256 verified.
- **SC-005**: ONNX inference on 224x224 image completes in under 1 second on standard laptop CPU (EUPE ViT-B is smaller than the other models).

## Assumptions

- `facebook/EUPE-ViT-B` on HuggingFace is publicly accessible and its `forward_features` method produces normalized CLS and patch tokens.
- EUPE uses standard ImageNet normalization based on its DINOv3 library heritage.
- The ONNX export will use the TorchScript legacy exporter at opset 14 with onnxsim simplification, matching the V-JEPA 2 export process that works with the `ort` crate.
- EUPE ViT-B has 768 embed_dim — the first model in the registry below 1024. No code changes are expected since the analysis pipeline is dimension-agnostic.
- The FAIR Research License (non-commercial) applies to the model weights; the ONNX re-hosting preserves this license.
