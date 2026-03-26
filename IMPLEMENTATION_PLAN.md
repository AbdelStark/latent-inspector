# latent-inspector: Implementation Plan

## Phase 1: Foundation (Week 1-2)
- ONNX Runtime integration via `ort` crate
- Model registry: download, cache, verify ONNX models from HuggingFace
- Image preprocessing: resize, normalize with model-specific params
- Feature extraction: run inference, extract CLS token + patch tokens
- Start with DINOv2 only (simplest, best documented ONNX export)
- Basic `inspect` command: load model, extract features, print shapes + norms

## Phase 2: Analysis Engine (Week 2-3)
- PCA via custom SVD implementation (ndarray-linalg or pure ndarray)
- Representation rank: singular value thresholding
- Feature variance spectrum: eigenvalue distribution
- Attention weight extraction (from ONNX intermediate outputs)
- Attention concentration: Gini coefficient
- Patch entropy: k-means clustering + Shannon entropy
- Dead dimension detection: near-zero variance scan
- `inspect` command fully functional with all per-model metrics

## Phase 3: Multi-Model Comparison (Week 3-4)
- Add all 6 models (DINOv2, DINOv3, MAE, I-JEPA, CLIP, SigLIP)
- Parallel inference via rayon (all models simultaneously)
- CLS cosine similarity matrix
- Centered Kernel Alignment (CKA) implementation
- Mutual k-NN overlap
- Patch correspondence via Hungarian matching
- `compare` command: side-by-side output for N models

## Phase 4: Visualization (Week 3-4)
- Terminal renderer: ratatui for rich inline display
  - Unicode block attention maps
  - ANSI color PCA projections
  - Formatted tables for metrics
- PNG export: attention overlays, PCA RGB images, heatmaps
- JSON export: structured metrics for scripting
- HTML report: self-contained interactive page

## Phase 5: Dataset Commands (Week 4)
- `neighbors` command: k-NN retrieval across image directory
- `similarity` command: CKA/k-NN between model pairs across dataset
- `drift` command: track representations across checkpoints
- Batch processing with progress bars (indicatif)
- Result caching: store embeddings for repeated queries

## Phase 6: Polish (Week 4)
- `models` command: list available, download specific
- Error handling, graceful degradation (model download fails, etc.)
- Documentation: README examples, --help text
- cargo publish to crates.io
- Blog post + demo recording for launch

## Key Technical Decisions

### ONNX Runtime vs Tract
ONNX Runtime (`ort` crate): faster, GPU support, wider model compatibility.
Tract: pure Rust, but slower and less compatible with complex ViT models.
Decision: ONNX Runtime for inference, pure Rust for analysis. Best of both worlds.

### Extracting intermediate representations from ONNX
ONNX models can be modified to output intermediate activations. We pre-process models to expose:
- Last-layer patch tokens (before head)
- CLS token
- Attention weights from each layer (if available in the graph)
Some models (MAE decoder-side) require custom ONNX graph surgery.

### PCA without heavy dependencies
We avoid linking to LAPACK/OpenBLAS. Instead:
- For small matrices (< 1024 dims): iterative power method for top-k eigenvalues
- For full SVD when needed: use ndarray-linalg with openblas-src as optional feature
- Default: power method (no system deps, works everywhere)
