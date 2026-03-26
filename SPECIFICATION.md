# latent-inspector: Specification v0.1.0

## Overview

A Rust CLI for inspecting and comparing learned representations across self-supervised vision models. Takes images as input, runs inference through multiple SSL models via ONNX Runtime, extracts intermediate representations, and computes a structured set of analyses that reveal how each model perceives the input.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                    latent-inspector                     │
├───────────────────────────────────────────────────────┤
│  cli/         CLI commands (compare, inspect, etc.)   │
│  models/      Model registry, ONNX loading, caching   │
│  extract/     Feature extraction, attention hooks      │
│  analysis/    PCA, CKA, k-NN, rank, variance, Gini   │
│  viz/         Terminal renderer, PNG export, HTML gen  │
│  dataset/     Image loading, batching, caching        │
└───────────────────────────────────────────────────────┘
```

## Core Pipeline

For each (image, model) pair:

1. **Load model**: ONNX Runtime session from cached model file
2. **Preprocess image**: Resize, normalize (model-specific mean/std)
3. **Run inference**: Forward pass, extract intermediate outputs
4. **Extract features**: CLS token [D], patch tokens [N, D], attention weights [L, H, N, N]
5. **Compute analysis**: PCA, rank, variance, attention concentration, etc.

For multi-model comparison:
6. **Cross-model metrics**: CLS cosine similarity, CKA, patch correspondence
7. **Render output**: Terminal, PNG, JSON, or HTML

## Model Interface

All models are normalized to a common representation:

```rust
pub struct ModelOutput {
    pub cls_token: Option<Array1<f32>>,     // [D] global representation
    pub patch_tokens: Array2<f32>,          // [N_patches, D] per-patch features
    pub attention_weights: Option<Array4<f32>>, // [layers, heads, N, N]
    pub model_info: ModelInfo,
}

pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub patch_size: u32,
    pub embed_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub method: SSLMethod,  // DINO, MAE, JEPA, CLIP, etc.
}
```

## Analysis Functions

### Per-model metrics:

| Metric | What it measures | How |
|--------|-----------------|-----|
| Representation rank | Effective dimensionality | Count singular values above threshold (1% of max) |
| Feature variance spectrum | Information distribution | PCA eigenvalue distribution |
| Attention concentration (Gini) | How focused the attention is | Gini coefficient of attention weights |
| Patch entropy | Diversity of patch representations | Entropy of k-means cluster assignments |
| Dead dimensions | Unused feature dimensions | Count dimensions with near-zero variance across patches |
| Feature norm distribution | Magnitude patterns | L2 norms of patch features (histogram) |

### Cross-model metrics:

| Metric | What it measures | How |
|--------|-----------------|-----|
| CLS cosine similarity | Global representation alignment | Cosine similarity between CLS tokens |
| Centered Kernel Alignment (CKA) | Representation similarity | Linear CKA between patch feature matrices |
| Mutual k-NN overlap | Agreement on similarity structure | % overlap in k=10 nearest neighbors |
| Patch correspondence | Spatial alignment | Hungarian matching on patch cosine similarities |
| Rank correlation | Feature importance agreement | Spearman correlation of PCA eigenvalue rankings |

## Visualization

### Terminal (default):
- Unicode block characters for attention maps (▓░▒)
- ANSI colors for PCA projections (RGB mapping)
- Tables for numeric metrics
- Box-drawing for comparison layouts

### PNG:
- Side-by-side attention maps overlaid on original image
- PCA 3-channel projection as RGB image
- Similarity matrices as heatmaps
- Feature histograms

### JSON:
- All numeric metrics as structured JSON
- Suitable for piping into analysis scripts

### HTML:
- Interactive report with hover-to-compare
- Zoomable attention maps
- Toggle between models

## CLI Commands

```bash
# Core commands
latent-inspector compare <image> --models <list> [--output <dir>] [--format terminal|png|json|html]
latent-inspector inspect <image> --model <name> [--layers all|last|<n>] [--output <dir>]
latent-inspector neighbors <image> --model <name> --dataset <dir> [--k 10]
latent-inspector similarity --model-a <name> --model-b <name> --dataset <dir> [--metric cka|knn|cosine]
latent-inspector drift --model <name> --checkpoints <dir> --dataset <dir>
latent-inspector models [--download <name>]  # List/download available models
```

## Model Registry

Models are ONNX files hosted on HuggingFace Hub. On first use:
1. Check `~/.cache/latent-inspector/<model_name>.onnx`
2. If missing, download from HuggingFace
3. Display progress bar during download
4. Verify SHA-256 hash

Supported models (v0.1):
- `dinov2-vit-l14` (DINOv2 ViT-L/14, 304M params, ~1.2GB)
- `dinov3-vit-l14` (DINOv3 distilled ViT-L, ~1.2GB)
- `mae-vit-l16` (MAE ViT-L/16, 304M params, ~1.2GB)
- `ijepa-vit-h14` (I-JEPA ViT-H/14, 632M params, ~2.4GB)
- `clip-vit-l14` (CLIP ViT-L/14, 304M params, ~1.2GB)
- `siglip-so400m` (SigLIP SO400M/14, 400M params, ~1.6GB)

## Dependencies

Core: ort (ONNX Runtime), ndarray, image, rayon, clap, serde/serde_json
Viz: ratatui, crossterm, indicatif
Network: reqwest (model download)
