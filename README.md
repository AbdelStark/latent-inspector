# latent-inspector

A fast Rust CLI for inspecting and comparing learned representations across self-supervised vision models. Feed it an image, get a structured comparison of how DINOv2, I-JEPA, MAE, CLIP, and SigLIP see the world — with real numbers, not vibes.

<table>
<tr>
<td width="50%"><img src="docs/assets/img/screenshots/tui-1.png" alt="Dashboard"/><br/><sub>Dashboard — model registry, image preview, architecture comparison</sub></td>
<td width="50%"><img src="docs/assets/img/screenshots/tui-2.png" alt="Inspector"/><br/><sub>Inspector — representation health gauges and PCA variance spectrum</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/img/screenshots/tui-3.png" alt="Compare"/><br/><sub>Compare — cross-model metrics, CLS similarity matrix, CKA and k-NN overlap</sub></td>
<td width="50%"><img src="docs/assets/img/screenshots/tui-4.png" alt="Spectrum"/><br/><sub>Spectrum — full PCA scree plot with 90%/99% thresholds and interpretation</sub></td>
</tr>
</table>

## Quick start

```bash
# Install
cargo install latent-inspector

# Or build from source with real ONNX inference
cargo build --features onnx-inference --release

# Compare two models on an image (models download automatically on first use)
latent-inspector compare photo.jpg --models dinov2-vit-l14,ijepa-vit-h14

# Deep-dive into a single model
latent-inspector inspect photo.jpg --model dinov2-vit-l14

# Profile a model's representation space over a dataset
latent-inspector profile --model dinov2-vit-l14 --dataset images/

# Interactive TUI
latent-inspector tui photo.jpg -m dinov2-vit-l14,ijepa-vit-h14

# See all available models and their status
latent-inspector models
```

## Why this exists

Self-supervised learning (SSL) models learn to represent images without labels, but they do so in fundamentally different ways:

- **DINOv2** learns patch-level features via self-distillation. Its representations naturally segment objects — patches on the elephant cluster together, patches on the background cluster together — without ever seeing a segmentation label.
- **I-JEPA** predicts missing patches in latent space (not pixel space). It learns to fill in what's "probably there" based on context, favoring abstract structure over texture.
- **MAE** reconstructs masked pixel regions. It must encode enough detail to literally redraw the masked patches.
- **CLIP** aligns images with text descriptions. Its representation is shaped by language, not just visual similarity.

These different training objectives create different internal "world models." latent-inspector makes those differences visible, measurable, and comparable with concrete metrics.

## Supported models

| Model | Architecture | Params | Method | Status |
|-------|-------------|--------|--------|--------|
| **DINOv2** | ViT-L/14 | 304M | Self-distillation + centering | **Ready** |
| **I-JEPA** | ViT-H/14 | 632M | Joint embedding predictive | **Ready** |
| DINOv3 | ViT-L/14 | 304M | Self-distillation + Gram anchoring | Planned |
| MAE | ViT-L/16 | 304M | Masked autoencoder | Planned |
| CLIP | ViT-L/14 | 304M | Contrastive image-text | Planned |
| V-JEPA 2 | ViT-L/16 | 304M | Video joint embedding predictive | Planned |
| SigLIP | ViT-SO400M/14 | 400M | Sigmoid contrastive image-text | Planned |

Models download automatically on first use (~1-2 GB each) and are cached in `~/.cache/latent-inspector/`. Downloads resume from partial transfers when possible. Override the cache location with `LATENT_INSPECTOR_CACHE_DIR`.

---

## Case study: How DINOv2 and I-JEPA see an elephant

This walkthrough uses a real elephant photograph to show what latent-inspector reveals about how two fundamentally different SSL approaches represent the same image. Every number below is from an actual ONNX inference run.

### Step 1: Compare both models

```bash
latent-inspector compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14
```

```
Model Comparison
================================================================================
Metric                dinov2-vit-l14  ijepa-vit-h14
--------------------------------------------------------------------------------
Repr. rank            60/1024         44/1280
Dead dimensions       0               0
Patch entropy         2.52            2.89
CLS L2 norm           46.3            N/A
Top-10 var%           66.8%           72.7%
Components@90%        31              22
================================================================================
```

#### What these numbers mean

**Representation rank** (60/1024 vs 44/1280): How many dimensions the model actually uses. DINOv2 spreads information across 60 effective dimensions out of 1024 total. I-JEPA is more concentrated — only 44 out of 1280. Neither has dead dimensions (dimensions that are zero for all patches), which means both models are efficient in their use of the embedding space.

**Patch entropy** (2.52 vs 2.89): How diverse the patch representations are across the image. Higher entropy means patches are more differentiated from each other. I-JEPA (2.89) creates more distinct per-patch representations than DINOv2 (2.52). This makes sense: I-JEPA's prediction objective forces it to encode fine-grained spatial context to predict what's missing, while DINOv2's distillation objective favors consistent global features.

**CLS L2 norm** (46.3 vs N/A): The magnitude of the global image representation. DINOv2 exposes a CLS (classification) token — a single vector that summarizes the entire image. I-JEPA's ONNX export does not include one. This architectural difference means CLS-based comparisons (like CLS cosine similarity) are unavailable for mixed DINOv2/I-JEPA comparisons. latent-inspector reports this explicitly as `N/A` rather than silently dropping the metric.

**Top-10 variance** (66.8% vs 72.7%): What fraction of total information is captured by the first 10 principal components. I-JEPA concentrates more variance into fewer dimensions — its representation is more "top-heavy." DINOv2 spreads information more evenly.

**Components@90%** (31 vs 22): How many PCA components are needed to explain 90% of the variance. I-JEPA needs only 22 components; DINOv2 needs 31. This confirms I-JEPA's representation is lower-dimensional in practice, despite having a wider embedding space (1280 vs 1024).

### Step 2: Cross-model similarity

The same `compare` command also outputs pairwise metrics:

```
Linear CKA:
              dinov2-vit-l14  ijepa-vit-h14
dinov2-vit... 1.000           0.329
ijepa-vit-h14 0.329           1.000

k-NN overlap (k=10):
              dinov2-vit-l14  ijepa-vit-h14
dinov2-vit... 1.000           0.278
ijepa-vit-h14 0.278           1.000
```

**Linear CKA = 0.329**: Centered Kernel Alignment measures whether two models organize their representations in similar geometric structures. A CKA of 1.0 means identical geometry; 0.0 means completely unrelated. At 0.329, DINOv2 and I-JEPA have *some* structural overlap but represent the elephant in substantially different ways. This is expected — self-distillation and latent prediction are fundamentally different training signals.

**k-NN overlap = 0.278**: For each patch, look at its 10 nearest neighbors in each model's representation space. Only 27.8% of neighbors overlap. This means when DINOv2 considers two patches "similar," I-JEPA often disagrees. The elephant's trunk patches might cluster with body patches in one model but with background-boundary patches in the other.

### Step 3: Deep-dive into a single model

```bash
latent-inspector inspect docs/assets/img/samples/elephant_sample_image.jpg \
  --model dinov2-vit-l14
```

```
Model: dinov2-vit-l14
============================================================
  Patches:          256
  Embed dim:        1024
  Effective rank:   60/1024
  Dead dimensions:  0
  Patch entropy:    2.523
  CLS L2 norm:      46.28
  Patch norm mean:  47.52 +/- 1.41
  Top-10 var%:      66.8%
  Components@90%:   31

  Variance spectrum (top 12 components):
    PC01: 17.17%  17.17% cum  ######
    PC02: 12.52%  29.70% cum  #####
    PC03:  9.07%  38.76% cum  ###
    PC04:  6.09%  44.85% cum  ##
    PC05:  5.15%  50.00% cum  ##
    PC06:  4.62%  54.61% cum  #
    PC07:  3.67%  58.28% cum  #
    PC08:  3.30%  61.58% cum  #
    PC09:  2.79%  64.37% cum  #
    PC10:  2.43%  66.79% cum
    PC11:  2.11%  68.90% cum
    PC12:  1.98%  70.88% cum
```

**Patch norm mean 47.52 +/- 1.41**: DINOv2 patch vectors have remarkably consistent magnitudes (standard deviation of only 1.41). This means no patch is dramatically more "activated" than others — the model distributes representational energy evenly across the image. By contrast, I-JEPA shows 33.77 +/- 6.14: much more variation, suggesting it gives some patches significantly stronger representations than others.

**Variance spectrum**: The first principal component captures 17.17% of variance — the single strongest "direction" in the representation. By PC05, we reach 50% cumulative variance. The gradual decay (rather than a sharp cliff) tells us DINOv2 uses a rich, multi-scale representation. No single axis dominates.

### Step 4: Export reports for sharing

Every command supports `--format terminal|json|html|png` and `--output <dir>`:

```bash
# Generate a self-contained HTML report bundle
latent-inspector compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14 \
  --format html --output elephant-report/

# What gets generated:
# elephant-report/
#   report.html          Interactive HTML with all metrics and charts
#   compare.json         Same data as structured JSON for automation
#   dinov2-vit-l14_pca.png   PCA projection (3 components as RGB)
#   ijepa-vit-h14_pca.png    PCA projection for I-JEPA
#   linear_cka.png       Cross-model CKA heatmap
#   knn_overlap_k10.png  Cross-model k-NN overlap heatmap
#   input_image.png      Copy of the input image
#   artifacts.json       Machine-readable manifest of all outputs
```

```bash
# Single-model deep-dive report
latent-inspector inspect docs/assets/img/samples/elephant_sample_image.jpg \
  --model dinov2-vit-l14 --format html --output dinov2-inspect/

# Outputs: report.html, inspect.json, dinov2-vit-l14_pca.png,
#   dinov2-vit-l14_variance.png, input_image.png, artifacts.json
```

```bash
# JSON for programmatic consumption
latent-inspector compare photo.jpg --models dinov2-vit-l14,ijepa-vit-h14 \
  --format json | jq '.comparisons[0].linear_cka'
# 0.329
```

### Key takeaway

DINOv2 and I-JEPA both produce rich representations of the elephant, but they organize information differently:

| Property | DINOv2 | I-JEPA | Interpretation |
|----------|--------|--------|----------------|
| Effective rank | 60/1024 | 44/1280 | DINOv2 uses more dimensions |
| Variance concentration | 66.8% in top 10 | 72.7% in top 10 | I-JEPA is more concentrated |
| Patch entropy | 2.52 | 2.89 | I-JEPA differentiates patches more |
| Patch norm std | 1.41 | 6.14 | DINOv2 is more uniform |
| CLS token | Yes (46.3 norm) | No | Different architectures |

The low CKA (0.329) and low k-NN overlap (0.278) confirm these are genuinely different world models — not just rescaled versions of the same representation.

---

## Commands reference

### `compare` — Side-by-side model comparison

```bash
latent-inspector compare <image> --models <model1>,<model2>[,...]
  [--format terminal|json|html|png]
  [--output <dir>]
  [--pca-components <n>]
```

Computes per-model metrics and pairwise cross-model similarity. Handles mismatched architectures gracefully: dimension-agnostic metrics (CKA, k-NN) are computed when patch counts match; dimension-dependent metrics (patch correspondence) and architecture-dependent metrics (CLS cosine) are reported as `N/A` with an explanation.

**Pairwise metrics:**
- **CLS cosine similarity** — Global image representation similarity (requires both models to export a CLS token)
- **Linear CKA** — Representation geometry alignment, invariant to linear transforms
- **k-NN overlap** — Neighborhood agreement: fraction of shared nearest neighbors across models
- **Mean patch correspondence** — Hungarian-matched optimal patch pairing similarity (requires matching embedding dimensions)

### `inspect` — Single model deep-dive

```bash
latent-inspector inspect <image> --model <model>
  [--format terminal|json|html|png]
  [--output <dir>]
  [--pca-components <n>]
```

Full representation analysis for one model: rank, entropy, variance spectrum, patch norm statistics, attention concentration (when available), and PCA projection. The variance spectrum shows the full scree plot — how information is distributed across principal components.

### `neighbors` — k-NN retrieval across a dataset

```bash
latent-inspector neighbors <image> --model <model> --dataset <dir>
  [--k <n>]
  [--format terminal|json|html|png]
  [--output <dir>]
```

Given a query image and a dataset directory, find the k most similar images according to the model. This reveals what a model considers "similar" — DINOv2 finds visually similar objects, while CLIP (when ready) will find semantically similar concepts. If the model doesn't expose a CLS token, the command uses mean-patch embeddings automatically.

### `similarity` — Cross-model alignment on a dataset

```bash
latent-inspector similarity --model-a <model> --model-b <model> --dataset <dir>
  [--format terminal|json|html|png]
  [--output <dir>]
```

Measures how similarly two models represent an entire dataset using linear CKA, mutual k-NN overlap, and (when both models expose CLS tokens) mean CLS cosine similarity. Runs inference in parallel across the dataset.

### `profile` — Representation space profiling over a dataset

```bash
latent-inspector profile --model <model> --dataset <dir>
  [--format terminal|json|html|png]
  [--output <dir>]
```

Generates a comprehensive representation fingerprint by running the model on every image in a dataset and computing both per-image metric aggregates and dataset-level space metrics:

- **Isotropy (cosine)** — How uniformly embeddings are spread in the representation space (1 - average pairwise cosine similarity)
- **Isotropy (partition)** — Singular value uniformity of the embedding matrix (Mu et al. 2018)
- **Uniformity** — Wang & Isola (2020) metric measuring spread on the unit hypersphere
- **Intrinsic dimensionality** — MLE estimate (Levina & Bickel 2004) of the representation manifold dimension

Per-image metrics (rank, entropy, Gini, variance concentration) are aggregated as mean/std/min/max across the dataset.

### `drift` — Track representation changes across checkpoints

```bash
latent-inspector drift --model <model> --checkpoints <dir> --dataset <dir>
  [--format terminal|json|html|png]
  [--output <dir>]
```

Load a directory of `.onnx` checkpoint files (different training stages), run inference on a shared dataset, and report consecutive CKA scores. This shows when and how much a model's representations shift during training. Checkpoints are processed in natural numeric order (`step-2.onnx` before `step-10.onnx`).

### `models` — Registry and cache status

```bash
latent-inspector models
  [--verbose]
  [--download <model>]
  [--format terminal|json|html]
  [--output <dir>]
```

Displays the full model registry with status, readiness, cache state, evidence status, and artifact inventory. Use `--verbose` for per-artifact details. Use `--download <model>` to pre-cache a model before running analysis.

### `validate` — Preprocessing and parity checks

```bash
latent-inspector validate --model <model>
  [--format terminal|json|html]
  [--output <dir>]
  [--refresh-goldens]
```

Validates a model's integration against checked-in contract and reference artifacts. Checks preprocessing parameters, tensor semantics (names, shapes, roles), and output parity against golden fixtures. Use `--refresh-goldens` to regenerate reference artifacts after a verified ONNX update.

### `tui` — Interactive terminal UI

```bash
latent-inspector tui [<image>] [-m <model1>,<model2>,...]
```

Interactive terminal interface with multiple views: dashboard (model registry overview), inspector (per-model metrics and variance spectrum), compare (cross-model pairwise matrices), spectrum (full PCA scree plot), file browser (select images), and help (keyboard shortcuts). Navigate with arrow keys, switch views with number keys.

## Output formats

Every analysis command supports four output formats:

| Format | Flag | Output | Use case |
|--------|------|--------|----------|
| **Terminal** | `--format terminal` (default) | Rich Unicode display, ASCII fallback | Interactive exploration |
| **JSON** | `--format json` | Structured metrics to stdout or file | Automation, scripting, dashboards |
| **HTML** | `--format html` | Self-contained report bundle | Sharing, documentation, review |
| **PNG** | `--format png` | PCA projections, heatmaps, charts | Presentations, papers |

When `--output <dir>` is provided, all formats also emit an `artifacts.json` manifest listing every generated file with byte sizes and SHA-256 digests. HTML bundles include companion JSON for both human and machine consumption.

Force ASCII output in non-Unicode terminals: `LATENT_INSPECTOR_FORCE_ASCII=1`.

## Metrics glossary

| Metric | What it measures | Range | Intuition |
|--------|-----------------|-------|-----------|
| **Effective rank** | Number of significant singular values | 1 to embed_dim | Higher = more expressive; the model uses more of its capacity |
| **Dead dimensions** | Embedding dimensions that are zero for all patches | 0 to embed_dim | Should be 0; non-zero means wasted capacity |
| **Patch entropy** | Diversity of patch representations (via k-means clustering) | 0 to log2(k) | Higher = patches are more differentiated from each other |
| **Attention Gini** | Concentration of attention weights | 0 to 1 | Higher = more focused attention; lower = diffuse |
| **CLS L2 norm** | Magnitude of the global image vector | 0+ | Varies by model; useful for cross-image comparison |
| **Patch norm mean/std** | Distribution of patch vector magnitudes | 0+ | Low std = uniform activation; high std = some patches dominate |
| **Top-10 variance %** | Information captured by first 10 PCA components | 0-100% | Higher = more concentrated representation |
| **Components@90%** | PCA components needed for 90% variance | 1 to embed_dim | Lower = more compressible representation |
| **Linear CKA** | Geometric similarity between two representations | 0 to 1 | 1 = identical geometry; 0 = unrelated |
| **k-NN overlap** | Neighborhood agreement between two models | 0 to 1 | 1 = same neighbors; 0 = completely different |
| **Patch correspondence** | Optimal assignment similarity (Hungarian matching) | 0 to 1 | How well patches can be aligned across models |
| **Isotropy (cosine)** | Spread of embeddings in the representation space | 0 to 1 | Higher = more uniform; near 0 = vectors clustered in a cone |
| **Isotropy (partition)** | Singular value uniformity (Mu et al. 2018) | 0 to 1 | Higher = eigenvalues more uniform; 0 = dominated by top components |
| **Uniformity** | Embedding spread on the unit hypersphere (Wang & Isola 2020) | -inf to 0 | More negative = better spread; 0 = all vectors identical |
| **Intrinsic dimensionality** | True manifold dimension (Levina & Bickel 2004 MLE) | 1+ | Lower than ambient dim = representations lie on a low-dim manifold |

## Validation and trust

Every report includes a validation summary showing whether the model's outputs can be trusted:

- **Validated** — Contract and parity checks pass against approved reference artifacts
- **Stale** — Evidence exists but doesn't match the current model configuration (needs refresh)
- **Unverified** — No evidence available (planned models, or stub backend)

Run `latent-inspector validate --model <name>` to check a model's integration status. Use `--refresh-goldens` to update reference artifacts after a verified ONNX export update.

## How model loading works

1. **First run**: download the ONNX artifact from HuggingFace Hub to `~/.cache/latent-inspector/`
2. **Load**: create an ONNX Runtime session and validate tensor names against the model graph
3. **Preprocess**: resize short edge to target size, center-crop to square, normalize with model-specific mean/std
4. **Extract**: run inference, split the output into patch tokens (and CLS token if available) via the common `ModelOutput` interface

Downloads resume from partial transfers. Cache integrity is verified via SHA-256. Use `latent-inspector models --verbose` to inspect the cache state of every artifact.

To convert a Hugging Face model to ONNX format for use with latent-inspector, use the [ONNX Community Converter](https://huggingface.co/spaces/onnx-community/convert-to-onnx).

## Development

```bash
# Build without ONNX (fast, uses stub backend for development)
cargo build

# Build with real ONNX inference
cargo build --features onnx-inference --release

# Run all tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Full CI pipeline
make all
```

The stub backend (`LATENT_INSPECTOR_MODEL_BACKEND=stub`) produces deterministic synthetic outputs for development and testing without downloading real models. All integration tests use the stub by default.

## License

MIT OR Apache-2.0
