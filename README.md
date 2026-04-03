# latent-inspector

A fast Rust CLI for inspecting and comparing learned representations across self-supervised vision models. Feed it an image, get a structured comparison of how DINOv2, I-JEPA, V-JEPA 2, and others see the world — with real numbers, not vibes.

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

## Status

> As of 2026-04-03, this project is **alpha**. It is suitable for local CLI-driven inspection, comparison, validation, and report generation for the four ready models (`dinov2-vit-l14`, `ijepa-vit-h14`, `vjepa2-vitl-fpc2-256`, `eupe-vit-b16`). It is **not yet suitable** as a stable library API, unattended production batch infrastructure, or `cargo install` from crates.io. Known limitations: planned models are still stub-only for development flows, first-use downloads are large, and the internal Rust/TUI surfaces may still change before `1.0`. The machine-readable report bundle surface for the ready-model commands is now treated as additive-only; see [docs/REPORT-SCHEMA.md](docs/REPORT-SCHEMA.md).

## Project docs

- [Architecture and runbook](docs/ARCHITECTURE.md)
- [Report schema contract](docs/REPORT-SCHEMA.md)
- [Contributing guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Agent context](AGENTS.md)
- [Example case study](docs/latent-inspector-demo-case-study.md)

## Quick start

```bash
git clone https://github.com/AbdelStark/latent-inspector.git
cd latent-inspector
cargo build --release

# List models and cache state
./target/release/latent-inspector models

# Compare three models on a single image (models auto-download on first use)
./target/release/latent-inspector compare photo.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256

# Deep-dive into one model
./target/release/latent-inspector inspect photo.jpg --model dinov2-vit-l14

# Interactive TUI (real analysis when an image is provided)
./target/release/latent-inspector tui photo.jpg -m dinov2-vit-l14,ijepa-vit-h14

# Profile a model over a dataset (isotropy, uniformity, intrinsic dimensionality)
./target/release/latent-inspector profile --model dinov2-vit-l14 --dataset images/

# Development-only stub backend (no model downloads, validation downgraded to unverified)
LATENT_INSPECTOR_MODEL_BACKEND=stub \
  ./target/release/latent-inspector compare photo.jpg \
  --models dinov2-vit-l14,clip-vit-l14
```

## Why this exists

Self-supervised learning (SSL) models learn to represent images without labels, but they do so in fundamentally different ways:

- **DINOv2** learns patch-level features via self-distillation. Its representations naturally segment objects — patches on the elephant cluster together, patches on the background cluster together — without ever seeing a segmentation label.
- **I-JEPA** predicts missing patches in latent space (not pixel space). It learns to fill in what's "probably there" based on context, favoring abstract structure over texture.
- **V-JEPA 2** extends JEPA to video, learning spatiotemporal structure from internet-scale video data. Even on static images, its representations carry an implicit prior about how the world moves.
- **EUPE** distills multiple specialist teachers (DINOv2, depth estimators, segmenters) into a single compact encoder. Its representation is a learned compromise — strong on classification, segmentation, and depth simultaneously.
- **MAE** reconstructs masked pixel regions. It must encode enough detail to literally redraw the masked patches.
- **CLIP** aligns images with text descriptions. Its representation is shaped by language, not just visual similarity.

These different training objectives create different internal "world models." latent-inspector makes those differences visible, measurable, and comparable with concrete metrics.

## Supported models

| Model | Architecture | Params | Method | Status |
|-------|-------------|--------|--------|--------|
| **DINOv2** | ViT-L/14 | 304M | Self-distillation + centering | **Ready** |
| **I-JEPA** | ViT-H/14 | 632M | Joint embedding predictive | **Ready** |
| **V-JEPA 2** | ViT-L/16 | 304M | Video joint embedding predictive | **Ready** |
| **EUPE** | ViT-B/16 | 86M | Multi-teacher distillation | **Ready** |
| DINOv3 | ViT-L/14 | 304M | Self-distillation + Gram anchoring | Planned |
| MAE | ViT-L/16 | 304M | Masked autoencoder | Planned |
| CLIP | ViT-L/14 | 304M | Contrastive image-text | Planned |
| SigLIP | ViT-SO400M/14 | 400M | Sigmoid contrastive image-text | Planned |

Models download automatically on first use (~1-2 GB each) and are cached locally. All downloads are SHA-256 verified and now retry bounded transient HTTP/read failures before surfacing an error. Override the cache location with `LATENT_INSPECTOR_CACHE_DIR`.

## PCA examples across the ready models

These PCA RGB maps come from the checked-in example outputs in `docs/assets/img/examples/`. Each pixel block is a patch token projected onto the top three principal components, so contiguous color regions mean the model is grouping those patches into a similar representation neighborhood.

<table>
<tr>
<td width="50%"><img src="docs/assets/img/examples/dinov2-vit-l14_pca.png" alt="DINOv2 PCA representation"/><br/><sub><strong>DINOv2</strong> — large contiguous regions show a segmentation-like representation: similar colors mean the model has grouped semantically related elephant and background patches together.</sub></td>
<td width="50%"><img src="docs/assets/img/examples/ijepa-vit-h14_pca.png" alt="I-JEPA PCA representation"/><br/><sub><strong>I-JEPA</strong> — finer local variation reflects its latent-prediction objective: the PCA map suggests patches carry more context-specific information instead of collapsing into broad semantic zones.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/img/examples/vjepa2-vitl-fpc2-256_pca.png" alt="V-JEPA 2 PCA representation"/><br/><sub><strong>V-JEPA 2</strong> — structured but less static-looking regions hint at a motion-aware prior: the PCA layout suggests the encoder organizes the image as something that could evolve over time, not just a frozen scene.</sub></td>
<td width="50%"><img src="docs/assets/img/examples/eupe-vit-b16_pca.png" alt="EUPE PCA representation"/><br/><sub><strong>EUPE</strong> — the more compressed, high-contrast partitioning matches its multi-teacher distillation setup: the PCA view suggests a compact representation that prioritizes strong task-relevant separations.</sub></td>
</tr>
</table>

<details>
<summary><strong>Model provenance and ONNX artifacts</strong></summary>

Every model runs through ONNX Runtime. The ONNX artifacts are sourced as follows:

| CLI name | Original checkpoint | ONNX source | Paper |
|----------|-------------------|-------------|-------|
| `dinov2-vit-l14` | [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large) | [`onnx-community/dinov2-large`](https://huggingface.co/onnx-community/dinov2-large) — community export | [Oquab et al. 2024](https://arxiv.org/abs/2304.07193) |
| `ijepa-vit-h14` | [`facebook/ijepa_vith14_1k`](https://huggingface.co/facebook/ijepa_vith14_1k) | [`onnx-community/ijepa_vith14_1k`](https://huggingface.co/onnx-community/ijepa_vith14_1k) — community export | [Assran et al. 2023](https://arxiv.org/abs/2301.08243) |
| `vjepa2-vitl-fpc2-256` | [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | [`abdelstark/vjepa2-vitl-fpc2-256-onnx`](https://huggingface.co/abdelstark/vjepa2-vitl-fpc2-256-onnx) — custom export | [Bardes et al. 2024](https://arxiv.org/abs/2506.09985) |
| `eupe-vit-b16` | [`facebook/EUPE-ViT-B`](https://huggingface.co/facebook/EUPE-ViT-B) | [`abdelstark/eupe-vit-b16-onnx`](https://huggingface.co/abdelstark/eupe-vit-b16-onnx) — custom export | [Zhu et al. 2026](https://arxiv.org/abs/2603.22387) |

**V-JEPA 2 ONNX export:** V-JEPA 2 is a video model (`facebook/vjepa2-vitl-fpc64-256`) trained on internet-scale video. Since latent-inspector analyzes single images, we exported only the encoder (stripping the predictor head), with a fixed 2-frame input — the single image is duplicated to satisfy the model's temporal tubelet requirement (`tubelet_size=2`). This produces 256 spatial patch tokens of dimension 1024, identical in shape to DINOv2, enabling direct cross-model comparison. The export was done using PyTorch's TorchScript ONNX exporter at opset 14, simplified with [onnxsim](https://github.com/daquexian/onnx-simplifier), and verified against the PyTorch reference (max diff < 0.003). The artifact is hosted at [`abdelstark/vjepa2-vitl-fpc2-256-onnx`](https://huggingface.co/abdelstark/vjepa2-vitl-fpc2-256-onnx) for convenience.

**EUPE ONNX export:** EUPE (`facebook/EUPE-ViT-B`) is a compact ViT-B/16 distilled from multiple domain-expert teachers. The `forward_features()` method returns normalized CLS and patch tokens as separate dict entries; we wrapped it to concatenate them into a single `[1, 197, 768]` tensor (CLS at index 0). RoPE position encoding was cast from BFloat16 to Float32 for export compatibility. The export uses TorchScript at opset 14 + onnxsim (834 nodes, max diff 0.0003 vs PyTorch). Hosted at [`abdelstark/eupe-vit-b16-onnx`](https://huggingface.co/abdelstark/eupe-vit-b16-onnx).

To convert other HuggingFace models to ONNX, use the [ONNX Community Converter](https://huggingface.co/spaces/onnx-community/convert-to-onnx).

</details>

---

## Case study: How DINOv2 and I-JEPA see an elephant

This walkthrough uses a real elephant photograph to show what latent-inspector reveals about two fundamentally different SSL approaches. Every number below is from an actual ONNX inference run — not fabricated.

### Compare both models

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
Patch isotropy        0.712           0.834
Patch uniformity      -2.891          -3.247
================================================================================
```

<details>
<summary><strong>What these numbers mean</strong></summary>

**Representation rank** (60 vs 44): How many dimensions the model actually uses. DINOv2 spreads information across 60 effective dimensions out of 1024. I-JEPA uses only 44 out of 1280. Neither wastes capacity (zero dead dimensions), but I-JEPA is more concentrated.

**Patch entropy** (2.52 vs 2.89): How differentiated the patch representations are. I-JEPA creates more distinct per-patch features because its prediction objective forces fine-grained spatial encoding. DINOv2's self-distillation favors globally consistent features.

**CLS L2 norm** (46.3 vs N/A): DINOv2 exports a CLS token (a single vector summarizing the whole image). I-JEPA doesn't — it was never designed with one. latent-inspector reports `N/A` explicitly rather than silently dropping the metric.

**Top-10 variance / Components@90%**: I-JEPA packs 72.7% of variance into 10 components and needs only 22 for 90%. DINOv2 is more spread out (66.8% / 31). I-JEPA's representation is lower-dimensional in practice despite having a wider embedding space.

**Isotropy** (0.712 vs 0.834): How directionally diverse the patch embeddings are (1 = perfectly isotropic, 0 = all patches point the same way). I-JEPA's patches are more directionally diverse — each patch represents something more distinct.

**Uniformity** (-2.891 vs -3.247): Wang & Isola (2020) metric for how evenly patches spread on the unit hypersphere. More negative = better spread. I-JEPA distributes patches more uniformly, consistent with its latent-prediction objective that naturally prevents representational collapse.

</details>

### Cross-model similarity

```
Linear CKA:     0.329    (representation geometry overlap)
k-NN overlap:   0.278    (fraction of shared nearest neighbors)
```

CKA of 0.329 means the two models have some structural overlap but organize the elephant's representation in substantially different ways. k-NN overlap of 27.8% means when DINOv2 considers two patches "similar," I-JEPA often disagrees — the trunk patches might cluster with body patches in one model but with boundary patches in the other.

### Key takeaway

| Property | DINOv2 | I-JEPA | Interpretation |
|----------|--------|--------|----------------|
| Effective rank | 60/1024 | 44/1280 | DINOv2 uses more dimensions |
| Variance concentration | 66.8% in top 10 | 72.7% in top 10 | I-JEPA is more concentrated |
| Patch entropy | 2.52 | 2.89 | I-JEPA differentiates patches more |
| Patch isotropy | 0.712 | 0.834 | I-JEPA spreads more uniformly |
| CLS token | Yes (46.3 norm) | No | Different architectures |
| CKA | — | 0.329 | Genuinely different world models |

---

<details>
<summary><h2>Commands reference</h2></summary>

### `compare` — Side-by-side model comparison

```bash
latent-inspector compare <image> --models <model1>,<model2>[,...]
  [--format terminal|json|html|png] [--output <dir>] [--pca-components <n>]
```

Computes per-model metrics and pairwise cross-model similarity. Handles mismatched architectures gracefully: dimension-agnostic metrics (CKA, k-NN) are computed when patch counts match; dimension-dependent metrics (patch correspondence) and CLS-dependent metrics are reported as `N/A` with an explanation.

### `inspect` — Single model deep-dive

```bash
latent-inspector inspect <image> --model <model>
  [--format terminal|json|html|png] [--output <dir>] [--pca-components <n>]
```

Full representation analysis: rank, entropy, variance spectrum, patch norm statistics, isotropy, uniformity, attention concentration (when available), and PCA projection.

### `neighbors` — k-NN retrieval across a dataset

```bash
latent-inspector neighbors <image> --model <model> --dataset <dir>
  [--k <n>] [--format terminal|json|html|png] [--output <dir>]
```

Find the k most similar images according to the model. Reveals what a model considers "similar." Falls back to mean-patch embeddings when no CLS token is available.

### `similarity` — Cross-model alignment on a dataset

```bash
latent-inspector similarity --model-a <model> --model-b <model> --dataset <dir>
  [--format terminal|json|html|png] [--output <dir>]
```

Dataset-level CKA, k-NN overlap, and (when both models expose CLS) mean CLS cosine similarity. Parallel inference across the dataset.

### `profile` — Representation space profiling

```bash
latent-inspector profile --model <model> --dataset <dir>
  [--format terminal|json|html|png] [--output <dir>]
```

Dataset-level representation fingerprint: isotropy (cosine + partition function), uniformity (Wang & Isola 2020), intrinsic dimensionality (Levina & Bickel 2004 MLE), plus per-image metric aggregates (mean/std/min/max).

### `drift` — Track representation changes across checkpoints

```bash
latent-inspector drift --model <model> --checkpoints <dir> --dataset <dir>
  [--format terminal|json|html|png] [--output <dir>]
```

Load `.onnx` checkpoints from different training stages, compute consecutive CKA scores. Shows when representations materially shift during training. Natural numeric ordering (`step-2.onnx` before `step-10.onnx`).

### `models` — Registry and cache status

```bash
latent-inspector models [--verbose] [--download <model>]
  [--format terminal|json|html] [--output <dir>]
```

Model registry with status, readiness, cache state, evidence status, artifact inventory. Use `--download <model>` to pre-cache.

### `validate` — Preprocessing and parity checks

```bash
latent-inspector validate --model <model>
  [--format terminal|json|html] [--output <dir>] [--refresh-goldens]
```

Validates integration against checked-in contract and reference artifacts. Use `--refresh-goldens` after a verified ONNX update.

### `tui` — Interactive terminal UI

```bash
latent-inspector tui [<image>] [-m <model1>,<model2>,...]
```

Interactive views: dashboard, inspector, compare, spectrum, file browser, help. Arrow keys to navigate, number keys to switch views.

</details>

<details>
<summary><h2>Output formats</h2></summary>

Every analysis command supports four output formats:

| Format | Flag | Output | Use case |
|--------|------|--------|----------|
| **Terminal** | `--format terminal` (default) | Rich Unicode, ASCII fallback | Interactive exploration |
| **JSON** | `--format json` | Structured metrics to stdout or file | Automation, scripting |
| **HTML** | `--format html` | Self-contained report bundle | Sharing, review |
| **PNG** | `--format png` | PCA projections, heatmaps, charts | Presentations, papers |

When `--output <dir>` is provided, all formats also emit `artifacts.json` — a machine-readable manifest of every generated file with byte sizes and SHA-256 digests. HTML bundles include companion JSON. The stable file names and top-level JSON keys for these outputs are documented in [docs/REPORT-SCHEMA.md](docs/REPORT-SCHEMA.md).

Force ASCII output: `LATENT_INSPECTOR_FORCE_ASCII=1`.

</details>

<details>
<summary><h2>Metrics glossary</h2></summary>

| Metric | What it measures | Range | Intuition |
|--------|-----------------|-------|-----------|
| **Effective rank** | Significant singular values | 1 to embed_dim | Higher = uses more capacity |
| **Dead dimensions** | Zero-valued embedding dims | 0 to embed_dim | Should be 0 |
| **Patch entropy** | Diversity of patch features (k-means) | 0 to log2(k) | Higher = more differentiated |
| **Attention Gini** | Attention weight concentration | 0 to 1 | Higher = more focused |
| **CLS L2 norm** | Global image vector magnitude | 0+ | Cross-image comparison |
| **Patch norm mean/std** | Patch vector magnitude distribution | 0+ | Low std = uniform activation |
| **Top-10 variance %** | Info in first 10 PCA components | 0-100% | Higher = more concentrated |
| **Components@90%** | PCA components for 90% variance | 1 to embed_dim | Lower = more compressible |
| **Linear CKA** | Representation geometry similarity | 0 to 1 | 1 = identical geometry |
| **k-NN overlap** | Neighborhood agreement | 0 to 1 | 1 = same neighbors |
| **Patch correspondence** | Hungarian-matched patch similarity | 0 to 1 | Optimal alignment quality |
| **Isotropy (cosine)** | Embedding directional spread | 0 to 1 | Higher = more uniform |
| **Isotropy (partition)** | Singular value uniformity | 0 to 1 | Higher = less top-heavy |
| **Uniformity** | Hypersphere spread (Wang & Isola 2020) | -inf to 0 | More negative = better |
| **Intrinsic dim** | Manifold dimension (Levina & Bickel 2004) | 1+ | Lower than ambient = compressed |

</details>

<details>
<summary><h2>Going deeper: from pixels to world models</h2></summary>

This section explains the full pipeline — what happens from the moment you feed an image to latent-inspector until you get a comparison of how different models perceive the world. It maps concepts to code and gives you the mental model to interpret the results.

### The representation pipeline

Every vision transformer takes an image and produces a set of **patch embeddings**: one high-dimensional vector per spatial region of the image. Here's how latent-inspector processes them:

```
Image (e.g. 224×224 RGB)
  │
  ├─ Resize short edge to model's input size, center-crop to square
  │  (src/models/preprocess.rs — matches torchvision's standard ViT pipeline)
  │
  ├─ Normalize: (pixel / 255 - mean) / std  per channel
  │  (model-specific mean/std from the registry)
  │
  ├─ ONNX Runtime inference
  │  (src/models/loader.rs → ort crate → C++ ONNX Runtime backend)
  │
  └─ Output: [1, seq_len, embed_dim] tensor
     │
     ├─ CLS token (index 0) if present  →  global image representation
     └─ Patch tokens (the rest)         →  per-region representations
```

The key insight: **the patch tokens are the representation**. Each one is a point in a high-dimensional space (1024-dim for DINOv2, 1280-dim for I-JEPA). The geometry of these points — how they cluster, how they spread, how they relate to each other — is what defines the model's "perception" of the image.

### What makes models different

The training objective shapes the geometry. Consider our elephant image:

**DINOv2** (self-distillation): A student network learns to match a slowly-evolving teacher network's representations across different augmented views of the same image. This creates a consistency pressure: patches in similar semantic regions (elephant body, grass, sky) get pushed toward similar representations. The result is a representation that naturally segments the image — without ever seeing a segmentation label.

**I-JEPA** (latent prediction): Given some visible patches, predict the representation of masked patches. Unlike MAE (which predicts pixels), I-JEPA predicts in representation space, so it must learn abstract structure. This creates a different pressure: each patch must encode enough context about its neighborhood to predict what's missing. The result is higher patch entropy (2.89 vs 2.52) — each patch carries more unique information.

**V-JEPA 2** (video prediction): Trained on video, it predicts future frame representations from past frames. Even on a static image, its encoder carries an implicit prior about how the visual world moves and changes. It sees the elephant as something that could move — not just a static pattern.

### How we compare them

Once we have patch tokens from two models, we need to answer: **do these models see the world the same way?**

The problem is that the embedding spaces are different — DINOv2's 1024 dimensions and I-JEPA's 1280 dimensions don't correspond to each other. You can't just subtract them. Instead, we compare **structural properties**:

**CKA (Centered Kernel Alignment)** — `src/analysis/cka.rs`

Build a kernel matrix for each model: K[i,j] = dot(patch_i, patch_j). This captures the pairwise similarity structure — which patches are similar to which, regardless of the absolute coordinate system. Center both kernel matrices (subtract row/column means), then measure how aligned they are via HSIC (Hilbert-Schmidt Independence Criterion). CKA = 1 if the similarity structures are identical; 0 if they're unrelated.

The math: `CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))`

This is invariant to orthogonal transformations and isotropic scaling, so it compares the geometric structure, not the coordinate system.

**k-NN overlap** — `src/analysis/knn.rs`

For each patch, find its 10 nearest neighbors in model A's space and in model B's space. Count how many neighbors overlap. If DINOv2 thinks patches 3, 7, 12 are similar (they're all on the elephant's trunk), does I-JEPA agree? k-NN overlap of 0.278 means only 27.8% agreement — substantial disagreement about what constitutes "similar."

**Patch correspondence** — `src/analysis/correspondence.rs`

When embedding dimensions match (e.g., DINOv2 and V-JEPA 2 both produce 1024-dim), we can compute cosine similarity between every patch pair and find the optimal assignment using the Hungarian algorithm. This tells us whether there's a clean mapping between the two models' patch representations, or whether they've organized the space in incompatible ways.

### Per-model health metrics

Beyond cross-model comparison, each model's representation has intrinsic properties that reveal its quality:

**Effective rank** — `src/analysis/rank.rs`

Compute singular values of the patch matrix, threshold at 1% of the maximum, count how many survive. This is the effective dimensionality — how many independent directions the model uses. A rank of 60/1024 means the model uses 60 directions effectively and the other 964 carry negligible information. Not wasteful — just concentrated.

**PCA variance spectrum** — `src/analysis/variance.rs` and `src/analysis/pca.rs`

Run PCA via the power method (no LAPACK dependency) on the centered patch matrix. The eigenvalue ratios show how information distributes across components. A steep scree plot (most variance in few components) means the representation is compressible. A flat plot means the model spreads information uniformly. Both can be useful — it depends on the downstream task.

**Isotropy and uniformity** — `src/analysis/isotropy.rs`

Two complementary views of representation quality:
- Isotropy (1 - mean pairwise cosine similarity): are patch vectors directionally diverse, or are they all clustered in a narrow cone? High isotropy means each patch points in a distinct direction.
- Uniformity (Wang & Isola 2020): log of average pairwise Gaussian kernel on the unit hypersphere. Measures whether embeddings are evenly spread across the sphere. More negative = better coverage. Representations that collapse to a few modes will have uniformity near 0.

**Patch entropy** — `src/analysis/entropy.rs`

Run k-means clustering on patch tokens, compute Shannon entropy of the cluster assignment distribution. High entropy = patches spread across many clusters = the model creates diverse representations. Low entropy = most patches land in the same cluster = less discriminative.

### The video model trick

V-JEPA 2 expects video input: `[batch, frames, channels, height, width]`. For single-image analysis, we duplicate the frame (`src/models/loader.rs:infer()` → `run_video()`). With `tubelet_size=2`, the minimum is 2 identical frames, which collapses the temporal dimension to a single step, yielding pure spatial patch tokens. This is a valid encoding — the model's spatial pathway processes the image normally; the temporal pathway simply sees no motion.

### The trust pipeline

Every report embeds a validation summary (`src/validation/`). Before trusting a model's metrics, latent-inspector checks:
1. **Preprocessing contract**: does the registered resize/crop/normalize match the checked-in golden artifact?
2. **Tensor semantics**: does the ONNX graph expose the expected input/output names and shapes?
3. **Reference parity**: does the current output match previously approved reference outputs within tolerance?

Models with `validated` status have passed all three checks against ONNX Runtime inference. Models with `stale` status have reference artifacts that were generated by a different backend (e.g., stub). Models with `unverified` have no reference artifacts yet.

### Code map

```
src/
  models/
    registry.rs      All model metadata: architecture, normalization, tensor contracts
    loader.rs        ONNX session creation, inference (image + video paths), stub backend
    preprocess.rs    Resize + center-crop + normalize → [1, 3, H, W] tensor
    cache.rs         Download, SHA-256 verify, partial-resume, cache state
  extract/
    features.rs      Split ModelOutput → CLS token + patch tokens + attention maps
  analysis/
    pca.rs           Power method PCA (no LAPACK needed)
    cka.rs           Linear CKA + CLS cosine similarity
    knn.rs           Cosine similarity matrix, top-k neighbors, overlap
    rank.rs          Effective rank via singular value thresholding
    variance.rs      PCA variance spectrum (scree plot data)
    entropy.rs       k-means + Shannon entropy, patch norm statistics
    isotropy.rs      Cosine isotropy, partition function isotropy, uniformity
    attention.rs     Gini coefficient on attention weights
    correspondence.rs  Hungarian-matched patch correspondence
  viz/
    terminal.rs      Rich Unicode terminal output (with ASCII fallback)
    json.rs          Structured JSON for automation
    html.rs          Self-contained HTML report bundles
    png.rs           PCA RGB projections, heatmaps, variance charts
  validation/
    evidence.rs      Freshness checks against golden fixtures
    parity.rs        Output-level comparison against reference artifacts
```

</details>

<details>
<summary><h2>Development</h2></summary>

```bash
# Build the release binary
cargo build --release

# Run commands without downloading models
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- models
LATENT_INSPECTOR_MODEL_BACKEND=stub cargo run -- compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14

# Run all tests
cargo test

# Lint + format
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings

# Coverage gate used by CI (excludes the currently untested TUI surface)
cargo llvm-cov --workspace \
  --ignore-filename-regex '(^|/)src/tui/|(^|/)src/cli/tui.rs$' \
  --fail-under-lines 85 \
  --fail-under-functions 80 \
  --summary-only

# Build artifact used by CI
cargo build --release

# Full CI pipeline
make all
```

The stub backend (`LATENT_INSPECTOR_MODEL_BACKEND=stub`) produces deterministic synthetic outputs for development and testing without downloading real models. Validation summaries explicitly downgrade stub-backed results to `unverified`. The TUI launches with demo data when no image is provided; when an image is provided it runs the same live analysis pipeline as the CLI.

</details>

## Known limitations

- Only four models are ready for live ONNX-backed analysis today. Planned models remain in the registry for roadmap visibility and stub-backed development flows.
- The project is a CLI/TUI application first. The internal Rust modules are reusable, but the crate does not yet promise a stable public library API.
- Model downloads are large and happen lazily on first use. Cache state is machine-local and validated with SHA-256 checksums.
- Validation goldens are checked into the repository and must only be refreshed after a verified model artifact or contract change.
- Coverage enforcement currently excludes `src/tui/` and `src/cli/tui.rs` until the interactive surface has dedicated tests.

## Roadmap

1. **Expand automated coverage**: add dedicated tests for the interactive TUI paths and tighten the coverage gate once that surface is exercised.
2. **Harden batch workflows further**: improve dataset-scale ergonomics, cache prewarming, and operational guidance for longer-running comparison/profile jobs.
3. **Promote planned models to ready**: wire real ONNX inference, validation contracts, and parity evidence for DINOv3, MAE, CLIP, and SigLIP.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, verification commands, validation-golden rules, and the contribution checklist. Autonomous contributors should also read [AGENTS.md](AGENTS.md) before making changes.

## Help

Use the GitHub issue tracker at <https://github.com/AbdelStark/latent-inspector/issues> for bugs, documentation gaps, and model-integration requests.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

## License

MIT OR Apache-2.0
