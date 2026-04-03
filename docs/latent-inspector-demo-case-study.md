# latent-inspector Demo Case Study

Reference material for the launch video. Every number is from real ONNX Runtime inference on `docs/assets/img/samples/elephant_sample_image.jpg`.

---

## Opening — The Hook

**Script**: "Four AI models look at this elephant. They all see something different. Here's how we know."

**Command to run live**:

```bash
latent-inspector compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16
```

**Key visual**: The PCA RGB projections side by side. Each image shows how a model "sees" the elephant — same image, four completely different internal representations.

---

## The Four Models — Who They Are

| Model | What it learned to do | Size | Key architectural choice |
|-------|----------------------|------|--------------------------|
| **DINOv2** | Match a student to a slowly-evolving teacher across augmented views | 304M, 1024-dim | Self-distillation — no labels, no reconstruction |
| **I-JEPA** | Predict missing patches in latent space from visible context | 632M, 1280-dim | Predicts representations, not pixels |
| **V-JEPA 2** | Predict future video frames in latent space | 304M, 1024-dim | Trained on video — sees static images through a temporal lens |
| **EUPE** | Match multiple specialist teachers simultaneously | 86M, 768-dim | Distilled from DINOv2 + depth + segmentation teachers |

**Demo talking point**: "Same ViT backbone, same patch tokenization, same elephant. The only difference is the training objective — what the model was asked to learn. That single choice reshapes the entire geometry of the representation."

---

## The Numbers — Per-Model Metrics

| Metric | DINOv2 | I-JEPA | V-JEPA 2 | EUPE |
|--------|--------|--------|----------|------|
| Effective rank | 60/1024 | 44/1280 | 64/1024 | 17/768 |
| Dead dimensions | 0 | 0 | 0 | 0 |
| Patch entropy | 2.52 | 2.89 | 2.36 | 2.84 |
| CLS L2 norm | 46.3 | N/A | N/A | 56.5 |
| Patch norm mean | 47.5 | 33.8 | 101.0 | 57.3 |
| Patch norm std | 1.4 | 6.1 | 6.7 | 1.5 |
| Top-10 var % | 66.8% | 72.7% | 58.1% | 88.8% |
| Components@90% | 31 | 22 | 38 | 12 |
| Isotropy | 0.796 | 0.788 | 0.678 | 0.026 |
| Uniformity | -2.70 | -2.72 | -2.50 | -0.10 |

### Talking points for each metric

**Effective rank** — "V-JEPA 2 uses the most dimensions (64). EUPE uses the fewest (17). More dimensions doesn't mean better — it means the model spreads information differently."

**Patch entropy** — "I-JEPA creates the most differentiated patches (2.89). Its prediction objective forces each patch to encode unique spatial context. V-JEPA 2 has the lowest (2.36) — trained on video, it sees the elephant as one coherent scene."

**Isotropy** — "This is where it gets wild. DINOv2 and I-JEPA have isotropy around 0.79 — their patches point in diverse directions. EUPE is at 0.026 — nearly zero. All its patches point almost the same way. Multi-teacher distillation collapsed the directional diversity into a compact universal feature."

**Top-10 variance** — "EUPE packs 88.8% of its information into just 10 components. You could throw away 758 of its 768 dimensions and keep most of the signal. DINOv2 needs 31 components for 90%. V-JEPA 2 needs 38. EUPE is the most compressible; V-JEPA 2 is the least."

**Uniformity** — "Wang & Isola's metric. DINOv2 and I-JEPA are around -2.7 — good spread on the unit hypersphere. EUPE is at -0.10 — nearly collapsed. Its patches are clustered in a tiny region of the sphere. This is the cost of universality: one compact feature set for every task."

---

## The PCA Visualizations — "How Models See"

**Script**: "These four images show the same elephant. Each pixel is colored by mapping the model's top 3 PCA components to RGB. Same-color regions have similar representations. Different models, different colors, different perception."

### What to point out in each visualization

**DINOv2**: Clear spatial clustering. The elephant body is one color, the background another. This is the "emergent segmentation" that made DINOv2 famous — the model learns to separate objects without any segmentation labels.

**I-JEPA**: More granular, more colors. The elephant's trunk, body, legs, and background each get distinct representations. I-JEPA's prediction objective forces it to encode fine-grained spatial structure — it needs to know exactly what's next to what.

**V-JEPA 2**: The most colorful. Highest effective rank (64 dimensions) means the most directions in representation space, which means the most distinct colors in the PCA projection. Trained on video, it encodes spatial information through a temporal lens.

**EUPE**: Smooth gradients, less spatial structure. With isotropy at 0.026, all patches point nearly the same direction — the PCA projection shows subtle gradients rather than sharp boundaries. This is what a "universal" representation looks like: general-purpose but not spatially discriminative.

**Demo talking point**: "Notice how DINOv2 naturally segments the elephant from the background — that's emergent object segmentation, no labels needed. Now look at EUPE — it's much smoother. That's the price of universality: when you optimize for every task, you lose the sharp spatial boundaries that single-objective models develop."

---

## Cross-Model Comparison — The CKA Matrix

```
              DINOv2    I-JEPA    V-JEPA 2  EUPE
DINOv2        1.000     0.329     0.358     0.044
I-JEPA        0.329     1.000     0.275     0.000
V-JEPA 2      0.358     0.275     1.000     0.038
EUPE          0.044     0.000     0.038     1.000
```

### Talking points

**Script**: "CKA measures whether two models organize their representations in similar geometric structures. 1.0 means identical; 0.0 means completely unrelated."

**DINOv2 ↔ V-JEPA 2 = 0.358** — "The closest pair. Makes sense — they share the exact same architecture (ViT-L, 24 layers, 1024-dim). Same capacity, different training signal. The architecture constrains the geometry more than we might expect."

**DINOv2 ↔ I-JEPA = 0.329** — "Close second. Despite very different training objectives, they share some geometric structure. Both are Meta FAIR models trained on images — perhaps the ImageNet prior creates common structure."

**I-JEPA ↔ V-JEPA 2 = 0.275** — "The JEPA 'family' is actually the most distant among the three. Despite sharing the JEPA prediction framework, image prediction and video prediction create substantially different geometries."

**Anything ↔ EUPE ≈ 0** — "EUPE is an alien. CKA of 0.044 with DINOv2, 0.000 with I-JEPA, 0.038 with V-JEPA 2. Multi-teacher distillation created a representation that has essentially zero structural similarity to any single-objective model. This is the most surprising finding."

---

## k-NN Overlap — Local Neighborhood Agreement

```
              DINOv2    I-JEPA    V-JEPA 2  EUPE
DINOv2        1.000     0.278     0.205     0.132
I-JEPA        0.278     1.000     0.202     0.100
V-JEPA 2      0.205     0.202     1.000     0.089
EUPE          0.132     0.100     0.089     1.000
```

**Script**: "CKA measures global geometry. k-NN overlap measures local neighborhoods — when DINOv2 says patches A and B are similar, does I-JEPA agree?"

**DINOv2 ↔ I-JEPA = 0.278** — "Highest local agreement. Both use 14px patches at 224px — same spatial granularity. They agree on roughly 1 in 4 nearest neighbors."

**EUPE ↔ V-JEPA 2 = 0.089** — "Lowest. Only 8.9% of local neighborhoods overlap. These models disagree about which patches are similar more than any other pair."

---

## Demo Commands — What to Show

### 1. Compare (terminal — the main demo)

```bash
latent-inspector compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16
```

### 2. HTML report (shareable)

```bash
latent-inspector compare docs/assets/img/samples/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16 \
  --format html --output elephant-report/
```

### 3. Single model deep-dive

```bash
latent-inspector inspect docs/assets/img/samples/elephant_sample_image.jpg \
  --model dinov2-vit-l14
```

### 4. Interactive TUI

```bash
latent-inspector tui docs/assets/img/samples/elephant_sample_image.jpg \
  -m dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16
```

### 5. Validate model integrity

```bash
latent-inspector validate --model dinov2-vit-l14 --model ijepa-vit-h14 \
  --model vjepa2-vitl-fpc2-256 --model eupe-vit-b16
```

### 6. Model registry

```bash
latent-inspector models --verbose
```

---

## The Story Arc for the Video

### Act 1: The Question (30 sec)
"These four AI models look at this elephant. They all produce representations — internal maps of what they see. But do they see the same thing? How would you even measure that?"

### Act 2: The Tool (60 sec)
"latent-inspector is a Rust CLI that runs real ONNX inference on multiple SSL models and compares their latent representations using rigorous metrics from the representation learning literature."

Show: `cargo install latent-inspector` → `compare` command → terminal output.

### Act 3: The PCA Visualization (45 sec)
"These four images are the same elephant seen through four different neural networks. Each patch of the image is colored by the model's top 3 PCA components. Same-color regions = similar representations."

Show: the four 448x448 PCA projections side by side. Point out DINOv2's clean segmentation vs EUPE's smooth gradients.

### Act 4: The Metrics (90 sec)
Walk through the numbers. Focus on:
- Effective rank: spread vs concentration
- Isotropy: 0.796 vs 0.026 (the most dramatic difference)
- CKA matrix: EUPE as an alien (0.000 with I-JEPA)

### Act 5: The Insight (30 sec)
"The training objective — not the architecture, not the data size — is what shapes how a model sees the world. Self-distillation, latent prediction, video prediction, and multi-teacher distillation create four fundamentally different world models from the same input."

### Act 6: The Closer (15 sec)
"latent-inspector is open source, written in Rust, and runs in seconds. Link in the description."

---

## Technical Notes for Credibility

- All models run through ONNX Runtime, not the original PyTorch — this is reproducible by anyone
- V-JEPA 2 is an encoder-only export from a video model; 2 duplicate frames satisfy tubelet_size=2
- EUPE is exported with fp32 RoPE (the original uses bf16 which the ONNX exporter can't handle)
- CKA is linear (kernel = X * X^T), invariant to orthogonal transforms — the gold standard for representation comparison
- k-NN overlap with k=10 measures local structure, complementary to CKA's global view
- PCA projections are normalized per-channel to [0, 255] — colors are relative within each model, not absolute across models
- All validation passes: DINOv2 (73 signals), I-JEPA (45), V-JEPA 2 (45), EUPE (73), zero drift

## What NOT to claim

- The PCA colors don't mean one model is "better" — they show different, not better/worse
- Low isotropy isn't inherently bad — EUPE was designed for efficiency, not spatial diversity
- CKA of 0.0 doesn't mean EUPE is useless — it means it organized information in a completely orthogonal way
- The numbers are from a single image — dataset-level analysis with `profile` would give population statistics
