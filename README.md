# latent-inspector

A fast CLI for inspecting and comparing learned representations across self-supervised vision models. Feed it an image, get a structured comparison of how DINOv2, DINOv3, MAE, I-JEPA, CLIP, and SigLIP see the world.

Current implementation status: Phase 1 is focused on DINOv2 model loading and inspection. The other models are listed in the registry as planned targets for the multi-model milestone and are not loadable yet.

```bash
cargo install latent-inspector

# Compare representations across models
latent-inspector compare photo.jpg --models dinov2-vit-l14,mae-vit-l16,ijepa-vit-h14,clip-vit-l14

# Inspect a single model's representation
latent-inspector inspect photo.jpg --model dinov2-vit-l14 --output report/

# Validate preprocessing, tensor semantics, and reference parity
latent-inspector validate --model dinov2-vit-l14 --format json --output validation/

# Find nearest neighbors across a dataset
latent-inspector neighbors photo.jpg --model dinov2-vit-l14 --dataset imagenet-val/

# Measure representation similarity between two models
latent-inspector similarity --model-a dinov2-vit-l14 --model-b ijepa-vit-h14 --dataset images/
```

## What it does

SSL models learn to represent images in very different ways. DINO learns patch-level features that segment objects without supervision. MAE learns to reconstruct masked regions. I-JEPA predicts in latent space. CLIP aligns images with text. Each approach creates a different internal "view" of the same image.

latent-inspector makes these differences visible and measurable.

### For each model, it computes:

- **Patch-level attention maps** — When a model export includes attention tensors, inspect reports expose patch overlays and attention concentration summaries.
- **Feature PCA projection** — Reduce the high-dimensional representation to 3 RGB channels. Same-color regions have similar features.
- **CLS token similarity** — How does the global representation compare across models?
- **Patch cosine similarity matrix** — Which patches in model A correspond to which patches in model B?
- **Representation rank** — Effective dimensionality of the learned features (higher = more expressive)
- **Feature variance spectrum** — Distribution of information across dimensions (concentrated vs spread)
- **k-NN patch classification** — How well do patches separate semantic categories without fine-tuning?

### Output formats:

- **Terminal** — Rich inline display with colored Unicode blocks (default)
- **PNG** — PCA projections plus comparison heatmaps or inspection variance charts
- **JSON** — Raw metrics plus pairwise overview matrices and highlights
- **HTML** — Interactive report with pairwise matrices, highlights, and embedded validation summaries
  plus copied image previews so exported bundles stay readable away from the
  source dataset

When a command writes files to an output directory, latent-inspector now also
emits `artifacts.json` in that directory. The manifest records the command,
requested format, the command context that produced the bundle, a top-line
summary of the run, primary report path when there is one, every generated
asset, and any attached validation statuses so automation can discover outputs
without hard-coding filenames or reverse-engineering the full report payload.
For HTML exports, the output directory now also includes the equivalent
structured JSON payload alongside the human-readable page, so a single run can
serve both people and downstream automation.

### Validation workflow:

- `validate` checks preprocessing against the approved model contract
- `validate` verifies the exported ONNX tensor name, shape, and CLS semantics
- `validate` compares observed outputs against checked-in aggregate and per-fixture reference evidence
- `compare`, `inspect`, `neighbors`, `similarity`, and `drift` now embed validation summaries in terminal, JSON, and HTML reports

## Why Rust?

Model inference runs via ONNX Runtime (C++ backend). The analysis pipeline (PCA, cosine similarity, k-NN, and attention-aware summaries when exports provide attention tensors) runs in native Rust. Parallel across all models via rayon.

On a MacBook M3 Pro, comparing 5 models on a single image takes ~3 seconds. The equivalent Python pipeline takes ~25 seconds.

For researchers processing thousands of images across multiple models, this matters.

## Supported models

| Model | Architecture | Method | Source | Status |
|-------|-------------|--------|--------|--------|
| DINOv2 | ViT-L/14 | Self-distillation + centering | Meta FAIR | Ready in Phase 1 |
| DINOv3 | ViT-7B (distilled to ViT-L) | Self-distillation + Gram anchoring | Meta FAIR | Planned |
| MAE | ViT-L/16 | Masked autoencoder (reconstruction) | Meta FAIR | Planned |
| I-JEPA | ViT-H/14 | Joint embedding predictive (latent prediction) | Meta FAIR | Planned |
| CLIP | ViT-L/14 | Contrastive image-text | OpenAI | Planned |
| SigLIP | ViT-SO400M/14 | Sigmoid contrastive image-text | Google | Planned |

Models are downloaded automatically on first use (~300MB-2GB each) and cached locally.
If a cache bundle is partial or contains an empty/corrupt artifact, latent-inspector
now refreshes only the missing or invalid files before creating the ONNX session.
Interrupted downloads keep their `.download-part` payloads and resume from the
last completed byte when the host supports HTTP range requests; otherwise the
cache layer falls back to a clean restart automatically.

For CI or isolated local runs, set `LATENT_INSPECTOR_CACHE_DIR=/tmp/latent-inspector-cache`
to override the default cache root.

## Validate a model integration

Run the dedicated validation command whenever you update an export or want to
confirm that a report is still source-aligned:

```bash
cargo run -- validate --model dinov2-vit-l14
cargo run -- validate --model dinov2-vit-l14 --format json --output tmp/validation
cargo run -- validate --model dinov2-vit-l14 --refresh-goldens
```

The validation summary reports preprocessing status, tensor semantics, approved
reference parity, caveats, and a plain-language recommendation for whether the
model is safe to interpret as source-aligned.
`validate` is the only command that re-executes the fixture set against the
active backend. The normal report commands (`compare`, `inspect`, `neighbors`,
`similarity`, `drift`, and `models`) reuse the checked-in approved evidence plus
freshness checks so trust summaries stay fast and deterministic.
If the checked-in contract or reference artifacts no longer match the current
registry profile, the summary now reports `stale` instead of treating outdated
evidence as a fresh pass or a runtime failure.
When the development stub backend is enabled through
`LATENT_INSPECTOR_MODEL_BACKEND=stub`, the command still exercises the report
and fixture plumbing but marks the run as `unverified`; synthetic stub outputs
are not treated as release-grade source-alignment evidence.
`validate --format html --output <dir>` now writes both `validation.html` and
`validation.json` so reviewed evidence bundles stay machine-readable.
If an export emits `NaN`/`Inf` tensors or a non-square patch grid, the analysis
now fails explicitly instead of producing misleading metrics or corrupted PCA
artefacts.

## Example: How different models see a street scene

```
$ latent-inspector compare street.jpg --models dinov2,mae,ijepa,clip

Model Comparison: street.jpg
═══════════════════════════════

                DINOv2-L    MAE-L       I-JEPA-H    CLIP-L
Repr. rank      487/1024    312/1024    445/1024    198/1024
Top-10 var%     23.4%       41.7%       28.1%       62.3%
Patch entropy   6.82        5.91        6.44        4.12
CLS L2 norm     18.4        N/A         16.2        12.7

Cross-model CLS cosine similarity:
             DINOv2  MAE     I-JEPA  CLIP
DINOv2       1.000   -       0.721   0.534
MAE          -       -       -       -
I-JEPA       0.721   -       1.000   0.488
CLIP         0.534   -       0.488   1.000

Attention concentration (Gini coefficient):
DINOv2: 0.72 (focused)   MAE: 0.31 (diffuse)
I-JEPA: 0.58 (moderate)  CLIP: 0.81 (very focused)

[PNG outputs saved to ./compare_street/]
```

## Analysis modes

### `compare` — Side-by-side model comparison
The main command. Takes an image and a list of models. Produces PCA projections, pairwise similarity matrices, highlight summaries, attention concentration metrics when available, and validation-aware reports. When compared models expose different patch grids or incompatible CLS / embedding spaces, `compare` now keeps the dimension-agnostic metrics, marks unsupported metrics as `N/A`, and explains the reason in terminal, JSON, and HTML outputs instead of silently dropping them. Matrix sections now also report how many model pairs were actually comparable for each metric, so mixed-model runs do not imply support that the compared exports do not provide. `--format json` prints the structured compare report to stdout by default or writes `compare.json` when `--output <dir>` is provided. `--format png` writes per-model PCA images plus pairwise heatmaps for CKA, k-NN overlap, and direct patch correspondence. `--format html` now writes `report.html`, the same structured payload as `compare.json`, those companion PNG assets, an input-image preview, and `artifacts.json` in a single bundle.

### `inspect` — Deep dive into a single model
Detailed analysis of one model's representation: rank/entropy metrics, attention concentration when available, dead dimension counts, variance spectrum, validation status, and exportable PCA + variance artefacts. When the backend exposes attention tensors, inspect reports also include an attention summary and an overlay projected back onto the source image. `--format json` prints the structured inspect report to stdout by default or writes `inspect.json` when `--output <dir>` is provided. `--format html` now writes a dedicated single-model report plus `inspect.json`, with the variance-spectrum breakdown, attention summary, validation summary, and linked artefacts instead of falling back to the generic compare layout.

### `neighbors` — k-NN retrieval across a dataset
Given an image and a dataset directory, find the most similar images according to each model. Reveals what each model considers "similar." DINO finds visually similar objects. CLIP finds semantically similar concepts. I-JEPA finds structurally similar scenes.
Dataset-backed commands recurse through nested directories and preserve relative
paths in their reports, so class-folder layouts remain legible in neighbor
lists.
`neighbors` now supports `--format terminal|json|html|png`; JSON prints to
stdout by default or writes `neighbors.json` when `--output <dir>` is provided,
while HTML/PNG emit a shareable report or ranking chart under
`neighbors_output/` (or the requested output directory). Terminal, JSON, and
HTML reports also attach the active model's validation summary so nearest-neighbor
results keep their trust context. If a model does not expose a CLS token, the
command now falls back to a mean-patch image embedding and records that basis in
terminal, JSON, and HTML reports. HTML exports also include the similarity chart
PNG that the standalone `png` surface writes, query/top-match previews, plus the
same structured payload as `neighbors.json`.

### `similarity` — Representation alignment between models
Centered Kernel Alignment (CKA) and mutual k-NN overlap between two models across a dataset. Answers: "How similarly do these two models represent the world?"
`similarity` now supports `--format terminal|json|html|png`; the JSON/HTML
reports include the computed metric set plus dataset processing summary, and the
PNG surface writes a compact metric chart for automation-friendly artifact
capture. HTML exports now embed that chart alongside dataset sample previews in
the report. Terminal, JSON, and HTML outputs also include validation summaries for
both compared models. Report payloads now also state that dataset-level
similarity metrics are computed from mean-patch embeddings, with CLS cosine
surfaced separately when available. HTML bundles also include `similarity.json`.

### `drift` — Track representation changes across checkpoints
Point it at a directory of `.onnx` checkpoints (different training stages). Each file is loaded as its own session while reusing the selected model's registered preprocessing and tensor contract, then the command reports consecutive checkpoint CKA scores across the dataset. This is useful for understanding when representations materially shift during training.
Checkpoint filenames are evaluated in natural numeric order, so names such as
`step-2.onnx` are processed before `step-10.onnx`.
If a supported image file in the dataset is unreadable or corrupt, the command
now skips that file, continues processing the rest of the dataset, and reports
the skipped paths in the terminal summary instead of aborting the whole run.
`drift` also supports `--format terminal|json|html|png`; the structured report
captures checkpoint ordering, aggregate drift highlights, and dataset skip
details, while the PNG output writes a consecutive-CKA chart to disk. HTML
exports embed that chart plus dataset sample previews when at least one
comparison runs. Terminal,
JSON, and HTML outputs now also surface per-checkpoint validation summaries so
training-stage drift is read alongside contract and parity caveats. Dataset-based
drift summaries now explicitly state that checkpoint comparisons use mean-patch
embeddings. HTML bundles also include `drift.json`.

## Dependencies

```toml
[dependencies]
ort = "2"                    # ONNX Runtime bindings
ndarray = "0.16"             # N-dimensional arrays
image = "0.25"               # Image I/O
rayon = "1.10"               # Parallel model inference
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ratatui = "0.29"             # Terminal visualization
crossterm = "0.28"
indicatif = "0.17"           # Progress bars
tracing = "0.1"
thiserror = "2.0"
dirs = "6"                   # Model cache directory
reqwest = { version = "0.12", features = ["blocking"] }  # Model download
```

## How model loading works

1. First run: for a ready model, download the ONNX artifact from HuggingFace Hub to `~/.cache/latent-inspector/`
2. Load via ONNX Runtime and validate the declared input/output tensor names against the graph
3. Preprocess to the model-specific input size and normalization stats
4. Extract patch features and CLS token into the common `ModelOutput` interface

If a download is interrupted mid-transfer, the cache keeps the partial file and
attempts to resume on the next run instead of restarting from zero whenever the
remote host honors byte-range requests.

In the current Phase 1 build, `dinov2-vit-l14` is the only loadable model. The remaining registry entries are intentionally marked as planned so the CLI does not imply support that has not been implemented yet.

Use `latent-inspector models` to inspect the live registry inventory. The
catalog now reports each model's phase status, runtime support
(`onnx-ready` vs `stub-only`), whether the local cache contains the full
artifact bundle, and whether the approved validation evidence is current,
stale, missing, or intentionally withheld for planned integrations.
Use `latent-inspector models --format json` to emit the same catalog as
structured JSON to stdout or `latent-inspector models --format json --output
tmp/models` to write `models.json` for automation. For a shareable report, run
`latent-inspector models --format html --output tmp/models` to generate
`models.html` alongside `models.json`.

## License

MIT OR Apache-2.0
