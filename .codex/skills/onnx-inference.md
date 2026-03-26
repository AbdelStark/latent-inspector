---
name: onnx-inference
description: ONNX Runtime integration via the `ort` crate — model loading, session management, feature extraction, intermediate output hooks, and model caching. Activate when working on model loading, inference, preprocessing, or ONNX graph inspection.
prerequisites: ort crate, ONNX model files
---

# ONNX Inference

<purpose>
Covers the ONNX Runtime inference pipeline: downloading models from HuggingFace, caching, creating sessions, preprocessing images, running inference, and extracting intermediate representations (CLS tokens, patch tokens, attention weights).
</purpose>

<context>
— 6 models, each 300MB-2.4GB ONNX files
— Cache location: `~/.cache/latent-inspector/<model_name>.onnx`
— Models downloaded from HuggingFace Hub with SHA-256 verification
— All models normalized to `ModelOutput` struct (see SPECIFICATION.md)
— `ort` crate v2 with `load-dynamic` feature for bundled ONNX Runtime
</context>

<procedure>
### Load a model
1. Check cache: `dirs::cache_dir()/.../latent-inspector/<model>.onnx`
2. If missing: download with reqwest + indicatif progress bar
3. Verify SHA-256 hash against registry
4. Create `ort::Session` with `SessionBuilder::new()?.with_model_from_file(path)?`
5. Validate input/output names match expected model interface

### Preprocess an image for inference
1. Load with `image::open(path)?`
2. Resize to model's expected input (typically 224×224 or 518×518 for DINOv2)
3. Convert to f32, normalize: `(pixel / 255.0 - mean) / std`
4. Reshape to NCHW: `[1, 3, H, W]`
5. Create `ort::Value` from ndarray

### Extract intermediate representations
1. Some models expose attention weights as named outputs — check graph
2. For models that don't: modify ONNX graph to expose intermediate nodes
3. Map output tensors to ModelOutput fields:
   — CLS token: first token of last layer output → `Array1<f32>` [D]
   — Patch tokens: remaining tokens → `Array2<f32>` [N_patches, D]
   — Attention weights: attention output nodes → `Array4<f32>` [L, H, N, N]

### Model-specific preprocessing params
| Model         | Input Size | Mean                    | Std                     |
|---------------|-----------|-------------------------|-------------------------|
| DINOv2 ViT-L  | 518×518   | [0.485, 0.456, 0.406]  | [0.229, 0.224, 0.225]  |
| DINOv3 ViT-L  | 518×518   | [0.485, 0.456, 0.406]  | [0.229, 0.224, 0.225]  |
| MAE ViT-L     | 224×224   | [0.485, 0.456, 0.406]  | [0.229, 0.224, 0.225]  |
| I-JEPA ViT-H  | 224×224   | [0.485, 0.456, 0.406]  | [0.229, 0.224, 0.225]  |
| CLIP ViT-L    | 224×224   | [0.481, 0.458, 0.408]  | [0.269, 0.261, 0.276]  |
| SigLIP SO400M | 384×384   | [0.5, 0.5, 0.5]        | [0.5, 0.5, 0.5]        |

[verify] — confirm exact values against model cards before implementation.
</procedure>

<patterns>
<do>
  — Reuse `ort::Session` across multiple images — session creation is expensive.
  — Use `ort::SessionBuilder` options for thread count: `.with_intra_threads(num_cpus::get())`
  — Store model metadata (input names, output names, dimensions) in the registry, not hardcoded.
  — Validate tensor shapes immediately after extraction — fail fast on mismatches.
  — Drop large tensors (attention weights) after computing metrics to free memory.
</do>
<dont>
  — Don't create a new Session per image — reuse sessions.
  — Don't assume output tensor names — inspect with `session.outputs` and map dynamically.
  — Don't load all 6 models into memory simultaneously unless needed — load on demand.
  — Don't use `ort::Session::run` with wrong input names — match exactly from model graph.
</dont>
</patterns>

<examples>
Example: Basic model loading and inference
```rust
use ort::{Session, SessionBuilder, Value};
use ndarray::Array4;

let session = SessionBuilder::new()?
    .with_model_from_file("path/to/model.onnx")?;

// Prepare input: [1, 3, 224, 224] f32 tensor
let input = Array4::<f32>::zeros((1, 3, 224, 224));
let input_value = Value::from_array(input)?;

let outputs = session.run(ort::inputs!["pixel_values" => input_value]?)?;
let features = outputs["last_hidden_state"].extract_tensor::<f32>()?;
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| "Invalid input name" | Model expects different input tensor name | Inspect `session.inputs` for actual names |
| Output shape unexpected | Model version mismatch | Print shapes, compare with model card |
| Session creation slow | Large model loading | This is expected (1-3s). Cache the Session. |
| ORT library not found | Feature flag issue | Use `ort = { features = ["load-dynamic"] }` |
| Download stalls | Large file + slow network | Implement timeout + retry with Range header resume |
</troubleshooting>

<references>
— src/models/registry.rs: Model metadata and URLs
— src/models/loader.rs: Session creation
— src/models/preprocess.rs: Image normalization
— SPECIFICATION.md: ModelOutput struct definition
</references>
