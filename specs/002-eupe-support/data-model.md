# Data Model: EUPE Registry Entry

## Registry Entry

```
RegistryEntry {
  info: ModelInfo {
    name: "eupe-vit-b16"
    architecture: "ViT-B/16"
    patch_size: 16
    embed_dim: 768
    num_layers: 12
    num_heads: 12
    method: SSLMethod::EUPE
    input_size: 224
    params_m: 86
  }
  availability: Availability::ready(...)
  artifacts: [
    ModelArtifact { relative_path: "eupe-vit-b16/model.onnx", ... }
    ModelArtifact { relative_path: "eupe-vit-b16/model.onnx_data", ... }
  ]
  norm_mean: [0.485, 0.456, 0.406]
  norm_std: [0.229, 0.224, 0.225]
  input_name: "input"                    // determined during export
  output_name: "output"                  // determined during export (combined CLS+patches)
  video_frames: None                     // standard image model
  validation: {
    source: "facebookresearch/eupe"
    tensor: PatchAndClsSequence, cls_expected=true, patch_count=196, embedding_dim=768
  }
}
```

## SSLMethod Extension

```
SSLMethod::EUPE  →  Display: "EUPE"
```

## Tensor Contract

```json
{
  "name": "output",
  "role": "patch-and-cls-sequence",
  "cls_expected": true,
  "batch_size": 1,
  "patch_count": 196,
  "embedding_dim": 768
}
```

## Cross-Model Compatibility Matrix

| Pair | CLS cosine | CKA | k-NN | Patch corr | Notes |
|------|-----------|-----|------|------------|-------|
| EUPE ↔ DINOv2 | **Yes** | Yes (min 196) | Yes (min 196) | No (768≠1024) | First CLS pair! |
| EUPE ↔ I-JEPA | No (I-JEPA no CLS) | Yes (min 196) | Yes (min 196) | No (768≠1280) | |
| EUPE ↔ V-JEPA 2 | No (V-JEPA no CLS) | Yes (min 196) | Yes (min 196) | No (768≠1024) | |
