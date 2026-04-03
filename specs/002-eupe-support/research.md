# Research: EUPE ONNX Export

## EUPE Model Architecture

**Decision**: Use `facebook/EUPE-ViT-B` (ViT-B/16, 86M params)

| Property | Value |
|----------|-------|
| Architecture | ViT-B/16 |
| Hidden size | 768 |
| Layers | 12 |
| Heads | 12 |
| Patch size | 16 |
| Input size | 224 (but also supports 256 per docs) |
| Params | 86M |
| CLS token | Yes (`x_norm_clstoken`) |
| Patches at 224px | 196 (14x14 grid) |
| Output keys | `x_norm_clstoken`, `x_norm_patchtokens` |
| Normalization | ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

## ONNX Export Strategy

**Decision**: Export via `forward_features` method, TorchScript legacy exporter at opset 14, simplify with onnxsim, external data format.

**Rationale**: This matches the V-JEPA 2 pattern that works with `ort 2.0.0-rc.12`. The dynamo exporter produces models the ort crate can't parse. The legacy TorchScript exporter + onnxsim produces compatible graphs.

**Alternatives considered**:
- Dynamo exporter at opset 18: produces newer protobuf format, ort can't load
- optimum-onnx: EUPE is custom architecture, not supported by optimum
- Direct `forward()` instead of `forward_features()`: returns unnormalized tokens

## Loading Method

**Decision**: Use `torch.hub.load()` with the EUPE repo, then call `forward_features()`

The model is loaded from HuggingFace via the DINOv3 library that EUPE builds on. The `forward_features` method returns normalized tokens (L2-normalized CLS and patch outputs).

## ONNX Output Naming

**Decision**: Map `x_norm_clstoken` and `x_norm_patchtokens` to a single combined output tensor.

The ONNX export needs to produce a single `[1, 197, 768]` tensor (CLS at index 0, patches at 1-196) to match the `PatchAndClsSequence` tensor role in the registry. This may require a wrapper module that concatenates the two outputs.

## Input Size

**Decision**: Use 224px input (not 256px)

The HuggingFace card mentions 256px but the standard ViT-B/16 uses 224px with 14x14 = 196 patches. Using 224px keeps the patch count at 196 and matches the contract.

## Cache Path

**Decision**: `eupe-vit-b16/model.onnx` + `eupe-vit-b16/model.onnx_data`

Follows the subdirectory convention used by I-JEPA and V-JEPA 2 for models with external data.

## HuggingFace Hosting

**Decision**: Upload to `abdelstark/eupe-vit-b16-onnx`

Same pattern as V-JEPA 2: custom ONNX export hosted on the maintainer's HuggingFace space.
