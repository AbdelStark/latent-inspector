# EUPE ONNX Export Procedure (Reproducible)

This document defines the **canonical** export workflow for `facebook/EUPE-ViT-B`.

It is designed to avoid the previous failure mode where ONNX outputs became nearly input-independent and produced misleading benchmarks.

## Source references used for this procedure

- **EUPE model card** (`facebook/EUPE-ViT-B`) states:
  - the model exposes class + patch tokens through `forward_features()` as `x_norm_clstoken` and `x_norm_patchtokens`.
  - for 224×224 input, the expected token count is 197 (1 CLS + 196 patches).
- **EUPE paper** (arXiv:2603.22387v2, Mar 31, 2026) frames training as a **proxy-teacher** pipeline (scale up to large proxy, then distill to efficient student), not direct multi-teacher-to-student distillation.
- **PyTorch ONNX exporter docs** recommend `torch.onnx.export(..., dynamo=True)` as the modern path.
- **ONNX docs** recommend verifying graph validity via `onnx.checker.check_model`.
- **ONNX Runtime docs** show explicit provider/session usage for reproducible execution.

## Script

Use the repository script:

```bash
python scripts/export_eupe_onnx.py \
  --model-id facebook/EUPE-ViT-B \
  --output artifacts/eupe-vit-b16/model.onnx \
  --validation-images docs/assets/img/samples \
  --max-images 5 \
  --atol 1e-3 --rtol 1e-3
```

The script performs all required steps:

1. Loads EUPE from Hugging Face with `trust_remote_code=True`.
2. Wraps `forward_features()` and concatenates
   `[x_norm_clstoken, x_norm_patchtokens] -> [B, 197, 768]`.
3. Exports with `torch.onnx.export(..., dynamo=True, external_data=True)`.
4. Runs `onnx.checker.check_model` on the exported graph.
5. Validates ONNX vs PyTorch on **5 images** with:
   - max absolute difference,
   - mean absolute difference,
   - cosine similarity,
   - strict `np.allclose` gate (`atol`/`rtol` configurable; default `1e-3`).
6. Runs an **input-independence gate**:
   - `cos(ONNX(zeros), ONNX(random)) < 0.85` must hold.
7. Writes a JSON validation report next to the model (or `--report`).

## Publish criteria

Do **not** publish a new ONNX artifact unless all are true:

- `validation_passed == true` in the report.
- Every image record has `allclose_pass == true`.
- `input_independence_cosine < input_independence_threshold`.

## Notes

- Default runtime is CPU ONNX Runtime to keep parity checks deterministic across environments.
- `onnxsim` is optional; if unavailable, export still proceeds and validates.
- If you change input size from 224, keep it aligned with downstream contracts/benchmarks.
