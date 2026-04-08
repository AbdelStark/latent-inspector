# EUPE ONNX Export Procedure

This document defines the canonical export workflow for `facebook/EUPE-ViT-B`.

The goal is simple: ship an ONNX artifact that is demonstrably input-sensitive and remains close to the upstream PyTorch model on real images.

## Ground truth

- The Hugging Face repo for `facebook/EUPE-ViT-B` ships a `.pt` checkpoint, not a Transformers `AutoModel`.
- The official load path is the upstream `facebookresearch/eupe` torch.hub entrypoint `eupe_vitb16`.
- `forward_features()` exposes `x_norm_clstoken` and `x_norm_patchtokens`.
- The paper describes EUPE as a proxy-distilled student, not direct multi-teacher-to-student distillation.

## Export command

```bash
python3 scripts/export_eupe_onnx.py \
  --output artifacts/eupe-vit-b16/model.onnx \
  --validation-images docs/assets/img/samples \
  --max-images 5
```

By default the script:

1. Downloads `EUPE-ViT-B.pt` from Hugging Face.
2. Loads the model through `torch.hub.load("facebookresearch/eupe:main", "eupe_vitb16", ...)`.
3. Wraps `forward_features()` into a single `[B, 197, 768]` tensor:
   `[x_norm_clstoken, x_norm_patchtokens]`.
4. Exports with the legacy TorchScript ONNX path (`dynamo=False`).
5. Rewrites the result as an ONNX external-data bundle:
   `model.onnx` + `model.onnx_data`.
6. Checks the graph with `onnx.checker.check_model`.
7. Validates ONNX vs PyTorch on 5 images.
8. Runs an input-independence gate:
   `cos(ONNX(zeros), ONNX(random)) < 0.85`.

## Why the legacy exporter

As of the current toolchain, the newer `torch.export` / `dynamo=True` ONNX exporter fails on EUPE during decomposition. Until that upstream exporter bug is fixed, the release export uses the legacy TorchScript-based path.

## Parity criteria

The script records strict `np.allclose()` status for visibility, but publication is gated by metrics that match the working export path:

- CLS cosine `>= 0.995`
- Patch cosine `>= 0.99`
- CLS mean abs diff `<= 0.03`
- Patch mean abs diff `<= 0.05`
- CLS max abs diff `<= 0.5`
- Patch max abs diff `<= 5.0`
- Input-independence cosine `< 0.85`

These thresholds cleanly reject the broken constant-bias export while accepting the corrected artifact.

## Publish checklist

Do not publish unless all are true:

- The script exits successfully.
- `validation_passed == true` in the JSON report.
- Every validation image has `threshold_pass == true`.
- `input_independence_cosine < input_independence_threshold`.

## Notes

- CPU ONNX Runtime is used for parity checks to keep them deterministic.
- `onnxsim` is optional.
- The Rust validation fixtures in `tests/fixtures/validation/` are release-artifact regression evidence for the shipped ONNX bundle. PyTorch source alignment is established by the export report, not by pretending the checked-in ONNX reference is a PyTorch capture.
