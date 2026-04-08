#!/usr/bin/env python3
"""Export facebook/EUPE-ViT-B to ONNX with explicit CLS+patch concatenation and parity checks.

Usage:
  python scripts/export_eupe_onnx.py \
    --output artifacts/eupe-vit-b16/model.onnx \
    --validation-images docs/assets/img/samples
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2 as T
from transformers import AutoModel


DEFAULT_MODEL_ID = "facebook/EUPE-ViT-B"
DEFAULT_OUTPUT_NAME = "last_hidden_state"


@dataclass
class PairMetrics:
    cosine: float
    max_abs_diff: float
    mean_abs_diff: float


@dataclass
class ValidationRecord:
    image: str
    cls: PairMetrics
    patch: PairMetrics
    allclose_atol: float
    allclose_rtol: float
    allclose_pass: bool


@dataclass
class ExportReport:
    model_id: str
    onnx_path: str
    opset: int
    image_size: int
    validation_records: List[ValidationRecord]
    validation_passed: bool
    input_independence_cosine: float
    input_independence_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-images", type=Path, default=Path("docs/assets/img/samples"))
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--input-independence-threshold", type=float, default=0.85)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-simplify", action="store_true")
    return parser.parse_args()


class EUPEWrapper(nn.Module):
    """Normalizes EUPE outputs into [B, 1 + N, D] with CLS at index 0."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.base.forward_features(pixel_values)
        if "x_norm_clstoken" not in features or "x_norm_patchtokens" not in features:
            raise RuntimeError(
                "EUPE forward_features() did not expose expected keys "
                "('x_norm_clstoken', 'x_norm_patchtokens')."
            )
        cls = features["x_norm_clstoken"].unsqueeze(1)
        patches = features["x_norm_patchtokens"]
        return torch.cat([cls, patches], dim=1)


def set_repro(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_preprocess(size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((size, size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_images(path: Path, max_images: int) -> List[Path]:
    if path.is_file():
        return [path]

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed
    )
    if len(images) < max_images:
        raise RuntimeError(
            f"Need at least {max_images} validation images, found {len(images)} in {path}."
        )
    return images[:max_images]


def load_tensor(image_path: Path, preprocess: T.Compose) -> torch.Tensor:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    tensor = preprocess(rgb)
    return tensor.unsqueeze(0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def run_torch(model: EUPEWrapper, x: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        y = model(x.to(device)).float().cpu().numpy()
    return y


def run_ort(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x.astype(np.float32)})[0]


def compare_outputs(torch_y: np.ndarray, ort_y: np.ndarray, atol: float, rtol: float) -> tuple[PairMetrics, PairMetrics, bool]:
    cls_t, cls_o = torch_y[:, :1, :], ort_y[:, :1, :]
    patch_t, patch_o = torch_y[:, 1:, :], ort_y[:, 1:, :]

    cls_metrics = PairMetrics(
        cosine=cosine_similarity(cls_t, cls_o),
        max_abs_diff=float(np.max(np.abs(cls_t - cls_o))),
        mean_abs_diff=float(np.mean(np.abs(cls_t - cls_o))),
    )
    patch_metrics = PairMetrics(
        cosine=cosine_similarity(patch_t, patch_o),
        max_abs_diff=float(np.max(np.abs(patch_t - patch_o))),
        mean_abs_diff=float(np.mean(np.abs(patch_t - patch_o))),
    )

    allclose_pass = bool(np.allclose(torch_y, ort_y, atol=atol, rtol=rtol))
    return cls_metrics, patch_metrics, allclose_pass


def maybe_simplify(model_path: Path) -> None:
    try:
        import onnxsim  # type: ignore

        simplified, ok = onnxsim.simplify(str(model_path))
        if not ok:
            raise RuntimeError("onnxsim reported unsuccessful simplification")
        onnx.save(simplified, str(model_path))
    except ImportError:
        print("[warn] onnxsim not installed; skipping simplification.")


def build_input_independence_probe(size: int) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros((1, 3, size, size), dtype=np.float32)

    rng = np.random.default_rng(13)
    rnd = rng.normal(loc=0.0, scale=1.0, size=(1, 3, size, size)).astype(np.float32)
    return zeros, rnd


def main() -> None:
    args = parse_args()
    set_repro()

    device = resolve_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    base = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    base.eval()
    base.to(device)

    model = EUPEWrapper(base).eval()
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device, dtype=torch.float32)

    print(f"Exporting ONNX to {args.output} ...")
    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=[DEFAULT_OUTPUT_NAME],
        opset_version=args.opset,
        dynamo=True,
        external_data=True,
    )

    if not args.skip_simplify:
        maybe_simplify(args.output)

    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.output), providers=providers)

    preprocess = build_preprocess(args.image_size)
    image_paths = collect_images(args.validation_images, args.max_images)

    records: List[ValidationRecord] = []
    for image_path in image_paths:
        tensor = load_tensor(image_path, preprocess)
        torch_y = run_torch(model, tensor, device)
        ort_y = run_ort(session, tensor.numpy())

        cls_metrics, patch_metrics, allclose_pass = compare_outputs(
            torch_y, ort_y, atol=args.atol, rtol=args.rtol
        )

        records.append(
            ValidationRecord(
                image=str(image_path),
                cls=cls_metrics,
                patch=patch_metrics,
                allclose_atol=args.atol,
                allclose_rtol=args.rtol,
                allclose_pass=allclose_pass,
            )
        )

    zeros, rnd = build_input_independence_probe(args.image_size)
    z_out = run_ort(session, zeros)
    r_out = run_ort(session, rnd)
    independence_cos = cosine_similarity(z_out, r_out)

    validation_passed = all(r.allclose_pass for r in records)
    gate_passed = independence_cos < args.input_independence_threshold

    report = ExportReport(
        model_id=args.model_id,
        onnx_path=str(args.output),
        opset=args.opset,
        image_size=args.image_size,
        validation_records=records,
        validation_passed=validation_passed and gate_passed,
        input_independence_cosine=independence_cos,
        input_independence_threshold=args.input_independence_threshold,
    )

    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print("\nValidation summary")
    print("------------------")
    for record in records:
        print(
            f"{record.image}: allclose={record.allclose_pass} "
            f"cls_max={record.cls.max_abs_diff:.6f} patch_max={record.patch.max_abs_diff:.6f} "
            f"cls_cos={record.cls.cosine:.6f} patch_cos={record.patch.cosine:.6f}"
        )
    print(
        f"input-independence cosine(zero,random)={independence_cos:.6f} "
        f"(threshold < {args.input_independence_threshold:.2f})"
    )

    if not validation_passed:
        raise SystemExit("Export failed validation: ONNX output is not allclose to PyTorch on validation images.")
    if not gate_passed:
        raise SystemExit("Export failed input-independence gate: output appears input-insensitive.")

    print(f"\nExport successful: {args.output}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
