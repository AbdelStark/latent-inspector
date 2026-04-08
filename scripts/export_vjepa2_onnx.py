#!/usr/bin/env python3
"""Export a corrected image-adapted V-JEPA 2 ONNX model with parity checks.

The official V-JEPA 2 checkpoints are video encoders. For single-image latent
inspection, this wrapper follows Meta's image-eval adaptation more closely than
the old 2-frame surrogate:

- accept a standard image tensor `[B, 3, H, W]`
- repeat it to 16 frames
- run the encoder-only vision trunk
- reshape the temporal-spatial tokens to `[B, 8, 256, 1024]`
- average over time to produce `[B, 256, 1024]`

Usage:
  python scripts/export_vjepa2_onnx.py \
    --output artifacts/vjepa2-vitl-img16-256/model.onnx \
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
from huggingface_hub import HfApi
from PIL import Image
from torchvision.transforms import v2 as T
from transformers import AutoModel


DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_OUTPUT_NAME = "last_hidden_state"


@dataclass
class PairMetrics:
    cosine: float
    max_abs_diff: float
    mean_abs_diff: float


@dataclass
class ValidationThresholds:
    min_patch_cosine: float
    max_patch_mean_abs_diff: float
    max_patch_max_abs_diff: float


@dataclass
class ValidationRecord:
    image: str
    patch: PairMetrics
    allclose_atol: float
    allclose_rtol: float
    allclose_pass: bool
    threshold_pass: bool


@dataclass
class ExportReport:
    model_id: str
    onnx_path: str
    opset: int
    image_size: int
    repeat_frames: int
    temporal_groups: int
    spatial_tokens: int
    thresholds: ValidationThresholds
    validation_records: List[ValidationRecord]
    validation_passed: bool
    input_independence_cosine: float
    input_independence_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--repeat-frames", type=int, default=16)
    parser.add_argument(
        "--validation-images", type=Path, default=Path("docs/assets/img/samples")
    )
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--min-patch-cosine", type=float, default=0.999)
    parser.add_argument("--max-patch-mean-abs-diff", type=float, default=0.01)
    parser.add_argument("--max-patch-max-abs-diff", type=float, default=0.5)
    parser.add_argument("--input-independence-threshold", type=float, default=0.85)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-simplify", action="store_true")
    parser.add_argument("--skip-external-data", action="store_true")
    parser.add_argument(
        "--publish-to",
        default=None,
        help="Optional Hugging Face repo id to upload the exported artifact bundle.",
    )
    return parser.parse_args()


class VJEPA2ImageWrapper(nn.Module):
    """Expose a stable image representation from the video-only backbone."""

    def __init__(self, base_model: nn.Module, repeat_frames: int):
        super().__init__()
        self.base = base_model
        self.repeat_frames = repeat_frames

        config = getattr(base_model, "config", None)
        if config is None:
            raise RuntimeError("V-JEPA 2 model is missing config metadata.")

        tubelet = int(getattr(config, "tubelet_size"))
        patch = int(getattr(config, "patch_size"))
        image_size = getattr(config, "image_size")
        if isinstance(image_size, (list, tuple)):
            if len(image_size) != 2 or image_size[0] != image_size[1]:
                raise RuntimeError(f"Unsupported non-square image size: {image_size}")
            image_size = int(image_size[0])
        else:
            image_size = int(image_size)

        if repeat_frames % tubelet != 0:
            raise RuntimeError(
                f"repeat_frames={repeat_frames} must be divisible by tubelet_size={tubelet}."
            )

        patches_per_side = image_size // patch
        self.temporal_groups = repeat_frames // tubelet
        self.spatial_tokens = patches_per_side * patches_per_side

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        video = pixel_values.unsqueeze(1).repeat(1, self.repeat_frames, 1, 1, 1)
        features = self.base.get_vision_features(video)

        if features.ndim != 3:
            raise RuntimeError(f"Expected [B, N, D] features, got {tuple(features.shape)}.")

        batch, sequence_len, hidden = features.shape
        expected = self.temporal_groups * self.spatial_tokens
        if sequence_len != expected:
            raise RuntimeError(
                f"Unexpected sequence length {sequence_len}; expected {expected} "
                f"from temporal_groups={self.temporal_groups} and spatial_tokens={self.spatial_tokens}."
            )

        features = features.reshape(batch, self.temporal_groups, self.spatial_tokens, hidden)
        return features.mean(dim=1)


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


def run_torch(model: VJEPA2ImageWrapper, x: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return model(x.to(device)).float().cpu().numpy()


def run_ort(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x.astype(np.float32)})[0]


def compare_outputs(
    torch_y: np.ndarray,
    ort_y: np.ndarray,
    atol: float,
    rtol: float,
    thresholds: ValidationThresholds,
) -> tuple[PairMetrics, bool, bool]:
    patch_metrics = PairMetrics(
        cosine=cosine_similarity(torch_y, ort_y),
        max_abs_diff=float(np.max(np.abs(torch_y - ort_y))),
        mean_abs_diff=float(np.mean(np.abs(torch_y - ort_y))),
    )
    allclose_pass = bool(np.allclose(torch_y, ort_y, atol=atol, rtol=rtol))
    threshold_pass = (
        patch_metrics.cosine >= thresholds.min_patch_cosine
        and patch_metrics.mean_abs_diff <= thresholds.max_patch_mean_abs_diff
        and patch_metrics.max_abs_diff <= thresholds.max_patch_max_abs_diff
    )
    return patch_metrics, allclose_pass, threshold_pass


def maybe_simplify(model_path: Path) -> None:
    try:
        import onnxsim  # type: ignore

        simplified, ok = onnxsim.simplify(str(model_path))
        if not ok:
            raise RuntimeError("onnxsim reported unsuccessful simplification")
        onnx.save(simplified, str(model_path))
    except ImportError:
        print("[warn] onnxsim not installed; skipping simplification.")


def externalize_weights(model_path: Path, weights_name: str = "model.onnx_data") -> Path:
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    external_path = model_path.with_name(weights_name)
    stale_path = model_path.with_name(f"{model_path.name}.data")
    if stale_path != external_path and stale_path.exists():
        stale_path.unlink()
    return external_path


def build_input_independence_probe(size: int) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros((1, 3, size, size), dtype=np.float32)
    rng = np.random.default_rng(13)
    rnd = rng.normal(loc=0.0, scale=1.0, size=(1, 3, size, size)).astype(np.float32)
    return zeros, rnd


def maybe_publish_bundle(repo_id: str, output_path: Path, report_path: Path | None) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    uploads = [(output_path, "model.onnx")]
    for external_data in (
        output_path.with_name("model.onnx_data"),
        output_path.with_name(f"{output_path.name}.data"),
    ):
        if external_data.exists():
            uploads.append((external_data, "model.onnx_data"))
            break
    if report_path is not None and report_path.exists():
        uploads.append((report_path, report_path.name))

    for src, dst in uploads:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=repo_id,
            repo_type="model",
        )


def main() -> None:
    args = parse_args()
    set_repro()
    device = resolve_device(args.device)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve() if args.report else output_path.with_suffix(".report.json")

    print(f"[info] Loading {args.model_id} on {device} ...")
    base_model = AutoModel.from_pretrained(args.model_id)
    base_model.eval()
    base_model.to(device)

    wrapper = VJEPA2ImageWrapper(base_model, repeat_frames=args.repeat_frames).eval()
    wrapper.to(device)

    preprocess = build_preprocess(args.image_size)
    sample = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32, device=device)

    print(f"[info] Exporting ONNX to {output_path} ...")
    torch.onnx.export(
        wrapper,
        sample,
        str(output_path),
        input_names=["pixel_values"],
        output_names=[DEFAULT_OUTPUT_NAME],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    if not args.skip_simplify:
        maybe_simplify(output_path)

    if not args.skip_external_data:
        external_path = externalize_weights(output_path)
        print(f"[info] Saved external data weights to {external_path}")

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    thresholds = ValidationThresholds(
        min_patch_cosine=args.min_patch_cosine,
        max_patch_mean_abs_diff=args.max_patch_mean_abs_diff,
        max_patch_max_abs_diff=args.max_patch_max_abs_diff,
    )

    validation_records: List[ValidationRecord] = []
    validation_passed = True
    for image_path in collect_images(args.validation_images, args.max_images):
        x = load_tensor(image_path, preprocess)
        torch_y = run_torch(wrapper, x, device)
        ort_y = run_ort(session, x.numpy())
        patch_metrics, allclose_pass, threshold_pass = compare_outputs(
            torch_y,
            ort_y,
            atol=args.atol,
            rtol=args.rtol,
            thresholds=thresholds,
        )
        validation_records.append(
            ValidationRecord(
                image=image_path.name,
                patch=patch_metrics,
                allclose_atol=args.atol,
                allclose_rtol=args.rtol,
                allclose_pass=allclose_pass,
                threshold_pass=threshold_pass,
            )
        )
        validation_passed &= threshold_pass
        print(
            "[check]",
            image_path.name,
            f"cos={patch_metrics.cosine:.6f}",
            f"mean_abs={patch_metrics.mean_abs_diff:.6f}",
            f"max_abs={patch_metrics.max_abs_diff:.6f}",
            f"allclose={allclose_pass}",
            f"threshold={threshold_pass}",
        )

    zeros, rnd = build_input_independence_probe(args.image_size)
    zeros_y = run_ort(session, zeros)
    rnd_y = run_ort(session, rnd)
    input_independence_cosine = cosine_similarity(zeros_y, rnd_y)
    print(
        "[check] input-independence",
        f"cos={input_independence_cosine:.6f}",
        f"threshold<{args.input_independence_threshold:.2f}",
    )
    validation_passed &= input_independence_cosine < args.input_independence_threshold

    report = ExportReport(
        model_id=args.model_id,
        onnx_path=str(output_path),
        opset=args.opset,
        image_size=args.image_size,
        repeat_frames=args.repeat_frames,
        temporal_groups=wrapper.temporal_groups,
        spatial_tokens=wrapper.spatial_tokens,
        thresholds=thresholds,
        validation_records=validation_records,
        validation_passed=validation_passed,
        input_independence_cosine=input_independence_cosine,
        input_independence_threshold=args.input_independence_threshold,
    )
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(f"[info] Wrote export report to {report_path}")

    if not validation_passed:
        raise SystemExit("Export validation failed.")

    if args.publish_to:
        maybe_publish_bundle(args.publish_to, output_path, report_path)
        print(f"[info] Published artifact bundle to https://huggingface.co/{args.publish_to}")


if __name__ == "__main__":
    main()
