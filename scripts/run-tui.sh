#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run-tui.sh — Launch latent-inspector TUI with real ONNX inference
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${LATENT_INSPECTOR_CACHE_DIR:-$HOME/.cache/latent-inspector}"
DEFAULT_MODEL="dinov2-vit-l14"
DEFAULT_IMAGE="$PROJECT_DIR/docs/assets/img/samples/buffalo_sample_image.jpg"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [IMAGE_PATH]

Launch the latent-inspector interactive TUI with real ONNX model inference.

Arguments:
  IMAGE_PATH          Path to an image file (default: sample buffalo image)

Options:
  -m, --model MODEL   Model to use (default: $DEFAULT_MODEL)
  -d, --demo          Launch in demo mode (no ONNX, synthetic data)
  --download-only     Download model and exit
  -h, --help          Show this help

Examples:
  $(basename "$0")                           # TUI with sample image + DINOv2
  $(basename "$0") photo.jpg                 # TUI with your image
  $(basename "$0") --demo                    # Demo mode (no download needed)
  $(basename "$0") --download-only           # Pre-download model only

Environment:
  LATENT_INSPECTOR_CACHE_DIR   Override model cache location
                               (default: ~/.cache/latent-inspector)
EOF
}

MODEL="$DEFAULT_MODEL"
IMAGE=""
DEMO=false
DOWNLOAD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--demo) DEMO=true; shift ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) IMAGE="$1"; shift ;;
    esac
done

cd "$PROJECT_DIR"

# ── Build ────────────────────────────────────────────────────────

if $DEMO; then
    echo "Building (stub mode)..."
    cargo build --release 2>&1 | tail -1
    echo ""
    exec cargo run --release -- tui
fi

echo "Building with ONNX inference..."
cargo build --features onnx-inference --release 2>&1 | tail -1

# ── Download model if needed ─────────────────────────────────────

MODEL_PATH="$CACHE_DIR/$MODEL.onnx"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo ""
    echo "Model '$MODEL' not found in cache."
    echo "Downloading (~1.1 GB for DINOv2)..."
    echo ""
    cargo run --features onnx-inference --release -- models --download "$MODEL"
fi

if $DOWNLOAD_ONLY; then
    echo ""
    echo "Model downloaded to: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

# ── Launch TUI ───────────────────────────────────────────────────

if [[ -z "$IMAGE" ]]; then
    IMAGE="$DEFAULT_IMAGE"
    echo "No image specified — using sample: $(basename "$IMAGE")"
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "Error: Image not found: $IMAGE" >&2
    exit 1
fi

echo "Launching TUI with $MODEL on $(basename "$IMAGE")..."
echo ""
exec cargo run --features onnx-inference --release -- tui "$IMAGE" -m "$MODEL"
