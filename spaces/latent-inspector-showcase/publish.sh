#!/usr/bin/env bash
set -euo pipefail

SPACE_ID="${1:-AbdelStark/latent-inspector-showcase}"
CREATE_IF_MISSING="${CREATE_IF_MISSING:-1}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install it first."
  exit 1
fi

cd "$(dirname "$0")"

echo "Checking HF auth..."
hf auth whoami >/dev/null

if [ "$CREATE_IF_MISSING" = "1" ]; then
  echo "Ensuring Space exists: $SPACE_ID"
  hf repo create "$SPACE_ID" --type space >/dev/null 2>&1 || true
fi

echo "Uploading to $SPACE_ID"
hf upload "$SPACE_ID" . --repo-type space

echo "Live URL: https://huggingface.co/spaces/$SPACE_ID"
echo "Done."
