#!/usr/bin/env bash
set -euo pipefail

SPACE_ID="${1:-AbdelStark/latent-inspector-showcase}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install it first."
  exit 1
fi

cd "$(dirname "$0")"

echo "Checking HF auth..."
hf auth whoami >/dev/null

echo "Uploading to $SPACE_ID"
hf upload "$SPACE_ID" . --repo-type space

echo "Done."
