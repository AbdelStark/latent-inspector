# latent-inspector — Slide Deck

This folder hosts the public slide deck and published HTML comparison reports for latent-inspector, deployed to GitHub Pages at:

**→ https://abdelstark.github.io/latent-inspector/**

## What's in here

- `index.html` — reveal.js slide deck (self-contained, no build step required)
- `slides.html` — alias of index.html, kept for direct-link stability
- `reports/` — published HTML report bundles and a reports index
- `*_pca.png` — PCA-projection stills from the four reference models (DINOv2, I-JEPA, V-JEPA 2, EUPE), generated with `latent-inspector compare`
- `elephant_sample_image.jpg` — the canonical sample image used throughout the deck

## Viewing locally

The deck is plain HTML with reveal.js loaded from a CDN — you can just open it:

```bash
open demo/index.html            # macOS
xdg-open demo/index.html        # Linux
start demo/index.html           # Windows
```

Or serve it with any static file server:

```bash
python3 -m http.server --directory demo 8000
# then visit http://localhost:8000/
```

## Regenerating the PCA stills

```bash
cargo build --release
./target/release/latent-inspector compare demo/elephant_sample_image.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16 \
  --output demo/ \
  --format png
```

## Deployment

This folder is deployed to GitHub Pages automatically by `.github/workflows/pages.yml` on any push to `main` that touches `demo/**`.

Published entry points:

- `/` — slide deck
- `/slides.html` — stable alias of the slide deck
- `/reports/` — reports index
- `/reports/20260408-123006/report.html` — sample four-model compare report
