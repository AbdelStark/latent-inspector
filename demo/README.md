# latent-inspector — Pages Site

This folder hosts the public landing page, presentation, and published HTML comparison reports for latent-inspector, deployed to GitHub Pages at:

**→ https://abdelstark.github.io/latent-inspector/**

## What's in here

- `index.html` — landing page with the two public entry points
- `slides.html` — reveal.js presentation, kept as the stable direct presentation URL
- `reports/` — published HTML report bundles and a reports index
- `*_pca.png` — PCA-projection stills from the four reference models (DINOv2, I-JEPA, V-JEPA 2, EUPE), generated with `latent-inspector compare`
- `elephant_sample_image.jpg` — the canonical sample image used throughout the deck

## Viewing locally

The landing page and presentation are plain HTML. You can just open them:

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
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-img16-256,eupe-vit-b16 \
  --output demo/ \
  --format png
```

## Deployment

This folder is deployed to GitHub Pages automatically by `.github/workflows/pages.yml` on any push to `main` that touches `demo/**`.

Published entry points:

- `/` — landing page with links to the presentation and sample report
- `/slides.html` — presentation: "How AI Models See the World"
- `/reports/` — reports index
- `/reports/20260408-123006/report.html` — sample four-model compare report
