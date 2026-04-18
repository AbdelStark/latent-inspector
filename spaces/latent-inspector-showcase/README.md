---
title: latent-inspector
author: AbdelStark
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Inspect how self-supervised vision models carve the same image differently.
---

# latent-inspector — Hugging Face Space package

This folder is a first real Hugging Face Space package for `latent-inspector`.

## What it is
A static HTML Space package that acts as the browser-native front door for the project.

It is optimized for:
- fast first impression
- interview / recruiter / researcher legibility
- visual proof over code spelunking
- easy linking from GitHub, X, and outreach

## Why static first
The project already has strong visual proof:
- PCA comparisons
- TUI screenshots
- slide deck
- detailed report

A static Space is the fastest truthful publication layer.

## Included
- `index.html`
- `styles.css`
- `assets/` with screenshots and PCA stills
- `publish.sh` helper for HF upload once authenticated
- `PUBLISH-MONDAY-CHECKLIST.md` with the exact ship steps and links

## Local preview
```bash
cd spaces/latent-inspector-showcase
python3 -m http.server 8123
# then open http://localhost:8123
```

## Publish to Hugging Face Spaces
Recommended Space ID:
- `AbdelStark/latent-inspector-showcase`

1. authenticate:
```bash
hf auth login
```
2. publish in one shot:
```bash
cd spaces/latent-inspector-showcase
./publish.sh AbdelStark/latent-inspector-showcase
```

This helper will:
- verify HF auth
- create the Space if it does not exist yet
- upload the package
- print the live URL

If the preferred Space name is already taken, use a variant like:
- `AbdelStark/latent-inspector-demo`
- `AbdelStark/latent-inspector-showcase-v1`

## Recommended next step after this package
After the static Space is live:
- add the HyperFrames promo video
- add a tighter report-view CTA
- optionally build a Gradio compare demo as a second Space or as a v2
