# latent-inspector HF Space — Monday 10:00 publish checklist

Target publish time:
- Monday 2026-04-20 10:00 CEST
- Monday 2026-04-20 08:00 UTC

Recommended Space ID:
- `AbdelStark/latent-inspector-showcase`

Expected live URL:
- https://huggingface.co/spaces/AbdelStark/latent-inspector-showcase

GitHub links:
- Branch: https://github.com/AbdelStark/latent-inspector/tree/hf-space-latent-inspector
- Compare / PR: https://github.com/AbdelStark/latent-inspector/compare/main...hf-space-latent-inspector?expand=1
- Space package folder: https://github.com/AbdelStark/latent-inspector/tree/hf-space-latent-inspector/spaces/latent-inspector-showcase

## 5-minute ship plan

### 1. Land the branch if it is still open
From repo root:
```bash
cd /home/developer/latent-inspector
```
If needed, open the PR:
```bash
gh pr create \
  --base main \
  --head hf-space-latent-inspector \
  --title "Add latent-inspector Hugging Face Space package" \
  --body "## Summary
- add first static Hugging Face Space package for latent-inspector
- add publish helper and Monday publish checklist
- package PCA stills and TUI screenshots into a browser-native project front door

## Why
This gives latent-inspector a much stronger public demo surface than repo-only presentation.

## Test plan
- local static preview via python3 -m http.server
- publish helper checked for hf CLI presence and repo-create + upload flow
"
```
Then merge it once you are happy.

### 2. Pull the latest main locally
```bash
cd /home/developer/latent-inspector
git checkout main
git pull --rebase origin main
```

### 3. Authenticate with Hugging Face
Quick check:
```bash
hf auth whoami
```
If needed:
```bash
hf auth login
```

### 4. Publish the Space
```bash
cd /home/developer/latent-inspector/spaces/latent-inspector-showcase
chmod +x publish.sh
./publish.sh AbdelStark/latent-inspector-showcase
```

### 5. Smoke-test the live page
Open:
- https://huggingface.co/spaces/AbdelStark/latent-inspector-showcase

Quick checks:
- hero loads cleanly
- PCA images render
- TUI screenshots render
- repo / slides / report links work
- mobile layout is readable enough

### 6. Optional same-morning upgrade
If you want a stronger first hit right away, add the HyperFrames promo video after the first publish and re-run:
```bash
./publish.sh AbdelStark/latent-inspector-showcase
```

## Fallback naming if the Space ID is unavailable
- `AbdelStark/latent-inspector-demo`
- `AbdelStark/latent-inspector-showcase-v1`

If you use a fallback, just replace the Space ID in the publish command above.
