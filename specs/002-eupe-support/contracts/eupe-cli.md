# CLI Contract: EUPE Model

## Model name

`eupe-vit-b16`

## Commands that accept EUPE

All existing commands — no new flags or options needed:

- `compare --models ...,eupe-vit-b16,...`
- `inspect --model eupe-vit-b16`
- `neighbors --model eupe-vit-b16 --dataset <dir>`
- `similarity --model-a eupe-vit-b16 --model-b <other> --dataset <dir>`
- `profile --model eupe-vit-b16 --dataset <dir>`
- `drift --model eupe-vit-b16 --checkpoints <dir> --dataset <dir>`
- `validate --model eupe-vit-b16`
- `models` (lists EUPE in catalog)
- `tui -m eupe-vit-b16,...`

## Expected outputs

EUPE produces 197 tokens: 1 CLS + 196 patches, embed_dim=768.

- CLS token: present (first model to pair with DINOv2 for CLS cosine)
- Attention weights: not exported (Gini = N/A)
- Embedding basis: CLS token preferred
