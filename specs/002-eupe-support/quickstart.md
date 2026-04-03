# Quickstart: Using EUPE

## Compare EUPE with DINOv2 (first CLS cosine pair)

```bash
latent-inspector compare photo.jpg --models dinov2-vit-l14,eupe-vit-b16
```

This is the first model pair where CLS cosine similarity is fully computable — both DINOv2 and EUPE expose CLS tokens.

## Inspect EUPE representation

```bash
latent-inspector inspect photo.jpg --model eupe-vit-b16
```

## Compare all four ready models

```bash
latent-inspector compare photo.jpg \
  --models dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16
```

## Generate HTML report

```bash
latent-inspector compare photo.jpg \
  --models dinov2-vit-l14,eupe-vit-b16 \
  --format html --output eupe-vs-dino/
```

## Validate model integration

```bash
latent-inspector validate --model eupe-vit-b16
```
