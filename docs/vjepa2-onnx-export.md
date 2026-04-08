# V-JEPA 2 ONNX Export Notes

This repo treats V-JEPA 2 as a still-image inspector target, even though the
official checkpoint is a video encoder.

## What changed

The retired `vjepa2-vitl-fpc2-256` adapter duplicated a still image to only
`2` frames. That export was numerically sound as ONNX, but it was the wrong
image abstraction: on natural images it diverged materially from the stable
repeated-frame image manifold used by Meta's own image-evaluation path.

The current canonical model is `vjepa2-vitl-img16-256`.

- input: `[B, 3, 256, 256]`
- wrapper: repeat the image to `16` frames
- backbone: official `facebook/vjepa2-vitl-fpc64-256` encoder
- reshape: `[B, 2048, 1024] -> [B, 8, 256, 1024]`
- reduce: average over the temporal axis to `[B, 256, 1024]`

The old CLI name `vjepa2-vitl-fpc2-256` remains as a backward-compatible alias,
but all docs and reports should use `vjepa2-vitl-img16-256`.

## Export command

```bash
source /tmp/latent-inspector-eupe-audit/bin/activate
python scripts/export_vjepa2_onnx.py \
  --output /tmp/vjepa2-vitl-img16-256/model.onnx \
  --validation-images docs/assets/img/samples \
  --publish-to abdelstark/vjepa2-vitl-img16-256-onnx
```

## Published artifact

- ONNX repo: <https://huggingface.co/abdelstark/vjepa2-vitl-img16-256-onnx>
- Source checkpoint: <https://huggingface.co/facebook/vjepa2-vitl-fpc64-256>
- Paper: <https://arxiv.org/abs/2506.09985>
- Reference implementation: <https://github.com/facebookresearch/vjepa2>

## Parity summary

The exporter writes a JSON parity report next to the ONNX bundle. For the
published artifact:

- worst patch cosine vs PyTorch across 5 sample images: `> 0.999999`
- worst patch mean abs diff: `1.37e-4`
- worst patch max abs diff: `0.00643`
- input-independence cosine `cos(zeros, random)`: `0.3168`

Those numbers confirm the ONNX is source-aligned for the intended 16-frame
image wrapper and is not suffering from the EUPE-style input-independent export
failure mode.
