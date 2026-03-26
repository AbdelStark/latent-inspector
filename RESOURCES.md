# latent-inspector: Resources

## Self-Supervised Vision Models

| Model | Paper | Key Innovation |
|-------|-------|---------------|
| DINO | Caron et al. 2021, arXiv:2104.14294 | Self-distillation with momentum teacher. Emergent segmentation in attention maps. |
| DINOv2 | Oquab et al. 2023, arXiv:2304.07193 | Scaled DINO + iBOT + SwAV centering on curated LVD-142M. Universal features without fine-tuning. |
| DINOv3 | Meta FAIR 2025, ai.meta.com/blog/dinov3 | 7B ViT with Gram Anchoring for stable dense features. Distilled to smaller models. |
| MAE | He et al. 2022, arXiv:2111.06377 | Masked autoencoder. Reconstruct masked patches. Features require fine-tuning. |
| I-JEPA | Assran et al. 2023, arXiv:2301.08243 | Predict masked regions in latent space (not pixel). No data augmentation needed. |
| V-JEPA 2 | Bardes et al. 2025, arXiv:2506.09985 | Video JEPA. 1B params on 1M hours video. Action-conditioned world model. |
| CLIP | Radford et al. 2021 | Contrastive image-text pretraining. Strong zero-shot but semantic, not spatial. |
| SigLIP | Zhai et al. 2023, arXiv:2303.15343 | Sigmoid loss replaces softmax in CLIP. Better scaling. |

## Representation Analysis Literature

| Paper | Key Finding |
|-------|------------|
| "Analyzing Local Representations of SSL ViTs" (2024) | DINO patches are more universal than MAE. MAE has high-variance "noise" dimensions. DINOv2 surprisingly less robust than DINO on some tasks. |
| "Representation Learning via CKA" (Kornblith 2019) | Centered Kernel Alignment as canonical measure of representation similarity. |
| "Do Vision Transformers See Like CNNs?" (Raghu 2021) | ViTs develop global attention early, unlike CNNs. Different representation geometry. |
| "What Do Self-Supervised Vision Transformers Learn?" (Park 2023) | Contrastive methods learn semantic features. Reconstruction methods learn textural features. |

## Rust Ecosystem

| Crate | Version | Purpose |
|-------|---------|---------|
| ort | 2.x | ONNX Runtime Rust bindings |
| ndarray | 0.16 | N-dimensional arrays for analysis |
| image | 0.25 | Image loading and processing |
| rayon | 1.10 | Parallel model inference |
| ratatui | 0.29 | Terminal UI rendering |
| clap | 4.x | CLI argument parsing |
| indicatif | 0.17 | Progress bars for download/batch |
| reqwest | 0.12 | HTTP for model downloads |

## Existing Tools (Competitors/Inspiration)

| Tool | Language | Limitation |
|------|----------|-----------|
| BertViz | Python | Text-only. No vision model support. Slow. |
| Attention Rollout | Python | Single model only. No cross-model comparison. |
| UMAP/t-SNE visualizers | Python | Global projection only. No patch-level analysis. |
| Embedding Projector (TensorBoard) | JS/Python | Interactive but heavyweight. No CLI. No comparison. |
| timm (PyTorch Image Models) | Python | Model zoo, not an analysis tool. |

## Gap latent-inspector fills

No existing tool provides: (a) multi-model comparison, (b) patch-level analysis, (c) quantitative metrics (rank, CKA, Gini), (d) fast Rust performance, (e) CLI-first UX, all in one package. Most researchers cobble together Jupyter notebooks with matplotlib. latent-inspector replaces that with a single command.
