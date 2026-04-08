use crate::errors::ModelError;
use serde::{Deserialize, Serialize};

/// Self-supervised learning method used to train the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SSLMethod {
    DINO,
    MAE,
    IJEPA,
    VJEPA2,
    EUPE,
    CLIP,
    SigLIP,
}

impl std::fmt::Display for SSLMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSLMethod::DINO => write!(f, "DINO"),
            SSLMethod::MAE => write!(f, "MAE"),
            SSLMethod::IJEPA => write!(f, "I-JEPA"),
            SSLMethod::VJEPA2 => write!(f, "V-JEPA 2"),
            SSLMethod::EUPE => write!(f, "EUPE"),
            SSLMethod::CLIP => write!(f, "CLIP"),
            SSLMethod::SigLIP => write!(f, "SigLIP"),
        }
    }
}

/// Implementation status for a registered model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    Ready,
    Planned,
}

impl std::fmt::Display for AvailabilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvailabilityStatus::Ready => write!(f, "ready"),
            AvailabilityStatus::Planned => write!(f, "planned"),
        }
    }
}

/// Phase-specific availability details for a registry entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Availability {
    pub status: AvailabilityStatus,
    pub phase: String,
    pub note: String,
}

impl Availability {
    fn ready(note: impl Into<String>) -> Self {
        Self {
            status: AvailabilityStatus::Ready,
            phase: "Phase 1".to_string(),
            note: note.into(),
        }
    }

    fn planned(phase: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            status: AvailabilityStatus::Planned,
            phase: phase.into(),
            note: note.into(),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.status, AvailabilityStatus::Ready)
    }
}

/// Download verification policy for an ONNX artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Checksum {
    Sha256(String),
    Pending { reason: String },
}

impl Checksum {
    pub fn label(&self) -> &'static str {
        match self {
            Checksum::Sha256(_) => "sha256",
            Checksum::Pending { .. } => "pending",
        }
    }

    pub fn note(&self) -> Option<&str> {
        match self {
            Checksum::Sha256(_) => None,
            Checksum::Pending { reason } => Some(reason.as_str()),
        }
    }
}

/// Metadata describing a registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Short identifier used on the CLI (e.g. "dinov2-vit-l14").
    pub name: String,
    /// Human-readable architecture string.
    pub architecture: String,
    /// Patch size in pixels.
    pub patch_size: u32,
    /// Embedding dimension.
    pub embed_dim: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// SSL training method.
    pub method: SSLMethod,
    /// Expected input image size (square).
    pub input_size: u32,
    /// Approximate parameter count (millions).
    pub params_m: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorRole {
    PatchSequence,
    PatchAndClsSequence,
}

impl std::fmt::Display for TensorRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorRole::PatchSequence => write!(f, "patch sequence"),
            TensorRole::PatchAndClsSequence => write!(f, "patch+cls sequence"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessContract {
    pub input_size: u32,
    pub resize_filter: String,
    pub color_space: String,
    pub layout: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub role: TensorRole,
    pub cls_expected: bool,
    pub batch_size: usize,
    pub patch_count: usize,
    pub embedding_dim: usize,
}

impl TensorContract {
    pub fn expected_sequence_len(&self) -> usize {
        self.patch_count + usize::from(self.cls_expected)
    }

    pub fn expected_shape(&self) -> Vec<usize> {
        vec![
            self.batch_size,
            self.expected_sequence_len(),
            self.embedding_dim,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityTolerances {
    pub patch_count_abs: usize,
    pub embedding_dim_abs: usize,
    pub patch_mean_abs: f32,
    pub patch_std_abs: f32,
    pub cls_l2_abs: f32,
    #[serde(default = "default_parity_signal_tolerance")]
    pub patch_rms_abs: f32,
    #[serde(default = "default_parity_signal_tolerance")]
    pub patch_signature_abs: f32,
    #[serde(default = "default_parity_signal_tolerance")]
    pub cls_signature_abs: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelValidationProfile {
    pub source: String,
    pub evidence_timestamp: String,
    pub fixture_set: String,
    pub preprocess: PreprocessContract,
    pub tensor: TensorContract,
    pub tolerances: ParityTolerances,
}

/// One file required to run a model from cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    /// Relative path within the cache directory.
    pub relative_path: String,
    /// HuggingFace Hub download URL.
    pub download_url: String,
    /// Verification policy for the downloaded file.
    pub checksum: Checksum,
}

/// Full registry entry including availability, download metadata, and validation contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub info: ModelInfo,
    pub availability: Availability,
    /// One or more files required to run the model from cache.
    pub artifacts: Vec<ModelArtifact>,
    /// Image normalization mean (RGB).
    pub norm_mean: [f32; 3],
    /// Image normalization std (RGB).
    pub norm_std: [f32; 3],
    /// Name of the ONNX input tensor.
    pub input_name: String,
    /// Name of the ONNX output tensor carrying patch tokens (and CLS at index 0).
    pub output_name: String,
    /// For video-backbone models: number of duplicated frames to construct
    /// the input tensor `[1, frames, 3, H, W]`. When `None`, the model
    /// accepts standard image input `[1, 3, H, W]`.
    pub video_frames: Option<u32>,
    /// Approved validation contract and parity configuration.
    pub validation: ModelValidationProfile,
}

impl RegistryEntry {
    pub fn is_ready(&self) -> bool {
        self.availability.is_ready()
    }

    pub fn availability_summary(&self) -> String {
        format!("{}: {}", self.availability.phase, self.availability.note)
    }

    pub fn ensure_ready(&self) -> Result<(), ModelError> {
        if self.is_ready() {
            Ok(())
        } else {
            Err(ModelError::Unavailable {
                name: self.info.name.clone(),
                reason: self.availability_summary(),
            })
        }
    }

    pub fn primary_artifact(&self) -> Result<&ModelArtifact, ModelError> {
        self.artifacts
            .first()
            .ok_or_else(|| ModelError::MissingDownloadMetadata(self.info.name.clone()))
    }

    pub fn verification_label(&self) -> &'static str {
        self.artifacts
            .first()
            .map(|artifact| artifact.checksum.label())
            .unwrap_or("pending")
    }

    pub fn verification_note(&self) -> Option<&str> {
        self.artifacts
            .first()
            .and_then(|artifact| artifact.checksum.note())
    }
}

fn default_parity_signal_tolerance() -> f32 {
    1e-3
}

fn default_validation_profile(
    source: &str,
    preprocess: PreprocessContract,
    tensor: TensorContract,
) -> ModelValidationProfile {
    ModelValidationProfile {
        source: source.to_string(),
        evidence_timestamp: "2026-03-27T12:00:00Z".to_string(),
        fixture_set: "standard".to_string(),
        preprocess,
        tensor,
        tolerances: ParityTolerances {
            patch_count_abs: 0,
            embedding_dim_abs: 0,
            patch_mean_abs: 1e-3,
            patch_std_abs: 1e-3,
            cls_l2_abs: 1e-3,
            patch_rms_abs: default_parity_signal_tolerance(),
            patch_signature_abs: default_parity_signal_tolerance(),
            cls_signature_abs: default_parity_signal_tolerance(),
        },
    }
}

/// Returns the full model registry.
///
/// # Model provenance
///
/// | CLI name              | Original checkpoint                    | ONNX artifact source                       |
/// |-----------------------|----------------------------------------|--------------------------------------------|
/// | `dinov2-vit-l14`      | `facebook/dinov2-large`                | `onnx-community/dinov2-large`              |
/// | `ijepa-vit-h14`       | `facebook/ijepa_vith14_1k`             | `onnx-community/ijepa_vith14_1k`           |
/// | `vjepa2-vitl-fpc2-256`| `facebook/vjepa2-vitl-fpc64-256`       | `abdelstark/vjepa2-vitl-fpc2-256-onnx` (*) |
///
/// (*) Custom encoder-only export: the V-JEPA 2 predictor is stripped and
///     the input is fixed to 2 frames (duplicated from a single image) so
///     that a video-backbone model produces comparable patch embeddings.
pub fn registry() -> Vec<RegistryEntry> {
    let phase_three_note =
        "Reserved for the multi-model milestone once ONNX output mapping and validation are in place.";

    vec![
        // ── DINOv2 ViT-L/14 ────────────────────────────────────────────
        // Paper: "DINOv2: Learning Robust Visual Features without Supervision"
        //        Oquab et al. 2024  — https://arxiv.org/abs/2304.07193
        // Source: facebook/dinov2-large (HuggingFace)
        // ONNX:   onnx-community/dinov2-large (community export)
        RegistryEntry {
            info: ModelInfo {
                name: "dinov2-vit-l14".to_string(),
                architecture: "ViT-L/14".to_string(),
                patch_size: 14,
                embed_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::DINO,
                input_size: 224,
                params_m: 304,
            },
            availability: Availability::ready(
                "Reference Phase 1 model with preprocessing, caching, and ONNX session loading wired end-to-end.",
            ),
            artifacts: vec![ModelArtifact {
                relative_path: "dinov2-vit-l14.onnx".to_string(),
                download_url:
                    "https://huggingface.co/onnx-community/dinov2-large/resolve/main/onnx/model.onnx".to_string(),
                checksum: Checksum::Sha256(
                    "305351060a1939d944e2dbe97dd64e4937ce5a220dce254e4cd74c7e4777d6ac"
                        .to_string(),
                ),
            }],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "facebookresearch/dinov2",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1024,
                },
            ),
        },
        RegistryEntry {
            info: ModelInfo {
                name: "dinov3-vit-l14".to_string(),
                architecture: "ViT-L/14".to_string(),
                patch_size: 14,
                embed_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::DINO,
                input_size: 224,
                params_m: 304,
            },
            availability: Availability::planned(
                "Phase 3",
                "Reserved for the multi-model milestone; artifact metadata still needs to be pinned.",
            ),
            artifacts: Vec::new(),
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "meta/dinov3",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1024,
                },
            ),
        },
        RegistryEntry {
            info: ModelInfo {
                name: "mae-vit-l16".to_string(),
                architecture: "ViT-L/16".to_string(),
                patch_size: 16,
                embed_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::MAE,
                input_size: 224,
                params_m: 304,
            },
            availability: Availability::planned("Phase 3", phase_three_note),
            artifacts: vec![ModelArtifact {
                relative_path: "mae-vit-l16.onnx".to_string(),
                download_url:
                    "https://huggingface.co/facebook/vit-mae-large/resolve/main/model.onnx".to_string(),
                checksum: Checksum::Pending {
                    reason: "Download metadata will be validated when MAE support lands in the comparison milestone."
                        .to_string(),
                },
            }],
            norm_mean: [0.5, 0.5, 0.5],
            norm_std: [0.5, 0.5, 0.5],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "facebookresearch/mae",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchSequence,
                    cls_expected: false,
                    batch_size: 1,
                    patch_count: 196,
                    embedding_dim: 1024,
                },
            ),
        },
        RegistryEntry {
            info: ModelInfo {
                name: "clip-vit-l14".to_string(),
                architecture: "ViT-L/14".to_string(),
                patch_size: 14,
                embed_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::CLIP,
                input_size: 224,
                params_m: 304,
            },
            availability: Availability::planned("Phase 3", phase_three_note),
            artifacts: vec![ModelArtifact {
                relative_path: "clip-vit-l14.onnx".to_string(),
                download_url:
                    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/onnx/visual.onnx".to_string(),
                checksum: Checksum::Pending {
                    reason: "Download metadata will be validated when CLIP support lands in the comparison milestone."
                        .to_string(),
                },
            }],
            norm_mean: [0.481_454_67, 0.457_827_5, 0.408_210_73],
            norm_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "openai/clip-vit-large-patch14",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.481_454_67, 0.457_827_5, 0.408_210_73],
                    std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1024,
                },
            ),
        },
        // ── I-JEPA ViT-H/14 ────────────────────────────────────────────
        // Paper: "Self-Supervised Learning from Images with a Joint-Embedding
        //         Predictive Architecture"
        //        Assran et al. 2023  — https://arxiv.org/abs/2301.08243
        // Source: facebook/ijepa_vith14_1k (HuggingFace)
        // ONNX:   onnx-community/ijepa_vith14_1k (community export)
        RegistryEntry {
            info: ModelInfo {
                name: "ijepa-vit-h14".to_string(),
                architecture: "ViT-H/14".to_string(),
                patch_size: 14,
                embed_dim: 1280,
                num_layers: 32,
                num_heads: 16,
                method: SSLMethod::IJEPA,
                input_size: 224,
                params_m: 632,
            },
            availability: Availability::ready(
                "ONNX community export with verified SHA-256 hashes for both model graph and external data.",
            ),
            artifacts: vec![
                ModelArtifact {
                    relative_path: "ijepa-vit-h14/model.onnx".to_string(),
                    download_url:
                        "https://huggingface.co/onnx-community/ijepa_vith14_1k/resolve/main/onnx/model.onnx".to_string(),
                    checksum: Checksum::Sha256(
                        "10b70b2151a5db382f03d52cfa0c223b0c6ea3c79e0f2068e0bc8f4ee2d6bfb8"
                            .to_string(),
                    ),
                },
                ModelArtifact {
                    relative_path: "ijepa-vit-h14/model.onnx_data".to_string(),
                    download_url:
                        "https://huggingface.co/onnx-community/ijepa_vith14_1k/resolve/main/onnx/model.onnx_data".to_string(),
                    checksum: Checksum::Sha256(
                        "82ab3565d733de48f2e142a2289b92f15b990bd461fe18c4d79635d4c34ade6f"
                            .to_string(),
                    ),
                },
            ],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "facebookresearch/ijepa",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchSequence,
                    cls_expected: false,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1280,
                },
            ),
        },
        // ── V-JEPA 2 ViT-L/16 ───────────────────────────────────────────
        // Paper: "V-JEPA 2: Self-Supervised Video Models Enable Understanding
        //         of Complex Real-World Interactions"
        //        Bardes et al. 2024  — https://arxiv.org/abs/2506.09985
        // Source: facebook/vjepa2-vitl-fpc64-256 (HuggingFace)
        // ONNX:   abdelstark/vjepa2-vitl-fpc2-256-onnx (custom export)
        //
        // Export method: encoder-only (predictor stripped), TorchScript ONNX
        // exporter at opset 14, simplified with onnxsim, 2-frame input (single
        // image duplicated to satisfy tubelet_size=2). Produces 256 spatial
        // patches × 1024 embed dim — same shape as DINOv2.
        RegistryEntry {
            info: ModelInfo {
                name: "vjepa2-vitl-fpc2-256".to_string(),
                architecture: "ViT-L/16".to_string(),
                patch_size: 16,
                embed_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::VJEPA2,
                input_size: 256,
                params_m: 304,
            },
            availability: Availability::ready(
                "V-JEPA 2 ViT-L encoder exported to ONNX with 2-frame video input for single-image latent inspection.",
            ),
            artifacts: vec![
                ModelArtifact {
                    relative_path: "vjepa2-vitl-fpc2-256/model.onnx".to_string(),
                    download_url:
                        "https://huggingface.co/abdelstark/vjepa2-vitl-fpc2-256-onnx/resolve/main/model.onnx"
                            .to_string(),
                    checksum: Checksum::Sha256(
                        "942f72f0f1afe2bea855160c8f11080cd4d322b54a04e5e671fb96beb8ce6537"
                            .to_string(),
                    ),
                },
                ModelArtifact {
                    relative_path: "vjepa2-vitl-fpc2-256/model.onnx_data".to_string(),
                    download_url:
                        "https://huggingface.co/abdelstark/vjepa2-vitl-fpc2-256-onnx/resolve/main/model.onnx_data"
                            .to_string(),
                    checksum: Checksum::Sha256(
                        "3a82e4d3cb3eaf1c13209daa08bb8ad7a408a4cc8bff3bee48978a8bfd34e640"
                            .to_string(),
                    ),
                },
            ],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values_videos".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: Some(2),
            validation: default_validation_profile(
                "facebookresearch/vjepa2",
                PreprocessContract {
                    input_size: 256,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "ntchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchSequence,
                    cls_expected: false,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1024,
                },
            ),
        },
        // ── EUPE ViT-B/16 ──────────────────────────────────────────────
        // Paper: "Efficient Universal Perception Encoder"
        //        Zhu et al. 2026  — https://arxiv.org/abs/2603.22387
        // Source: facebook/EUPE-ViT-B (HuggingFace)
        // ONNX:   abdelstark/eupe-vit-b16-onnx (custom export)
        //
        // Export method: official facebookresearch/eupe torch.hub loader +
        // forward_features() wrapper concatenating CLS + patches.
        // Exported with the legacy TorchScript ONNX path because the newer
        // torch.export/dynamo exporter currently fails on EUPE.
        // Trained with proxy distillation: a compact student distilled from a
        // single large proxy teacher that itself aggregates multiple experts.
        RegistryEntry {
            info: ModelInfo {
                name: "eupe-vit-b16".to_string(),
                architecture: "ViT-B/16".to_string(),
                patch_size: 16,
                embed_dim: 768,
                num_layers: 12,
                num_heads: 12,
                method: SSLMethod::EUPE,
                input_size: 224,
                params_m: 86,
            },
            availability: Availability::ready(
                "EUPE ViT-B/16 proxy-distilled from a large universal teacher into a compact efficient encoder.",
            ),
            artifacts: vec![
                ModelArtifact {
                    relative_path: "eupe-vit-b16/model.onnx".to_string(),
                    download_url:
                        "https://huggingface.co/abdelstark/eupe-vit-b16-onnx/resolve/main/model.onnx"
                            .to_string(),
                    checksum: Checksum::Sha256(
                        "e0f0572f72afbb4857c960bbd99c3a6d71c8c7834b5cf0ed99632f53ba384f0f"
                            .to_string(),
                    ),
                },
                ModelArtifact {
                    relative_path: "eupe-vit-b16/model.onnx_data".to_string(),
                    download_url:
                        "https://huggingface.co/abdelstark/eupe-vit-b16-onnx/resolve/main/model.onnx_data"
                            .to_string(),
                    checksum: Checksum::Sha256(
                        "66fd1a1988c68746d49083026b9fbffc278c60061eaed9ef5c89d80b5716da87"
                            .to_string(),
                    ),
                },
            ],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "facebookresearch/eupe",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 196,
                    embedding_dim: 768,
                },
            ),
        },
        RegistryEntry {
            info: ModelInfo {
                name: "siglip-so400m".to_string(),
                architecture: "ViT-SO400M/14".to_string(),
                patch_size: 14,
                embed_dim: 1152,
                num_layers: 27,
                num_heads: 16,
                method: SSLMethod::SigLIP,
                input_size: 224,
                params_m: 400,
            },
            availability: Availability::planned("Phase 3", phase_three_note),
            artifacts: vec![ModelArtifact {
                relative_path: "siglip-so400m.onnx".to_string(),
                download_url:
                    "https://huggingface.co/google/siglip-so400m-patch14-224/resolve/main/onnx/model.onnx".to_string(),
                checksum: Checksum::Pending {
                    reason: "Download metadata will be validated when SigLIP support lands in the comparison milestone."
                        .to_string(),
                },
            }],
            norm_mean: [0.5, 0.5, 0.5],
            norm_std: [0.5, 0.5, 0.5],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
            video_frames: None,
            validation: default_validation_profile(
                "google/siglip-so400m-patch14-224",
                PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 256,
                    embedding_dim: 1152,
                },
            ),
        },
    ]
}

/// Look up a registry entry by model name.
pub fn find(name: &str) -> Option<RegistryEntry> {
    registry().into_iter().find(|entry| entry.info.name == name)
}

/// Look up a model that is ready for inference in the current phase.
pub fn find_ready(name: &str) -> Result<RegistryEntry, ModelError> {
    let entry = find(name).ok_or_else(|| ModelError::NotFound(name.to_string()))?;
    entry.ensure_ready()?;
    Ok(entry)
}

/// List all known model names.
pub fn model_names() -> Vec<String> {
    registry()
        .into_iter()
        .map(|entry| entry.info.name)
        .collect()
}

/// List model names that are ready to load in the current implementation phase.
pub fn ready_model_names() -> Vec<String> {
    registry()
        .into_iter()
        .filter(RegistryEntry::is_ready)
        .map(|entry| entry.info.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_contains_current_and_planned_models() {
        let names = model_names();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"dinov2-vit-l14".to_string()));
        assert!(names.contains(&"dinov3-vit-l14".to_string()));
        assert!(names.contains(&"mae-vit-l16".to_string()));
        assert!(names.contains(&"clip-vit-l14".to_string()));
        assert!(names.contains(&"ijepa-vit-h14".to_string()));
        assert!(names.contains(&"vjepa2-vitl-fpc2-256".to_string()));
        assert!(names.contains(&"eupe-vit-b16".to_string()));
        assert!(names.contains(&"siglip-so400m".to_string()));
    }

    #[test]
    fn test_ready_models() {
        let ready = ready_model_names();
        assert_eq!(
            ready,
            vec![
                "dinov2-vit-l14".to_string(),
                "ijepa-vit-h14".to_string(),
                "vjepa2-vitl-fpc2-256".to_string(),
                "eupe-vit-b16".to_string(),
            ]
        );
    }

    #[test]
    fn test_find_ready_rejects_planned_models() {
        let error = find_ready("clip-vit-l14").unwrap_err();
        assert!(matches!(error, ModelError::Unavailable { .. }));
    }
}
