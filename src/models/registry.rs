use serde::{Deserialize, Serialize};

/// Self-supervised learning method used to train the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SSLMethod {
    DINO,
    MAE,
    IJEPA,
    CLIP,
    SigLIP,
}

impl std::fmt::Display for SSLMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSLMethod::DINO => write!(f, "DINO"),
            SSLMethod::MAE => write!(f, "MAE"),
            SSLMethod::IJEPA => write!(f, "I-JEPA"),
            SSLMethod::CLIP => write!(f, "CLIP"),
            SSLMethod::SigLIP => write!(f, "SigLIP"),
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

/// Full registry entry including download metadata.
#[derive(Debug, Clone)]
pub struct ModelArtifact {
    /// Relative path within the cache directory.
    pub relative_path: String,
    /// HuggingFace Hub download URL.
    pub download_url: String,
    /// Expected SHA-256 hex digest of the downloaded file.
    pub sha256: String,
}

/// Full registry entry including download metadata.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub info: ModelInfo,
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
}

/// Returns the full model registry.
pub fn registry() -> Vec<RegistryEntry> {
    vec![
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
            artifacts: vec![ModelArtifact {
                relative_path: "dinov2-vit-l14.onnx".to_string(),
                download_url:
                    "https://huggingface.co/onnx-community/dinov2-large/resolve/main/onnx/model.onnx".to_string(),
                sha256: "placeholder_sha256_dinov2_l14".to_string(),
            }],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
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
            artifacts: vec![ModelArtifact {
                relative_path: "mae-vit-l16.onnx".to_string(),
                download_url:
                    "https://huggingface.co/facebook/vit-mae-large/resolve/main/model.onnx".to_string(),
                sha256: "placeholder_sha256_mae_l16".to_string(),
            }],
            norm_mean: [0.5, 0.5, 0.5],
            norm_std: [0.5, 0.5, 0.5],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
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
            artifacts: vec![ModelArtifact {
                relative_path: "clip-vit-l14.onnx".to_string(),
                download_url:
                    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/onnx/visual.onnx".to_string(),
                sha256: "placeholder_sha256_clip_l14".to_string(),
            }],
            norm_mean: [0.48145467, 0.4578275, 0.40821073],
            norm_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
        },
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
            artifacts: vec![
                ModelArtifact {
                    relative_path: "ijepa-vit-h14/model.onnx".to_string(),
                    download_url:
                        "https://huggingface.co/onnx-community/ijepa_vith14_1k/resolve/main/onnx/model.onnx"
                            .to_string(),
                    sha256: "placeholder_sha256_ijepa_h14".to_string(),
                },
                ModelArtifact {
                    relative_path: "ijepa-vit-h14/model.onnx_data".to_string(),
                    download_url:
                        "https://huggingface.co/onnx-community/ijepa_vith14_1k/resolve/main/onnx/model.onnx_data"
                            .to_string(),
                    sha256: "placeholder_sha256_ijepa_h14_data".to_string(),
                },
            ],
            norm_mean: [0.485, 0.456, 0.406],
            norm_std: [0.229, 0.224, 0.225],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
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
            artifacts: vec![ModelArtifact {
                relative_path: "siglip-so400m.onnx".to_string(),
                download_url:
                    "https://huggingface.co/google/siglip-so400m-patch14-224/resolve/main/onnx/model.onnx".to_string(),
                sha256: "placeholder_sha256_siglip_so400m".to_string(),
            }],
            norm_mean: [0.5, 0.5, 0.5],
            norm_std: [0.5, 0.5, 0.5],
            input_name: "pixel_values".to_string(),
            output_name: "last_hidden_state".to_string(),
        },
    ]
}

/// Look up a registry entry by model name.
pub fn find(name: &str) -> Option<RegistryEntry> {
    registry().into_iter().find(|e| e.info.name == name)
}

/// List all known model names.
pub fn model_names() -> Vec<String> {
    registry().into_iter().map(|e| e.info.name).collect()
}

impl RegistryEntry {
    pub fn primary_artifact(&self) -> &ModelArtifact {
        &self.artifacts[0]
    }
}
