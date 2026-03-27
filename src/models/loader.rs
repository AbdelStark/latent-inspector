use crate::errors::ModelError;
use crate::models::cache;
use crate::models::preprocess::{self, PreprocessConfig};
use crate::models::registry::{self, RegistryEntry};
use image::DynamicImage;
use ndarray::{Array1, Array2, Array4};
use tracing::info;

/// Normalized output from any model.
pub struct ModelOutput {
    /// Global CLS token representation `[D]`. `None` for models without CLS (e.g. MAE encoder).
    pub cls_token: Option<Array1<f32>>,
    /// Per-patch features `[N_patches, D]`.
    pub patch_tokens: Array2<f32>,
    /// Attention weights `[layers, heads, N, N]`. `None` when not exported.
    pub attention_weights: Option<ndarray::Array4<f32>>,
    /// Metadata about the model.
    pub model_info: registry::ModelInfo,
}

/// A loaded model session ready to run inference.
pub struct ModelSession {
    entry: RegistryEntry,
    inner: SessionInner,
}

/// Inner session — either a real ONNX session or a stub for testing.
enum SessionInner {
    #[cfg(feature = "onnx-inference")]
    Onnx(ort::Session),
    Stub,
}

impl ModelSession {
    /// Load a model by name.  Downloads if not cached.
    pub fn load(model_name: &str) -> Result<Self, ModelError> {
        let entry = registry::find(model_name).ok_or_else(|| {
            ModelError::NotFound(model_name.to_string())
        })?;

        let path = cache::model_path(model_name)?;

        if !path.exists() {
            info!("Model '{}' not found in cache — downloading", model_name);
            cache::download(model_name, &entry.download_url, &path, &entry.sha256)?;
        } else {
            info!("Model '{}' found in cache at {}", model_name, path.display());
        }

        let inner = Self::create_session(&path)?;

        Ok(Self { entry, inner })
    }

    fn create_session(path: &std::path::Path) -> Result<SessionInner, ModelError> {
        #[cfg(feature = "onnx-inference")]
        {
            if path.exists() {
                let session = ort::Session::builder()
                    .map_err(|e| ModelError::SessionCreation(e.to_string()))?
                    .commit_from_file(path)
                    .map_err(|e| ModelError::SessionCreation(e.to_string()))?;
                return Ok(SessionInner::Onnx(session));
            }
        }
        let _ = path; // suppress unused warning in stub mode
        Ok(SessionInner::Stub)
    }

    /// Run inference on a preprocessed image tensor `[1, 3, H, W]`.
    pub fn run(&self, _tensor: &Array4<f32>) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let n_patches = ((info.input_size / info.patch_size) as usize).pow(2);
        let d = info.embed_dim as usize;

        match &self.inner {
            #[cfg(feature = "onnx-inference")]
            SessionInner::Onnx(session) => {
                use ndarray::ArrayD;

                let input_shape = tensor.shape().to_vec();
                let flat: Vec<f32> = tensor.iter().copied().collect();
                let ort_tensor = ort::Value::from_array(
                    ndarray::Array::from_shape_vec(input_shape.as_slice(), flat)
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?,
                )
                .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let outputs = session
                    .run(ort::inputs![self.entry.input_name.as_str() => ort_tensor]
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?)
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let hidden: ort::Value = outputs[self.entry.output_name.as_str()].clone();
                let hidden_array: ndarray::ArrayViewD<f32> = hidden
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                // Shape: [1, seq_len, D] where seq_len = 1 (CLS) + N_patches
                let seq_len = hidden_array.shape()[1];
                let has_cls = seq_len == n_patches + 1;

                let cls_token = if has_cls {
                    let cls = hidden_array
                        .slice(ndarray::s![0, 0, ..])
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;
                    Some(cls)
                } else {
                    None
                };

                let patch_start = if has_cls { 1 } else { 0 };
                let patch_tokens = hidden_array
                    .slice(ndarray::s![0, patch_start.., ..])
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                Ok(ModelOutput {
                    cls_token,
                    patch_tokens,
                    attention_weights: None,
                    model_info: info.clone(),
                })
            }
            SessionInner::Stub => {
                // Deterministic stub data — useful for tests and demos
                let cls_token = Some(Array1::from_elem(d, 0.1_f32));
                let patch_tokens = Array2::from_elem((n_patches, d), 0.1_f32);
                Ok(ModelOutput {
                    cls_token,
                    patch_tokens,
                    attention_weights: None,
                    model_info: info.clone(),
                })
            }
        }
    }

    /// Preprocess an image and run inference in one call.
    pub fn infer(&self, img: &DynamicImage) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let cfg = PreprocessConfig::new(info.input_size, self.entry.norm_mean, self.entry.norm_std);
        let tensor = preprocess::preprocess(img, &cfg)?;
        self.run(&tensor)
    }

    /// Model metadata.
    pub fn info(&self) -> &registry::ModelInfo {
        &self.entry.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry;

    #[test]
    fn test_stub_inference_shapes() {
        // Use a stub session (path doesn't exist → stub)
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let session = ModelSession {
            inner: SessionInner::Stub,
            entry: entry.clone(),
        };
        let img = image::DynamicImage::new_rgb8(224, 224);
        let output = session.infer(&img).unwrap();

        let n_patches = ((entry.info.input_size / entry.info.patch_size) as usize).pow(2);
        let d = entry.info.embed_dim as usize;

        assert_eq!(output.patch_tokens.shape(), &[n_patches, d]);
        assert!(output.cls_token.is_some());
        assert_eq!(output.cls_token.unwrap().len(), d);
    }
}
