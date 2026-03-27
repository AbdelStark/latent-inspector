use crate::errors::ModelError;
#[cfg(feature = "onnx-inference")]
use crate::models::cache;
use crate::models::preprocess::{self, PreprocessConfig};
use crate::models::registry::{self, RegistryEntry};
use image::DynamicImage;
use ndarray::{Array1, Array2, Array4};
#[cfg(feature = "onnx-inference")]
use tracing::info;

/// Normalized output from any model.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Global CLS token representation `[D]`. `None` for models without CLS (e.g. MAE encoder).
    pub cls_token: Option<Array1<f32>>,
    /// Per-patch features `[N_patches, D]`.
    pub patch_tokens: Array2<f32>,
    /// Attention weights `[layers, heads, N, N]`. `None` when not exported.
    pub attention_weights: Option<ndarray::Array4<f32>>,
    /// Metadata about the model.
    pub model_info: registry::ModelInfo,
    /// Explicit metadata about the consumed tensor.
    pub tensor_metadata: OutputTensorMetadata,
}

#[derive(Debug, Clone)]
pub struct OutputTensorMetadata {
    pub input_name: String,
    pub input_shape: Vec<usize>,
    pub output_name: String,
    pub output_shape: Vec<usize>,
    pub sequence_has_cls: bool,
    pub observed_patch_count: usize,
    pub embedding_dim: usize,
}

/// A loaded model session ready to run inference.
pub struct ModelSession {
    entry: RegistryEntry,
    inner: SessionInner,
}

/// Inner session — either a real ONNX session or a stub for testing.
enum SessionInner {
    #[cfg(feature = "onnx-inference")]
    Onnx(std::sync::Mutex<ort::session::Session>),
    Stub,
}

impl ModelSession {
    /// Load a model by name.  Downloads if not cached.
    pub fn load(model_name: &str) -> Result<Self, ModelError> {
        let entry = registry::find(model_name)
            .ok_or_else(|| ModelError::NotFound(model_name.to_string()))?;

        #[cfg(not(feature = "onnx-inference"))]
        {
            Ok(Self {
                entry,
                inner: SessionInner::Stub,
            })
        }
        #[cfg(feature = "onnx-inference")]
        {
            let path = cache::model_path(model_name)?;

            if !cache::is_cached(model_name)? {
                info!("Model '{}' not found in cache — downloading", model_name);
                cache::download(model_name, &entry)?;
            } else {
                info!(
                    "Model '{}' found in cache at {}",
                    model_name,
                    path.display()
                );
            }

            let inner = Self::create_session(&path)?;

            Ok(Self { entry, inner })
        }
    }

    #[cfg(feature = "onnx-inference")]
    fn create_session(path: &std::path::Path) -> Result<SessionInner, ModelError> {
        #[cfg(feature = "onnx-inference")]
        {
            if path.exists() {
                let session = ort::session::Session::builder()
                    .map_err(|e| ModelError::SessionCreation(e.to_string()))?
                    .commit_from_file(path)
                    .map_err(|e| ModelError::SessionCreation(e.to_string()))?;
                return Ok(SessionInner::Onnx(std::sync::Mutex::new(session)));
            }
        }
        let _ = path;
        Ok(SessionInner::Stub)
    }

    /// Run inference on a preprocessed image tensor `[1, 3, H, W]`.
    pub fn run(&self, tensor: &Array4<f32>) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let contract = &self.entry.validation.tensor;
        let n_patches = contract.patch_count;
        let d = contract.embedding_dim;
        let input_shape = tensor.shape().to_vec();
        #[cfg(not(feature = "onnx-inference"))]
        let _ = tensor;

        match &self.inner {
            #[cfg(feature = "onnx-inference")]
            SessionInner::Onnx(session) => {
                let input_shape = tensor.shape().to_vec();
                let flat: Vec<f32> = tensor.iter().copied().collect();
                let ort_tensor = ort::value::TensorRef::from_array_view((
                    input_shape.as_slice(),
                    flat.as_slice(),
                ))
                .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let mut session = session.lock().map_err(|e| {
                    ModelError::InferenceFailed(format!("Failed to lock ONNX session: {e}"))
                })?;
                let outputs = session
                    .run(ort::inputs![self.entry.input_name.as_str() => ort_tensor])
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let hidden = outputs
                    .get(self.entry.output_name.as_str())
                    .ok_or_else(|| {
                        ModelError::InferenceFailed(format!(
                            "Output tensor '{}' not found",
                            self.entry.output_name
                        ))
                    })?;
                let (hidden_shape, hidden_data) = hidden
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;
                let hidden_shape: Vec<usize> =
                    hidden_shape.iter().map(|&dim| dim as usize).collect();
                let hidden_array = ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&hidden_shape),
                    hidden_data.to_vec(),
                )
                .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                // Shape: [1, seq_len, D] where seq_len = 1 (CLS) + N_patches
                let seq_len = hidden_array.shape()[1];
                let has_cls = if contract.cls_expected {
                    seq_len == n_patches + 1 || seq_len > n_patches
                } else {
                    false
                };

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
                    tensor_metadata: OutputTensorMetadata {
                        input_name: self.entry.input_name.clone(),
                        input_shape,
                        output_name: self.entry.output_name.clone(),
                        output_shape: hidden_shape,
                        sequence_has_cls: has_cls,
                        observed_patch_count: seq_len.saturating_sub(patch_start),
                        embedding_dim: d,
                    },
                })
            }
            SessionInner::Stub => {
                // Deterministic stub data — useful for tests and demos
                let cls_token = contract.cls_expected.then(|| Array1::from_elem(d, 0.1_f32));
                let patch_tokens = Array2::from_elem((n_patches, d), 0.1_f32);
                Ok(ModelOutput {
                    cls_token,
                    patch_tokens,
                    attention_weights: None,
                    model_info: info.clone(),
                    tensor_metadata: OutputTensorMetadata {
                        input_name: self.entry.input_name.clone(),
                        input_shape,
                        output_name: self.entry.output_name.clone(),
                        output_shape: contract.expected_shape(),
                        sequence_has_cls: contract.cls_expected,
                        observed_patch_count: n_patches,
                        embedding_dim: d,
                    },
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

    pub fn entry(&self) -> &RegistryEntry {
        &self.entry
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

        let n_patches = entry.validation.tensor.patch_count;
        let d = entry.validation.tensor.embedding_dim;

        assert_eq!(output.patch_tokens.shape(), &[n_patches, d]);
        assert!(output.cls_token.is_some());
        assert_eq!(output.cls_token.unwrap().len(), d);
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 257, 1024]);
    }

    #[test]
    fn test_stub_mae_semantics_skip_cls() {
        let entry = registry::find("mae-vit-l16").unwrap();
        let session = ModelSession {
            inner: SessionInner::Stub,
            entry,
        };
        let img = image::DynamicImage::new_rgb8(224, 224);
        let output = session.infer(&img).unwrap();

        assert!(output.cls_token.is_none());
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 196, 1024]);
    }
}
