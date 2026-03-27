use crate::errors::ModelError;
use crate::models::cache;
use crate::models::preprocess::{self, PreprocessConfig};
use crate::models::registry::{self, RegistryEntry};
use image::DynamicImage;
use ndarray::{Array1, Array2, Array4, Axis, Ix3};
use ort::session::Session;
use ort::value::TensorRef;
use tracing::info;

/// Normalized output from any model.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Global CLS token representation `[D]`. `None` for models without CLS.
    pub cls_token: Option<Array1<f32>>,
    /// Per-patch features `[N_patches, D]`.
    pub patch_tokens: Array2<f32>,
    /// Attention weights `[layers, heads, N, N]`. `None` when not exported.
    pub attention_weights: Option<Array4<f32>>,
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

/// Explicit backend used by a `ModelSession`.
enum SessionInner {
    Onnx(Session),
    Stub,
}

impl ModelSession {
    /// Load a model by name. Downloads the artifact if it is ready and not yet cached.
    pub fn load(model_name: &str) -> Result<Self, ModelError> {
        let entry = registry::find_ready(model_name)?;
        if use_stub_backend() {
            info!(
                "Using explicit stub backend for '{}' via LATENT_INSPECTOR_MODEL_BACKEND=stub",
                model_name
            );
            return Ok(Self {
                entry,
                inner: SessionInner::Stub,
            });
        }

        let path = cache::model_path(model_name)?;

        if !cache::is_cached(model_name)? {
            info!("Model '{}' not found in cache, downloading", model_name);
            cache::download(model_name, &entry)?;
        } else {
            info!(
                "Model '{}' found in cache at {}",
                model_name,
                path.display()
            );
        }

        let inner = Self::create_session(&entry, &path)?;
        Ok(Self { entry, inner })
    }

    fn create_session(
        entry: &RegistryEntry,
        path: &std::path::Path,
    ) -> Result<SessionInner, ModelError> {
        let intra_threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1);

        let mut builder = Session::builder()
            .map_err(|e| ModelError::SessionCreation(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| ModelError::SessionCreation(e.to_string()))?;

        let session = builder
            .commit_from_file(path)
            .map_err(|e| ModelError::SessionCreation(e.to_string()))?;

        Self::validate_graph(entry, &session)?;
        Ok(SessionInner::Onnx(session))
    }

    fn validate_graph(entry: &RegistryEntry, session: &Session) -> Result<(), ModelError> {
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        if !input_names.iter().any(|name| name == &entry.input_name) {
            return Err(ModelError::GraphMismatch {
                name: entry.info.name.clone(),
                kind: "input".to_string(),
                expected: entry.input_name.clone(),
                available: input_names,
            });
        }

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();
        if !output_names.iter().any(|name| name == &entry.output_name) {
            return Err(ModelError::GraphMismatch {
                name: entry.info.name.clone(),
                kind: "output".to_string(),
                expected: entry.output_name.clone(),
                available: output_names,
            });
        }

        Ok(())
    }

    /// Run inference on a preprocessed image tensor `[1, 3, H, W]`.
    pub fn run(&mut self, tensor: &Array4<f32>) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let contract = &self.entry.validation.tensor;
        let expected_patches = contract.patch_count;
        let expected_dim = contract.embedding_dim;
        let input_shape = tensor.shape().to_vec();

        match &mut self.inner {
            SessionInner::Onnx(session) => {
                let input_data = tensor.as_slice().ok_or_else(|| {
                    ModelError::InferenceFailed(
                        "Input tensor must be contiguous in memory".to_string(),
                    )
                })?;
                let input = TensorRef::from_array_view((tensor.shape(), input_data))
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let outputs = session
                    .run(ort::inputs![self.entry.input_name.as_str() => input])
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let hidden = outputs
                    .get(self.entry.output_name.as_str())
                    .ok_or_else(|| ModelError::GraphMismatch {
                        name: info.name.clone(),
                        kind: "output".to_string(),
                        expected: self.entry.output_name.clone(),
                        available: outputs.keys().map(str::to_string).collect(),
                    })?;

                let (shape, values) = hidden
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                let output_shape: Vec<usize> = shape
                    .iter()
                    .map(|&dim| {
                        usize::try_from(dim).map_err(|_| {
                            ModelError::InferenceFailed(format!(
                                "Output '{}' contains invalid dimension {dim}",
                                self.entry.output_name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let hidden_array =
                    ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&output_shape), values)
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?
                        .into_dimensionality::<Ix3>()
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

                if hidden_array.shape()[0] != contract.batch_size {
                    return Err(ModelError::InferenceFailed(format!(
                        "Expected batch dimension {} for '{}', got {:?}",
                        contract.batch_size, self.entry.output_name, output_shape
                    )));
                }

                if hidden_array.shape()[2] != expected_dim {
                    return Err(ModelError::InferenceFailed(format!(
                        "Expected embed dim {} for '{}', got {:?}",
                        expected_dim, self.entry.output_name, output_shape
                    )));
                }

                let seq_len = hidden_array.shape()[1];
                let has_cls = if contract.cls_expected {
                    seq_len == expected_patches + 1
                } else {
                    false
                };

                if seq_len != expected_patches && !has_cls {
                    return Err(ModelError::InferenceFailed(format!(
                        "Expected {} or {} tokens for '{}', got {}",
                        expected_patches,
                        expected_patches + usize::from(contract.cls_expected),
                        info.name,
                        seq_len
                    )));
                }

                let tokens = hidden_array.index_axis(Axis(0), 0);
                let cls_token = has_cls.then(|| tokens.index_axis(Axis(0), 0).to_owned());
                let patch_start = usize::from(has_cls);
                let patch_tokens = tokens.slice(ndarray::s![patch_start.., ..]).to_owned();

                Ok(ModelOutput {
                    cls_token,
                    patch_tokens,
                    attention_weights: None,
                    model_info: info.clone(),
                    tensor_metadata: OutputTensorMetadata {
                        input_name: self.entry.input_name.clone(),
                        input_shape,
                        output_name: self.entry.output_name.clone(),
                        output_shape,
                        sequence_has_cls: has_cls,
                        observed_patch_count: expected_patches,
                        embedding_dim: expected_dim,
                    },
                })
            }
            SessionInner::Stub => {
                let cls_token = contract
                    .cls_expected
                    .then(|| Array1::from_elem(expected_dim, 0.1_f32));
                let patch_tokens = Array2::from_elem((expected_patches, expected_dim), 0.1_f32);
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
                        observed_patch_count: expected_patches,
                        embedding_dim: expected_dim,
                    },
                })
            }
        }
    }

    /// Preprocess an image and run inference in one call.
    pub fn infer(&mut self, img: &DynamicImage) -> Result<ModelOutput, ModelError> {
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

fn use_stub_backend() -> bool {
    std::env::var("LATENT_INSPECTOR_MODEL_BACKEND")
        .map(|value| value.eq_ignore_ascii_case("stub"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ModelSession {
        fn stubbed(entry: RegistryEntry) -> Self {
            Self {
                inner: SessionInner::Stub,
                entry,
            }
        }
    }

    #[test]
    fn test_stub_inference_shapes() {
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut session = ModelSession::stubbed(entry.clone());
        let img = image::DynamicImage::new_rgb8(224, 224);
        let output = session.infer(&img).unwrap();

        let n_patches = entry.validation.tensor.patch_count;
        let embed_dim = entry.validation.tensor.embedding_dim;

        assert_eq!(output.patch_tokens.shape(), &[n_patches, embed_dim]);
        assert!(output.cls_token.is_some());
        assert_eq!(output.cls_token.unwrap().len(), embed_dim);
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 257, 1024]);
    }

    #[test]
    fn test_stub_mae_semantics_skip_cls() {
        let entry = registry::find("mae-vit-l16").unwrap();
        let mut session = ModelSession::stubbed(entry);
        let img = image::DynamicImage::new_rgb8(224, 224);
        let output = session.infer(&img).unwrap();

        assert!(output.cls_token.is_none());
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 196, 1024]);
    }

    #[test]
    fn test_load_rejects_planned_models() {
        let result = ModelSession::load("clip-vit-l14");
        assert!(matches!(result, Err(ModelError::Unavailable { .. })));
    }
}
