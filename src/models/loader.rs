use crate::errors::ModelError;
use crate::models::cache;
use crate::models::preprocess::{self, PreprocessConfig};
use crate::models::registry::{self, RegistryEntry};
use image::DynamicImage;
use ndarray::{Array1, Array2, Array4, Axis, Ix3};
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use std::path::Path;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBackend {
    OnnxRuntime,
    Stub,
}

impl InferenceBackend {
    pub fn label(self) -> &'static str {
        match self {
            InferenceBackend::OnnxRuntime => "onnx-runtime",
            InferenceBackend::Stub => "stub",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            InferenceBackend::OnnxRuntime => "ONNX Runtime",
            InferenceBackend::Stub => "Stub backend",
        }
    }
}

impl std::fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A loaded model session ready to run inference.
pub struct ModelSession {
    entry: RegistryEntry,
    inner: SessionInner,
}

/// Explicit backend used by a `ModelSession`.
enum SessionInner {
    Onnx(Session),
    Stub(StubConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceOutputLayout {
    ContractDriven,
    PatchOnly,
}

impl SessionInner {
    fn backend(&self) -> InferenceBackend {
        match self {
            SessionInner::Onnx(_) => InferenceBackend::OnnxRuntime,
            SessionInner::Stub(_) => InferenceBackend::Stub,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StubConfig {
    seed: u64,
    checkpoint_progress: Option<f32>,
    variant_seed: u64,
}

impl StubConfig {
    fn standard(model_name: &str) -> Self {
        Self {
            seed: stub_seed_value(model_name),
            checkpoint_progress: None,
            variant_seed: 0,
        }
    }

    fn checkpoint(model_name: &str, path: &Path) -> Self {
        Self {
            seed: stub_seed_value(model_name),
            checkpoint_progress: checkpoint_progress_value(path),
            variant_seed: stub_seed(path),
        }
    }
}

impl ModelSession {
    /// Load a model by name. Downloads the artifact if it is ready and not yet cached.
    pub fn load(model_name: &str) -> Result<Self, ModelError> {
        Self::load_impl(model_name, false)
    }

    /// Load a model for report-oriented analysis. In stub mode this can use
    /// planned registry entries so multi-model flows can be exercised before the
    /// real ONNX integration is available.
    pub fn load_for_analysis(model_name: &str) -> Result<Self, ModelError> {
        Self::load_impl(model_name, true)
    }

    /// Load a model session from an explicit ONNX artifact path while reusing the
    /// registered preprocessing and tensor contracts for the selected model.
    pub fn load_checkpoint(model_name: &str, artifact_path: &Path) -> Result<Self, ModelError> {
        let entry = registry::find_ready(model_name)?;
        validate_artifact_path(&entry, artifact_path)?;

        if use_stub_backend() {
            info!(
                "Using explicit stub backend for checkpoint '{}' via LATENT_INSPECTOR_MODEL_BACKEND=stub",
                artifact_path.display()
            );
            return Ok(Self {
                entry,
                inner: SessionInner::Stub(StubConfig::checkpoint(model_name, artifact_path)),
            });
        }

        let inner = Self::create_session(&entry, artifact_path)?;
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

    fn output_from_hidden(
        entry: &RegistryEntry,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        hidden_array: ndarray::ArrayView3<'_, f32>,
        layout: SequenceOutputLayout,
    ) -> Result<ModelOutput, ModelError> {
        let info = &entry.info;
        let contract = &entry.validation.tensor;

        if hidden_array.shape()[0] != contract.batch_size {
            return Err(ModelError::InferenceFailed(format!(
                "Expected batch dimension {} for '{}', got {:?}",
                contract.batch_size, entry.output_name, output_shape
            )));
        }

        let observed_dim = hidden_array.shape()[2];
        if observed_dim != contract.embedding_dim {
            return Err(ModelError::InferenceFailed(format!(
                "Expected embed dim {} for '{}', got {:?}",
                contract.embedding_dim, entry.output_name, output_shape
            )));
        }

        let seq_len = hidden_array.shape()[1];
        let (sequence_has_cls, patch_start) = match layout {
            SequenceOutputLayout::ContractDriven => {
                let expected_patches = contract.patch_count;
                let expected_with_cls = expected_patches + 1;
                let has_cls = contract.cls_expected && seq_len == expected_with_cls;

                if seq_len != expected_patches && seq_len != expected_with_cls {
                    return Err(ModelError::InferenceFailed(format!(
                        "Expected {} or {} tokens for '{}', got {}",
                        expected_patches, expected_with_cls, info.name, seq_len
                    )));
                }

                if contract.cls_expected && !has_cls {
                    return Err(ModelError::InferenceFailed(format!(
                        "Contract for '{}' expects a CLS token (sequence length {}), \
                         but got {} tokens (patches only)",
                        info.name, expected_with_cls, seq_len
                    )));
                }

                (has_cls, usize::from(has_cls))
            }
            SequenceOutputLayout::PatchOnly => {
                if contract.cls_expected {
                    return Err(ModelError::InferenceFailed(format!(
                        "Patch-only output handling cannot satisfy the CLS-bearing contract for '{}'",
                        info.name
                    )));
                }
                if seq_len != contract.patch_count {
                    return Err(ModelError::InferenceFailed(format!(
                        "Expected {} patch tokens for '{}', got {}",
                        contract.patch_count, info.name, seq_len
                    )));
                }

                (false, 0)
            }
        };

        let tokens = hidden_array.index_axis(Axis(0), 0);
        let cls_token = sequence_has_cls.then(|| tokens.index_axis(Axis(0), 0).to_owned());
        let patch_tokens = tokens.slice(ndarray::s![patch_start.., ..]).to_owned();
        let observed_patch_count = patch_tokens.nrows();

        Ok(ModelOutput {
            cls_token,
            patch_tokens,
            attention_weights: None,
            model_info: info.clone(),
            tensor_metadata: OutputTensorMetadata {
                input_name: entry.input_name.clone(),
                input_shape,
                output_name: entry.output_name.clone(),
                output_shape,
                sequence_has_cls,
                observed_patch_count,
                embedding_dim: observed_dim,
            },
        })
    }

    /// Run inference on a preprocessed image tensor `[1, 3, H, W]`.
    pub fn run(&mut self, tensor: &Array4<f32>) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let contract = &self.entry.validation.tensor;
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
                Self::output_from_hidden(
                    &self.entry,
                    input_shape,
                    output_shape,
                    hidden_array,
                    SequenceOutputLayout::ContractDriven,
                )
            }
            SessionInner::Stub(config) => {
                let (cls_token, patch_tokens, attention_weights) =
                    seeded_stub_features(*config, tensor, contract, info.num_heads as usize);
                let sequence_has_cls = cls_token.is_some();
                let observed_patch_count = patch_tokens.nrows();
                let embedding_dim = patch_tokens.ncols();

                Ok(ModelOutput {
                    cls_token,
                    patch_tokens,
                    attention_weights,
                    model_info: info.clone(),
                    tensor_metadata: OutputTensorMetadata {
                        input_name: self.entry.input_name.clone(),
                        input_shape,
                        output_name: self.entry.output_name.clone(),
                        output_shape: vec![
                            contract.batch_size,
                            observed_patch_count + usize::from(sequence_has_cls),
                            embedding_dim,
                        ],
                        sequence_has_cls,
                        observed_patch_count,
                        embedding_dim,
                    },
                })
            }
        }
    }

    /// Preprocess an image and run inference in one call.
    ///
    /// For video-backbone models (e.g. V-JEPA 2), the preprocessed frame is
    /// duplicated to form the required multi-frame input tensor.
    pub fn infer(&mut self, img: &DynamicImage) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
        let cfg = PreprocessConfig::new(info.input_size, self.entry.norm_mean, self.entry.norm_std);
        let tensor = preprocess::preprocess(img, &cfg)?;

        if let Some(num_frames) = self.entry.video_frames {
            // Build [1, F, 3, H, W] by duplicating the single frame F times.
            let frame = tensor.index_axis(Axis(0), 0); // [3, H, W]
            let frames: Vec<_> = (0..num_frames)
                .map(|_| frame.view().insert_axis(Axis(0)))
                .collect();
            let video = ndarray::concatenate(Axis(0), &frames)
                .map_err(|e| ModelError::InferenceFailed(e.to_string()))?
                .insert_axis(Axis(0)); // [1, F, 3, H, W]
            self.run_video(&video)
        } else {
            self.run(&tensor)
        }
    }

    /// Run inference with a 5D video tensor `[1, F, 3, H, W]`.
    fn run_video(&mut self, tensor: &ndarray::Array5<f32>) -> Result<ModelOutput, ModelError> {
        let info = &self.entry.info;
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
                                "ONNX output dimension {dim} is not a valid usize"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Output should be [batch, patches, embed_dim]
                let hidden_array =
                    ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&output_shape), values)
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?
                        .into_dimensionality::<Ix3>()
                        .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;
                Self::output_from_hidden(
                    &self.entry,
                    input_shape,
                    output_shape,
                    hidden_array,
                    SequenceOutputLayout::PatchOnly,
                )
            }
            SessionInner::Stub(_) => {
                // For stub mode, generate features from the first frame
                let frame = tensor.index_axis(Axis(0), 0); // [F, 3, H, W]
                let first_frame = frame.index_axis(Axis(0), 0).insert_axis(Axis(0)).to_owned(); // [1, 3, H, W]
                self.run(&first_frame)
            }
        }
    }

    /// Model metadata.
    pub fn info(&self) -> &registry::ModelInfo {
        &self.entry.info
    }

    pub fn entry(&self) -> &RegistryEntry {
        &self.entry
    }

    pub fn backend(&self) -> InferenceBackend {
        self.inner.backend()
    }

    fn load_impl(model_name: &str, allow_planned_stub: bool) -> Result<Self, ModelError> {
        let stub_backend = use_stub_backend();
        let entry = if stub_backend && allow_planned_stub {
            registry::find(model_name)
                .ok_or_else(|| ModelError::NotFound(model_name.to_string()))?
        } else {
            registry::find_ready(model_name)?
        };

        if stub_backend {
            info!(
                "Using explicit stub backend for '{}' via LATENT_INSPECTOR_MODEL_BACKEND=stub",
                model_name
            );
            if allow_planned_stub && !entry.is_ready() {
                info!(
                    "Allowing planned model '{}' in stub analysis mode for development-only report generation",
                    model_name
                );
            }
            return Ok(Self {
                entry,
                inner: SessionInner::Stub(StubConfig::standard(model_name)),
            });
        }

        let path = cache::ensure_artifacts(model_name, &entry)?;
        info!(
            "Model '{}' is ready from cache bundle rooted at {}",
            model_name,
            path.display()
        );

        let inner = Self::create_session(&entry, &path)?;
        Ok(Self { entry, inner })
    }
}

fn use_stub_backend() -> bool {
    std::env::var("LATENT_INSPECTOR_MODEL_BACKEND")
        .map(|value| value.eq_ignore_ascii_case("stub"))
        .unwrap_or(false)
}

/// FNV-1a hash — deterministic across Rust versions unlike `DefaultHasher`.
fn stub_seed_value(value: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn validate_artifact_path(entry: &RegistryEntry, artifact_path: &Path) -> Result<(), ModelError> {
    if !artifact_path.is_file() {
        return Err(ModelError::InvalidArtifactPath {
            name: entry.info.name.clone(),
            path: artifact_path.display().to_string(),
            reason: "file does not exist".to_string(),
        });
    }

    if std::fs::metadata(artifact_path)?.len() == 0 {
        return Err(ModelError::InvalidArtifactPath {
            name: entry.info.name.clone(),
            path: artifact_path.display().to_string(),
            reason: "file is empty".to_string(),
        });
    }

    let missing_companions = checkpoint_companion_paths(entry, artifact_path)
        .into_iter()
        .filter(|path| !path.is_file())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();

    if missing_companions.is_empty() {
        Ok(())
    } else {
        Err(ModelError::InvalidArtifactPath {
            name: entry.info.name.clone(),
            path: artifact_path.display().to_string(),
            reason: format!(
                "missing required companion artifacts: {}",
                missing_companions.join(", ")
            ),
        })
    }
}

fn checkpoint_companion_paths(
    entry: &RegistryEntry,
    artifact_path: &Path,
) -> Vec<std::path::PathBuf> {
    let Ok(primary_artifact) = entry.primary_artifact() else {
        return Vec::new();
    };

    let primary_relative = Path::new(&primary_artifact.relative_path);
    let registry_root = primary_relative.parent();
    let artifact_dir = artifact_path.parent().unwrap_or_else(|| Path::new("."));

    entry
        .artifacts
        .iter()
        .skip(1)
        .map(|artifact| {
            let artifact_relative = Path::new(&artifact.relative_path);
            let suffix = registry_root
                .and_then(|root| artifact_relative.strip_prefix(root).ok())
                .unwrap_or(artifact_relative);
            artifact_dir.join(suffix)
        })
        .collect()
}

fn stub_seed(path: &Path) -> u64 {
    stub_seed_value(&path.to_string_lossy())
}

fn checkpoint_progress_value(path: &Path) -> Option<f32> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(checkpoint_progress_from_label)
}

fn seeded_stub_features(
    config: StubConfig,
    tensor: &Array4<f32>,
    contract: &registry::TensorContract,
    model_heads: usize,
) -> (Option<Array1<f32>>, Array2<f32>, Option<Array4<f32>>) {
    let seed = config.seed;
    let checkpoint_position = config.checkpoint_progress.unwrap_or(0.0);
    let checkpoint_sample_offset = checkpoint_position.max(0.0).round() as usize;
    let checkpoint_progress = config
        .checkpoint_progress
        .map(|progress| (progress.max(0.0) + 1.0).ln())
        .unwrap_or(0.0);
    let checkpoint_drift = [
        checkpoint_progress * 0.16,
        checkpoint_progress.powi(2) * 0.02,
        checkpoint_progress.sqrt() * 0.05,
    ];
    let (mean, stddev) = tensor_moments(tensor);
    let flat_tensor = tensor.iter().copied().collect::<Vec<_>>();
    let patch_components = (0..contract.patch_count)
        .map(|patch_idx| {
            let sample_a = sampled_tensor_value(
                &flat_tensor,
                seed ^ 0x1111_1111,
                patch_idx + checkpoint_sample_offset,
                checkpoint_sample_offset,
            );
            let sample_b = sampled_tensor_value(
                &flat_tensor,
                seed ^ 0x2222_2222,
                patch_idx,
                checkpoint_sample_offset + 1,
            );
            [
                sample_a + mean * 0.45,
                sample_b - stddev * 0.30,
                stub_wave(seed ^ 0x3333_3333, patch_idx),
                stub_wave(seed ^ 0x4444_4444, patch_idx * 3 + 1) * (0.35 + stddev.abs()),
            ]
        })
        .collect::<Vec<_>>();
    let dim_components = (0..contract.embedding_dim)
        .map(|dim_idx| {
            let sample_a = sampled_tensor_value(
                &flat_tensor,
                seed ^ 0x5555_5555,
                checkpoint_sample_offset,
                dim_idx + checkpoint_sample_offset,
            );
            let sample_b = sampled_tensor_value(
                &flat_tensor,
                seed ^ 0x6666_6666,
                checkpoint_sample_offset + 1,
                dim_idx,
            );
            [
                sample_a + mean * 0.35,
                sample_b + stddev * 0.20,
                stub_wave(seed ^ 0x7777_7777, dim_idx),
                stub_wave(seed ^ 0x8888_8888, dim_idx * 5 + 2) * (0.30 + mean.abs()),
            ]
        })
        .collect::<Vec<_>>();
    let patch_tokens = Array2::from_shape_fn(
        (contract.patch_count, contract.embedding_dim),
        |(patch_idx, dim_idx)| {
            let patch = patch_components[patch_idx];
            let dim = dim_components[dim_idx];
            let patch_axis = normalized_axis(patch_idx, contract.patch_count);
            let dim_axis = normalized_axis(dim_idx, contract.embedding_dim);
            let checkpoint_mix_a = 0.60 + checkpoint_drift[0] * patch_axis;
            let checkpoint_mix_b = 0.25 - checkpoint_drift[0] * dim_axis;
            let checkpoint_mix_c = 0.10 + checkpoint_drift[1] * zero_mean_quadratic(dim_axis);
            let checkpoint_mix_d = 0.05 + checkpoint_drift[1] * zero_mean_quadratic(patch_axis);
            let structured_checkpoint_drift = checkpoint_drift[0] * patch_axis * dim_axis
                + checkpoint_drift[1]
                    * zero_mean_quadratic(patch_axis)
                    * zero_mean_quadratic(dim_axis)
                + checkpoint_drift[2]
                    * stub_wave(seed ^ 0xABCD_EF01, patch_idx * 7 + dim_idx * 3 + 1)
                    * stub_wave(seed ^ 0x1020_3040, dim_idx * 11 + patch_idx + 3);
            let variant_jitter =
                checkpoint_variant_jitter(config.variant_seed, 0x0F0F_F0F0, patch_idx, dim_idx);
            patch[0] * dim[0] * checkpoint_mix_a
                + patch[1] * dim[1] * checkpoint_mix_b
                + patch[2] * dim[2] * checkpoint_mix_c
                + patch[3] * dim[3] * checkpoint_mix_d
                + mean * 0.12
                + stddev * 0.08
                + structured_checkpoint_drift
                + variant_jitter
        },
    );

    let cls_token = contract.cls_expected.then(|| {
        Array1::from_shape_fn(contract.embedding_dim, |dim_idx| {
            let dim = dim_components[dim_idx];
            let cls_source = sampled_tensor_value(
                &flat_tensor,
                seed ^ 0x9999_9999,
                checkpoint_sample_offset,
                dim_idx + checkpoint_sample_offset * 2,
            );
            let dim_axis = normalized_axis(dim_idx, contract.embedding_dim);
            let structured_checkpoint_drift = checkpoint_drift[0] * dim_axis
                + checkpoint_drift[1] * zero_mean_quadratic(dim_axis)
                + checkpoint_drift[2] * stub_wave(seed ^ 0x5566_7788, dim_idx * 13 + 5);
            cls_source * 0.50
                + dim[0] * (mean + 0.40)
                + dim[1] * 0.15
                + dim[2] * 0.05
                + stddev * 0.12
                + structured_checkpoint_drift
                + checkpoint_variant_jitter(config.variant_seed, 0xDEAD_BEEF, 0, dim_idx)
        })
    });

    let attention_weights = Some(seeded_stub_attention(
        config,
        &flat_tensor,
        contract,
        &patch_tokens,
        cls_token.as_ref(),
        mean,
        stddev,
        model_heads,
    ));

    (cls_token, patch_tokens, attention_weights)
}

#[allow(clippy::too_many_arguments)]
fn seeded_stub_attention(
    config: StubConfig,
    flat_tensor: &[f32],
    contract: &registry::TensorContract,
    patch_tokens: &Array2<f32>,
    cls_token: Option<&Array1<f32>>,
    mean: f32,
    stddev: f32,
    model_heads: usize,
) -> Array4<f32> {
    let has_cls = contract.cls_expected && cls_token.is_some();
    let patch_start = usize::from(has_cls);
    let token_count = contract.patch_count + patch_start;
    let layers = 2usize;
    let heads = model_heads.clamp(1, 4);
    let checkpoint_offset = config.checkpoint_progress.unwrap_or(0.0).max(0.0).round() as usize;
    let focus_x =
        (sampled_tensor_value(flat_tensor, config.seed ^ 0xA1A1_A1A1, checkpoint_offset, 0) + mean)
            .clamp(-1.0, 1.0)
            * 0.45;
    let focus_y = (sampled_tensor_value(
        flat_tensor,
        config.seed ^ 0xB2B2_B2B2,
        checkpoint_offset + 1,
        1,
    ) - stddev)
        .clamp(-1.0, 1.0)
        * 0.45;

    let mut token_signatures = Vec::with_capacity(token_count);
    if let Some(cls) = cls_token {
        let signature = cls.iter().copied().sum::<f32>() / cls.len().max(1) as f32;
        token_signatures.push(signature);
    }
    token_signatures.extend((0..contract.patch_count).map(|patch_idx| {
        let row = patch_tokens.row(patch_idx);
        row.iter().copied().sum::<f32>() / row.len().max(1) as f32
    }));

    let mut attention = Array4::<f32>::zeros((layers, heads, token_count, token_count));
    for layer_idx in 0..layers {
        for head_idx in 0..heads {
            let head_focus_x =
                focus_x + normalized_axis(head_idx, heads) * 0.18 + layer_idx as f32 * 0.04;
            let head_focus_y =
                focus_y - normalized_axis(layer_idx, layers) * 0.18 + head_idx as f32 * 0.02;
            for query_idx in 0..token_count {
                let (query_x, query_y) = if has_cls && query_idx == 0 {
                    (head_focus_x * 0.4, head_focus_y * 0.4)
                } else {
                    patch_coordinates(query_idx.saturating_sub(patch_start), contract.patch_count)
                };

                let mut row_sum = 0.0_f32;
                for key_idx in 0..token_count {
                    let mut score = 0.02
                        + 0.05
                            * stub_unit(
                                config.seed ^ 0xC3C3_C3C3,
                                layer_idx * token_count + query_idx,
                                head_idx * token_count + key_idx,
                            );

                    if has_cls && key_idx == 0 {
                        score += 0.10
                            + 0.02 * (layer_idx + head_idx) as f32
                            + token_signatures[query_idx].abs() * 0.02;
                    } else {
                        let patch_idx = key_idx.saturating_sub(patch_start);
                        let (key_x, key_y) = patch_coordinates(patch_idx, contract.patch_count);
                        let dx = query_x - key_x;
                        let dy = query_y - key_y;
                        let distance = dx * dx + dy * dy;
                        let locality = (-distance * 4.5).exp();
                        let focus_distance =
                            (key_x - head_focus_x).powi(2) + (key_y - head_focus_y).powi(2);
                        let cls_focus = (-focus_distance * 6.0).exp();
                        let compatibility = (token_signatures[query_idx]
                            * token_signatures[key_idx])
                            .abs()
                            .min(2.0);

                        score += 0.18 * locality
                            + 0.16 * compatibility
                            + 0.08
                                * sampled_tensor_value(
                                    flat_tensor,
                                    config.seed ^ 0xD4D4_D4D4,
                                    query_idx + checkpoint_offset,
                                    key_idx + layer_idx * 3 + head_idx,
                                )
                                .abs();

                        if has_cls && query_idx == 0 {
                            score += 0.35 * cls_focus;
                        }
                    }

                    attention[[layer_idx, head_idx, query_idx, key_idx]] = score.max(1e-6);
                    row_sum += attention[[layer_idx, head_idx, query_idx, key_idx]];
                }

                if row_sum > 0.0 {
                    for key_idx in 0..token_count {
                        attention[[layer_idx, head_idx, query_idx, key_idx]] /= row_sum;
                    }
                }
            }
        }
    }

    attention
}

fn checkpoint_progress_from_label(label: &str) -> Option<f32> {
    let mut value = 0_u64;
    let mut current = 0_u64;
    let mut in_digits = false;
    let mut found = false;

    for byte in label.bytes() {
        if byte.is_ascii_digit() {
            in_digits = true;
            current = current
                .saturating_mul(10)
                .saturating_add(u64::from(byte - b'0'));
        } else if in_digits {
            found = true;
            value = value
                .saturating_mul(1_000_000)
                .saturating_add(current.min(999_999));
            current = 0;
            in_digits = false;
        }
    }

    if in_digits {
        found = true;
        value = value
            .saturating_mul(1_000_000)
            .saturating_add(current.min(999_999));
    }

    found.then_some(value as f32)
}

fn normalized_axis(index: usize, len: usize) -> f32 {
    if len <= 1 {
        0.0
    } else {
        index as f32 / (len - 1) as f32 - 0.5
    }
}

fn patch_coordinates(index: usize, patch_count: usize) -> (f32, f32) {
    let grid = (patch_count as f32).sqrt().round() as usize;
    if grid > 0 && grid * grid == patch_count {
        let row = index / grid;
        let col = index % grid;
        (normalized_axis(col, grid), normalized_axis(row, grid))
    } else {
        (normalized_axis(index, patch_count), 0.0)
    }
}

fn zero_mean_quadratic(value: f32) -> f32 {
    value * value - (1.0 / 12.0)
}

fn checkpoint_variant_jitter(seed: u64, salt: u64, major: usize, minor: usize) -> f32 {
    if seed == 0 {
        0.0
    } else {
        (stub_unit(seed ^ salt, major, minor) - 0.5) * 0.01
    }
}

fn tensor_moments(tensor: &Array4<f32>) -> (f32, f32) {
    let count = tensor.len().max(1) as f32;
    let mean = tensor.iter().copied().sum::<f32>() / count;
    let variance = tensor
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / count;
    (mean, variance.sqrt())
}

fn sampled_tensor_value(values: &[f32], seed: u64, major: usize, minor: usize) -> f32 {
    let len = values.len().max(1);
    let index = ((seed as usize)
        .wrapping_add((major + 1).wrapping_mul(131))
        .wrapping_add((minor + 1).wrapping_mul(17)))
        % len;
    values[index]
}

fn stub_wave(seed: u64, index: usize) -> f32 {
    (stub_unit(seed, index, 0) - 0.5) * 2.0
}

fn stub_unit(seed: u64, major: usize, minor: usize) -> f32 {
    let mixed = seed
        .wrapping_add((major as u64 + 1).wrapping_mul(0x9E37_79B1_85EB_CA87))
        .wrapping_add((minor as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    (mixed % 10_000) as f32 / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::linear_cka;
    use crate::extract::ExtractedFeatures;
    use crate::models::registry::{Checksum, ModelArtifact};
    use crate::TEST_PROCESS_ENV_LOCK;
    use std::ffi::OsString;
    use std::fs;
    use tempfile::tempdir;

    impl ModelSession {
        fn stubbed(entry: RegistryEntry) -> Self {
            Self {
                inner: SessionInner::Stub(StubConfig::standard(&entry.info.name)),
                entry,
            }
        }
    }

    struct ModelBackendEnvGuard {
        previous: Option<OsString>,
    }

    impl ModelBackendEnvGuard {
        fn set(value: &str) -> Self {
            let previous = std::env::var_os("LATENT_INSPECTOR_MODEL_BACKEND");
            std::env::set_var("LATENT_INSPECTOR_MODEL_BACKEND", value);
            Self { previous }
        }
    }

    impl Drop for ModelBackendEnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("LATENT_INSPECTOR_MODEL_BACKEND", value),
                None => std::env::remove_var("LATENT_INSPECTOR_MODEL_BACKEND"),
            }
        }
    }

    #[test]
    fn test_patch_only_output_records_observed_metadata() {
        let entry = registry::find("vjepa2-vitl-fpc2-256").unwrap();
        let session = ModelSession::stubbed(entry.clone());
        let hidden = ndarray::Array3::<f32>::zeros((
            entry.validation.tensor.batch_size,
            entry.validation.tensor.patch_count,
            entry.validation.tensor.embedding_dim,
        ));

        let output = ModelSession::output_from_hidden(
            &session.entry,
            vec![1, 3, 256, 256],
            hidden.shape().to_vec(),
            hidden.view(),
            SequenceOutputLayout::PatchOnly,
        )
        .unwrap();

        assert!(output.cls_token.is_none());
        assert!(!output.tensor_metadata.sequence_has_cls);
        assert_eq!(
            output.tensor_metadata.observed_patch_count,
            entry.validation.tensor.patch_count
        );
        assert_eq!(
            output.tensor_metadata.embedding_dim,
            entry.validation.tensor.embedding_dim
        );
    }

    #[test]
    fn test_patch_only_output_rejects_unexpected_sequence_length() {
        let entry = registry::find("vjepa2-vitl-fpc2-256").unwrap();
        let session = ModelSession::stubbed(entry.clone());
        let hidden = ndarray::Array3::<f32>::zeros((
            entry.validation.tensor.batch_size,
            entry.validation.tensor.patch_count + 1,
            entry.validation.tensor.embedding_dim,
        ));

        let error = ModelSession::output_from_hidden(
            &session.entry,
            vec![1, 3, 256, 256],
            hidden.shape().to_vec(),
            hidden.view(),
            SequenceOutputLayout::PatchOnly,
        )
        .unwrap_err();

        assert!(matches!(error, ModelError::InferenceFailed(_)));
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
        assert!(output.attention_weights.is_some());
        assert_eq!(output.cls_token.unwrap().len(), embed_dim);
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 257, 1024]);
        assert_eq!(output.attention_weights.unwrap().shape(), &[2, 4, 257, 257]);
    }

    #[test]
    fn test_stub_mae_semantics_skip_cls() {
        let entry = registry::find("mae-vit-l16").unwrap();
        let mut session = ModelSession::stubbed(entry);
        let img = image::DynamicImage::new_rgb8(224, 224);
        let output = session.infer(&img).unwrap();

        assert!(output.cls_token.is_none());
        assert!(output.attention_weights.is_some());
        assert_eq!(output.tensor_metadata.output_shape, vec![1, 196, 1024]);
    }

    #[test]
    fn test_load_rejects_planned_models() {
        let result = ModelSession::load("clip-vit-l14");
        assert!(matches!(result, Err(ModelError::Unavailable { .. })));
    }

    #[test]
    fn test_load_for_analysis_allows_planned_models_in_stub_mode() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _guard = ModelBackendEnvGuard::set("stub");

        let mut session = ModelSession::load_for_analysis("clip-vit-l14").unwrap();
        let output = session
            .infer(&image::DynamicImage::new_rgb8(224, 224))
            .unwrap();

        assert_eq!(session.info().name, "clip-vit-l14");
        assert_eq!(output.patch_tokens.shape(), &[256, 1024]);
        assert!(output.cls_token.is_some());
    }

    #[test]
    fn test_checkpoint_stub_sessions_vary_by_path() {
        let dir = tempdir().unwrap();
        let first_path = dir.path().join("step-0001.onnx");
        let second_path = dir.path().join("step-0002.onnx");
        fs::write(&first_path, b"stub checkpoint").unwrap();
        fs::write(&second_path, b"stub checkpoint").unwrap();

        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _guard = ModelBackendEnvGuard::set("stub");

        let mut first = ModelSession::load_checkpoint("dinov2-vit-l14", &first_path).unwrap();
        let mut second = ModelSession::load_checkpoint("dinov2-vit-l14", &second_path).unwrap();
        let img = image::DynamicImage::new_rgb8(224, 224);

        let first_output = first.infer(&img).unwrap();
        let second_output = second.infer(&img).unwrap();

        assert_ne!(
            first_output.patch_tokens[[0, 0]],
            second_output.patch_tokens[[0, 0]]
        );
    }

    #[test]
    fn test_checkpoint_progress_from_label_uses_numeric_segments() {
        assert_eq!(checkpoint_progress_from_label("step-10"), Some(10.0));
        assert_eq!(
            checkpoint_progress_from_label("epoch-2-step-10"),
            Some(2_000_010.0)
        );
        assert_eq!(checkpoint_progress_from_label("checkpoint-final"), None);
    }

    #[test]
    fn test_checkpoint_stub_sessions_reflect_numeric_progression() {
        let dir = tempdir().unwrap();
        let first_path = dir.path().join("step-1.onnx");
        let second_path = dir.path().join("step-2.onnx");
        let third_path = dir.path().join("step-10.onnx");
        fs::write(&first_path, b"stub checkpoint").unwrap();
        fs::write(&second_path, b"stub checkpoint").unwrap();
        fs::write(&third_path, b"stub checkpoint").unwrap();

        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _guard = ModelBackendEnvGuard::set("stub");

        let mut first = ModelSession::load_checkpoint("dinov2-vit-l14", &first_path).unwrap();
        let mut second = ModelSession::load_checkpoint("dinov2-vit-l14", &second_path).unwrap();
        let mut third = ModelSession::load_checkpoint("dinov2-vit-l14", &third_path).unwrap();
        let images = [
            image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                224,
                224,
                image::Rgb([8, 32, 64]),
            )),
            image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                224,
                224,
                image::Rgb([220, 120, 40]),
            )),
        ];

        let first_matrix = checkpoint_embedding_matrix(&mut first, &images);
        let second_matrix = checkpoint_embedding_matrix(&mut second, &images);
        let third_matrix = checkpoint_embedding_matrix(&mut third, &images);

        let cka_12 = linear_cka(&first_matrix, &second_matrix).unwrap();
        let cka_2_10 = linear_cka(&second_matrix, &third_matrix).unwrap();

        assert!(
            cka_12 > cka_2_10,
            "expected later checkpoint gap to drift more: step-1->2={cka_12:.4}, step-2->10={cka_2_10:.4}"
        );
    }

    #[test]
    fn test_standard_stub_sessions_vary_by_input_content() {
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut session = ModelSession::stubbed(entry);

        let dark = image::DynamicImage::new_rgb8(224, 224);
        let bright = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            224,
            224,
            image::Rgb([255, 255, 255]),
        ));

        let dark_output = session.infer(&dark).unwrap();
        let bright_output = session.infer(&bright).unwrap();

        assert_ne!(
            dark_output.patch_tokens[[0, 0]],
            bright_output.patch_tokens[[0, 0]]
        );
    }

    #[test]
    fn test_load_checkpoint_requires_existing_artifact() {
        let missing = Path::new("/definitely/missing/checkpoint.onnx");
        let result = ModelSession::load_checkpoint("dinov2-vit-l14", missing);

        assert!(matches!(
            result,
            Err(ModelError::InvalidArtifactPath { .. })
        ));
    }

    #[test]
    fn test_validate_artifact_path_rejects_empty_checkpoint_files() {
        let dir = tempdir().unwrap();
        let artifact_path = dir.path().join("model.onnx");
        fs::write(&artifact_path, []).unwrap();
        let entry = registry::find("dinov2-vit-l14").unwrap();

        let result = validate_artifact_path(&entry, &artifact_path);

        assert!(matches!(
            result,
            Err(ModelError::InvalidArtifactPath { reason, .. }) if reason == "file is empty"
        ));
    }

    #[test]
    fn test_validate_artifact_path_requires_companion_artifacts() {
        let dir = tempdir().unwrap();
        let artifact_path = dir.path().join("model.onnx");
        fs::write(&artifact_path, b"onnx").unwrap();

        let mut entry = registry::find("dinov2-vit-l14").unwrap();
        entry.artifacts = vec![
            ModelArtifact {
                relative_path: "bundle/model.onnx".to_string(),
                download_url: "https://example.invalid/model.onnx".to_string(),
                checksum: Checksum::Pending {
                    reason: "test fixture".to_string(),
                },
            },
            ModelArtifact {
                relative_path: "bundle/model.onnx_data".to_string(),
                download_url: "https://example.invalid/model.onnx_data".to_string(),
                checksum: Checksum::Pending {
                    reason: "test fixture".to_string(),
                },
            },
        ];

        let result = validate_artifact_path(&entry, &artifact_path);

        assert!(matches!(
            result,
            Err(ModelError::InvalidArtifactPath { reason, .. })
                if reason.contains("model.onnx_data")
        ));
    }

    #[test]
    fn test_validate_artifact_path_accepts_complete_companion_bundle() {
        let dir = tempdir().unwrap();
        let artifact_path = dir.path().join("model.onnx");
        let companion_path = dir.path().join("model.onnx_data");
        fs::write(&artifact_path, b"onnx").unwrap();
        fs::write(&companion_path, b"external-data").unwrap();

        let mut entry = registry::find("dinov2-vit-l14").unwrap();
        entry.artifacts = vec![
            ModelArtifact {
                relative_path: "bundle/model.onnx".to_string(),
                download_url: "https://example.invalid/model.onnx".to_string(),
                checksum: Checksum::Pending {
                    reason: "test fixture".to_string(),
                },
            },
            ModelArtifact {
                relative_path: "bundle/model.onnx_data".to_string(),
                download_url: "https://example.invalid/model.onnx_data".to_string(),
                checksum: Checksum::Pending {
                    reason: "test fixture".to_string(),
                },
            },
        ];

        validate_artifact_path(&entry, &artifact_path).unwrap();
    }

    fn checkpoint_embedding_matrix(
        session: &mut ModelSession,
        images: &[image::DynamicImage],
    ) -> Array2<f32> {
        let mut rows = Vec::new();
        for image in images {
            let output = session.infer(image).unwrap();
            let features = ExtractedFeatures::from_output(output).unwrap();
            rows.push(features.mean_patch());
        }

        let dim = rows.first().map(|row| row.len()).unwrap_or(0);
        let mut matrix = Array2::<f32>::zeros((rows.len(), dim));
        for (index, row) in rows.iter().enumerate() {
            matrix.row_mut(index).assign(row);
        }
        matrix
    }
}
