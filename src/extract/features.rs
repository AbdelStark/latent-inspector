use crate::analysis::finite::{
    ensure_finite_1d, ensure_finite_2d, ensure_finite_4d, square_grid_side,
};
use crate::errors::AnalysisError;
use crate::models::ModelOutput;
use ndarray::{Array1, Array2, Array4, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingBasis {
    ClsToken,
    MeanPatch,
}

impl EmbeddingBasis {
    pub fn label(self) -> &'static str {
        match self {
            EmbeddingBasis::ClsToken => "CLS token",
            EmbeddingBasis::MeanPatch => "Mean patch",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            EmbeddingBasis::ClsToken => {
                "Global image embedding taken directly from the model CLS token."
            }
            EmbeddingBasis::MeanPatch => {
                "Global image embedding built by averaging patch tokens because no CLS token is available."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionMapBasis {
    ClsToPatch,
    MeanTokenToPatch,
}

impl AttentionMapBasis {
    pub fn label(self) -> &'static str {
        match self {
            AttentionMapBasis::ClsToPatch => "CLS-to-patch attention",
            AttentionMapBasis::MeanTokenToPatch => "Mean token-to-patch attention",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            AttentionMapBasis::ClsToPatch => {
                "Average attention from the CLS token into the patch grid."
            }
            AttentionMapBasis::MeanTokenToPatch => {
                "Average attention from all tokens into the patch grid."
            }
        }
    }
}

/// Summary of extracted features for one model/image pair.
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// CLS token `[D]`.
    pub cls_token: Option<Array1<f32>>,
    /// Patch token matrix `[N_patches, D]`.
    pub patch_tokens: Array2<f32>,
    /// Attention weights `[layers, heads, N, N]` when exported by the model.
    pub attention_weights: Option<Array4<f32>>,
    /// Whether the token sequence includes a CLS token prefix.
    pub sequence_has_cls: bool,
    /// L2 norm of the CLS token (if present).
    pub cls_norm: Option<f32>,
    /// L2 norms of each patch token `[N_patches]`.
    pub patch_norms: Array1<f32>,
    /// Number of patches.
    pub n_patches: usize,
    /// Feature dimension.
    pub embed_dim: usize,
}

impl ExtractedFeatures {
    /// Build from a raw `ModelOutput`.
    pub fn from_output(output: ModelOutput) -> Result<Self, AnalysisError> {
        let shape = output.patch_tokens.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(AnalysisError::EmptyInput(
                "patch_tokens has zero-size dimension".into(),
            ));
        }

        ensure_finite_2d(&output.patch_tokens, "patch tokens")?;
        if let Some(cls_token) = output.cls_token.as_ref() {
            ensure_finite_1d(cls_token, "CLS token")?;
        }
        if let Some(attention_weights) = output.attention_weights.as_ref() {
            ensure_finite_4d(attention_weights, "attention weights")?;
        }

        let n_patches = shape[0];
        let embed_dim = shape[1];

        let cls_norm = output.cls_token.as_ref().map(l2_norm);
        let patch_norms = output
            .patch_tokens
            .rows()
            .into_iter()
            .map(|row| l2_norm(&row.to_owned()))
            .collect::<Array1<f32>>();

        Ok(Self {
            cls_token: output.cls_token,
            patch_tokens: output.patch_tokens,
            attention_weights: output.attention_weights,
            sequence_has_cls: output.tensor_metadata.sequence_has_cls,
            cls_norm,
            patch_norms,
            n_patches,
            embed_dim,
        })
    }

    /// Returns a row-normalized version of the patch tokens (each patch divided by its L2 norm).
    pub fn normalized_patch_tokens(&self) -> Array2<f32> {
        let mut out = self.patch_tokens.clone();
        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            let norm = self.patch_norms[i].max(1e-8);
            row.mapv_inplace(|v| v / norm);
        }
        out
    }

    /// Mean of patch tokens `[D]`.
    ///
    /// Safe: `from_output` validates patch_tokens has at least one row.
    pub fn mean_patch(&self) -> Array1<f32> {
        self.patch_tokens
            .mean_axis(Axis(0))
            .expect("ExtractedFeatures: patch_tokens must be non-empty")
    }

    /// Global embedding for the requested basis, when available.
    pub fn embedding_for_basis(&self, basis: EmbeddingBasis) -> Option<Array1<f32>> {
        match basis {
            EmbeddingBasis::ClsToken => self.cls_token.clone(),
            EmbeddingBasis::MeanPatch => Some(self.mean_patch()),
        }
    }

    /// Preferred global embedding for image-level comparisons.
    pub fn preferred_global_embedding(&self) -> (EmbeddingBasis, Array1<f32>) {
        if let Some(cls) = self.cls_token.clone() {
            (EmbeddingBasis::ClsToken, cls)
        } else {
            (EmbeddingBasis::MeanPatch, self.mean_patch())
        }
    }

    pub fn attention_dimensions(&self) -> Option<(usize, usize, usize)> {
        let weights = self.attention_weights.as_ref()?;
        let shape = weights.shape();
        if shape.len() != 4 || shape[2] != shape[3] {
            return None;
        }
        Some((shape[0], shape[1], shape[2]))
    }

    pub fn attention_map(&self) -> Option<(AttentionMapBasis, Array2<f32>)> {
        let weights = self.attention_weights.as_ref()?;
        let grid_size = attention_grid_size(self.n_patches)?;
        let shape = weights.shape();
        if shape.len() != 4 || shape[2] != shape[3] {
            return None;
        }

        let has_cls = self.sequence_has_cls
            && self.cls_token.is_some()
            && shape[2] == self.n_patches.saturating_add(1);
        let patch_start = usize::from(has_cls);
        let token_count = shape[2];
        if token_count < patch_start + self.n_patches {
            return None;
        }

        let basis = if has_cls {
            AttentionMapBasis::ClsToPatch
        } else {
            AttentionMapBasis::MeanTokenToPatch
        };
        let normalizer = if has_cls {
            (shape[0] * shape[1]).max(1) as f32
        } else {
            (shape[0] * shape[1] * token_count).max(1) as f32
        };

        let values = (0..self.n_patches)
            .map(|patch_idx| {
                let key_idx = patch_start + patch_idx;
                let mut total = 0.0_f32;
                for layer_idx in 0..shape[0] {
                    for head_idx in 0..shape[1] {
                        if has_cls {
                            total += weights[[layer_idx, head_idx, 0, key_idx]];
                        } else {
                            for query_idx in 0..token_count {
                                total += weights[[layer_idx, head_idx, query_idx, key_idx]];
                            }
                        }
                    }
                }
                total / normalizer
            })
            .collect::<Vec<_>>();

        Array2::from_shape_vec((grid_size, grid_size), values)
            .ok()
            .map(|map| (basis, map))
    }
}

fn l2_norm(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn attention_grid_size(n_patches: usize) -> Option<usize> {
    square_grid_side(n_patches, "attention map").ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dummy_output(n: usize, d: usize) -> ModelOutput {
        use crate::models::{ModelInfo, OutputTensorMetadata, SSLMethod};
        ModelOutput {
            cls_token: Some(Array1::from_elem(d, 1.0_f32)),
            patch_tokens: Array2::from_elem((n, d), 0.5_f32),
            attention_weights: None,
            model_info: ModelInfo {
                name: "test".into(),
                architecture: "ViT".into(),
                patch_size: 14,
                embed_dim: d as u32,
                num_layers: 12,
                num_heads: 8,
                method: SSLMethod::DINO,
                input_size: 224,
                params_m: 100,
            },
            tensor_metadata: OutputTensorMetadata {
                input_name: "pixel_values".into(),
                input_shape: vec![1, 3, 224, 224],
                output_name: "last_hidden_state".into(),
                output_shape: vec![1, n + 1, d],
                sequence_has_cls: true,
                observed_patch_count: n,
                embedding_dim: d,
            },
        }
    }

    #[test]
    fn test_extracted_shapes() {
        let out = dummy_output(256, 1024);
        let feat = ExtractedFeatures::from_output(out).unwrap();
        assert_eq!(feat.n_patches, 256);
        assert_eq!(feat.embed_dim, 1024);
        assert_eq!(feat.patch_norms.len(), 256);
    }

    #[test]
    fn test_cls_norm() {
        // CLS = [1.0; D], norm = sqrt(D)
        let d = 4;
        let out = dummy_output(16, d);
        let feat = ExtractedFeatures::from_output(out).unwrap();
        let expected = (d as f32).sqrt();
        approx::assert_relative_eq!(feat.cls_norm.unwrap(), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_patch() {
        let out = dummy_output(16, 4);
        let feat = ExtractedFeatures::from_output(out).unwrap();
        let mean = feat.mean_patch();
        // All values are 0.5
        for v in mean.iter() {
            approx::assert_relative_eq!(*v, 0.5, epsilon = 1e-5);
        }
    }

    #[test]
    fn preferred_embedding_uses_cls_when_available() {
        let feat = ExtractedFeatures::from_output(dummy_output(16, 4)).unwrap();
        let (basis, embedding) = feat.preferred_global_embedding();

        assert_eq!(basis, EmbeddingBasis::ClsToken);
        assert_eq!(embedding.len(), 4);
    }

    #[test]
    fn preferred_embedding_falls_back_to_mean_patch_without_cls() {
        let mut output = dummy_output(16, 4);
        output.cls_token = None;
        output.tensor_metadata.sequence_has_cls = false;
        output.tensor_metadata.output_shape = vec![1, 16, 4];
        let feat = ExtractedFeatures::from_output(output).unwrap();
        let (basis, embedding) = feat.preferred_global_embedding();

        assert_eq!(basis, EmbeddingBasis::MeanPatch);
        assert_eq!(embedding, feat.mean_patch());
    }

    #[test]
    fn attention_map_uses_cls_attention_when_present() {
        let mut output = dummy_output(4, 4);
        output.attention_weights = Some(
            Array4::from_shape_vec(
                (1, 1, 5, 5),
                vec![
                    0.1, 0.4, 0.3, 0.1, 0.1, //
                    0.2, 0.2, 0.2, 0.2, 0.2, //
                    0.2, 0.2, 0.2, 0.2, 0.2, //
                    0.2, 0.2, 0.2, 0.2, 0.2, //
                    0.2, 0.2, 0.2, 0.2, 0.2, //
                ],
            )
            .unwrap(),
        );
        let feat = ExtractedFeatures::from_output(output).unwrap();

        let (basis, map) = feat.attention_map().unwrap();

        assert_eq!(basis, AttentionMapBasis::ClsToPatch);
        assert_eq!(map.shape(), &[2, 2]);
        approx::assert_relative_eq!(map[[0, 0]], 0.4, epsilon = 1e-5);
        approx::assert_relative_eq!(map[[0, 1]], 0.3, epsilon = 1e-5);
    }

    #[test]
    fn attention_map_falls_back_to_mean_token_attention_without_cls() {
        let mut output = dummy_output(4, 4);
        output.cls_token = None;
        output.tensor_metadata.sequence_has_cls = false;
        output.tensor_metadata.output_shape = vec![1, 4, 4];
        output.attention_weights = Some(
            Array4::from_shape_vec(
                (1, 1, 4, 4),
                vec![
                    0.1, 0.4, 0.3, 0.2, //
                    0.1, 0.4, 0.3, 0.2, //
                    0.1, 0.4, 0.3, 0.2, //
                    0.1, 0.4, 0.3, 0.2, //
                ],
            )
            .unwrap(),
        );
        let feat = ExtractedFeatures::from_output(output).unwrap();

        let (basis, map) = feat.attention_map().unwrap();

        assert_eq!(basis, AttentionMapBasis::MeanTokenToPatch);
        assert_eq!(map.shape(), &[2, 2]);
        approx::assert_relative_eq!(map[[0, 0]], 0.1, epsilon = 1e-5);
        approx::assert_relative_eq!(map[[1, 1]], 0.2, epsilon = 1e-5);
    }

    #[test]
    fn extracted_features_reject_non_finite_patch_tokens() {
        let mut output = dummy_output(4, 4);
        output.patch_tokens[[1, 2]] = f32::NAN;

        let error = ExtractedFeatures::from_output(output).unwrap_err();

        assert!(matches!(error, AnalysisError::NonFiniteValues { .. }));
        assert!(error.to_string().contains("patch tokens"));
    }

    #[test]
    fn extracted_features_reject_non_finite_attention_weights() {
        let mut output = dummy_output(4, 4);
        let mut attention = Array4::from_elem((1, 1, 5, 5), 0.2_f32);
        attention[[0, 0, 0, 3]] = f32::INFINITY;
        output.attention_weights = Some(attention);

        let error = ExtractedFeatures::from_output(output).unwrap_err();

        assert!(matches!(error, AnalysisError::NonFiniteValues { .. }));
        assert!(error.to_string().contains("attention weights"));
    }
}
