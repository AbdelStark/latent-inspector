use crate::errors::AnalysisError;
use crate::models::ModelOutput;
use ndarray::{Array1, Array2, Axis};

/// Summary of extracted features for one model/image pair.
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// CLS token `[D]`.
    pub cls_token: Option<Array1<f32>>,
    /// Patch token matrix `[N_patches, D]`.
    pub patch_tokens: Array2<f32>,
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

        let n_patches = shape[0];
        let embed_dim = shape[1];

        let cls_norm = output.cls_token.as_ref().map(l2_norm);
        let patch_norms = output
            .patch_tokens
            .rows()
            .into_iter()
            .map(|row| l2_norm_view(&row))
            .collect::<Array1<f32>>();

        Ok(Self {
            cls_token: output.cls_token,
            patch_tokens: output.patch_tokens,
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
    pub fn mean_patch(&self) -> Array1<f32> {
        self.patch_tokens.mean_axis(Axis(0)).unwrap()
    }
}

fn l2_norm(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_norm_view(v: &ndarray::ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
}
