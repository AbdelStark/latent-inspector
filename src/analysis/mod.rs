pub mod attention;
pub mod cka;
pub mod correspondence;
pub mod entropy;
pub mod knn;
pub mod pca;
pub mod rank;
pub mod variance;

pub use attention::{gini, mean_gini, per_head_gini};
pub use cka::{cls_cosine_similarity, linear_cka};
pub use correspondence::{patch_correspondence, patch_cosine_similarity, CorrespondenceResult};
pub use entropy::{patch_entropy, patch_norm_stats, shannon_entropy, NormStats};
pub use knn::{cosine_similarity_matrix, knn_overlap, top_k_neighbors};
pub use pca::{pca, transform, PcaResult};
pub use rank::{dead_dimensions, effective_rank};
pub use variance::{variance_spectrum, VarianceSpectrum};

use crate::errors::AnalysisError;
use crate::extract::ExtractedFeatures;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Full set of per-model analysis metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub n_patches: usize,
    pub embed_dim: usize,
    pub effective_rank: usize,
    pub dead_dimensions: usize,
    pub patch_entropy: f32,
    pub cls_l2_norm: Option<f32>,
    pub patch_norm_mean: f32,
    pub patch_norm_std: f32,
    pub top10_variance_pct: f32,
    pub components_90pct: usize,
}

/// Compute all per-model metrics for the given features.
pub fn compute_metrics(
    features: &ExtractedFeatures,
    model_name: &str,
) -> Result<ModelMetrics, AnalysisError> {
    let rank = effective_rank(&features.patch_tokens, 0.01, 64)?;
    let dead = dead_dimensions(&features.patch_tokens, 1e-6);
    let entropy = patch_entropy(&features.patch_tokens, 8, 30)?;
    let norm_stats = patch_norm_stats(&features.patch_tokens);
    let spec = variance_spectrum(&features.patch_tokens, 16)?;

    Ok(ModelMetrics {
        model_name: model_name.to_string(),
        n_patches: features.n_patches,
        embed_dim: features.embed_dim,
        effective_rank: rank,
        dead_dimensions: dead,
        patch_entropy: entropy,
        cls_l2_norm: features.cls_norm,
        patch_norm_mean: norm_stats.mean,
        patch_norm_std: norm_stats.std,
        top10_variance_pct: spec.top10_concentration * 100.0,
        components_90pct: spec.components_90pct,
    })
}

/// Cross-model comparison metrics between two models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub model_a: String,
    pub model_b: String,
    pub cls_cosine_sim: Option<f32>,
    pub linear_cka: f32,
    pub knn_overlap_k10: f32,
    pub mean_patch_correspondence: Option<f32>,
}

/// Compute cross-model comparison metrics.
pub fn compute_comparison(
    a: &ExtractedFeatures,
    b: &ExtractedFeatures,
    name_a: &str,
    name_b: &str,
) -> Result<ComparisonMetrics, AnalysisError> {
    // Align patch counts (use the minimum)
    let n = a.n_patches.min(b.n_patches);
    let pa: Array2<f32> = a.patch_tokens.slice(ndarray::s![..n, ..]).to_owned();
    let pb: Array2<f32> = b.patch_tokens.slice(ndarray::s![..n, ..]).to_owned();

    let cls_sim = match (&a.cls_token, &b.cls_token) {
        (Some(ca), Some(cb)) if ca.len() == cb.len() => Some(cls_cosine_similarity(ca, cb)),
        _ => None,
    };

    let cka = linear_cka(&pa, &pb)?;
    let overlap = knn_overlap(&pa, &pb, 10)?;
    let mean_patch_correspondence = if pa.shape()[1] == pb.shape()[1] {
        Some(patch_correspondence(&pa, &pb)?.mean_similarity)
    } else {
        warn!(
            model_a = name_a,
            model_b = name_b,
            embed_dim_a = pa.shape()[1],
            embed_dim_b = pb.shape()[1],
            "Skipping direct-space comparison metrics for mismatched embedding dimensions"
        );
        None
    };

    Ok(ComparisonMetrics {
        model_a: name_a.to_string(),
        model_b: name_b.to_string(),
        cls_cosine_sim: cls_sim,
        linear_cka: cka,
        knn_overlap_k10: overlap,
        mean_patch_correspondence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::{ModelInfo, SSLMethod};
    use crate::models::ModelOutput;
    use ndarray::{Array1, Array2};

    fn features(name: &str, n_patches: usize, embed_dim: usize) -> ExtractedFeatures {
        ExtractedFeatures::from_output(ModelOutput {
            cls_token: Some(Array1::from_elem(embed_dim, 1.0_f32)),
            patch_tokens: Array2::from_shape_fn((n_patches, embed_dim), |(i, j)| {
                (i * embed_dim + j) as f32
            }),
            attention_weights: None,
            model_info: ModelInfo {
                name: name.to_string(),
                architecture: "ViT".to_string(),
                patch_size: 14,
                embed_dim: embed_dim as u32,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::DINO,
                input_size: 224,
                params_m: 300,
            },
        })
        .unwrap()
    }

    #[test]
    fn comparison_keeps_dimension_agnostic_metrics_for_mixed_widths() {
        let a = features("dinov2-vit-l14", 256, 1024);
        let b = features("ijepa-vit-h14", 256, 1280);

        let comparison = compute_comparison(&a, &b, "dinov2-vit-l14", "ijepa-vit-h14").unwrap();

        assert!(comparison.linear_cka.is_finite());
        assert!(comparison.knn_overlap_k10.is_finite());
        assert_eq!(comparison.cls_cosine_sim, None);
        assert_eq!(comparison.mean_patch_correspondence, None);
    }

    #[test]
    fn comparison_preserves_direct_metrics_for_matching_widths() {
        let a = features("dinov2-vit-l14", 256, 1024);
        let b = features("clip-vit-l14", 256, 1024);

        let comparison = compute_comparison(&a, &b, "dinov2-vit-l14", "clip-vit-l14").unwrap();

        assert!(comparison.cls_cosine_sim.is_some());
        assert!(comparison.mean_patch_correspondence.is_some());
    }
}
