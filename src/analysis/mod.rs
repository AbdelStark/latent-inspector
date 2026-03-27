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
    pub mean_patch_correspondence: f32,
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
        (Some(ca), Some(cb)) => Some(cls_cosine_similarity(ca, cb)),
        _ => None,
    };

    let cka = linear_cka(&pa, &pb)?;
    let overlap = knn_overlap(&pa, &pb, 10)?;
    let corr = patch_correspondence(&pa, &pb)?;

    Ok(ComparisonMetrics {
        model_a: name_a.to_string(),
        model_b: name_b.to_string(),
        cls_cosine_sim: cls_sim,
        linear_cka: cka,
        knn_overlap_k10: overlap,
        mean_patch_correspondence: corr.mean_similarity,
    })
}
