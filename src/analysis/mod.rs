pub mod attention;
pub mod cka;
pub mod coherence;
pub mod correspondence;
pub mod entropy;
pub(crate) mod finite;
pub mod intrinsic_dim;
pub mod isotropy;
pub mod knn;
pub mod pca;
pub mod rank;
pub mod rankme;
pub mod spectral_decay;
pub mod uniformity;
pub mod variance;

pub use attention::{gini, mean_gini, per_head_gini};
pub use cka::{cls_cosine_similarity, linear_cka};
pub use coherence::{spatial_coherence, spatial_coherence_map};
pub use correspondence::{patch_correspondence, patch_cosine_similarity, CorrespondenceResult};
pub use entropy::{patch_entropy, patch_norm_stats, shannon_entropy, NormStats};
pub use finite::square_grid_side;
pub use intrinsic_dim::{intrinsic_dimensionality, intrinsic_dimensionality_default};
pub use isotropy::{isotropy_score, partition_isotropy};
pub use knn::{cosine_similarity_matrix, knn_overlap, top_k_neighbors};
pub use pca::{pca, transform, transform_top_k, PcaResult};
pub use rank::{dead_dimensions, effective_rank};
pub use rankme::{rankme, rankme_from_spectrum};
pub use spectral_decay::spectral_decay_from_spectrum;
pub use uniformity::uniformity;
pub use variance::{variance_spectrum, variance_spectrum_from_pca_result, VarianceSpectrum};

use crate::errors::AnalysisError;
use crate::extract::ExtractedFeatures;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Maximum number of PCA components computed for full analysis pipelines.
pub const MAX_PCA_COMPONENTS: usize = 64;

/// Number of PCA components computed for the interactive TUI (lighter weight).
pub const TUI_PCA_COMPONENTS: usize = 32;

/// Full set of per-model analysis metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub n_patches: usize,
    pub embed_dim: usize,
    pub effective_rank: usize,
    pub dead_dimensions: usize,
    pub patch_entropy: f32,
    pub attention_gini: Option<f32>,
    pub cls_l2_norm: Option<f32>,
    pub patch_norm_mean: f32,
    pub patch_norm_std: f32,
    pub top10_variance_pct: f32,
    pub components_90pct: usize,
    /// Isotropy of the patch embedding space (0 = collapsed, 1 = uniform).
    pub patch_isotropy: f32,
    /// Uniformity of the patch embeddings on the unit hypersphere (more negative = better spread).
    pub patch_uniformity: f32,
    /// Spatial coherence: mean cosine similarity between adjacent patches on the grid.
    /// High values (~1) indicate smooth, segmentation-like representations (e.g. DINOv2).
    /// Low values (~0) indicate spatially differentiated representations (e.g. I-JEPA).
    /// `None` when the patch count is not a perfect square grid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial_coherence: Option<f32>,
    /// RankMe: smooth effective rank via Shannon entropy of normalized singular values.
    /// exp(H(p)) where p is the normalized singular value distribution.
    /// Range [1, k] — higher means richer, more spread-out representations.
    /// Reference: Garrido et al., ICML 2023.
    pub rankme: f32,
    /// Spectral decay exponent (beta): power-law fit to eigenvalue decay.
    /// Higher beta = faster decay = more concentrated representation.
    /// `None` when fewer than 2 positive eigenvalues exist.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_decay: Option<f32>,
}

/// Compute all per-model metrics for the given features.
pub fn compute_metrics(
    features: &ExtractedFeatures,
    model_name: &str,
) -> Result<ModelMetrics, AnalysisError> {
    let spec = variance_spectrum(&features.patch_tokens, MAX_PCA_COMPONENTS)?;
    model_metrics_from_spectrum(features, model_name, &spec)
}

/// Build per-model metrics while reusing an already computed variance spectrum.
pub fn model_metrics_from_spectrum(
    features: &ExtractedFeatures,
    model_name: &str,
    spec: &VarianceSpectrum,
) -> Result<ModelMetrics, AnalysisError> {
    let max_ev = spec
        .explained_variance
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let rank = if max_ev <= 0.0 {
        0
    } else {
        let threshold = 0.01 * max_ev;
        spec.explained_variance
            .iter()
            .filter(|&&ev| ev > threshold)
            .count()
    };
    let dead = dead_dimensions(&features.patch_tokens, 1e-6);
    let entropy = patch_entropy(&features.patch_tokens, 8, 30)?;
    let attention_gini = features
        .attention_weights
        .as_ref()
        .map(mean_gini)
        .transpose()?;
    let norm_stats = patch_norm_stats(&features.patch_tokens);

    // Per-image patch space metrics (need >= 2 patches)
    let iso = if features.n_patches >= 2 {
        isotropy_score(&features.patch_tokens)?
    } else {
        0.0
    };
    let uni = if features.n_patches >= 2 {
        uniformity(&features.patch_tokens)?
    } else {
        0.0
    };

    // Spatial coherence: gracefully degrade when patch count is not a perfect square
    let sc = spatial_coherence(&features.patch_tokens).ok();

    // Smooth effective rank (RankMe) and spectral decay rate
    let rm = rankme_from_spectrum(spec);
    let sd = spectral_decay_from_spectrum(spec);

    Ok(ModelMetrics {
        model_name: model_name.to_string(),
        n_patches: features.n_patches,
        embed_dim: features.embed_dim,
        effective_rank: rank,
        dead_dimensions: dead,
        patch_entropy: entropy,
        attention_gini,
        cls_l2_norm: features.cls_norm,
        patch_norm_mean: norm_stats.mean,
        patch_norm_std: norm_stats.std,
        top10_variance_pct: spec.top10_concentration * 100.0,
        components_90pct: spec.components_90pct,
        patch_isotropy: iso,
        patch_uniformity: uni,
        spatial_coherence: sc,
        rankme: rm,
        spectral_decay: sd,
    })
}

/// How two models' patch grids were aligned for comparison.
///
/// When models have different patch counts, metrics are computed over the
/// smaller count and the alignment is documented here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComparisonAlignment {
    pub patch_count_a: usize,
    pub patch_count_b: usize,
    pub compared_patch_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ComparisonAlignment {
    /// Human-readable description of the alignment (e.g. "256 shared patches").
    pub fn summary(&self) -> String {
        if self.patch_count_a == self.patch_count_b {
            format!("{} shared patches", self.compared_patch_count)
        } else {
            format!(
                "{} shared patches (from {} vs {})",
                self.compared_patch_count, self.patch_count_a, self.patch_count_b
            )
        }
    }
}

/// A labelled warning attached to a comparison metric explaining why its
/// value may be unreliable or incomplete.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetricCaveat {
    pub key: String,
    pub label: String,
    pub reason: String,
}

/// Cross-model comparison metrics between two models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub model_a: String,
    pub model_b: String,
    pub alignment: ComparisonAlignment,
    pub cls_cosine_sim: Option<f32>,
    pub linear_cka: f32,
    pub knn_overlap_k10: f32,
    pub mean_patch_correspondence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metric_caveats: Vec<MetricCaveat>,
}

impl ComparisonMetrics {
    /// Returns `true` if the comparison has any alignment notes or metric caveats.
    pub fn has_caveats(&self) -> bool {
        self.alignment.note.is_some() || !self.metric_caveats.is_empty()
    }

    /// Format all caveats as displayable text lines.
    pub fn caveat_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(note) = &self.alignment.note {
            lines.push(format!("Patch alignment: {note}"));
        }
        lines.extend(
            self.metric_caveats
                .iter()
                .map(|caveat| format!("{}: {}", caveat.label, caveat.reason)),
        );
        lines
    }
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

    let alignment = ComparisonAlignment {
        patch_count_a: a.n_patches,
        patch_count_b: b.n_patches,
        compared_patch_count: n,
        note: (a.n_patches != b.n_patches).then(|| {
            format!(
                "Compared the first {n} shared patches because the models expose different patch grids ({} vs {}).",
                a.n_patches, b.n_patches
            )
        }),
    };
    let mut metric_caveats = Vec::new();

    let cls_sim = match (&a.cls_token, &b.cls_token) {
        (Some(ca), Some(cb)) if ca.len() == cb.len() => Some(cls_cosine_similarity(ca, cb)),
        (Some(ca), Some(cb)) => {
            metric_caveats.push(MetricCaveat {
                key: "cls_cosine_sim".to_string(),
                label: "CLS cosine similarity".to_string(),
                reason: format!(
                    "Unavailable because CLS dimensions differ ({} vs {}).",
                    ca.len(),
                    cb.len()
                ),
            });
            None
        }
        (None, None) => {
            metric_caveats.push(MetricCaveat {
                key: "cls_cosine_sim".to_string(),
                label: "CLS cosine similarity".to_string(),
                reason: "Unavailable because neither model exposes a CLS token.".to_string(),
            });
            None
        }
        _ => {
            metric_caveats.push(MetricCaveat {
                key: "cls_cosine_sim".to_string(),
                label: "CLS cosine similarity".to_string(),
                reason: "Unavailable because only one model exposes a CLS token.".to_string(),
            });
            None
        }
    };

    let cka = linear_cka(&pa, &pb)?;
    let overlap = knn_overlap(&pa, &pb, 10)?;
    let mean_patch_correspondence = if pa.shape()[1] == pb.shape()[1] {
        Some(patch_correspondence(&pa, &pb)?.mean_similarity)
    } else {
        metric_caveats.push(MetricCaveat {
            key: "mean_patch_correspondence".to_string(),
            label: "Mean patch correspondence".to_string(),
            reason: format!(
                "Unavailable because embedding dimensions differ ({} vs {}).",
                pa.shape()[1],
                pb.shape()[1]
            ),
        });
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
        alignment,
        cls_cosine_sim: cls_sim,
        linear_cka: cka,
        knn_overlap_k10: overlap,
        mean_patch_correspondence,
        metric_caveats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelInfo, ModelOutput, OutputTensorMetadata, SSLMethod};
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
            tensor_metadata: OutputTensorMetadata {
                input_name: "pixel_values".into(),
                input_shape: vec![1, 3, 224, 224],
                output_name: "last_hidden_state".into(),
                output_shape: vec![1, n_patches + 1, embed_dim],
                sequence_has_cls: true,
                observed_patch_count: n_patches,
                embedding_dim: embed_dim,
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
        assert!(comparison
            .metric_caveats
            .iter()
            .any(|caveat| caveat.key == "cls_cosine_sim"));
        assert!(comparison
            .metric_caveats
            .iter()
            .any(|caveat| caveat.key == "mean_patch_correspondence"));
    }

    #[test]
    fn comparison_preserves_direct_metrics_for_matching_widths() {
        let a = features("dinov2-vit-l14", 256, 1024);
        let b = features("clip-vit-l14", 256, 1024);

        let comparison = compute_comparison(&a, &b, "dinov2-vit-l14", "clip-vit-l14").unwrap();

        assert!(comparison.cls_cosine_sim.is_some());
        assert!(comparison.mean_patch_correspondence.is_some());
        assert!(comparison.metric_caveats.is_empty());
        assert!(comparison.alignment.note.is_none());
    }

    #[test]
    fn comparison_records_patch_alignment_truncation() {
        let a = features("dinov2-vit-l14", 256, 1024);
        let b = features("mae-vit-l16", 196, 1024);

        let comparison = compute_comparison(&a, &b, "dinov2-vit-l14", "mae-vit-l16").unwrap();

        assert_eq!(comparison.alignment.compared_patch_count, 196);
        assert_eq!(comparison.alignment.patch_count_a, 256);
        assert_eq!(comparison.alignment.patch_count_b, 196);
        assert!(comparison.alignment.note.is_some());
    }

    #[test]
    fn metrics_can_reuse_a_precomputed_spectrum() {
        let feat = features("dinov2-vit-l14", 64, 32);
        let direct = compute_metrics(&feat, "dinov2-vit-l14").unwrap();
        let spec = variance_spectrum(&feat.patch_tokens, 16).unwrap();
        let reused = model_metrics_from_spectrum(&feat, "dinov2-vit-l14", &spec).unwrap();

        assert_eq!(reused.model_name, direct.model_name);
        assert_eq!(reused.effective_rank, direct.effective_rank);
        assert_eq!(reused.dead_dimensions, direct.dead_dimensions);
        approx::assert_relative_eq!(
            reused.top10_variance_pct,
            direct.top10_variance_pct,
            epsilon = 1e-4
        );
        assert_eq!(reused.components_90pct, direct.components_90pct);
        approx::assert_relative_eq!(reused.patch_isotropy, direct.patch_isotropy, epsilon = 1e-4);
        approx::assert_relative_eq!(
            reused.patch_uniformity,
            direct.patch_uniformity,
            epsilon = 1e-4
        );
    }

    #[test]
    fn compute_metrics_includes_finite_isotropy_and_uniformity() {
        let feat = features("dinov2-vit-l14", 64, 32);
        let metrics = compute_metrics(&feat, "dinov2-vit-l14").unwrap();

        assert!(metrics.patch_isotropy.is_finite());
        assert!(metrics.patch_isotropy >= 0.0);
        assert!(metrics.patch_isotropy <= 1.0);
        assert!(metrics.patch_uniformity.is_finite());
        assert!(metrics.patch_uniformity <= 0.0);
    }

    #[test]
    fn compute_metrics_handles_two_patches() {
        // Minimal viable patch count — isotropy/uniformity should compute
        let feat = features("dinov2-vit-l14", 2, 32);
        let metrics = compute_metrics(&feat, "dinov2-vit-l14").unwrap();

        assert!(metrics.patch_isotropy.is_finite());
        assert!(metrics.patch_uniformity.is_finite());
    }
}
