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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComparisonAlignment {
    pub patch_count_a: usize,
    pub patch_count_b: usize,
    pub compared_patch_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ComparisonAlignment {
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
    pub fn has_caveats(&self) -> bool {
        self.alignment.note.is_some() || !self.metric_caveats.is_empty()
    }

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
}
