use crate::analysis::{ComparisonMetrics, ModelMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseMatrix {
    pub labels: Vec<String>,
    pub rows: Vec<Vec<Option<f32>>>,
}

impl PairwiseMatrix {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn has_off_diagonal_values(&self) -> bool {
        self.rows.iter().enumerate().any(|(row_idx, row)| {
            row.iter()
                .enumerate()
                .any(|(col_idx, value)| row_idx != col_idx && value.is_some())
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHighlight {
    pub label: String,
    pub model: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonHighlight {
    pub label: String,
    pub model_a: String,
    pub model_b: String,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareOverview {
    pub model_highlights: Vec<ModelHighlight>,
    pub comparison_highlights: Vec<ComparisonHighlight>,
    pub cls_cosine_matrix: PairwiseMatrix,
    pub linear_cka_matrix: PairwiseMatrix,
    pub knn_overlap_matrix: PairwiseMatrix,
    pub correspondence_matrix: PairwiseMatrix,
}

pub fn build_compare_overview(
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
) -> CompareOverview {
    let labels = metrics
        .iter()
        .map(|metric| metric.model_name.clone())
        .collect::<Vec<_>>();

    CompareOverview {
        model_highlights: build_model_highlights(metrics),
        comparison_highlights: build_comparison_highlights(comparisons),
        cls_cosine_matrix: build_pairwise_matrix(&labels, comparisons, MetricKind::ClsCosine),
        linear_cka_matrix: build_pairwise_matrix(&labels, comparisons, MetricKind::LinearCka),
        knn_overlap_matrix: build_pairwise_matrix(&labels, comparisons, MetricKind::KnnOverlap),
        correspondence_matrix: build_pairwise_matrix(
            &labels,
            comparisons,
            MetricKind::MeanPatchCorrespondence,
        ),
    }
}

#[derive(Clone, Copy)]
enum MetricKind {
    ClsCosine,
    LinearCka,
    KnnOverlap,
    MeanPatchCorrespondence,
}

fn build_pairwise_matrix(
    labels: &[String],
    comparisons: &[ComparisonMetrics],
    kind: MetricKind,
) -> PairwiseMatrix {
    let mut rows = vec![vec![None; labels.len()]; labels.len()];
    let indexes = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();

    for (index, row) in rows.iter_mut().enumerate() {
        row[index] = Some(1.0);
    }

    for comparison in comparisons {
        let Some(&row) = indexes.get(comparison.model_a.as_str()) else {
            continue;
        };
        let Some(&col) = indexes.get(comparison.model_b.as_str()) else {
            continue;
        };
        let value = metric_value(comparison, kind);
        rows[row][col] = value;
        rows[col][row] = value;
    }

    PairwiseMatrix {
        labels: labels.to_vec(),
        rows,
    }
}

fn metric_value(comparison: &ComparisonMetrics, kind: MetricKind) -> Option<f32> {
    match kind {
        MetricKind::ClsCosine => comparison.cls_cosine_sim,
        MetricKind::LinearCka => Some(comparison.linear_cka),
        MetricKind::KnnOverlap => Some(comparison.knn_overlap_k10),
        MetricKind::MeanPatchCorrespondence => comparison.mean_patch_correspondence,
    }
}

fn build_model_highlights(metrics: &[ModelMetrics]) -> Vec<ModelHighlight> {
    let mut highlights = Vec::new();

    if let Some(metric) = metrics
        .iter()
        .max_by_key(|metric| (metric.effective_rank, usize::MAX - metric.dead_dimensions))
    {
        highlights.push(ModelHighlight {
            label: "Highest effective rank".to_string(),
            model: metric.model_name.clone(),
            value: format!("{}/{}", metric.effective_rank, metric.embed_dim),
        });
    }

    if let Some(metric) = metrics.iter().max_by(|a, b| {
        a.patch_entropy
            .partial_cmp(&b.patch_entropy)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push(ModelHighlight {
            label: "Highest patch entropy".to_string(),
            model: metric.model_name.clone(),
            value: format!("{:.2}", metric.patch_entropy),
        });
    }

    if let Some(metric) = metrics.iter().max_by(|a, b| {
        a.top10_variance_pct
            .partial_cmp(&b.top10_variance_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push(ModelHighlight {
            label: "Most top-heavy variance".to_string(),
            model: metric.model_name.clone(),
            value: format!("{:.1}%", metric.top10_variance_pct),
        });
    }

    highlights
}

fn build_comparison_highlights(comparisons: &[ComparisonMetrics]) -> Vec<ComparisonHighlight> {
    let mut highlights = Vec::new();

    if let Some(comparison) = comparisons.iter().max_by(|a, b| {
        a.linear_cka
            .partial_cmp(&b.linear_cka)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push(ComparisonHighlight {
            label: "Strongest CKA alignment".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.linear_cka,
        });
    }

    if let Some(comparison) = comparisons.iter().min_by(|a, b| {
        a.linear_cka
            .partial_cmp(&b.linear_cka)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push(ComparisonHighlight {
            label: "Weakest CKA alignment".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.linear_cka,
        });
    }

    if let Some(comparison) = comparisons.iter().max_by(|a, b| {
        a.knn_overlap_k10
            .partial_cmp(&b.knn_overlap_k10)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push(ComparisonHighlight {
            label: "Highest neighborhood overlap".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.knn_overlap_k10,
        });
    }

    if let Some(comparison) = comparisons
        .iter()
        .filter(|comparison| comparison.mean_patch_correspondence.is_some())
        .max_by(|a, b| {
            a.mean_patch_correspondence
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&b.mean_patch_correspondence.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        highlights.push(ComparisonHighlight {
            label: "Strongest patch correspondence".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.mean_patch_correspondence.unwrap_or_default(),
        });
    }

    highlights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> Vec<ModelMetrics> {
        vec![
            ModelMetrics {
                model_name: "dinov2".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 300,
                dead_dimensions: 4,
                patch_entropy: 6.1,
                cls_l2_norm: Some(1.0),
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 25.0,
                components_90pct: 64,
            },
            ModelMetrics {
                model_name: "clip".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 210,
                dead_dimensions: 2,
                patch_entropy: 5.0,
                cls_l2_norm: Some(1.0),
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 41.0,
                components_90pct: 52,
            },
        ]
    }

    fn comparisons() -> Vec<ComparisonMetrics> {
        vec![ComparisonMetrics {
            model_a: "dinov2".into(),
            model_b: "clip".into(),
            cls_cosine_sim: Some(0.42),
            linear_cka: 0.77,
            knn_overlap_k10: 0.33,
            mean_patch_correspondence: Some(0.51),
        }]
    }

    #[test]
    fn compare_overview_builds_symmetric_matrices() {
        let overview = build_compare_overview(&metrics(), &comparisons());

        assert_eq!(overview.linear_cka_matrix.rows[0][0], Some(1.0));
        assert_eq!(overview.linear_cka_matrix.rows[1][1], Some(1.0));
        assert_eq!(overview.linear_cka_matrix.rows[0][1], Some(0.77));
        assert_eq!(overview.linear_cka_matrix.rows[1][0], Some(0.77));
        assert_eq!(overview.cls_cosine_matrix.rows[0][1], Some(0.42));
        assert_eq!(overview.correspondence_matrix.rows[0][1], Some(0.51));
    }

    #[test]
    fn compare_overview_includes_highlights() {
        let overview = build_compare_overview(&metrics(), &comparisons());

        assert!(overview
            .model_highlights
            .iter()
            .any(|highlight| highlight.label == "Highest effective rank"));
        assert!(overview
            .comparison_highlights
            .iter()
            .any(|highlight| highlight.label == "Strongest CKA alignment"));
    }
}
