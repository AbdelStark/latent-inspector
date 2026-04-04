use crate::analysis::{ComparisonMetrics, ModelMetrics, VarianceSpectrum};
use crate::dataset::DatasetProcessingSummary;
use crate::extract::{AttentionMapBasis, EmbeddingBasis};
use crate::validation::report::ModelValidationSummary;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseMetricUnavailability {
    pub reason: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseMetricSupport {
    pub supported_pairs: usize,
    pub total_pairs: usize,
    pub unavailable_pairs: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unavailable_reasons: Vec<PairwiseMetricUnavailability>,
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
    pub cls_cosine_support: PairwiseMetricSupport,
    pub linear_cka_matrix: PairwiseMatrix,
    pub linear_cka_support: PairwiseMetricSupport,
    pub knn_overlap_matrix: PairwiseMatrix,
    pub knn_overlap_support: PairwiseMetricSupport,
    pub correspondence_matrix: PairwiseMatrix,
    pub correspondence_support: PairwiseMetricSupport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub image: String,
    pub requested_models: Vec<String>,
    pub metrics: Vec<ModelMetrics>,
    pub comparisons: Vec<ComparisonMetrics>,
    pub overview: CompareOverview,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<ModelValidationSummary>,
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
        cls_cosine_matrix: build_pairwise_matrix(
            &labels,
            metrics,
            comparisons,
            MetricKind::ClsCosine,
        ),
        cls_cosine_support: build_pairwise_support(comparisons, MetricKind::ClsCosine),
        linear_cka_matrix: build_pairwise_matrix(
            &labels,
            metrics,
            comparisons,
            MetricKind::LinearCka,
        ),
        linear_cka_support: build_pairwise_support(comparisons, MetricKind::LinearCka),
        knn_overlap_matrix: build_pairwise_matrix(
            &labels,
            metrics,
            comparisons,
            MetricKind::KnnOverlap,
        ),
        knn_overlap_support: build_pairwise_support(comparisons, MetricKind::KnnOverlap),
        correspondence_matrix: build_pairwise_matrix(
            &labels,
            metrics,
            comparisons,
            MetricKind::MeanPatchCorrespondence,
        ),
        correspondence_support: build_pairwise_support(
            comparisons,
            MetricKind::MeanPatchCorrespondence,
        ),
    }
}

pub fn build_compare_report(
    image: impl Into<String>,
    requested_models: Vec<String>,
    metrics: Vec<ModelMetrics>,
    comparisons: Vec<ComparisonMetrics>,
    validation: Vec<ModelValidationSummary>,
) -> CompareReport {
    let overview = build_compare_overview(&metrics, &comparisons);
    CompareReport {
        image: image.into(),
        requested_models,
        metrics,
        comparisons,
        overview,
        validation,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VarianceSpectrumReport {
    pub ratios: Vec<f32>,
    pub cumulative: Vec<f32>,
    pub components_90pct: usize,
    pub components_99pct: usize,
    pub top10_concentration: f32,
}

impl From<&VarianceSpectrum> for VarianceSpectrumReport {
    fn from(value: &VarianceSpectrum) -> Self {
        Self {
            ratios: value.ratios.to_vec(),
            cumulative: value.cumulative.to_vec(),
            components_90pct: value.components_90pct,
            components_99pct: value.components_99pct,
            top10_concentration: value.top10_concentration,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InspectAttentionSummary {
    pub mean_gini: f32,
    pub layers: usize,
    pub heads: usize,
    pub token_count: usize,
    pub map_basis: AttentionMapBasis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectReport {
    pub image: String,
    pub model: String,
    pub metrics: ModelMetrics,
    pub validation: ModelValidationSummary,
    pub variance_spectrum: VarianceSpectrumReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<InspectAttentionSummary>,
}

pub fn build_inspect_report(
    image: impl Into<String>,
    model: impl Into<String>,
    metrics: ModelMetrics,
    validation: ModelValidationSummary,
    variance_spectrum: &VarianceSpectrum,
    attention: Option<InspectAttentionSummary>,
) -> InspectReport {
    InspectReport {
        image: image.into(),
        model: model.into(),
        metrics,
        validation,
        variance_spectrum: variance_spectrum.into(),
        attention,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborMatch {
    pub rank: usize,
    pub image: String,
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborsReport {
    pub query_image: String,
    pub dataset: String,
    pub model: String,
    pub embedding_basis: EmbeddingBasis,
    pub requested_k: usize,
    pub dataset_summary: DatasetProcessingSummary,
    pub neighbors: Vec<NeighborMatch>,
    pub validation: ModelValidationSummary,
}

impl NeighborsReport {
    pub fn similarity_series(&self) -> Vec<f32> {
        self.neighbors
            .iter()
            .map(|neighbor| neighbor.similarity)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetricValue {
    pub key: String,
    pub label: String,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityReport {
    pub model_a: String,
    pub model_b: String,
    pub dataset: String,
    pub dataset_embedding_basis: EmbeddingBasis,
    pub requested_metric: String,
    pub sample_count: usize,
    pub dataset_summary: DatasetProcessingSummary,
    pub metrics: Vec<SimilarityMetricValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<ModelValidationSummary>,
}

impl SimilarityReport {
    pub fn metric_series(&self) -> Vec<f32> {
        self.metrics.iter().map(|metric| metric.value).collect()
    }

    pub fn metric_value(&self, key: &str) -> Option<f32> {
        self.metrics
            .iter()
            .find(|metric| metric.key == key)
            .map(|metric| metric.value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftStep {
    pub from_checkpoint: String,
    pub to_checkpoint: String,
    pub linear_cka: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub model: String,
    pub checkpoints: String,
    pub dataset: String,
    pub dataset_embedding_basis: EmbeddingBasis,
    pub checkpoint_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_summary: Option<DatasetProcessingSummary>,
    pub drift: Vec<DriftStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_consecutive_cka: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub largest_shift: Option<DriftStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<ModelValidationSummary>,
}

impl DriftReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: impl Into<String>,
        checkpoints: impl Into<String>,
        dataset: impl Into<String>,
        dataset_embedding_basis: EmbeddingBasis,
        checkpoint_names: Vec<String>,
        dataset_summary: Option<DatasetProcessingSummary>,
        drift: Vec<DriftStep>,
        validation: Vec<ModelValidationSummary>,
    ) -> Self {
        let mean_consecutive_cka = (!drift.is_empty())
            .then(|| drift.iter().map(|step| step.linear_cka).sum::<f32>() / drift.len() as f32);
        let largest_shift = drift
            .iter()
            .min_by(|left, right| left.linear_cka.total_cmp(&right.linear_cka))
            .cloned();

        Self {
            model: model.into(),
            checkpoints: checkpoints.into(),
            dataset: dataset.into(),
            dataset_embedding_basis,
            checkpoint_names,
            dataset_summary,
            drift,
            mean_consecutive_cka,
            largest_shift,
            validation,
        }
    }

    pub fn cka_series(&self) -> Vec<f32> {
        self.drift.iter().map(|step| step.linear_cka).collect()
    }
}

/// Per-image metrics collected during profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileImageMetrics {
    pub image: String,
    pub effective_rank: usize,
    pub dead_dimensions: usize,
    pub patch_entropy: f32,
    pub attention_gini: Option<f32>,
    pub cls_l2_norm: Option<f32>,
    pub patch_norm_mean: f32,
    pub patch_norm_std: f32,
    pub top10_variance_pct: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial_coherence: Option<f32>,
}

/// Aggregate statistics for a single metric across all profiled images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStatistic {
    pub key: String,
    pub label: String,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

/// Space-level metrics computed over the entire embedding matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceMetrics {
    pub isotropy_cosine: f32,
    pub isotropy_partition: f32,
    pub uniformity: f32,
    pub intrinsic_dimensionality: f32,
}

/// Comprehensive representation profile for a model over a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    pub model: String,
    pub dataset: String,
    pub embedding_basis: EmbeddingBasis,
    pub sample_count: usize,
    pub embed_dim: usize,
    pub dataset_summary: DatasetProcessingSummary,
    pub space_metrics: SpaceMetrics,
    pub aggregate_metrics: Vec<AggregateStatistic>,
    pub per_image_metrics: Vec<ProfileImageMetrics>,
    pub validation: ModelValidationSummary,
}

impl ProfileReport {
    /// Returns a flat list of the key space-level metric values for charting.
    pub fn space_metric_series(&self) -> Vec<(&str, f32)> {
        vec![
            ("Isotropy (cosine)", self.space_metrics.isotropy_cosine),
            (
                "Isotropy (partition)",
                self.space_metrics.isotropy_partition,
            ),
            ("Uniformity", self.space_metrics.uniformity),
            ("Intrinsic dim", self.space_metrics.intrinsic_dimensionality),
        ]
    }
}

/// Build aggregate statistics from a list of per-image metric values.
pub fn build_aggregate(key: &str, label: &str, values: &[f32]) -> AggregateStatistic {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n.max(1.0);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    AggregateStatistic {
        key: key.to_string(),
        label: label.to_string(),
        mean,
        std,
        min,
        max,
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
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
    kind: MetricKind,
) -> PairwiseMatrix {
    let mut rows = vec![vec![None; labels.len()]; labels.len()];
    let indexes = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();

    let metric_indexes = metrics
        .iter()
        .enumerate()
        .map(|(index, metric)| (metric.model_name.as_str(), index))
        .collect::<HashMap<_, _>>();

    for (index, row) in rows.iter_mut().enumerate() {
        row[index] = indexes
            .get(labels[index].as_str())
            .and_then(|_| metric_indexes.get(labels[index].as_str()))
            .and_then(|metric_index| diagonal_metric_value(kind, &metrics[*metric_index]));
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

fn build_pairwise_support(
    comparisons: &[ComparisonMetrics],
    kind: MetricKind,
) -> PairwiseMetricSupport {
    let supported_pairs = comparisons
        .iter()
        .filter(|comparison| metric_value(comparison, kind).is_some())
        .count();
    let total_pairs = comparisons.len();

    let mut reasons = HashMap::<String, usize>::new();
    for comparison in comparisons {
        if let Some(reason) = metric_unavailable_reason(comparison, kind) {
            *reasons.entry(reason.to_string()).or_insert(0) += 1;
        }
    }

    let mut unavailable_reasons = reasons
        .into_iter()
        .map(|(reason, count)| PairwiseMetricUnavailability { reason, count })
        .collect::<Vec<_>>();
    unavailable_reasons.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.reason.cmp(&right.reason))
    });

    PairwiseMetricSupport {
        supported_pairs,
        total_pairs,
        unavailable_pairs: total_pairs.saturating_sub(supported_pairs),
        unavailable_reasons,
    }
}

fn diagonal_metric_value(kind: MetricKind, metric: &ModelMetrics) -> Option<f32> {
    match kind {
        MetricKind::ClsCosine => metric.cls_l2_norm.map(|_| 1.0),
        MetricKind::LinearCka | MetricKind::KnnOverlap | MetricKind::MeanPatchCorrespondence => {
            Some(1.0)
        }
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

fn metric_unavailable_reason(comparison: &ComparisonMetrics, kind: MetricKind) -> Option<&str> {
    comparison
        .metric_caveats
        .iter()
        .find(|caveat| caveat.key == kind.key())
        .map(|caveat| caveat.reason.as_str())
}

impl MetricKind {
    fn key(self) -> &'static str {
        match self {
            MetricKind::ClsCosine => "cls_cosine_sim",
            MetricKind::LinearCka => "linear_cka",
            MetricKind::KnnOverlap => "knn_overlap_k10",
            MetricKind::MeanPatchCorrespondence => "mean_patch_correspondence",
        }
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

    if let Some(metric) = metrics
        .iter()
        .max_by(|a, b| a.patch_entropy.total_cmp(&b.patch_entropy))
    {
        highlights.push(ModelHighlight {
            label: "Highest patch entropy".to_string(),
            model: metric.model_name.clone(),
            value: format!("{:.2}", metric.patch_entropy),
        });
    }

    if let Some(metric) = metrics
        .iter()
        .max_by(|a, b| a.top10_variance_pct.total_cmp(&b.top10_variance_pct))
    {
        highlights.push(ModelHighlight {
            label: "Most top-heavy variance".to_string(),
            model: metric.model_name.clone(),
            value: format!("{:.1}%", metric.top10_variance_pct),
        });
    }

    if let Some(metric) = metrics
        .iter()
        .filter(|metric| metric.attention_gini.is_some())
        .max_by(|a, b| {
            a.attention_gini
                .unwrap_or(f32::NEG_INFINITY)
                .total_cmp(&b.attention_gini.unwrap_or(f32::NEG_INFINITY))
        })
    {
        highlights.push(ModelHighlight {
            label: "Most focused attention".to_string(),
            model: metric.model_name.clone(),
            value: format!("{:.2}", metric.attention_gini.unwrap_or_default()),
        });
    }

    highlights
}

fn build_comparison_highlights(comparisons: &[ComparisonMetrics]) -> Vec<ComparisonHighlight> {
    let mut highlights = Vec::new();

    if let Some(comparison) = comparisons
        .iter()
        .max_by(|a, b| a.linear_cka.total_cmp(&b.linear_cka))
    {
        highlights.push(ComparisonHighlight {
            label: "Strongest CKA alignment".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.linear_cka,
        });
    }

    if let Some(comparison) = comparisons
        .iter()
        .min_by(|a, b| a.linear_cka.total_cmp(&b.linear_cka))
    {
        highlights.push(ComparisonHighlight {
            label: "Weakest CKA alignment".to_string(),
            model_a: comparison.model_a.clone(),
            model_b: comparison.model_b.clone(),
            value: comparison.linear_cka,
        });
    }

    if let Some(comparison) = comparisons
        .iter()
        .max_by(|a, b| a.knn_overlap_k10.total_cmp(&b.knn_overlap_k10))
    {
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
                .total_cmp(&b.mean_patch_correspondence.unwrap_or(f32::NEG_INFINITY))
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
    use crate::analysis::VarianceSpectrum;
    use crate::validation::report::{
        CheckSummary, ModelValidationSummary, ParityValidationSummary, TensorValidationSummary,
        ValidationStatus,
    };
    use ndarray::Array1;

    fn metrics() -> Vec<ModelMetrics> {
        vec![
            ModelMetrics {
                model_name: "dinov2".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 300,
                dead_dimensions: 4,
                patch_entropy: 6.1,
                attention_gini: Some(0.63),
                cls_l2_norm: Some(1.0),
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 25.0,
                components_90pct: 64,
                patch_isotropy: 0.65,
                patch_uniformity: -2.1,
                spatial_coherence: Some(0.72),
                rankme: 22.4,
                spectral_decay: Some(1.35),
            },
            ModelMetrics {
                model_name: "clip".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 210,
                dead_dimensions: 2,
                patch_entropy: 5.0,
                attention_gini: Some(0.48),
                cls_l2_norm: Some(1.0),
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 41.0,
                components_90pct: 52,
                patch_isotropy: 0.65,
                patch_uniformity: -2.1,
                spatial_coherence: Some(0.72),
                rankme: 15.8,
                spectral_decay: Some(1.92),
            },
        ]
    }

    fn comparisons() -> Vec<ComparisonMetrics> {
        vec![ComparisonMetrics {
            model_a: "dinov2".into(),
            model_b: "clip".into(),
            alignment: crate::analysis::ComparisonAlignment {
                patch_count_a: 256,
                patch_count_b: 256,
                compared_patch_count: 256,
                note: None,
            },
            cls_cosine_sim: Some(0.42),
            linear_cka: 0.77,
            knn_overlap_k10: 0.33,
            mean_patch_correspondence: Some(0.51),
            metric_caveats: Vec::new(),
        }]
    }

    fn validation_summary(model: &str) -> ModelValidationSummary {
        ModelValidationSummary::from_checks(
            model,
            "2026-03-28T00:00:00Z",
            CheckSummary::validated("Preprocess matches contract."),
            vec![TensorValidationSummary {
                name: "last_hidden_state".into(),
                role: "patch embeddings".into(),
                status: ValidationStatus::Validated,
                summary: "Tensor semantics match the registry contract.".into(),
            }],
            ParityValidationSummary::new(
                ValidationStatus::Validated,
                "Reference parity matches approved evidence.",
            ),
        )
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
        assert_eq!(overview.cls_cosine_support.supported_pairs, 1);
        assert_eq!(overview.cls_cosine_support.total_pairs, 1);
        assert!(overview.cls_cosine_support.unavailable_reasons.is_empty());
    }

    #[test]
    fn compare_overview_includes_highlights() {
        let overview = build_compare_overview(&metrics(), &comparisons());

        assert!(overview
            .model_highlights
            .iter()
            .any(|highlight| highlight.label == "Highest effective rank"));
        assert!(overview
            .model_highlights
            .iter()
            .any(|highlight| highlight.label == "Most focused attention"));
        assert!(overview
            .comparison_highlights
            .iter()
            .any(|highlight| highlight.label == "Strongest CKA alignment"));
    }

    #[test]
    fn compare_overview_handles_non_finite_highlight_values_without_panicking() {
        let mut model_metrics = metrics();
        model_metrics[0].patch_entropy = f32::NAN;
        model_metrics[1].top10_variance_pct = f32::INFINITY;
        let mut comparison_metrics = comparisons();
        comparison_metrics[0].linear_cka = f32::NAN;

        let overview = build_compare_overview(&model_metrics, &comparison_metrics);

        assert!(!overview.model_highlights.is_empty());
        assert!(!overview.comparison_highlights.is_empty());
    }

    #[test]
    fn compare_report_tracks_requested_models() {
        let report = build_compare_report(
            "images/street.png",
            vec!["dinov2".into(), "clip".into()],
            metrics(),
            comparisons(),
            vec![validation_summary("dinov2"), validation_summary("clip")],
        );

        assert_eq!(report.image, "images/street.png");
        assert_eq!(report.requested_models, vec!["dinov2", "clip"]);
        assert_eq!(report.overview.linear_cka_matrix.rows[0][1], Some(0.77));
        assert_eq!(report.validation.len(), 2);
    }

    #[test]
    fn compare_overview_marks_clsless_diagonal_unavailable_and_tracks_support() {
        let metrics = vec![
            ModelMetrics {
                model_name: "dinov2".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 300,
                dead_dimensions: 4,
                patch_entropy: 6.1,
                attention_gini: Some(0.63),
                cls_l2_norm: Some(1.0),
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 25.0,
                components_90pct: 64,
                patch_isotropy: 0.65,
                patch_uniformity: -2.1,
                spatial_coherence: Some(0.72),
                rankme: 22.4,
                spectral_decay: Some(1.35),
            },
            ModelMetrics {
                model_name: "mae".into(),
                n_patches: 196,
                embed_dim: 1024,
                effective_rank: 210,
                dead_dimensions: 2,
                patch_entropy: 5.0,
                attention_gini: None,
                cls_l2_norm: None,
                patch_norm_mean: 2.0,
                patch_norm_std: 0.4,
                top10_variance_pct: 41.0,
                components_90pct: 52,
                patch_isotropy: 0.65,
                patch_uniformity: -2.1,
                spatial_coherence: Some(0.72),
                rankme: 15.8,
                spectral_decay: Some(1.92),
            },
        ];
        let comparisons = vec![ComparisonMetrics {
            model_a: "dinov2".into(),
            model_b: "mae".into(),
            alignment: crate::analysis::ComparisonAlignment {
                patch_count_a: 256,
                patch_count_b: 196,
                compared_patch_count: 196,
                note: Some(
                    "Compared the first 196 shared patches because the models expose different patch grids (256 vs 196)."
                        .into(),
                ),
            },
            cls_cosine_sim: None,
            linear_cka: 0.77,
            knn_overlap_k10: 0.33,
            mean_patch_correspondence: Some(0.51),
            metric_caveats: vec![crate::analysis::MetricCaveat {
                key: "cls_cosine_sim".into(),
                label: "CLS cosine similarity".into(),
                reason: "Unavailable because only one model exposes a CLS token.".into(),
            }],
        }];

        let overview = build_compare_overview(&metrics, &comparisons);

        assert_eq!(overview.cls_cosine_matrix.rows[0][0], Some(1.0));
        assert_eq!(overview.cls_cosine_matrix.rows[1][1], None);
        assert_eq!(overview.cls_cosine_matrix.rows[0][1], None);
        assert_eq!(overview.cls_cosine_support.supported_pairs, 0);
        assert_eq!(overview.cls_cosine_support.total_pairs, 1);
        assert_eq!(overview.cls_cosine_support.unavailable_pairs, 1);
        assert_eq!(
            overview.cls_cosine_support.unavailable_reasons,
            vec![PairwiseMetricUnavailability {
                reason: "Unavailable because only one model exposes a CLS token.".into(),
                count: 1,
            }]
        );
    }

    #[test]
    fn inspect_report_materializes_variance_spectrum() {
        let report = build_inspect_report(
            "images/street.png",
            "dinov2",
            metrics().into_iter().next().unwrap(),
            validation_summary("dinov2"),
            &VarianceSpectrum {
                explained_variance: Array1::from_vec(vec![5.0, 3.0, 2.0]),
                ratios: Array1::from_vec(vec![0.5, 0.3, 0.2]),
                cumulative: Array1::from_vec(vec![0.5, 0.8, 1.0]),
                components_90pct: 3,
                components_99pct: 3,
                top10_concentration: 1.0,
            },
            Some(InspectAttentionSummary {
                mean_gini: 0.63,
                layers: 2,
                heads: 4,
                token_count: 257,
                map_basis: AttentionMapBasis::ClsToPatch,
            }),
        );

        assert_eq!(report.image, "images/street.png");
        assert_eq!(report.model, "dinov2");
        assert_eq!(report.variance_spectrum.components_90pct, 3);
        assert_eq!(report.variance_spectrum.ratios, vec![0.5, 0.3, 0.2]);
        assert_eq!(report.attention.as_ref().unwrap().mean_gini, 0.63);
    }

    #[test]
    fn neighbors_report_exposes_similarity_series() {
        let report = NeighborsReport {
            query_image: "query.png".into(),
            dataset: "dataset".into(),
            model: "dinov2".into(),
            embedding_basis: EmbeddingBasis::ClsToken,
            requested_k: 2,
            dataset_summary: DatasetProcessingSummary {
                discovered: 3,
                loaded: 2,
                skipped: 1,
                skipped_examples: Vec::new(),
            },
            neighbors: vec![
                NeighborMatch {
                    rank: 1,
                    image: "class-a/leaf".into(),
                    similarity: 0.91,
                },
                NeighborMatch {
                    rank: 2,
                    image: "root".into(),
                    similarity: 0.82,
                },
            ],
            validation: validation_summary("dinov2"),
        };

        assert_eq!(report.similarity_series(), vec![0.91, 0.82]);
    }

    #[test]
    fn similarity_report_supports_metric_lookup() {
        let report = SimilarityReport {
            model_a: "dinov2".into(),
            model_b: "clip".into(),
            dataset: "dataset".into(),
            dataset_embedding_basis: EmbeddingBasis::MeanPatch,
            requested_metric: "all".into(),
            sample_count: 4,
            dataset_summary: DatasetProcessingSummary {
                discovered: 4,
                loaded: 4,
                skipped: 0,
                skipped_examples: Vec::new(),
            },
            metrics: vec![
                SimilarityMetricValue {
                    key: "linear_cka".into(),
                    label: "Linear CKA".into(),
                    value: 0.77,
                },
                SimilarityMetricValue {
                    key: "knn_overlap_k10".into(),
                    label: "k-NN overlap (k=10)".into(),
                    value: 0.43,
                },
            ],
            note: None,
            validation: vec![validation_summary("dinov2"), validation_summary("clip")],
        };

        assert_eq!(report.metric_value("linear_cka"), Some(0.77));
        assert_eq!(report.metric_series(), vec![0.77, 0.43]);
        assert_eq!(report.metric_value("mean_cls_cosine"), None);
    }

    #[test]
    fn drift_report_computes_aggregate_fields() {
        let report = DriftReport::new(
            "dinov2",
            "checkpoints",
            "dataset",
            EmbeddingBasis::MeanPatch,
            vec!["step-1".into(), "step-2".into(), "step-10".into()],
            Some(DatasetProcessingSummary {
                discovered: 3,
                loaded: 3,
                skipped: 0,
                skipped_examples: Vec::new(),
            }),
            vec![
                DriftStep {
                    from_checkpoint: "step-1".into(),
                    to_checkpoint: "step-2".into(),
                    linear_cka: 0.93,
                },
                DriftStep {
                    from_checkpoint: "step-2".into(),
                    to_checkpoint: "step-10".into(),
                    linear_cka: 0.71,
                },
            ],
            vec![validation_summary("step-1"), validation_summary("step-2")],
        );

        assert_eq!(report.mean_consecutive_cka, Some(0.82));
        assert_eq!(
            report.largest_shift,
            Some(DriftStep {
                from_checkpoint: "step-2".into(),
                to_checkpoint: "step-10".into(),
                linear_cka: 0.71,
            })
        );
        assert_eq!(report.cka_series(), vec![0.93, 0.71]);
        assert_eq!(report.validation.len(), 2);
    }
}
