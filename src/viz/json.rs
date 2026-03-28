use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::errors::VizError;
use crate::models::ModelCatalogReport;
use crate::validation::report::ModelValidationSummary;
use crate::viz::report::{
    CompareReport, DriftReport, InspectReport, NeighborsReport, SimilarityReport,
};
use serde::Serialize;
use std::path::Path;

/// Render model metrics as pretty-printed JSON to stdout.
pub fn print_metrics(metrics: &[ModelMetrics]) -> Result<(), VizError> {
    print_value(metrics)
}

/// Render comparison metrics as pretty-printed JSON to stdout.
pub fn print_comparison(comparisons: &[ComparisonMetrics]) -> Result<(), VizError> {
    print_value(comparisons)
}

pub fn print_validation_summaries(summaries: &[ModelValidationSummary]) -> Result<(), VizError> {
    print_value(summaries)
}

pub fn print_compare_report(report: &CompareReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_compare_report(report: &CompareReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

pub fn print_inspect_report(report: &InspectReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_inspect_report(report: &InspectReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

pub fn write_validation_report(
    summaries: &[ModelValidationSummary],
    path: &Path,
) -> Result<(), VizError> {
    write_value(summaries, path)
}

pub fn print_neighbors_report(report: &NeighborsReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_neighbors_report(report: &NeighborsReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

pub fn print_similarity_report(report: &SimilarityReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_similarity_report(report: &SimilarityReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

pub fn print_drift_report(report: &DriftReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_drift_report(report: &DriftReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

pub fn print_model_catalog(report: &ModelCatalogReport) -> Result<(), VizError> {
    print_value(report)
}

pub fn write_model_catalog(report: &ModelCatalogReport, path: &Path) -> Result<(), VizError> {
    write_value(report, path)
}

fn print_value<T: Serialize + ?Sized>(value: &T) -> Result<(), VizError> {
    let json = serialize_pretty(value)?;
    println!("{json}");
    Ok(())
}

fn write_value<T: Serialize + ?Sized>(value: &T, path: &Path) -> Result<(), VizError> {
    let json = serialize_pretty(value)?;
    std::fs::write(path, json)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", path.display())))?;
    Ok(())
}

fn serialize_pretty<T: Serialize + ?Sized>(value: &T) -> Result<String, VizError> {
    serde_json::to_string_pretty(value)
        .map_err(|e| VizError::Html(format!("JSON serialization failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_metrics() -> ModelMetrics {
        ModelMetrics {
            model_name: "test".into(),
            n_patches: 256,
            embed_dim: 1024,
            effective_rank: 128,
            dead_dimensions: 0,
            patch_entropy: 2.5,
            cls_l2_norm: Some(18.4),
            patch_norm_mean: 12.0,
            patch_norm_std: 2.0,
            top10_variance_pct: 45.0,
            components_90pct: 32,
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let m = dummy_metrics();
        let json = serde_json::to_string_pretty(&m).unwrap();
        let parsed: ModelMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_name, "test");
        assert_eq!(parsed.effective_rank, 128);
    }
}
