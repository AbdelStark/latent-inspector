use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::errors::VizError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Render model metrics as pretty-printed JSON to stdout.
pub fn print_metrics(metrics: &[ModelMetrics]) -> Result<(), VizError> {
    let json = serde_json::to_string_pretty(metrics)
        .map_err(|e| VizError::Html(format!("JSON serialization failed: {e}")))?;
    println!("{json}");
    Ok(())
}

/// Render comparison metrics as pretty-printed JSON to stdout.
pub fn print_comparison(comparisons: &[ComparisonMetrics]) -> Result<(), VizError> {
    let json = serde_json::to_string_pretty(comparisons)
        .map_err(|e| VizError::Html(format!("JSON serialization failed: {e}")))?;
    println!("{json}");
    Ok(())
}

/// Write a combined report (metrics + comparisons) to a JSON file.
pub fn write_report(
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
    path: &Path,
) -> Result<(), VizError> {
    let mut report: HashMap<&str, Value> = HashMap::new();
    report.insert(
        "metrics",
        serde_json::to_value(metrics).map_err(|e| VizError::Html(e.to_string()))?,
    );
    report.insert(
        "comparisons",
        serde_json::to_value(comparisons).map_err(|e| VizError::Html(e.to_string()))?,
    );

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| VizError::Html(format!("JSON serialization failed: {e}")))?;

    std::fs::write(path, json)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", path.display())))?;

    Ok(())
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
