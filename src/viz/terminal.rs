//! Terminal rendering using Unicode block characters and ANSI colors.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::dataset::DatasetProcessingSummary;
use crate::validation::report::ModelValidationSummary;
use crate::viz::report::{
    CompareOverview, DriftReport, NeighborsReport, PairwiseMatrix, SimilarityReport,
};
use ndarray::Array2;

const BLOCK_CHARS: &[char] = &[' ', '░', '▒', '▓', '█'];

/// Map a normalized value in `[0, 1]` to a Unicode block character.
fn value_to_block(v: f32) -> char {
    let idx = (v.clamp(0.0, 1.0) * (BLOCK_CHARS.len() - 1) as f32).round() as usize;
    BLOCK_CHARS[idx.min(BLOCK_CHARS.len() - 1)]
}

/// Render a 2-D attention map `[H_patches, W_patches]` as Unicode blocks.
pub fn render_attention_map(map: &Array2<f32>, width: usize) -> String {
    let (h, w) = (map.shape()[0], map.shape()[1]);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);

    let mut out = String::new();
    for row in 0..h.min(width) {
        for col in 0..w.min(width) {
            let v = (map[[row, col]] - min) / range;
            out.push(value_to_block(v));
        }
        out.push('\n');
    }
    out
}

/// Print a metrics table for multiple models.
pub fn print_metrics_table(metrics: &[ModelMetrics]) {
    println!();
    println!("Model Comparison");
    println!("{}", "═".repeat(80));

    // Header
    print!("{:<22}", "Metric");
    for m in metrics {
        print!("{:<16}", truncate(&m.model_name, 15));
    }
    println!();
    println!("{}", "─".repeat(80));

    // Rows
    print!("{:<22}", "Repr. rank");
    for m in metrics {
        print!("{:<16}", format!("{}/{}", m.effective_rank, m.embed_dim));
    }
    println!();

    print!("{:<22}", "Dead dimensions");
    for m in metrics {
        print!("{:<16}", m.dead_dimensions);
    }
    println!();

    print!("{:<22}", "Patch entropy");
    for m in metrics {
        print!("{:<16}", format!("{:.2}", m.patch_entropy));
    }
    println!();

    print!("{:<22}", "CLS L2 norm");
    for m in metrics {
        let val = m
            .cls_l2_norm
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "N/A".into());
        print!("{:<16}", val);
    }
    println!();

    print!("{:<22}", "Top-10 var%");
    for m in metrics {
        print!("{:<16}", format!("{:.1}%", m.top10_variance_pct));
    }
    println!();

    print!("{:<22}", "Components@90%");
    for m in metrics {
        print!("{:<16}", m.components_90pct);
    }
    println!();

    println!("{}", "═".repeat(80));
}

/// Print a cross-model CLS cosine similarity matrix.
pub fn print_cls_similarity_matrix(comparisons: &[ComparisonMetrics], model_names: &[&str]) {
    println!();
    println!("Cross-model CLS cosine similarity:");
    let w = 10;
    print!("{:<12}", "");
    for name in model_names {
        print!("{:<width$}", truncate(name, w - 1), width = w);
    }
    println!();

    for &a in model_names {
        print!("{:<12}", truncate(a, 11));
        for &b in model_names {
            if a == b {
                print!("{:<width$}", "1.000", width = w);
            } else {
                let val = comparisons
                    .iter()
                    .find(|c| {
                        (c.model_a == a && c.model_b == b) || (c.model_a == b && c.model_b == a)
                    })
                    .and_then(|c| c.cls_cosine_sim)
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or_else(|| "-".into());
                print!("{:<width$}", val, width = w);
            }
        }
        println!();
    }
}

pub fn print_compare_overview(overview: &CompareOverview) {
    if !overview.model_highlights.is_empty() {
        println!();
        println!("Highlights");
        println!("{}", "═".repeat(80));
        for highlight in &overview.model_highlights {
            println!(
                "{:<28} {:<22} {}",
                truncate(&highlight.label, 27),
                truncate(&highlight.model, 21),
                highlight.value
            );
        }
        println!("{}", "═".repeat(80));
    }

    if !overview.comparison_highlights.is_empty() {
        println!();
        println!("Comparison Highlights");
        println!("{}", "═".repeat(80));
        for highlight in &overview.comparison_highlights {
            println!(
                "{:<28} {:<20} ↔ {:<20} {:.3}",
                truncate(&highlight.label, 27),
                truncate(&highlight.model_a, 19),
                truncate(&highlight.model_b, 19),
                highlight.value
            );
        }
        println!("{}", "═".repeat(80));
    }

    print_pairwise_matrix("CLS cosine similarity", &overview.cls_cosine_matrix);
    print_pairwise_matrix("Linear CKA", &overview.linear_cka_matrix);
    print_pairwise_matrix("k-NN overlap (k=10)", &overview.knn_overlap_matrix);
    print_pairwise_matrix("Mean patch correspondence", &overview.correspondence_matrix);
}

pub fn print_pairwise_matrix(title: &str, matrix: &PairwiseMatrix) {
    if matrix.len() < 2 || !matrix.has_off_diagonal_values() {
        return;
    }

    println!();
    println!("{title}:");
    let width = 12;
    print!("{:<14}", "");
    for name in &matrix.labels {
        print!("{:<width$}", truncate(name, width - 1), width = width);
    }
    println!();

    for (row_idx, name) in matrix.labels.iter().enumerate() {
        print!("{:<14}", truncate(name, 13));
        for value in &matrix.rows[row_idx] {
            let rendered = value
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "N/A".to_string());
            print!("{:<width$}", rendered, width = width);
        }
        println!();
    }
}

pub fn print_validation_summaries(summaries: &[ModelValidationSummary]) {
    println!();
    println!("Validation Summary");
    println!("{}", "═".repeat(100));
    println!(
        "{:<20} {:<12} {:<12} {:<18} Recommendation",
        "Model", "Status", "Parity", "Evidence"
    );
    println!("{}", "─".repeat(100));

    for summary in summaries {
        println!(
            "{:<20} {:<12} {:<12} {:<18} {}",
            truncate(&summary.model, 19),
            summary.status.label(),
            summary.parity.status.label(),
            truncate(&summary.evidence_timestamp, 17),
            truncate(&summary.recommendation, 44),
        );

        if !summary.caveats.is_empty() {
            println!("  caveats: {}", summary.caveats.join(" | "));
        }
        if !summary.parity.deltas.is_empty() {
            let labels = summary
                .parity
                .deltas
                .iter()
                .take(3)
                .map(|delta| delta.name.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            println!("  deltas:  {labels}");
        }
    }

    println!("{}", "═".repeat(100));
}

pub fn print_neighbors_report(report: &NeighborsReport) {
    println!();
    println!("Nearest neighbors for {}", report.query_image);
    println!("Model: {}  k={}", report.model, report.requested_k);
    println!("{}", "─".repeat(50));
    for neighbor in &report.neighbors {
        println!(
            "  {:2}. {:40} sim={:.4}",
            neighbor.rank,
            truncate(&neighbor.image, 40),
            neighbor.similarity
        );
    }
    print_dataset_processing_summary(&report.dataset_summary);
}

pub fn print_similarity_report(report: &SimilarityReport) {
    println!();
    println!(
        "Representation similarity: {} vs {}",
        report.model_a, report.model_b
    );
    println!("Dataset: {} images", report.sample_count);
    println!("{}", "═".repeat(55));

    for metric in &report.metrics {
        println!("  {:<22} {:.4}", format!("{}:", metric.label), metric.value);
    }

    if let Some(note) = &report.note {
        println!("  Mean CLS cosine sim: {note}");
    }

    print_dataset_processing_summary(&report.dataset_summary);
}

pub fn print_dataset_processing_summary(summary: &DatasetProcessingSummary) {
    println!();
    println!("Dataset Summary");
    println!("{}", "═".repeat(84));
    println!("Supported files: {}", summary.discovered);
    println!("Loaded images:   {}", summary.loaded);
    println!("Skipped images:  {}", summary.skipped);

    for skipped in &summary.skipped_examples {
        println!(
            "  skipped: {} ({})",
            truncate(&skipped.path, 36),
            truncate(&skipped.reason, 40)
        );
    }

    if summary.skipped > summary.skipped_examples.len() {
        println!(
            "  skipped: … plus {} more files",
            summary.skipped - summary.skipped_examples.len()
        );
    }

    println!("{}", "═".repeat(84));
}

pub fn print_drift_report(report: &DriftReport) {
    if let Some(summary) = &report.dataset_summary {
        print_dataset_processing_summary(summary);
    }

    let rows = report
        .drift
        .iter()
        .map(|step| {
            (
                step.from_checkpoint.clone(),
                step.to_checkpoint.clone(),
                step.linear_cka,
            )
        })
        .collect::<Vec<_>>();
    print_drift_summary(&report.checkpoint_names, &rows);
}

pub fn print_drift_summary(checkpoints: &[String], drift_rows: &[(String, String, f32)]) {
    println!();
    println!("Representation Drift");
    println!("{}", "═".repeat(84));

    if checkpoints.is_empty() {
        println!("No .onnx checkpoint files were found.");
        println!("{}", "═".repeat(84));
        return;
    }

    println!("Checkpoints: {}", checkpoints.join(" -> "));
    println!("{}", "─".repeat(84));

    if drift_rows.is_empty() {
        println!("Need at least two checkpoints to compute consecutive drift.");
        println!("{}", "═".repeat(84));
        return;
    }

    for (from, to, cka) in drift_rows {
        println!(
            "{:<26} -> {:<26} CKA={:.4}",
            truncate(from, 25),
            truncate(to, 25),
            cka
        );
    }

    let mean_cka = drift_rows.iter().map(|(_, _, cka)| cka).sum::<f32>() / drift_rows.len() as f32;
    if let Some((from, to, cka)) = drift_rows
        .iter()
        .min_by(|left, right| left.2.total_cmp(&right.2))
    {
        println!("{}", "─".repeat(84));
        println!("Mean consecutive CKA: {:.4}", mean_cka);
        println!(
            "Largest shift: {} -> {} ({:.4})",
            truncate(from, 22),
            truncate(to, 22),
            cka
        );
    }

    println!("{}", "═".repeat(84));
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_to_block() {
        assert_eq!(value_to_block(0.0), ' ');
        assert_eq!(value_to_block(1.0), '█');
    }

    #[test]
    fn test_render_attention_map_shape() {
        let map = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f32);
        let rendered = render_attention_map(&map, 8);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0].chars().count(), 4);
    }

    #[test]
    fn test_truncate_adds_ellipsis() {
        assert_eq!(truncate("checkpoint-0000001", 8), "checkpo…");
    }
}
