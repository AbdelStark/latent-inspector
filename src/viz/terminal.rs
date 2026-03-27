//! Terminal rendering using Unicode block characters and ANSI colors.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::validation::report::ModelValidationSummary;
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
    }

    println!("{}", "═".repeat(100));
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
}
