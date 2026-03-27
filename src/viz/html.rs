//! Self-contained interactive HTML report generation.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::errors::VizError;
use crate::validation::report::ModelValidationSummary;
use crate::viz::report::{build_compare_overview, CompareOverview, PairwiseMatrix};
use std::path::Path;

/// Generate a self-contained HTML report and write it to `output_path`.
pub fn write_report(
    image_name: &str,
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
    output_path: &Path,
) -> Result<(), VizError> {
    write_report_with_validation(image_name, metrics, comparisons, &[], output_path)
}

pub fn write_report_with_validation(
    image_name: &str,
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
    validation: &[ModelValidationSummary],
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_html(image_name, metrics, comparisons, validation);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn write_validation_report(
    validation: &[ModelValidationSummary],
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_validation_html(validation);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

fn render_html(
    image_name: &str,
    metrics: &[ModelMetrics],
    comparisons: &[ComparisonMetrics],
    validation: &[ModelValidationSummary],
) -> String {
    let overview = build_compare_overview(metrics, comparisons);
    let comparison_rows = comparisons
        .iter()
        .map(|comparison| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td></tr>",
                escape_html(&comparison.model_a),
                escape_html(&comparison.model_b),
                comparison
                    .cls_cosine_sim
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "N/A".to_string()),
                comparison.linear_cka,
                comparison.knn_overlap_k10,
                comparison
                    .mean_patch_correspondence
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "N/A".to_string()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let metrics_rows = metrics
        .iter()
        .map(|metric| {
            format!(
                "<tr><td>{}</td><td>{}/{}</td><td>{}</td><td>{:.2}</td><td>{}</td><td>{:.1}%</td><td>{}</td></tr>",
                escape_html(&metric.model_name),
                metric.effective_rank,
                metric.embed_dim,
                metric.dead_dimensions,
                metric.patch_entropy,
                metric
                    .cls_l2_norm
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "N/A".to_string()),
                metric.top10_variance_pct,
                metric.components_90pct,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let validated_count = validation
        .iter()
        .filter(|summary| summary.status.label() == "validated")
        .count();
    let highlights = render_overview_cards(&overview);
    let comparison_table = render_comparison_table(&comparison_rows);
    let matrix_sections = render_matrix_sections(&overview);
    let validation_section = if validation.is_empty() {
        "<p class=\"empty-state\">No validation evidence was attached to this report.</p>"
            .to_string()
    } else {
        let validation_rows = validation
            .iter()
            .map(render_validation_row)
            .collect::<Vec<_>>()
            .join("\n");
        format!("<div class=\"validation-grid\">{validation_rows}</div>")
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>latent-inspector: {image_name}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0d1117;
    --panel: #161b22;
    --panel-2: #11161d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #79c0ff;
    --ok: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
    --border: #30363d;
    --chip: #1b2230;
  }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem; background: radial-gradient(circle at top, #182032 0%, var(--bg) 45%); color: var(--text); }}
  h1, h2, h3 {{ color: var(--accent); }}
  h3 {{ margin-top: 0; }}
  .panel {{ background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.25rem; margin: 1rem 0 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th {{ background: var(--panel); padding: 0.5rem 1rem; text-align: left; color: var(--accent); border-bottom: 2px solid var(--border); }}
  td {{ padding: 0.4rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; }}
  tr:hover td {{ background: #161b22; }}
  .badge {{ font-size: 0.8em; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: #1f2937; text-transform: uppercase; letter-spacing: 0.04em; }}
  .badge.validated {{ color: var(--ok); border-color: rgba(63,185,80,0.35); }}
  .badge.partial, .badge.stale {{ color: var(--warn); border-color: rgba(210,153,34,0.35); }}
  .badge.failed, .badge.unverified {{ color: var(--bad); border-color: rgba(248,81,73,0.35); }}
  .caveat {{ color: var(--muted); margin: 0.3rem 0 0; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; margin-top: 1rem; }}
  .stat-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 1rem; background: rgba(255,255,255,0.03); }}
  .stat-card strong {{ display: block; font-size: 1.4rem; margin-top: 0.25rem; }}
  .chip-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.8rem; }}
  .chip-card {{ border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem 1rem; background: var(--chip); }}
  .chip-card p {{ margin: 0.3rem 0 0; color: var(--muted); }}
  .matrix-grid {{ display: grid; gap: 1rem; }}
  .matrix-card {{ border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem 1rem; background: rgba(255,255,255,0.02); }}
  .validation-grid {{ display: grid; gap: 1rem; }}
  .validation-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.02); }}
  .empty-state {{ color: var(--muted); margin: 0; }}
  code {{ color: #c9d1d9; }}
</style>
</head>
<body>
<h1>latent-inspector</h1>
<div class="panel">
  <p>Image: <code>{image_name}</code></p>
  <div class="stats-grid">
    <div class="stat-card"><span>Models analysed</span><strong>{}</strong></div>
    <div class="stat-card"><span>Pairwise comparisons</span><strong>{}</strong></div>
    <div class="stat-card"><span>Validated reports</span><strong>{}</strong></div>
  </div>
</div>

<div class="panel">
<h2>Highlights</h2>
{}
</div>

<div class="panel">
<h2>Per-model metrics</h2>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Repr. rank</th>
      <th>Dead dims</th>
      <th>Patch entropy</th>
      <th>CLS L2 norm</th>
      <th>Top-10 var%</th>
      <th>Components@90%</th>
    </tr>
  </thead>
  <tbody>
    {metrics_rows}
  </tbody>
</table>
</div>

<div class="panel">
<h2>Cross-model comparison</h2>
{}
</div>

<div class="panel">
<h2>Pairwise matrices</h2>
{}
</div>

<div class="panel">
<h2>Validation Summary</h2>
{}
</div>

<footer style="margin-top:3rem;color:#8b949e;font-size:0.8em">
  Generated by <a href="https://github.com/AbdelStark/latent-inspector" style="color:#79c0ff">latent-inspector</a>
</footer>
</body>
</html>"#,
        metrics.len(),
        comparisons.len(),
        validated_count,
        highlights,
        comparison_table,
        matrix_sections,
        validation_section,
    )
}

fn render_validation_html(validation: &[ModelValidationSummary]) -> String {
    render_html("validation-run", &[], &[], validation)
}

fn render_validation_row(summary: &ModelValidationSummary) -> String {
    let tensor_summary = summary
        .tensors
        .iter()
        .map(|tensor| {
            format!(
                "{}: {}",
                escape_html(&tensor.name),
                escape_html(&tensor.summary)
            )
        })
        .collect::<Vec<_>>()
        .join("<br>");
    let caveats = if summary.caveats.is_empty() {
        "<p class=\"caveat\">No open caveats.</p>".to_string()
    } else {
        summary
            .caveats
            .iter()
            .map(|caveat| format!("<p class=\"caveat\">{}</p>", escape_html(caveat)))
            .collect::<Vec<_>>()
            .join("")
    };

    format!(
        "<article class=\"validation-card\"><div style=\"display:flex;justify-content:space-between;align-items:center;gap:1rem\"><strong>{}</strong><span class=\"badge {}\">{}</span></div><p>{}</p><p><strong>Preprocess:</strong> {}</p><p><strong>Tensor semantics:</strong> {}</p><p><strong>Parity:</strong> {}</p>{}</article>",
        escape_html(&summary.model),
        summary.status.label(),
        summary.status.label(),
        escape_html(&summary.recommendation),
        escape_html(&summary.preprocess.summary),
        tensor_summary,
        escape_html(&summary.parity.summary),
        caveats
    )
}

fn render_overview_cards(overview: &CompareOverview) -> String {
    let mut cards = overview
        .model_highlights
        .iter()
        .map(|highlight| {
            format!(
                "<article class=\"chip-card\"><strong>{}</strong><p>{}: {}</p></article>",
                escape_html(&highlight.label),
                escape_html(&highlight.model),
                escape_html(&highlight.value),
            )
        })
        .collect::<Vec<_>>();

    cards.extend(overview.comparison_highlights.iter().map(|highlight| {
        format!(
            "<article class=\"chip-card\"><strong>{}</strong><p>{} ↔ {}: {:.3}</p></article>",
            escape_html(&highlight.label),
            escape_html(&highlight.model_a),
            escape_html(&highlight.model_b),
            highlight.value,
        )
    }));

    if cards.is_empty() {
        "<p class=\"empty-state\">No summary highlights were generated for this report.</p>"
            .to_string()
    } else {
        format!("<div class=\"chip-grid\">{}</div>", cards.join("\n"))
    }
}

fn render_matrix_sections(overview: &CompareOverview) -> String {
    let matrices = [
        ("CLS cosine similarity", &overview.cls_cosine_matrix),
        ("Linear CKA", &overview.linear_cka_matrix),
        ("k-NN overlap (k=10)", &overview.knn_overlap_matrix),
        ("Mean patch correspondence", &overview.correspondence_matrix),
    ];

    let cards = matrices
        .into_iter()
        .filter(|(_, matrix)| matrix.len() >= 2 && matrix.has_off_diagonal_values())
        .map(|(title, matrix)| {
            format!(
                "<article class=\"matrix-card\"><h3>{}</h3>{}</article>",
                escape_html(title),
                render_matrix_table(matrix),
            )
        })
        .collect::<Vec<_>>();

    if cards.is_empty() {
        "<p class=\"empty-state\">Pairwise matrices require at least two comparable models.</p>"
            .to_string()
    } else {
        format!("<div class=\"matrix-grid\">{}</div>", cards.join("\n"))
    }
}

fn render_comparison_table(rows: &str) -> String {
    if rows.is_empty() {
        "<p class=\"empty-state\">No pairwise comparisons were generated.</p>".to_string()
    } else {
        format!(
            "<table>
  <thead>
    <tr>
      <th>Model A</th>
      <th>Model B</th>
      <th>CLS cosine sim</th>
      <th>Linear CKA</th>
      <th>k-NN overlap (k=10)</th>
      <th>Mean patch corr.</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>"
        )
    }
}

fn render_matrix_table(matrix: &PairwiseMatrix) -> String {
    let header = matrix
        .labels
        .iter()
        .map(|label| format!("<th>{}</th>", escape_html(label)))
        .collect::<Vec<_>>()
        .join("");
    let rows = matrix
        .rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let cells = row
                .iter()
                .map(|value| {
                    format!(
                        "<td>{}</td>",
                        value
                            .map(|value| format!("{value:.3}"))
                            .unwrap_or_else(|| "N/A".to_string())
                    )
                })
                .collect::<Vec<_>>()
                .join("");
            format!(
                "<tr><th>{}</th>{}</tr>",
                escape_html(&matrix.labels[row_idx]),
                cells
            )
        })
        .collect::<Vec<_>>()
        .join("");

    format!(
        "<table><thead><tr><th></th>{}</tr></thead><tbody>{}</tbody></table>",
        header, rows
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_html_contains_title() {
        let html = render_html("test.jpg", &[], &[], &[]);
        assert!(html.contains("test.jpg"));
    }

    #[test]
    fn test_html_contains_pairwise_matrices_when_comparisons_exist() {
        let metrics = vec![
            ModelMetrics {
                model_name: "dinov2".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 256,
                dead_dimensions: 4,
                patch_entropy: 5.4,
                cls_l2_norm: Some(12.0),
                patch_norm_mean: 6.0,
                patch_norm_std: 1.0,
                top10_variance_pct: 20.0,
                components_90pct: 48,
            },
            ModelMetrics {
                model_name: "clip".into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 192,
                dead_dimensions: 8,
                patch_entropy: 4.8,
                cls_l2_norm: Some(10.0),
                patch_norm_mean: 5.0,
                patch_norm_std: 1.1,
                top10_variance_pct: 35.0,
                components_90pct: 36,
            },
        ];
        let comparisons = vec![ComparisonMetrics {
            model_a: "dinov2".into(),
            model_b: "clip".into(),
            cls_cosine_sim: Some(0.55),
            linear_cka: 0.71,
            knn_overlap_k10: 0.32,
            mean_patch_correspondence: Some(0.48),
        }];

        let html = render_html("test.jpg", &metrics, &comparisons, &[]);
        assert!(html.contains("Pairwise matrices"));
        assert!(html.contains("Linear CKA"));
        assert!(html.contains("Strongest CKA alignment"));
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn test_write_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("report.html");
        write_report("img.jpg", &[], &[], &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
    }
}
