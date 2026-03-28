//! Self-contained interactive HTML report generation.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::dataset::DatasetProcessingSummary;
use crate::errors::VizError;
use crate::models::ModelCatalogReport;
use crate::validation::report::ModelValidationSummary;
use crate::viz::report::{
    build_compare_overview, CompareOverview, DriftReport, InspectReport, NeighborsReport,
    PairwiseMatrix, SimilarityReport,
};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct InspectHtmlAssets {
    pub pca_image: Option<String>,
    pub variance_image: Option<String>,
}

impl InspectHtmlAssets {
    fn is_empty(&self) -> bool {
        self.pca_image.is_none() && self.variance_image.is_none()
    }
}

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

pub fn write_neighbors_report(
    report: &NeighborsReport,
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_neighbors_html(report);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn write_similarity_report(
    report: &SimilarityReport,
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_similarity_html(report);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn write_drift_report(report: &DriftReport, output_path: &Path) -> Result<(), VizError> {
    let html = render_drift_html(report);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn write_model_catalog_report(
    report: &ModelCatalogReport,
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_model_catalog_html(report);
    std::fs::write(output_path, &html)
        .map_err(|e| VizError::Html(format!("Failed to write {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn write_inspect_report(report: &InspectReport, output_path: &Path) -> Result<(), VizError> {
    write_inspect_report_with_assets(report, &InspectHtmlAssets::default(), output_path)
}

pub fn write_inspect_report_with_assets(
    report: &InspectReport,
    assets: &InspectHtmlAssets,
    output_path: &Path,
) -> Result<(), VizError> {
    let html = render_inspect_html(report, assets);
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
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&comparison.model_a),
                escape_html(&comparison.model_b),
                escape_html(&comparison.alignment.summary()),
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
                render_metric_caveats(comparison),
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
    let validation_section = render_validation_section_body(
        validation,
        "No validation evidence was attached to this report.",
    );

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
  .delta-list {{ margin: 0.5rem 0 0; padding-left: 1.1rem; color: var(--muted); }}
  .delta-list li {{ margin: 0.25rem 0; }}
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

fn render_neighbors_html(report: &NeighborsReport) -> String {
    let rows = report
        .neighbors
        .iter()
        .map(|neighbor| {
            format!(
                "<tr><td>{}</td><td><code>{}</code></td><td>{:.4}</td></tr>",
                neighbor.rank,
                escape_html(&neighbor.image),
                neighbor.similarity,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let table = if rows.is_empty() {
        "<p class=\"empty-state\">No neighbors were returned for this query.</p>".to_string()
    } else {
        format!(
            "<table><thead><tr><th>Rank</th><th>Image</th><th>Cosine similarity</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    };

    let sections = vec![
        ("Top Matches", table),
        (
            "Dataset Processing",
            render_dataset_summary_html(&report.dataset_summary),
        ),
        (
            "Validation Summary",
            render_validation_section_body(
                std::slice::from_ref(&report.validation),
                "No validation evidence was attached to this report.",
            ),
        ),
    ];

    render_secondary_html(
        "Nearest Neighbors",
        &format!(
            "Query <code>{}</code> searched with model <code>{}</code> using a <strong>{}</strong> global embedding.",
            escape_html(&report.query_image),
            escape_html(&report.model),
            escape_html(report.embedding_basis.label()),
        ),
        &[
            (
                "Embedding basis",
                report.embedding_basis.label().to_string(),
            ),
            ("Requested k", report.requested_k.to_string()),
            ("Neighbors returned", report.neighbors.len().to_string()),
            ("Loaded images", report.dataset_summary.loaded.to_string()),
        ],
        &sections,
    )
}

fn render_similarity_html(report: &SimilarityReport) -> String {
    let metrics = if report.metrics.is_empty() {
        "<p class=\"empty-state\">No similarity metric was available for the selected mode.</p>"
            .to_string()
    } else {
        let rows = report
            .metrics
            .iter()
            .map(|metric| {
                format!(
                    "<tr><td>{}</td><td>{:.4}</td></tr>",
                    escape_html(&metric.label),
                    metric.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>")
    };
    let note = report
        .note
        .as_ref()
        .map(|note| format!("<p class=\"caveat\">{}</p>", escape_html(note)))
        .unwrap_or_default();

    let sections = vec![
        ("Similarity Metrics", format!("{metrics}{note}")),
        (
            "Dataset Processing",
            render_dataset_summary_html(&report.dataset_summary),
        ),
        (
            "Validation Summary",
            render_validation_section_body(
                &report.validation,
                "No validation evidence was attached to this report.",
            ),
        ),
    ];

    render_secondary_html(
        "Representation Similarity",
        &format!(
            "<code>{}</code> vs <code>{}</code> across <code>{}</code>. Dataset-level similarity metrics use <strong>{}</strong> embeddings.",
            escape_html(&report.model_a),
            escape_html(&report.model_b),
            escape_html(&report.dataset),
            escape_html(report.dataset_embedding_basis.label()),
        ),
        &[
            (
                "Dataset embedding basis",
                report.dataset_embedding_basis.label().to_string(),
            ),
            ("Requested mode", report.requested_metric.clone()),
            ("Loaded samples", report.sample_count.to_string()),
            ("Metrics reported", report.metrics.len().to_string()),
        ],
        &sections,
    )
}

fn render_drift_html(report: &DriftReport) -> String {
    let rows = if report.drift.is_empty() {
        "<p class=\"empty-state\">Need at least two checkpoints to compute consecutive drift.</p>"
            .to_string()
    } else {
        let body = report
            .drift
            .iter()
            .map(|step| {
                format!(
                    "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{:.4}</td></tr>",
                    escape_html(&step.from_checkpoint),
                    escape_html(&step.to_checkpoint),
                    step.linear_cka
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<table><thead><tr><th>From</th><th>To</th><th>Linear CKA</th></tr></thead><tbody>{body}</tbody></table>"
        )
    };
    let largest_shift = report
        .largest_shift
        .as_ref()
        .map(|step| {
            format!(
                "<p><strong>Largest shift:</strong> <code>{}</code> → <code>{}</code> ({:.4})</p>",
                escape_html(&step.from_checkpoint),
                escape_html(&step.to_checkpoint),
                step.linear_cka,
            )
        })
        .unwrap_or_else(|| {
            "<p class=\"empty-state\">No drift highlight available yet.</p>".to_string()
        });
    let summary_section = report
        .dataset_summary
        .as_ref()
        .map(render_dataset_summary_html)
        .unwrap_or_else(|| {
            "<p class=\"empty-state\">Dataset processing did not run because no checkpoints were available.</p>"
                .to_string()
        });

    let sections = vec![
        ("Consecutive Drift", rows),
        ("Highlights", largest_shift),
        ("Dataset Processing", summary_section),
        (
            "Validation Summary",
            render_validation_section_body(
                &report.validation,
                "No validation evidence was attached to this report.",
            ),
        ),
    ];

    render_secondary_html(
        "Representation Drift",
        &format!(
            "Model <code>{}</code> across checkpoints in <code>{}</code>. Consecutive drift uses <strong>{}</strong> embeddings.",
            escape_html(&report.model),
            escape_html(&report.checkpoints),
            escape_html(report.dataset_embedding_basis.label()),
        ),
        &[
            (
                "Dataset embedding basis",
                report.dataset_embedding_basis.label().to_string(),
            ),
            ("Checkpoints", report.checkpoint_names.len().to_string()),
            ("Consecutive comparisons", report.drift.len().to_string()),
            (
                "Mean CKA",
                report
                    .mean_consecutive_cka
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "N/A".to_string()),
            ),
        ],
        &sections,
    )
}

fn render_model_catalog_html(report: &ModelCatalogReport) -> String {
    let fixture_status = match (
        &report.fixture_set,
        &report.evidence_timestamp,
        &report.fixture_error,
    ) {
        (_, _, Some(error)) => format!("Unavailable: {}", escape_html(error)),
        (Some(fixture_set), Some(timestamp), None) => format!(
            "<code>{}</code> @ <code>{}</code>",
            escape_html(fixture_set),
            escape_html(timestamp),
        ),
        (Some(fixture_set), None, None) => {
            format!("<code>{}</code>", escape_html(fixture_set))
        }
        _ => "Unavailable".to_string(),
    };

    let sections = vec![
        (
            "Fixture Provenance",
            format!(
                "<p><strong>Validation fixtures:</strong> {}</p><p><strong>Evidence summary:</strong> {} approved, {} stale, {} missing, {} unverified</p>",
                fixture_status,
                report.summary.evidence.approved,
                report.summary.evidence.stale,
                report.summary.evidence.missing,
                report.summary.evidence.unverified,
            ),
        ),
        ("Model Inventory", render_model_catalog_table(report)),
    ];

    render_secondary_html(
        "Model inventory",
        "Registry availability, cache state, and validation evidence for each known integration.",
        &[
            ("Registered models", report.summary.total_models.to_string()),
            ("Ready now", report.summary.ready_models.to_string()),
            ("Cached bundles", report.summary.cached_models.to_string()),
            (
                "Approved evidence",
                report.summary.evidence.approved.to_string(),
            ),
        ],
        &sections,
    )
}

fn render_inspect_html(report: &InspectReport, assets: &InspectHtmlAssets) -> String {
    let metrics = &report.metrics;
    let variance = &report.variance_spectrum;
    let variance_rows = variance
        .ratios
        .iter()
        .zip(variance.cumulative.iter())
        .take(12)
        .enumerate()
        .map(|(index, (ratio, cumulative))| {
            let width = (ratio.clamp(0.0, 1.0) * 100.0).round();
            format!(
                "<tr><td>PC{:02}</td><td>{:.2}%</td><td>{:.2}%</td><td><div class=\"spectrum-bar-track\"><div class=\"spectrum-bar-fill\" style=\"width:{width:.0}%\"></div></div></td></tr>",
                index + 1,
                ratio * 100.0,
                cumulative * 100.0,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let variance_table = if variance_rows.is_empty() {
        "<p class=\"empty-state\">Variance spectrum was unavailable for this report.</p>"
            .to_string()
    } else {
        format!(
            "<table><thead><tr><th>Component</th><th>Variance</th><th>Cumulative</th><th>Profile</th></tr></thead><tbody>{variance_rows}</tbody></table>"
        )
    };
    let mut sections = vec![
        (
            "Representation Metrics",
            format!(
                "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>\
                 <tr><td>Patch tokens</td><td>{}</td></tr>\
                 <tr><td>Embedding dimension</td><td>{}</td></tr>\
                 <tr><td>Effective rank</td><td>{}/{}</td></tr>\
                 <tr><td>Dead dimensions</td><td>{}</td></tr>\
                 <tr><td>Patch entropy</td><td>{:.3}</td></tr>\
                 <tr><td>CLS L2 norm</td><td>{}</td></tr>\
                 <tr><td>Patch norm mean ± std</td><td>{:.2} ± {:.2}</td></tr>\
                 <tr><td>Top-10 variance concentration</td><td>{:.1}%</td></tr>\
                 </tbody></table>",
                metrics.n_patches,
                metrics.embed_dim,
                metrics.effective_rank,
                metrics.embed_dim,
                metrics.dead_dimensions,
                metrics.patch_entropy,
                metrics
                    .cls_l2_norm
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "N/A".to_string()),
                metrics.patch_norm_mean,
                metrics.patch_norm_std,
                metrics.top10_variance_pct,
            ),
        ),
        (
            "Variance Spectrum",
            format!(
                "<p>Top principal components of the patch embedding space for <code>{}</code>.</p>\
                 <div class=\"stats-grid\">\
                 <div class=\"stat-card\"><span>Components @ 90%</span><strong>{}</strong></div>\
                 <div class=\"stat-card\"><span>Components @ 99%</span><strong>{}</strong></div>\
                 <div class=\"stat-card\"><span>Top-10 concentration</span><strong>{:.1}%</strong></div>\
                 <div class=\"stat-card\"><span>Components shown</span><strong>{}</strong></div>\
                 </div>{}",
                escape_html(&report.model),
                variance.components_90pct,
                variance.components_99pct,
                variance.top10_concentration * 100.0,
                variance.ratios.len(),
                variance_table,
            ),
        ),
    ];
    if !assets.is_empty() {
        sections.push(("Visual Artefacts", render_inspect_asset_gallery(assets)));
    }
    sections.push((
        "Validation Summary",
        render_validation_section_body(
            std::slice::from_ref(&report.validation),
            "No validation evidence was attached to this report.",
        ),
    ));

    render_secondary_html(
        "Representation Inspect",
        &format!(
            "Image <code>{}</code> analysed with model <code>{}</code>.",
            escape_html(&report.image),
            escape_html(&report.model),
        ),
        &[
            ("Model", report.model.clone()),
            ("Patch tokens", metrics.n_patches.to_string()),
            ("Embed dim", metrics.embed_dim.to_string()),
            ("Effective rank", metrics.effective_rank.to_string()),
        ],
        &sections,
    )
}

fn render_inspect_asset_gallery(assets: &InspectHtmlAssets) -> String {
    let mut cards = Vec::new();

    if let Some(path) = &assets.pca_image {
        cards.push(format!(
            "<article class=\"inspect-asset-card\"><h3>PCA Projection</h3><img src=\"{}\" alt=\"Inspect PCA projection\" /><p class=\"caveat\">Patch-space RGB projection derived from the top three PCA components.</p></article>",
            escape_html(path),
        ));
    }
    if let Some(path) = &assets.variance_image {
        cards.push(format!(
            "<article class=\"inspect-asset-card\"><h3>Variance Chart</h3><img src=\"{}\" alt=\"Inspect variance spectrum chart\" /><p class=\"caveat\">Component-wise variance concentration across the inspected representation.</p></article>",
            escape_html(path),
        ));
    }

    if cards.is_empty() {
        "<p class=\"empty-state\">No inspect artefacts were generated for this report.</p>"
            .to_string()
    } else {
        format!(
            "<div class=\"inspect-asset-grid\">{}</div>",
            cards.join("\n")
        )
    }
}

fn render_validation_section_body(
    validation: &[ModelValidationSummary],
    empty_message: &str,
) -> String {
    if validation.is_empty() {
        format!(
            "<p class=\"empty-state\">{}</p>",
            escape_html(empty_message)
        )
    } else {
        let validation_rows = validation
            .iter()
            .map(render_validation_row)
            .collect::<Vec<_>>()
            .join("\n");
        format!("<div class=\"validation-grid\">{validation_rows}</div>")
    }
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
    let parity_deltas = if summary.parity.deltas.is_empty() {
        String::new()
    } else {
        let items = summary
            .parity
            .deltas
            .iter()
            .take(5)
            .map(|delta| {
                format!(
                    "<li><code>{}</code>: {} vs {}</li>",
                    escape_html(&delta.name),
                    escape_html(&delta.observed),
                    escape_html(&delta.expected),
                )
            })
            .collect::<Vec<_>>()
            .join("");
        format!("<ul class=\"delta-list\">{items}</ul>")
    };

    format!(
        "<article class=\"validation-card\"><div style=\"display:flex;justify-content:space-between;align-items:center;gap:1rem\"><strong>{}</strong><span class=\"badge {}\">{}</span></div><p>{}</p><p><strong>Preprocess:</strong> {}</p><p><strong>Tensor semantics:</strong> {}</p><p><strong>Parity:</strong> {}</p>{}{}</article>",
        escape_html(&summary.model),
        summary.status.label(),
        summary.status.label(),
        escape_html(&summary.recommendation),
        escape_html(&summary.preprocess.summary),
        tensor_summary,
        escape_html(&summary.parity.summary),
        caveats,
        parity_deltas
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
      <th>Patch alignment</th>
      <th>CLS cosine sim</th>
      <th>Linear CKA</th>
      <th>k-NN overlap (k=10)</th>
      <th>Mean patch corr.</th>
      <th>Metric caveats</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
            </table>"
        )
    }
}

fn render_model_catalog_table(report: &ModelCatalogReport) -> String {
    if report.entries.is_empty() {
        return "<p class=\"empty-state\">No models are registered.</p>".to_string();
    }

    let rows = report
        .entries
        .iter()
        .map(|entry| {
            format!(
                "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&entry.name),
                escape_html(&entry.phase),
                escape_html(&entry.availability_status.to_string()),
                escape_html(entry.evidence_status.label()),
                escape_html(entry.cache_status.label()),
                escape_html(&entry.verification_label),
                escape_html(&entry.method.to_string()),
                entry.params_m,
                render_model_catalog_details(entry),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "<table><thead><tr><th>Name</th><th>Phase</th><th>Status</th><th>Evidence</th><th>Cache</th><th>Verify</th><th>Method</th><th>Params (M)</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>"
    )
}

fn render_model_catalog_details(entry: &crate::models::ModelInventoryEntry) -> String {
    let mut parts = vec![
        format!(
            "<p><strong>Availability:</strong> {}</p>",
            escape_html(&entry.availability_note),
        ),
        format!(
            "<p><strong>Architecture:</strong> {} | {}x{} | dim {} | {} layers / {} heads</p>",
            escape_html(&entry.architecture),
            entry.input_size,
            entry.input_size,
            entry.embed_dim,
            entry.num_layers,
            entry.num_heads,
        ),
        format!(
            "<p><strong>Evidence:</strong> {} [{} @ {}]</p>",
            escape_html(&entry.evidence_summary),
            escape_html(&entry.approved_fixture_set),
            escape_html(&entry.approved_evidence_timestamp),
        ),
        format!(
            "<p><strong>Cache:</strong> {}</p>",
            escape_html(&entry.cache_summary),
        ),
    ];

    if let Some(note) = &entry.verification_note {
        parts.push(format!(
            "<p><strong>Verification note:</strong> {}</p>",
            escape_html(note),
        ));
    }

    if !entry.evidence_details.is_empty() {
        let items = entry
            .evidence_details
            .iter()
            .map(|detail| format!("<li>{}</li>", escape_html(detail)))
            .collect::<Vec<_>>()
            .join("");
        parts.push(format!(
            "<p><strong>Evidence details:</strong></p><ul>{items}</ul>"
        ));
    }

    if !entry.artifacts.is_empty() {
        let items = entry
            .artifacts
            .iter()
            .map(|artifact| {
                format!(
                    "<li><code>{}</code><br/><a href=\"{}\">{}</a></li>",
                    escape_html(&artifact.relative_path),
                    escape_html(&artifact.url),
                    escape_html(&artifact.url),
                )
            })
            .collect::<Vec<_>>()
            .join("");
        parts.push(format!(
            "<p><strong>Artifacts:</strong></p><ul>{items}</ul>"
        ));
    }

    parts.join("")
}

fn render_metric_caveats(comparison: &ComparisonMetrics) -> String {
    let mut caveats = Vec::new();
    if let Some(note) = &comparison.alignment.note {
        caveats.push(format!("Patch alignment: {}", escape_html(note)));
    }
    caveats.extend(comparison.metric_caveats.iter().map(|caveat| {
        format!(
            "{}: {}",
            escape_html(&caveat.label),
            escape_html(&caveat.reason)
        )
    }));

    if caveats.is_empty() {
        "None".to_string()
    } else {
        caveats.join("<br/>")
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

fn render_secondary_html(
    title: &str,
    subtitle: &str,
    stats: &[(&str, String)],
    sections: &[(&str, String)],
) -> String {
    let stats_html = render_secondary_stats(stats);
    let sections_html = sections
        .iter()
        .map(|(heading, body)| {
            format!(
                "<div class=\"panel\"><h2>{}</h2>{}</div>",
                escape_html(heading),
                body
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>latent-inspector: {}</title>
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
  }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem; background: radial-gradient(circle at top, #182032 0%, var(--bg) 45%); color: var(--text); }}
  h1, h2 {{ color: var(--accent); }}
  .panel {{ background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.25rem; margin: 1rem 0 1.5rem; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; margin-top: 1rem; }}
  .stat-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 1rem; background: rgba(255,255,255,0.03); }}
  .stat-card strong {{ display: block; font-size: 1.4rem; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th {{ background: var(--panel); padding: 0.5rem 1rem; text-align: left; color: var(--accent); border-bottom: 2px solid var(--border); }}
  td {{ padding: 0.4rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; }}
  tr:hover td {{ background: #161b22; }}
  .badge {{ font-size: 0.8em; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: #1f2937; text-transform: uppercase; letter-spacing: 0.04em; }}
  .badge.validated {{ color: var(--ok); border-color: rgba(63,185,80,0.35); }}
  .badge.partial, .badge.stale {{ color: var(--warn); border-color: rgba(210,153,34,0.35); }}
  .badge.failed, .badge.unverified {{ color: var(--bad); border-color: rgba(248,81,73,0.35); }}
  .validation-grid {{ display: grid; gap: 1rem; }}
  .validation-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.02); }}
  .inspect-asset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
  .inspect-asset-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.02); }}
  .inspect-asset-card h3 {{ margin-bottom: 0.75rem; }}
  .inspect-asset-card img {{ width: 100%; display: block; border-radius: 10px; border: 1px solid var(--border); background: rgba(255,255,255,0.02); }}
  .delta-list {{ margin: 0.5rem 0 0; padding-left: 1.1rem; color: var(--muted); }}
  .delta-list li {{ margin: 0.25rem 0; }}
  .spectrum-bar-track {{ width: 100%; min-width: 140px; height: 0.75rem; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }}
  .spectrum-bar-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, #58a6ff, #3fb950); }}
  .empty-state, .caveat, li {{ color: var(--muted); }}
  ul {{ margin: 0.6rem 0 0; padding-left: 1.2rem; }}
  code {{ color: #c9d1d9; }}
</style>
</head>
<body>
<h1>latent-inspector</h1>
<div class="panel">
  <h2>{}</h2>
  <p>{}</p>
  {}
</div>
{}
</body>
</html>"#,
        escape_html(title),
        escape_html(title),
        subtitle,
        stats_html,
        sections_html
    )
}

fn render_secondary_stats(stats: &[(&str, String)]) -> String {
    if stats.is_empty() {
        return String::new();
    }

    let cards = stats
        .iter()
        .map(|(label, value)| {
            format!(
                "<div class=\"stat-card\"><span>{}</span><strong>{}</strong></div>",
                escape_html(label),
                escape_html(value),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("<div class=\"stats-grid\">{cards}</div>")
}

fn render_dataset_summary_html(summary: &DatasetProcessingSummary) -> String {
    let skipped = if summary.skipped_examples.is_empty() {
        "<p class=\"empty-state\">No skipped images.</p>".to_string()
    } else {
        let items = summary
            .skipped_examples
            .iter()
            .map(|item| {
                format!(
                    "<li><code>{}</code>: {}</li>",
                    escape_html(&item.path),
                    escape_html(&item.reason),
                )
            })
            .collect::<Vec<_>>()
            .join("");
        format!("<ul>{items}</ul>")
    };

    format!(
        "<p><strong>Supported files:</strong> {}</p><p><strong>Loaded images:</strong> {}</p><p><strong>Skipped images:</strong> {}</p>{}",
        summary.discovered,
        summary.loaded,
        summary.skipped,
        skipped,
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
    use crate::dataset::{DatasetProcessingSummary, SkippedImage};
    use crate::models::{build_model_catalog, EvidenceStatus};
    use crate::validation::report::{
        CheckSummary, ModelValidationSummary, ParityValidationSummary, TensorValidationSummary,
        ValidationStatus,
    };
    use crate::viz::report::{
        DriftReport, DriftStep, InspectReport, NeighborMatch, NeighborsReport,
        SimilarityMetricValue, SimilarityReport, VarianceSpectrumReport,
    };
    use tempfile::tempdir;

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

    fn inspect_report(model: &str) -> InspectReport {
        InspectReport {
            image: "fixture.png".into(),
            model: model.into(),
            metrics: ModelMetrics {
                model_name: model.into(),
                n_patches: 256,
                embed_dim: 1024,
                effective_rank: 212,
                dead_dimensions: 6,
                patch_entropy: 5.47,
                cls_l2_norm: Some(14.3),
                patch_norm_mean: 6.1,
                patch_norm_std: 0.8,
                top10_variance_pct: 28.5,
                components_90pct: 41,
            },
            validation: validation_summary(model),
            variance_spectrum: VarianceSpectrumReport {
                ratios: vec![0.28, 0.19, 0.13, 0.09],
                cumulative: vec![0.28, 0.47, 0.60, 0.69],
                components_90pct: 41,
                components_99pct: 88,
                top10_concentration: 0.62,
            },
        }
    }

    fn model_catalog_report() -> ModelCatalogReport {
        build_model_catalog(None)
    }

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
            alignment: crate::analysis::ComparisonAlignment {
                patch_count_a: 256,
                patch_count_b: 256,
                compared_patch_count: 256,
                note: None,
            },
            cls_cosine_sim: Some(0.55),
            linear_cka: 0.71,
            knn_overlap_k10: 0.32,
            mean_patch_correspondence: Some(0.48),
            metric_caveats: Vec::new(),
        }];

        let html = render_html("test.jpg", &metrics, &comparisons, &[]);
        assert!(html.contains("Pairwise matrices"));
        assert!(html.contains("Linear CKA"));
        assert!(html.contains("Strongest CKA alignment"));
    }

    #[test]
    fn test_html_renders_comparison_alignment_and_caveats() {
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
                model_name: "mae".into(),
                n_patches: 196,
                embed_dim: 1024,
                effective_rank: 192,
                dead_dimensions: 8,
                patch_entropy: 4.8,
                cls_l2_norm: None,
                patch_norm_mean: 5.0,
                patch_norm_std: 1.1,
                top10_variance_pct: 35.0,
                components_90pct: 36,
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
            linear_cka: 0.71,
            knn_overlap_k10: 0.32,
            mean_patch_correspondence: Some(0.48),
            metric_caveats: vec![crate::analysis::MetricCaveat {
                key: "cls_cosine_sim".into(),
                label: "CLS cosine similarity".into(),
                reason: "Unavailable because only one model exposes a CLS token.".into(),
            }],
        }];

        let html = render_html("test.jpg", &metrics, &comparisons, &[]);
        assert!(html.contains("Patch alignment"));
        assert!(html.contains("196 shared patches (from 256 vs 196)"));
        assert!(html.contains("Metric caveats"));
        assert!(html.contains("only one model exposes a CLS token"));
    }

    #[test]
    fn test_render_inspect_html_includes_variance_spectrum_and_validation() {
        let report = inspect_report("dinov2-vit-l14");
        let assets = InspectHtmlAssets {
            pca_image: Some("dinov2-vit-l14_pca.png".into()),
            variance_image: Some("dinov2-vit-l14_variance.png".into()),
        };
        let html = render_inspect_html(&report, &assets);

        assert!(html.contains("Representation Inspect"));
        assert!(html.contains("Variance Spectrum"));
        assert!(html.contains("Components @ 99%"));
        assert!(html.contains("PC01"));
        assert!(html.contains("Visual Artefacts"));
        assert!(html.contains("dinov2-vit-l14_pca.png"));
        assert!(html.contains("Validation Summary"));
        assert!(html.contains("dinov2-vit-l14"));
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

    #[test]
    fn test_write_inspect_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inspect.html");
        let report = inspect_report("dinov2-vit-l14");

        write_inspect_report(&report, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Variance Spectrum"));
        assert!(content.contains("fixture.png"));
    }

    #[test]
    fn test_write_inspect_report_with_assets() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inspect-assets.html");
        let report = inspect_report("dinov2-vit-l14");
        let assets = InspectHtmlAssets {
            pca_image: Some("dinov2-vit-l14_pca.png".into()),
            variance_image: Some("dinov2-vit-l14_variance.png".into()),
        };

        write_inspect_report_with_assets(&report, &assets, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Visual Artefacts"));
        assert!(content.contains("dinov2-vit-l14_variance.png"));
    }

    #[test]
    fn test_write_neighbors_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("neighbors.html");
        let report = NeighborsReport {
            query_image: "query.png".into(),
            dataset: "dataset".into(),
            model: "dinov2".into(),
            embedding_basis: crate::extract::EmbeddingBasis::ClsToken,
            requested_k: 2,
            dataset_summary: DatasetProcessingSummary {
                discovered: 3,
                loaded: 2,
                skipped: 1,
                skipped_examples: vec![SkippedImage {
                    path: "broken.png".into(),
                    reason: "decode failed".into(),
                }],
            },
            neighbors: vec![NeighborMatch {
                rank: 1,
                image: "class-a/leaf".into(),
                similarity: 0.91,
            }],
            validation: validation_summary("dinov2"),
        };

        write_neighbors_report(&report, &path).unwrap();
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Nearest Neighbors"));
        assert!(html.contains("class-a/leaf"));
        assert!(html.contains("Validation Summary"));
        assert!(html.contains("CLS token"));
        assert!(html.contains("Reference parity matches approved evidence."));
    }

    #[test]
    fn test_write_similarity_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("similarity.html");
        let report = SimilarityReport {
            model_a: "dinov2".into(),
            model_b: "clip".into(),
            dataset: "dataset".into(),
            dataset_embedding_basis: crate::extract::EmbeddingBasis::MeanPatch,
            requested_metric: "all".into(),
            sample_count: 2,
            dataset_summary: DatasetProcessingSummary {
                discovered: 2,
                loaded: 2,
                skipped: 0,
                skipped_examples: Vec::new(),
            },
            metrics: vec![SimilarityMetricValue {
                key: "linear_cka".into(),
                label: "Linear CKA".into(),
                value: 0.77,
            }],
            note: Some("N/A (CLS tokens unavailable)".into()),
            validation: vec![validation_summary("dinov2"), validation_summary("clip")],
        };

        write_similarity_report(&report, &path).unwrap();
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Representation Similarity"));
        assert!(html.contains("Linear CKA"));
        assert!(html.contains("CLS tokens unavailable"));
        assert!(html.contains("Validation Summary"));
        assert!(html.contains("Mean patch"));
        assert!(html.contains("dinov2"));
        assert!(html.contains("clip"));
    }

    #[test]
    fn test_write_drift_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("drift.html");
        let report = DriftReport::new(
            "dinov2",
            "checkpoints",
            "dataset",
            crate::extract::EmbeddingBasis::MeanPatch,
            vec!["step-1".into(), "step-2".into()],
            Some(DatasetProcessingSummary {
                discovered: 2,
                loaded: 2,
                skipped: 0,
                skipped_examples: Vec::new(),
            }),
            vec![DriftStep {
                from_checkpoint: "step-1".into(),
                to_checkpoint: "step-2".into(),
                linear_cka: 0.88,
            }],
            vec![validation_summary("step-1"), validation_summary("step-2")],
        );

        write_drift_report(&report, &path).unwrap();
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Representation Drift"));
        assert!(html.contains("step-1"));
        assert!(html.contains("Largest shift"));
        assert!(html.contains("Validation Summary"));
        assert!(html.contains("Mean patch"));
    }

    #[test]
    fn test_write_model_catalog_report() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("models.html");
        let report = model_catalog_report();

        write_model_catalog_report(&report, &path).unwrap();

        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Model inventory"));
        assert!(html.contains("Fixture Provenance"));
        assert!(html.contains("dinov2-vit-l14"));
        assert!(html.contains("approved"));
    }

    #[test]
    fn test_model_catalog_report_includes_summary_counts() {
        let report = model_catalog_report();
        let html = render_model_catalog_html(&report);

        assert_eq!(report.summary.evidence.approved, 1);
        assert_eq!(report.summary.evidence.unverified, 5);
        assert!(html.contains("Registered models"));
        assert!(html.contains("Approved evidence"));
        assert!(html.contains(EvidenceStatus::Approved.label()));
    }
}
