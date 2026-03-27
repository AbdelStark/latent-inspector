//! Self-contained interactive HTML report generation.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::errors::VizError;
use crate::validation::report::ModelValidationSummary;
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
    let metrics_rows = metrics
        .iter()
        .map(|m| {
            format!(
                "<tr><td>{}</td><td>{}/{}</td><td>{}</td><td>{:.2}</td><td>{}</td><td>{:.1}%</td></tr>",
                escape_html(&m.model_name),
                m.effective_rank,
                m.embed_dim,
                m.dead_dimensions,
                m.patch_entropy,
                m.cls_l2_norm.map(|v| format!("{:.1}", v)).unwrap_or("N/A".into()),
                m.top10_variance_pct,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let comparison_rows = comparisons
        .iter()
        .map(|c| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                escape_html(&c.model_a),
                escape_html(&c.model_b),
                c.cls_cosine_sim.map(|v| format!("{:.3}", v)).unwrap_or("N/A".into()),
                c.linear_cka,
                c.knn_overlap_k10,
                c.mean_patch_correspondence
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or("N/A".into()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let validation_rows = validation
        .iter()
        .map(render_validation_row)
        .collect::<Vec<_>>()
        .join("\n");

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
  }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem; background: radial-gradient(circle at top, #182032 0%, var(--bg) 45%); color: var(--text); }}
  h1, h2 {{ color: var(--accent); }}
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
  .validation-grid {{ display: grid; gap: 1rem; }}
  .validation-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.02); }}
</style>
</head>
<body>
<h1>latent-inspector</h1>
<div class="panel">
  <p>Image: <code>{image_name}</code></p>
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
    </tr>
  </thead>
  <tbody>
    {metrics_rows}
  </tbody>
</table>
</div>

<div class="panel">
<h2>Cross-model comparison</h2>
<table>
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
    {comparison_rows}
  </tbody>
</table>
</div>

<div class="panel">
<h2>Validation Summary</h2>
<div class="validation-grid">
{validation_rows}
</div>
</div>

<footer style="margin-top:3rem;color:#8b949e;font-size:0.8em">
  Generated by <a href="https://github.com/AbdelStark/latent-inspector" style="color:#79c0ff">latent-inspector</a>
</footer>
</body>
</html>"#
    )
}

fn render_validation_html(validation: &[ModelValidationSummary]) -> String {
    render_html("validation-run", &[], &[], validation)
}

fn render_validation_row(summary: &ModelValidationSummary) -> String {
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
        "<article class=\"validation-card\"><div style=\"display:flex;justify-content:space-between;align-items:center;gap:1rem\"><strong>{}</strong><span class=\"badge {}\">{}</span></div><p>{}</p><p><strong>Preprocess:</strong> {}</p><p><strong>Parity:</strong> {}</p>{}</article>",
        escape_html(&summary.model),
        summary.status.label(),
        summary.status.label(),
        escape_html(&summary.recommendation),
        escape_html(&summary.preprocess.summary),
        escape_html(&summary.parity.summary),
        caveats
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
