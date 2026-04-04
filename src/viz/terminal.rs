//! Terminal rendering with Unicode-first output and ASCII fallbacks.

use crate::analysis::{ComparisonMetrics, ModelMetrics};
use crate::dataset::DatasetProcessingSummary;
use crate::models::{ModelCatalogReport, ModelDownloadReport, ModelReadinessStatus};
use crate::validation::report::ModelValidationSummary;
use crate::viz::report::{
    CompareOverview, DriftReport, NeighborsReport, PairwiseMatrix, PairwiseMetricSupport,
    ProfileReport, SimilarityReport,
};
use ndarray::Array2;
use std::io::IsTerminal;

const UNICODE_BLOCK_CHARS: &[char] = &[' ', '░', '▒', '▓', '█'];
const ASCII_BLOCK_CHARS: &[char] = &[' ', '.', ':', '=', '#'];
const FORCE_ASCII_ENV: &str = "LATENT_INSPECTOR_FORCE_ASCII";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalGlyphs {
    Unicode,
    Ascii,
}

impl TerminalGlyphs {
    fn current() -> Self {
        if env_truthy(FORCE_ASCII_ENV)
            || term_is_dumb()
            || !std::io::stdout().is_terminal()
            || !locale_supports_unicode()
        {
            Self::Ascii
        } else {
            Self::Unicode
        }
    }

    fn heavy_rule(self) -> char {
        match self {
            TerminalGlyphs::Unicode => '═',
            TerminalGlyphs::Ascii => '=',
        }
    }

    fn light_rule(self) -> char {
        match self {
            TerminalGlyphs::Unicode => '─',
            TerminalGlyphs::Ascii => '-',
        }
    }

    fn pair_separator(self) -> &'static str {
        match self {
            TerminalGlyphs::Unicode => " ↔ ",
            TerminalGlyphs::Ascii => " <-> ",
        }
    }

    fn dimension_separator(self) -> &'static str {
        match self {
            TerminalGlyphs::Unicode => "×",
            TerminalGlyphs::Ascii => "x",
        }
    }

    fn ellipsis(self) -> &'static str {
        match self {
            TerminalGlyphs::Unicode => "…",
            TerminalGlyphs::Ascii => "...",
        }
    }

    fn plus_minus_separator(self) -> &'static str {
        match self {
            TerminalGlyphs::Unicode => "±",
            TerminalGlyphs::Ascii => "+/-",
        }
    }

    fn block_chars(self) -> &'static [char] {
        match self {
            TerminalGlyphs::Unicode => UNICODE_BLOCK_CHARS,
            TerminalGlyphs::Ascii => ASCII_BLOCK_CHARS,
        }
    }

    fn bar_char(self) -> char {
        match self {
            TerminalGlyphs::Unicode => '█',
            TerminalGlyphs::Ascii => '#',
        }
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var_os(name)
        .map(|value| {
            matches!(
                value.to_string_lossy().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn term_is_dumb() -> bool {
    std::env::var_os("TERM")
        .map(|value| value.to_string_lossy().eq_ignore_ascii_case("dumb"))
        .unwrap_or(false)
}

fn locale_supports_unicode() -> bool {
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .filter_map(std::env::var_os)
        .find(|value| !value.is_empty());

    let Some(locale) = locale else {
        return true;
    };

    let upper = locale.to_string_lossy().to_ascii_uppercase();
    upper.contains("UTF-8") || upper.contains("UTF8")
}

fn repeat_char(ch: char, width: usize) -> String {
    std::iter::repeat(ch).take(width).collect()
}

/// Render a heavy horizontal divider appropriate for the current terminal.
pub fn heavy_rule(width: usize) -> String {
    repeat_char(TerminalGlyphs::current().heavy_rule(), width)
}

pub(crate) fn light_rule(width: usize) -> String {
    repeat_char(TerminalGlyphs::current().light_rule(), width)
}

pub(crate) fn pair_separator() -> &'static str {
    TerminalGlyphs::current().pair_separator()
}

pub(crate) fn dimension_separator() -> &'static str {
    TerminalGlyphs::current().dimension_separator()
}

/// Render a plus/minus separator appropriate for the current terminal.
pub fn plus_minus_separator() -> &'static str {
    TerminalGlyphs::current().plus_minus_separator()
}

/// Render a solid progress bar segment appropriate for the current terminal.
pub fn bar(width: usize) -> String {
    repeat_char(TerminalGlyphs::current().bar_char(), width)
}

fn value_to_block_with_glyphs(v: f32, glyphs: TerminalGlyphs) -> char {
    let block_chars = glyphs.block_chars();
    let idx = (v.clamp(0.0, 1.0) * (block_chars.len() - 1) as f32).round() as usize;
    block_chars[idx.min(block_chars.len() - 1)]
}

/// Render a 2-D attention map `[H_patches, W_patches]` as Unicode blocks.
pub fn render_attention_map(map: &Array2<f32>, width: usize) -> String {
    render_attention_map_with_glyphs(map, width, TerminalGlyphs::current())
}

fn render_attention_map_with_glyphs(
    map: &Array2<f32>,
    width: usize,
    glyphs: TerminalGlyphs,
) -> String {
    let (h, w) = (map.shape()[0], map.shape()[1]);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);

    let mut out = String::new();
    for row in 0..h.min(width) {
        for col in 0..w.min(width) {
            let v = (map[[row, col]] - min) / range;
            out.push(value_to_block_with_glyphs(v, glyphs));
        }
        out.push('\n');
    }
    out
}

/// Print a metrics table for multiple models.
pub fn print_metrics_table(metrics: &[ModelMetrics]) {
    println!();
    println!("Model Comparison");
    println!("{}", heavy_rule(80));

    // Header
    print!("{:<22}", "Metric");
    for m in metrics {
        print!("{:<16}", truncate(&m.model_name, 15));
    }
    println!();
    println!("{}", light_rule(80));

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

    print!("{:<22}", "Attention Gini");
    for m in metrics {
        let val = m
            .attention_gini
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".into());
        print!("{:<16}", val);
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

    print!("{:<22}", "Patch isotropy");
    for m in metrics {
        print!("{:<16}", format!("{:.3}", m.patch_isotropy));
    }
    println!();

    print!("{:<22}", "Patch uniformity");
    for m in metrics {
        print!("{:<16}", format!("{:.3}", m.patch_uniformity));
    }
    println!();

    print!("{:<22}", "Spatial coherence");
    for m in metrics {
        print!(
            "{:<16}",
            m.spatial_coherence
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "N/A".to_string())
        );
    }
    println!();

    print!("{:<22}", "RankMe");
    for m in metrics {
        print!("{:<16}", format!("{:.1}", m.rankme));
    }
    println!();

    print!("{:<22}", "Spectral decay");
    for m in metrics {
        print!(
            "{:<16}",
            m.spectral_decay
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "N/A".to_string())
        );
    }
    println!();

    println!("{}", heavy_rule(80));
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
        println!("{}", heavy_rule(80));
        for highlight in &overview.model_highlights {
            println!(
                "{:<28} {:<22} {}",
                truncate(&highlight.label, 27),
                truncate(&highlight.model, 21),
                highlight.value
            );
        }
        println!("{}", heavy_rule(80));
    }

    if !overview.comparison_highlights.is_empty() {
        println!();
        println!("Comparison Highlights");
        println!("{}", heavy_rule(80));
        for highlight in &overview.comparison_highlights {
            println!(
                "{:<28} {}{}{} {:.3}",
                truncate(&highlight.label, 27),
                truncate(&highlight.model_a, 19),
                pair_separator(),
                truncate(&highlight.model_b, 19),
                highlight.value
            );
        }
        println!("{}", heavy_rule(80));
    }

    print_pairwise_matrix(
        "CLS cosine similarity",
        &overview.cls_cosine_matrix,
        &overview.cls_cosine_support,
    );
    print_pairwise_matrix(
        "Linear CKA",
        &overview.linear_cka_matrix,
        &overview.linear_cka_support,
    );
    print_pairwise_matrix(
        "k-NN overlap (k=10)",
        &overview.knn_overlap_matrix,
        &overview.knn_overlap_support,
    );
    print_pairwise_matrix(
        "Mean patch correspondence",
        &overview.correspondence_matrix,
        &overview.correspondence_support,
    );
}

pub fn print_comparison_caveats(comparisons: &[ComparisonMetrics]) {
    let caveated = comparisons
        .iter()
        .filter(|comparison| comparison.has_caveats())
        .collect::<Vec<_>>();
    if caveated.is_empty() {
        return;
    }

    println!();
    println!("Comparison Caveats");
    println!("{}", heavy_rule(100));
    for comparison in caveated {
        println!(
            "{}{}{}",
            truncate(&comparison.model_a, 32),
            pair_separator(),
            truncate(&comparison.model_b, 32),
        );
        for line in comparison.caveat_lines() {
            println!("  {line}");
        }
        println!("{}", light_rule(100));
    }
}

pub fn print_pairwise_matrix(
    title: &str,
    matrix: &PairwiseMatrix,
    support: &PairwiseMetricSupport,
) {
    if matrix.len() < 2 {
        return;
    }

    println!();
    println!("{title}:");
    println!(
        "Comparable model pairs: {}/{}",
        support.supported_pairs, support.total_pairs
    );
    if support.unavailable_pairs > 0 {
        println!("Unavailable pairs: {}", support.unavailable_pairs);
        for reason in &support.unavailable_reasons {
            println!("  x{} {}", reason.count, reason.reason);
        }
    }
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

pub fn print_model_catalog(report: &ModelCatalogReport, verbose: bool) {
    println!();
    println!("Available models ({})", report.summary.total_models);
    println!("{}", heavy_rule(148));
    println!(
        "{:<20} {:<10} {:<24} {:<12} {:<12} {:<10} {:<8} {:<12} {:>10}",
        "Name",
        "Status",
        "Readiness",
        "Runtime",
        "Evidence",
        "Cache",
        "Verify",
        "Method",
        "Params (M)"
    );
    println!("{}", light_rule(148));

    for entry in &report.entries {
        println!(
            "{:<20} {:<10} {:<24} {:<12} {:<12} {:<10} {:<8} {:<12} {:>10}",
            truncate(&entry.name, 19),
            entry.availability_status.to_string(),
            truncate(entry.readiness_status.label(), 23),
            truncate(entry.runtime_support.label(), 11),
            entry.evidence_status.label(),
            entry.cache_status.label(),
            truncate(&entry.verification_label, 7),
            entry.method,
            entry.params_m,
        );

        if verbose {
            println!("    Phase: {}", entry.phase);
            println!("    Note: {}", entry.availability_note);
            println!(
                "    Readiness: {} ({})",
                entry.readiness_summary,
                entry.readiness_status.label(),
            );
            for step in &entry.next_steps {
                println!("      next: {}", step);
            }
            println!("    Runtime: {}", entry.runtime_summary);
            println!("    Arch: {}", entry.architecture);
            println!(
                "    Input: {}{}{}",
                entry.input_size,
                dimension_separator(),
                entry.input_size
            );
            println!("    Embed dim: {}", entry.embed_dim);
            println!(
                "    Layers: {}, Heads: {}",
                entry.num_layers, entry.num_heads
            );
            println!(
                "    Evidence: {} [{} @ {}]",
                entry.evidence_summary,
                entry.approved_fixture_set,
                entry.approved_evidence_timestamp,
            );
            for detail in &entry.evidence_details {
                println!("      detail: {}", detail);
            }
            println!("    Cache: {}", entry.cache_summary);
            println!(
                "    Artifacts: {} total, {} usable, {} verified, {} pending verification, {} missing, {} invalid, {} unusable, {} unknown",
                entry.artifact_summary.total,
                entry.artifact_summary.usable,
                entry.artifact_summary.verified,
                entry.artifact_summary.pending_verification,
                entry.artifact_summary.missing,
                entry.artifact_summary.invalid,
                entry.artifact_summary.unusable,
                entry.artifact_summary.unknown,
            );
            if let Some(note) = &entry.verification_note {
                println!("    Verify note: {}", note);
            }
            for artifact in &entry.artifacts {
                println!(
                    "    Artifact: {} [{} | {}]",
                    artifact.relative_path,
                    artifact.cache_status.label(),
                    artifact.verification_label,
                );
                println!("      Path: {}", artifact.absolute_path);
                if let Some(byte_size) = artifact.byte_size {
                    println!("      Bytes: {}", byte_size);
                }
                println!("      Cache: {}", artifact.cache_summary);
                if let Some(note) = &artifact.verification_note {
                    println!("      Verify note: {}", note);
                }
                println!("      URL: {}", artifact.url);
            }
        } else if !matches!(
            entry.readiness_status,
            ModelReadinessStatus::Ready | ModelReadinessStatus::Planned
        ) {
            println!("  readiness: {}", entry.readiness_summary);
        }
    }

    println!("{}", heavy_rule(148));

    let fixture_summary = if let Some(error) = &report.fixture_error {
        format!("unavailable ({error})")
    } else {
        match (&report.fixture_set, &report.evidence_timestamp) {
            (Some(fixture_set), Some(timestamp)) => format!("{fixture_set} @ {timestamp}"),
            (Some(fixture_set), None) => fixture_set.clone(),
            _ => "unavailable".to_string(),
        }
    };

    println!("Validation fixtures: {fixture_summary}");
    println!(
        "Evidence summary: {} approved, {} stale, {} missing, {} unverified",
        report.summary.evidence.approved,
        report.summary.evidence.stale,
        report.summary.evidence.missing,
        report.summary.evidence.unverified,
    );
    println!(
        "Readiness summary: {} ready, {} need download, {} need evidence refresh, {} need validation, {} planned, {} blocked",
        report.summary.readiness.ready,
        report.summary.readiness.needs_download,
        report.summary.readiness.needs_evidence_refresh,
        report.summary.readiness.needs_validation,
        report.summary.readiness.planned,
        report.summary.readiness.blocked,
    );
    println!(
        "Artifact summary: {} total, {} usable, {} verified, {} pending verification, {} missing, {} invalid, {} unusable, {} unknown",
        report.summary.artifacts.total,
        report.summary.artifacts.usable,
        report.summary.artifacts.verified,
        report.summary.artifacts.pending_verification,
        report.summary.artifacts.missing,
        report.summary.artifacts.invalid,
        report.summary.artifacts.unusable,
        report.summary.artifacts.unknown,
    );
}

pub fn print_model_download_report(report: &ModelDownloadReport) {
    println!();
    println!("Model download");
    println!("{}", heavy_rule(116));
    println!("Model: {}", report.model);
    println!("Action: {}", report.action.label());
    println!("Summary: {}", report.summary);
    println!(
        "Readiness: {} ({})",
        report.entry.readiness_summary,
        report.entry.readiness_status.label(),
    );
    if !report.entry.next_steps.is_empty() {
        println!("Next steps:");
        for step in &report.entry.next_steps {
            println!("  - {}", step);
        }
    }

    if !report.artifact_changes.is_empty() {
        println!("Artifact changes:");
        for artifact in &report.artifact_changes {
            println!(
                "  - {}: {} -> {}{}",
                artifact.relative_path,
                artifact.previous_status.label(),
                artifact.current_status.label(),
                if artifact.downloaded {
                    " (downloaded)"
                } else if artifact.repaired {
                    " (repaired)"
                } else {
                    ""
                },
            );
            if let Some(byte_size) = artifact.byte_size {
                println!("      bytes: {}", byte_size);
            }
            println!("      detail: {}", artifact.cache_summary);
        }
    }

    println!("{}", heavy_rule(116));
}

pub fn print_validation_summaries(summaries: &[ModelValidationSummary]) {
    println!();
    println!("Validation Summary");
    println!("{}", heavy_rule(116));
    println!(
        "{:<20} {:<12} {:<12} {:<12} {:<18} Recommendation",
        "Model", "Status", "Backend", "Parity", "Evidence"
    );
    println!("{}", light_rule(116));

    for summary in summaries {
        println!(
            "{:<20} {:<12} {:<12} {:<12} {:<18} {}",
            truncate(&summary.model, 19),
            summary.status.label(),
            truncate(summary.backend.kind.label(), 11),
            summary.parity.status.label(),
            truncate(&summary.evidence_timestamp, 17),
            truncate(&summary.recommendation, 42),
        );

        for line in validation_detail_lines(summary) {
            println!("{line}");
        }
    }

    println!("{}", heavy_rule(116));
}

fn summarize_tensor_checks(summary: &ModelValidationSummary) -> String {
    summary
        .tensors
        .iter()
        .map(|tensor| format!("{}={}", tensor.name, tensor.status.label()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_parity_delta(delta: &crate::validation::report::ParitySignalDelta) -> String {
    match (delta.abs_diff, delta.tolerance) {
        (Some(abs_diff), Some(tolerance)) => format!(
            "{} obs={} exp={} diff={:.6} tol={:.6}",
            delta.name, delta.observed, delta.expected, abs_diff, tolerance
        ),
        _ => format!(
            "{} obs={} exp={}",
            delta.name, delta.observed, delta.expected
        ),
    }
}

fn validation_detail_lines(summary: &ModelValidationSummary) -> Vec<String> {
    let mut lines = vec![
        format!(
            "  evidence: fixture-set={} artifact={}",
            summary.parity.fixture_set.as_deref().unwrap_or("n/a"),
            truncate(summary.parity.artifact_id.as_deref().unwrap_or("n/a"), 78),
        ),
        format!(
            "  checks:   preprocess={} tensors={} parity={} (checked {}, drifted {})",
            summary.preprocess.status.label(),
            summarize_tensor_checks(summary),
            summary.parity.status.label(),
            summary.parity.checked_signals,
            summary.parity.drifted_signals,
        ),
    ];

    if !summary.backend.status.is_validated() {
        lines.push(format!("  backend: {}", summary.backend.summary));
    }
    if !summary.caveats.is_empty() {
        lines.push(format!("  caveats: {}", summary.caveats.join(" | ")));
    }
    if !summary.parity.drifted_fixtures.is_empty() {
        lines.push(format!(
            "  fixtures: {}",
            summary
                .parity
                .drifted_fixtures
                .iter()
                .take(2)
                .map(|fixture| format!(
                    "{} ({})",
                    fixture.fixture_id,
                    truncate(&fixture.signals.join(", "), 54)
                ))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    if !summary.parity.deltas.is_empty() {
        lines.push(format!(
            "  deltas:  {}",
            summary
                .parity
                .deltas
                .iter()
                .take(2)
                .map(format_parity_delta)
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }

    lines
}

pub fn print_neighbors_report(report: &NeighborsReport) {
    println!();
    println!("Nearest neighbors for {}", report.query_image);
    println!("Model: {}  k={}", report.model, report.requested_k);
    println!("Embedding basis: {}", report.embedding_basis.label());
    println!("{}", light_rule(50));
    for neighbor in &report.neighbors {
        println!(
            "  {:2}. {:40} sim={:.4}",
            neighbor.rank,
            truncate(&neighbor.image, 40),
            neighbor.similarity
        );
    }
    print_dataset_processing_summary(&report.dataset_summary);
    print_validation_summaries(std::slice::from_ref(&report.validation));
}

pub fn print_similarity_report(report: &SimilarityReport) {
    println!();
    println!(
        "Representation similarity: {} vs {}",
        report.model_a, report.model_b
    );
    println!("Dataset: {} images", report.sample_count);
    println!(
        "Dataset embedding basis: {}",
        report.dataset_embedding_basis.label()
    );
    println!("{}", heavy_rule(55));

    for metric in &report.metrics {
        println!("  {:<22} {:.4}", format!("{}:", metric.label), metric.value);
    }

    if let Some(note) = &report.note {
        println!("  Mean CLS cosine sim: {note}");
    }

    print_dataset_processing_summary(&report.dataset_summary);
    if !report.validation.is_empty() {
        print_validation_summaries(&report.validation);
    }
}

pub fn print_dataset_processing_summary(summary: &DatasetProcessingSummary) {
    println!();
    println!("Dataset Summary");
    println!("{}", heavy_rule(84));
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
            "  skipped: {} plus {} more files",
            TerminalGlyphs::current().ellipsis(),
            summary.skipped - summary.skipped_examples.len()
        );
    }

    println!("{}", heavy_rule(84));
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
    print_drift_summary(
        &report.checkpoint_names,
        report.dataset_embedding_basis.label(),
        &rows,
    );
    if !report.validation.is_empty() {
        print_validation_summaries(&report.validation);
    }
}

pub fn print_drift_summary(
    checkpoints: &[String],
    dataset_embedding_basis: &str,
    drift_rows: &[(String, String, f32)],
) {
    println!();
    println!("Representation Drift");
    println!("{}", heavy_rule(84));

    if checkpoints.is_empty() {
        println!("No .onnx checkpoint files were found.");
        println!("{}", heavy_rule(84));
        return;
    }

    println!("Checkpoints: {}", checkpoints.join(" -> "));
    println!("Dataset embedding basis: {dataset_embedding_basis}");
    println!("{}", light_rule(84));

    if drift_rows.is_empty() {
        println!("Need at least two checkpoints to compute consecutive drift.");
        println!("{}", heavy_rule(84));
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
        println!("{}", light_rule(84));
        println!("Mean consecutive CKA: {:.4}", mean_cka);
        println!(
            "Largest shift: {} -> {} ({:.4})",
            truncate(from, 22),
            truncate(to, 22),
            cka
        );
    }

    println!("{}", heavy_rule(84));
}

pub fn print_profile_report(report: &ProfileReport) {
    println!();
    println!("Representation Profile: {}", report.model);
    println!(
        "Dataset: {} ({} images)",
        report.dataset, report.sample_count
    );
    println!("Embedding dimension: {}", report.embed_dim);
    println!("Embedding basis: {}", report.embedding_basis.label());
    println!("{}", heavy_rule(68));

    println!();
    println!("Space-Level Metrics");
    println!("{}", light_rule(68));
    println!(
        "  {:<32} {:.4}",
        "Isotropy (cosine):", report.space_metrics.isotropy_cosine
    );
    println!(
        "  {:<32} {:.4}",
        "Isotropy (partition):", report.space_metrics.isotropy_partition
    );
    println!(
        "  {:<32} {:.4}",
        "Uniformity:", report.space_metrics.uniformity
    );
    println!(
        "  {:<32} {:.1}",
        "Intrinsic dimensionality:", report.space_metrics.intrinsic_dimensionality
    );

    if !report.aggregate_metrics.is_empty() {
        println!();
        println!(
            "Per-Image Metric Aggregates ({} images)",
            report.sample_count
        );
        println!("{}", light_rule(68));
        println!(
            "  {:<28} {:>8} {:>8} {:>8} {:>8}",
            "Metric", "Mean", "Std", "Min", "Max"
        );
        println!("  {}", light_rule(64));
        for agg in &report.aggregate_metrics {
            println!(
                "  {:<28} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                truncate(&format!("{}:", agg.label), 27),
                agg.mean,
                agg.std,
                agg.min,
                agg.max
            );
        }
    }

    print_dataset_processing_summary(&report.dataset_summary);
    print_validation_summaries(std::slice::from_ref(&report.validation));
}

fn truncate(s: &str, max: usize) -> String {
    truncate_with_glyphs(s, max, TerminalGlyphs::current())
}

fn truncate_with_glyphs(s: &str, max: usize, glyphs: TerminalGlyphs) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }

    let ellipsis = glyphs.ellipsis();
    let ellipsis_len = ellipsis.chars().count();
    if max <= ellipsis_len {
        return ellipsis.chars().take(max).collect();
    }

    let mut truncated = s.chars().take(max - ellipsis_len).collect::<String>();
    truncated.push_str(ellipsis);
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::InferenceBackend;
    use crate::validation::report::{
        CheckSummary, ModelValidationSummary, ParitySignalDelta, ParityValidationSummary,
        TensorValidationSummary, ValidationStatus,
    };

    #[test]
    fn test_value_to_block() {
        assert_eq!(
            value_to_block_with_glyphs(0.0, TerminalGlyphs::Unicode),
            ' '
        );
        assert_eq!(
            value_to_block_with_glyphs(1.0, TerminalGlyphs::Unicode),
            '█'
        );
        assert_eq!(value_to_block_with_glyphs(1.0, TerminalGlyphs::Ascii), '#');
    }

    #[test]
    fn test_render_attention_map_shape() {
        let map = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f32);
        let rendered = render_attention_map_with_glyphs(&map, 8, TerminalGlyphs::Ascii);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0].chars().count(), 4);
        assert!(rendered.is_ascii());
    }

    #[test]
    fn test_truncate_adds_ellipsis() {
        assert_eq!(
            truncate_with_glyphs("checkpoint-0000001", 8, TerminalGlyphs::Unicode),
            "checkpo…"
        );
        assert_eq!(
            truncate_with_glyphs("checkpoint-0000001", 8, TerminalGlyphs::Ascii),
            "check..."
        );
    }

    #[test]
    fn validation_detail_lines_include_fixture_drift_preview() {
        let parity = ParityValidationSummary {
            status: ValidationStatus::Failed,
            summary: "Reference parity drift detected.".into(),
            artifact_id: Some("dinov2-vit-l14:standard:2026-03-27T12:00:00Z".into()),
            fixture_set: Some("standard".into()),
            checked_signals: 0,
            drifted_signals: 0,
            deltas: vec![ParitySignalDelta {
                name: "fixtures.gradient-224.patch_signature[0]".into(),
                observed: "9.900000".into(),
                expected: "0.100000".into(),
                abs_diff: Some(9.8),
                tolerance: Some(0.001),
            }],
            drifted_fixtures: Vec::new(),
        }
        .with_diagnostics(73);
        let summary = ModelValidationSummary::from_checks(
            "dinov2-vit-l14",
            "2026-03-27T12:00:00Z",
            CheckSummary::validated("Preprocess matches contract."),
            vec![TensorValidationSummary {
                name: "last_hidden_state".into(),
                role: "patch embeddings".into(),
                status: ValidationStatus::Validated,
                summary: "Tensor semantics match.".into(),
            }],
            parity,
        )
        .with_backend(InferenceBackend::Stub);

        let lines = validation_detail_lines(&summary);

        assert!(lines
            .iter()
            .any(|line| line.contains("fixture-set=standard")));
        assert!(lines
            .iter()
            .any(|line| line.contains("checked 73, drifted 1")));
        assert!(lines
            .iter()
            .any(|line| line.contains("fixtures: gradient-224")));
        assert!(lines.iter().any(|line| line.contains("patch_signature[0]")));
    }
}
