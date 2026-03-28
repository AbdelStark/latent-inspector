use crate::analysis::{compute_comparison, compute_metrics, ComparisonMetrics, ModelMetrics};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::OutputFormat;
use clap::Args;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::info;

#[derive(Args, Debug)]
pub struct CompareArgs {
    /// Path to the input image.
    pub image: PathBuf,

    /// Comma-separated list of model names (e.g. "dinov2-vit-l14,clip-vit-l14").
    #[arg(short, long, value_delimiter = ',', default_values_t = vec!["dinov2-vit-l14".to_string()])]
    pub models: Vec<String>,

    /// Output directory for PNG/JSON/HTML artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,
}

pub fn run(args: CompareArgs) -> Result<(), Error> {
    info!("Comparing {} models on {:?}", args.models.len(), args.image);

    // Load and preprocess image
    let img = image::open(&args.image)?;
    let display_labels = disambiguate_labels(&args.models);

    // Load all model sessions (parallel)
    let mut sessions: Vec<(String, String, ModelSession)> = args
        .models
        .iter()
        .zip(display_labels.iter())
        .map(|(name, display_label)| {
            let session = ModelSession::load(name)?;
            Ok((display_label.clone(), name.clone(), session))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let validation_summaries = sessions
        .iter_mut()
        .map(|(display_label, _, session)| {
            let mut summary = summarize_session_or_unverified(session, None);
            summary.model = display_label.clone();
            summary
        })
        .collect::<Vec<_>>();

    // Run inference in parallel
    let outputs: Vec<(String, ExtractedFeatures)> = sessions
        .par_iter_mut()
        .map(
            |(display_label, _, session): &mut (String, String, ModelSession)| {
                info!("Running inference for {display_label}");
                let output = session.infer(&img)?;
                let features = ExtractedFeatures::from_output(output)?;
                Ok((display_label.clone(), features))
            },
        )
        .collect::<Result<Vec<_>, Error>>()?;

    // Compute per-model metrics
    let metrics: Vec<ModelMetrics> = outputs
        .iter()
        .map(|(display_label, feat)| compute_metrics(feat, display_label))
        .collect::<Result<Vec<_>, _>>()?;

    // Compute pairwise comparisons
    let mut comparisons: Vec<ComparisonMetrics> = Vec::new();
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let (name_a, feat_a) = &outputs[i];
            let (name_b, feat_b) = &outputs[j];
            let cmp = compute_comparison(feat_a, feat_b, name_a, name_b)?;
            comparisons.push(cmp);
        }
    }

    let report = crate::viz::report::build_compare_report(
        args.image.display().to_string(),
        args.models.clone(),
        metrics,
        comparisons,
        validation_summaries,
    );

    // Render output
    match args.format {
        OutputFormat::Terminal => {
            crate::viz::terminal::print_metrics_table(&report.metrics);
            crate::viz::terminal::print_compare_overview(&report.overview);
            crate::viz::terminal::print_validation_summaries(&report.validation);
        }
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("compare.json");
                crate::viz::json::write_compare_report(&report, &path)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_compare_report(&report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("compare_output"));
            std::fs::create_dir_all(&outdir)?;
            let image_name = args
                .image
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image");
            crate::viz::html::write_report_with_validation(
                image_name,
                &report.metrics,
                &report.comparisons,
                &report.validation,
                &outdir.join("report.html"),
            )?;
            println!("Report written to {}/report.html", outdir.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("compare_output"));
            std::fs::create_dir_all(&outdir)?;
            // PCA RGB images
            for (name, feat) in &outputs {
                let pca_result = crate::analysis::pca(&feat.patch_tokens, 3, 300)?;
                let projected = crate::analysis::transform(&feat.patch_tokens, &pca_result);
                let grid = (feat.n_patches as f32).sqrt() as usize;
                let path = outdir.join(format!("{}_pca.png", slugify(name)));
                crate::viz::png::save_pca_rgb(&projected, grid, &path)?;
            }
            save_pairwise_heatmaps(&outdir, &report.overview)?;
            println!("PNG outputs saved to {}", outdir.display());
        }
    }

    Ok(())
}

fn disambiguate_labels(models: &[String]) -> Vec<String> {
    let mut totals: HashMap<&str, usize> = HashMap::new();
    for model in models {
        *totals.entry(model.as_str()).or_insert(0) += 1;
    }

    let mut seen: HashMap<&str, usize> = HashMap::new();
    models
        .iter()
        .map(|model| {
            let count = totals.get(model.as_str()).copied().unwrap_or(1);
            if count == 1 {
                return model.clone();
            }

            let entry = seen.entry(model.as_str()).or_insert(0);
            *entry += 1;
            format!("{model}#{}", *entry)
        })
        .collect()
}

fn slugify(label: &str) -> String {
    let mut slug = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if matches!(ch, '-' | '_') {
            slug.push(ch);
        } else {
            slug.push('_');
        }
    }
    slug.trim_matches('_').to_string()
}

fn save_pairwise_heatmaps(
    outdir: &std::path::Path,
    overview: &crate::viz::report::CompareOverview,
) -> Result<(), Error> {
    let heatmaps = [
        ("cls_cosine", &overview.cls_cosine_matrix),
        ("linear_cka", &overview.linear_cka_matrix),
        ("knn_overlap_k10", &overview.knn_overlap_matrix),
        ("patch_correspondence", &overview.correspondence_matrix),
    ];

    for (name, matrix) in heatmaps {
        if matrix.len() >= 2 && matrix.has_off_diagonal_values() {
            crate::viz::png::save_pairwise_heatmap(matrix, &outdir.join(format!("{name}.png")))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_models_receive_stable_suffixes() {
        let labels = disambiguate_labels(&[
            "dinov2-vit-l14".to_string(),
            "clip-vit-l14".to_string(),
            "dinov2-vit-l14".to_string(),
        ]);

        assert_eq!(
            labels,
            vec![
                "dinov2-vit-l14#1".to_string(),
                "clip-vit-l14".to_string(),
                "dinov2-vit-l14#2".to_string()
            ]
        );
    }

    #[test]
    fn slugify_replaces_non_filename_characters() {
        assert_eq!(slugify("dinov2-vit-l14#2"), "dinov2-vit-l14_2");
    }
}
