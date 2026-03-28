use crate::analysis::{compute_comparison, compute_metrics, ComparisonMetrics, ModelMetrics};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::assets;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::OutputFormat;
use clap::Args;
use rayon::prelude::*;
use serde_json::json;
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
            let session = ModelSession::load_for_analysis(name)?;
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
            crate::viz::terminal::print_comparison_caveats(&report.comparisons);
            crate::viz::terminal::print_validation_summaries(&report.validation);
        }
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("compare.json");
                crate::viz::json::write_compare_report(&report, &path)?;
                OutputArtifactManifest::new("compare", OutputFormat::Json)
                    .with_primary_artifact("compare.json")
                    .with_context(compare_manifest_context(&args))
                    .with_summary(compare_manifest_summary(&report))
                    .add_artifact("compare.json", ArtifactKind::Json, "Compare report")
                    .with_validation(&report.validation)
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_compare_report(&report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("compare_output"));
            std::fs::create_dir_all(&outdir)?;
            let pca_artifacts = render_pca_artifacts(&outputs, &outdir)?;
            let heatmap_artifacts = render_pairwise_heatmap_artifacts(&report.overview, &outdir)?;
            let source_preview = assets::write_preview_image(
                &img,
                &outdir,
                "input_image.png",
                "Input image",
                "Source image used for this compare report.",
            )?;
            let assets = crate::viz::html::CompareHtmlAssets {
                source_images: vec![source_preview],
                pca_images: pca_artifacts,
                heatmaps: heatmap_artifacts,
            };
            let image_name = args
                .image
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image");
            crate::viz::json::write_compare_report(&report, &outdir.join("compare.json"))?;
            crate::viz::html::write_report_with_validation_and_assets(
                image_name,
                &report.metrics,
                &report.comparisons,
                &report.validation,
                &assets,
                &outdir.join("report.html"),
            )?;
            let mut manifest = OutputArtifactManifest::new("compare", OutputFormat::Html)
                .with_primary_artifact("report.html")
                .with_context(compare_manifest_context(&args))
                .with_summary(compare_manifest_summary(&report))
                .add_artifact("report.html", ArtifactKind::Html, "Compare report")
                .add_artifact("compare.json", ArtifactKind::Json, "Compare report data")
                .with_validation(&report.validation);
            manifest = add_visual_artifacts(manifest, &assets.source_images);
            manifest = add_visual_artifacts(manifest, &assets.pca_images);
            manifest = add_visual_artifacts(manifest, &assets.heatmaps);
            manifest.write_to_dir(&outdir)?;
            println!("Report written to {}/report.html", outdir.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("compare_output"));
            std::fs::create_dir_all(&outdir)?;
            let pca_artifacts = render_pca_artifacts(&outputs, &outdir)?;
            let heatmap_artifacts = render_pairwise_heatmap_artifacts(&report.overview, &outdir)?;
            let mut manifest = OutputArtifactManifest::new("compare", OutputFormat::Png)
                .with_context(compare_manifest_context(&args))
                .with_summary(compare_manifest_summary(&report))
                .with_validation(&report.validation);
            manifest = add_visual_artifacts(manifest, &pca_artifacts);
            manifest = add_visual_artifacts(manifest, &heatmap_artifacts);
            manifest.write_to_dir(&outdir)?;
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

fn render_pca_artifacts(
    outputs: &[(String, ExtractedFeatures)],
    outdir: &std::path::Path,
) -> Result<Vec<crate::viz::html::VisualAsset>, Error> {
    let mut artifacts = Vec::with_capacity(outputs.len());
    for (name, feat) in outputs {
        let pca_result = crate::analysis::pca(&feat.patch_tokens, 3, 300)?;
        let projected = crate::analysis::transform(&feat.patch_tokens, &pca_result);
        let grid = patch_grid_side(feat.n_patches, &format!("{name} PCA rendering"))?;
        let filename = format!("{}_pca.png", assets::slugify_filename(name));
        let path = outdir.join(&filename);
        crate::viz::png::save_pca_rgb(&projected, grid, &path)?;
        artifacts.push(assets::visual_asset(
            filename,
            format!("{name} PCA projection"),
            format!(
                "Patch-space RGB projection derived from the top three PCA components for {name}."
            ),
        ));
    }
    Ok(artifacts)
}

fn patch_grid_side(patch_count: usize, context: &str) -> Result<usize, Error> {
    let grid = (patch_count as f64).sqrt().round() as usize;
    if grid > 0 && grid.checked_mul(grid) == Some(patch_count) {
        Ok(grid)
    } else {
        Err(crate::errors::AnalysisError::InvalidPatchGrid {
            context: context.to_string(),
            patch_count,
        }
        .into())
    }
}

fn render_pairwise_heatmap_artifacts(
    overview: &crate::viz::report::CompareOverview,
    outdir: &std::path::Path,
) -> Result<Vec<crate::viz::html::VisualAsset>, Error> {
    let heatmaps = [
        (
            "cls_cosine.png",
            "CLS cosine heatmap",
            "Cross-model CLS cosine similarity matrix.",
            &overview.cls_cosine_matrix,
        ),
        (
            "linear_cka.png",
            "Linear CKA heatmap",
            "Cross-model representation alignment measured with linear CKA.",
            &overview.linear_cka_matrix,
        ),
        (
            "knn_overlap_k10.png",
            "k-NN overlap heatmap",
            "Cross-model neighborhood agreement using k=10.",
            &overview.knn_overlap_matrix,
        ),
        (
            "patch_correspondence.png",
            "Patch correspondence heatmap",
            "Direct patch-space correspondence for models with matching embedding dimensions.",
            &overview.correspondence_matrix,
        ),
    ];

    let mut artifacts = Vec::new();
    for (filename, title, description, matrix) in heatmaps {
        if matrix.len() < 2 || !matrix.has_off_diagonal_values() {
            continue;
        }
        crate::viz::png::save_pairwise_heatmap(matrix, &outdir.join(filename))?;
        artifacts.push(assets::visual_asset(filename, title, description));
    }

    Ok(artifacts)
}

fn add_visual_artifacts(
    mut manifest: OutputArtifactManifest,
    artifacts: &[crate::viz::html::VisualAsset],
) -> OutputArtifactManifest {
    for artifact in artifacts {
        manifest = manifest.add_artifact(
            artifact.path.clone(),
            ArtifactKind::Png,
            artifact.description.clone(),
        );
    }
    manifest
}

fn compare_manifest_context(args: &CompareArgs) -> serde_json::Value {
    json!({
        "image": args.image.display().to_string(),
        "models": args.models,
    })
}

fn compare_manifest_summary(report: &crate::viz::report::CompareReport) -> serde_json::Value {
    json!({
        "model_count": report.metrics.len(),
        "comparison_count": report.comparisons.len(),
        "model_highlights": report.overview.model_highlights,
        "comparison_highlights": report.overview.comparison_highlights,
        "pairwise_support": {
            "cls_cosine": report.overview.cls_cosine_support,
            "linear_cka": report.overview.linear_cka_support,
            "knn_overlap_k10": report.overview.knn_overlap_support,
            "mean_patch_correspondence": report.overview.correspondence_support,
        },
    })
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
        assert_eq!(
            crate::viz::assets::slugify_filename("dinov2-vit-l14#2"),
            "dinov2-vit-l14_2"
        );
    }

    #[test]
    fn patch_grid_side_rejects_non_square_patch_counts() {
        let error = patch_grid_side(15, "compare PCA").unwrap_err();

        assert!(error
            .to_string()
            .contains("does not form a square patch grid"));
    }
}
