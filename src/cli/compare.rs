use crate::analysis::{compute_comparison, compute_metrics, ComparisonMetrics, ModelMetrics};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::viz::OutputFormat;
use clap::Args;
use rayon::prelude::*;
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

    // Load all model sessions (parallel)
    let sessions: Vec<(String, ModelSession)> = args
        .models
        .iter()
        .map(|name| {
            let session = ModelSession::load(name)?;
            Ok((name.clone(), session))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Run inference in parallel
    let outputs: Vec<(String, ExtractedFeatures)> = sessions
        .par_iter()
        .map(|(name, session): &(String, ModelSession)| {
            info!("Running inference for {name}");
            let output = session.infer(&img)?;
            let features = ExtractedFeatures::from_output(output)?;
            Ok((name.clone(), features))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Compute per-model metrics
    let metrics: Vec<ModelMetrics> = outputs
        .iter()
        .map(|(name, feat)| compute_metrics(feat, name))
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

    // Render output
    let model_name_refs: Vec<&str> = args.models.iter().map(String::as_str).collect();
    match args.format {
        OutputFormat::Terminal => {
            crate::viz::terminal::print_metrics_table(&metrics);
            crate::viz::terminal::print_cls_similarity_matrix(&comparisons, &model_name_refs);
        }
        OutputFormat::Json => {
            crate::viz::json::print_metrics(&metrics)?;
            crate::viz::json::print_comparison(&comparisons)?;
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
            crate::viz::html::write_report(
                image_name,
                &metrics,
                &comparisons,
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
                let path = outdir.join(format!("{name}_pca.png"));
                crate::viz::png::save_pca_rgb(&projected, grid, &path)?;
            }
            println!("PNG outputs saved to {}", outdir.display());
        }
    }

    Ok(())
}
