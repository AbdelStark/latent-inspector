use crate::analysis::{cls_cosine_similarity, knn_overlap, linear_cka};
use crate::errors::Error;
use crate::extract::{EmbeddingBasis, ExtractedFeatures};
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::report::{SimilarityMetricValue, SimilarityReport};
use crate::viz::OutputFormat;
use clap::Args;
use ndarray::Array2;
use std::path::PathBuf;
use tracing::info;

#[derive(Args, Debug)]
pub struct SimilarityArgs {
    /// First model.
    #[arg(long)]
    pub model_a: String,

    /// Second model.
    #[arg(long)]
    pub model_b: String,

    /// Dataset directory.
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Similarity metric to use.
    #[arg(short, long, default_value = "cka", value_parser = ["cka", "knn", "cosine", "all"])]
    pub metric: String,

    /// Output directory for JSON/HTML/PNG artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,
}

pub fn run(args: SimilarityArgs) -> Result<(), Error> {
    info!(
        "Measuring similarity {} vs {} on {:?}",
        args.model_a, args.model_b, args.dataset
    );

    let mut session_a = ModelSession::load_for_analysis(&args.model_a)?;
    let mut session_b = ModelSession::load_for_analysis(&args.model_b)?;
    let (label_a, label_b) = similarity_validation_labels(&args.model_a, &args.model_b);

    let mut validation_a = summarize_session_or_unverified(&mut session_a, None);
    validation_a.model = label_a;
    let mut validation_b = summarize_session_or_unverified(&mut session_b, None);
    validation_b.model = label_b;

    let mut cls_a: Vec<ndarray::Array1<f32>> = Vec::new();
    let mut cls_b: Vec<ndarray::Array1<f32>> = Vec::new();

    let mut patch_rows_a: Vec<ndarray::Array1<f32>> = Vec::new();
    let mut patch_rows_b: Vec<ndarray::Array1<f32>> = Vec::new();

    let dataset_summary = crate::dataset::for_each_image(&args.dataset, true, |_, img| {
        let out_a = session_a.infer(&img)?;
        let out_b = session_b.infer(&img)?;

        let feat_a = ExtractedFeatures::from_output(out_a)?;
        let feat_b = ExtractedFeatures::from_output(out_b)?;

        let mean_a = feat_a.mean_patch();
        let mean_b = feat_b.mean_patch();

        if let (Some(ca), Some(cb)) = (feat_a.cls_token, feat_b.cls_token) {
            cls_a.push(ca);
            cls_b.push(cb);
        }

        patch_rows_a.push(mean_a);
        patch_rows_b.push(mean_b);
        Ok::<(), Error>(())
    })?;
    info!("Dataset: {} supported images", dataset_summary.discovered);

    if !dataset_summary.has_loaded_images() || patch_rows_a.is_empty() {
        return Err(crate::errors::DatasetError::NoUsableImages(
            args.dataset.display().to_string(),
        )
        .into());
    }

    let n = patch_rows_a.len();
    let da = patch_rows_a[0].len();
    let db = patch_rows_b[0].len();

    let mut mat_a = Array2::<f32>::zeros((n, da));
    let mut mat_b = Array2::<f32>::zeros((n, db));
    for i in 0..n {
        mat_a.row_mut(i).assign(&patch_rows_a[i]);
        mat_b.row_mut(i).assign(&patch_rows_b[i]);
    }

    let mut metrics = Vec::new();
    if matches!(args.metric.as_str(), "cka" | "all") {
        metrics.push(SimilarityMetricValue {
            key: "linear_cka".to_string(),
            label: "Linear CKA".to_string(),
            value: linear_cka(&mat_a, &mat_b)?,
        });
    }

    if matches!(args.metric.as_str(), "knn" | "all") {
        metrics.push(SimilarityMetricValue {
            key: "knn_overlap_k10".to_string(),
            label: "k-NN overlap (k=10)".to_string(),
            value: knn_overlap(&mat_a, &mat_b, 10)?,
        });
    }

    let note = if matches!(args.metric.as_str(), "cosine" | "all") {
        match mean_cls_cosine(&cls_a, &cls_b) {
            Ok(mean_sim) => {
                metrics.push(SimilarityMetricValue {
                    key: "mean_cls_cosine".to_string(),
                    label: "Mean CLS cosine sim".to_string(),
                    value: mean_sim,
                });
                None
            }
            Err(note) => Some(note),
        }
    } else {
        None
    };

    let report = SimilarityReport {
        model_a: args.model_a.clone(),
        model_b: args.model_b.clone(),
        dataset: args.dataset.display().to_string(),
        dataset_embedding_basis: EmbeddingBasis::MeanPatch,
        requested_metric: args.metric.clone(),
        sample_count: n,
        dataset_summary,
        metrics,
        note,
        validation: vec![validation_a, validation_b],
    };
    render_output(&args, &report)?;

    Ok(())
}

fn similarity_validation_labels(model_a: &str, model_b: &str) -> (String, String) {
    if model_a == model_b {
        (format!("{model_a}#1"), format!("{model_b}#2"))
    } else {
        (model_a.to_string(), model_b.to_string())
    }
}

fn mean_cls_cosine(
    cls_a: &[ndarray::Array1<f32>],
    cls_b: &[ndarray::Array1<f32>],
) -> Result<f32, String> {
    if cls_a.is_empty() {
        return Err("N/A (CLS tokens unavailable)".to_string());
    }

    let same_width = cls_a.iter().zip(cls_b).all(|(a, b)| a.len() == b.len());
    if !same_width {
        return Err(format!(
            "N/A (embedding dims differ: {} vs {})",
            cls_a[0].len(),
            cls_b[0].len()
        ));
    }

    let total = cls_a
        .iter()
        .zip(cls_b.iter())
        .map(|(left, right)| cls_cosine_similarity(left, right))
        .sum::<f32>();
    Ok(total / cls_a.len() as f32)
}

fn render_output(args: &SimilarityArgs, report: &SimilarityReport) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => crate::viz::terminal::print_similarity_report(report),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("similarity.json");
                crate::viz::json::write_similarity_report(report, &path)?;
                OutputArtifactManifest::new("similarity", OutputFormat::Json)
                    .with_primary_artifact("similarity.json")
                    .add_artifact("similarity.json", ArtifactKind::Json, "Similarity report")
                    .with_validation(&report.validation)
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_similarity_report(report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("similarity_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("report.html");
            crate::viz::html::write_similarity_report(report, &path)?;
            OutputArtifactManifest::new("similarity", OutputFormat::Html)
                .with_primary_artifact("report.html")
                .add_artifact("report.html", ArtifactKind::Html, "Similarity report")
                .with_validation(&report.validation)
                .write_to_dir(&outdir)?;
            println!("Report written to {}", path.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("similarity_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("similarity.png");
            crate::viz::png::save_series_chart(&report.metric_series(), &path)?;
            OutputArtifactManifest::new("similarity", OutputFormat::Png)
                .with_primary_artifact("similarity.png")
                .add_artifact(
                    "similarity.png",
                    ArtifactKind::Png,
                    "Similarity metric chart",
                )
                .with_validation(&report.validation)
                .write_to_dir(&outdir)?;
            println!("PNG saved to {}", path.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::similarity_validation_labels;

    #[test]
    fn duplicate_similarity_models_receive_stable_validation_labels() {
        assert_eq!(
            similarity_validation_labels("dinov2-vit-l14", "dinov2-vit-l14"),
            (
                "dinov2-vit-l14#1".to_string(),
                "dinov2-vit-l14#2".to_string()
            )
        );
    }
}
