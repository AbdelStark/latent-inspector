use crate::analysis::linear_cka;
use crate::errors::Error;
use crate::extract::{EmbeddingBasis, ExtractedFeatures};
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::report::{DriftReport, DriftStep};
use crate::viz::{terminal, OutputFormat};
use clap::Args;
use ndarray::Array2;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Args, Debug)]
pub struct DriftArgs {
    /// Model name (same architecture across all checkpoints).
    #[arg(short, long)]
    pub model: String,

    /// Directory containing checkpoint ONNX files (named by step/epoch).
    #[arg(short, long)]
    pub checkpoints: PathBuf,

    /// Dataset directory to measure drift on.
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Output directory for JSON/HTML/PNG artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,
}

pub fn run(args: DriftArgs) -> Result<(), Error> {
    info!(
        "Measuring drift for {} on {:?}",
        args.model, args.checkpoints
    );

    // Scan checkpoint directory for ONNX files
    let mut ckpt_paths: Vec<PathBuf> = std::fs::read_dir(&args.checkpoints)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("onnx"))
        .collect();

    if ckpt_paths.is_empty() {
        let report = DriftReport::new(
            args.model.clone(),
            args.checkpoints.display().to_string(),
            args.dataset.display().to_string(),
            EmbeddingBasis::MeanPatch,
            Vec::new(),
            None,
            Vec::new(),
            Vec::new(),
        );
        render_output(&args, &report)?;
        return Ok(());
    }

    ckpt_paths.sort_by(|left, right| natural_checkpoint_cmp(left, right));
    let checkpoint_names = ckpt_paths
        .iter()
        .map(|path| checkpoint_name(path))
        .collect::<Vec<_>>();
    info!("Dataset across {} checkpoints", checkpoint_names.len());

    // For each checkpoint, embed the dataset
    let mut all_embeddings: Vec<(String, Array2<f32>)> = Vec::new();
    let mut dataset_summary = None;
    let mut validation = Vec::with_capacity(ckpt_paths.len());

    for ckpt_path in &ckpt_paths {
        let ckpt_name = checkpoint_name(ckpt_path);
        info!("Processing checkpoint: {ckpt_name}");

        let mut session = ModelSession::load_checkpoint(&args.model, ckpt_path)?;
        let mut summary = summarize_session_or_unverified(&mut session, None);
        summary.model = ckpt_name.clone();
        summary.caveats.push(
            "Checkpoint drift runs reuse the registered preprocessing and tensor contract, while reference parity remains anchored to the approved release artifact rather than this checkpoint."
                .to_string(),
        );
        validation.push(summary);
        let (embedding, summary) = embed_dataset(&mut session, &args.dataset)?;
        if dataset_summary.is_none() {
            dataset_summary = Some(summary);
        }
        all_embeddings.push((ckpt_name, embedding));
    }

    let mut drift_rows = Vec::new();
    for window in all_embeddings.windows(2) {
        let (name_a, mat_a) = &window[0];
        let (name_b, mat_b) = &window[1];
        let cka = linear_cka(mat_a, mat_b)?;
        drift_rows.push(DriftStep {
            from_checkpoint: name_a.clone(),
            to_checkpoint: name_b.clone(),
            linear_cka: cka,
        });
    }

    let report = DriftReport::new(
        args.model.clone(),
        args.checkpoints.display().to_string(),
        args.dataset.display().to_string(),
        EmbeddingBasis::MeanPatch,
        checkpoint_names,
        dataset_summary,
        drift_rows,
        validation,
    );
    render_output(&args, &report)?;

    Ok(())
}

fn render_output(args: &DriftArgs, report: &DriftReport) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => terminal::print_drift_report(report),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("drift.json");
                crate::viz::json::write_drift_report(report, &path)?;
                OutputArtifactManifest::new("drift", OutputFormat::Json)
                    .with_primary_artifact("drift.json")
                    .add_artifact("drift.json", ArtifactKind::Json, "Drift report")
                    .with_validation(&report.validation)
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_drift_report(report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("drift_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("report.html");
            crate::viz::html::write_drift_report(report, &path)?;
            OutputArtifactManifest::new("drift", OutputFormat::Html)
                .with_primary_artifact("report.html")
                .add_artifact("report.html", ArtifactKind::Html, "Drift report")
                .with_validation(&report.validation)
                .write_to_dir(&outdir)?;
            println!("Report written to {}", path.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("drift_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("consecutive_cka.png");
            crate::viz::png::save_series_chart(&report.cka_series(), &path)?;
            OutputArtifactManifest::new("drift", OutputFormat::Png)
                .with_primary_artifact("consecutive_cka.png")
                .add_artifact(
                    "consecutive_cka.png",
                    ArtifactKind::Png,
                    "Consecutive checkpoint CKA chart",
                )
                .with_validation(&report.validation)
                .write_to_dir(&outdir)?;
            println!("PNG saved to {}", path.display());
        }
    }

    Ok(())
}

fn embed_dataset(
    session: &mut ModelSession,
    dataset_dir: &Path,
) -> Result<(Array2<f32>, crate::dataset::DatasetProcessingSummary), Error> {
    let mut rows = Vec::new();
    let summary = crate::dataset::for_each_image(dataset_dir, true, |_, img| {
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        rows.push(features.mean_patch());
        Ok::<(), Error>(())
    })?;

    if !summary.has_loaded_images() || rows.is_empty() {
        return Err(
            crate::errors::DatasetError::NoUsableImages(dataset_dir.display().to_string()).into(),
        );
    }

    let n = rows.len();
    let d = rows.first().map(|row| row.len()).unwrap_or(0);
    let mut matrix = Array2::<f32>::zeros((n, d));
    for (index, row) in rows.iter().enumerate() {
        matrix.row_mut(index).assign(row);
    }
    Ok((matrix, summary))
}

fn checkpoint_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn natural_checkpoint_cmp(left: &Path, right: &Path) -> Ordering {
    natural_cmp(&checkpoint_name(left), &checkpoint_name(right))
}

fn natural_cmp(left: &str, right: &str) -> Ordering {
    let left_bytes = left.as_bytes();
    let right_bytes = right.as_bytes();
    let mut left_index = 0;
    let mut right_index = 0;

    while left_index < left_bytes.len() && right_index < right_bytes.len() {
        let left_byte = left_bytes[left_index];
        let right_byte = right_bytes[right_index];

        if left_byte.is_ascii_digit() && right_byte.is_ascii_digit() {
            let left_start = left_index;
            let right_start = right_index;

            while left_index < left_bytes.len() && left_bytes[left_index].is_ascii_digit() {
                left_index += 1;
            }
            while right_index < right_bytes.len() && right_bytes[right_index].is_ascii_digit() {
                right_index += 1;
            }

            let ordering = compare_numeric_slices(
                &left_bytes[left_start..left_index],
                &right_bytes[right_start..right_index],
            );
            if ordering != Ordering::Equal {
                return ordering;
            }
            continue;
        }

        let ordering = left_byte
            .to_ascii_lowercase()
            .cmp(&right_byte.to_ascii_lowercase());
        if ordering != Ordering::Equal {
            return ordering;
        }

        left_index += 1;
        right_index += 1;
    }

    left_bytes.len().cmp(&right_bytes.len())
}

fn compare_numeric_slices(left: &[u8], right: &[u8]) -> Ordering {
    let left_trimmed = trim_leading_zeroes(left);
    let right_trimmed = trim_leading_zeroes(right);

    match left_trimmed.len().cmp(&right_trimmed.len()) {
        Ordering::Equal => match left_trimmed.cmp(right_trimmed) {
            Ordering::Equal => left.len().cmp(&right.len()),
            ordering => ordering,
        },
        ordering => ordering,
    }
}

fn trim_leading_zeroes(bytes: &[u8]) -> &[u8] {
    let trimmed = bytes
        .iter()
        .position(|byte| *byte != b'0')
        .map(|index| &bytes[index..])
        .unwrap_or(&[]);
    if trimmed.is_empty() {
        b"0"
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_checkpoint_sort_is_natural() {
        let mut paths = vec![
            PathBuf::from("step-10.onnx"),
            PathBuf::from("step-2.onnx"),
            PathBuf::from("step-1.onnx"),
        ];

        paths.sort_by(|left, right| natural_checkpoint_cmp(left, right));

        let names = paths
            .iter()
            .map(|path| checkpoint_name(path))
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["step-1", "step-2", "step-10"]);
    }
}
