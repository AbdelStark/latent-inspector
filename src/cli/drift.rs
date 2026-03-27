use crate::analysis::linear_cka;
use crate::dataset::ImageEntry;
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::viz::terminal;
use clap::Args;
use ndarray::Array2;
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
        terminal::print_drift_summary(&[], &[]);
        return Ok(());
    }

    ckpt_paths.sort();
    let checkpoint_names = ckpt_paths
        .iter()
        .map(|path| checkpoint_name(path))
        .collect::<Vec<_>>();

    // Load dataset image list
    let dataset_entries = crate::dataset::scan_images(&args.dataset)?;
    info!(
        "Dataset: {} images across {} checkpoints",
        dataset_entries.len(),
        checkpoint_names.len()
    );

    // For each checkpoint, embed the dataset
    let mut all_embeddings: Vec<(String, Array2<f32>)> = Vec::new();

    for ckpt_path in &ckpt_paths {
        let ckpt_name = checkpoint_name(ckpt_path);
        info!("Processing checkpoint: {ckpt_name}");

        let mut session = ModelSession::load_checkpoint(&args.model, ckpt_path)?;
        let embedding = embed_dataset(&mut session, &dataset_entries)?;
        all_embeddings.push((ckpt_name, embedding));
    }

    let mut drift_rows = Vec::new();
    for window in all_embeddings.windows(2) {
        let (name_a, mat_a) = &window[0];
        let (name_b, mat_b) = &window[1];
        let cka = linear_cka(mat_a, mat_b)?;
        drift_rows.push((name_a.clone(), name_b.clone(), cka));
    }

    terminal::print_drift_summary(&checkpoint_names, &drift_rows);

    Ok(())
}

fn embed_dataset(
    session: &mut ModelSession,
    dataset_entries: &[ImageEntry],
) -> Result<Array2<f32>, Error> {
    let mut rows: Vec<ndarray::Array1<f32>> = Vec::with_capacity(dataset_entries.len());

    for entry in dataset_entries {
        let img = crate::dataset::load_image(&entry.path)?;
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        rows.push(features.mean_patch());
    }

    let n = rows.len();
    let d = rows.first().map(|row| row.len()).unwrap_or(0);
    let mut matrix = Array2::<f32>::zeros((n, d));
    for (index, row) in rows.iter().enumerate() {
        matrix.row_mut(index).assign(row);
    }
    Ok(matrix)
}

fn checkpoint_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string()
}
