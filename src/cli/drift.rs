use crate::analysis::linear_cka;
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use clap::Args;
use ndarray::Array2;
use std::path::PathBuf;
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
        println!(
            "No .onnx checkpoint files found in {}",
            args.checkpoints.display()
        );
        return Ok(());
    }

    ckpt_paths.sort();
    println!("Found {} checkpoints", ckpt_paths.len());

    // Load dataset image list
    let dataset_entries = crate::dataset::scan_images(&args.dataset)?;
    info!("Dataset: {} images", dataset_entries.len());

    // For each checkpoint, embed the dataset
    let mut all_embeddings: Vec<(String, Array2<f32>)> = Vec::new();

    for ckpt_path in &ckpt_paths {
        let ckpt_name = ckpt_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        info!("Processing checkpoint: {ckpt_name}");

        // For now, use the default model (stub mode) — real usage needs checkpoint loading
        let session = ModelSession::load(&args.model)?;
        let mut rows: Vec<ndarray::Array1<f32>> = Vec::new();

        for entry in &dataset_entries {
            let img = crate::dataset::load_image(&entry.path)?;
            let output = session.infer(&img)?;
            let features = ExtractedFeatures::from_output(output)?;
            rows.push(features.mean_patch());
        }

        let n = rows.len();
        let d = rows[0].len();
        let mut mat = Array2::<f32>::zeros((n, d));
        for (i, r) in rows.iter().enumerate() {
            mat.row_mut(i).assign(r);
        }
        all_embeddings.push((ckpt_name.to_string(), mat));
    }

    // Print CKA between consecutive checkpoints
    println!("\nRepresentation drift (CKA between consecutive checkpoints):");
    println!("{}", "─".repeat(60));
    for window in all_embeddings.windows(2) {
        let (name_a, mat_a) = &window[0];
        let (name_b, mat_b) = &window[1];
        let cka = linear_cka(mat_a, mat_b).unwrap_or(0.0);
        println!("  {name_a} → {name_b}: CKA = {:.4}", cka);
    }

    Ok(())
}
