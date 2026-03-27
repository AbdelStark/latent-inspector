use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use clap::Args;
use std::path::PathBuf;
use tracing::info;

#[derive(Args, Debug)]
pub struct NeighborsArgs {
    /// Query image path.
    pub image: PathBuf,

    /// Model to use.
    #[arg(short, long, default_value = "dinov2-vit-l14")]
    pub model: String,

    /// Dataset directory to search.
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Number of nearest neighbors to return.
    #[arg(short = 'k', long, default_value_t = 10)]
    pub k: usize,
}

pub fn run(args: NeighborsArgs) -> Result<(), Error> {
    info!(
        "Finding {} neighbors for {:?} using {}",
        args.k, args.image, args.model
    );

    let mut session = ModelSession::load(&args.model)?;

    // Embed query image
    let query_img = image::open(&args.image)?;
    let query_output = session.infer(&query_img)?;
    let query_features = ExtractedFeatures::from_output(query_output)?;
    let query_cls = query_features.cls_token.as_ref().ok_or_else(|| {
        crate::errors::AnalysisError::EmptyInput(
            "Model has no CLS token for neighbor search".into(),
        )
    })?;

    let mut embeddings: Vec<(String, ndarray::Array1<f32>)> = Vec::new();
    let dataset_summary = crate::dataset::for_each_image(&args.dataset, true, |entry, img| {
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        if let Some(cls) = features.cls_token {
            embeddings.push((entry.stem, cls));
        }
        Ok::<(), Error>(())
    })?;
    info!(
        "Dataset size: {} supported images",
        dataset_summary.discovered
    );

    if !dataset_summary.has_loaded_images() {
        return Err(crate::errors::DatasetError::NoUsableImages(
            args.dataset.display().to_string(),
        )
        .into());
    }

    if embeddings.is_empty() {
        return Err(crate::errors::AnalysisError::EmptyInput(
            "Dataset produced no CLS embeddings for neighbor search".into(),
        )
        .into());
    }

    // Build similarity scores between query and all dataset entries
    let _d = query_cls.len();
    let mut scores: Vec<(f32, &str)> = embeddings
        .iter()
        .map(|(name, emb)| {
            let dot: f32 = query_cls.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let na = query_cls
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            let nb = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            (dot / (na * nb), name.as_str())
        })
        .collect();

    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("\nNearest neighbors for {}", args.image.display());
    println!("Model: {}  k={}", args.model, args.k);
    println!("{}", "─".repeat(50));
    for (rank, (sim, name)) in scores.iter().take(args.k).enumerate() {
        println!("  {:2}. {:40} sim={:.4}", rank + 1, name, sim);
    }
    crate::viz::terminal::print_dataset_processing_summary(&dataset_summary);

    Ok(())
}
