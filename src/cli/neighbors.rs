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
    info!("Finding {} neighbors for {:?} using {}", args.k, args.image, args.model);

    let session = ModelSession::load(&args.model)?;

    // Embed query image
    let query_img = image::open(&args.image)?;
    let query_output = session.infer(&query_img)?;
    let query_features = ExtractedFeatures::from_output(query_output)?;
    let query_cls = query_features.cls_token.as_ref().ok_or_else(|| {
        crate::errors::AnalysisError::EmptyInput("Model has no CLS token for neighbor search".into())
    })?;

    // Embed all dataset images
    let dataset = crate::dataset::DatasetIterator::new(&args.dataset, true)?;
    let total = dataset.len();
    info!("Dataset size: {total} images");

    let mut embeddings: Vec<(String, ndarray::Array1<f32>)> = Vec::new();

    for result in dataset {
        let (entry, img) = result?;
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        if let Some(cls) = features.cls_token {
            embeddings.push((entry.stem.clone(), cls));
        }
    }

    if embeddings.is_empty() {
        println!("No valid embeddings found in dataset.");
        return Ok(());
    }

    // Build similarity scores between query and all dataset entries
    let _d = query_cls.len();
    let mut scores: Vec<(f32, &str)> = embeddings
        .iter()
        .map(|(name, emb)| {
            let dot: f32 = query_cls.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let na = query_cls.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
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

    Ok(())
}
