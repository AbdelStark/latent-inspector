use crate::dataset::ImageEntry;
use crate::errors::Error;
use crate::extract::{EmbeddingBasis, ExtractedFeatures};
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::assets;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::report::{NeighborMatch, NeighborsReport};
use crate::viz::OutputFormat;
use clap::Args;
use serde_json::json;
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

    /// Output directory for JSON/HTML/PNG artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,
}

pub fn run(args: NeighborsArgs) -> Result<(), Error> {
    info!(
        "Finding {} neighbors for {:?} using {}",
        args.k, args.image, args.model
    );

    let mut session = ModelSession::load_for_analysis(&args.model)?;
    let validation = summarize_session_or_unverified(&mut session, None);

    // Embed query image
    let query_img = image::open(&args.image)?;
    let query_output = session.infer(&query_img)?;
    let query_features = ExtractedFeatures::from_output(query_output)?;
    let (embedding_basis, query_embedding) = query_features.preferred_global_embedding();

    let mut embeddings: Vec<(ImageEntry, ndarray::Array1<f32>)> = Vec::new();
    let dataset_summary = crate::dataset::for_each_image(&args.dataset, true, |entry, img| {
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        let embedding = extract_neighbor_embedding(&features, embedding_basis);
        embeddings.push((entry, embedding));
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
            "Dataset produced no usable global embeddings for neighbor search".into(),
        )
        .into());
    }

    // Build similarity scores between query and all dataset entries
    let mut scores: Vec<(f32, ImageEntry)> = embeddings
        .iter()
        .map(|(entry, emb)| {
            let dot: f32 = query_embedding
                .iter()
                .zip(emb.iter())
                .map(|(a, b)| a * b)
                .sum();
            let na = query_embedding
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            let nb = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            (dot / (na * nb), entry.clone())
        })
        .collect();

    scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    let preview_sources = scores
        .iter()
        .take(args.k.min(4))
        .enumerate()
        .map(|(rank, (similarity, entry))| NeighborPreviewSource {
            rank: rank + 1,
            similarity: *similarity,
            entry: entry.clone(),
        })
        .collect::<Vec<_>>();
    let neighbors = scores
        .iter()
        .take(args.k)
        .enumerate()
        .map(|(rank, (similarity, entry))| NeighborMatch {
            rank: rank + 1,
            image: entry.stem.clone(),
            similarity: *similarity,
        })
        .collect::<Vec<_>>();
    let report = NeighborsReport {
        query_image: args.image.display().to_string(),
        dataset: args.dataset.display().to_string(),
        model: args.model.clone(),
        embedding_basis,
        requested_k: args.k,
        dataset_summary,
        neighbors,
        validation,
    };
    render_output(&args, &report, &query_img, &preview_sources)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct NeighborPreviewSource {
    rank: usize,
    similarity: f32,
    entry: ImageEntry,
}

fn extract_neighbor_embedding(
    features: &ExtractedFeatures,
    basis: EmbeddingBasis,
) -> ndarray::Array1<f32> {
    if let Some(embedding) = features.embedding_for_basis(basis) {
        embedding
    } else {
        features.mean_patch()
    }
}

fn render_output(
    args: &NeighborsArgs,
    report: &NeighborsReport,
    query_image: &image::DynamicImage,
    preview_sources: &[NeighborPreviewSource],
) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => crate::viz::terminal::print_neighbors_report(report),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("neighbors.json");
                crate::viz::json::write_neighbors_report(report, &path)?;
                OutputArtifactManifest::new("neighbors", OutputFormat::Json)
                    .with_primary_artifact("neighbors.json")
                    .with_context(neighbors_manifest_context(args))
                    .with_summary(neighbors_manifest_summary(report))
                    .add_artifact("neighbors.json", ArtifactKind::Json, "Neighbors report")
                    .with_validation(std::slice::from_ref(&report.validation))
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_neighbors_report(report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("neighbors_output"));
            std::fs::create_dir_all(&outdir)?;
            let assets = render_neighbors_assets(report, query_image, preview_sources, &outdir)?;
            crate::viz::json::write_neighbors_report(report, &outdir.join("neighbors.json"))?;
            let path = outdir.join("report.html");
            crate::viz::html::write_neighbors_report_with_assets(report, &assets, &path)?;
            let mut manifest = OutputArtifactManifest::new("neighbors", OutputFormat::Html)
                .with_primary_artifact("report.html")
                .with_context(neighbors_manifest_context(args))
                .with_summary(neighbors_manifest_summary(report))
                .add_artifact("report.html", ArtifactKind::Html, "Neighbors report")
                .add_artifact(
                    "neighbors.json",
                    ArtifactKind::Json,
                    "Neighbors report data",
                )
                .with_validation(std::slice::from_ref(&report.validation));
            for asset in &assets.visuals {
                manifest = manifest.add_artifact(
                    asset.path.clone(),
                    ArtifactKind::Png,
                    asset.description.clone(),
                );
            }
            manifest.write_to_dir(&outdir)?;
            println!("Report written to {}", path.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("neighbors_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("neighbors.png");
            crate::viz::png::save_series_chart(&report.similarity_series(), &path)?;
            OutputArtifactManifest::new("neighbors", OutputFormat::Png)
                .with_primary_artifact("neighbors.png")
                .with_context(neighbors_manifest_context(args))
                .with_summary(neighbors_manifest_summary(report))
                .add_artifact(
                    "neighbors.png",
                    ArtifactKind::Png,
                    "Neighbors similarity chart",
                )
                .with_validation(std::slice::from_ref(&report.validation))
                .write_to_dir(&outdir)?;
            println!("PNG saved to {}", path.display());
        }
    }

    Ok(())
}

fn neighbors_manifest_context(args: &NeighborsArgs) -> serde_json::Value {
    json!({
        "query_image": args.image.display().to_string(),
        "dataset": args.dataset.display().to_string(),
        "model": args.model,
        "requested_k": args.k,
    })
}

fn neighbors_manifest_summary(report: &NeighborsReport) -> serde_json::Value {
    json!({
        "returned_neighbors": report.neighbors.len(),
        "embedding_basis": report.embedding_basis,
        "dataset_summary": report.dataset_summary,
        "top_neighbor": report.neighbors.first(),
    })
}

fn render_neighbors_assets(
    report: &NeighborsReport,
    query_image: &image::DynamicImage,
    preview_sources: &[NeighborPreviewSource],
    outdir: &std::path::Path,
) -> Result<crate::viz::html::GalleryAssets, Error> {
    let mut visuals = vec![assets::write_preview_image(
        query_image,
        outdir,
        "query_image.png",
        "Query image",
        format!("Source query image searched against {}.", report.dataset),
    )?];

    for preview in preview_sources {
        let filename = format!(
            "neighbor_{:02}_{}.png",
            preview.rank,
            assets::slugify_filename(&preview.entry.stem)
        );
        visuals.push(assets::write_preview_from_path(
            &preview.entry.path,
            outdir,
            &filename,
            format!("Neighbor #{}: {}", preview.rank, preview.entry.stem),
            format!(
                "Dataset match ranked #{} with cosine similarity {:.4}.",
                preview.rank, preview.similarity
            ),
        )?);
    }

    let filename = "neighbors.png";
    crate::viz::png::save_series_chart(&report.similarity_series(), &outdir.join(filename))?;
    visuals.push(assets::visual_asset(
        filename,
        "Neighbor similarity chart",
        "Rank-ordered cosine similarity for the returned neighbor set.",
    ));

    Ok(crate::viz::html::GalleryAssets { visuals })
}
