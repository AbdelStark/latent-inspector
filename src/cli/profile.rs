use crate::analysis::{
    compute_metrics, intrinsic_dimensionality_default, isotropy_score, partition_isotropy,
    uniformity,
};
use crate::errors::Error;
use crate::extract::{EmbeddingBasis, ExtractedFeatures};
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::assets;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::report::{build_aggregate, ProfileImageMetrics, ProfileReport, SpaceMetrics};
use crate::viz::OutputFormat;
use clap::Args;
use ndarray::Array2;
use serde_json::json;
use std::path::PathBuf;
use tracing::info;

#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Model to profile.
    #[arg(short, long, default_value = "dinov2-vit-l14")]
    pub model: String,

    /// Dataset directory to profile over.
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Output directory for JSON/HTML/PNG artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,
}

/// Per-image data collected during the profiling pass.
struct ProfileSample {
    image: String,
    embedding: ndarray::Array1<f32>,
    metrics: crate::analysis::ModelMetrics,
}

/// Execute the `profile` subcommand: compute a comprehensive representation
/// profile for a model over a dataset, including space-level metrics
/// (isotropy, uniformity, intrinsic dimensionality) and per-image metric
/// aggregates.
pub fn run(args: ProfileArgs) -> Result<(), Error> {
    info!("Profiling model {} on {:?}", args.model, args.dataset);

    let mut session = ModelSession::load_for_analysis(&args.model)?;
    let validation = summarize_session_or_unverified(&mut session, None);

    let (dataset_summary, samples) = crate::dataset::map_images_parallel(
        &args.dataset,
        true,
        || ModelSession::load_for_analysis(&args.model).map_err(Error::from),
        |session, entry, img| {
            let output = session.infer(&img)?;
            let features = ExtractedFeatures::from_output(output)?;
            let metrics = compute_metrics(&features, &args.model)?;
            let (_, embedding) = features.preferred_global_embedding();

            Ok(Some(ProfileSample {
                image: entry.stem,
                embedding,
                metrics,
            }))
        },
    )?;

    if !dataset_summary.has_loaded_images() || samples.is_empty() {
        return Err(crate::errors::DatasetError::NoUsableImages(
            args.dataset.display().to_string(),
        )
        .into());
    }

    let n = samples.len();
    let embed_dim = samples[0].embedding.len();
    info!("Profiled {n} images with {embed_dim}-dim embeddings");

    // Build embedding matrix [N, D] for space-level metrics
    let mut embedding_matrix = Array2::<f32>::zeros((n, embed_dim));
    for (i, sample) in samples.iter().enumerate() {
        embedding_matrix.row_mut(i).assign(&sample.embedding);
    }

    // Determine embedding basis from the first sample
    let embedding_basis = if samples[0].metrics.cls_l2_norm.is_some() {
        EmbeddingBasis::ClsToken
    } else {
        EmbeddingBasis::MeanPatch
    };

    // Compute space-level metrics
    let iso_cosine = isotropy_score(&embedding_matrix)?;
    let iso_partition = partition_isotropy(&embedding_matrix)?;
    let unif = uniformity(&embedding_matrix)?;
    let intrinsic_dim = intrinsic_dimensionality_default(&embedding_matrix)?;

    let space_metrics = SpaceMetrics {
        isotropy_cosine: iso_cosine,
        isotropy_partition: iso_partition,
        uniformity: unif,
        intrinsic_dimensionality: intrinsic_dim,
    };

    // Build per-image metrics and aggregates
    let per_image_metrics: Vec<ProfileImageMetrics> = samples
        .iter()
        .map(|s| ProfileImageMetrics {
            image: s.image.clone(),
            effective_rank: s.metrics.effective_rank,
            dead_dimensions: s.metrics.dead_dimensions,
            patch_entropy: s.metrics.patch_entropy,
            attention_gini: s.metrics.attention_gini,
            cls_l2_norm: s.metrics.cls_l2_norm,
            patch_norm_mean: s.metrics.patch_norm_mean,
            patch_norm_std: s.metrics.patch_norm_std,
            top10_variance_pct: s.metrics.top10_variance_pct,
            spatial_coherence: s.metrics.spatial_coherence,
        })
        .collect();

    let aggregate_metrics = build_aggregates(&samples);

    let report = ProfileReport {
        model: args.model.clone(),
        dataset: args.dataset.display().to_string(),
        embedding_basis,
        sample_count: n,
        embed_dim,
        dataset_summary,
        space_metrics,
        aggregate_metrics,
        per_image_metrics,
        validation,
    };

    render_output(&args, &report)?;
    Ok(())
}

fn build_aggregates(samples: &[ProfileSample]) -> Vec<crate::viz::report::AggregateStatistic> {
    let ranks: Vec<f32> = samples
        .iter()
        .map(|s| s.metrics.effective_rank as f32)
        .collect();
    let dead: Vec<f32> = samples
        .iter()
        .map(|s| s.metrics.dead_dimensions as f32)
        .collect();
    let entropy: Vec<f32> = samples.iter().map(|s| s.metrics.patch_entropy).collect();
    let top10: Vec<f32> = samples
        .iter()
        .map(|s| s.metrics.top10_variance_pct)
        .collect();
    let norm_mean: Vec<f32> = samples.iter().map(|s| s.metrics.patch_norm_mean).collect();
    let norm_std: Vec<f32> = samples.iter().map(|s| s.metrics.patch_norm_std).collect();

    let mut aggs = vec![
        build_aggregate("effective_rank", "Effective rank", &ranks),
        build_aggregate("dead_dimensions", "Dead dimensions", &dead),
        build_aggregate("patch_entropy", "Patch entropy", &entropy),
        build_aggregate("top10_variance_pct", "Top-10 variance %", &top10),
        build_aggregate("patch_norm_mean", "Patch norm mean", &norm_mean),
        build_aggregate("patch_norm_std", "Patch norm std", &norm_std),
    ];

    let gini: Vec<f32> = samples
        .iter()
        .filter_map(|s| s.metrics.attention_gini)
        .collect();
    if !gini.is_empty() {
        aggs.push(build_aggregate("attention_gini", "Attention Gini", &gini));
    }

    let coherence: Vec<f32> = samples
        .iter()
        .filter_map(|s| s.metrics.spatial_coherence)
        .collect();
    if !coherence.is_empty() {
        aggs.push(build_aggregate(
            "spatial_coherence",
            "Spatial coherence",
            &coherence,
        ));
    }

    aggs
}

fn render_output(args: &ProfileArgs, report: &ProfileReport) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => crate::viz::terminal::print_profile_report(report),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("profile.json");
                crate::viz::json::write_profile_report(report, &path)?;
                OutputArtifactManifest::new("profile", OutputFormat::Json)
                    .with_primary_artifact("profile.json")
                    .with_context(profile_manifest_context(args))
                    .with_summary(profile_manifest_summary(report))
                    .add_artifact("profile.json", ArtifactKind::Json, "Profile report")
                    .with_validation(std::slice::from_ref(&report.validation))
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_profile_report(report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("profile_output"));
            std::fs::create_dir_all(&outdir)?;
            let assets = render_profile_assets(report, &outdir)?;
            let mut manifest = OutputArtifactManifest::new("profile", OutputFormat::Html)
                .with_primary_artifact("report.html")
                .with_context(profile_manifest_context(args))
                .with_summary(profile_manifest_summary(report))
                .add_artifact("report.html", ArtifactKind::Html, "Profile report")
                .add_artifact("profile.json", ArtifactKind::Json, "Profile report data")
                .with_validation(std::slice::from_ref(&report.validation));
            for asset in &assets.visuals {
                manifest = manifest.add_artifact(
                    asset.path.clone(),
                    ArtifactKind::Png,
                    asset.description.clone(),
                );
            }
            crate::viz::json::write_profile_report(report, &outdir.join("profile.json"))?;
            let path = outdir.join("report.html");
            let bundle = manifest.finalize_for_bundle_display(&outdir)?;
            crate::viz::html::write_profile_report_with_assets_and_bundle(
                report,
                &assets,
                Some(&bundle),
                &path,
            )?;
            manifest.write_to_dir(&outdir)?;
            println!("Report written to {}", path.display());
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("profile_output"));
            std::fs::create_dir_all(&outdir)?;
            let series: Vec<f32> = report
                .space_metric_series()
                .iter()
                .map(|(_, v)| *v)
                .collect();
            let path = outdir.join("profile.png");
            crate::viz::png::save_series_chart(&series, &path)?;
            OutputArtifactManifest::new("profile", OutputFormat::Png)
                .with_primary_artifact("profile.png")
                .with_context(profile_manifest_context(args))
                .with_summary(profile_manifest_summary(report))
                .add_artifact("profile.png", ArtifactKind::Png, "Space metrics chart")
                .with_validation(std::slice::from_ref(&report.validation))
                .write_to_dir(&outdir)?;
            println!("PNG saved to {}", path.display());
        }
    }

    Ok(())
}

fn profile_manifest_context(args: &ProfileArgs) -> serde_json::Value {
    json!({
        "model": args.model,
        "dataset": args.dataset.display().to_string(),
    })
}

fn profile_manifest_summary(report: &ProfileReport) -> serde_json::Value {
    json!({
        "sample_count": report.sample_count,
        "embed_dim": report.embed_dim,
        "embedding_basis": report.embedding_basis,
        "isotropy_cosine": report.space_metrics.isotropy_cosine,
        "isotropy_partition": report.space_metrics.isotropy_partition,
        "uniformity": report.space_metrics.uniformity,
        "intrinsic_dimensionality": report.space_metrics.intrinsic_dimensionality,
        "dataset_summary": report.dataset_summary,
    })
}

fn render_profile_assets(
    report: &ProfileReport,
    outdir: &std::path::Path,
) -> Result<crate::viz::html::GalleryAssets, Error> {
    let mut visuals = Vec::new();

    // Space metrics chart
    let series: Vec<f32> = report.aggregate_metrics.iter().map(|a| a.mean).collect();
    if !series.is_empty() {
        let filename = "aggregate_means.png";
        crate::viz::png::save_series_chart(&series, &outdir.join(filename))?;
        visuals.push(assets::visual_asset(
            filename,
            "Aggregate metric means",
            "Mean values of per-image metrics across the dataset.",
        ));
    }

    Ok(crate::viz::html::GalleryAssets { visuals })
}
