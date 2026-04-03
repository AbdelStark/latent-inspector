use crate::analysis::{model_metrics_from_spectrum, pca, transform, variance_spectrum};
use crate::errors::Error;
use crate::extract::{AttentionMapBasis, ExtractedFeatures};
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::assets;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::report::InspectAttentionSummary;
use crate::viz::OutputFormat;
use clap::Args;
use serde_json::json;
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to the input image.
    pub image: PathBuf,

    /// Model to use.
    #[arg(short, long, default_value = "dinov2-vit-l14")]
    pub model: String,

    /// Output directory for artefacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,

    /// Number of PCA components to show in variance spectrum.
    #[arg(long, default_value_t = 32)]
    pub pca_components: usize,
}

/// Execute the `inspect` subcommand: extract features from a single model,
/// compute per-model metrics (rank, entropy, variance spectrum), and render.
pub fn run(args: InspectArgs) -> Result<(), Error> {
    info!("Inspecting {} on {:?}", args.model, args.image);

    let img = image::open(&args.image)?;
    let mut session = ModelSession::load_for_analysis(&args.model)?;
    let output = session.infer(&img)?;
    let features = ExtractedFeatures::from_output(output)?;
    let validation_summary = summarize_session_or_unverified(&mut session, None);

    let requested_components = args
        .pca_components
        .clamp(1, crate::analysis::MAX_PCA_COMPONENTS);
    let spectrum = variance_spectrum(&features.patch_tokens, crate::analysis::MAX_PCA_COMPONENTS)?;
    let metrics = model_metrics_from_spectrum(&features, &args.model, &spectrum)?;
    let display_spectrum = spectrum.truncated(requested_components);
    let attention = build_inspect_attention_summary(&features, metrics.attention_gini);
    let report = crate::viz::report::build_inspect_report(
        args.image.display().to_string(),
        args.model.clone(),
        metrics,
        validation_summary,
        &display_spectrum,
        attention,
    );

    match args.format {
        OutputFormat::Terminal => {
            println!("\nModel: {}", report.model);
            println!("{}", crate::viz::terminal::heavy_rule(60));
            println!("  Patches:          {}", report.metrics.n_patches);
            println!("  Embed dim:        {}", report.metrics.embed_dim);
            println!(
                "  Effective rank:   {}/{}",
                report.metrics.effective_rank, report.metrics.embed_dim
            );
            println!("  Dead dimensions:  {}", report.metrics.dead_dimensions);
            println!("  Patch entropy:    {:.3}", report.metrics.patch_entropy);
            if let Some(attention) = &report.attention {
                println!("  Attention gini:   {:.3}", attention.mean_gini);
            }
            if let Some(norm) = report.metrics.cls_l2_norm {
                println!("  CLS L2 norm:      {:.2}", norm);
            }
            println!(
                "  Patch norm mean:  {:.2} {} {:.2}",
                report.metrics.patch_norm_mean,
                crate::viz::terminal::plus_minus_separator(),
                report.metrics.patch_norm_std
            );
            println!(
                "  Top-10 var%:      {:.1}%",
                report.metrics.top10_variance_pct
            );
            println!("  Components@90%:   {}", report.metrics.components_90pct);
            println!("  Patch isotropy:   {:.3}", report.metrics.patch_isotropy);
            println!("  Patch uniformity: {:.3}", report.metrics.patch_uniformity);
            if let Some(attention) = &report.attention {
                println!(
                    "  Attention source: {} ({} layers x {} heads)",
                    attention.map_basis.label(),
                    attention.layers,
                    attention.heads,
                );
                if let Some((_, map)) = features.attention_map() {
                    println!();
                    println!("  Attention map:");
                    print!("{}", crate::viz::terminal::render_attention_map(&map, 16));
                }
            }
            println!();
            println!(
                "  Variance spectrum (top {} components):",
                report.variance_spectrum.ratios.len()
            );
            for (i, (&ratio, &cum)) in report
                .variance_spectrum
                .ratios
                .iter()
                .zip(report.variance_spectrum.cumulative.iter())
                .enumerate()
            {
                let bar_len = (ratio * 40.0) as usize;
                let bar = crate::viz::terminal::bar(bar_len);
                println!(
                    "    PC{:02}: {:5.2}%  {:5.2}% cum  {}",
                    i + 1,
                    ratio * 100.0,
                    cum * 100.0,
                    bar
                );
            }
            crate::viz::terminal::print_validation_summaries(std::slice::from_ref(
                &report.validation,
            ));
        }
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("inspect.json");
                crate::viz::json::write_inspect_report(&report, &path)?;
                OutputArtifactManifest::new("inspect", OutputFormat::Json)
                    .with_primary_artifact("inspect.json")
                    .with_context(inspect_manifest_context(&args, requested_components))
                    .with_summary(inspect_manifest_summary(&report))
                    .add_artifact("inspect.json", ArtifactKind::Json, "Inspect report")
                    .with_validation(std::slice::from_ref(&report.validation))
                    .write_to_dir(outdir)?;
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_inspect_report(&report)?;
            }
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            let assets = write_inspect_visual_artifacts(Some(&img), &features, &report, &outdir)?;
            let manifest = build_inspect_manifest(
                &report,
                Some(&assets),
                OutputFormat::Png,
                inspect_manifest_context(&args, requested_components),
            );
            manifest.write_to_dir(&outdir)?;
            println!("PNG saved to {}", outdir.display());
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            let assets = write_inspect_visual_artifacts(Some(&img), &features, &report, &outdir)?;
            let manifest = build_inspect_manifest(
                &report,
                Some(&assets),
                OutputFormat::Html,
                inspect_manifest_context(&args, requested_components),
            )
            .add_artifact("inspect.json", ArtifactKind::Json, "Inspect report data")
            .add_artifact("report.html", ArtifactKind::Html, "Inspect report")
            .with_primary_artifact("report.html");
            crate::viz::json::write_inspect_report(&report, &outdir.join("inspect.json"))?;
            let bundle = manifest.finalize_for_bundle_display(&outdir)?;
            crate::viz::html::write_inspect_report_with_assets_and_bundle(
                &report,
                &assets,
                Some(&bundle),
                &outdir.join("report.html"),
            )?;
            manifest.write_to_dir(&outdir)?;
            println!("Report written to {}/report.html", outdir.display());
        }
    }

    Ok(())
}

fn write_inspect_visual_artifacts(
    source_image: Option<&image::DynamicImage>,
    features: &ExtractedFeatures,
    report: &crate::viz::report::InspectReport,
    outdir: &Path,
) -> Result<crate::viz::html::InspectHtmlAssets, Error> {
    let prefix = assets::slugify_filename(&report.model);
    let pca_filename = format!("{prefix}_pca.png");
    let variance_filename = format!("{prefix}_variance.png");
    let attention_filename = format!("{prefix}_attention.png");
    let pca_result = pca(&features.patch_tokens, 3, 300)?;
    let projected = transform(&features.patch_tokens, &pca_result);
    let grid = patch_grid_side(
        features.n_patches,
        &format!("{} PCA rendering", report.model),
    )?;

    crate::viz::png::save_pca_rgb(&projected, grid, &outdir.join(&pca_filename))?;
    crate::viz::png::save_variance_spectrum_chart(
        &report.variance_spectrum.ratios,
        &outdir.join(&variance_filename),
    )?;
    let attention_image =
        if let (Some(image), Some((basis, map))) = (source_image, features.attention_map()) {
            crate::viz::png::save_attention_overlay(
                image,
                &map,
                &outdir.join(&attention_filename),
                0.45,
            )?;
            Some(assets::visual_asset(
                attention_filename,
                "Attention Overlay",
                format!(
                    "{} projected back onto the input image.",
                    basis.description()
                ),
            ))
        } else {
            None
        };

    Ok(crate::viz::html::InspectHtmlAssets {
        source_image: source_image
            .map(|image| {
                assets::write_preview_image(
                    image,
                    outdir,
                    "input_image.png",
                    "Input image",
                    format!("Source image inspected with {}.", report.model),
                )
            })
            .transpose()?,
        pca_image: Some(assets::visual_asset(
            pca_filename,
            "PCA Projection",
            "Patch-space RGB projection derived from the top three PCA components.",
        )),
        variance_image: Some(assets::visual_asset(
            variance_filename,
            "Variance Chart",
            "Component-wise variance concentration across the inspected representation.",
        )),
        attention_image,
    })
}

fn patch_grid_side(patch_count: usize, context: &str) -> Result<usize, Error> {
    Ok(crate::analysis::square_grid_side(patch_count, context)?)
}

fn build_inspect_manifest(
    report: &crate::viz::report::InspectReport,
    assets: Option<&crate::viz::html::InspectHtmlAssets>,
    format: OutputFormat,
    context: serde_json::Value,
) -> OutputArtifactManifest {
    let mut manifest = OutputArtifactManifest::new("inspect", format)
        .with_context(context)
        .with_summary(inspect_manifest_summary(report))
        .with_validation(std::slice::from_ref(&report.validation));

    if let Some(assets) = assets {
        if let Some(source_image) = &assets.source_image {
            manifest = manifest.add_artifact(
                source_image.path.clone(),
                ArtifactKind::Png,
                source_image.description.clone(),
            );
        }
        if let Some(pca_image) = &assets.pca_image {
            manifest = manifest.add_artifact(
                pca_image.path.clone(),
                ArtifactKind::Png,
                pca_image.description.clone(),
            );
        }
        if let Some(variance_image) = &assets.variance_image {
            manifest = manifest.add_artifact(
                variance_image.path.clone(),
                ArtifactKind::Png,
                variance_image.description.clone(),
            );
        }
        if let Some(attention_image) = &assets.attention_image {
            manifest = manifest.add_artifact(
                attention_image.path.clone(),
                ArtifactKind::Png,
                attention_image.description.clone(),
            );
        }
    }

    manifest
}

fn inspect_manifest_context(args: &InspectArgs, requested_components: usize) -> serde_json::Value {
    json!({
        "image": args.image.display().to_string(),
        "model": args.model,
        "pca_components": requested_components,
    })
}

fn inspect_manifest_summary(report: &crate::viz::report::InspectReport) -> serde_json::Value {
    json!({
        "effective_rank": report.metrics.effective_rank,
        "patch_entropy": report.metrics.patch_entropy,
        "attention_gini": report.metrics.attention_gini,
        "components_90pct": report.metrics.components_90pct,
        "components_99pct": report.variance_spectrum.components_99pct,
        "top10_variance_pct": report.metrics.top10_variance_pct,
    })
}

fn build_inspect_attention_summary(
    features: &ExtractedFeatures,
    attention_gini: Option<f32>,
) -> Option<InspectAttentionSummary> {
    let mean_gini = attention_gini?;
    let (layers, heads, token_count) = features.attention_dimensions()?;
    let map_basis = features
        .attention_map()
        .map(|(basis, _)| basis)
        .unwrap_or_else(|| {
            if features.sequence_has_cls && features.cls_token.is_some() {
                AttentionMapBasis::ClsToPatch
            } else {
                AttentionMapBasis::MeanTokenToPatch
            }
        });

    Some(InspectAttentionSummary {
        mean_gini,
        layers,
        heads,
        token_count,
        map_basis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_grid_side_rejects_non_square_patch_counts() {
        let error = patch_grid_side(10, "inspect PCA").unwrap_err();

        assert!(error
            .to_string()
            .contains("does not form a square patch grid"));
    }
}
