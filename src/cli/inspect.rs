use crate::analysis::{compute_metrics, pca, transform, variance_spectrum};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::OutputFormat;
use clap::Args;
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

pub fn run(args: InspectArgs) -> Result<(), Error> {
    info!("Inspecting {} on {:?}", args.model, args.image);

    let img = image::open(&args.image)?;
    let mut session = ModelSession::load_for_analysis(&args.model)?;
    let output = session.infer(&img)?;
    let features = ExtractedFeatures::from_output(output)?;
    let validation_summary = summarize_session_or_unverified(&mut session, None);

    let metrics = compute_metrics(&features, &args.model)?;
    let spectrum = variance_spectrum(&features.patch_tokens, args.pca_components.min(64))?;
    let report = crate::viz::report::build_inspect_report(
        args.image.display().to_string(),
        args.model.clone(),
        metrics,
        validation_summary,
        &spectrum,
    );

    match args.format {
        OutputFormat::Terminal => {
            println!("\nModel: {}", report.model);
            println!("{}", "═".repeat(60));
            println!("  Patches:          {}", report.metrics.n_patches);
            println!("  Embed dim:        {}", report.metrics.embed_dim);
            println!(
                "  Effective rank:   {}/{}",
                report.metrics.effective_rank, report.metrics.embed_dim
            );
            println!("  Dead dimensions:  {}", report.metrics.dead_dimensions);
            println!("  Patch entropy:    {:.3}", report.metrics.patch_entropy);
            if let Some(norm) = report.metrics.cls_l2_norm {
                println!("  CLS L2 norm:      {:.2}", norm);
            }
            println!(
                "  Patch norm mean:  {:.2} ± {:.2}",
                report.metrics.patch_norm_mean, report.metrics.patch_norm_std
            );
            println!(
                "  Top-10 var%:      {:.1}%",
                report.metrics.top10_variance_pct
            );
            println!("  Components@90%:   {}", report.metrics.components_90pct);
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
                let bar = "█".repeat(bar_len);
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
                println!("JSON report written to {}", path.display());
            } else {
                crate::viz::json::print_inspect_report(&report)?;
            }
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            write_inspect_visual_artifacts(&features, &report, &outdir)?;
            println!("PNG saved to {}", outdir.display());
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            let assets = write_inspect_visual_artifacts(&features, &report, &outdir)?;
            crate::viz::html::write_inspect_report_with_assets(
                &report,
                &assets,
                &outdir.join("report.html"),
            )?;
            println!("Report written to {}/report.html", outdir.display());
        }
    }

    Ok(())
}

fn write_inspect_visual_artifacts(
    features: &ExtractedFeatures,
    report: &crate::viz::report::InspectReport,
    outdir: &Path,
) -> Result<crate::viz::html::InspectHtmlAssets, Error> {
    let prefix = slugify(&report.model);
    let pca_filename = format!("{prefix}_pca.png");
    let variance_filename = format!("{prefix}_variance.png");
    let pca_result = pca(&features.patch_tokens, 3, 300)?;
    let projected = transform(&features.patch_tokens, &pca_result);
    let grid = (features.n_patches as f32).sqrt() as usize;

    crate::viz::png::save_pca_rgb(&projected, grid, &outdir.join(&pca_filename))?;
    crate::viz::png::save_variance_spectrum_chart(
        &report.variance_spectrum.ratios,
        &outdir.join(&variance_filename),
    )?;

    Ok(crate::viz::html::InspectHtmlAssets {
        pca_image: Some(pca_filename),
        variance_image: Some(variance_filename),
    })
}

fn slugify(label: &str) -> String {
    let mut slug = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if matches!(ch, '-' | '_') {
            slug.push(ch);
        } else {
            slug.push('_');
        }
    }
    slug.trim_matches('_').to_string()
}
