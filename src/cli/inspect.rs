use crate::analysis::{compute_metrics, variance_spectrum};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use crate::validation::summarize_session_or_unverified;
use crate::viz::OutputFormat;
use clap::Args;
use std::path::PathBuf;
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
    let mut session = ModelSession::load(&args.model)?;
    let output = session.infer(&img)?;
    let features = ExtractedFeatures::from_output(output)?;
    let validation_summary = summarize_session_or_unverified(&mut session, None);

    let metrics = compute_metrics(&features, &args.model)?;
    let spectrum = variance_spectrum(&features.patch_tokens, args.pca_components.min(64))?;

    match args.format {
        OutputFormat::Terminal => {
            println!("\nModel: {}", args.model);
            println!("{}", "═".repeat(60));
            println!("  Patches:          {}", features.n_patches);
            println!("  Embed dim:        {}", features.embed_dim);
            println!(
                "  Effective rank:   {}/{}",
                metrics.effective_rank, metrics.embed_dim
            );
            println!("  Dead dimensions:  {}", metrics.dead_dimensions);
            println!("  Patch entropy:    {:.3}", metrics.patch_entropy);
            if let Some(norm) = metrics.cls_l2_norm {
                println!("  CLS L2 norm:      {:.2}", norm);
            }
            println!(
                "  Patch norm mean:  {:.2} ± {:.2}",
                metrics.patch_norm_mean, metrics.patch_norm_std
            );
            println!("  Top-10 var%:      {:.1}%", metrics.top10_variance_pct);
            println!("  Components@90%:   {}", metrics.components_90pct);
            println!();
            println!(
                "  Variance spectrum (top {} components):",
                spectrum.ratios.len()
            );
            for (i, (&ratio, &cum)) in spectrum
                .ratios
                .iter()
                .zip(spectrum.cumulative.iter())
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
                &validation_summary,
            ));
        }
        OutputFormat::Json => {
            let data = serde_json::json!({
                "metrics": metrics,
                "validation": validation_summary,
                "variance_spectrum": {
                    "ratios": spectrum.ratios.to_vec(),
                    "cumulative": spectrum.cumulative.to_vec(),
                    "components_90pct": spectrum.components_90pct,
                    "components_99pct": spectrum.components_99pct,
                }
            });
            println!("{}", serde_json::to_string_pretty(&data)?);
        }
        OutputFormat::Png => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            let pca_result = crate::analysis::pca(&features.patch_tokens, 3, 300)?;
            let projected = crate::analysis::transform(&features.patch_tokens, &pca_result);
            let grid = (features.n_patches as f32).sqrt() as usize;
            let path = outdir.join(format!("{}_pca.png", args.model));
            crate::viz::png::save_pca_rgb(&projected, grid, &path)?;
            crate::viz::png::save_variance_spectrum_chart(
                &spectrum.ratios.to_vec(),
                &outdir.join(format!("{}_variance.png", args.model)),
            )?;
            println!("PNG saved to {}", outdir.display());
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .unwrap_or_else(|| PathBuf::from("inspect_output"));
            std::fs::create_dir_all(&outdir)?;
            let image_name = args
                .image
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image");
            crate::viz::html::write_report_with_validation(
                image_name,
                &[metrics],
                &[],
                &[validation_summary],
                &outdir.join("report.html"),
            )?;
            println!("Report written to {}/report.html", outdir.display());
        }
    }

    Ok(())
}
