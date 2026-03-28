use crate::errors::{Error, ValidationError};
use crate::models::{build_model_catalog, cache, registry};
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::OutputFormat;
use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct ModelsArgs {
    /// Download a specific model by name.
    #[arg(long)]
    pub download: Option<String>,

    /// Show all models including size information.
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format for the model catalog.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,

    /// Output directory for JSON or HTML artifacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: ModelsArgs) -> Result<(), Error> {
    if matches!(args.format, OutputFormat::Png) {
        return Err(ValidationError::Usage(
            "models only supports terminal, json, or html output.".to_string(),
        )
        .into());
    }

    if let Some(name) = args.download {
        return download_model(&name);
    }

    list_models(&args)?;
    Ok(())
}

fn list_models(args: &ModelsArgs) -> Result<(), Error> {
    let report = build_model_catalog(None);

    match args.format {
        OutputFormat::Terminal => {
            crate::viz::terminal::print_model_catalog(&report, args.verbose);
            println!(
                "Run `latent-inspector models --download dinov2-vit-l14` to cache the Phase 1 model."
            );
            let cache_path = cache::cache_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            println!("Cache dir: {}", cache_path);
        }
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("models.json");
                crate::viz::json::write_model_catalog(&report, &path)?;
                OutputArtifactManifest::new("models", OutputFormat::Json)
                    .with_primary_artifact("models.json")
                    .add_artifact("models.json", ArtifactKind::Json, "Model catalog")
                    .write_to_dir(outdir)?;
                println!("Model catalog written to {}", path.display());
            } else {
                crate::viz::json::print_model_catalog(&report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("models_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("models.html");
            crate::viz::html::write_model_catalog_report(&report, &path)?;
            OutputArtifactManifest::new("models", OutputFormat::Html)
                .with_primary_artifact("models.html")
                .add_artifact("models.html", ArtifactKind::Html, "Model catalog")
                .write_to_dir(&outdir)?;
            println!("Model catalog written to {}", path.display());
        }
        OutputFormat::Png => unreachable!("validated earlier"),
    }

    Ok(())
}

fn download_model(name: &str) -> Result<(), Error> {
    let entry = registry::find(name).ok_or_else(|| {
        crate::errors::ModelError::NotFound(format!(
            "Unknown model '{name}'. Run `latent-inspector models` to see available models."
        ))
    })?;
    entry.ensure_ready()?;

    let dest = cache::model_path(name)?;

    if cache::is_cached(name)? {
        println!("Model '{name}' is already cached at {}.", dest.display());
        return Ok(());
    }

    println!("Downloading {name} ({} M params)…", entry.info.params_m);
    cache::download(name, &entry)?;
    println!("✓ {name} saved to {}.", dest.display());
    Ok(())
}
