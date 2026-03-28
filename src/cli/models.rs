use crate::errors::Error;
use crate::models::{build_model_catalog, cache, registry};
use clap::Args;

#[derive(Args, Debug)]
pub struct ModelsArgs {
    /// Download a specific model by name.
    #[arg(long)]
    pub download: Option<String>,

    /// Show all models including size information.
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: ModelsArgs) -> Result<(), Error> {
    if let Some(name) = args.download {
        return download_model(&name);
    }

    list_models(args.verbose)?;
    Ok(())
}

fn list_models(verbose: bool) -> Result<(), Error> {
    let report = build_model_catalog(None);
    crate::viz::terminal::print_model_catalog(&report, verbose);
    println!("Run `latent-inspector models --download dinov2-vit-l14` to cache the Phase 1 model.");
    let cache_path = cache::cache_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("Cache dir: {}", cache_path);
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
