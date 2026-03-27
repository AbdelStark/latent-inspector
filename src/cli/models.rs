use crate::errors::Error;
use crate::models::{cache, registry};
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

    list_models(args.verbose);
    Ok(())
}

fn list_models(verbose: bool) {
    let entries = registry::registry();

    println!("\nAvailable models ({}):", entries.len());
    println!("{}", "═".repeat(88));
    println!(
        "{:<22} {:<10} {:<10} {:<14} {:<10} {:<8}",
        "Name", "Status", "Verify", "Method", "Params (M)", "Cached"
    );
    println!("{}", "─".repeat(88));

    for entry in &entries {
        let cached = cache::is_cached(&entry.info.name)
            .map(|c| if c { "✓" } else { "✗" })
            .unwrap_or("?");

        println!(
            "{:<22} {:<10} {:<10} {:<14} {:<10} {:<8}",
            entry.info.name,
            entry.availability.status.to_string(),
            entry.verification_label(),
            entry.info.method,
            entry.info.params_m,
            cached,
        );

        if verbose {
            println!("    Phase: {}", entry.availability.phase);
            println!("    Note: {}", entry.availability.note);
            println!("    Arch: {}", entry.info.architecture);
            println!(
                "    Input: {}×{}",
                entry.info.input_size, entry.info.input_size
            );
            println!("    Embed dim: {}", entry.info.embed_dim);
            println!(
                "    Layers: {}, Heads: {}",
                entry.info.num_layers, entry.info.num_heads
            );
            if entry.is_ready() {
                for artifact in &entry.artifacts {
                    println!("    Artifact: {}", artifact.relative_path);
                    println!("    URL: {}", artifact.download_url);
                }
            }
            if let Some(note) = entry.verification_note() {
                println!("    Verify note: {}", note);
            }
        }
    }

    println!("{}", "═".repeat(88));
    println!("Run `latent-inspector models --download dinov2-vit-l14` to cache the Phase 1 model.");
    let cache_path = cache::cache_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("Cache dir: {}", cache_path);
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
