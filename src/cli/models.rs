use crate::errors::{Error, ValidationError};
use crate::models::{
    build_model_catalog, build_model_download_report, cache, registry, ModelCatalogReport,
    ModelDownloadAction, ModelDownloadReport,
};
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::OutputFormat;
use clap::Args;
use serde_json::json;
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

    if let Some(name) = args.download.as_deref() {
        return download_model(&args, name);
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
                    .with_context(models_manifest_context(args))
                    .with_summary(models_manifest_summary(&report))
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
            let manifest = OutputArtifactManifest::new("models", OutputFormat::Html)
                .with_primary_artifact("models.html")
                .with_context(models_manifest_context(args))
                .with_summary(models_manifest_summary(&report))
                .add_artifact("models.html", ArtifactKind::Html, "Model catalog")
                .add_artifact("models.json", ArtifactKind::Json, "Model catalog data");
            crate::viz::json::write_model_catalog(&report, &outdir.join("models.json"))?;
            crate::viz::html::write_model_catalog_report_with_bundle(
                &report,
                Some(&manifest),
                &path,
            )?;
            manifest.write_to_dir(&outdir)?;
            println!("Model catalog written to {}", path.display());
        }
        OutputFormat::Png => unreachable!("validated earlier"),
    }

    Ok(())
}

fn download_model(args: &ModelsArgs, name: &str) -> Result<(), Error> {
    let entry = registry::find(name).ok_or_else(|| {
        crate::errors::ModelError::NotFound(format!(
            "Unknown model '{name}'. Run `latent-inspector models` to see available models."
        ))
    })?;
    entry.ensure_ready()?;

    let previous_artifacts = cache::inspect_registry_artifacts(&entry)?;
    let action = if previous_artifacts
        .iter()
        .all(|artifact| artifact.cache_status.is_usable())
        && !previous_artifacts.is_empty()
    {
        ModelDownloadAction::AlreadyCached
    } else {
        if matches!(args.format, OutputFormat::Terminal) {
            println!("Downloading {name} ({} M params)…", entry.info.params_m);
        }
        cache::download(name, &entry)?;
        ModelDownloadAction::Downloaded
    };

    let report = build_model_catalog(None).filter_to_names(&[name.to_string()]);
    let download_report = build_model_download_report(
        action,
        &previous_artifacts,
        take_single_entry(report, name)?,
    );
    render_download_output(args, &download_report)?;
    Ok(())
}

fn render_download_output(args: &ModelsArgs, report: &ModelDownloadReport) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => crate::viz::terminal::print_model_download_report(report),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("download.json");
                crate::viz::json::write_model_download_report(report, &path)?;
                OutputArtifactManifest::new("models", OutputFormat::Json)
                    .with_primary_artifact("download.json")
                    .with_context(download_manifest_context(args, report))
                    .with_summary(download_manifest_summary(report))
                    .add_artifact("download.json", ArtifactKind::Json, "Model download report")
                    .write_to_dir(outdir)?;
                println!("Model download report written to {}", path.display());
            } else {
                crate::viz::json::print_model_download_report(report)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("models_download_output"));
            std::fs::create_dir_all(&outdir)?;
            let path = outdir.join("download.html");
            let manifest = OutputArtifactManifest::new("models", OutputFormat::Html)
                .with_primary_artifact("download.html")
                .with_context(download_manifest_context(args, report))
                .with_summary(download_manifest_summary(report))
                .add_artifact("download.html", ArtifactKind::Html, "Model download report")
                .add_artifact("download.json", ArtifactKind::Json, "Model download data");
            crate::viz::json::write_model_download_report(report, &outdir.join("download.json"))?;
            crate::viz::html::write_model_download_report_with_bundle(
                report,
                Some(&manifest),
                &path,
            )?;
            manifest.write_to_dir(&outdir)?;
            println!("Model download report written to {}", path.display());
        }
        OutputFormat::Png => unreachable!("validated earlier"),
    }

    Ok(())
}

fn take_single_entry(
    report: ModelCatalogReport,
    name: &str,
) -> Result<crate::models::ModelInventoryEntry, Error> {
    report.entries.into_iter().next().ok_or_else(|| {
        ValidationError::Usage(format!(
            "Model catalog did not contain an entry for '{name}' after download."
        ))
        .into()
    })
}

fn models_manifest_context(args: &ModelsArgs) -> serde_json::Value {
    json!({
        "verbose": args.verbose,
        "mode": "catalog",
    })
}

fn models_manifest_summary(
    report: &crate::models::inventory::ModelCatalogReport,
) -> serde_json::Value {
    json!({
        "fixture_set": report.fixture_set,
        "evidence_timestamp": report.evidence_timestamp,
        "fixture_error": report.fixture_error,
        "summary": report.summary,
    })
}

fn download_manifest_context(args: &ModelsArgs, report: &ModelDownloadReport) -> serde_json::Value {
    json!({
        "verbose": args.verbose,
        "mode": "download",
        "model": report.model,
        "action": report.action,
    })
}

fn download_manifest_summary(report: &ModelDownloadReport) -> serde_json::Value {
    json!({
        "summary": report.summary,
        "readiness_status": report.entry.readiness_status,
        "readiness_summary": report.entry.readiness_summary,
        "downloaded_artifacts": report.downloaded_artifact_count(),
        "repaired_artifacts": report.repaired_artifact_count(),
    })
}
