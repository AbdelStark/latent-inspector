use crate::errors::{Error, ModelError, ValidationError};
use crate::models::ModelSession;
use crate::validation::fixtures::load_fixture_set;
use crate::validation::report::{ModelValidationSummary, ValidationStatus};
use crate::validation::validate_session_with_fixture_set;
use crate::viz::manifest::{ArtifactKind, OutputArtifactManifest};
use crate::viz::OutputFormat;
use clap::Args;
use serde_json::json;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// One or more model identifiers to validate.
    #[arg(long = "model", required = true, value_delimiter = ',')]
    pub models: Vec<String>,

    /// Fixture set identifier or manifest path.
    #[arg(long)]
    pub fixture_set: Option<String>,

    /// Output format.
    #[arg(short, long, default_value = "terminal")]
    pub format: OutputFormat,

    /// Output directory for JSON or HTML artifacts.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Rewrite approved reference artifacts with current observed outputs.
    #[arg(long)]
    pub refresh_goldens: bool,
}

pub fn run(args: ValidateArgs) -> Result<(), Error> {
    if args.models.is_empty() {
        return Err(
            ValidationError::Usage("At least one --model value is required.".to_string()).into(),
        );
    }

    if matches!(args.format, OutputFormat::Png) {
        return Err(ValidationError::Usage(
            "validate only supports terminal, json, or html output.".to_string(),
        )
        .into());
    }

    let fixture_set = load_fixture_set(args.fixture_set.as_deref())?;
    let mut summaries = Vec::with_capacity(args.models.len());

    for model in &args.models {
        let mut session = ModelSession::load(model).map_err(|err| match err {
            ModelError::NotFound(_) => {
                ValidationError::Usage(format!("Unknown model identifier: {model}"))
            }
            other => ValidationError::FailedValidation {
                model: model.clone(),
                reason: other.to_string(),
            },
        })?;

        let validation_result =
            validate_session_with_fixture_set(&mut session, &fixture_set, args.refresh_goldens);

        match validation_result {
            Ok(summary) => summaries.push(summary),
            Err(err @ ValidationError::Usage(_))
            | Err(err @ ValidationError::MissingFixtures(_)) => return Err(err.into()),
            Err(ValidationError::FailedValidation { reason, .. })
            | Err(ValidationError::ContractMismatch { reason, .. }) => {
                summaries.push(
                    ModelValidationSummary::failed(
                        model,
                        &fixture_set.manifest.evidence_timestamp,
                        reason,
                    )
                    .with_backend(session.backend()),
                );
            }
            Err(other) => {
                summaries.push(
                    ModelValidationSummary::failed(
                        model,
                        &fixture_set.manifest.evidence_timestamp,
                        other.to_string(),
                    )
                    .with_backend(session.backend()),
                );
            }
        }
    }

    render_validate_output(&args, &summaries)?;

    if summaries
        .iter()
        .all(|summary| matches!(summary.status, ValidationStatus::Validated))
    {
        Ok(())
    } else {
        Err(ValidationError::FailedValidation {
            model: args.models.join(","),
            reason: "One or more models did not reach validated status.".to_string(),
        }
        .into())
    }
}

fn render_validate_output(
    args: &ValidateArgs,
    summaries: &[ModelValidationSummary],
) -> Result<(), Error> {
    match args.format {
        OutputFormat::Terminal => crate::viz::terminal::print_validation_summaries(summaries),
        OutputFormat::Json => {
            if let Some(outdir) = &args.output {
                std::fs::create_dir_all(outdir)?;
                let path = outdir.join("validation.json");
                crate::viz::json::write_validation_report(summaries, &path)?;
                OutputArtifactManifest::new("validate", OutputFormat::Json)
                    .with_primary_artifact("validation.json")
                    .with_context(validate_manifest_context(args))
                    .with_summary(validate_manifest_summary(args, summaries))
                    .add_artifact("validation.json", ArtifactKind::Json, "Validation report")
                    .with_validation(summaries)
                    .write_to_dir(outdir)?;
                println!(
                    "Validation report written to {}/validation.json",
                    outdir.display()
                );
            } else {
                crate::viz::json::print_validation_summaries(summaries)?;
            }
        }
        OutputFormat::Html => {
            let outdir = args
                .output
                .clone()
                .unwrap_or_else(|| PathBuf::from("validation_output"));
            std::fs::create_dir_all(&outdir)?;
            let manifest = OutputArtifactManifest::new("validate", OutputFormat::Html)
                .with_primary_artifact("validation.html")
                .with_context(validate_manifest_context(args))
                .with_summary(validate_manifest_summary(args, summaries))
                .add_artifact("validation.html", ArtifactKind::Html, "Validation report")
                .add_artifact(
                    "validation.json",
                    ArtifactKind::Json,
                    "Validation report data",
                )
                .with_validation(summaries);
            crate::viz::json::write_validation_report(summaries, &outdir.join("validation.json"))?;
            crate::viz::html::write_validation_report_with_bundle(
                summaries,
                Some(&manifest),
                &outdir.join("validation.html"),
            )?;
            manifest.write_to_dir(&outdir)?;
            println!(
                "Validation report written to {}/validation.html",
                outdir.display()
            );
        }
        OutputFormat::Png => unreachable!("validated earlier"),
    }

    Ok(())
}

fn validate_manifest_context(args: &ValidateArgs) -> serde_json::Value {
    json!({
        "models": args.models,
        "fixture_set": args.fixture_set,
        "refresh_goldens": args.refresh_goldens,
    })
}

fn validate_manifest_summary(
    args: &ValidateArgs,
    summaries: &[ModelValidationSummary],
) -> serde_json::Value {
    let failed_models = summaries
        .iter()
        .filter(|summary| !matches!(summary.status, ValidationStatus::Validated))
        .map(|summary| summary.model.clone())
        .collect::<Vec<_>>();

    json!({
        "model_count": summaries.len(),
        "refresh_goldens": args.refresh_goldens,
        "evidence_timestamp": summaries.first().map(|summary| summary.evidence_timestamp.clone()),
        "failed_models": failed_models,
    })
}
