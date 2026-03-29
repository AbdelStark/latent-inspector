pub(crate) mod evidence;
pub mod fixtures;
pub mod freshness;
pub mod parity;
pub mod report;
pub mod semantics;

use crate::errors::{ModelError, ValidationError};
use crate::models::ModelSession;
use evidence::summarize_registered_evidence;
use fixtures::{build_reference_artifact_id, load_fixture_set, ContractArtifact, LoadedFixtureSet};
use parity::{build_reference_artifact, evaluate_reference_parity, summarize_outputs};
use report::ModelValidationSummary;
use semantics::{evaluate_preprocess_contract, evaluate_tensor_semantics};

pub use fixtures::{
    ContractFixture, FixturePattern, FixtureSignalSummary, MaterializedFixture, ReferenceArtifact,
    ReferenceSignals, ValidationFixtureManifest, ValidationFixtureSpec,
};
pub use report::{
    CheckSummary, ModelValidationSummary as ValidationSummary, ParitySignalDelta,
    ParityValidationSummary, TensorValidationSummary, ValidationStatus,
};

fn map_model_error(model: &str, err: ModelError) -> ValidationError {
    match err {
        ModelError::NotFound(_) => {
            ValidationError::Usage(format!("Unknown model identifier: {model}"))
        }
        other => ValidationError::FailedValidation {
            model: model.to_string(),
            reason: other.to_string(),
        },
    }
}

fn load_contract(
    fixture_set: &LoadedFixtureSet,
    model: &str,
) -> Result<ContractArtifact, ValidationError> {
    fixture_set.load_contract(model)
}

pub fn validate_session(
    session: &mut ModelSession,
    fixture_selection: Option<&str>,
    refresh_goldens: bool,
) -> Result<ModelValidationSummary, ValidationError> {
    let fixture_set = load_fixture_set(fixture_selection)?;
    validate_session_with_fixture_set(session, &fixture_set, refresh_goldens)
}

pub fn validate_model(
    model: &str,
    fixture_selection: Option<&str>,
    refresh_goldens: bool,
) -> Result<ModelValidationSummary, ValidationError> {
    let mut session = ModelSession::load(model).map_err(|err| map_model_error(model, err))?;
    validate_session(&mut session, fixture_selection, refresh_goldens)
}

pub fn summarize_session_or_unverified(
    session: &mut ModelSession,
    fixture_selection: Option<&str>,
) -> ModelValidationSummary {
    if !session.entry().is_ready() {
        return ModelValidationSummary::unverified(
            &session.info().name,
            &session.entry().validation.evidence_timestamp,
            format!(
                "{} is still a planned integration. Development reports may use stubbed outputs, but source-alignment validation remains withheld until the real ONNX path is wired.",
                session.info().name
            ),
        )
        .with_backend(session.backend());
    }

    match summarize_registered_evidence(session.entry(), fixture_selection) {
        Ok(summary) => summary.with_backend(session.backend()),
        Err(err) => ModelValidationSummary::unverified(
            &session.info().name,
            &session.entry().validation.evidence_timestamp,
            format!("Approved validation evidence could not be loaded: {err}"),
        )
        .with_backend(session.backend()),
    }
}

pub fn validate_session_with_fixture_set(
    session: &mut ModelSession,
    fixture_set: &LoadedFixtureSet,
    refresh_goldens: bool,
) -> Result<ModelValidationSummary, ValidationError> {
    let model = session.info().name.clone();
    let contract = load_contract(fixture_set, &model)?;
    let preprocess = evaluate_preprocess_contract(session.entry(), &contract, fixture_set)?;
    let fixtures = fixture_set.materialize_fixtures()?;

    if fixtures.is_empty() {
        return Err(ValidationError::MissingFixtures(format!(
            "Fixture set '{}' does not contain any fixture definitions",
            fixture_set.manifest.fixture_set
        )));
    }

    let mut outputs = Vec::with_capacity(fixtures.len());
    for fixture in &fixtures {
        let output =
            session
                .infer(&fixture.image)
                .map_err(|err| ValidationError::FailedValidation {
                    model: model.clone(),
                    reason: format!("Inference failed on fixture '{}': {err}", fixture.spec.id),
                })?;
        outputs.push(output);
    }

    let tensors = evaluate_tensor_semantics(session.entry(), &contract, fixture_set, &outputs[0]);
    let observed = summarize_outputs(fixtures.as_slice(), outputs.as_slice());

    let reference = if refresh_goldens {
        let artifact = build_reference_artifact(
            &model,
            &session.entry().validation,
            session.backend(),
            observed.clone(),
        );
        fixture_set.write_reference(&model, &artifact)?;
        artifact
    } else {
        fixture_set.load_reference(&model)?
    };

    let parity = evaluate_reference_parity(
        session.entry(),
        fixture_set,
        session.backend(),
        &observed,
        &reference,
    );
    let evidence_timestamp = reference.evidence_timestamp.clone();
    let artifact_id = reference.artifact_id.clone();
    let fixture_set_id = reference.fixture_set.clone();
    let summary = ModelValidationSummary::from_checks(
        model,
        evidence_timestamp,
        preprocess,
        tensors,
        parity.with_artifact(Some(artifact_id), Some(fixture_set_id)),
    )
    .with_backend(session.backend());

    Ok(summary)
}

pub fn default_artifact_id(model: &str, fixture_set: &str, evidence_timestamp: &str) -> String {
    build_reference_artifact_id(model, fixture_set, evidence_timestamp)
}
