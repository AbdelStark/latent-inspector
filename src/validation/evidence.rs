use crate::errors::ValidationError;
use crate::models::registry::RegistryEntry;
use crate::validation::fixtures::{load_fixture_set, LoadedFixtureSet};
use crate::validation::freshness::{
    parity_evidence_freshness, preprocess_evidence_freshness, tensor_evidence_freshness,
};
use crate::validation::report::{
    CheckSummary, ModelValidationSummary, ParityValidationSummary, TensorValidationSummary,
    ValidationStatus,
};

pub(crate) fn summarize_registered_evidence(
    entry: &RegistryEntry,
    fixture_selection: Option<&str>,
) -> Result<ModelValidationSummary, ValidationError> {
    let fixture_set = load_fixture_set(fixture_selection)?;
    summarize_registered_evidence_with_fixture_set(entry, &fixture_set)
}

pub(crate) fn summarize_registered_evidence_with_fixture_set(
    entry: &RegistryEntry,
    fixture_set: &LoadedFixtureSet,
) -> Result<ModelValidationSummary, ValidationError> {
    let contract = fixture_set.load_contract(&entry.info.name)?;
    let reference = fixture_set.load_reference(&entry.info.name)?;

    let preprocess = summarize_check(
        preprocess_evidence_freshness(entry, &contract, fixture_set).reasons(),
        "Approved preprocessing evidence matches the current registry contract.",
        "Approved preprocessing evidence is stale against the current registry contract",
    );
    let tensor = summarize_tensor_check(
        &contract.profile.tensor.name,
        &contract.profile.tensor.role.to_string(),
        tensor_evidence_freshness(entry, &contract, fixture_set).reasons(),
        "Approved tensor semantics evidence matches the current registry contract.",
        "Approved tensor semantics evidence is stale against the current registry contract",
    );
    let parity = summarize_parity_check(
        parity_evidence_freshness(entry, &reference, fixture_set).reasons(),
        "Approved reference parity evidence matches the current registry contract.",
        "Approved reference parity evidence is stale against the current registry contract",
    )
    .with_artifact(
        Some(reference.artifact_id.clone()),
        Some(reference.fixture_set.clone()),
    );

    Ok(ModelValidationSummary::from_checks(
        &entry.info.name,
        reference.evidence_timestamp,
        preprocess,
        vec![tensor],
        parity,
    ))
}

fn summarize_check(
    reasons: &[String],
    validated_summary: &str,
    stale_summary: &str,
) -> CheckSummary {
    let (status, summary) = summarize_status(reasons, validated_summary, stale_summary);
    CheckSummary::new(status, summary)
}

fn summarize_tensor_check(
    name: &str,
    role: &str,
    reasons: &[String],
    validated_summary: &str,
    stale_summary: &str,
) -> TensorValidationSummary {
    let (status, summary) = summarize_status(reasons, validated_summary, stale_summary);
    TensorValidationSummary {
        name: name.to_string(),
        role: role.to_string(),
        status,
        summary,
    }
}

fn summarize_parity_check(
    reasons: &[String],
    validated_summary: &str,
    stale_summary: &str,
) -> ParityValidationSummary {
    let (status, summary) = summarize_status(reasons, validated_summary, stale_summary);
    ParityValidationSummary::new(status, summary)
}

fn summarize_status(
    reasons: &[String],
    validated_summary: &str,
    stale_summary: &str,
) -> (ValidationStatus, String) {
    if reasons.is_empty() {
        (ValidationStatus::Validated, validated_summary.to_string())
    } else {
        (
            ValidationStatus::Stale,
            format!("{stale_summary}: {}", reasons.join("; ")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("validation")
    }

    fn copy_fixture_dir() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        for entry in fs::read_dir(fixture_root()).unwrap() {
            let entry = entry.unwrap();
            let src = entry.path();
            let dest = dir.path().join(entry.file_name());
            if src.is_file() {
                fs::copy(src, dest).unwrap();
            }
        }
        dir
    }

    fn read_json(path: &Path) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn registered_evidence_summary_reports_validated_ready_model() {
        let fixture_set = load_fixture_set(None).unwrap();
        let entry = crate::models::registry::find("dinov2-vit-l14").unwrap();

        let summary = summarize_registered_evidence_with_fixture_set(&entry, &fixture_set).unwrap();

        assert_eq!(summary.status, ValidationStatus::Validated);
        assert_eq!(summary.preprocess.status, ValidationStatus::Validated);
        assert_eq!(summary.tensors[0].status, ValidationStatus::Validated);
        assert_eq!(summary.parity.status, ValidationStatus::Validated);
        assert!(summary.caveats.is_empty());
        assert_eq!(
            summary.parity.artifact_id.as_deref(),
            Some("dinov2-vit-l14:standard:2026-03-27T12:00:00Z")
        );
    }

    #[test]
    fn registered_evidence_summary_flags_stale_contract_artifact() {
        let fixtures = copy_fixture_dir();
        let contract_path = fixtures.path().join("dinov2-vit-l14.contract.json");
        let mut contract = read_json(&contract_path);
        contract["profile"]["evidence_timestamp"] = Value::from("2026-03-28T00:00:00Z");
        fs::write(
            &contract_path,
            serde_json::to_string_pretty(&contract).unwrap(),
        )
        .unwrap();

        let manifest_path = fixtures.path().join("manifest.json");
        let fixture_set = load_fixture_set(Some(manifest_path.to_str().unwrap())).unwrap();
        let entry = crate::models::registry::find("dinov2-vit-l14").unwrap();

        let summary = summarize_registered_evidence_with_fixture_set(&entry, &fixture_set).unwrap();

        assert_eq!(summary.status, ValidationStatus::Stale);
        assert_eq!(summary.preprocess.status, ValidationStatus::Stale);
        assert_eq!(summary.tensors[0].status, ValidationStatus::Stale);
        assert_eq!(summary.parity.status, ValidationStatus::Validated);
        assert!(summary
            .caveats
            .iter()
            .any(|caveat| caveat.contains("artifact evidence timestamp")));
    }
}
