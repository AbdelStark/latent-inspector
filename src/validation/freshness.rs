use crate::models::registry::RegistryEntry;
use crate::models::InferenceBackend;
use crate::validation::fixtures::{
    build_reference_artifact_id, ContractArtifact, LoadedFixtureSet, ReferenceArtifact,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreshnessReport {
    reasons: Vec<String>,
}

impl FreshnessReport {
    pub fn new(reasons: Vec<String>) -> Self {
        Self { reasons }
    }

    pub fn is_stale(&self) -> bool {
        !self.reasons.is_empty()
    }

    pub fn reasons(&self) -> &[String] {
        &self.reasons
    }
}

pub fn preprocess_evidence_freshness(
    entry: &RegistryEntry,
    contract: &ContractArtifact,
    fixture_set: &LoadedFixtureSet,
) -> FreshnessReport {
    let mut reasons = shared_profile_reasons(
        entry,
        &contract.model,
        &contract.profile.source,
        &contract.profile.fixture_set,
        &contract.profile.evidence_timestamp,
        fixture_set,
    );

    if contract.profile.preprocess != entry.validation.preprocess {
        reasons.push(
            "approved preprocessing evidence no longer matches the current registry contract"
                .to_string(),
        );
    }

    FreshnessReport::new(reasons)
}

pub fn tensor_evidence_freshness(
    entry: &RegistryEntry,
    contract: &ContractArtifact,
    fixture_set: &LoadedFixtureSet,
) -> FreshnessReport {
    let mut reasons = shared_profile_reasons(
        entry,
        &contract.model,
        &contract.profile.source,
        &contract.profile.fixture_set,
        &contract.profile.evidence_timestamp,
        fixture_set,
    );

    if contract.profile.tensor != entry.validation.tensor {
        reasons.push(
            "approved tensor semantics evidence no longer matches the current registry contract"
                .to_string(),
        );
    }

    FreshnessReport::new(reasons)
}

pub fn parity_evidence_freshness(
    entry: &RegistryEntry,
    reference: &ReferenceArtifact,
    fixture_set: &LoadedFixtureSet,
) -> FreshnessReport {
    let mut reasons = shared_profile_reasons(
        entry,
        &reference.model,
        &reference.source,
        &reference.fixture_set,
        &reference.evidence_timestamp,
        fixture_set,
    );

    let expected_artifact_id = build_reference_artifact_id(
        &entry.info.name,
        &entry.validation.fixture_set,
        &entry.validation.evidence_timestamp,
    );
    if reference.artifact_id != expected_artifact_id {
        reasons.push(format!(
            "reference artifact id '{}' does not match the current approved identity '{}'",
            reference.artifact_id, expected_artifact_id
        ));
    }

    if reference.tolerances != entry.validation.tolerances {
        reasons.push(
            "approved parity tolerances no longer match the current registry contract".to_string(),
        );
    }

    let expected_backend = if entry.is_ready() {
        InferenceBackend::OnnxRuntime
    } else {
        InferenceBackend::Stub
    };
    if reference.backend != expected_backend {
        reasons.push(format!(
            "reference backend '{}' does not match the expected '{}' execution path for this model",
            reference.backend, expected_backend
        ));
    }

    FreshnessReport::new(reasons)
}

fn shared_profile_reasons(
    entry: &RegistryEntry,
    artifact_model: &str,
    artifact_source: &str,
    artifact_fixture_set: &str,
    artifact_timestamp: &str,
    fixture_set: &LoadedFixtureSet,
) -> Vec<String> {
    let expected = &entry.validation;
    let mut reasons = Vec::new();

    if artifact_model != entry.info.name {
        reasons.push(format!(
            "artifact model '{}' does not match the requested model '{}'",
            artifact_model, entry.info.name
        ));
    }

    if artifact_source != expected.source {
        reasons.push(format!(
            "artifact source '{}' does not match the current registry source '{}'",
            artifact_source, expected.source
        ));
    }

    if artifact_fixture_set != expected.fixture_set {
        reasons.push(format!(
            "artifact fixture set '{}' does not match the current registry fixture set '{}'",
            artifact_fixture_set, expected.fixture_set
        ));
    }

    if fixture_set.manifest.fixture_set != expected.fixture_set {
        reasons.push(format!(
            "fixture manifest '{}' does not match the current registry fixture set '{}'",
            fixture_set.manifest.fixture_set, expected.fixture_set
        ));
    }

    if artifact_timestamp != expected.evidence_timestamp {
        reasons.push(format!(
            "artifact evidence timestamp '{}' does not match the current registry timestamp '{}'",
            artifact_timestamp, expected.evidence_timestamp
        ));
    }

    if fixture_set.manifest.evidence_timestamp != expected.evidence_timestamp {
        reasons.push(format!(
            "fixture manifest timestamp '{}' does not match the current registry timestamp '{}'",
            fixture_set.manifest.evidence_timestamp, expected.evidence_timestamp
        ));
    }

    reasons
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry;
    use crate::validation::fixtures::load_fixture_set;

    #[test]
    fn contract_freshness_flags_outdated_preprocess_evidence() {
        let fixture_set = load_fixture_set(None).unwrap();
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut contract = fixture_set.load_contract("dinov2-vit-l14").unwrap();
        contract.profile.evidence_timestamp = "2026-03-28T00:00:00Z".to_string();

        let freshness = preprocess_evidence_freshness(&entry, &contract, &fixture_set);
        assert!(freshness.is_stale());
        assert!(freshness
            .reasons()
            .iter()
            .any(|reason| reason.contains("artifact evidence timestamp")));
    }

    #[test]
    fn tensor_freshness_flags_contract_tensor_drift() {
        let fixture_set = load_fixture_set(None).unwrap();
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut contract = fixture_set.load_contract("dinov2-vit-l14").unwrap();
        contract.profile.tensor.embedding_dim += 1;

        let freshness = tensor_evidence_freshness(&entry, &contract, &fixture_set);
        assert!(freshness.is_stale());
        assert!(freshness
            .reasons()
            .iter()
            .any(|reason| reason.contains("tensor semantics evidence")));
    }

    #[test]
    fn parity_freshness_flags_reference_identity_drift() {
        let fixture_set = load_fixture_set(None).unwrap();
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
        reference.artifact_id = "unexpected".to_string();

        let freshness = parity_evidence_freshness(&entry, &reference, &fixture_set);
        assert!(freshness.is_stale());
        assert!(freshness
            .reasons()
            .iter()
            .any(|reason| reason.contains("reference artifact id")));
    }

    #[test]
    fn parity_freshness_flags_reference_backend_drift() {
        let fixture_set = load_fixture_set(None).unwrap();
        let entry = registry::find("dinov2-vit-l14").unwrap();
        let mut reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
        reference.backend = InferenceBackend::Stub;

        let freshness = parity_evidence_freshness(&entry, &reference, &fixture_set);
        assert!(freshness.is_stale());
        assert!(freshness
            .reasons()
            .iter()
            .any(|reason| reason.contains("reference backend")));
    }
}
