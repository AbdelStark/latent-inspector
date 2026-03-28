use crate::models::cache;
use crate::models::registry::{self, AvailabilityStatus, RegistryEntry, SSLMethod};
use crate::validation::fixtures::{load_fixture_set, LoadedFixtureSet};
use crate::validation::freshness::{
    parity_evidence_freshness, preprocess_evidence_freshness, tensor_evidence_freshness,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheStatus {
    Complete,
    Missing,
    Unknown,
}

impl CacheStatus {
    pub fn label(self) -> &'static str {
        match self {
            CacheStatus::Complete => "complete",
            CacheStatus::Missing => "missing",
            CacheStatus::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvidenceStatus {
    Approved,
    Stale,
    Missing,
    Unverified,
}

impl EvidenceStatus {
    pub fn label(self) -> &'static str {
        match self {
            EvidenceStatus::Approved => "approved",
            EvidenceStatus::Stale => "stale",
            EvidenceStatus::Missing => "missing",
            EvidenceStatus::Unverified => "unverified",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeSupport {
    OnnxReady,
    StubOnly,
}

impl RuntimeSupport {
    pub fn label(self) -> &'static str {
        match self {
            RuntimeSupport::OnnxReady => "onnx-ready",
            RuntimeSupport::StubOnly => "stub-only",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactInventory {
    pub relative_path: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInventoryEntry {
    pub name: String,
    pub availability_status: AvailabilityStatus,
    pub phase: String,
    pub availability_note: String,
    pub runtime_support: RuntimeSupport,
    pub runtime_summary: String,
    pub method: SSLMethod,
    pub params_m: u32,
    pub architecture: String,
    pub input_size: u32,
    pub embed_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub verification_label: String,
    pub verification_note: Option<String>,
    pub cache_status: CacheStatus,
    pub cache_summary: String,
    pub evidence_status: EvidenceStatus,
    pub evidence_summary: String,
    pub evidence_details: Vec<String>,
    pub approved_fixture_set: String,
    pub approved_evidence_timestamp: String,
    pub artifacts: Vec<ModelArtifactInventory>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceStatusCounts {
    pub approved: usize,
    pub stale: usize,
    pub missing: usize,
    pub unverified: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalogSummary {
    pub total_models: usize,
    pub ready_models: usize,
    pub planned_models: usize,
    pub cached_models: usize,
    pub evidence: EvidenceStatusCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalogReport {
    pub fixture_set: Option<String>,
    pub evidence_timestamp: Option<String>,
    pub fixture_error: Option<String>,
    pub summary: ModelCatalogSummary,
    pub entries: Vec<ModelInventoryEntry>,
}

impl ModelCatalogReport {
    pub fn evidence_count(&self, status: EvidenceStatus) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.evidence_status == status)
            .count()
    }

    pub fn cache_count(&self, status: CacheStatus) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.cache_status == status)
            .count()
    }

    pub fn ready_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.availability_status == AvailabilityStatus::Ready)
            .count()
    }

    pub fn planned_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.availability_status == AvailabilityStatus::Planned)
            .count()
    }

    fn build_summary(&self) -> ModelCatalogSummary {
        ModelCatalogSummary {
            total_models: self.entries.len(),
            ready_models: self.ready_count(),
            planned_models: self.planned_count(),
            cached_models: self.cache_count(CacheStatus::Complete),
            evidence: EvidenceStatusCounts {
                approved: self.evidence_count(EvidenceStatus::Approved),
                stale: self.evidence_count(EvidenceStatus::Stale),
                missing: self.evidence_count(EvidenceStatus::Missing),
                unverified: self.evidence_count(EvidenceStatus::Unverified),
            },
        }
    }
}

pub fn build_model_catalog(fixture_selection: Option<&str>) -> ModelCatalogReport {
    let fixture_result = load_fixture_set(fixture_selection);
    let (fixture_set, fixture_error, fixture_name, evidence_timestamp) = match fixture_result {
        Ok(fixture_set) => {
            let fixture_name = Some(fixture_set.manifest.fixture_set.clone());
            let evidence_timestamp = Some(fixture_set.manifest.evidence_timestamp.clone());
            (Some(fixture_set), None, fixture_name, evidence_timestamp)
        }
        Err(err) => (None, Some(err.to_string()), None, None),
    };

    let entries = registry::registry()
        .into_iter()
        .map(|entry| build_inventory_entry(&entry, fixture_set.as_ref(), fixture_error.as_deref()))
        .collect();

    let mut report = ModelCatalogReport {
        fixture_set: fixture_name,
        evidence_timestamp,
        fixture_error,
        summary: ModelCatalogSummary {
            total_models: 0,
            ready_models: 0,
            planned_models: 0,
            cached_models: 0,
            evidence: EvidenceStatusCounts {
                approved: 0,
                stale: 0,
                missing: 0,
                unverified: 0,
            },
        },
        entries,
    };
    report.summary = report.build_summary();
    report
}

fn build_inventory_entry(
    entry: &RegistryEntry,
    fixture_set: Option<&LoadedFixtureSet>,
    fixture_error: Option<&str>,
) -> ModelInventoryEntry {
    let (runtime_support, runtime_summary) = runtime_support(entry);
    let (cache_status, cache_summary) = match cache::is_cached(&entry.info.name) {
        Ok(true) => (
            CacheStatus::Complete,
            "Required artifact bundle is present in the local cache.".to_string(),
        ),
        Ok(false) => (
            CacheStatus::Missing,
            "Required artifact bundle is missing or incomplete in the local cache.".to_string(),
        ),
        Err(err) => (
            CacheStatus::Unknown,
            format!("Cache state could not be determined: {err}"),
        ),
    };

    let (evidence_status, evidence_summary, evidence_details) =
        assess_evidence(entry, fixture_set, fixture_error);

    ModelInventoryEntry {
        name: entry.info.name.clone(),
        availability_status: entry.availability.status.clone(),
        phase: entry.availability.phase.clone(),
        availability_note: entry.availability.note.clone(),
        runtime_support,
        runtime_summary,
        method: entry.info.method.clone(),
        params_m: entry.info.params_m,
        architecture: entry.info.architecture.clone(),
        input_size: entry.info.input_size,
        embed_dim: entry.info.embed_dim,
        num_layers: entry.info.num_layers,
        num_heads: entry.info.num_heads,
        verification_label: entry.verification_label().to_string(),
        verification_note: entry.verification_note().map(str::to_string),
        cache_status,
        cache_summary,
        evidence_status,
        evidence_summary,
        evidence_details,
        approved_fixture_set: entry.validation.fixture_set.clone(),
        approved_evidence_timestamp: entry.validation.evidence_timestamp.clone(),
        artifacts: entry
            .artifacts
            .iter()
            .map(|artifact| ModelArtifactInventory {
                relative_path: artifact.relative_path.clone(),
                url: artifact.download_url.clone(),
            })
            .collect(),
    }
}

fn runtime_support(entry: &RegistryEntry) -> (RuntimeSupport, String) {
    if entry.is_ready() {
        (
            RuntimeSupport::OnnxReady,
            "Normal runs load the registered ONNX artifact. The stub backend remains available only when explicitly forced for development workflows.".to_string(),
        )
    } else {
        (
            RuntimeSupport::StubOnly,
            "Normal runs remain blocked until this integration is promoted to ready. Only the development stub backend can be used for analysis scaffolding.".to_string(),
        )
    }
}

fn assess_evidence(
    entry: &RegistryEntry,
    fixture_set: Option<&LoadedFixtureSet>,
    fixture_error: Option<&str>,
) -> (EvidenceStatus, String, Vec<String>) {
    if !entry.is_ready() {
        return (
            EvidenceStatus::Unverified,
            "Validation evidence is intentionally withheld until this integration is promoted from planned to ready.".to_string(),
            vec![entry.availability.note.clone()],
        );
    }

    let Some(fixture_set) = fixture_set else {
        let mut details = Vec::new();
        if let Some(error) = fixture_error {
            details.push(error.to_string());
        }
        return (
            EvidenceStatus::Missing,
            "Validation evidence could not be inspected because the fixture manifest was unavailable.".to_string(),
            details,
        );
    };

    let contract = match fixture_set.load_contract(&entry.info.name) {
        Ok(contract) => contract,
        Err(err) => {
            return (
                EvidenceStatus::Missing,
                "Approved validation contract could not be loaded from the fixture set."
                    .to_string(),
                vec![err.to_string()],
            );
        }
    };

    let reference = match fixture_set.load_reference(&entry.info.name) {
        Ok(reference) => reference,
        Err(err) => {
            return (
                EvidenceStatus::Missing,
                "Approved parity reference artifact could not be loaded from the fixture set."
                    .to_string(),
                vec![err.to_string()],
            );
        }
    };

    let mut stale_reasons = Vec::new();
    extend_unique(
        &mut stale_reasons,
        preprocess_evidence_freshness(entry, &contract, fixture_set).reasons(),
    );
    extend_unique(
        &mut stale_reasons,
        tensor_evidence_freshness(entry, &contract, fixture_set).reasons(),
    );
    extend_unique(
        &mut stale_reasons,
        parity_evidence_freshness(entry, &reference, fixture_set).reasons(),
    );

    if stale_reasons.is_empty() {
        (
            EvidenceStatus::Approved,
            "Approved validation contract and parity artifacts are current for the active registry profile.".to_string(),
            Vec::new(),
        )
    } else {
        (
            EvidenceStatus::Stale,
            "Approved validation evidence is stale against the active registry profile."
                .to_string(),
            stale_reasons,
        )
    }
}

fn extend_unique(target: &mut Vec<String>, items: &[String]) {
    for item in items {
        if !target.iter().any(|existing| existing == item) {
            target.push(item.clone());
        }
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
    fn ready_model_has_approved_evidence_in_default_catalog() {
        let report = build_model_catalog(None);
        let dinov2 = report
            .entries
            .iter()
            .find(|entry| entry.name == "dinov2-vit-l14")
            .unwrap();

        assert_eq!(dinov2.evidence_status, EvidenceStatus::Approved);
        assert!(dinov2.evidence_details.is_empty());
        assert_eq!(report.summary.total_models, report.entries.len());
        assert_eq!(report.summary.ready_models, 1);
        assert_eq!(report.summary.evidence.approved, 1);
    }

    #[test]
    fn planned_models_remain_unverified_even_with_reference_artifacts() {
        let report = build_model_catalog(None);
        let planned = report
            .entries
            .iter()
            .find(|entry| entry.name == "mae-vit-l16")
            .unwrap();

        assert_eq!(planned.evidence_status, EvidenceStatus::Unverified);
        assert!(planned.evidence_summary.contains("intentionally withheld"));
    }

    #[test]
    fn stale_contract_marks_ready_model_as_stale() {
        let fixtures = copy_fixture_dir();
        let contract_path = fixtures.path().join("dinov2-vit-l14.contract.json");
        let mut contract = read_json(&contract_path);
        contract["profile"]["evidence_timestamp"] = Value::from("2026-03-28T00:00:00Z");
        fs::write(
            &contract_path,
            serde_json::to_string_pretty(&contract).unwrap(),
        )
        .unwrap();

        let manifest = fixtures.path().join("manifest.json");
        let report = build_model_catalog(Some(manifest.to_str().unwrap()));
        let dinov2 = report
            .entries
            .iter()
            .find(|entry| entry.name == "dinov2-vit-l14")
            .unwrap();

        assert_eq!(dinov2.evidence_status, EvidenceStatus::Stale);
        assert!(dinov2
            .evidence_details
            .iter()
            .any(|detail| detail.contains("artifact evidence timestamp")));
    }

    #[test]
    fn missing_fixture_manifest_marks_ready_model_as_missing() {
        let report = build_model_catalog(Some("/tmp/latent-inspector-missing-manifest.json"));
        let dinov2 = report
            .entries
            .iter()
            .find(|entry| entry.name == "dinov2-vit-l14")
            .unwrap();

        assert!(report.fixture_error.is_some());
        assert_eq!(dinov2.evidence_status, EvidenceStatus::Missing);
        assert!(dinov2
            .evidence_summary
            .contains("fixture manifest was unavailable"));
    }
}
