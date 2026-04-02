use crate::models::cache::{self, ArtifactCacheStatus};
use crate::models::registry::{self, AvailabilityStatus, RegistryEntry, SSLMethod};
use crate::validation::evidence::summarize_registered_evidence_with_fixture_set;
use crate::validation::fixtures::{load_fixture_set, LoadedFixtureSet};
use crate::validation::ValidationStatus;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelReadinessStatus {
    Ready,
    NeedsDownload,
    NeedsEvidenceRefresh,
    NeedsValidation,
    Planned,
    Blocked,
}

impl ModelReadinessStatus {
    pub fn label(self) -> &'static str {
        match self {
            ModelReadinessStatus::Ready => "ready",
            ModelReadinessStatus::NeedsDownload => "needs-download",
            ModelReadinessStatus::NeedsEvidenceRefresh => "needs-evidence-refresh",
            ModelReadinessStatus::NeedsValidation => "needs-validation",
            ModelReadinessStatus::Planned => "planned",
            ModelReadinessStatus::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReadinessCounts {
    pub ready: usize,
    pub needs_download: usize,
    pub needs_evidence_refresh: usize,
    pub needs_validation: usize,
    pub planned: usize,
    pub blocked: usize,
}

impl ModelReadinessCounts {
    fn record(&mut self, status: ModelReadinessStatus) {
        match status {
            ModelReadinessStatus::Ready => self.ready += 1,
            ModelReadinessStatus::NeedsDownload => self.needs_download += 1,
            ModelReadinessStatus::NeedsEvidenceRefresh => self.needs_evidence_refresh += 1,
            ModelReadinessStatus::NeedsValidation => self.needs_validation += 1,
            ModelReadinessStatus::Planned => self.planned += 1,
            ModelReadinessStatus::Blocked => self.blocked += 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactInventory {
    pub relative_path: String,
    pub url: String,
    pub absolute_path: String,
    pub cache_status: ArtifactCacheStatus,
    pub cache_summary: String,
    pub byte_size: Option<u64>,
    pub verification_label: String,
    pub verification_note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelDownloadAction {
    Downloaded,
    AlreadyCached,
}

impl ModelDownloadAction {
    pub fn label(self) -> &'static str {
        match self {
            ModelDownloadAction::Downloaded => "downloaded",
            ModelDownloadAction::AlreadyCached => "already-cached",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactTransition {
    pub relative_path: String,
    pub previous_status: ArtifactCacheStatus,
    pub current_status: ArtifactCacheStatus,
    pub downloaded: bool,
    pub repaired: bool,
    pub byte_size: Option<u64>,
    pub cache_summary: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactInventorySummary {
    pub total: usize,
    pub usable: usize,
    pub verified: usize,
    pub pending_verification: usize,
    pub missing: usize,
    pub invalid: usize,
    pub unusable: usize,
    pub unknown: usize,
}

impl ArtifactInventorySummary {
    fn from_artifacts(artifacts: &[ModelArtifactInventory]) -> Self {
        Self::from_statuses(artifacts.iter().map(|artifact| artifact.cache_status))
    }

    fn from_statuses<I>(statuses: I) -> Self
    where
        I: IntoIterator<Item = ArtifactCacheStatus>,
    {
        let mut summary = Self::default();
        for status in statuses {
            summary.total += 1;
            match status {
                ArtifactCacheStatus::PresentVerified => {
                    summary.usable += 1;
                    summary.verified += 1;
                }
                ArtifactCacheStatus::PresentUnverified => {
                    summary.usable += 1;
                    summary.pending_verification += 1;
                }
                ArtifactCacheStatus::Missing => summary.missing += 1,
                ArtifactCacheStatus::Invalid => summary.invalid += 1,
                ArtifactCacheStatus::Unusable => summary.unusable += 1,
                ArtifactCacheStatus::Unknown => summary.unknown += 1,
            }
        }
        summary
    }

    fn merge(&mut self, other: &Self) {
        self.total += other.total;
        self.usable += other.usable;
        self.verified += other.verified;
        self.pending_verification += other.pending_verification;
        self.missing += other.missing;
        self.invalid += other.invalid;
        self.unusable += other.unusable;
        self.unknown += other.unknown;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInventoryEntry {
    pub name: String,
    pub availability_status: AvailabilityStatus,
    pub phase: String,
    pub availability_note: String,
    pub readiness_status: ModelReadinessStatus,
    pub readiness_summary: String,
    pub next_steps: Vec<String>,
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
    pub artifact_summary: ArtifactInventorySummary,
    pub evidence_status: EvidenceStatus,
    pub evidence_summary: String,
    pub evidence_details: Vec<String>,
    pub approved_fixture_set: String,
    pub approved_evidence_timestamp: String,
    pub artifacts: Vec<ModelArtifactInventory>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceStatusCounts {
    pub approved: usize,
    pub stale: usize,
    pub missing: usize,
    pub unverified: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalogSummary {
    pub total_models: usize,
    pub ready_models: usize,
    pub planned_models: usize,
    pub cached_models: usize,
    pub readiness: ModelReadinessCounts,
    pub artifacts: ArtifactInventorySummary,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadReport {
    pub model: String,
    pub action: ModelDownloadAction,
    pub summary: String,
    pub entry: ModelInventoryEntry,
    pub artifact_changes: Vec<ModelArtifactTransition>,
}

impl ModelDownloadReport {
    pub fn downloaded_artifact_count(&self) -> usize {
        self.artifact_changes
            .iter()
            .filter(|artifact| artifact.downloaded)
            .count()
    }

    pub fn repaired_artifact_count(&self) -> usize {
        self.artifact_changes
            .iter()
            .filter(|artifact| artifact.repaired)
            .count()
    }
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

    pub fn filter_to_names(mut self, names: &[String]) -> Self {
        if names.is_empty() {
            return self;
        }

        self.entries
            .retain(|entry| names.iter().any(|name| name == &entry.name));
        self.summary = self.build_summary();
        self
    }

    fn build_summary(&self) -> ModelCatalogSummary {
        let mut artifacts = ArtifactInventorySummary::default();
        let mut readiness = ModelReadinessCounts::default();
        for entry in &self.entries {
            artifacts.merge(&entry.artifact_summary);
            readiness.record(entry.readiness_status);
        }

        ModelCatalogSummary {
            total_models: self.entries.len(),
            ready_models: self.ready_count(),
            planned_models: self.planned_count(),
            cached_models: self.cache_count(CacheStatus::Complete),
            readiness,
            artifacts,
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
        // Placeholder — immediately replaced by build_summary() below.
        summary: ModelCatalogSummary::default(),
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
    let abr = build_artifact_inventory(entry);

    let (evidence_status, evidence_summary, evidence_details) =
        assess_evidence(entry, fixture_set, fixture_error);
    let (readiness_status, readiness_summary, next_steps) = assess_readiness(
        entry,
        abr.cache_status,
        &abr.artifact_summary,
        evidence_status,
        abr.verification_note.as_deref(),
    );

    ModelInventoryEntry {
        name: entry.info.name.clone(),
        availability_status: entry.availability.status.clone(),
        phase: entry.availability.phase.clone(),
        availability_note: entry.availability.note.clone(),
        readiness_status,
        readiness_summary,
        next_steps,
        runtime_support,
        runtime_summary,
        method: entry.info.method.clone(),
        params_m: entry.info.params_m,
        architecture: entry.info.architecture.clone(),
        input_size: entry.info.input_size,
        embed_dim: entry.info.embed_dim,
        num_layers: entry.info.num_layers,
        num_heads: entry.info.num_heads,
        verification_label: abr.verification_label,
        verification_note: abr.verification_note,
        cache_status: abr.cache_status,
        cache_summary: abr.cache_summary,
        artifact_summary: abr.artifact_summary,
        evidence_status,
        evidence_summary,
        evidence_details,
        approved_fixture_set: entry.validation.fixture_set.clone(),
        approved_evidence_timestamp: entry.validation.evidence_timestamp.clone(),
        artifacts: abr.artifacts,
    }
}

struct ArtifactBuildResult {
    artifacts: Vec<ModelArtifactInventory>,
    artifact_summary: ArtifactInventorySummary,
    cache_status: CacheStatus,
    cache_summary: String,
    verification_label: String,
    verification_note: Option<String>,
}

fn build_artifact_inventory(entry: &RegistryEntry) -> ArtifactBuildResult {
    if entry.artifacts.is_empty() {
        return ArtifactBuildResult {
            artifacts: Vec::new(),
            artifact_summary: ArtifactInventorySummary::default(),
            cache_status: CacheStatus::Missing,
            cache_summary: "No download artifacts are pinned for this registry entry yet."
                .to_string(),
            verification_label: "pending".to_string(),
            verification_note: Some(
                "Artifact metadata has not been pinned for this integration yet.".to_string(),
            ),
        };
    }

    let (artifacts, cache_status, cache_summary) = match cache::inspect_registry_artifacts(entry) {
        Ok(artifacts) => {
            let inventories = artifacts
                .into_iter()
                .map(|artifact| ModelArtifactInventory {
                    relative_path: artifact.relative_path,
                    url: artifact.url,
                    absolute_path: artifact.absolute_path,
                    cache_status: artifact.cache_status,
                    cache_summary: artifact.cache_summary,
                    byte_size: artifact.byte_size,
                    verification_label: artifact.verification_label,
                    verification_note: artifact.verification_note,
                })
                .collect::<Vec<_>>();
            let summary = ArtifactInventorySummary::from_artifacts(&inventories);
            let status = if summary.total > 0 && summary.usable == summary.total {
                CacheStatus::Complete
            } else {
                CacheStatus::Missing
            };
            let summary_text = render_cache_summary(&summary);
            (inventories, status, summary_text)
        }
        Err(err) => (
            build_unknown_artifacts(entry, &err.to_string()),
            CacheStatus::Unknown,
            format!("Cache state could not be determined: {err}"),
        ),
    };

    let artifact_summary = ArtifactInventorySummary::from_artifacts(&artifacts);
    let (verification_label, verification_note) = summarize_verification(&artifacts);

    ArtifactBuildResult {
        artifacts,
        artifact_summary,
        cache_status,
        cache_summary,
        verification_label,
        verification_note,
    }
}

fn build_unknown_artifacts(entry: &RegistryEntry, reason: &str) -> Vec<ModelArtifactInventory> {
    let cache_root = cache::cache_dir().ok();
    entry
        .artifacts
        .iter()
        .map(|artifact| ModelArtifactInventory {
            relative_path: artifact.relative_path.clone(),
            url: artifact.download_url.clone(),
            absolute_path: cache_root
                .as_ref()
                .map(|root| root.join(&artifact.relative_path).display().to_string())
                .unwrap_or_else(|| artifact.relative_path.clone()),
            cache_status: ArtifactCacheStatus::Unknown,
            cache_summary: format!("Cache inspection failed: {reason}"),
            byte_size: None,
            verification_label: artifact.checksum.label().to_string(),
            verification_note: artifact.checksum.note().map(str::to_string),
        })
        .collect()
}

fn summarize_verification(artifacts: &[ModelArtifactInventory]) -> (String, Option<String>) {
    if artifacts.is_empty() {
        return ("pending".to_string(), None);
    }

    let mut labels = Vec::new();
    let mut notes = Vec::new();
    for artifact in artifacts {
        push_unique(&mut labels, artifact.verification_label.clone());
        if let Some(note) = &artifact.verification_note {
            push_unique(&mut notes, note.clone());
        }
    }

    let label = if labels.len() == 1 {
        labels.remove(0)
    } else {
        "mixed".to_string()
    };
    let note = (!notes.is_empty()).then(|| notes.join(" | "));

    (label, note)
}

fn push_unique(items: &mut Vec<String>, value: String) {
    if !items.iter().any(|existing| existing == &value) {
        items.push(value);
    }
}

fn render_cache_summary(summary: &ArtifactInventorySummary) -> String {
    if summary.total == 0 {
        return "No download artifacts are pinned for this registry entry yet.".to_string();
    }

    if summary.unknown > 0 {
        return format!(
            "Cache inspection is incomplete: {} of {} artifact states are unknown.",
            summary.unknown, summary.total
        );
    }

    if summary.usable == summary.total {
        if summary.pending_verification > 0 {
            return format!(
                "All {} artifacts are usable; {} checksum-verified and {} pending verification metadata.",
                summary.total, summary.verified, summary.pending_verification
            );
        }

        return format!(
            "All {} artifacts are usable and checksum-verified.",
            summary.total
        );
    }

    format!(
        "{} of {} artifacts are usable ({} missing, {} invalid, {} unusable).",
        summary.usable, summary.total, summary.missing, summary.invalid, summary.unusable
    )
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

    match summarize_registered_evidence_with_fixture_set(entry, fixture_set) {
        Ok(summary) => match summary.status {
            ValidationStatus::Validated => (
                EvidenceStatus::Approved,
                "Approved validation contract and parity artifacts are current for the active registry profile.".to_string(),
                Vec::new(),
            ),
            ValidationStatus::Stale => (
                EvidenceStatus::Stale,
                "Approved validation evidence is stale against the active registry profile."
                    .to_string(),
                summary.caveats,
            ),
            ValidationStatus::Failed | ValidationStatus::Partial | ValidationStatus::Unverified => (
                EvidenceStatus::Missing,
                "Validation evidence could not be fully interpreted from the fixture set."
                    .to_string(),
                summary.caveats,
            ),
        },
        Err(err) => (
            EvidenceStatus::Missing,
            "Approved validation evidence could not be loaded from the fixture set."
                .to_string(),
            vec![err.to_string()],
        ),
    }
}

pub fn build_model_download_report(
    action: ModelDownloadAction,
    previous_artifacts: &[cache::CachedArtifactInfo],
    entry: ModelInventoryEntry,
) -> ModelDownloadReport {
    let artifact_changes = entry
        .artifacts
        .iter()
        .map(|artifact| {
            let previous_status = previous_artifacts
                .iter()
                .find(|candidate| candidate.relative_path == artifact.relative_path)
                .map(|candidate| candidate.cache_status)
                .unwrap_or(ArtifactCacheStatus::Missing);
            let current_status = artifact.cache_status;
            let downloaded = !previous_status.is_usable() && current_status.is_usable();
            let repaired = matches!(previous_status, ArtifactCacheStatus::Invalid)
                && current_status.is_usable();

            ModelArtifactTransition {
                relative_path: artifact.relative_path.clone(),
                previous_status,
                current_status,
                downloaded,
                repaired,
                byte_size: artifact.byte_size,
                cache_summary: artifact.cache_summary.clone(),
            }
        })
        .collect::<Vec<_>>();

    let downloaded_count = artifact_changes
        .iter()
        .filter(|artifact| artifact.downloaded)
        .count();
    let repaired_count = artifact_changes
        .iter()
        .filter(|artifact| artifact.repaired)
        .count();
    let summary = match action {
        ModelDownloadAction::AlreadyCached => format!(
            "Cache already contained the full artifact bundle for {}. {}",
            entry.name, entry.readiness_summary
        ),
        ModelDownloadAction::Downloaded => {
            let mut summary = format!(
                "Downloaded {} {} for {}.",
                downloaded_count.max(1),
                item_label(downloaded_count.max(1), "artifact", "artifacts"),
                entry.name,
            );
            if repaired_count > 0 {
                summary.push_str(&format!(
                    " Repaired {} invalid {} in the process.",
                    repaired_count,
                    item_label(repaired_count, "artifact", "artifacts"),
                ));
            }
            summary.push(' ');
            summary.push_str(&entry.readiness_summary);
            summary
        }
    };

    ModelDownloadReport {
        model: entry.name.clone(),
        action,
        summary,
        entry,
        artifact_changes,
    }
}

fn assess_readiness(
    entry: &RegistryEntry,
    cache_status: CacheStatus,
    artifact_summary: &ArtifactInventorySummary,
    evidence_status: EvidenceStatus,
    verification_note: Option<&str>,
) -> (ModelReadinessStatus, String, Vec<String>) {
    let model = entry.info.name.as_str();
    let mut next_steps = Vec::new();

    if !entry.is_ready() {
        next_steps.push(format!(
            "Promote {model} from {} to a ready ONNX integration before normal runs.",
            entry.availability.phase
        ));
        next_steps.push(
            "Until then, only the explicit stub backend is appropriate for development-only report scaffolding."
                .to_string(),
        );
        return (
            ModelReadinessStatus::Planned,
            "This integration is still planned, so normal ONNX runs remain intentionally disabled."
                .to_string(),
            next_steps,
        );
    }

    if matches!(cache_status, CacheStatus::Unknown) || artifact_summary.unusable > 0 {
        next_steps.push(format!(
            "Inspect the local cache bundle for {model} and remove or repair unusable artifacts."
        ));
        next_steps.push(format!(
            "After the cache is healthy, rerun `latent-inspector models --download {model}` to restore the expected bundle."
        ));
        if matches!(evidence_status, EvidenceStatus::Stale) {
            next_steps.push(format!(
                "Once the cache is healthy, refresh the approved validation evidence for {model}."
            ));
        }
        if let Some(note) = verification_note {
            next_steps.push(format!("Verification metadata note: {note}"));
        }
        let summary = if artifact_summary.unusable > 0 {
            format!(
                "{} unusable {} are blocking normal runs and require manual cache repair.",
                artifact_summary.unusable,
                item_label(artifact_summary.unusable, "artifact", "artifacts"),
            )
        } else {
            "Cache inspection could not determine whether the artifact bundle is usable."
                .to_string()
        };
        return (ModelReadinessStatus::Blocked, summary, next_steps);
    }

    let actionable_downloads = artifact_summary.missing + artifact_summary.invalid;
    if cache_status != CacheStatus::Complete || actionable_downloads > 0 {
        next_steps.push(format!(
            "Run `latent-inspector models --download {model}` to download or refresh the required artifact bundle."
        ));
        match evidence_status {
            EvidenceStatus::Stale => next_steps.push(format!(
                "After the cache bundle is complete, refresh the approved validation evidence for {model}."
            )),
            EvidenceStatus::Missing => next_steps.push(format!(
                "After the cache bundle is complete, run `latent-inspector validate --model {model}` to record approved evidence."
            )),
            EvidenceStatus::Approved | EvidenceStatus::Unverified => {}
        }
        if artifact_summary.pending_verification > 0 {
            next_steps.push(format!(
                "Checksum metadata is still pending for {} {}; keep the cache provenance documented until hashes are pinned.",
                artifact_summary.pending_verification,
                item_label(
                    artifact_summary.pending_verification,
                    "artifact",
                    "artifacts",
                ),
            ));
        }

        let summary = match (artifact_summary.missing, artifact_summary.invalid) {
            (missing, 0) => format!(
                "{} required {} are missing from the local cache bundle.",
                missing,
                item_label(missing, "artifact", "artifacts"),
            ),
            (0, invalid) => format!(
                "{} cached {} must be refreshed before normal runs are safe.",
                invalid,
                item_label(invalid, "artifact", "artifacts"),
            ),
            (missing, invalid) => format!(
                "{} {} are missing and {} {} must be refreshed before normal runs are safe.",
                missing,
                item_label(missing, "artifact", "artifacts"),
                invalid,
                item_label(invalid, "artifact", "artifacts"),
            ),
        };
        return (ModelReadinessStatus::NeedsDownload, summary, next_steps);
    }

    match evidence_status {
        EvidenceStatus::Approved => {
            if artifact_summary.pending_verification > 0 {
                next_steps.push(format!(
                    "Pin checksum metadata for {} {} so future downloads can be fully verified.",
                    artifact_summary.pending_verification,
                    item_label(
                        artifact_summary.pending_verification,
                        "artifact",
                        "artifacts",
                    ),
                ));
                return (
                    ModelReadinessStatus::Ready,
                    format!(
                        "Ready to run: the cache bundle is complete and approved validation evidence is current. Checksum metadata is still pending for {} {}.",
                        artifact_summary.pending_verification,
                        item_label(
                            artifact_summary.pending_verification,
                            "artifact",
                            "artifacts",
                        ),
                    ),
                    next_steps,
                );
            }

            (
                ModelReadinessStatus::Ready,
                "Ready to run: the cache bundle is complete and approved validation evidence is current."
                    .to_string(),
                next_steps,
            )
        }
        EvidenceStatus::Stale => {
            next_steps.push(format!(
                "Refresh the checked-in contract or reference evidence for {model} so reports can be treated as source-aligned again."
            ));
            if artifact_summary.pending_verification > 0 {
                next_steps.push(format!(
                    "Pin checksum metadata for {} {} once the evidence refresh is complete.",
                    artifact_summary.pending_verification,
                    item_label(
                        artifact_summary.pending_verification,
                        "artifact",
                        "artifacts",
                    ),
                ));
            }
            (
                ModelReadinessStatus::NeedsEvidenceRefresh,
                "The model can run, but the approved validation evidence is stale against the active registry profile."
                    .to_string(),
                next_steps,
            )
        }
        EvidenceStatus::Missing => {
            next_steps.push(format!(
                "Run `latent-inspector validate --model {model}` against the approved fixture set and commit the resulting evidence."
            ));
            if artifact_summary.pending_verification > 0 {
                next_steps.push(format!(
                    "Pin checksum metadata for {} {} alongside the validation refresh.",
                    artifact_summary.pending_verification,
                    item_label(
                        artifact_summary.pending_verification,
                        "artifact",
                        "artifacts",
                    ),
                ));
            }
            (
                ModelReadinessStatus::NeedsValidation,
                "The artifact bundle is runnable, but approved validation evidence is missing or could not be loaded."
                    .to_string(),
                next_steps,
            )
        }
        EvidenceStatus::Unverified => (
            ModelReadinessStatus::Planned,
            "This integration is still planned, so release-grade validation remains intentionally withheld."
                .to_string(),
            next_steps,
        ),
    }
}

fn item_label(count: usize, singular: &'static str, plural: &'static str) -> &'static str {
    if count == 1 {
        singular
    } else {
        plural
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
        assert_eq!(dinov2.artifact_summary.total, 1);
        assert_eq!(report.summary.artifacts.total, 6);
        assert!(dinov2.evidence_details.is_empty());
        assert_eq!(report.summary.total_models, report.entries.len());
        assert_eq!(report.summary.ready_models, 2);
        // Only DINOv2 has ONNX-backed evidence; I-JEPA is ready but its
        // reference artifacts were generated by the stub backend, so evidence
        // is stale rather than approved.
        assert_eq!(report.summary.evidence.approved, 1);
    }

    #[test]
    fn artifact_summary_counts_usable_missing_and_unknown_states() {
        let summary = ArtifactInventorySummary::from_statuses([
            ArtifactCacheStatus::PresentVerified,
            ArtifactCacheStatus::PresentUnverified,
            ArtifactCacheStatus::Missing,
            ArtifactCacheStatus::Invalid,
            ArtifactCacheStatus::Unknown,
        ]);

        assert_eq!(summary.total, 5);
        assert_eq!(summary.usable, 2);
        assert_eq!(summary.verified, 1);
        assert_eq!(summary.pending_verification, 1);
        assert_eq!(summary.missing, 1);
        assert_eq!(summary.invalid, 1);
        assert_eq!(summary.unknown, 1);
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

    #[test]
    fn readiness_needs_download_when_ready_model_is_uncached() {
        let entry = crate::models::registry::find("dinov2-vit-l14").unwrap();
        let artifact_summary = ArtifactInventorySummary {
            total: 1,
            missing: 1,
            ..ArtifactInventorySummary::default()
        };

        let (status, summary, next_steps) = assess_readiness(
            &entry,
            CacheStatus::Missing,
            &artifact_summary,
            EvidenceStatus::Approved,
            None,
        );

        assert_eq!(status, ModelReadinessStatus::NeedsDownload);
        assert!(summary.contains("missing"));
        assert!(next_steps
            .iter()
            .any(|step| step.contains("models --download dinov2-vit-l14")));
    }

    #[test]
    fn readiness_marks_complete_model_ready_even_with_pending_checksum_metadata() {
        let entry = crate::models::registry::find("dinov2-vit-l14").unwrap();
        let artifact_summary = ArtifactInventorySummary {
            total: 1,
            usable: 1,
            pending_verification: 1,
            ..ArtifactInventorySummary::default()
        };

        let (status, summary, next_steps) = assess_readiness(
            &entry,
            CacheStatus::Complete,
            &artifact_summary,
            EvidenceStatus::Approved,
            Some("SHA-256 pin pending"),
        );

        assert_eq!(status, ModelReadinessStatus::Ready);
        assert!(summary.contains("Ready to run"));
        assert!(next_steps
            .iter()
            .any(|step| step.contains("checksum metadata")));
    }
}
