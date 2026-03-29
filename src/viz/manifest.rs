use crate::errors::VizError;
use crate::validation::report::{ModelValidationSummary, ValidationStatus};
use crate::viz::OutputFormat;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const ARTIFACT_MANIFEST_FILENAME: &str = "artifacts.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactKind {
    Json,
    Html,
    Png,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputArtifact {
    pub path: String,
    pub kind: ArtifactKind,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl OutputArtifact {
    pub fn new(path: impl Into<String>, kind: ArtifactKind, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind,
            label: label.into(),
            byte_size: None,
            sha256: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactValidationRecord {
    pub model: String,
    pub status: ValidationStatus,
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactValidationOverview {
    pub overall_status: ValidationStatus,
    pub validated: usize,
    pub partial: usize,
    pub unverified: usize,
    pub stale: usize,
    pub failed: usize,
}

impl ArtifactValidationOverview {
    fn from_summaries(summaries: &[ModelValidationSummary]) -> Option<Self> {
        if summaries.is_empty() {
            return None;
        }

        let mut overall_status = ValidationStatus::Validated;
        let mut validated = 0;
        let mut partial = 0;
        let mut unverified = 0;
        let mut stale = 0;
        let mut failed = 0;

        for summary in summaries {
            overall_status = overall_status.combine(summary.status);
            match summary.status {
                ValidationStatus::Validated => validated += 1,
                ValidationStatus::Partial => partial += 1,
                ValidationStatus::Unverified => unverified += 1,
                ValidationStatus::Stale => stale += 1,
                ValidationStatus::Failed => failed += 1,
            }
        }

        Some(Self {
            overall_status,
            validated,
            partial,
            unverified,
            stale,
            failed,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputArtifactManifest {
    pub command: String,
    pub format: OutputFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Value>,
    pub artifacts: Vec<OutputArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<ArtifactValidationRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_summary: Option<ArtifactValidationOverview>,
}

impl OutputArtifactManifest {
    pub fn new(command: impl Into<String>, format: OutputFormat) -> Self {
        Self {
            command: command.into(),
            format,
            primary_artifact: None,
            context: None,
            summary: None,
            artifacts: Vec::new(),
            validation: Vec::new(),
            validation_summary: None,
        }
    }

    pub fn with_primary_artifact(mut self, path: impl Into<String>) -> Self {
        self.primary_artifact = Some(path.into());
        self
    }

    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_summary(mut self, summary: Value) -> Self {
        self.summary = Some(summary);
        self
    }

    pub fn add_artifact(
        mut self,
        path: impl Into<String>,
        kind: ArtifactKind,
        label: impl Into<String>,
    ) -> Self {
        self.artifacts.push(OutputArtifact::new(path, kind, label));
        self
    }

    pub fn with_validation(mut self, summaries: &[ModelValidationSummary]) -> Self {
        self.validation = summaries
            .iter()
            .map(|summary| ArtifactValidationRecord {
                model: summary.model.clone(),
                status: summary.status,
                recommendation: summary.recommendation.clone(),
            })
            .collect();
        self.validation_summary = ArtifactValidationOverview::from_summaries(summaries);
        self
    }

    pub fn finalize_for_bundle_display(&self, outdir: &Path) -> Result<Self, VizError> {
        let excluded = if self.format == OutputFormat::Html {
            self.primary_artifact
                .as_deref()
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        self.finalize_for_dir_with_exclusions(outdir, &excluded)
    }

    pub fn finalize_for_dir(&self, outdir: &Path) -> Result<Self, VizError> {
        self.finalize_for_dir_with_exclusions(outdir, &[])
    }

    fn finalize_for_dir_with_exclusions(
        &self,
        outdir: &Path,
        excluded_paths: &[&str],
    ) -> Result<Self, VizError> {
        self.validate()?;

        let excluded = excluded_paths.iter().copied().collect::<HashSet<_>>();
        let mut manifest = self.clone();
        for artifact in &mut manifest.artifacts {
            if excluded.contains(artifact.path.as_str()) {
                artifact.byte_size = None;
                artifact.sha256 = None;
                continue;
            }

            let path = outdir.join(&artifact.path);
            let metadata = std::fs::metadata(&path).map_err(|err| {
                VizError::Artifact(format!(
                    "Bundle artifact '{}' could not be inspected: {err}",
                    artifact.path
                ))
            })?;
            if !metadata.is_file() {
                return Err(VizError::Artifact(format!(
                    "Bundle artifact '{}' is not a regular file.",
                    artifact.path
                )));
            }

            artifact.byte_size = Some(metadata.len());
            artifact.sha256 = Some(digest_file(&path)?);
        }

        Ok(manifest)
    }

    pub fn write_to_dir(&self, outdir: &Path) -> Result<PathBuf, VizError> {
        let path = outdir.join(ARTIFACT_MANIFEST_FILENAME);
        let manifest = self.finalize_for_dir(outdir)?;
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|err| VizError::Artifact(format!("Manifest serialization failed: {err}")))?;
        std::fs::write(&path, json).map_err(|err| {
            VizError::Artifact(format!("Failed to write {}: {err}", path.display()))
        })?;
        Ok(path)
    }

    fn validate(&self) -> Result<(), VizError> {
        if self.artifacts.is_empty() {
            return Err(VizError::Artifact(
                "Bundle manifest must include at least one declared artifact.".to_string(),
            ));
        }

        let mut seen = HashSet::new();
        for artifact in &self.artifacts {
            validate_artifact_path(&artifact.path)?;
            if !seen.insert(artifact.path.as_str()) {
                return Err(VizError::Artifact(format!(
                    "Bundle manifest contains duplicate artifact path '{}'.",
                    artifact.path
                )));
            }
        }

        if let Some(primary) = &self.primary_artifact {
            validate_artifact_path(primary)?;
            if !self
                .artifacts
                .iter()
                .any(|artifact| artifact.path == *primary)
            {
                return Err(VizError::Artifact(format!(
                    "Primary artifact '{}' is not listed in the bundle artifact table.",
                    primary
                )));
            }
        }

        Ok(())
    }
}

fn validate_artifact_path(path: &str) -> Result<(), VizError> {
    if path.trim().is_empty() {
        return Err(VizError::Artifact(
            "Bundle artifact paths must not be empty.".to_string(),
        ));
    }

    let artifact_path = Path::new(path);
    if artifact_path.is_absolute() {
        return Err(VizError::Artifact(format!(
            "Bundle artifact path '{}' must be relative to the output directory.",
            path
        )));
    }

    if artifact_path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(VizError::Artifact(format!(
            "Bundle artifact path '{}' must not escape the output directory.",
            path
        )));
    }

    Ok(())
}

fn digest_file(path: &Path) -> Result<String, VizError> {
    let mut file = std::fs::File::open(path).map_err(|err| {
        VizError::Artifact(format!(
            "Failed to open {} for digesting: {err}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let read = file.read(&mut buffer).map_err(|err| {
            VizError::Artifact(format!(
                "Failed to read {} for digesting: {err}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn manifest_write_captures_primary_artifact_and_validation() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("report.html"), "<html></html>").unwrap();
        std::fs::write(dir.path().join("dinov2-vit-l14_pca.png"), b"png").unwrap();
        let manifest = OutputArtifactManifest::new("inspect", OutputFormat::Html)
            .with_primary_artifact("report.html")
            .with_context(serde_json::json!({
                "image": "fixture.png",
                "model": "dinov2-vit-l14"
            }))
            .with_summary(serde_json::json!({
                "effective_rank": 8,
                "patch_entropy": 4.2
            }))
            .add_artifact("report.html", ArtifactKind::Html, "Inspect report")
            .add_artifact(
                "dinov2-vit-l14_pca.png",
                ArtifactKind::Png,
                "PCA projection",
            )
            .with_validation(&[ModelValidationSummary::unverified(
                "dinov2-vit-l14",
                "2026-03-27T12:00:00Z",
                "Stub backend is active.",
            )]);

        let path = manifest.write_to_dir(dir.path()).unwrap();
        let payload = std::fs::read_to_string(path).unwrap();
        let parsed: OutputArtifactManifest = serde_json::from_str(&payload).unwrap();

        assert_eq!(parsed.command, "inspect");
        assert_eq!(parsed.format, OutputFormat::Html);
        assert_eq!(parsed.primary_artifact.as_deref(), Some("report.html"));
        assert_eq!(parsed.context.as_ref().unwrap()["image"], "fixture.png");
        assert_eq!(parsed.summary.as_ref().unwrap()["effective_rank"], 8);
        assert_eq!(parsed.artifacts.len(), 2);
        assert_eq!(parsed.artifacts[0].kind, ArtifactKind::Html);
        assert_eq!(parsed.artifacts[1].path, "dinov2-vit-l14_pca.png");
        assert!(parsed.artifacts[0].byte_size.is_some());
        assert!(parsed.artifacts[0].sha256.is_some());
        assert!(parsed.artifacts[1].byte_size.is_some());
        assert!(parsed.artifacts[1].sha256.is_some());
        assert_eq!(parsed.validation.len(), 1);
        assert_eq!(parsed.validation[0].model, "dinov2-vit-l14");
        assert_eq!(parsed.validation[0].status, ValidationStatus::Unverified);
        assert_eq!(
            parsed.validation_summary.as_ref().unwrap().overall_status,
            ValidationStatus::Unverified
        );
        assert_eq!(parsed.validation_summary.as_ref().unwrap().unverified, 1);
    }

    #[test]
    fn manifest_rejects_duplicate_or_escaping_paths() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("report.html"), "<html></html>").unwrap();

        let duplicate = OutputArtifactManifest::new("inspect", OutputFormat::Html)
            .with_primary_artifact("report.html")
            .add_artifact("report.html", ArtifactKind::Html, "Inspect report")
            .add_artifact("report.html", ArtifactKind::Json, "Duplicate artifact");
        let duplicate_error = duplicate.write_to_dir(dir.path()).unwrap_err().to_string();
        assert!(duplicate_error.contains("duplicate artifact path"));

        let escaping = OutputArtifactManifest::new("inspect", OutputFormat::Json)
            .with_primary_artifact("../outside.json")
            .add_artifact("../outside.json", ArtifactKind::Json, "Escaping artifact");
        let escaping_error = escaping.write_to_dir(dir.path()).unwrap_err().to_string();
        assert!(escaping_error.contains("must not escape the output directory"));
    }

    #[test]
    fn bundle_display_skips_primary_html_metadata() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("report.html"), "<html></html>").unwrap();
        std::fs::write(dir.path().join("compare.json"), "{\"ok\":true}").unwrap();

        let manifest = OutputArtifactManifest::new("compare", OutputFormat::Html)
            .with_primary_artifact("report.html")
            .add_artifact("report.html", ArtifactKind::Html, "Compare report")
            .add_artifact("compare.json", ArtifactKind::Json, "Compare data");

        let display = manifest.finalize_for_bundle_display(dir.path()).unwrap();
        let report = display
            .artifacts
            .iter()
            .find(|artifact| artifact.path == "report.html")
            .unwrap();
        assert!(report.byte_size.is_none());
        assert!(report.sha256.is_none());

        let data = display
            .artifacts
            .iter()
            .find(|artifact| artifact.path == "compare.json")
            .unwrap();
        assert!(data.byte_size.is_some());
        assert!(data.sha256.is_some());
    }
}
