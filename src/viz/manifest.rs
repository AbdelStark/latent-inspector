use crate::errors::VizError;
use crate::validation::report::{ModelValidationSummary, ValidationStatus};
use crate::viz::OutputFormat;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

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
}

impl OutputArtifact {
    pub fn new(path: impl Into<String>, kind: ArtifactKind, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind,
            label: label.into(),
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

    pub fn write_to_dir(&self, outdir: &Path) -> Result<PathBuf, VizError> {
        let path = outdir.join(ARTIFACT_MANIFEST_FILENAME);
        let json = serde_json::to_string_pretty(self)
            .map_err(|err| VizError::Artifact(format!("Manifest serialization failed: {err}")))?;
        std::fs::write(&path, json).map_err(|err| {
            VizError::Artifact(format!("Failed to write {}: {err}", path.display()))
        })?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn manifest_write_captures_primary_artifact_and_validation() {
        let dir = tempdir().unwrap();
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
        assert_eq!(parsed.validation.len(), 1);
        assert_eq!(parsed.validation[0].model, "dinov2-vit-l14");
        assert_eq!(parsed.validation[0].status, ValidationStatus::Unverified);
        assert_eq!(
            parsed.validation_summary.as_ref().unwrap().overall_status,
            ValidationStatus::Unverified
        );
        assert_eq!(parsed.validation_summary.as_ref().unwrap().unverified, 1);
    }
}
