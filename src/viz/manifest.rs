use crate::errors::VizError;
use crate::validation::report::{ModelValidationSummary, ValidationStatus};
use crate::viz::OutputFormat;
use serde::{Deserialize, Serialize};
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
pub struct OutputArtifactManifest {
    pub command: String,
    pub format: OutputFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_artifact: Option<String>,
    pub artifacts: Vec<OutputArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<ArtifactValidationRecord>,
}

impl OutputArtifactManifest {
    pub fn new(command: impl Into<String>, format: OutputFormat) -> Self {
        Self {
            command: command.into(),
            format,
            primary_artifact: None,
            artifacts: Vec::new(),
            validation: Vec::new(),
        }
    }

    pub fn with_primary_artifact(mut self, path: impl Into<String>) -> Self {
        self.primary_artifact = Some(path.into());
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
        assert_eq!(parsed.artifacts.len(), 2);
        assert_eq!(parsed.artifacts[0].kind, ArtifactKind::Html);
        assert_eq!(parsed.artifacts[1].path, "dinov2-vit-l14_pca.png");
        assert_eq!(parsed.validation.len(), 1);
        assert_eq!(parsed.validation[0].model, "dinov2-vit-l14");
        assert_eq!(parsed.validation[0].status, ValidationStatus::Unverified);
    }
}
