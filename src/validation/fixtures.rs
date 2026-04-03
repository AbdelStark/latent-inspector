use crate::errors::ValidationError;
use crate::models::registry::{ModelValidationProfile, ParityTolerances};
use crate::models::InferenceBackend;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFixtureManifest {
    pub fixture_set: String,
    pub evidence_timestamp: String,
    pub fixtures: Vec<ValidationFixtureSpec>,
    pub models: HashMap<String, ContractFixture>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractFixture {
    pub contract: String,
    pub reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFixtureSpec {
    pub id: String,
    pub description: String,
    pub width: u32,
    pub height: u32,
    pub pattern: FixturePattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FixturePattern {
    Gradient,
    Checkerboard,
    CenterSquare,
}

#[derive(Debug, Clone)]
pub struct MaterializedFixture {
    pub spec: ValidationFixtureSpec,
    pub image: DynamicImage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractArtifact {
    pub model: String,
    pub profile: ModelValidationProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSignals {
    pub tensor_name: String,
    pub output_shape: Vec<usize>,
    pub cls_present: bool,
    pub patch_count: usize,
    pub embedding_dim: usize,
    pub patch_mean: f32,
    pub patch_std: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch_rms: Option<f32>,
    pub cls_l2_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixtures: Vec<FixtureSignalSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureSignalSummary {
    pub id: String,
    pub patch_mean: f32,
    pub patch_std: f32,
    pub patch_rms: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patch_signature: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cls_l2_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cls_signature: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceArtifact {
    pub model: String,
    pub fixture_set: String,
    pub evidence_timestamp: String,
    pub artifact_id: String,
    pub source: String,
    #[serde(default = "default_reference_backend")]
    pub backend: InferenceBackend,
    pub tolerances: ParityTolerances,
    pub observed: ReferenceSignals,
}

#[derive(Debug, Clone)]
pub struct LoadedFixtureSet {
    pub manifest_path: PathBuf,
    pub manifest: ValidationFixtureManifest,
}

impl LoadedFixtureSet {
    pub fn manifest_dir(&self) -> &Path {
        self.manifest_path
            .parent()
            .unwrap_or(self.manifest_path.as_path())
    }

    pub fn materialize_fixtures(&self) -> Result<Vec<MaterializedFixture>, ValidationError> {
        self.manifest
            .fixtures
            .iter()
            .cloned()
            .map(|spec| {
                Ok(MaterializedFixture {
                    image: materialize_pattern(&spec),
                    spec,
                })
            })
            .collect()
    }

    pub fn load_contract(&self, model: &str) -> Result<ContractArtifact, ValidationError> {
        let files = self.model_files(model)?;
        let path = self.manifest_dir().join(&files.contract);
        load_json(&path).map_err(|err| match err {
            ValidationError::Io(_) | ValidationError::Json(_) => ValidationError::MissingFixtures(
                format!("Failed to load contract fixture {}: {err}", path.display()),
            ),
            other => other,
        })
    }

    pub fn load_reference(&self, model: &str) -> Result<ReferenceArtifact, ValidationError> {
        let files = self.model_files(model)?;
        let path = self.manifest_dir().join(&files.reference);
        load_json(&path).map_err(|err| match err {
            ValidationError::Io(_) | ValidationError::Json(_) => {
                ValidationError::MissingFixtures(format!(
                    "Failed to load reference artifact {}: {err}",
                    path.display()
                ))
            }
            other => other,
        })
    }

    pub fn write_reference(
        &self,
        model: &str,
        artifact: &ReferenceArtifact,
    ) -> Result<PathBuf, ValidationError> {
        let files = self.model_files(model)?;
        let path = self.manifest_dir().join(&files.reference);
        let json = serde_json::to_string_pretty(artifact)?;
        fs::write(&path, json)?;
        Ok(path)
    }

    fn model_files(&self, model: &str) -> Result<&ContractFixture, ValidationError> {
        self.manifest.models.get(model).ok_or_else(|| {
            ValidationError::MissingFixtures(format!(
                "Fixture manifest '{}' does not declare artifacts for model '{}'",
                self.manifest.fixture_set, model
            ))
        })
    }
}

pub fn default_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("validation")
        .join("manifest.json")
}

pub fn resolve_manifest_path(selection: Option<&str>) -> Result<PathBuf, ValidationError> {
    match selection {
        None => Ok(default_manifest_path()),
        Some("standard") => Ok(default_manifest_path()),
        Some(candidate) => {
            let path = PathBuf::from(candidate);
            if path.exists() {
                Ok(path)
            } else {
                Err(ValidationError::Usage(format!(
                    "Unknown fixture-set '{}'. Pass 'standard' or a manifest path.",
                    candidate
                )))
            }
        }
    }
}

pub fn load_fixture_set(selection: Option<&str>) -> Result<LoadedFixtureSet, ValidationError> {
    let manifest_path = resolve_manifest_path(selection)?;
    if !manifest_path.exists() {
        return Err(ValidationError::MissingFixtures(format!(
            "Validation manifest not found at {}",
            manifest_path.display()
        )));
    }

    let manifest = load_json(&manifest_path)?;
    Ok(LoadedFixtureSet {
        manifest_path,
        manifest,
    })
}

pub fn build_reference_artifact_id(
    model: &str,
    fixture_set: &str,
    evidence_timestamp: &str,
) -> String {
    format!("{model}:{fixture_set}:{evidence_timestamp}")
}

fn default_reference_backend() -> InferenceBackend {
    InferenceBackend::Stub
}

fn materialize_pattern(spec: &ValidationFixtureSpec) -> DynamicImage {
    let image: RgbImage = match spec.pattern {
        FixturePattern::Gradient => ImageBuffer::from_fn(spec.width, spec.height, |x, y| {
            let r = ((x * 255) / spec.width.max(1)) as u8;
            let g = ((y * 255) / spec.height.max(1)) as u8;
            let b = (((x + y) * 255) / (spec.width + spec.height).max(1)) as u8;
            Rgb([r, g, b])
        }),
        FixturePattern::Checkerboard => ImageBuffer::from_fn(spec.width, spec.height, |x, y| {
            let light = ((x / 16) + (y / 16)) % 2 == 0;
            let value = if light { 224 } else { 32 };
            Rgb([value, 64, 255 - value])
        }),
        FixturePattern::CenterSquare => ImageBuffer::from_fn(spec.width, spec.height, |x, y| {
            let inside = x > spec.width / 4
                && x < (spec.width * 3) / 4
                && y > spec.height / 4
                && y < (spec.height * 3) / 4;
            if inside {
                Rgb([255, 196, 32])
            } else {
                Rgb([12, 18, 28])
            }
        }),
    };

    DynamicImage::ImageRgb8(image)
}

fn load_json<T>(path: &Path) -> Result<T, ValidationError>
where
    T: for<'de> Deserialize<'de>,
{
    let data = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fixture_set_loads() {
        let fixture_set = load_fixture_set(None).unwrap();
        assert_eq!(fixture_set.manifest.fixture_set, "standard");
        assert!(!fixture_set.manifest.fixtures.is_empty());
    }

    #[test]
    fn materialized_fixture_matches_dimensions() {
        let fixture_set = load_fixture_set(None).unwrap();
        let fixture = fixture_set.materialize_fixtures().unwrap().remove(0);
        assert_eq!(fixture.image.width(), fixture.spec.width);
        assert_eq!(fixture.image.height(), fixture.spec.height);
    }
}
