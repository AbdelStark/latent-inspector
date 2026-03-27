use crate::errors::ModelError;
use crate::models::registry::{self, Checksum, ModelArtifact, RegistryEntry};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const CACHE_DIR_ENV: &str = "LATENT_INSPECTOR_CACHE_DIR";

#[derive(Debug, Clone)]
enum ArtifactCacheState {
    Missing,
    PresentUnverified,
    PresentVerified,
    Invalid(String),
    Unusable(String),
}

impl ArtifactCacheState {
    fn is_usable(&self) -> bool {
        matches!(
            self,
            ArtifactCacheState::PresentUnverified | ArtifactCacheState::PresentVerified
        )
    }

    fn needs_download(&self) -> bool {
        matches!(
            self,
            ArtifactCacheState::Missing | ArtifactCacheState::Invalid(_)
        )
    }

    fn detail(&self) -> Option<&str> {
        match self {
            ArtifactCacheState::Invalid(reason) | ArtifactCacheState::Unusable(reason) => {
                Some(reason.as_str())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct ArtifactInspection {
    artifact: ModelArtifact,
    path: PathBuf,
    state: ArtifactCacheState,
}

#[derive(Debug, Clone)]
struct CacheInspection {
    artifacts: Vec<ArtifactInspection>,
}

impl CacheInspection {
    fn is_complete(&self) -> bool {
        !self.artifacts.is_empty()
            && self
                .artifacts
                .iter()
                .all(|artifact| artifact.state.is_usable())
    }

    fn repairable_artifacts(&self) -> impl Iterator<Item = &ArtifactInspection> {
        self.artifacts
            .iter()
            .filter(|artifact| artifact.state.needs_download())
    }

    fn first_unusable(&self) -> Option<&ArtifactInspection> {
        self.artifacts
            .iter()
            .find(|artifact| matches!(artifact.state, ArtifactCacheState::Unusable(_)))
    }

    fn incomplete_reason(&self) -> String {
        self.artifacts
            .iter()
            .filter(|artifact| !artifact.state.is_usable())
            .map(|artifact| match &artifact.state {
                ArtifactCacheState::Missing => {
                    format!("{}: missing", artifact.artifact.relative_path)
                }
                ArtifactCacheState::Invalid(reason) | ArtifactCacheState::Unusable(reason) => {
                    format!("{}: {reason}", artifact.artifact.relative_path)
                }
                ArtifactCacheState::PresentUnverified | ArtifactCacheState::PresentVerified => {
                    artifact.artifact.relative_path.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Returns the cache directory: `~/.cache/latent-inspector/`.
pub fn cache_dir() -> Result<PathBuf, ModelError> {
    let dir = std::env::var_os(CACHE_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(|| {
            dirs::cache_dir()
                .map(|base| base.join("latent-inspector"))
                .ok_or_else(|| {
                    ModelError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Could not determine cache directory",
                    ))
                })
        })?;
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Returns the expected path for a model's primary ONNX file in the cache.
pub fn model_path(model_name: &str) -> Result<PathBuf, ModelError> {
    let entry = registry_entry(model_name)?;
    artifact_path(entry.primary_artifact()?)
}

/// Returns true if every required artifact for the model is already cached.
pub fn is_cached(model_name: &str) -> Result<bool, ModelError> {
    let entry = registry_entry(model_name)?;
    Ok(inspect_cache(&entry)?.is_complete())
}

/// Ensure every required artifact is present and usable, redownloading only the
/// missing or invalid files in the cache bundle.
pub fn ensure_artifacts(model_name: &str, entry: &RegistryEntry) -> Result<PathBuf, ModelError> {
    entry.ensure_ready()?;

    let initial = inspect_cache(entry)?;
    if let Some(problem) = initial.first_unusable() {
        return Err(ModelError::InvalidArtifactPath {
            name: model_name.to_string(),
            path: problem.path.display().to_string(),
            reason: problem
                .state
                .detail()
                .unwrap_or("artifact is not usable")
                .to_string(),
        });
    }

    for artifact in initial.repairable_artifacts() {
        if let Some(detail) = artifact.state.detail() {
            warn!(
                "Refreshing cached artifact {} for '{}': {}",
                artifact.artifact.relative_path, model_name, detail
            );
        } else {
            info!(
                "Caching missing artifact {} for '{}'",
                artifact.artifact.relative_path, model_name
            );
        }

        if let Some(parent) = artifact.path.parent() {
            fs::create_dir_all(parent)?;
        }
        download_artifact(model_name, &artifact.artifact, &artifact.path)?;
    }

    let repaired = inspect_cache(entry)?;
    if let Some(problem) = repaired.first_unusable() {
        return Err(ModelError::InvalidArtifactPath {
            name: model_name.to_string(),
            path: problem.path.display().to_string(),
            reason: problem
                .state
                .detail()
                .unwrap_or("artifact is not usable")
                .to_string(),
        });
    }

    if !repaired.is_complete() {
        return Err(ModelError::DownloadFailed {
            name: model_name.to_string(),
            reason: format!(
                "artifact bundle remained incomplete after download: {}",
                repaired.incomplete_reason()
            ),
        });
    }

    model_path(model_name)
}

/// Download every artifact for a model and verify integrity according to each
/// artifact's checksum policy.
pub fn download(model_name: &str, entry: &RegistryEntry) -> Result<(), ModelError> {
    ensure_artifacts(model_name, entry)?;
    Ok(())
}

fn download_artifact(
    model_name: &str,
    artifact: &ModelArtifact,
    dest: &Path,
) -> Result<(), ModelError> {
    info!(
        "Downloading {} from {}",
        artifact.relative_path, artifact.download_url
    );

    let mut response = reqwest::blocking::get(&artifact.download_url)
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| ModelError::DownloadFailed {
            name: model_name.to_string(),
            reason: e.to_string(),
        })?;

    let progress = ProgressBar::new(response.content_length().unwrap_or(0));
    progress.set_style(progress_style(model_name)?);
    progress.set_message(format!("Downloading {}", artifact.relative_path));

    let tmp = temp_download_path(dest)?;
    let result = download_to_file(&mut response, artifact, &tmp, model_name, &progress);

    if let Err(error) = result {
        progress.abandon_with_message(format!("Failed {}", artifact.relative_path));
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }

    progress.finish_with_message(format!("Downloaded {}", artifact.relative_path));
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::rename(&tmp, dest)?;
    info!(
        "Model artifact {} saved to {}",
        artifact.relative_path,
        dest.display()
    );
    Ok(())
}

fn progress_style(model_name: &str) -> Result<ProgressStyle, ModelError> {
    ProgressStyle::default_bar()
        .template(
            "{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
             {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
        )
        .map(|style| style.progress_chars("#>-"))
        .map_err(|e| ModelError::DownloadFailed {
            name: model_name.to_string(),
            reason: e.to_string(),
        })
}

fn download_to_file(
    response: &mut reqwest::blocking::Response,
    artifact: &ModelArtifact,
    path: &Path,
    model_name: &str,
    progress: &ProgressBar,
) -> Result<(), ModelError> {
    let mut file = fs::File::create(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut hasher = match &artifact.checksum {
        Checksum::Sha256(_) => Some(Sha256::new()),
        Checksum::Pending { reason } => {
            warn!(
                "Skipping checksum verification for {} until metadata is pinned: {}",
                artifact.relative_path, reason
            );
            None
        }
    };

    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| ModelError::DownloadFailed {
                name: model_name.to_string(),
                reason: e.to_string(),
            })?;

        if read == 0 {
            break;
        }

        let chunk = &buffer[..read];
        file.write_all(chunk)?;
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(chunk);
        }
        progress.inc(read as u64);
    }

    file.flush()?;

    if let (Checksum::Sha256(expected), Some(hasher)) = (&artifact.checksum, hasher) {
        let actual = hex::encode(hasher.finalize());
        verify_sha256_digest(expected, &actual, model_name)?;
    }

    Ok(())
}

fn registry_entry(model_name: &str) -> Result<RegistryEntry, ModelError> {
    registry::find(model_name).ok_or_else(|| ModelError::NotFound(model_name.to_string()))
}

fn artifact_path(artifact: &ModelArtifact) -> Result<PathBuf, ModelError> {
    Ok(cache_dir()?.join(&artifact.relative_path))
}

fn inspect_cache(entry: &RegistryEntry) -> Result<CacheInspection, ModelError> {
    let artifacts = entry
        .artifacts
        .iter()
        .cloned()
        .map(|artifact| inspect_artifact(&entry.info.name, artifact))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CacheInspection { artifacts })
}

fn inspect_artifact(
    model_name: &str,
    artifact: ModelArtifact,
) -> Result<ArtifactInspection, ModelError> {
    let path = artifact_path(&artifact)?;

    let state = if !path.exists() {
        ArtifactCacheState::Missing
    } else if !path.is_file() {
        ArtifactCacheState::Unusable("path exists but is not a file".to_string())
    } else if fs::metadata(&path)?.len() == 0 {
        ArtifactCacheState::Invalid("file is empty".to_string())
    } else {
        match &artifact.checksum {
            Checksum::Sha256(expected) => match verify_sha256(&path, expected, model_name) {
                Ok(()) => ArtifactCacheState::PresentVerified,
                Err(ModelError::VerificationFailed {
                    expected, actual, ..
                }) => ArtifactCacheState::Invalid(format!(
                    "checksum mismatch (expected {expected}, got {actual})"
                )),
                Err(err) => ArtifactCacheState::Unusable(err.to_string()),
            },
            Checksum::Pending { .. } => ArtifactCacheState::PresentUnverified,
        }
    };

    Ok(ArtifactInspection {
        artifact,
        path,
        state,
    })
}

fn temp_download_path(dest: &Path) -> Result<PathBuf, ModelError> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| ModelError::InvalidArtifactPath {
            name: "cache".to_string(),
            path: dest.display().to_string(),
            reason: "artifact path has no file name".to_string(),
        })?;
    let mut temp_name = file_name.to_os_string();
    temp_name.push(".download-part");
    Ok(dest.with_file_name(temp_name))
}

/// Verify the SHA-256 hash of a file.
pub fn verify_sha256(path: &Path, expected: &str, model_name: &str) -> Result<(), ModelError> {
    debug!("Verifying SHA-256 for {}", path.display());
    let actual = digest_file(path)?;
    verify_sha256_digest(expected, &actual, model_name)
}

fn digest_file(path: &Path) -> Result<String, ModelError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

fn verify_sha256_digest(expected: &str, actual: &str, model_name: &str) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::VerificationFailed {
            name: model_name.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{LazyLock, Mutex};
    use tempfile::tempdir;

    static CACHE_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct CacheDirEnvGuard {
        previous: Option<OsString>,
    }

    impl CacheDirEnvGuard {
        fn set(path: &Path) -> Self {
            let previous = std::env::var_os(CACHE_DIR_ENV);
            std::env::set_var(CACHE_DIR_ENV, path);
            Self { previous }
        }
    }

    impl Drop for CacheDirEnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(path) => std::env::set_var(CACHE_DIR_ENV, path),
                None => std::env::remove_var(CACHE_DIR_ENV),
            }
        }
    }

    #[test]
    fn test_cache_dir_created() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let result = cache_dir();
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_path_format() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let path = model_path("dinov2-vit-l14").unwrap();
        assert!(path.to_str().unwrap().ends_with("dinov2-vit-l14.onnx"));
    }

    #[test]
    fn test_external_data_model_path_format() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let path = model_path("ijepa-vit-h14").unwrap();
        assert!(path.to_str().unwrap().ends_with("ijepa-vit-h14/model.onnx"));
    }

    #[test]
    fn test_sha256_verification_accepts_expected_hash() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.bin");
        fs::write(&file, b"hello world").unwrap();

        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        let result = verify_sha256(&file, expected, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_sha256_verification_rejects_wrong_hash() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.bin");
        fs::write(&file, b"hello world").unwrap();

        let result = verify_sha256(&file, "not-the-right-hash", "test");
        assert!(matches!(result, Err(ModelError::VerificationFailed { .. })));
    }

    #[test]
    fn test_cache_dir_uses_env_override() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        let path = cache_dir().unwrap();
        assert_eq!(path, dir.path());
    }

    #[test]
    fn test_is_cached_requires_complete_artifact_bundle() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        let primary = model_path("ijepa-vit-h14").unwrap();
        fs::create_dir_all(primary.parent().unwrap()).unwrap();
        fs::write(&primary, b"onnx").unwrap();

        assert!(!is_cached("ijepa-vit-h14").unwrap());

        let companion = dir.path().join("ijepa-vit-h14/model.onnx_data");
        fs::write(companion, b"external-data").unwrap();

        assert!(is_cached("ijepa-vit-h14").unwrap());
    }

    #[test]
    fn test_is_cached_rejects_empty_artifact_files() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        let path = model_path("dinov2-vit-l14").unwrap();
        fs::write(path, []).unwrap();

        assert!(!is_cached("dinov2-vit-l14").unwrap());
    }

    #[test]
    fn test_temp_download_paths_do_not_collide_for_external_data() {
        let primary = Path::new("/tmp/model.onnx");
        let companion = Path::new("/tmp/model.onnx_data");

        let primary_tmp = temp_download_path(primary).unwrap();
        let companion_tmp = temp_download_path(companion).unwrap();

        assert_ne!(primary_tmp, companion_tmp);
        assert!(primary_tmp
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(".download-part"));
        assert!(companion_tmp
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(".download-part"));
    }

    #[test]
    fn test_inspect_artifact_detects_checksum_mismatch() {
        let _lock = CACHE_ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        let artifact = ModelArtifact {
            relative_path: "bundle/model.onnx".to_string(),
            download_url: "https://example.invalid/model.onnx".to_string(),
            checksum: Checksum::Sha256("deadbeef".to_string()),
        };
        let path = dir.path().join("bundle/model.onnx");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not-the-right-content").unwrap();

        let inspection = inspect_artifact("test-model", artifact).unwrap();
        assert!(matches!(inspection.state, ArtifactCacheState::Invalid(_)));
    }
}
