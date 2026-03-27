use crate::errors::ModelError;
use crate::models::registry::{self, Checksum, ModelArtifact, RegistryEntry};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Returns the cache directory: `~/.cache/latent-inspector/`.
pub fn cache_dir() -> Result<PathBuf, ModelError> {
    let base = dirs::cache_dir().ok_or_else(|| {
        ModelError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Could not determine cache directory",
        ))
    })?;
    let dir = base.join("latent-inspector");
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
    if entry.artifacts.is_empty() {
        return Ok(false);
    }

    for artifact in &entry.artifacts {
        if !artifact_path(artifact)?.exists() {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Download every artifact for a model and verify integrity according to each
/// artifact's checksum policy.
pub fn download(model_name: &str, entry: &RegistryEntry) -> Result<(), ModelError> {
    entry.ensure_ready()?;

    for artifact in &entry.artifacts {
        let dest = artifact_path(artifact)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        download_artifact(model_name, artifact, &dest)?;
    }

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

    let tmp = dest.with_extension("onnx.tmp");
    let result = download_to_file(&mut response, artifact, &tmp, model_name, &progress);

    if let Err(error) = result {
        progress.abandon_with_message(format!("Failed {}", artifact.relative_path));
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }

    progress.finish_with_message(format!("Downloaded {}", artifact.relative_path));
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
    use tempfile::tempdir;

    #[test]
    fn test_cache_dir_created() {
        let result = cache_dir();
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_path_format() {
        let path = model_path("dinov2-vit-l14").unwrap();
        assert!(path.to_str().unwrap().ends_with("dinov2-vit-l14.onnx"));
    }

    #[test]
    fn test_external_data_model_path_format() {
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
}
