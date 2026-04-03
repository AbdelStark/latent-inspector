use crate::errors::ModelError;
use crate::models::registry::{self, Checksum, ModelArtifact, RegistryEntry};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{CONTENT_RANGE, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};
use tracing::{debug, info, warn};

const CACHE_DIR_ENV: &str = "LATENT_INSPECTOR_CACHE_DIR";
const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const DOWNLOAD_USER_AGENT: &str = concat!("latent-inspector/", env!("CARGO_PKG_VERSION"));
const VERIFICATION_CACHE_SUFFIX: &str = ".sha256-cache.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactCacheStatus {
    Missing,
    PresentUnverified,
    PresentVerified,
    Invalid,
    Unusable,
    Unknown,
}

impl ArtifactCacheStatus {
    pub fn label(self) -> &'static str {
        match self {
            ArtifactCacheStatus::Missing => "missing",
            ArtifactCacheStatus::PresentUnverified => "present-unverified",
            ArtifactCacheStatus::PresentVerified => "present-verified",
            ArtifactCacheStatus::Invalid => "invalid",
            ArtifactCacheStatus::Unusable => "unusable",
            ArtifactCacheStatus::Unknown => "unknown",
        }
    }

    pub fn is_usable(self) -> bool {
        matches!(
            self,
            ArtifactCacheStatus::PresentUnverified | ArtifactCacheStatus::PresentVerified
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedArtifactInfo {
    pub relative_path: String,
    pub url: String,
    pub absolute_path: String,
    pub cache_status: ArtifactCacheStatus,
    pub cache_summary: String,
    pub byte_size: Option<u64>,
    pub verification_label: String,
    pub verification_note: Option<String>,
}

impl CachedArtifactInfo {
    fn from_inspection(inspection: &ArtifactInspection) -> Self {
        let byte_size = fs::metadata(&inspection.path)
            .ok()
            .filter(|metadata| metadata.is_file())
            .map(|metadata| metadata.len());
        Self {
            relative_path: inspection.artifact.relative_path.clone(),
            url: inspection.artifact.download_url.clone(),
            absolute_path: inspection.path.display().to_string(),
            cache_status: inspection.state.status(),
            cache_summary: artifact_state_summary(&inspection.state, &inspection.artifact),
            byte_size,
            verification_label: inspection.artifact.checksum.label().to_string(),
            verification_note: inspection.artifact.checksum.note().map(str::to_string),
        }
    }
}

#[derive(Debug, Clone)]
enum ArtifactCacheState {
    Missing,
    PresentUnverified,
    PresentVerified,
    Invalid(String),
    Unusable(String),
}

impl ArtifactCacheState {
    fn status(&self) -> ArtifactCacheStatus {
        match self {
            ArtifactCacheState::Missing => ArtifactCacheStatus::Missing,
            ArtifactCacheState::PresentUnverified => ArtifactCacheStatus::PresentUnverified,
            ArtifactCacheState::PresentVerified => ArtifactCacheStatus::PresentVerified,
            ArtifactCacheState::Invalid(_) => ArtifactCacheStatus::Invalid,
            ArtifactCacheState::Unusable(_) => ArtifactCacheStatus::Unusable,
        }
    }

    fn is_usable(&self) -> bool {
        self.status().is_usable()
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VerificationCacheRecord {
    expected_sha256: String,
    file_size: u64,
    modified_unix_secs: u64,
    modified_subsec_nanos: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileFingerprint {
    file_size: u64,
    modified_unix_secs: u64,
    modified_subsec_nanos: u32,
}

impl FileFingerprint {
    fn from_path(path: &Path) -> Result<Self, ModelError> {
        let metadata = fs::metadata(path)?;
        let modified = metadata
            .modified()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let modified = modified
            .duration_since(UNIX_EPOCH)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok(Self {
            file_size: metadata.len(),
            modified_unix_secs: modified.as_secs(),
            modified_subsec_nanos: modified.subsec_nanos(),
        })
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

pub fn inspect_model_artifacts(model_name: &str) -> Result<Vec<CachedArtifactInfo>, ModelError> {
    let entry = registry_entry(model_name)?;
    inspect_registry_artifacts(&entry)
}

pub fn inspect_registry_artifacts(
    entry: &RegistryEntry,
) -> Result<Vec<CachedArtifactInfo>, ModelError> {
    let inspection = inspect_cache(entry)?;
    Ok(inspection
        .artifacts
        .iter()
        .map(CachedArtifactInfo::from_inspection)
        .collect())
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

    let progress = ProgressBar::new(0);
    progress.set_style(progress_style(model_name)?);
    progress.set_message(format!("Downloading {}", artifact.relative_path));

    let tmp = temp_download_path(dest)?;
    let result = download_to_file(artifact, &tmp, model_name, &progress);

    if let Err(error) = result {
        progress.abandon_with_message(format!("Failed {}", artifact.relative_path));
        return Err(error);
    }

    progress.finish_with_message(format!("Downloaded {}", artifact.relative_path));
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::rename(&tmp, dest)?;
    if let Checksum::Sha256(expected) = &artifact.checksum {
        persist_verification_cache(dest, expected);
    }
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
    artifact: &ModelArtifact,
    path: &Path,
    model_name: &str,
    progress: &ProgressBar,
) -> Result<(), ModelError> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(DOWNLOAD_CONNECT_TIMEOUT)
        .timeout(DOWNLOAD_REQUEST_TIMEOUT)
        .user_agent(DOWNLOAD_USER_AGENT)
        .build()
        .map_err(|e: reqwest::Error| ModelError::DownloadFailed {
            name: model_name.to_string(),
            reason: e.to_string(),
        })?;
    let mut response = open_download_stream(&client, artifact, path, model_name, progress)?;
    let mut file = open_download_file(path, response.mode)?;
    stream_response_to_file(
        &mut response.inner,
        &mut file,
        model_name,
        progress,
        artifact.relative_path.as_str(),
    )?;
    file.flush()?;
    drop(file);

    match &artifact.checksum {
        Checksum::Sha256(expected) => {
            if let Err(error) = verify_sha256_uncached(path, expected, model_name) {
                let _ = fs::remove_file(path);
                return Err(error);
            }
        }
        Checksum::Pending { reason } => {
            warn!(
                "Skipping checksum verification for {} until metadata is pinned: {}",
                artifact.relative_path, reason
            );
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum DownloadMode {
    Fresh,
    Resume,
}

struct DownloadResponse {
    inner: reqwest::blocking::Response,
    mode: DownloadMode,
}

fn open_download_stream(
    client: &reqwest::blocking::Client,
    artifact: &ModelArtifact,
    path: &Path,
    model_name: &str,
    progress: &ProgressBar,
) -> Result<DownloadResponse, ModelError> {
    let partial_len = partial_download_len(path)?;
    if partial_len > 0 {
        info!(
            "Resuming {} from byte {}",
            artifact.relative_path, partial_len
        );
        let request = client
            .get(&artifact.download_url)
            .header(RANGE, format!("bytes={partial_len}-"));
        let response = send_download_request(request, model_name)?;

        match response.status() {
            reqwest::StatusCode::PARTIAL_CONTENT => {
                if content_range_start(response.headers()) == Some(partial_len) {
                    set_progress_bounds(progress, partial_len, response.content_length());
                    return Ok(DownloadResponse {
                        inner: response,
                        mode: DownloadMode::Resume,
                    });
                }

                warn!(
                    "Server returned unexpected Content-Range while resuming {}; restarting from scratch",
                    artifact.relative_path
                );
                reset_partial_download(path)?;
                return fresh_download_stream(client, artifact, model_name, progress);
            }
            reqwest::StatusCode::OK => {
                warn!(
                    "Server ignored Range request for {}; restarting download from scratch",
                    artifact.relative_path
                );
                reset_partial_download(path)?;
                set_progress_bounds(progress, 0, response.content_length());
                return Ok(DownloadResponse {
                    inner: response,
                    mode: DownloadMode::Fresh,
                });
            }
            reqwest::StatusCode::RANGE_NOT_SATISFIABLE => {
                warn!(
                    "Partial download for {} is no longer valid; restarting from scratch",
                    artifact.relative_path
                );
                reset_partial_download(path)?;
                return fresh_download_stream(client, artifact, model_name, progress);
            }
            _ => {
                return Err(download_status_error(response, model_name));
            }
        }
    }

    fresh_download_stream(client, artifact, model_name, progress)
}

fn fresh_download_stream(
    client: &reqwest::blocking::Client,
    artifact: &ModelArtifact,
    model_name: &str,
    progress: &ProgressBar,
) -> Result<DownloadResponse, ModelError> {
    let response = send_download_request(client.get(&artifact.download_url), model_name)?;
    set_progress_bounds(progress, 0, response.content_length());
    Ok(DownloadResponse {
        inner: response,
        mode: DownloadMode::Fresh,
    })
}

fn send_download_request(
    request: reqwest::blocking::RequestBuilder,
    model_name: &str,
) -> Result<reqwest::blocking::Response, ModelError> {
    request.send().map_err(|e| ModelError::DownloadFailed {
        name: model_name.to_string(),
        reason: e.to_string(),
    })
}

fn download_status_error(response: reqwest::blocking::Response, model_name: &str) -> ModelError {
    let status = response.status();
    ModelError::DownloadFailed {
        name: model_name.to_string(),
        reason: format!("HTTP {status} while downloading artifact"),
    }
}

fn open_download_file(path: &Path, mode: DownloadMode) -> Result<fs::File, ModelError> {
    match mode {
        DownloadMode::Fresh => fs::File::create(path).map_err(ModelError::from),
        DownloadMode::Resume => fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(ModelError::from),
    }
}

fn stream_response_to_file(
    response: &mut reqwest::blocking::Response,
    file: &mut fs::File,
    model_name: &str,
    progress: &ProgressBar,
    artifact_name: &str,
) -> Result<(), ModelError> {
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| ModelError::DownloadFailed {
                name: model_name.to_string(),
                reason: format!("failed to read {artifact_name}: {e}"),
            })?;

        if read == 0 {
            break;
        }

        file.write_all(&buffer[..read])?;
        progress.inc(read as u64);
    }

    Ok(())
}

fn partial_download_len(path: &Path) -> Result<u64, ModelError> {
    match fs::metadata(path) {
        Ok(metadata) => {
            if !metadata.is_file() {
                return Err(ModelError::InvalidArtifactPath {
                    name: "cache".to_string(),
                    path: path.display().to_string(),
                    reason: "partial download path exists but is not a file".to_string(),
                });
            }
            Ok(metadata.len())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error.into()),
    }
}

fn reset_partial_download(path: &Path) -> Result<(), ModelError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn set_progress_bounds(progress: &ProgressBar, offset: u64, remaining: Option<u64>) {
    let total = remaining
        .map(|remaining| remaining + offset)
        .unwrap_or(offset);
    progress.set_length(total);
    progress.set_position(offset);
}

fn content_range_start(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_content_range_start)
}

fn parse_content_range_start(value: &str) -> Option<u64> {
    let range = value.strip_prefix("bytes ")?;
    let (span, _) = range.split_once('/')?;
    let (start, _) = span.split_once('-')?;
    start.parse().ok()
}

fn registry_entry(model_name: &str) -> Result<RegistryEntry, ModelError> {
    registry::find(model_name).ok_or_else(|| ModelError::NotFound(model_name.to_string()))
}

fn artifact_path(artifact: &ModelArtifact) -> Result<PathBuf, ModelError> {
    Ok(cache_dir()?.join(&artifact.relative_path))
}

fn artifact_state_summary(state: &ArtifactCacheState, artifact: &ModelArtifact) -> String {
    match state {
        ArtifactCacheState::Missing => "Artifact is not cached.".to_string(),
        ArtifactCacheState::PresentUnverified => match artifact.checksum.note() {
            Some(note) => format!("Artifact is cached; verification metadata is pending ({note})."),
            None => "Artifact is cached; verification metadata is pending.".to_string(),
        },
        ArtifactCacheState::PresentVerified => {
            "Artifact is cached and checksum-verified.".to_string()
        }
        ArtifactCacheState::Invalid(reason) => {
            format!("Artifact must be refreshed before use: {reason}.")
        }
        ArtifactCacheState::Unusable(reason) => {
            format!("Artifact path is unusable: {reason}.")
        }
    }
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
            Checksum::Sha256(expected) => match verify_sha256_cached(&path, expected, model_name) {
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
    let result = verify_sha256_uncached(path, expected, model_name);
    match &result {
        Ok(()) => persist_verification_cache(path, expected),
        Err(_) => clear_verification_cache(path),
    }
    result
}

fn verify_sha256_cached(path: &Path, expected: &str, model_name: &str) -> Result<(), ModelError> {
    if verification_cache_matches(path, expected) {
        debug!("Reusing cached SHA-256 verification for {}", path.display());
        return Ok(());
    }

    verify_sha256(path, expected, model_name)
}

fn verify_sha256_uncached(path: &Path, expected: &str, model_name: &str) -> Result<(), ModelError> {
    debug!("Verifying SHA-256 for {}", path.display());
    let actual = digest_file(path)?;
    verify_sha256_digest(expected, &actual, model_name)
}

fn verification_cache_matches(path: &Path, expected: &str) -> bool {
    try_verification_cache_matches(path, expected).unwrap_or_else(|err| {
        warn!(
            "Ignoring SHA-256 cache metadata for {}: {}",
            path.display(),
            err
        );
        false
    })
}

fn try_verification_cache_matches(path: &Path, expected: &str) -> Result<bool, ModelError> {
    let cache_path = verification_cache_path(path)?;
    let payload = match fs::read_to_string(&cache_path) {
        Ok(payload) => payload,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let record: VerificationCacheRecord = serde_json::from_str(&payload).map_err(|err| {
        ModelError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid verification cache metadata at {}: {err}",
                cache_path.display()
            ),
        ))
    })?;
    if record.expected_sha256 != expected {
        return Ok(false);
    }

    let fingerprint = FileFingerprint::from_path(path)?;
    Ok(record.file_size == fingerprint.file_size
        && record.modified_unix_secs == fingerprint.modified_unix_secs
        && record.modified_subsec_nanos == fingerprint.modified_subsec_nanos)
}

fn persist_verification_cache(path: &Path, expected: &str) {
    if let Err(err) = try_persist_verification_cache(path, expected) {
        warn!(
            "Failed to persist SHA-256 cache metadata for {}: {}",
            path.display(),
            err
        );
    }
}

fn try_persist_verification_cache(path: &Path, expected: &str) -> Result<(), ModelError> {
    let fingerprint = FileFingerprint::from_path(path)?;
    let record = VerificationCacheRecord {
        expected_sha256: expected.to_string(),
        file_size: fingerprint.file_size,
        modified_unix_secs: fingerprint.modified_unix_secs,
        modified_subsec_nanos: fingerprint.modified_subsec_nanos,
    };
    let payload = serde_json::to_vec(&record).map_err(|err| {
        ModelError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize verification cache metadata: {err}"),
        ))
    })?;
    fs::write(verification_cache_path(path)?, payload)?;
    Ok(())
}

fn clear_verification_cache(path: &Path) {
    let cache_path = match verification_cache_path(path) {
        Ok(path) => path,
        Err(err) => {
            warn!(
                "Failed to resolve SHA-256 cache metadata path for {}: {}",
                path.display(),
                err
            );
            return;
        }
    };

    match fs::remove_file(&cache_path) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => warn!(
            "Failed to remove stale SHA-256 cache metadata at {}: {}",
            cache_path.display(),
            err
        ),
    }
}

fn verification_cache_path(path: &Path) -> Result<PathBuf, ModelError> {
    let file_name = path
        .file_name()
        .ok_or_else(|| ModelError::InvalidArtifactPath {
            name: "cache".to_string(),
            path: path.display().to_string(),
            reason: "artifact path has no file name".to_string(),
        })?;
    let mut cache_name = file_name.to_os_string();
    cache_name.push(VERIFICATION_CACHE_SUFFIX);
    Ok(path.with_file_name(cache_name))
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
    use crate::TEST_PROCESS_ENV_LOCK;
    use std::ffi::OsString;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use tempfile::tempdir;

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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RequestLog {
        path: String,
        range: Option<String>,
    }

    #[derive(Debug, Clone)]
    struct ResponseScenario {
        status_code: u16,
        body: Vec<u8>,
        content_range: Option<String>,
        cut_after: Option<usize>,
    }

    impl ResponseScenario {
        fn ok(body: Vec<u8>) -> Self {
            Self {
                status_code: 200,
                body,
                content_range: None,
                cut_after: None,
            }
        }

        fn partial(body: Vec<u8>, content_range: String) -> Self {
            Self {
                status_code: 206,
                body,
                content_range: Some(content_range),
                cut_after: None,
            }
        }

        fn interrupted(body: Vec<u8>, cut_after: usize) -> Self {
            Self {
                status_code: 200,
                body,
                content_range: None,
                cut_after: Some(cut_after),
            }
        }
    }

    struct TestServer {
        base_url: String,
        requests: Arc<Mutex<Vec<RequestLog>>>,
        join_handle: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn spawn(scenarios: Vec<ResponseScenario>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let request_log = Arc::clone(&requests);

            let join_handle = thread::spawn(move || {
                for scenario in scenarios {
                    let (mut stream, _) = listener.accept().unwrap();
                    let reader = stream.try_clone().unwrap();
                    let mut reader = BufReader::new(reader);
                    let mut request_line = String::new();
                    reader.read_line(&mut request_line).unwrap();

                    let path = request_line
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .to_string();

                    let mut range = None;
                    loop {
                        let mut line = String::new();
                        let bytes = reader.read_line(&mut line).unwrap();
                        if bytes == 0 || line == "\r\n" {
                            break;
                        }

                        if let Some((name, value)) = line.split_once(':') {
                            if name.eq_ignore_ascii_case("range") {
                                range = Some(value.trim().to_string());
                            }
                        }
                    }

                    request_log.lock().unwrap().push(RequestLog { path, range });

                    let reason = match scenario.status_code {
                        200 => "OK",
                        206 => "Partial Content",
                        416 => "Range Not Satisfiable",
                        _ => "Test Response",
                    };

                    let mut headers = vec![
                        format!("HTTP/1.1 {} {}", scenario.status_code, reason),
                        format!("Content-Length: {}", scenario.body.len()),
                        "Connection: close".to_string(),
                    ];
                    if let Some(content_range) = &scenario.content_range {
                        headers.push(format!("Content-Range: {content_range}"));
                    }
                    headers.push(String::new());
                    headers.push(String::new());
                    stream.write_all(headers.join("\r\n").as_bytes()).unwrap();

                    let bytes_to_send = scenario.cut_after.unwrap_or(scenario.body.len());
                    stream
                        .write_all(&scenario.body[..bytes_to_send.min(scenario.body.len())])
                        .unwrap();
                    stream.flush().unwrap();
                }
            });

            Self {
                base_url: format!("http://{addr}"),
                requests,
                join_handle: Some(join_handle),
            }
        }

        fn url(&self, path: &str) -> String {
            format!("{}{}", self.base_url, path)
        }

        fn requests(&self) -> Vec<RequestLog> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            if let Some(handle) = self.join_handle.take() {
                handle.join().unwrap();
            }
        }
    }

    #[test]
    fn test_cache_dir_created() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let result = cache_dir();
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_path_format() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let path = model_path("dinov2-vit-l14").unwrap();
        assert!(path.to_str().unwrap().ends_with("dinov2-vit-l14.onnx"));
    }

    #[test]
    fn test_external_data_model_path_format() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
    fn test_sha256_verification_cache_tracks_file_fingerprint() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.bin");
        fs::write(&file, b"hello world").unwrap();

        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        verify_sha256(&file, expected, "test").unwrap();

        let cache_path = verification_cache_path(&file).unwrap();
        assert!(cache_path.is_file());
        assert!(verification_cache_matches(&file, expected));

        fs::write(&file, b"changed content with a different size").unwrap();
        assert!(!verification_cache_matches(&file, expected));
    }

    #[test]
    fn test_cache_dir_uses_env_override() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        let path = cache_dir().unwrap();
        assert_eq!(path, dir.path());
    }

    #[test]
    fn test_is_cached_requires_complete_artifact_bundle() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _guard = CacheDirEnvGuard::set(dir.path());

        // I-JEPA requires two artifacts (model.onnx + model.onnx_data).
        // With only the primary artifact present, the bundle is incomplete.
        let primary = model_path("ijepa-vit-h14").unwrap();
        fs::create_dir_all(primary.parent().unwrap()).unwrap();
        fs::write(&primary, b"onnx").unwrap();

        assert!(!is_cached("ijepa-vit-h14").unwrap());

        // Even with both artifacts present, fake data fails SHA-256
        // verification, so the bundle is still not usable.
        let companion = dir.path().join("ijepa-vit-h14/model.onnx_data");
        fs::write(companion, b"external-data").unwrap();

        assert!(!is_cached("ijepa-vit-h14").unwrap());

        // Verify the artifacts are reported as invalid (checksum mismatch),
        // not missing.
        let artifacts = inspect_model_artifacts("ijepa-vit-h14").unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(artifacts
            .iter()
            .all(|a| a.cache_status == ArtifactCacheStatus::Invalid));
    }

    #[test]
    fn test_is_cached_rejects_empty_artifact_files() {
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = TEST_PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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

    #[test]
    fn test_parse_content_range_start() {
        assert_eq!(
            parse_content_range_start("bytes 4096-8191/16384"),
            Some(4096)
        );
        assert_eq!(parse_content_range_start("items 1-2/3"), None);
    }

    #[test]
    fn test_download_artifact_resumes_partial_transfer() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("bundle/model.onnx");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();

        let content = (0..160_000)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let expected_checksum = hex::encode(Sha256::digest(&content));
        let cut_after = 48_000;
        let server = TestServer::spawn(vec![
            ResponseScenario::interrupted(content.clone(), cut_after),
            ResponseScenario::partial(
                content[cut_after..].to_vec(),
                format!("bytes {cut_after}-159999/160000"),
            ),
        ]);

        let artifact = ModelArtifact {
            relative_path: "bundle/model.onnx".to_string(),
            download_url: server.url("/model.onnx"),
            checksum: Checksum::Sha256(expected_checksum),
        };

        let first = download_artifact("test-model", &artifact, &dest);
        assert!(matches!(first, Err(ModelError::DownloadFailed { .. })));

        let temp_path = temp_download_path(&dest).unwrap();
        let partial_len = fs::metadata(&temp_path).unwrap().len();
        assert!(partial_len >= cut_after as u64);
        assert!(partial_len < content.len() as u64);

        download_artifact("test-model", &artifact, &dest).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), content);
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].range, None);
        assert_eq!(requests[1].range, Some(format!("bytes={partial_len}-")));
    }

    #[test]
    fn test_download_artifact_persists_verification_cache_after_success() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("bundle/model.onnx");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();

        let content = b"verified artifact".to_vec();
        let expected_checksum = hex::encode(Sha256::digest(&content));
        let server = TestServer::spawn(vec![ResponseScenario::ok(content)]);

        let artifact = ModelArtifact {
            relative_path: "bundle/model.onnx".to_string(),
            download_url: server.url("/model.onnx"),
            checksum: Checksum::Sha256(expected_checksum.clone()),
        };

        download_artifact("test-model", &artifact, &dest).unwrap();

        assert!(verification_cache_path(&dest).unwrap().is_file());
        assert!(verification_cache_matches(&dest, &expected_checksum));
    }

    #[test]
    fn test_download_artifact_restarts_when_range_is_ignored() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("bundle/model.onnx");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();

        let content = (0..96_000)
            .map(|index| (index % 241) as u8)
            .collect::<Vec<_>>();
        let expected_checksum = hex::encode(Sha256::digest(&content));
        let cut_after = 24_000;
        let server = TestServer::spawn(vec![
            ResponseScenario::interrupted(content.clone(), cut_after),
            ResponseScenario::ok(content.clone()),
        ]);

        let artifact = ModelArtifact {
            relative_path: "bundle/model.onnx".to_string(),
            download_url: server.url("/model.onnx"),
            checksum: Checksum::Sha256(expected_checksum),
        };

        let first = download_artifact("test-model", &artifact, &dest);
        assert!(matches!(first, Err(ModelError::DownloadFailed { .. })));

        let temp_path = temp_download_path(&dest).unwrap();
        let partial_len = fs::metadata(&temp_path).unwrap().len();
        assert!(partial_len >= cut_after as u64);

        download_artifact("test-model", &artifact, &dest).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), content);
        assert!(!temp_path.exists());

        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].range, None);
        assert_eq!(requests[1].range, Some(format!("bytes={partial_len}-")));
    }
}
