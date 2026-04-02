//! Shared test helpers used across integration test files.
#![allow(dead_code)]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

pub fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

pub fn read_artifact_manifest(dir: &Path) -> Value {
    read_json(&dir.join("artifacts.json"))
}

pub fn artifact_entry<'a>(manifest: &'a Value, path: &str) -> &'a Value {
    manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["path"] == path)
        .unwrap_or_else(|| panic!("missing artifact entry for {path}"))
}

pub fn assert_artifact_metadata(manifest: &Value, path: &str) {
    let artifact = artifact_entry(manifest, path);
    assert!(artifact["byte_size"].as_u64().unwrap() > 0);
    assert_eq!(artifact["sha256"].as_str().unwrap().len(), 64);
}

pub fn sha256_preview(digest: &str) -> String {
    if digest.len() > 16 {
        format!("{}…", &digest[..16])
    } else {
        digest.to_string()
    }
}

pub fn digest_preview_for(path: &Path) -> String {
    let digest = hex::encode(Sha256::digest(fs::read(path).unwrap()));
    sha256_preview(&digest)
}
