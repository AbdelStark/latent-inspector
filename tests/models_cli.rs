mod common;
use common::*;

use std::fs;
use std::process::Command;
use tempfile::tempdir;

fn models_command(cache_dir: &tempfile::TempDir) -> Command {
    let mut command = Command::new(bin());
    command.env("LATENT_INSPECTOR_CACHE_DIR", cache_dir.path());
    command
}

#[test]
fn models_output_includes_evidence_and_fixture_summary() {
    let cache_dir = tempdir().unwrap();
    let output = models_command(&cache_dir)
        .args(["models"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Available models"));
    assert!(stdout.contains("Readiness"));
    assert!(stdout.contains("Runtime"));
    assert!(stdout.contains("Evidence"));
    assert!(stdout.contains("Validation fixtures:"));
    assert!(stdout.contains("Evidence summary:"));
    assert!(stdout.contains("Readiness summary:"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("needs-download"));
    assert!(stdout.contains("onnx-ready"));
    assert!(stdout.contains("approved"));
    assert!(stdout.contains("mae-vit-l16"));
    assert!(stdout.contains("stub-only"));
    assert!(stdout.contains("unverified"));
    assert!(stdout.contains("Artifact summary:"));
    assert!(stdout.contains("8 total"));
}

#[test]
fn models_verbose_output_includes_evidence_and_cache_details() {
    let cache_dir = tempdir().unwrap();
    // Empty cache dir — DINOv2 artifact will be "missing".
    let output = models_command(&cache_dir)
        .args(["models", "--verbose"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(
        "Evidence: Approved validation contract and parity artifacts are current for the active registry profile."
    ));
    assert!(stdout.contains("Runtime: Normal runs load the registered ONNX artifact."));
    assert!(stdout.contains("[standard @ 2026-03-27T12:00:00Z]"));
    assert!(stdout.contains("Artifact: dinov2-vit-l14.onnx [missing |"));
    assert!(stdout.contains("Path: "));
    assert!(stdout.contains("Cache dir:"));
}

#[test]
fn models_json_output_writes_structured_catalog() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();
    let output = models_command(&cache_dir)
        .args([
            "models",
            "--format",
            "json",
            "--output",
            outdir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload = read_json(&outdir.path().join("models.json"));
    assert_eq!(payload["summary"]["total_models"], 7);
    assert_eq!(payload["summary"]["ready_models"], 3);
    assert_eq!(payload["summary"]["evidence"]["approved"], 1);
    assert_eq!(payload["summary"]["artifacts"]["total"], 8);
    assert_eq!(payload["summary"]["artifacts"]["usable"], 0);
    assert_eq!(payload["summary"]["readiness"]["ready"], 0);
    assert_eq!(payload["summary"]["readiness"]["needs_download"], 3);
    assert_eq!(payload["summary"]["readiness"]["planned"], 4);
    assert_eq!(payload["entries"].as_array().unwrap().len(), 7);
    let dinov2 = payload["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "dinov2-vit-l14")
        .unwrap();
    assert_eq!(dinov2["readiness_status"], "needs-download");
    assert!(dinov2["readiness_summary"]
        .as_str()
        .unwrap()
        .contains("missing"));
    assert!(dinov2["next_steps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|step| step
            .as_str()
            .unwrap()
            .contains("models --download dinov2-vit-l14")));
    assert_eq!(dinov2["runtime_support"], "onnx-ready");
    assert_eq!(dinov2["artifact_summary"]["total"], 1);
    assert_eq!(
        dinov2["artifacts"][0]["relative_path"],
        "dinov2-vit-l14.onnx"
    );
    assert_eq!(dinov2["artifacts"][0]["cache_status"], "missing");
    assert_eq!(
        dinov2["artifacts"][0]["absolute_path"],
        cache_dir
            .path()
            .join("dinov2-vit-l14.onnx")
            .display()
            .to_string()
    );
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "models.json");
    assert_eq!(manifest["context"]["mode"], "catalog");
    assert_eq!(manifest["context"]["verbose"], false);
    assert_eq!(manifest["artifacts"][0]["path"], "models.json");
    assert_eq!(manifest["summary"]["summary"]["total_models"], 7);
    assert_eq!(manifest["summary"]["summary"]["ready_models"], 3);
    assert!(manifest["validation_summary"].is_null());
    assert_artifact_metadata(&manifest, "models.json");
}

#[test]
fn models_html_output_writes_shareable_catalog() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();
    let output = models_command(&cache_dir)
        .args([
            "models",
            "--format",
            "html",
            "--output",
            outdir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let html = fs::read_to_string(outdir.path().join("models.html")).unwrap();
    assert!(html.contains("Model inventory"));
    assert!(html.contains("Export Bundle"));
    assert!(html.contains("artifacts.json"));
    assert!(html.contains("Validation fixtures:"));
    assert!(html.contains("Readiness summary:"));
    assert!(html.contains("Runtime"));
    assert!(html.contains("Readiness"));
    assert!(html.contains("dinov2-vit-l14"));
    assert!(html.contains("Registry availability, cache state, and validation evidence"));
    assert!(html.contains("Ready to run"));
    assert!(html.contains("Artifact details"));
    assert!(html.contains("SHA-256"));
    let payload = read_json(&outdir.path().join("models.json"));
    assert_eq!(payload["summary"]["total_models"], 7);
    assert_eq!(payload["summary"]["ready_models"], 3);
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "html");
    assert_eq!(manifest["primary_artifact"], "models.html");
    assert_eq!(manifest["context"]["mode"], "catalog");
    assert_eq!(manifest["summary"]["fixture_set"], "standard");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "models.html"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "models.json"));
    assert_artifact_metadata(&manifest, "models.html");
    assert_artifact_metadata(&manifest, "models.json");
    assert!(html.contains(&digest_preview_for(&outdir.path().join("models.json"))));
}

#[test]
fn models_rejects_png_output() {
    let cache_dir = tempdir().unwrap();
    let output = models_command(&cache_dir)
        .args(["models", "--format", "png"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("models only supports terminal, json, or html output"));
}

/// Download report structure test.
///
/// Note: with real SHA-256 checksums, staging fake ONNX data produces
/// `invalid` status. If network is unavailable, the download will fail.
/// This test is marked `#[ignore]` for CI environments without network.
#[test]
#[ignore]
fn models_download_json_output_writes_structured_report() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();

    let output = models_command(&cache_dir)
        .args([
            "models",
            "--download",
            "dinov2-vit-l14",
            "--format",
            "json",
            "--output",
            outdir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload = read_json(&outdir.path().join("download.json"));
    assert_eq!(payload["model"], "dinov2-vit-l14");
    assert!(
        payload["action"] == "downloaded" || payload["action"] == "already-cached",
        "unexpected action: {}",
        payload["action"]
    );
    assert_eq!(payload["entry"]["readiness_status"], "ready");
    assert_eq!(payload["artifact_changes"].as_array().unwrap().len(), 1);
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "download.json");
    assert_eq!(manifest["context"]["mode"], "download");
    assert_eq!(manifest["context"]["model"], "dinov2-vit-l14");
    assert_artifact_metadata(&manifest, "download.json");
}

/// Download HTML report test. Requires network access.
#[test]
#[ignore]
fn models_download_html_output_writes_shareable_report() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();

    let output = models_command(&cache_dir)
        .args([
            "models",
            "--download",
            "dinov2-vit-l14",
            "--format",
            "html",
            "--output",
            outdir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let html = fs::read_to_string(outdir.path().join("download.html")).unwrap();
    assert!(html.contains("Model download"));
    assert!(html.contains("Artifact Changes"));
    assert!(html.contains("Ready to run"));
    assert!(html.contains("SHA-256"));
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["primary_artifact"], "download.html");
    assert_eq!(manifest["context"]["mode"], "download");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["path"] == "download.html"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["path"] == "download.json"));
    assert_artifact_metadata(&manifest, "download.html");
    assert_artifact_metadata(&manifest, "download.json");
}
