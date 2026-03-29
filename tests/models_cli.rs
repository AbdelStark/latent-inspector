use serde_json::Value;
use std::fs;
use std::process::Command;
use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

fn read_json(path: &std::path::Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn read_artifact_manifest(dir: &std::path::Path) -> Value {
    read_json(&dir.join("artifacts.json"))
}

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
    assert!(stdout.contains("6 total"));
}

#[test]
fn models_verbose_output_includes_evidence_and_cache_details() {
    let cache_dir = tempdir().unwrap();
    fs::write(cache_dir.path().join("dinov2-vit-l14.onnx"), b"fake onnx").unwrap();
    let output = models_command(&cache_dir)
        .args(["models", "--verbose"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Readiness: Ready to run"));
    assert!(stdout.contains(
        "Evidence: Approved validation contract and parity artifacts are current for the active registry profile."
    ));
    assert!(stdout.contains("Runtime: Normal runs load the registered ONNX artifact."));
    assert!(stdout.contains("next: Pin checksum metadata"));
    assert!(stdout.contains("[standard @ 2026-03-27T12:00:00Z]"));
    assert!(stdout.contains("Artifacts: 1 total, 1 usable, 0 verified, 1 pending verification"));
    assert!(stdout.contains("Artifact: dinov2-vit-l14.onnx [present-unverified | pending]"));
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
    assert_eq!(payload["summary"]["total_models"], 6);
    assert_eq!(payload["summary"]["ready_models"], 1);
    assert_eq!(payload["summary"]["evidence"]["approved"], 1);
    assert_eq!(payload["summary"]["artifacts"]["total"], 6);
    assert_eq!(payload["summary"]["artifacts"]["usable"], 0);
    assert_eq!(payload["summary"]["readiness"]["ready"], 0);
    assert_eq!(payload["summary"]["readiness"]["needs_download"], 1);
    assert_eq!(payload["summary"]["readiness"]["planned"], 5);
    assert_eq!(payload["entries"].as_array().unwrap().len(), 6);
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
    assert_eq!(manifest["summary"]["summary"]["total_models"], 6);
    assert_eq!(manifest["summary"]["summary"]["ready_models"], 1);
    assert!(manifest["validation_summary"].is_null());
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
    let payload = read_json(&outdir.path().join("models.json"));
    assert_eq!(payload["summary"]["total_models"], 6);
    assert_eq!(payload["summary"]["ready_models"], 1);
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

#[test]
fn models_download_json_output_writes_structured_report_when_cached() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();
    fs::write(cache_dir.path().join("dinov2-vit-l14.onnx"), b"fake onnx").unwrap();

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
    assert_eq!(payload["action"], "already-cached");
    assert_eq!(payload["entry"]["readiness_status"], "ready");
    assert_eq!(payload["artifact_changes"].as_array().unwrap().len(), 1);
    assert_eq!(
        payload["artifact_changes"][0]["previous_status"],
        "present-unverified"
    );
    assert_eq!(
        payload["artifact_changes"][0]["current_status"],
        "present-unverified"
    );
    assert_eq!(payload["artifact_changes"][0]["downloaded"], false);
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "download.json");
    assert_eq!(manifest["context"]["mode"], "download");
    assert_eq!(manifest["context"]["model"], "dinov2-vit-l14");
}

#[test]
fn models_download_html_output_writes_shareable_report_when_cached() {
    let outdir = tempdir().unwrap();
    let cache_dir = tempdir().unwrap();
    fs::write(cache_dir.path().join("dinov2-vit-l14.onnx"), b"fake onnx").unwrap();

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
    assert!(html.contains("already-cached"));
    assert!(html.contains("Ready to run"));
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["primary_artifact"], "download.html");
    assert_eq!(manifest["context"]["mode"], "download");
    assert_eq!(manifest["artifacts"][0]["path"], "download.html");
    assert_eq!(manifest["artifacts"][1]["path"], "download.json");
}
