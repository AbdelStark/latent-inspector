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

#[test]
fn models_output_includes_evidence_and_fixture_summary() {
    let output = Command::new(bin()).args(["models"]).output().unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Available models"));
    assert!(stdout.contains("Runtime"));
    assert!(stdout.contains("Evidence"));
    assert!(stdout.contains("Validation fixtures:"));
    assert!(stdout.contains("Evidence summary:"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("onnx-ready"));
    assert!(stdout.contains("approved"));
    assert!(stdout.contains("mae-vit-l16"));
    assert!(stdout.contains("stub-only"));
    assert!(stdout.contains("unverified"));
}

#[test]
fn models_verbose_output_includes_evidence_and_cache_details() {
    let output = Command::new(bin())
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
    assert!(stdout.contains("Cache: "));
    assert!(stdout.contains("Artifact: dinov2-vit-l14.onnx"));
    assert!(stdout.contains("Cache dir:"));
}

#[test]
fn models_json_output_writes_structured_catalog() {
    let outdir = tempdir().unwrap();
    let output = Command::new(bin())
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
    assert_eq!(payload["entries"].as_array().unwrap().len(), 6);
    let dinov2 = payload["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "dinov2-vit-l14")
        .unwrap();
    assert_eq!(dinov2["runtime_support"], "onnx-ready");
    assert_eq!(
        dinov2["artifacts"][0]["relative_path"],
        "dinov2-vit-l14.onnx"
    );
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "models.json");
    assert_eq!(manifest["artifacts"][0]["path"], "models.json");
}

#[test]
fn models_html_output_writes_shareable_catalog() {
    let outdir = tempdir().unwrap();
    let output = Command::new(bin())
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
    assert!(html.contains("Validation fixtures:"));
    assert!(html.contains("Runtime"));
    assert!(html.contains("dinov2-vit-l14"));
    assert!(html.contains("Registry availability, cache state, and validation evidence"));
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "models");
    assert_eq!(manifest["format"], "html");
    assert_eq!(manifest["primary_artifact"], "models.html");
    assert_eq!(manifest["artifacts"][0]["path"], "models.html");
}

#[test]
fn models_rejects_png_output() {
    let output = Command::new(bin())
        .args(["models", "--format", "png"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("models only supports terminal, json, or html output"));
}
