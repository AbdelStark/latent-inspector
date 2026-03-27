use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::{tempdir, TempDir};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("validation")
}

fn copy_fixture_dir() -> TempDir {
    let dir = tempdir().unwrap();
    for entry in fs::read_dir(fixture_root()).unwrap() {
        let entry = entry.unwrap();
        let src = entry.path();
        let dest = dir.path().join(entry.file_name());
        if src.is_file() {
            fs::copy(src, dest).unwrap();
        }
    }
    dir
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args(args)
        .output()
        .unwrap()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn write_test_image(dir: &Path) -> PathBuf {
    let path = dir.join("fixture.png");
    let image = image::RgbImage::from_fn(224, 224, |x, y| {
        image::Rgb([(x % 255) as u8, (y % 255) as u8, ((x + y) % 255) as u8])
    });
    image.save(&path).unwrap();
    path
}

#[test]
fn validate_terminal_succeeds_for_known_model() {
    let output = run(&["validate", "--model", "dinov2-vit-l14"]);

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Validation Summary"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("validated"));
}

#[test]
fn validate_unknown_model_returns_usage_exit_code() {
    let output = run(&["validate", "--model", "not-a-model"]);

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown model identifier"));
}

#[test]
fn validate_json_output_matches_contract_shape() {
    let outdir = tempdir().unwrap();
    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--format",
        "json",
        "--output",
        outdir.path().to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["model"], "dinov2-vit-l14");
    assert_eq!(summary["status"], "validated");
    assert!(summary["preprocess"]["summary"].is_string());
    assert!(summary["parity"]["artifact_id"].is_string());
    assert!(summary["tensors"].is_array());
}

#[test]
fn validate_detects_reference_drift() {
    let fixtures = copy_fixture_dir();
    let reference_path = fixtures.path().join("dinov2-vit-l14.reference.json");
    let mut reference = read_json(&reference_path);
    reference["observed"]["fixtures"][0]["patch_signature"][0] = Value::from(9.9);
    fs::write(
        &reference_path,
        serde_json::to_string_pretty(&reference).unwrap(),
    )
    .unwrap();

    let outdir = tempdir().unwrap();
    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--fixture-set",
        fixtures.path().join("manifest.json").to_str().unwrap(),
        "--format",
        "json",
        "--output",
        outdir.path().to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["status"], "failed");
    assert_eq!(summary["parity"]["status"], "failed");
    assert!(summary["parity"]["deltas"]
        .as_array()
        .unwrap()
        .iter()
        .any(|delta| delta["name"] == "fixtures.gradient-224.patch_signature[0]"));
}

#[test]
fn validate_marks_stale_contract_evidence_without_reporting_runtime_failure() {
    let fixtures = copy_fixture_dir();
    let contract_path = fixtures.path().join("dinov2-vit-l14.contract.json");
    let mut contract = read_json(&contract_path);
    contract["profile"]["evidence_timestamp"] = Value::from("2026-03-28T00:00:00Z");
    fs::write(
        &contract_path,
        serde_json::to_string_pretty(&contract).unwrap(),
    )
    .unwrap();

    let outdir = tempdir().unwrap();
    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--fixture-set",
        fixtures.path().join("manifest.json").to_str().unwrap(),
        "--format",
        "json",
        "--output",
        outdir.path().to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["status"], "stale");
    assert_eq!(summary["preprocess"]["status"], "stale");
    assert_eq!(summary["tensors"][0]["status"], "stale");
    assert_eq!(summary["parity"]["status"], "validated");
}

#[test]
fn validate_marks_stale_reference_identity_without_reporting_parity_drift() {
    let fixtures = copy_fixture_dir();
    let reference_path = fixtures.path().join("dinov2-vit-l14.reference.json");
    let mut reference = read_json(&reference_path);
    reference["artifact_id"] = Value::from("dinov2-vit-l14:standard:outdated");
    reference["observed"]["fixtures"][0]["patch_signature"][0] = Value::from(42.0);
    fs::write(
        &reference_path,
        serde_json::to_string_pretty(&reference).unwrap(),
    )
    .unwrap();

    let outdir = tempdir().unwrap();
    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--fixture-set",
        fixtures.path().join("manifest.json").to_str().unwrap(),
        "--format",
        "json",
        "--output",
        outdir.path().to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["status"], "stale");
    assert_eq!(summary["parity"]["status"], "stale");
    assert!(summary["parity"]["deltas"].is_null());
}

#[test]
fn validate_refresh_goldens_rewrites_reference_artifact() {
    let fixtures = copy_fixture_dir();
    let reference_path = fixtures.path().join("dinov2-vit-l14.reference.json");
    let original = read_json(&reference_path);
    let mut reference = read_json(&reference_path);
    reference["observed"]["fixtures"][0]["patch_signature"][0] = Value::from(9.9);
    fs::write(
        &reference_path,
        serde_json::to_string_pretty(&reference).unwrap(),
    )
    .unwrap();

    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--fixture-set",
        fixtures.path().join("manifest.json").to_str().unwrap(),
        "--refresh-goldens",
    ]);

    assert_eq!(output.status.code(), Some(0));
    let refreshed = read_json(&reference_path);
    assert_eq!(refreshed, original);
}

#[test]
fn inspect_json_includes_validation_summary() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output = run(&[
        "inspect",
        image.to_str().unwrap(),
        "--model",
        "dinov2-vit-l14",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(0));
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["validation"]["model"], "dinov2-vit-l14");
    assert_eq!(payload["validation"]["status"], "validated");
}

#[test]
fn compare_html_includes_validation_summary() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("compare");
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14",
        "--format",
        "html",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let html = fs::read_to_string(output_dir.join("report.html")).unwrap();
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14"));
}

#[test]
fn compare_json_includes_pairwise_overview() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14,dinov2-vit-l14",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(0));
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let labels = payload["overview"]["linear_cka_matrix"]["labels"]
        .as_array()
        .unwrap();
    assert_eq!(labels[0], Value::from("dinov2-vit-l14#1"));
    assert_eq!(labels[1], Value::from("dinov2-vit-l14#2"));
    assert!(payload["overview"]["comparison_highlights"].is_array());
}

#[test]
fn compare_png_writes_pairwise_heatmaps() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("compare-png");
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14,dinov2-vit-l14",
        "--format",
        "png",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output_dir.join("dinov2-vit-l14_1_pca.png").exists());
    assert!(output_dir.join("dinov2-vit-l14_2_pca.png").exists());
    assert!(output_dir.join("linear_cka.png").exists());
    assert!(output_dir.join("knn_overlap_k10.png").exists());
    assert!(output_dir.join("patch_correspondence.png").exists());
}

#[test]
fn inspect_png_writes_variance_chart() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("inspect-png");
    let output = run(&[
        "inspect",
        image.to_str().unwrap(),
        "--model",
        "dinov2-vit-l14",
        "--format",
        "png",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output_dir.join("dinov2-vit-l14_pca.png").exists());
    assert!(output_dir.join("dinov2-vit-l14_variance.png").exists());
}
