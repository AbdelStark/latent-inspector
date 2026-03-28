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

fn read_artifact_manifest(dir: &Path) -> Value {
    read_json(&dir.join("artifacts.json"))
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
fn validate_terminal_marks_stub_backend_unverified() {
    let output = run(&["validate", "--model", "dinov2-vit-l14"]);

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Validation Summary"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("unverified"));
    assert!(stdout.contains("stub"));
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

    assert_eq!(output.status.code(), Some(1));
    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["model"], "dinov2-vit-l14");
    assert_eq!(summary["status"], "unverified");
    assert_eq!(summary["backend"]["kind"], "stub");
    assert_eq!(summary["backend"]["status"], "unverified");
    assert!(summary["preprocess"]["summary"].is_string());
    assert!(summary["parity"]["artifact_id"].is_string());
    assert!(summary["tensors"].is_array());
    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "validate");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "validation.json");
    assert_eq!(manifest["artifacts"][0]["path"], "validation.json");
    assert_eq!(manifest["validation"][0]["status"], "unverified");
}

#[test]
fn validate_html_output_writes_companion_json_bundle() {
    let outdir = tempdir().unwrap();
    let output = run(&[
        "validate",
        "--model",
        "dinov2-vit-l14",
        "--format",
        "html",
        "--output",
        outdir.path().to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let html = fs::read_to_string(outdir.path().join("validation.html")).unwrap();
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14"));
    assert!(html.contains("Stub backend"));

    let payload = read_json(&outdir.path().join("validation.json"));
    let summary = &payload[0];
    assert_eq!(summary["model"], "dinov2-vit-l14");
    assert_eq!(summary["status"], "unverified");

    let manifest = read_artifact_manifest(outdir.path());
    assert_eq!(manifest["command"], "validate");
    assert_eq!(manifest["format"], "html");
    assert_eq!(manifest["primary_artifact"], "validation.html");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "validation.html"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "validation.json"));
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
    assert_eq!(summary["parity"]["status"], "unverified");
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

    assert_eq!(output.status.code(), Some(1));
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
    assert_eq!(payload["validation"]["status"], "unverified");
    assert_eq!(payload["validation"]["backend"]["kind"], "stub");
}

#[test]
fn inspect_html_includes_variance_spectrum_and_validation_summary() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("inspect");
    let output = run(&[
        "inspect",
        image.to_str().unwrap(),
        "--model",
        "dinov2-vit-l14",
        "--format",
        "html",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let html = fs::read_to_string(output_dir.join("report.html")).unwrap();
    assert!(html.contains("Representation Inspect"));
    assert!(html.contains("Variance Spectrum"));
    assert!(html.contains("Visual Artefacts"));
    assert!(html.contains("dinov2-vit-l14_pca.png"));
    assert!(html.contains("dinov2-vit-l14_variance.png"));
    assert!(html.contains("Components @ 99%"));
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14"));
    assert!(html.contains("Stub backend"));
    let payload = read_json(&output_dir.join("inspect.json"));
    assert_eq!(payload["model"], "dinov2-vit-l14");
    assert_eq!(payload["validation"]["status"], "unverified");
    assert!(output_dir.join("dinov2-vit-l14_pca.png").exists());
    assert!(output_dir.join("dinov2-vit-l14_variance.png").exists());
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "inspect");
    assert_eq!(manifest["format"], "html");
    assert_eq!(manifest["primary_artifact"], "report.html");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "report.html"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "inspect.json"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_pca.png"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_variance.png"));
    assert_eq!(manifest["validation"][0]["status"], "unverified");
}

#[test]
fn compare_html_embeds_visual_assets_and_validation_summary() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("compare");
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14,dinov2-vit-l14",
        "--format",
        "html",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let html = fs::read_to_string(output_dir.join("report.html")).unwrap();
    assert!(html.contains("Visual Artefacts"));
    assert!(html.contains("Per-model PCA projections"));
    assert!(html.contains("Pairwise metric heatmaps"));
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14#1"));
    assert!(html.contains("dinov2-vit-l14#2"));
    assert!(html.contains("dinov2-vit-l14_1_pca.png"));
    assert!(html.contains("dinov2-vit-l14_2_pca.png"));
    assert!(html.contains("linear_cka.png"));
    let payload = read_json(&output_dir.join("compare.json"));
    assert_eq!(
        payload["requested_models"],
        Value::from(vec!["dinov2-vit-l14", "dinov2-vit-l14"])
    );
    assert_eq!(payload["validation"].as_array().unwrap().len(), 2);
    assert!(output_dir.join("dinov2-vit-l14_1_pca.png").exists());
    assert!(output_dir.join("dinov2-vit-l14_2_pca.png").exists());
    assert!(output_dir.join("cls_cosine.png").exists());
    assert!(output_dir.join("linear_cka.png").exists());
    assert!(output_dir.join("knn_overlap_k10.png").exists());
    assert!(output_dir.join("patch_correspondence.png").exists());
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "compare");
    assert_eq!(manifest["format"], "html");
    assert_eq!(manifest["primary_artifact"], "report.html");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "compare.json"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_1_pca.png"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "linear_cka.png"));
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
    assert_eq!(payload["image"], Value::from(image.display().to_string()));
    assert_eq!(
        payload["requested_models"],
        Value::from(vec!["dinov2-vit-l14", "dinov2-vit-l14"])
    );
    let labels = payload["overview"]["linear_cka_matrix"]["labels"]
        .as_array()
        .unwrap();
    assert_eq!(labels[0], Value::from("dinov2-vit-l14#1"));
    assert_eq!(labels[1], Value::from("dinov2-vit-l14#2"));
    assert!(payload["overview"]["comparison_highlights"].is_array());
    assert_eq!(payload["validation"].as_array().unwrap().len(), 2);
}

#[test]
fn compare_json_output_writes_structured_report() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("compare-json");
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14,dinov2-vit-l14",
        "--format",
        "json",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("compare.json"));

    let payload = read_json(&output_dir.join("compare.json"));
    assert_eq!(payload["image"], Value::from(image.display().to_string()));
    assert_eq!(payload["metrics"].as_array().unwrap().len(), 2);
    assert!(payload["overview"]["cls_cosine_matrix"]["rows"].is_array());
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "compare");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "compare.json");
    assert_eq!(manifest["validation"].as_array().unwrap().len(), 2);
}

#[test]
fn compare_json_reports_alignment_and_metric_caveats_for_stubbed_planned_models() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output = run(&[
        "compare",
        image.to_str().unwrap(),
        "--models",
        "dinov2-vit-l14,mae-vit-l16,siglip-so400m",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(0));
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();

    let comparisons = payload["comparisons"].as_array().unwrap();
    let dinov2_mae = comparisons
        .iter()
        .find(|comparison| {
            comparison["model_a"] == "dinov2-vit-l14" && comparison["model_b"] == "mae-vit-l16"
        })
        .unwrap();
    assert_eq!(dinov2_mae["alignment"]["compared_patch_count"], 196);
    assert!(dinov2_mae["alignment"]["note"]
        .as_str()
        .unwrap()
        .contains("different patch grids"));
    assert!(dinov2_mae["metric_caveats"]
        .as_array()
        .unwrap()
        .iter()
        .any(|caveat| caveat["key"] == "cls_cosine_sim"
            && caveat["reason"]
                .as_str()
                .unwrap()
                .contains("only one model exposes a CLS token")));

    let dinov2_siglip = comparisons
        .iter()
        .find(|comparison| {
            comparison["model_a"] == "dinov2-vit-l14" && comparison["model_b"] == "siglip-so400m"
        })
        .unwrap();
    assert!(dinov2_siglip["metric_caveats"]
        .as_array()
        .unwrap()
        .iter()
        .any(|caveat| caveat["key"] == "mean_patch_correspondence"
            && caveat["reason"]
                .as_str()
                .unwrap()
                .contains("embedding dimensions differ")));

    let validation = payload["validation"].as_array().unwrap();
    assert_eq!(validation[0]["status"], "unverified");
    assert_eq!(validation[0]["backend"]["kind"], "stub");
    assert_eq!(validation[0]["backend"]["status"], "unverified");
    assert_eq!(validation[1]["status"], "unverified");
    assert_eq!(validation[2]["status"], "unverified");
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
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "compare");
    assert_eq!(manifest["format"], "png");
    assert!(manifest["primary_artifact"].is_null());
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_1_pca.png"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "linear_cka.png"));
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
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "inspect");
    assert_eq!(manifest["format"], "png");
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_pca.png"));
    assert!(manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|artifact| artifact["path"] == "dinov2-vit-l14_variance.png"));
}

#[test]
fn inspect_json_output_writes_structured_report() {
    let dir = tempdir().unwrap();
    let image = write_test_image(dir.path());
    let output_dir = dir.path().join("inspect-json");
    let output = run(&[
        "inspect",
        image.to_str().unwrap(),
        "--model",
        "dinov2-vit-l14",
        "--format",
        "json",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inspect.json"));

    let payload = read_json(&output_dir.join("inspect.json"));
    assert_eq!(payload["image"], Value::from(image.display().to_string()));
    assert_eq!(payload["model"], "dinov2-vit-l14");
    assert_eq!(payload["validation"]["status"], "unverified");
    assert_eq!(payload["validation"]["backend"]["kind"], "stub");
    assert!(payload["variance_spectrum"]["ratios"].is_array());
    assert!(payload["variance_spectrum"]["top10_concentration"].is_number());
    let manifest = read_artifact_manifest(&output_dir);
    assert_eq!(manifest["command"], "inspect");
    assert_eq!(manifest["format"], "json");
    assert_eq!(manifest["primary_artifact"], "inspect.json");
    assert_eq!(manifest["validation"][0]["status"], "unverified");
}
