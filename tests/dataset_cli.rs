use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

fn write_image(path: &Path, offset: u8) {
    let image = image::RgbImage::from_fn(224, 224, |x, y| {
        image::Rgb([
            ((x as u8).wrapping_add(offset)) % 255,
            ((y as u8).wrapping_add(offset / 2)) % 255,
            ((x as u8).wrapping_add(y as u8).wrapping_add(offset)) % 255,
        ])
    });
    image.save(path).unwrap();
}

fn write_query_image(dir: &Path) -> PathBuf {
    let path = dir.join("query.png");
    write_image(&path, 7);
    path
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn neighbors_recurses_into_nested_dataset_and_reports_skips() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-a");
    fs::create_dir_all(&nested).unwrap();

    let query_path = write_query_image(dir.path());
    write_image(&dataset_dir.join("root.png"), 11);
    write_image(&nested.join("leaf.png"), 29);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "neighbors",
            query_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--k",
            "2",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Nearest neighbors"));
    assert!(stdout.contains("class-a/leaf"));
    assert!(stdout.contains("Dataset Summary"));
    assert!(stdout.contains("Validation Summary"));
    assert!(stdout.contains("unverified"));
    assert!(stdout.contains("stub"));
    assert!(stdout.contains("Skipped images:"));
    assert!(stdout.contains("broken.png"));
}

#[test]
fn similarity_recurses_into_nested_dataset_and_reports_skips() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-b").join("deep");
    fs::create_dir_all(&nested).unwrap();

    write_image(&dataset_dir.join("root.png"), 17);
    write_image(&nested.join("leaf.png"), 31);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "similarity",
            "--model-a",
            "dinov2-vit-l14",
            "--model-b",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--metric",
            "all",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Representation similarity"));
    assert!(stdout.contains("Linear CKA:"));
    assert!(stdout.contains("k-NN overlap (k=10):"));
    assert!(stdout.contains("Mean CLS cosine sim:"));
    assert!(stdout.contains("Dataset Summary"));
    assert!(stdout.contains("Validation Summary"));
    assert!(stdout.contains("unverified"));
    assert!(stdout.contains("stub"));
    assert!(stdout.contains("broken.png"));
}

#[test]
fn neighbors_json_output_writes_structured_report() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-a");
    fs::create_dir_all(&nested).unwrap();

    let query_path = write_query_image(dir.path());
    write_image(&dataset_dir.join("root.png"), 11);
    write_image(&nested.join("leaf.png"), 29);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    let output_dir = dir.path().join("neighbors-output");
    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "neighbors",
            query_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--k",
            "2",
            "--format",
            "json",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload = read_json(&output_dir.join("neighbors.json"));
    assert_eq!(payload["model"], "dinov2-vit-l14");
    assert_eq!(payload["embedding_basis"], "cls-token");
    assert_eq!(payload["requested_k"], 2);
    assert_eq!(payload["dataset_summary"]["loaded"], 2);
    assert_eq!(payload["dataset_summary"]["skipped"], 1);
    assert_eq!(payload["validation"]["model"], "dinov2-vit-l14");
    assert_eq!(payload["validation"]["status"], "unverified");
    assert_eq!(payload["validation"]["backend"]["kind"], "stub");
    assert_eq!(payload["neighbors"].as_array().unwrap().len(), 2);
    assert!(payload["neighbors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|neighbor| neighbor["image"] == "class-a/leaf"));
}

#[test]
fn neighbors_html_output_includes_validation_summary() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-a");
    fs::create_dir_all(&nested).unwrap();

    let query_path = write_query_image(dir.path());
    write_image(&dataset_dir.join("root.png"), 11);
    write_image(&nested.join("leaf.png"), 29);

    let output_dir = dir.path().join("neighbors-html");
    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "neighbors",
            query_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--k",
            "2",
            "--format",
            "html",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let html = fs::read_to_string(output_dir.join("report.html")).unwrap();
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14"));
}

#[test]
fn similarity_json_output_writes_structured_report() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-b").join("deep");
    fs::create_dir_all(&nested).unwrap();

    write_image(&dataset_dir.join("root.png"), 17);
    write_image(&nested.join("leaf.png"), 31);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    let output_dir = dir.path().join("similarity-output");
    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "similarity",
            "--model-a",
            "dinov2-vit-l14",
            "--model-b",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--metric",
            "all",
            "--format",
            "json",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload = read_json(&output_dir.join("similarity.json"));
    assert_eq!(payload["model_a"], "dinov2-vit-l14");
    assert_eq!(payload["model_b"], "dinov2-vit-l14");
    assert_eq!(payload["requested_metric"], "all");
    assert_eq!(payload["sample_count"], 2);
    assert_eq!(payload["dataset_summary"]["skipped"], 1);
    assert_eq!(payload["validation"].as_array().unwrap().len(), 2);
    assert_eq!(payload["validation"][0]["status"], "unverified");
    assert_eq!(payload["validation"][1]["status"], "unverified");
    assert_eq!(payload["validation"][0]["backend"]["kind"], "stub");
    assert_eq!(payload["validation"][1]["backend"]["kind"], "stub");
    assert_eq!(payload["dataset_embedding_basis"], "mean-patch");
    let metric_keys = payload["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|metric| metric["key"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(metric_keys.contains(&"linear_cka"));
    assert!(metric_keys.contains(&"knn_overlap_k10"));
    assert!(metric_keys.contains(&"mean_cls_cosine"));
}

#[test]
fn similarity_html_output_includes_validation_summary() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-b").join("deep");
    fs::create_dir_all(&nested).unwrap();

    write_image(&dataset_dir.join("root.png"), 17);
    write_image(&nested.join("leaf.png"), 31);

    let output_dir = dir.path().join("similarity-html");
    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "similarity",
            "--model-a",
            "dinov2-vit-l14",
            "--model-b",
            "dinov2-vit-l14",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--metric",
            "all",
            "--format",
            "html",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let html = fs::read_to_string(output_dir.join("report.html")).unwrap();
    assert!(html.contains("Validation Summary"));
    assert!(html.contains("dinov2-vit-l14#1"));
    assert!(html.contains("dinov2-vit-l14#2"));
}

#[test]
fn similarity_json_supports_planned_stub_models_for_analysis() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-b").join("deep");
    fs::create_dir_all(&nested).unwrap();

    write_image(&dataset_dir.join("root.png"), 17);
    write_image(&nested.join("leaf.png"), 31);

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "similarity",
            "--model-a",
            "dinov2-vit-l14",
            "--model-b",
            "mae-vit-l16",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--metric",
            "all",
            "--format",
            "json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["model_a"], "dinov2-vit-l14");
    assert_eq!(payload["model_b"], "mae-vit-l16");
    assert_eq!(payload["dataset_embedding_basis"], "mean-patch");
    assert_eq!(payload["note"], "N/A (CLS tokens unavailable)");
    assert_eq!(
        payload["validation"].as_array().unwrap()[0]["status"],
        "unverified"
    );
    assert_eq!(
        payload["validation"].as_array().unwrap()[1]["status"],
        "unverified"
    );
}

#[test]
fn neighbors_json_falls_back_to_mean_patch_for_clsless_models() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let nested = dataset_dir.join("class-a");
    fs::create_dir_all(&nested).unwrap();

    let query_path = write_query_image(dir.path());
    write_image(&dataset_dir.join("root.png"), 11);
    write_image(&nested.join("leaf.png"), 29);

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "neighbors",
            query_path.to_str().unwrap(),
            "--model",
            "mae-vit-l16",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--k",
            "2",
            "--format",
            "json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["model"], "mae-vit-l16");
    assert_eq!(payload["embedding_basis"], "mean-patch");
    assert_eq!(payload["validation"]["status"], "unverified");
    assert_eq!(payload["neighbors"].as_array().unwrap().len(), 2);
}
