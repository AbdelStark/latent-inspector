use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

fn write_dataset_image(dir: &Path, name: &str, offset: u8) -> PathBuf {
    let path = dir.join(format!("{name}.png"));
    let image = image::RgbImage::from_fn(224, 224, |x, y| {
        image::Rgb([
            ((x as u8).wrapping_add(offset)) % 255,
            ((y as u8).wrapping_add(offset / 2)) % 255,
            ((x as u8).wrapping_add(y as u8).wrapping_add(offset)) % 255,
        ])
    });
    image.save(&path).unwrap();
    path
}

fn extract_cka_scores(stdout: &str) -> Vec<f32> {
    stdout
        .lines()
        .filter_map(|line| line.split("CKA=").nth(1))
        .filter_map(|value| value.trim().parse::<f32>().ok())
        .collect()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn drift_reports_consecutive_checkpoint_scores() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let checkpoints_dir = dir.path().join("checkpoints");
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(&checkpoints_dir).unwrap();

    write_dataset_image(&dataset_dir, "sample-a", 5);
    write_dataset_image(&dataset_dir, "sample-b", 29);
    let nested = dataset_dir.join("nested");
    fs::create_dir_all(&nested).unwrap();
    write_dataset_image(&nested, "sample-c", 73);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    fs::write(checkpoints_dir.join("step-1.onnx"), b"checkpoint-a").unwrap();
    fs::write(checkpoints_dir.join("step-2.onnx"), b"checkpoint-b").unwrap();
    fs::write(checkpoints_dir.join("step-10.onnx"), b"checkpoint-c").unwrap();

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "drift",
            "--model",
            "dinov2-vit-l14",
            "--checkpoints",
            checkpoints_dir.to_str().unwrap(),
            "--dataset",
            dataset_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Representation Drift"));
    assert!(stdout.contains("Dataset Summary"));
    assert!(stdout.contains("Validation Summary"));
    assert!(stdout.contains("broken.png"));
    assert!(stdout.contains("step-1"));
    assert!(stdout.contains("step-2"));
    assert!(stdout.contains("step-10"));
    assert!(stdout.contains("Largest shift:"));

    let step_1_index = stdout.find("step-1").unwrap();
    let step_2_index = stdout.find("step-2").unwrap();
    let step_10_index = stdout.find("step-10").unwrap();
    assert!(step_1_index < step_2_index);
    assert!(step_2_index < step_10_index);

    let scores = extract_cka_scores(&stdout);
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|score| *score < 0.9999));
}

#[test]
fn drift_json_and_png_outputs_are_written() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let checkpoints_dir = dir.path().join("checkpoints");
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(&checkpoints_dir).unwrap();

    write_dataset_image(&dataset_dir, "sample-a", 5);
    write_dataset_image(&dataset_dir, "sample-b", 29);
    fs::write(dataset_dir.join("broken.png"), b"not an image").unwrap();

    fs::write(checkpoints_dir.join("step-1.onnx"), b"checkpoint-a").unwrap();
    fs::write(checkpoints_dir.join("step-2.onnx"), b"checkpoint-b").unwrap();
    fs::write(checkpoints_dir.join("step-10.onnx"), b"checkpoint-c").unwrap();

    let json_output_dir = dir.path().join("drift-json");
    let json_output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "drift",
            "--model",
            "dinov2-vit-l14",
            "--checkpoints",
            checkpoints_dir.to_str().unwrap(),
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--format",
            "json",
            "--output",
            json_output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(json_output.status.code(), Some(0));

    let payload = read_json(&json_output_dir.join("drift.json"));
    assert_eq!(payload["model"], "dinov2-vit-l14");
    assert_eq!(payload["dataset_embedding_basis"], "mean-patch");
    assert_eq!(
        payload["checkpoint_names"],
        Value::Array(vec![
            Value::from("step-1"),
            Value::from("step-2"),
            Value::from("step-10"),
        ])
    );
    assert_eq!(payload["drift"].as_array().unwrap().len(), 2);
    assert_eq!(payload["dataset_summary"]["skipped"], 1);
    assert_eq!(payload["largest_shift"]["from_checkpoint"], "step-2");
    assert_eq!(payload["largest_shift"]["to_checkpoint"], "step-10");
    assert_eq!(payload["validation"].as_array().unwrap().len(), 3);
    assert_eq!(payload["validation"][0]["model"], "step-1");
    assert!(payload["validation"][0]["caveats"]
        .as_array()
        .unwrap()
        .iter()
        .any(|caveat| caveat
            .as_str()
            .unwrap()
            .contains("approved release artifact")));
    assert!(
        payload["drift"][0]["linear_cka"].as_f64().unwrap()
            > payload["drift"][1]["linear_cka"].as_f64().unwrap()
    );

    let png_output_dir = dir.path().join("drift-png");
    let png_output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "drift",
            "--model",
            "dinov2-vit-l14",
            "--checkpoints",
            checkpoints_dir.to_str().unwrap(),
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--format",
            "png",
            "--output",
            png_output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(png_output.status.code(), Some(0));
    assert!(png_output_dir.join("consecutive_cka.png").exists());
}

#[test]
fn drift_html_output_includes_validation_summary() {
    let dir = tempdir().unwrap();
    let dataset_dir = dir.path().join("dataset");
    let checkpoints_dir = dir.path().join("checkpoints");
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(&checkpoints_dir).unwrap();

    write_dataset_image(&dataset_dir, "sample-a", 5);
    write_dataset_image(&dataset_dir, "sample-b", 29);
    fs::write(checkpoints_dir.join("step-1.onnx"), b"checkpoint-a").unwrap();
    fs::write(checkpoints_dir.join("step-2.onnx"), b"checkpoint-b").unwrap();

    let output_dir = dir.path().join("drift-html");
    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "drift",
            "--model",
            "dinov2-vit-l14",
            "--checkpoints",
            checkpoints_dir.to_str().unwrap(),
            "--dataset",
            dataset_dir.to_str().unwrap(),
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
    assert!(html.contains("step-1"));
    assert!(html.contains("step-2"));
}
