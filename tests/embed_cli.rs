mod common;
use common::*;

use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

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

#[test]
fn embed_single_image_global_level_outputs_jsonl() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path, 42);
    let output_path = dir.path().join("embeddings.jsonl");

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "embed",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--level",
            "global",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "Expected 1 JSONL line for 1 image");

    let record: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(record["model"], "dinov2-vit-l14");
    assert!(!record["embedding"].as_array().unwrap().is_empty());
    assert!(record["embed_dim"].as_u64().unwrap() > 0);
    assert!(record["image"].as_str().unwrap().contains("photo.png"));
}

#[test]
fn embed_directory_produces_one_line_per_image() {
    let dir = tempdir().unwrap();
    let img_dir = dir.path().join("images");
    fs::create_dir_all(&img_dir).unwrap();
    write_image(&img_dir.join("a.png"), 1);
    write_image(&img_dir.join("b.png"), 2);
    write_image(&img_dir.join("c.jpg"), 3);
    let output_path = dir.path().join("embeddings.jsonl");

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "embed",
            img_dir.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--level",
            "global",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 3, "Expected 3 JSONL lines for 3 images");

    // Each line should be valid JSON
    for line in &lines {
        let record: Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["model"], "dinov2-vit-l14");
    }
}

#[test]
fn embed_patches_level_includes_patch_data() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path, 42);
    let output_path = dir.path().join("patches.jsonl");

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "embed",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--level",
            "patches",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = fs::read_to_string(&output_path).unwrap();
    let record: Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    assert!(record["n_patches"].as_u64().unwrap() > 0);
    assert!(!record["patches"].as_array().unwrap().is_empty());
    // patches should be n_patches * embed_dim elements
    let n_patches = record["n_patches"].as_u64().unwrap() as usize;
    let embed_dim = record["embed_dim"].as_u64().unwrap() as usize;
    assert_eq!(
        record["patches"].as_array().unwrap().len(),
        n_patches * embed_dim
    );
}

#[test]
fn embed_full_level_includes_both() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path, 42);
    let output_path = dir.path().join("full.jsonl");

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "embed",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--level",
            "full",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = fs::read_to_string(&output_path).unwrap();
    let record: Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    // Full level has both embedding and patches
    assert!(!record["embedding"].as_array().unwrap().is_empty());
    assert!(!record["patches"].as_array().unwrap().is_empty());
    assert!(record["basis"].as_str().is_some());
}

#[test]
fn embed_to_stdout_when_no_output_specified() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path, 42);

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "embed",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
        ])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let record: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(record["model"], "dinov2-vit-l14");
}
