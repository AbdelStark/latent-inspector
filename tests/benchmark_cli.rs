mod common;
use common::*;

use serde_json::Value;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

fn write_image(path: &Path) {
    let image = image::RgbImage::from_fn(224, 224, |x, y| {
        image::Rgb([
            (x as u8) % 255,
            (y as u8) % 255,
            ((x + y) as u8) % 255,
        ])
    });
    image.save(path).unwrap();
}

#[test]
fn benchmark_terminal_output_shows_latency_stats() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path);

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "benchmark",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--warmup",
            "1",
            "-n",
            "3",
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
    assert!(stdout.contains("Benchmark Results"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("Min:"));
    assert!(stdout.contains("Mean:"));
    assert!(stdout.contains("Median:"));
    assert!(stdout.contains("P95:"));
    assert!(stdout.contains("Throughput:"));
    assert!(stdout.contains("images/sec"));
}

#[test]
fn benchmark_json_output_has_correct_structure() {
    let dir = tempdir().unwrap();
    let img_path = dir.path().join("photo.png");
    write_image(&img_path);
    let output_path = dir.path().join("bench.json");

    let output = Command::new(bin())
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args([
            "benchmark",
            img_path.to_str().unwrap(),
            "--model",
            "dinov2-vit-l14",
            "--warmup",
            "1",
            "-n",
            "5",
            "--format",
            "json",
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

    let json: Value = read_json(&output_path);
    assert_eq!(json["model"], "dinov2-vit-l14");
    assert_eq!(json["timed_iterations"], 5);
    assert_eq!(json["warmup_iterations"], 1);
    assert!(json["total"]["count"].as_u64().unwrap() == 5);
    assert!(json["total"]["min_ms"].as_f64().unwrap() >= 0.0);
    assert!(json["total"]["mean_ms"].as_f64().unwrap() >= 0.0);
    assert!(json["total"]["median_ms"].as_f64().unwrap() >= 0.0);
    assert!(json["total"]["p95_ms"].as_f64().unwrap() >= 0.0);
    assert!(json["throughput_img_per_sec"].as_f64().unwrap() > 0.0);
}
