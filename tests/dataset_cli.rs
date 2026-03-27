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
    assert!(stdout.contains("broken.png"));
}
