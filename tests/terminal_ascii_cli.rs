mod common;
use common::*;

use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

fn fixture_image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("assets")
        .join("img")
        .join("samples")
        .join("cat.jpeg")
}

fn forced_ascii_command() -> Command {
    let mut command = Command::new(bin());
    command
        .env("LATENT_INSPECTOR_FORCE_ASCII", "1")
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .env("TERM", "dumb");
    command
}

#[test]
fn models_terminal_output_falls_back_to_ascii() {
    let cache_dir = tempdir().unwrap();
    let output = forced_ascii_command()
        .env("LATENT_INSPECTOR_CACHE_DIR", cache_dir.path())
        .args(["models"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_ascii());
    assert!(stdout.contains("Available models"));
    assert!(stdout.contains("Validation fixtures:"));
    assert!(stdout.contains("Artifact summary:"));
    assert!(!stdout.contains('═'));
    assert!(!stdout.contains('…'));
}

#[test]
fn inspect_terminal_output_falls_back_to_ascii() {
    let output = forced_ascii_command()
        .env("LATENT_INSPECTOR_MODEL_BACKEND", "stub")
        .args(["inspect", fixture_image().to_str().unwrap()])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_ascii());
    assert!(stdout.contains("Model: dinov2-vit-l14"));
    assert!(stdout.contains("Attention map:"));
    assert!(stdout.contains("Variance spectrum"));
    assert!(stdout.contains("Validation Summary"));
    assert!(!stdout.contains('█'));
    assert!(!stdout.contains('…'));
}
