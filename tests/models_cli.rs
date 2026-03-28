use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latent-inspector")
}

#[test]
fn models_output_includes_evidence_and_fixture_summary() {
    let output = Command::new(bin()).args(["models"]).output().unwrap();

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Available models"));
    assert!(stdout.contains("Evidence"));
    assert!(stdout.contains("Validation fixtures:"));
    assert!(stdout.contains("Evidence summary:"));
    assert!(stdout.contains("dinov2-vit-l14"));
    assert!(stdout.contains("approved"));
    assert!(stdout.contains("mae-vit-l16"));
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
    assert!(stdout.contains("[standard @ 2026-03-27T12:00:00Z]"));
    assert!(stdout.contains("Cache: "));
    assert!(stdout.contains("Artifact: dinov2-vit-l14.onnx"));
    assert!(stdout.contains("Cache dir:"));
}
