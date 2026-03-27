use latent_inspector::validation::fixtures::load_fixture_set;
use latent_inspector::validation::parity::compare_against_reference;
use latent_inspector::validation::{validate_model, ValidationStatus};

#[test]
fn approved_reference_artifacts_match_stub_outputs() {
    std::env::set_var("LATENT_INSPECTOR_MODEL_BACKEND", "stub");

    let summary = validate_model("dinov2-vit-l14", None, false).unwrap();
    assert_eq!(
        summary.status,
        ValidationStatus::Validated,
        "dinov2-vit-l14"
    );
    assert_eq!(
        summary.parity.status,
        ValidationStatus::Validated,
        "dinov2-vit-l14"
    );

    std::env::remove_var("LATENT_INSPECTOR_MODEL_BACKEND");
}

#[test]
fn parity_comparison_flags_golden_regressions() {
    let fixture_set = load_fixture_set(None).unwrap();
    let reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
    let observed = reference.observed.clone();

    let mut drifted = reference.clone();
    drifted.observed.fixtures[0].patch_signature[0] += 2.0;

    let parity = compare_against_reference(&observed, &drifted);
    assert_eq!(parity.status, ValidationStatus::Failed);
    assert!(parity
        .deltas
        .iter()
        .any(|delta| delta.name == "fixtures.gradient-224.patch_signature[0]"));
}
