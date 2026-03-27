use latent_inspector::validation::fixtures::load_fixture_set;
use latent_inspector::validation::parity::compare_against_reference;
use latent_inspector::validation::{validate_model, ValidationStatus};

#[test]
fn approved_reference_artifacts_match_stub_outputs() {
    for model in [
        "dinov2-vit-l14",
        "mae-vit-l16",
        "clip-vit-l14",
        "ijepa-vit-h14",
        "siglip-so400m",
    ] {
        let summary = validate_model(model, None, false).unwrap();
        assert_eq!(summary.status, ValidationStatus::Validated, "{model}");
        assert_eq!(
            summary.parity.status,
            ValidationStatus::Validated,
            "{model}"
        );
    }
}

#[test]
fn parity_comparison_flags_golden_regressions() {
    let fixture_set = load_fixture_set(None).unwrap();
    let reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
    let observed = reference.observed.clone();

    let mut drifted = reference.clone();
    drifted.observed.patch_mean = 2.0;

    let parity = compare_against_reference(&observed, &drifted);
    assert_eq!(parity.status, ValidationStatus::Failed);
    assert!(parity.deltas.iter().any(|delta| delta.name == "patch_mean"));
}
