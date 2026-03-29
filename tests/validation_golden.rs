use latent_inspector::models::InferenceBackend;
use latent_inspector::validation::fixtures::load_fixture_set;
use latent_inspector::validation::parity::compare_against_reference;
use latent_inspector::validation::ValidationStatus;

#[test]
fn approved_dinov2_reference_artifact_records_live_backend() {
    let fixture_set = load_fixture_set(None).unwrap();
    let reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
    assert_eq!(reference.backend, InferenceBackend::OnnxRuntime);
    assert_eq!(
        compare_against_reference(&reference.observed, &reference).status,
        ValidationStatus::Validated
    );
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
