use latent_inspector::models::ModelSession;
use latent_inspector::validation::fixtures::load_fixture_set;
use latent_inspector::validation::parity::{compare_against_reference, summarize_outputs};
use latent_inspector::validation::ValidationStatus;

#[test]
fn approved_reference_artifacts_match_stub_outputs() {
    std::env::set_var("LATENT_INSPECTOR_MODEL_BACKEND", "stub");

    let fixture_set = load_fixture_set(None).unwrap();
    let reference = fixture_set.load_reference("dinov2-vit-l14").unwrap();
    let fixtures = fixture_set.materialize_fixtures().unwrap();
    let mut session = ModelSession::load("dinov2-vit-l14").unwrap();
    let outputs = fixtures
        .iter()
        .map(|fixture| session.infer(&fixture.image).unwrap())
        .collect::<Vec<_>>();

    let observed = summarize_outputs(&fixtures, &outputs);
    let parity = compare_against_reference(&observed, &reference);
    assert_eq!(parity.status, ValidationStatus::Validated, "dinov2-vit-l14");

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
