use crate::models::loader::ModelOutput;
use crate::models::registry::{ModelValidationProfile, ParityTolerances};
use crate::validation::fixtures::{
    build_reference_artifact_id, FixtureSignalSummary, MaterializedFixture, ReferenceArtifact,
    ReferenceSignals,
};
use crate::validation::report::{ParitySignalDelta, ParityValidationSummary, ValidationStatus};

const SIGNATURE_SAMPLES: usize = 8;

pub fn summarize_outputs(
    fixtures: &[MaterializedFixture],
    outputs: &[ModelOutput],
) -> ReferenceSignals {
    let first = &outputs[0];
    let fixture_summaries = outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            let fixture_id = fixtures
                .get(index)
                .map(|fixture| fixture.spec.id.clone())
                .unwrap_or_else(|| format!("fixture-{index}"));
            summarize_fixture_output(fixture_id, output)
        })
        .collect::<Vec<_>>();

    let patch_mean = average_fixture_signal(&fixture_summaries, |fixture| fixture.patch_mean);
    let patch_std = average_fixture_signal(&fixture_summaries, |fixture| fixture.patch_std);
    let patch_rms = average_fixture_signal(&fixture_summaries, |fixture| fixture.patch_rms);
    let cls_values = fixture_summaries
        .iter()
        .filter_map(|fixture| fixture.cls_l2_norm)
        .collect::<Vec<_>>();
    let cls_l2_norm = (!cls_values.is_empty())
        .then(|| cls_values.iter().copied().sum::<f32>() / cls_values.len() as f32);

    ReferenceSignals {
        tensor_name: first.tensor_metadata.output_name.clone(),
        output_shape: first.tensor_metadata.output_shape.clone(),
        cls_present: first.tensor_metadata.sequence_has_cls,
        patch_count: first.tensor_metadata.observed_patch_count,
        embedding_dim: first.tensor_metadata.embedding_dim,
        patch_mean,
        patch_std,
        patch_rms: Some(patch_rms),
        cls_l2_norm,
        fixtures: fixture_summaries,
    }
}

pub fn compare_against_reference(
    observed: &ReferenceSignals,
    reference: &ReferenceArtifact,
) -> ParityValidationSummary {
    let mut deltas = Vec::new();

    if observed.tensor_name != reference.observed.tensor_name {
        deltas.push(ParitySignalDelta {
            name: "tensor_name".to_string(),
            observed: observed.tensor_name.clone(),
            expected: reference.observed.tensor_name.clone(),
            abs_diff: None,
            tolerance: None,
        });
    }

    if observed.output_shape != reference.observed.output_shape {
        deltas.push(ParitySignalDelta {
            name: "output_shape".to_string(),
            observed: format!("{:?}", observed.output_shape),
            expected: format!("{:?}", reference.observed.output_shape),
            abs_diff: None,
            tolerance: None,
        });
    }

    if observed.cls_present != reference.observed.cls_present {
        deltas.push(ParitySignalDelta {
            name: "cls_present".to_string(),
            observed: observed.cls_present.to_string(),
            expected: reference.observed.cls_present.to_string(),
            abs_diff: None,
            tolerance: None,
        });
    }

    compare_numeric(
        "patch_count",
        observed.patch_count as f32,
        reference.observed.patch_count as f32,
        reference.tolerances.patch_count_abs as f32,
        &mut deltas,
    );
    compare_numeric(
        "embedding_dim",
        observed.embedding_dim as f32,
        reference.observed.embedding_dim as f32,
        reference.tolerances.embedding_dim_abs as f32,
        &mut deltas,
    );
    compare_numeric(
        "patch_mean",
        observed.patch_mean,
        reference.observed.patch_mean,
        reference.tolerances.patch_mean_abs,
        &mut deltas,
    );
    compare_numeric(
        "patch_std",
        observed.patch_std,
        reference.observed.patch_std,
        reference.tolerances.patch_std_abs,
        &mut deltas,
    );

    if let Some(expected_patch_rms) = reference.observed.patch_rms {
        match observed.patch_rms {
            Some(observed_patch_rms) => compare_numeric(
                "patch_rms",
                observed_patch_rms,
                expected_patch_rms,
                reference.tolerances.patch_rms_abs,
                &mut deltas,
            ),
            None => deltas.push(ParitySignalDelta {
                name: "patch_rms".to_string(),
                observed: "null".to_string(),
                expected: format!("{expected_patch_rms:.6}"),
                abs_diff: None,
                tolerance: Some(reference.tolerances.patch_rms_abs),
            }),
        }
    }

    match (observed.cls_l2_norm, reference.observed.cls_l2_norm) {
        (Some(observed), Some(expected)) => compare_numeric(
            "cls_l2_norm",
            observed,
            expected,
            reference.tolerances.cls_l2_abs,
            &mut deltas,
        ),
        (None, None) => {}
        (observed, expected) => deltas.push(ParitySignalDelta {
            name: "cls_l2_norm".to_string(),
            observed: observed
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            expected: expected
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            abs_diff: None,
            tolerance: Some(reference.tolerances.cls_l2_abs),
        }),
    }

    compare_fixture_summaries(
        &observed.fixtures,
        &reference.observed.fixtures,
        &reference.tolerances,
        &mut deltas,
    );

    let status = if deltas.is_empty() {
        ValidationStatus::Validated
    } else {
        ValidationStatus::Failed
    };
    let summary = if deltas.is_empty() {
        "Aggregate and per-fixture signals stayed within approved tolerance on the standard validation fixture set."
            .to_string()
    } else {
        format!(
            "Reference parity drift detected in {} checked signals.",
            deltas.len()
        )
    };

    ParityValidationSummary {
        status,
        summary,
        artifact_id: Some(reference.artifact_id.clone()),
        fixture_set: Some(reference.fixture_set.clone()),
        deltas,
    }
}

pub fn build_reference_artifact(
    model: &str,
    profile: &ModelValidationProfile,
    observed: ReferenceSignals,
) -> ReferenceArtifact {
    ReferenceArtifact {
        model: model.to_string(),
        fixture_set: profile.fixture_set.clone(),
        evidence_timestamp: profile.evidence_timestamp.clone(),
        artifact_id: build_reference_artifact_id(
            model,
            &profile.fixture_set,
            &profile.evidence_timestamp,
        ),
        source: profile.source.clone(),
        tolerances: profile.tolerances.clone(),
        observed,
    }
}

fn summarize_fixture_output(fixture_id: String, output: &ModelOutput) -> FixtureSignalSummary {
    let patch_values = output.patch_tokens.iter().copied().collect::<Vec<_>>();
    let (patch_mean, patch_std) = mean_and_stddev(&patch_values);
    let patch_rms = rms(&patch_values);
    let patch_signature = sample_signature(&patch_values);

    let (cls_l2_norm, cls_signature) = match output.cls_token.as_ref() {
        Some(cls) => {
            let values = cls.iter().copied().collect::<Vec<_>>();
            (Some(l2_norm(&values)), sample_signature(&values))
        }
        None => (None, Vec::new()),
    };

    FixtureSignalSummary {
        id: fixture_id,
        patch_mean,
        patch_std,
        patch_rms,
        patch_signature,
        cls_l2_norm,
        cls_signature,
    }
}

fn average_fixture_signal(
    fixtures: &[FixtureSignalSummary],
    accessor: impl Fn(&FixtureSignalSummary) -> f32,
) -> f32 {
    fixtures.iter().map(accessor).sum::<f32>() / fixtures.len().max(1) as f32
}

fn mean_and_stddev(values: &[f32]) -> (f32, f32) {
    let count = values.len().max(1) as f32;
    let mean = values.iter().copied().sum::<f32>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / count;
    (mean, variance.sqrt())
}

fn rms(values: &[f32]) -> f32 {
    let count = values.len().max(1) as f32;
    (values.iter().map(|value| value * value).sum::<f32>() / count).sqrt()
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn sample_signature(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let sample_count = SIGNATURE_SAMPLES.min(values.len());
    if sample_count == 1 {
        return vec![values[0]];
    }

    let last_index = values.len() - 1;
    (0..sample_count)
        .map(|index| {
            let position = index * last_index / (sample_count - 1);
            values[position]
        })
        .collect()
}

fn compare_fixture_summaries(
    observed: &[FixtureSignalSummary],
    reference: &[FixtureSignalSummary],
    tolerances: &ParityTolerances,
    deltas: &mut Vec<ParitySignalDelta>,
) {
    if reference.is_empty() {
        return;
    }

    if observed.len() != reference.len() {
        deltas.push(ParitySignalDelta {
            name: "fixture_count".to_string(),
            observed: observed.len().to_string(),
            expected: reference.len().to_string(),
            abs_diff: None,
            tolerance: None,
        });
    }

    for (observed_fixture, reference_fixture) in observed.iter().zip(reference.iter()) {
        let fixture_name = format!("fixtures.{}", reference_fixture.id);

        if observed_fixture.id != reference_fixture.id {
            deltas.push(ParitySignalDelta {
                name: format!("{fixture_name}.id"),
                observed: observed_fixture.id.clone(),
                expected: reference_fixture.id.clone(),
                abs_diff: None,
                tolerance: None,
            });
        }

        compare_numeric(
            &format!("{fixture_name}.patch_mean"),
            observed_fixture.patch_mean,
            reference_fixture.patch_mean,
            tolerances.patch_mean_abs,
            deltas,
        );
        compare_numeric(
            &format!("{fixture_name}.patch_std"),
            observed_fixture.patch_std,
            reference_fixture.patch_std,
            tolerances.patch_std_abs,
            deltas,
        );
        compare_numeric(
            &format!("{fixture_name}.patch_rms"),
            observed_fixture.patch_rms,
            reference_fixture.patch_rms,
            tolerances.patch_rms_abs,
            deltas,
        );
        compare_vector(
            &format!("{fixture_name}.patch_signature"),
            &observed_fixture.patch_signature,
            &reference_fixture.patch_signature,
            tolerances.patch_signature_abs,
            deltas,
        );

        match (observed_fixture.cls_l2_norm, reference_fixture.cls_l2_norm) {
            (Some(observed_cls), Some(expected_cls)) => compare_numeric(
                &format!("{fixture_name}.cls_l2_norm"),
                observed_cls,
                expected_cls,
                tolerances.cls_l2_abs,
                deltas,
            ),
            (None, None) => {}
            (observed_cls, expected_cls) => deltas.push(ParitySignalDelta {
                name: format!("{fixture_name}.cls_l2_norm"),
                observed: observed_cls
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "null".to_string()),
                expected: expected_cls
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "null".to_string()),
                abs_diff: None,
                tolerance: Some(tolerances.cls_l2_abs),
            }),
        }

        compare_vector(
            &format!("{fixture_name}.cls_signature"),
            &observed_fixture.cls_signature,
            &reference_fixture.cls_signature,
            tolerances.cls_signature_abs,
            deltas,
        );
    }
}

fn compare_vector(
    name: &str,
    observed: &[f32],
    expected: &[f32],
    tolerance: f32,
    deltas: &mut Vec<ParitySignalDelta>,
) {
    if observed.len() != expected.len() {
        deltas.push(ParitySignalDelta {
            name: format!("{name}.len"),
            observed: observed.len().to_string(),
            expected: expected.len().to_string(),
            abs_diff: None,
            tolerance: None,
        });
    }

    for (index, (observed_value, expected_value)) in
        observed.iter().zip(expected.iter()).enumerate()
    {
        compare_numeric(
            &format!("{name}[{index}]"),
            *observed_value,
            *expected_value,
            tolerance,
            deltas,
        );
    }
}

fn compare_numeric(
    name: &str,
    observed: f32,
    expected: f32,
    tolerance: f32,
    deltas: &mut Vec<ParitySignalDelta>,
) {
    let abs_diff = (observed - expected).abs();
    if abs_diff > tolerance {
        deltas.push(ParitySignalDelta {
            name: name.to_string(),
            observed: format!("{observed:.6}"),
            expected: format!("{expected:.6}"),
            abs_diff: Some(abs_diff),
            tolerance: Some(tolerance),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::{ModelInfo, SSLMethod};
    use crate::models::OutputTensorMetadata;
    use crate::validation::fixtures::{FixturePattern, ValidationFixtureSpec};
    use image::DynamicImage;
    use ndarray::{Array1, Array2};

    fn test_output(patch_scale: f32, cls_scale: f32) -> ModelOutput {
        ModelOutput {
            cls_token: Some(Array1::from_shape_fn(6, |idx| {
                cls_scale + idx as f32 * 0.05
            })),
            patch_tokens: Array2::from_shape_fn((4, 6), |(row, col)| {
                patch_scale + row as f32 * 0.2 + col as f32 * 0.03
            }),
            attention_weights: None,
            model_info: ModelInfo {
                name: "dinov2-vit-l14".to_string(),
                architecture: "ViT-L/14".to_string(),
                patch_size: 14,
                embed_dim: 6,
                num_layers: 24,
                num_heads: 16,
                method: SSLMethod::DINO,
                input_size: 224,
                params_m: 304,
            },
            tensor_metadata: OutputTensorMetadata {
                input_name: "pixel_values".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_name: "last_hidden_state".to_string(),
                output_shape: vec![1, 5, 6],
                sequence_has_cls: true,
                observed_patch_count: 4,
                embedding_dim: 6,
            },
        }
    }

    fn test_fixture(id: &str) -> MaterializedFixture {
        MaterializedFixture {
            spec: ValidationFixtureSpec {
                id: id.to_string(),
                description: "test".to_string(),
                width: 224,
                height: 224,
                pattern: FixturePattern::Gradient,
            },
            image: DynamicImage::new_rgb8(224, 224),
        }
    }

    #[test]
    fn summarize_outputs_tracks_fixture_level_signals() {
        let fixtures = vec![
            test_fixture("gradient-224"),
            test_fixture("center-square-224"),
        ];
        let outputs = vec![test_output(0.1, 0.3), test_output(0.6, 0.8)];

        let summary = summarize_outputs(&fixtures, &outputs);

        assert_eq!(summary.fixtures.len(), 2);
        assert_eq!(summary.fixtures[0].id, "gradient-224");
        assert_eq!(summary.fixtures[1].id, "center-square-224");
        assert_eq!(summary.fixtures[0].patch_signature.len(), SIGNATURE_SAMPLES);
        assert!(summary.patch_rms.unwrap_or_default() > 0.0);
    }

    #[test]
    fn compare_against_reference_flags_fixture_signature_drift() {
        let tolerances = ParityTolerances {
            patch_count_abs: 0,
            embedding_dim_abs: 0,
            patch_mean_abs: 1e-3,
            patch_std_abs: 1e-3,
            cls_l2_abs: 1e-3,
            patch_rms_abs: 1e-3,
            patch_signature_abs: 1e-3,
            cls_signature_abs: 1e-3,
        };
        let observed = summarize_outputs(&[test_fixture("gradient-224")], &[test_output(0.1, 0.3)]);
        let mut reference = build_reference_artifact(
            "dinov2-vit-l14",
            &ModelValidationProfile {
                source: "facebookresearch/dinov2".to_string(),
                evidence_timestamp: "2026-03-27T12:00:00Z".to_string(),
                fixture_set: "standard".to_string(),
                preprocess: crate::models::registry::PreprocessContract {
                    input_size: 224,
                    resize_filter: "lanczos3".to_string(),
                    color_space: "rgb".to_string(),
                    layout: "nchw".to_string(),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                tensor: crate::models::registry::TensorContract {
                    name: "last_hidden_state".to_string(),
                    role: crate::models::registry::TensorRole::PatchAndClsSequence,
                    cls_expected: true,
                    batch_size: 1,
                    patch_count: 4,
                    embedding_dim: 6,
                },
                tolerances: tolerances.clone(),
            },
            observed.clone(),
        );
        reference.observed.fixtures[0].patch_signature[0] += 1.0;

        let parity = compare_against_reference(&observed, &reference);

        assert_eq!(parity.status, ValidationStatus::Failed);
        assert!(parity
            .deltas
            .iter()
            .any(|delta| delta.name == "fixtures.gradient-224.patch_signature[0]"));
    }
}
