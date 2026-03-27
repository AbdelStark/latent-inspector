use crate::models::loader::ModelOutput;
use crate::models::registry::ModelValidationProfile;
use crate::validation::fixtures::{
    build_reference_artifact_id, ReferenceArtifact, ReferenceSignals,
};
use crate::validation::report::{ParitySignalDelta, ParityValidationSummary, ValidationStatus};

pub fn summarize_outputs(
    _model: &str,
    _fixture_set: &str,
    outputs: &[ModelOutput],
) -> ReferenceSignals {
    let first = &outputs[0];
    let patch_mean = outputs
        .iter()
        .map(|output| {
            let sum = output
                .patch_tokens
                .iter()
                .fold(0.0_f64, |acc, value| acc + f64::from(*value));
            (sum / output.patch_tokens.len() as f64) as f32
        })
        .sum::<f32>()
        / outputs.len() as f32;
    let patch_std = outputs
        .iter()
        .map(|output| {
            let mean = output
                .patch_tokens
                .iter()
                .fold(0.0_f64, |acc, value| acc + f64::from(*value))
                / output.patch_tokens.len() as f64;
            let variance = output
                .patch_tokens
                .iter()
                .map(|value| {
                    let centered = f64::from(*value) - mean;
                    centered * centered
                })
                .sum::<f64>()
                / output.patch_tokens.len() as f64;
            variance.sqrt() as f32
        })
        .sum::<f32>()
        / outputs.len() as f32;
    let cls_l2_norm = {
        let values = outputs
            .iter()
            .filter_map(|output| output.cls_token.as_ref())
            .map(|cls| {
                cls.iter()
                    .fold(0.0_f64, |acc, value| {
                        acc + f64::from(*value) * f64::from(*value)
                    })
                    .sqrt() as f32
            })
            .collect::<Vec<_>>();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().copied().sum::<f32>() / values.len() as f32)
        }
    };

    ReferenceSignals {
        tensor_name: first.tensor_metadata.output_name.clone(),
        output_shape: first.tensor_metadata.output_shape.clone(),
        cls_present: first.tensor_metadata.sequence_has_cls,
        patch_count: first.tensor_metadata.observed_patch_count,
        embedding_dim: first.tensor_metadata.embedding_dim,
        patch_mean,
        patch_std,
        cls_l2_norm,
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

    let status = if deltas.is_empty() {
        ValidationStatus::Validated
    } else {
        ValidationStatus::Failed
    };
    let summary = if deltas.is_empty() {
        "Compared signals stayed within approved tolerance on the standard validation fixture set."
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
