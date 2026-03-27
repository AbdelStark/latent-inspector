use crate::errors::ValidationError;
use crate::models::loader::ModelOutput;
use crate::models::preprocess::{preprocess, PreprocessConfig};
use crate::models::registry::RegistryEntry;
use crate::validation::fixtures::{ContractArtifact, LoadedFixtureSet};
use crate::validation::report::{CheckSummary, TensorValidationSummary, ValidationStatus};

pub fn evaluate_preprocess_contract(
    entry: &RegistryEntry,
    contract: &ContractArtifact,
    fixture_set: &LoadedFixtureSet,
) -> Result<CheckSummary, ValidationError> {
    let expected = &contract.profile.preprocess;
    let mut mismatches = Vec::new();

    if entry.validation.preprocess.input_size != expected.input_size {
        mismatches.push(format!(
            "input size mismatch: registry={} contract={}",
            entry.validation.preprocess.input_size, expected.input_size
        ));
    }

    if entry.validation.preprocess.resize_filter != expected.resize_filter {
        mismatches.push(format!(
            "resize filter mismatch: registry={} contract={}",
            entry.validation.preprocess.resize_filter, expected.resize_filter
        ));
    }

    if entry.validation.preprocess.color_space != expected.color_space {
        mismatches.push(format!(
            "color space mismatch: registry={} contract={}",
            entry.validation.preprocess.color_space, expected.color_space
        ));
    }

    if entry.validation.preprocess.layout != expected.layout {
        mismatches.push(format!(
            "tensor layout mismatch: registry={} contract={}",
            entry.validation.preprocess.layout, expected.layout
        ));
    }

    if entry.validation.preprocess.mean != expected.mean {
        mismatches.push("normalization mean does not match the approved contract".to_string());
    }

    if entry.validation.preprocess.std != expected.std {
        mismatches.push("normalization std does not match the approved contract".to_string());
    }

    let fixture = fixture_set
        .materialize_fixtures()?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ValidationError::MissingFixtures("No validation fixtures available".to_string())
        })?;
    let cfg = PreprocessConfig::new(entry.info.input_size, entry.norm_mean, entry.norm_std);
    let tensor =
        preprocess(&fixture.image, &cfg).map_err(|err| ValidationError::ContractMismatch {
            model: entry.info.name.clone(),
            reason: err.to_string(),
        })?;
    let expected_shape = [
        1,
        3,
        expected.input_size as usize,
        expected.input_size as usize,
    ];
    if tensor.shape() != expected_shape.as_slice() {
        mismatches.push(format!(
            "preprocessed tensor shape mismatch: observed={:?} expected={:?}",
            tensor.shape(),
            expected_shape
        ));
    }

    let pixel = fixture.image.to_rgb8().get_pixel(0, 0).0;
    for (channel, (mean, std)) in expected.mean.iter().zip(expected.std.iter()).enumerate() {
        let raw = pixel[channel] as f32 / 255.0;
        let expected_value = (raw - mean) / std;
        let observed = tensor[[0, channel, 0, 0]];
        if (observed - expected_value).abs() > 1e-5 {
            mismatches.push(format!(
                "channel {} normalization drift: observed={observed:.6} expected={expected_value:.6}",
                channel
            ));
        }
    }

    if mismatches.is_empty() {
        Ok(CheckSummary::validated(
            "Input size, normalization, resize filter, and channel layout match the approved source-model contract.",
        ))
    } else {
        Ok(CheckSummary::failed(format!(
            "Preprocessing contract mismatches detected: {}.",
            mismatches.join("; ")
        )))
    }
}

pub fn evaluate_tensor_semantics(
    entry: &RegistryEntry,
    contract: &ContractArtifact,
    output: &ModelOutput,
) -> Vec<TensorValidationSummary> {
    let expected = &contract.profile.tensor;
    let observed_shape = output.tensor_metadata.output_shape.clone();
    let expected_shape = expected.expected_shape();
    let mut mismatches = Vec::new();

    if output.tensor_metadata.output_name != expected.name {
        mismatches.push(format!(
            "output name mismatch: observed={} expected={}",
            output.tensor_metadata.output_name, expected.name
        ));
    }

    if observed_shape != expected_shape {
        mismatches.push(format!(
            "output shape mismatch: observed={observed_shape:?} expected={expected_shape:?}"
        ));
    }

    if output.tensor_metadata.sequence_has_cls != expected.cls_expected {
        mismatches.push(format!(
            "CLS semantics mismatch: observed={} expected={}",
            output.tensor_metadata.sequence_has_cls, expected.cls_expected
        ));
    }

    if output.tensor_metadata.observed_patch_count != expected.patch_count {
        mismatches.push(format!(
            "patch count mismatch: observed={} expected={}",
            output.tensor_metadata.observed_patch_count, expected.patch_count
        ));
    }

    if output.tensor_metadata.embedding_dim != expected.embedding_dim {
        mismatches.push(format!(
            "embedding width mismatch: observed={} expected={}",
            output.tensor_metadata.embedding_dim, expected.embedding_dim
        ));
    }

    if output.cls_token.is_some() != expected.cls_expected {
        mismatches.push(format!(
            "CLS token presence mismatch: observed={} expected={}",
            output.cls_token.is_some(),
            expected.cls_expected
        ));
    }

    let status = if mismatches.is_empty() {
        ValidationStatus::Validated
    } else {
        ValidationStatus::Failed
    };

    let summary = if mismatches.is_empty() {
        format!(
            "{} matches the expected {} contract for {} patch tokens.",
            expected.name, expected.role, expected.patch_count
        )
    } else {
        format!(
            "Tensor semantic mismatches detected for {}: {}.",
            entry.info.name,
            mismatches.join("; ")
        )
    };

    vec![TensorValidationSummary {
        name: expected.name.clone(),
        role: expected.role.to_string(),
        status,
        summary,
    }]
}
