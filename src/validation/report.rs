use crate::models::InferenceBackend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Validated,
    Partial,
    Unverified,
    Failed,
    Stale,
}

impl ValidationStatus {
    pub fn severity(self) -> usize {
        match self {
            ValidationStatus::Validated => 0,
            ValidationStatus::Partial => 1,
            ValidationStatus::Unverified => 2,
            ValidationStatus::Stale => 3,
            ValidationStatus::Failed => 4,
        }
    }

    pub fn combine(self, other: Self) -> Self {
        if self.severity() >= other.severity() {
            self
        } else {
            other
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ValidationStatus::Validated => "validated",
            ValidationStatus::Partial => "partial",
            ValidationStatus::Unverified => "unverified",
            ValidationStatus::Failed => "failed",
            ValidationStatus::Stale => "stale",
        }
    }

    pub fn is_validated(self) -> bool {
        matches!(self, ValidationStatus::Validated)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckSummary {
    pub status: ValidationStatus,
    pub summary: String,
}

impl CheckSummary {
    pub fn new(status: ValidationStatus, summary: impl Into<String>) -> Self {
        Self {
            status,
            summary: summary.into(),
        }
    }

    pub fn validated(summary: impl Into<String>) -> Self {
        Self::new(ValidationStatus::Validated, summary)
    }

    pub fn partial(summary: impl Into<String>) -> Self {
        Self::new(ValidationStatus::Partial, summary)
    }

    pub fn unverified(summary: impl Into<String>) -> Self {
        Self::new(ValidationStatus::Unverified, summary)
    }

    pub fn failed(summary: impl Into<String>) -> Self {
        Self::new(ValidationStatus::Failed, summary)
    }

    pub fn stale(summary: impl Into<String>) -> Self {
        Self::new(ValidationStatus::Stale, summary)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorValidationSummary {
    pub name: String,
    pub role: String,
    pub status: ValidationStatus,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParitySignalDelta {
    pub name: String,
    pub observed: String,
    pub expected: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abs_diff: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureParityDrift {
    pub fixture_id: String,
    pub signal_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityValidationSummary {
    pub status: ValidationStatus,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixture_set: Option<String>,
    pub checked_signals: usize,
    pub drifted_signals: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<ParitySignalDelta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drifted_fixtures: Vec<FixtureParityDrift>,
}

impl ParityValidationSummary {
    pub fn new(status: ValidationStatus, summary: impl Into<String>) -> Self {
        Self {
            status,
            summary: summary.into(),
            artifact_id: None,
            fixture_set: None,
            checked_signals: 0,
            drifted_signals: 0,
            deltas: Vec::new(),
            drifted_fixtures: Vec::new(),
        }
    }

    pub fn with_artifact(
        mut self,
        artifact_id: Option<String>,
        fixture_set: Option<String>,
    ) -> Self {
        self.artifact_id = artifact_id;
        self.fixture_set = fixture_set;
        self
    }

    pub fn with_diagnostics(mut self, checked_signals: usize) -> Self {
        self.checked_signals = checked_signals;
        self.drifted_signals = self.deltas.len();
        self.drifted_fixtures = summarize_fixture_drifts(&self.deltas);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendValidationSummary {
    pub kind: InferenceBackend,
    pub status: ValidationStatus,
    pub summary: String,
}

impl BackendValidationSummary {
    pub fn for_backend(kind: InferenceBackend) -> Self {
        match kind {
            InferenceBackend::OnnxRuntime => Self {
                kind,
                status: ValidationStatus::Validated,
                summary: "Validation ran against live ONNX Runtime execution.".to_string(),
            },
            InferenceBackend::Stub => Self {
                kind,
                status: ValidationStatus::Unverified,
                summary: "Stub backend is active. Outputs are synthetic development scaffolding and do not establish source alignment."
                    .to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationSummary {
    pub model: String,
    pub status: ValidationStatus,
    pub evidence_timestamp: String,
    pub backend: BackendValidationSummary,
    pub preprocess: CheckSummary,
    pub tensors: Vec<TensorValidationSummary>,
    pub parity: ParityValidationSummary,
    pub caveats: Vec<String>,
    pub recommendation: String,
}

impl ModelValidationSummary {
    pub fn from_checks(
        model: impl Into<String>,
        evidence_timestamp: impl Into<String>,
        preprocess: CheckSummary,
        tensors: Vec<TensorValidationSummary>,
        parity: ParityValidationSummary,
    ) -> Self {
        let mut summary = Self {
            model: model.into(),
            evidence_timestamp: evidence_timestamp.into(),
            status: ValidationStatus::Validated,
            backend: BackendValidationSummary::for_backend(InferenceBackend::OnnxRuntime),
            preprocess,
            tensors,
            parity,
            caveats: Vec::new(),
            recommendation: String::new(),
        };
        summary.recompute_aggregate_fields();
        summary
    }

    pub fn unverified(
        model: impl Into<String>,
        evidence_timestamp: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        let reason = reason.into();
        let mut summary = Self {
            model: model.into(),
            evidence_timestamp: evidence_timestamp.into(),
            status: ValidationStatus::Unverified,
            backend: BackendValidationSummary::for_backend(InferenceBackend::OnnxRuntime),
            preprocess: CheckSummary::unverified("Preprocessing evidence was not loaded."),
            tensors: vec![TensorValidationSummary {
                name: "last_hidden_state".to_string(),
                role: "consumed tensor".to_string(),
                status: ValidationStatus::Unverified,
                summary: "Tensor semantics were not revalidated for this report surface."
                    .to_string(),
            }],
            parity: ParityValidationSummary::new(
                ValidationStatus::Unverified,
                "Reference parity evidence is unavailable for this report surface.",
            ),
            caveats: vec![reason],
            recommendation: String::new(),
        };
        summary.recompute_aggregate_fields();
        summary
    }

    pub fn failed(
        model: impl Into<String>,
        evidence_timestamp: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        let reason = reason.into();
        let mut summary = Self {
            model: model.into(),
            evidence_timestamp: evidence_timestamp.into(),
            status: ValidationStatus::Failed,
            backend: BackendValidationSummary::for_backend(InferenceBackend::OnnxRuntime),
            preprocess: CheckSummary::failed("Preprocessing validation did not complete."),
            tensors: vec![TensorValidationSummary {
                name: "last_hidden_state".to_string(),
                role: "consumed tensor".to_string(),
                status: ValidationStatus::Failed,
                summary: "Tensor semantics could not be validated because execution failed."
                    .to_string(),
            }],
            parity: ParityValidationSummary::new(
                ValidationStatus::Failed,
                "Reference parity could not be evaluated because validation execution failed.",
            ),
            caveats: vec![reason],
            recommendation: String::new(),
        };
        summary.recompute_aggregate_fields();
        summary
    }

    pub fn with_backend(mut self, backend: InferenceBackend) -> Self {
        self.backend = BackendValidationSummary::for_backend(backend);
        if matches!(backend, InferenceBackend::Stub) {
            self.degrade_for_stub_backend();
        }
        self.recompute_aggregate_fields();
        self
    }

    fn recompute_aggregate_fields(&mut self) {
        let mut status = self
            .backend
            .status
            .combine(self.preprocess.status)
            .combine(self.parity.status);
        for tensor in &self.tensors {
            status = status.combine(tensor.status);
        }

        let existing_caveats = std::mem::take(&mut self.caveats);
        let mut caveats = build_caveats(&self.preprocess, &self.tensors, &self.parity);
        extend_unique(&mut caveats, existing_caveats);

        self.status = status;
        self.caveats = caveats;
        self.recommendation = recommendation_for(status).to_string();
    }

    fn degrade_for_stub_backend(&mut self) {
        for tensor in &mut self.tensors {
            tensor.status = tensor.status.combine(ValidationStatus::Unverified);
            append_detail(
                &mut tensor.summary,
                "Stub backend synthesizes output tensors and does not execute or inspect the live ONNX graph.",
            );
        }

        self.parity.status = self.parity.status.combine(ValidationStatus::Unverified);
        append_detail(
            &mut self.parity.summary,
            "Stub backend parity only compares synthetic development outputs and is not release evidence for source alignment.",
        );
    }
}

fn summarize_fixture_drifts(deltas: &[ParitySignalDelta]) -> Vec<FixtureParityDrift> {
    let mut drifts = Vec::new();

    for delta in deltas {
        let Some((fixture_id, signal)) = split_fixture_signal(&delta.name) else {
            continue;
        };

        if let Some(existing) = drifts
            .iter_mut()
            .find(|drift: &&mut FixtureParityDrift| drift.fixture_id == fixture_id)
        {
            existing.signal_count += 1;
            if !existing
                .signals
                .iter()
                .any(|existing_signal| existing_signal == signal)
            {
                existing.signals.push(signal.to_string());
            }
            continue;
        }

        drifts.push(FixtureParityDrift {
            fixture_id: fixture_id.to_string(),
            signal_count: 1,
            signals: vec![signal.to_string()],
        });
    }

    drifts
}

fn split_fixture_signal(name: &str) -> Option<(&str, &str)> {
    let scoped = name.strip_prefix("fixtures.")?;
    scoped.split_once('.')
}

fn build_caveats(
    preprocess: &CheckSummary,
    tensors: &[TensorValidationSummary],
    parity: &ParityValidationSummary,
) -> Vec<String> {
    let mut caveats = Vec::new();
    if !preprocess.status.is_validated() {
        caveats.push(preprocess.summary.clone());
    }
    for tensor in tensors {
        if !tensor.status.is_validated() {
            caveats.push(tensor.summary.clone());
        }
    }
    if !parity.status.is_validated() {
        caveats.push(parity.summary.clone());
        let detail_count = parity.deltas.len().min(3);
        for delta in parity.deltas.iter().take(detail_count) {
            caveats.push(describe_parity_delta(delta));
        }
        if parity.deltas.len() > detail_count {
            caveats.push(format!(
                "{} additional parity deltas omitted from the summary.",
                parity.deltas.len() - detail_count
            ));
        }
    }
    caveats
}

fn append_detail(summary: &mut String, detail: &str) {
    if summary.contains(detail) {
        return;
    }

    if !summary.ends_with('.') {
        summary.push('.');
    }
    summary.push(' ');
    summary.push_str(detail);
}

fn extend_unique(target: &mut Vec<String>, items: Vec<String>) {
    for item in items {
        if !target.iter().any(|existing| existing == &item) {
            target.push(item);
        }
    }
}

fn describe_parity_delta(delta: &ParitySignalDelta) -> String {
    match (delta.abs_diff, delta.tolerance) {
        (Some(abs_diff), Some(tolerance)) => format!(
            "{} drifted (observed {}, expected {}, abs diff {:.6}, tolerance {:.6}).",
            delta.name, delta.observed, delta.expected, abs_diff, tolerance
        ),
        _ => format!(
            "{} changed (observed {}, expected {}).",
            delta.name, delta.observed, delta.expected
        ),
    }
}

pub fn recommendation_for(status: ValidationStatus) -> &'static str {
    match status {
        ValidationStatus::Validated => {
            "Safe to interpret as source-aligned for supported report features."
        }
        ValidationStatus::Partial => {
            "Use with caution; some validation evidence is intentionally partial."
        }
        ValidationStatus::Unverified => {
            "Treat outputs as unverified until validation evidence is restored."
        }
        ValidationStatus::Failed => {
            "Do not rely on source alignment until the reported validation failures are resolved."
        }
        ValidationStatus::Stale => {
            "Refresh the approved evidence before relying on this integration for review or release decisions."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_combines_by_severity() {
        assert_eq!(
            ValidationStatus::Validated.combine(ValidationStatus::Failed),
            ValidationStatus::Failed
        );
        assert_eq!(
            ValidationStatus::Stale.combine(ValidationStatus::Partial),
            ValidationStatus::Stale
        );
    }

    #[test]
    fn stub_backend_downgrades_source_alignment_claims() {
        let summary = ModelValidationSummary::from_checks(
            "dinov2-vit-l14",
            "2026-03-28T00:00:00Z",
            CheckSummary::validated("Preprocess matches contract."),
            vec![TensorValidationSummary {
                name: "last_hidden_state".into(),
                role: "patch embeddings".into(),
                status: ValidationStatus::Validated,
                summary: "Tensor semantics match the registry contract.".into(),
            }],
            ParityValidationSummary::new(
                ValidationStatus::Validated,
                "Reference parity matches approved evidence.",
            ),
        )
        .with_backend(InferenceBackend::Stub);

        assert_eq!(summary.status, ValidationStatus::Unverified);
        assert_eq!(summary.backend.kind, InferenceBackend::Stub);
        assert_eq!(summary.tensors[0].status, ValidationStatus::Unverified);
        assert_eq!(summary.parity.status, ValidationStatus::Unverified);
        assert!(summary.backend.summary.contains("Stub backend is active"));
    }

    #[test]
    fn parity_diagnostics_group_fixture_deltas() {
        let parity = ParityValidationSummary {
            status: ValidationStatus::Failed,
            summary: "Reference parity drift detected.".into(),
            artifact_id: Some("artifact".into()),
            fixture_set: Some("standard".into()),
            checked_signals: 0,
            drifted_signals: 0,
            deltas: vec![
                ParitySignalDelta {
                    name: "patch_mean".into(),
                    observed: "1.0".into(),
                    expected: "0.0".into(),
                    abs_diff: Some(1.0),
                    tolerance: Some(0.1),
                },
                ParitySignalDelta {
                    name: "fixtures.gradient-224.patch_signature[0]".into(),
                    observed: "1.0".into(),
                    expected: "0.0".into(),
                    abs_diff: Some(1.0),
                    tolerance: Some(0.1),
                },
                ParitySignalDelta {
                    name: "fixtures.gradient-224.cls_signature[1]".into(),
                    observed: "1.0".into(),
                    expected: "0.0".into(),
                    abs_diff: Some(1.0),
                    tolerance: Some(0.1),
                },
                ParitySignalDelta {
                    name: "fixtures.center-square-224.patch_std".into(),
                    observed: "1.0".into(),
                    expected: "0.0".into(),
                    abs_diff: Some(1.0),
                    tolerance: Some(0.1),
                },
            ],
            drifted_fixtures: Vec::new(),
        }
        .with_diagnostics(19);

        assert_eq!(parity.checked_signals, 19);
        assert_eq!(parity.drifted_signals, 4);
        assert_eq!(parity.drifted_fixtures.len(), 2);
        assert_eq!(parity.drifted_fixtures[0].fixture_id, "gradient-224");
        assert_eq!(parity.drifted_fixtures[0].signal_count, 2);
        assert_eq!(
            parity.drifted_fixtures[0].signals,
            vec![
                "patch_signature[0]".to_string(),
                "cls_signature[1]".to_string()
            ]
        );
    }
}
