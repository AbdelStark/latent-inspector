# Contract: Validation Report Payload

## Purpose

Define the structured validation summary that report renderers consume and that
JSON output exposes directly.

## Payload Shape

```json
{
  "model": "dinov2-vit-l14",
  "status": "validated",
  "evidence_timestamp": "2026-03-27T12:00:00Z",
  "preprocess": {
    "status": "validated",
    "summary": "Input size, normalization, and channel order match the approved source-model contract."
  },
  "tensors": [
    {
      "name": "last_hidden_state",
      "role": "patch+cls sequence",
      "status": "validated",
      "summary": "Sequence layout matches the expected CLS-plus-patch contract."
    }
  ],
  "parity": {
    "status": "validated",
    "summary": "Compared signals stayed within approved tolerance on the standard validation fixture set."
  },
  "caveats": [],
  "recommendation": "Safe to interpret as source-aligned for supported report features."
}
```

## Field Semantics

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `model` | string | Yes | Registered model identifier. |
| `status` | enum | Yes | Overall validation state: `validated`, `partial`, `unverified`, `failed`, or `stale`. |
| `evidence_timestamp` | string | Yes | Timestamp or version marker of the evidence cited by the summary. |
| `preprocess` | object | Yes | Validation status and explanation for input preparation. |
| `tensors` | array | Yes | One entry per consumed tensor contract included in the report. |
| `parity` | object | Yes | Validation status and explanation for reference comparison. |
| `caveats` | array of strings | Yes | Ordered warnings and limitations. Must be non-empty for non-validated states. |
| `recommendation` | string | Yes | Plain-language trust guidance for the user or reviewer. |

## Renderer Guarantees

- Terminal output must present `status`, parity outcome, and caveats in a short
  scan-friendly summary.
- JSON output must preserve all fields without dropping low-confidence states.
- HTML output must render the same payload with additional explanatory text, not
  a conflicting interpretation.
- Existing compare and inspect reports may omit tensor-by-tensor detail only if
  the aggregated explanation still reflects the same underlying payload.
