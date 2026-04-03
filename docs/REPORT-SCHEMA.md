# Report Schema

## Scope

As of 2026-04-03, the non-terminal CLI outputs for the ready-model surface are a compatibility contract for automation and sharing.

Covered commands:

- `compare`
- `inspect`
- `neighbors`
- `similarity`
- `profile`
- `drift`
- `models`
- `validate`

This contract covers:

- Primary artifact filenames
- The presence and spelling of current top-level JSON keys
- `artifacts.json` manifest structure
- Current enum spellings already emitted by JSON reports and manifests

Allowed additive changes:

- New optional fields
- New artifact entries in a bundle manifest
- New metric entries inside arrays that already use stable `key` fields
- Additional validation caveats or recommendations

Breaking changes require a changelog entry and matching updates to this document, `README.md`, and `AGENTS.md`.

Out of scope:

- Terminal text formatting
- TUI layout and interactions
- Internal Rust module APIs
- Exact numeric values in reports
- Planned-model live-runtime behavior before those models become ready

## Primary Artifacts

| Command surface | JSON | HTML | PNG |
|---|---|---|---|
| `compare` | `compare.json` | `report.html` plus `compare.json` | No single primary PNG; see `artifacts.json` |
| `inspect` | `inspect.json` | `report.html` plus `inspect.json` | No single primary PNG; see `artifacts.json` |
| `neighbors` | `neighbors.json` | `report.html` plus `neighbors.json` | `neighbors.png` |
| `similarity` | `similarity.json` | `report.html` plus `similarity.json` | `similarity.png` |
| `profile` | `profile.json` | `report.html` plus `profile.json` | `profile.png` |
| `drift` | `drift.json` | `report.html` plus `drift.json` | `consecutive_cka.png` |
| `models` catalog | `models.json` | `models.html` plus `models.json` | Not supported |
| `models --download <model>` | `download.json` | `download.html` plus `download.json` | Not supported |
| `validate` | `validation.json` | `validation.html` plus `validation.json` | Not supported |

## Bundle Manifest Contract

Every non-terminal output written with `--output <dir>` also writes `artifacts.json`.

Top-level manifest keys:

- `command`
- `format`
- `primary_artifact`
- `context`
- `summary`
- `artifacts`
- `validation`
- `validation_summary`

Artifact entries in `artifacts`:

- `path`: always relative to the output directory
- `kind`: `json`, `html`, or `png`
- `label`: human-readable description
- `byte_size`: omitted only for excluded bundle-display entries
- `sha256`: omitted only for excluded bundle-display entries

Validation entries in `validation`:

- `model`
- `status`
- `recommendation`

Validation summary fields:

- `overall_status`
- `validated`
- `partial`
- `unverified`
- `stale`
- `failed`

## Primary JSON Shapes

### `compare.json`

Top-level keys:

- `image`
- `requested_models`
- `metrics`
- `comparisons`
- `overview`
- `validation`

### `inspect.json`

Top-level keys:

- `image`
- `model`
- `metrics`
- `validation`
- `variance_spectrum`
- `attention`

`attention` is nullable and omitted when the model/report surface has no attention map.

### `neighbors.json`

Top-level keys:

- `query_image`
- `dataset`
- `model`
- `embedding_basis`
- `requested_k`
- `dataset_summary`
- `neighbors`
- `validation`

### `similarity.json`

Top-level keys:

- `model_a`
- `model_b`
- `dataset`
- `dataset_embedding_basis`
- `requested_metric`
- `sample_count`
- `dataset_summary`
- `metrics`
- `note`
- `validation`

`note` is nullable and omitted when there is no explanatory caveat.

### `profile.json`

Top-level keys:

- `model`
- `dataset`
- `embedding_basis`
- `sample_count`
- `embed_dim`
- `dataset_summary`
- `space_metrics`
- `aggregate_metrics`
- `per_image_metrics`
- `validation`

### `drift.json`

Top-level keys:

- `model`
- `checkpoints`
- `dataset`
- `dataset_embedding_basis`
- `checkpoint_names`
- `dataset_summary`
- `drift`
- `mean_consecutive_cka`
- `largest_shift`
- `validation`

`dataset_summary`, `mean_consecutive_cka`, and `largest_shift` may be omitted when the report surface has no corresponding value.

### `models.json`

Top-level keys:

- `fixture_set`
- `evidence_timestamp`
- `fixture_error`
- `summary`
- `entries`

### `download.json`

Top-level keys:

- `model`
- `action`
- `summary`
- `entry`
- `artifact_changes`

### `validation.json`

This file is a JSON array of per-model validation summaries.

Each entry contains:

- `model`
- `status`
- `evidence_timestamp`
- `backend`
- `preprocess`
- `tensors`
- `parity`
- `caveats`
- `recommendation`

## HTML Contract

HTML bundles are shareable report surfaces, not the machine contract. The stable automation surface for HTML output is:

- The primary HTML filename listed above
- The presence of a companion JSON report in the same output directory
- The presence of `artifacts.json`

Automation should consume the companion JSON and manifest rather than scraping HTML.
