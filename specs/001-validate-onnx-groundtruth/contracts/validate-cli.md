# Contract: `latent-inspector validate`

## Purpose

Provide a maintainer-facing CLI workflow that validates one or more model
integrations against explicit preprocessing and tensor-semantic contracts and
compares them to approved reference evidence.

## Command Shape

```text
latent-inspector validate --model <name> [--model <name> ...]
                          [--fixture-set <id-or-path>]
                          [--format terminal|json|html]
                          [--output <dir>]
                          [--refresh-goldens]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model <name>` | Yes | One or more registered model identifiers to validate. |
| `--fixture-set <id-or-path>` | No | Shared validation fixture set. Defaults to the standard checked-in set. |
| `--format <terminal\|json\|html>` | No | Output surface for the validation summary. Defaults to `terminal`. |
| `--output <dir>` | No | Output directory for JSON or HTML artifacts. |
| `--refresh-goldens` | No | Maintainer-only mode that rewrites approved golden artifacts after an intentional, reviewed change. |

## Behavioral Contract

- The command validates preprocessing rules against the declared model contract
  before reporting reference parity results.
- The command verifies every required consumed tensor and fails if a required
  tensor is missing or semantically incompatible with the profile.
- The command compares the exported model against approved reference evidence on
  the selected fixture set.
- The command emits one structured validation summary per requested model.
- When `--refresh-goldens` is not present, the command does not overwrite
  approved golden artifacts.

## Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| `0` | All requested models are validated or intentionally partial with a caller-selected non-failing policy. |
| `1` | One or more requested models fail required contract or parity checks. |
| `2` | Command usage error, missing fixtures, or unknown model identifier. |

## Output Guarantees

- `terminal`: human-readable summary table with validation status, reference
  parity result, and caveats.
- `json`: machine-readable payload matching the validation report contract.
- `html`: rendered report containing the same summary data plus explanatory
  narrative for preprocessing, tensor semantics, and trust caveats.
