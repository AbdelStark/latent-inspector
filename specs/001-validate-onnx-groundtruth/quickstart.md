# Quickstart: Model Validation Evidence

## Scenario 1: Validate one model integration

1. Ensure the target model is registered. In the default stub-backed test path,
   the command stays offline-safe; with real ONNX inference enabled, the export
   must be present in the local cache or downloadable through the existing
   model download flow.
2. Run the validation workflow for a single model against the default fixture
   set:

   ```bash
   cargo run -- validate --model dinov2-vit-l14
   ```

3. Confirm the command reports:
   - preprocessing validation status
   - tensor-semantic validation status
   - reference parity status
   - any caveats preventing the model from being treated as source-aligned
   - `stale` evidence when the checked-in contract or reference artifacts no
     longer match the current registry profile

## Scenario 2: Review machine-readable evidence

1. Write a JSON artifact for reviewer or CI consumption:

   ```bash
   cargo run -- validate --model clip-vit-l14 --format json --output tmp/validation
   ```

2. Inspect the generated report and confirm it contains one validation summary
   per requested model with explicit confidence, artifact identity, and caveat
   fields.

## Scenario 3: See trust summaries in normal report flows

1. Generate a comparison or inspection report after validation evidence exists:

   ```bash
   cargo run -- compare image.jpg --models dinov2-vit-l14,clip-vit-l14 --format html --output tmp/compare
   cargo run -- inspect image.jpg --model dinov2-vit-l14 --format terminal
   ```

2. Confirm each report surface includes a validation summary that explains what
   was checked, what the consumed tensor means, and whether the user should
   trust the displayed outputs.

## Scenario 4: Refresh approved golden evidence after an intentional export change

1. Review the export change and confirm that the expected behavior shift is
   intentional.
2. Regenerate approved golden artifacts explicitly:

   ```bash
   cargo run -- validate --model dinov2-vit-l14 --refresh-goldens
   ```

3. Re-run the normal validation workflow without refresh and confirm the new
   baseline passes with the refreshed evidence and produces the updated
   artifact identity in the report.
4. If the command still reports `stale`, refresh or update the checked-in
   contract/fixture metadata as well; `--refresh-goldens` only rewrites the
   reference artifacts.
