---
name: testing
description: Testing strategy for latent-inspector — unit tests with cargo test, integration tests, floating-point assertions with approx, property-based testing for numerical code, and test fixtures for ONNX models. Activate when writing tests, debugging test failures, or setting up test infrastructure.
prerequisites: cargo test, approx crate, tempfile crate
---

# Testing

<purpose>
Covers the testing approach: inline unit tests per module, integration tests for CLI commands, floating-point comparison strategies, test fixtures for model output, and property-based testing for numerical correctness.
</purpose>

<context>
— Unit tests: `#[cfg(test)] mod tests` inline in each module
— Integration tests: `tests/` directory, test CLI commands end-to-end
— Floating-point: use `approx::assert_relative_eq!` with epsilon
— No real model files in tests — use synthetic data or small fixture tensors
— CI must run without GPU and without downloading 1GB+ model files
</context>

<procedure>
### Write unit tests for a numerical function
1. Test known-answer cases (hand-computed or from reference implementation)
2. Test edge cases: zero input, single element, very large dimensions
3. Test mathematical properties: PCA eigenvalues are non-negative and sorted, CKA is symmetric, cosine similarity in [-1, 1]
4. Use `assert_relative_eq!(actual, expected, epsilon = 1e-5)` for float comparison
5. Keep test data small: [4, 8] matrices, not [256, 1024]

### Write integration tests for CLI
1. Create temp directory with `tempfile::TempDir`
2. Place a small test image (generate programmatically or include as fixture)
3. Run CLI command via `std::process::Command` or test the library API directly
4. Assert: exit code 0, expected output files exist, JSON output parses correctly
5. Do NOT depend on real ONNX models — mock the inference layer

### Test without real models
1. Create `MockSession` that returns predetermined `ModelOutput`
2. Use trait-based design: `trait ModelBackend { fn infer(&self, image: &Image) -> Result<ModelOutput> }`
3. Real implementation uses `ort::Session`, tests use `MockSession`
4. This decouples analysis/viz tests from ONNX Runtime availability

### Property-based testing for analysis
1. PCA: for any input matrix, eigenvalues are non-negative
2. CKA: CKA(X, X) == 1.0, CKA(X, Y) == CKA(Y, X)
3. Cosine similarity: result in [-1.0, 1.0]
4. Gini: result in [0.0, 1.0], uniform input → Gini ≈ 0
5. Rank: result in [0, min(N, D)]
</procedure>

<patterns>
<do>
  — Use `#[test]` for unit tests, one test per behavior.
  — Name tests descriptively: `test_pca_eigenvalues_are_sorted`, `test_cka_is_symmetric`.
  — Use `approx::assert_relative_eq!` for ALL float comparisons.
  — Generate test matrices with `ndarray::Array2::from_shape_fn` for reproducible random data.
  — Use `#[ignore]` for tests that require model downloads, so CI can skip them.
</do>
<dont>
  — Don't compare floats with `==` — always use epsilon-based comparison.
  — Don't use large test data — keep matrices under [16, 32] for fast tests.
  — Don't depend on network access in default tests.
  — Don't test implementation details — test behavior and mathematical properties.
</dont>
</patterns>

<examples>
Example: Testing PCA properties
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_pca_eigenvalues_non_negative() {
        let data = Array2::from_shape_fn((10, 4), |(i, j)| (i * 4 + j) as f32);
        let result = pca(&data, 3).unwrap();
        for &ev in result.eigenvalues.iter() {
            assert!(ev >= 0.0, "eigenvalue {ev} is negative");
        }
    }

    #[test]
    fn test_pca_eigenvalues_sorted_descending() {
        let data = Array2::from_shape_fn((10, 4), |(i, j)| (i * 4 + j) as f32);
        let result = pca(&data, 3).unwrap();
        for w in result.eigenvalues.windows(2) {
            assert!(w[0] >= w[1], "eigenvalues not sorted: {} < {}", w[0], w[1]);
        }
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Float assertion fails with tiny diff | Epsilon too tight | Use `epsilon = 1e-4` for power method results |
| Test passes locally, fails in CI | Platform float differences | Use `epsilon = 1e-3` or `max_relative = 1e-4` |
| Integration test hangs | Waiting for stdin/network | Ensure no interactive prompts, mock network calls |
| `#[ignore]` tests not running | Not using `--ignored` flag | `cargo test -- --ignored` for model-dependent tests |
</troubleshooting>

<references>
— tests/: Integration test directory
— Inline tests in each src/ module
— approx crate docs: relative/absolute epsilon comparison
</references>
