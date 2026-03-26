---
name: analysis-pipeline
description: Numerical analysis pipeline — PCA, CKA, k-NN, representation rank, variance spectrum, Gini coefficient, patch entropy, and cross-model metrics. Activate when implementing or debugging any metric computation in the analysis/ module.
prerequisites: ndarray, approx (for tests)
---

# Analysis Pipeline

<purpose>
Covers the mathematical analysis layer: all per-model and cross-model metrics. PCA via power method, CKA for representation similarity, k-NN for neighbor overlap, and statistical measures (rank, variance, Gini, entropy). All computation is pure Rust on ndarray arrays.
</purpose>

<context>
— Input: ModelOutput with patch_tokens [N, D], cls_token [D], attention_weights [L, H, N, N]
— Per-model: rank, variance spectrum, Gini, patch entropy, dead dimensions, norm distribution
— Cross-model: CLS cosine similarity, CKA, mutual k-NN overlap, patch correspondence
— Default PCA: power method (no LAPACK). Optional: full SVD via `full-svd` feature.
— All metrics return f32 or small ndarray results, not large tensors.
</context>

<procedure>
### PCA (power method — default)
1. Input: patch_tokens `Array2<f32>` [N, D]
2. Center: subtract mean along axis 0
3. Compute covariance: `X^T * X / (N-1)`
4. Iterate power method for top-k eigenvalues:
   a. Random initial vector
   b. Multiply by covariance matrix
   c. Normalize
   d. Repeat until convergence (or max 100 iterations)
   e. Deflate matrix, repeat for next eigenvalue
5. Output: eigenvalues [k], eigenvectors [D, k]

### Representation rank
1. Compute singular values (top-k via power method on X^T * X)
2. Threshold: count values > 1% of maximum singular value
3. Return: effective rank / total dimensions

### Centered Kernel Alignment (CKA)
1. Input: features_A [N, D_a], features_B [N, D_b] (same N images)
2. Compute linear kernels: K_a = X_a * X_a^T, K_b = X_b * X_b^T
3. Center kernels: H * K * H where H = I - 1/n * 11^T
4. CKA = HSIC(K_a, K_b) / sqrt(HSIC(K_a, K_a) * HSIC(K_b, K_b))
5. HSIC = trace(K_a_centered * K_b_centered) / (n-1)^2
6. Output: scalar in [0, 1]

### k-NN overlap
1. For each image, compute k=10 nearest neighbors in feature space
2. Compare neighbor sets between model A and model B
3. Overlap = |intersection| / k, averaged across all images
4. Output: scalar in [0, 1]

### Gini coefficient (attention concentration)
1. Input: attention_weights for a single head [N, N]
2. Flatten, sort ascending
3. Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
4. Output: scalar in [0, 1] (0 = uniform, 1 = maximally concentrated)
</procedure>

<patterns>
<do>
  — Use `ndarray::Array2::dot()` for matrix multiplication — it's optimized.
  — Center data before PCA/CKA — uncentered results are meaningless.
  — Validate input shapes at function entry with debug_assert!.
  — Return structured results: `PcaResult { eigenvalues, eigenvectors, explained_variance }`.
  — Use `f64` internally for accumulations (CKA, Gini) to avoid float precision issues, cast result to f32.
</do>
<dont>
  — Don't allocate large intermediate matrices when a streaming computation works (e.g., cosine similarity row-by-row).
  — Don't compute full SVD when only top-k eigenvalues are needed — use power method.
  — Don't assume square inputs — N (patches) ≠ D (dimensions) in general.
  — Don't forget to handle edge cases: all-zero features (dead model), single patch, N < k for k-NN.
</dont>
</patterns>

<examples>
Example: Cosine similarity between two CLS tokens
```rust
use ndarray::Array1;

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| PCA eigenvalues negative | Numerical instability | Use f64 for covariance, clamp eigenvalues to >= 0 |
| CKA > 1.0 | Centering error | Verify H matrix construction, check (n-1)^2 denominator |
| k-NN returns fewer than k | Dataset smaller than k | Use min(k, dataset_size - 1) |
| Gini = NaN | All-zero attention weights | Check for zero-sum before computing, return 0.0 |
| Power method doesn't converge | Repeated eigenvalues | Add small random perturbation, increase max iterations |
</troubleshooting>

<references>
— src/analysis/pca.rs: PCA implementation
— src/analysis/cka.rs: CKA implementation
— src/analysis/knn.rs: k-NN overlap
— src/analysis/attention.rs: Gini coefficient
— SPECIFICATION.md: Metric definitions table
— RESOURCES.md: CKA paper (Kornblith 2019)
</references>
