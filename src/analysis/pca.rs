//! PCA via iterative power method — no LAPACK dependency.

use crate::analysis::finite::ensure_finite_2d;
use crate::errors::AnalysisError;
use ndarray::{Array1, Array2, Axis};

/// Result of a PCA computation.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Top-k eigenvectors, shape `[k, D]` (rows = components).
    pub components: Array2<f32>,
    /// Explained variance per component (eigenvalues), length `k`.
    pub explained_variance: Array1<f32>,
    /// Fraction of total variance explained, length `k`.
    pub explained_variance_ratio: Array1<f32>,
    /// Per-sample mean used for centering `[D]`.
    pub mean: Array1<f32>,
}

/// Compute top-`k` principal components of `data` `[N, D]` using power iteration.
///
/// # Errors
/// Returns `AnalysisError` if data has too few samples or fails to converge.
pub fn pca(data: &Array2<f32>, k: usize, max_iter: usize) -> Result<PcaResult, AnalysisError> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "PCA requires at least 2 samples, got {n}"
        )));
    }
    ensure_finite_2d(data, "input data for PCA")?;
    let k = k.min(d).min(n - 1).max(1);

    // Centre the data — safe: n >= 2 from guard above
    let mean = data
        .mean_axis(Axis(0))
        .ok_or_else(|| AnalysisError::EmptyInput("PCA input cannot be empty".into()))?;
    let mut centred = data.to_owned();
    for mut row in centred.rows_mut() {
        row -= &mean;
    }

    let mut components = Array2::<f32>::zeros((k, d));
    let mut eigenvalues = Array1::<f32>::zeros(k);

    for i in 0..k {
        // Deterministic init.
        let mut v: Array1<f32> =
            Array1::from_iter((0..d).map(|j| if j == i % d { 1.0 } else { 0.0 }));
        normalize_inplace(&mut v);

        let mut prev_lambda: Option<f32> = None;

        for iter in 0..max_iter {
            // u = X * v  [N]
            let u = centred.dot(&v);
            // v_new = X^T * u  [D]
            let v_new = centred.t().dot(&u);

            let lambda = norm(&v_new);
            if lambda < 1e-10 {
                break;
            }

            let next = v_new / lambda;
            let aligned = next.dot(&v).abs();

            if let Some(previous) = prev_lambda {
                let relative_change = (lambda - previous).abs() / previous.max(1e-10);
                v = next;
                if relative_change < 1e-6 || (1.0 - aligned) < 1e-6 {
                    break;
                }
            } else {
                v = next;
            }

            if iter == max_iter - 1 {
                return Err(AnalysisError::ConvergenceFailed {
                    iterations: max_iter,
                    reason: format!("component {i} did not converge"),
                });
            }

            prev_lambda = Some(lambda);
        }

        let u = centred.dot(&v);
        let eigenvalue = u.dot(&u) / (n - 1) as f32;
        eigenvalues[i] = eigenvalue;
        components.row_mut(i).assign(&v);

        // Deflate without allocating a temporary scaled component per row.
        let projection = centred.dot(&v);
        for (row_index, mut row) in centred.rows_mut().into_iter().enumerate() {
            let scale = projection[row_index];
            for (value, component) in row.iter_mut().zip(v.iter()) {
                *value -= component * scale;
            }
        }
    }

    let total_variance: f32 = eigenvalues.sum();
    let explained_variance_ratio = if total_variance > 1e-10 {
        eigenvalues.mapv(|e| e / total_variance)
    } else {
        Array1::zeros(k)
    };

    Ok(PcaResult {
        components,
        explained_variance: eigenvalues,
        explained_variance_ratio,
        mean,
    })
}

/// Project `data` onto PCA components (after centering with `result.mean`).
pub fn transform(data: &Array2<f32>, result: &PcaResult) -> Array2<f32> {
    let mut centred = data.to_owned();
    for mut row in centred.rows_mut() {
        row -= &result.mean;
    }
    // [N, k] = [N, D] * [D, k]
    centred.dot(&result.components.t())
}

fn norm(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize_inplace(v: &mut Array1<f32>) {
    let n = norm(v).max(1e-10);
    v.mapv_inplace(|x| x / n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pca_shape() {
        let data = Array2::from_shape_fn((100, 32), |(i, j)| (i * j) as f32 / 100.0);
        let result = pca(&data, 4, 200).unwrap();
        assert_eq!(result.components.shape(), &[4, 32]);
        assert_eq!(result.explained_variance.len(), 4);
        assert_eq!(result.explained_variance_ratio.len(), 4);
    }

    #[test]
    fn test_explained_variance_sums_to_one() {
        let data = Array2::from_shape_fn((50, 8), |(i, j)| (i + j) as f32);
        let result = pca(&data, 8, 300).unwrap();
        let total: f32 = result.explained_variance_ratio.sum();
        // Should sum to ~1.0 (may be slightly off due to deflation)
        assert_relative_eq!(total, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_pca_too_few_samples() {
        let data = Array2::from_elem((1, 8), 1.0_f32);
        assert!(pca(&data, 2, 100).is_err());
    }

    #[test]
    fn test_transform_shape() {
        let data = Array2::from_shape_fn((20, 16), |(i, j)| (i * j) as f32 / 50.0);
        let result = pca(&data, 3, 200).unwrap();
        let projected = transform(&data, &result);
        assert_eq!(projected.shape(), &[20, 3]);
    }

    #[test]
    fn test_pca_handles_wide_matrices() {
        let data = Array2::from_shape_fn((8, 64), |(i, j)| ((i * 11 + j * 3) % 17) as f32);
        let result = pca(&data, 4, 200).unwrap();
        let projected = transform(&data, &result);

        assert_eq!(result.components.shape(), &[4, 64]);
        assert_eq!(projected.shape(), &[8, 4]);
        assert!(result.explained_variance.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn test_pca_rejects_non_finite_values() {
        let mut data = Array2::from_elem((8, 4), 1.0_f32);
        data[[5, 2]] = f32::NAN;

        let error = pca(&data, 2, 100).unwrap_err();

        assert!(matches!(error, AnalysisError::NonFiniteValues { .. }));
    }
}
