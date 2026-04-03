//! Maximum Likelihood Estimation of intrinsic dimensionality.
//!
//! Implements the Levina & Bickel (2004) estimator for the intrinsic
//! dimensionality of a representation manifold. This estimates the true
//! number of degrees of freedom in the data, which is often much lower
//! than the ambient embedding dimension.
//!
//! A model with 1024-dim embeddings might have an intrinsic dimensionality
//! of only 20-50, meaning the representations live on a low-dimensional
//! manifold within the high-dimensional space.

use crate::analysis::finite::ensure_finite_2d;
use crate::errors::AnalysisError;
use ndarray::Array2;

/// Estimate the intrinsic dimensionality of the data using the MLE method
/// of Levina & Bickel (2004).
///
/// Input: `embeddings` of shape `[N, D]`. Uses `k` nearest neighbors
/// for the local estimate at each point, then averages.
///
/// Returns the estimated intrinsic dimensionality (a positive float).
///
/// # Errors
/// Returns `AnalysisError` if there are too few samples or values are non-finite.
pub fn intrinsic_dimensionality(embeddings: &Array2<f32>, k: usize) -> Result<f32, AnalysisError> {
    let n = embeddings.shape()[0];
    if n < k + 1 {
        return Err(AnalysisError::InsufficientData(format!(
            "Intrinsic dimensionality with k={k} requires at least {} samples, got {n}",
            k + 1
        )));
    }
    if k < 2 {
        return Err(AnalysisError::InsufficientData(
            "Intrinsic dimensionality requires k >= 2".into(),
        ));
    }
    ensure_finite_2d(embeddings, "embeddings for intrinsic dimensionality")?;

    // For each point, find k nearest neighbors and compute local dimension estimate
    let mut dim_estimates = Vec::with_capacity(n);

    for i in 0..n {
        let row_i = embeddings.row(i);

        // Compute squared distances to all other points
        let mut distances: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let row_j = embeddings.row(j);
                row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .collect();

        // Partial sort to get k smallest distances
        distances.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
        let mut k_distances: Vec<f32> = distances[..k].to_vec();
        k_distances.sort_by(|a, b| a.total_cmp(b));

        // Convert to actual distances (sqrt)
        let k_dists: Vec<f32> = k_distances.iter().map(|d| d.sqrt().max(1e-10)).collect();

        // MLE estimate: m_k(x) = 1 / (1/(k-1) * sum_{j=1}^{k-1} log(T_k / T_j))
        // where T_j is the distance to the j-th neighbor
        let t_k = k_dists[k - 1];
        let log_t_k = t_k.ln();

        let sum_log_ratios: f32 = k_dists[..(k - 1)]
            .iter()
            .map(|t_j| log_t_k - t_j.ln())
            .sum();

        if sum_log_ratios > 1e-10 {
            let local_dim = (k - 1) as f32 / sum_log_ratios;
            dim_estimates.push(local_dim);
        }
    }

    if dim_estimates.is_empty() {
        return Err(AnalysisError::InsufficientData(
            "All local dimensionality estimates were degenerate".into(),
        ));
    }

    // Average over all points
    let mean_dim = dim_estimates.iter().sum::<f32>() / dim_estimates.len() as f32;
    Ok(mean_dim)
}

/// Compute intrinsic dimensionality with default k=10.
///
/// # Errors
/// Returns `AnalysisError` if there are too few samples.
pub fn intrinsic_dimensionality_default(embeddings: &Array2<f32>) -> Result<f32, AnalysisError> {
    let n = embeddings.shape()[0];
    let k = 10.min(n.saturating_sub(1)).max(2);
    intrinsic_dimensionality(embeddings, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn intrinsic_dim_of_line_is_near_one() {
        // Points along a 1-D line embedded in 16-D space
        let data = Array2::from_shape_fn(
            (100, 16),
            |(i, j)| {
                if j == 0 {
                    i as f32 * 0.1
                } else {
                    0.0
                }
            },
        );
        let dim = intrinsic_dimensionality(&data, 5).unwrap();
        assert!(dim < 2.5, "Expected ~1D for a line, got {dim}");
    }

    #[test]
    fn intrinsic_dim_of_plane_is_near_two() {
        // Points on a 2-D plane embedded in 16-D space
        let data = Array2::from_shape_fn((200, 16), |(i, j)| {
            if j == 0 {
                (i / 14) as f32 * 0.1
            } else if j == 1 {
                (i % 14) as f32 * 0.1
            } else {
                0.0
            }
        });
        let dim = intrinsic_dimensionality(&data, 5).unwrap();
        assert!(
            dim > 1.3 && dim < 3.5,
            "Expected ~2D for a plane, got {dim}"
        );
    }

    #[test]
    fn intrinsic_dim_of_high_dimensional_is_higher_than_line() {
        // Line data: 1-D
        let line = Array2::from_shape_fn(
            (100, 16),
            |(i, j)| {
                if j == 0 {
                    i as f32 * 0.1
                } else {
                    0.0
                }
            },
        );
        let dim_line = intrinsic_dimensionality(&line, 5).unwrap();

        // Multi-dimensional data: independent variation along each axis
        let multi = Array2::from_shape_fn((100, 8), |(i, j)| {
            // Use a different linear combination per dimension to create higher-rank data
            (i as f32 * (j + 1) as f32 * 0.01) + ((i * 7 + j * 13) as f32).sin() * 0.5
        });
        let dim_multi = intrinsic_dimensionality(&multi, 5).unwrap();

        assert!(
            dim_multi > dim_line,
            "Expected multi-dim ({dim_multi}) > line ({dim_line})"
        );
    }

    #[test]
    fn intrinsic_dim_requires_enough_samples() {
        let data = Array2::from_elem((3, 8), 1.0_f32);
        assert!(intrinsic_dimensionality(&data, 5).is_err());
    }

    #[test]
    fn intrinsic_dim_rejects_non_finite() {
        let mut data = Array2::from_elem((20, 4), 1.0_f32);
        data[[5, 2]] = f32::NAN;
        assert!(intrinsic_dimensionality(&data, 3).is_err());
    }

    #[test]
    fn default_k_adapts_to_small_datasets() {
        let data = Array2::from_shape_fn((5, 4), |(i, j)| (i * 4 + j) as f32);
        let dim = intrinsic_dimensionality_default(&data).unwrap();
        assert!(dim > 0.0, "Expected positive dimension, got {dim}");
    }
}
