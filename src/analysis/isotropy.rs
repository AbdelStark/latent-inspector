//! Isotropy metrics for representation space analysis.
//!
//! Isotropy measures how uniformly representation vectors are distributed
//! across the embedding space. Low isotropy (close to 0) indicates vectors
//! are clustered in a narrow cone — common with contrastive learning methods.
//! High isotropy (close to 1) means vectors fill the space more uniformly.

use crate::analysis::finite::ensure_finite_2d;
use crate::errors::AnalysisError;
use ndarray::{Array2, Axis};

/// Compute the isotropy score of a set of embeddings.
///
/// Isotropy is defined as 1 minus the average pairwise cosine similarity
/// between all embedding pairs. The result is clamped to `[0, 1]`:
///
/// - **0.0** — all vectors point in the same direction (collapsed)
/// - **~1.0** — vectors are uniformly spread (orthogonal on average)
///
/// Values above 1.0 (anti-correlated embeddings where average cosine < 0)
/// are clamped to 1.0 since they indicate maximum dispersion.
///
/// Input: `embeddings` of shape `[N, D]` where N is the number of samples
/// and D is the embedding dimension.
///
/// # Errors
/// Returns `AnalysisError` if there are fewer than 2 embeddings or values are non-finite.
pub fn isotropy_score(embeddings: &Array2<f32>) -> Result<f32, AnalysisError> {
    let n = embeddings.shape()[0];
    if n < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "Isotropy requires at least 2 embeddings, got {n}"
        )));
    }
    ensure_finite_2d(embeddings, "embeddings for isotropy")?;

    // Compute L2 norms
    let norms: Vec<f32> = embeddings
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10))
        .collect();

    // Compute average pairwise cosine similarity
    let mut total_sim = 0.0_f64;
    let mut pair_count = 0_u64;

    for i in 0..n {
        let row_i = embeddings.row(i);
        for j in (i + 1)..n {
            let row_j = embeddings.row(j);
            let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let cosine = (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0);
            total_sim += cosine as f64;
            pair_count += 1;
        }
    }

    let avg_cosine = if pair_count > 0 {
        (total_sim / pair_count as f64) as f32
    } else {
        0.0
    };

    // Isotropy = 1 - avg_cosine_similarity, clamped to [0, 1].
    // Anti-correlated embeddings (avg_cosine < 0) are clamped to 1.0.
    Ok((1.0 - avg_cosine).clamp(0.0, 1.0))
}

/// Compute the partition function-based isotropy score (Mu et al., 2018).
///
/// This measures how uniformly the singular values of the embedding matrix
/// are distributed. An isotropic representation has roughly equal singular
/// values. The score is exp(-variance of log singular values), ranging
/// from 0 (anisotropic, dominated by top components) to 1 (perfectly isotropic).
///
/// Input: `embeddings` of shape `[N, D]`.
///
/// # Errors
/// Returns `AnalysisError` if data is insufficient or non-finite.
pub fn partition_isotropy(embeddings: &Array2<f32>) -> Result<f32, AnalysisError> {
    let n = embeddings.shape()[0];
    let d = embeddings.shape()[1];
    if n < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "Partition isotropy requires at least 2 embeddings, got {n}"
        )));
    }
    ensure_finite_2d(embeddings, "embeddings for partition isotropy")?;

    // Center the data
    let mean = embeddings
        .mean_axis(Axis(0))
        .ok_or_else(|| AnalysisError::EmptyInput("isotropy input cannot be empty".into()))?;
    let mut centered = embeddings.to_owned();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }

    // Compute eigenvalues via C = X^T X / (n-1), then power iteration
    // We use variance of the log of the eigenvalues
    let k = d.min(n - 1).min(32); // Compute top-k eigenvalues
    let eigenvalues = top_eigenvalues(&centered, k, 100)?;

    let positive_eigenvalues: Vec<f32> = eigenvalues.into_iter().filter(|&e| e > 1e-10).collect();
    if positive_eigenvalues.is_empty() {
        return Ok(0.0);
    }

    let log_eigenvalues: Vec<f32> = positive_eigenvalues.iter().map(|e| e.ln()).collect();
    let mean_log = log_eigenvalues.iter().sum::<f32>() / log_eigenvalues.len() as f32;
    let variance_log = log_eigenvalues
        .iter()
        .map(|l| (l - mean_log).powi(2))
        .sum::<f32>()
        / log_eigenvalues.len() as f32;

    // Score = exp(-variance), bounded to [0, 1]
    Ok((-variance_log).exp().clamp(0.0, 1.0))
}

/// Extract top-k eigenvalues of the covariance matrix using power iteration.
fn top_eigenvalues(
    centered: &Array2<f32>,
    k: usize,
    max_iter: usize,
) -> Result<Vec<f32>, AnalysisError> {
    let (_n, d) = (centered.shape()[0], centered.shape()[1]);
    let n_minus_1 = (centered.shape()[0] - 1).max(1) as f32;

    let mut deflated = centered.to_owned();
    let mut eigenvalues = Vec::with_capacity(k);

    for i in 0..k {
        let mut v = ndarray::Array1::<f32>::zeros(d);
        v[i % d] = 1.0;

        for _ in 0..max_iter {
            let u = deflated.dot(&v);
            let v_new = deflated.t().dot(&u);
            let lambda = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            if lambda < 1e-12 {
                break;
            }
            let next = v_new / lambda;
            let converged = (1.0 - next.dot(&v).abs()) < 1e-6;
            v = next;
            if converged {
                break;
            }
        }

        let u = deflated.dot(&v);
        let eigenvalue = u.dot(&u) / n_minus_1;
        eigenvalues.push(eigenvalue);

        // Deflate
        let projection = deflated.dot(&v);
        for (row_idx, mut row) in deflated.rows_mut().into_iter().enumerate() {
            let scale = projection[row_idx];
            for (value, comp) in row.iter_mut().zip(v.iter()) {
                *value -= comp * scale;
            }
        }
    }

    Ok(eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn isotropy_of_identical_vectors_is_near_zero() {
        let data = Array2::from_shape_fn((50, 16), |(_i, j)| (j + 1) as f32);
        let score = isotropy_score(&data).unwrap();
        assert!(score < 0.05, "Expected low isotropy, got {score}");
    }

    #[test]
    fn isotropy_of_orthogonal_basis_is_high() {
        let mut data = Array2::zeros((16, 16));
        for i in 0..16 {
            data[[i, i]] = 1.0;
        }
        let score = isotropy_score(&data).unwrap();
        assert!(score > 0.9, "Expected high isotropy, got {score}");
    }

    #[test]
    fn isotropy_requires_two_samples() {
        let data = Array2::from_elem((1, 8), 1.0_f32);
        assert!(isotropy_score(&data).is_err());
    }

    #[test]
    fn isotropy_rejects_non_finite() {
        let mut data = Array2::from_elem((4, 4), 1.0_f32);
        data[[1, 2]] = f32::NAN;
        assert!(isotropy_score(&data).is_err());
    }

    #[test]
    fn partition_isotropy_of_uniform_eigenvalues_is_high() {
        // Create data with roughly equal variance in all directions
        let data = Array2::from_shape_fn((100, 8), |(i, j)| ((i * 7 + j * 13) % 37) as f32 / 37.0);
        let score = partition_isotropy(&data).unwrap();
        assert!(
            score > 0.3,
            "Expected moderate-to-high partition isotropy, got {score}"
        );
    }

    #[test]
    fn partition_isotropy_of_rank_one_data_is_low() {
        // Data lies on a single direction
        let data = Array2::from_shape_fn((50, 16), |(i, _j)| i as f32);
        let score = partition_isotropy(&data).unwrap();
        assert!(score < 0.3, "Expected low partition isotropy, got {score}");
    }
}
