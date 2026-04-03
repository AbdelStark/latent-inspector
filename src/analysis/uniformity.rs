//! Uniformity metric for representation space analysis.
//!
//! Measures how uniformly features are distributed on the unit hypersphere,
//! following Wang & Isola (2020) "Understanding Contrastive Representation
//! Learning through Alignment and Uniformity on the Hypersphere".
//!
//! The uniformity loss is defined as:
//!   L_uniform = log E[exp(-t * ||f(x) - f(y)||^2)]
//!
//! where the expectation is over all pairs (x, y) drawn independently.
//! More negative values indicate better uniformity (more spread out).

use crate::analysis::finite::ensure_finite_2d;
use crate::errors::AnalysisError;
use ndarray::Array2;

/// Default temperature parameter for uniformity computation.
const DEFAULT_T: f32 = 2.0;

/// Compute the uniformity metric for a set of L2-normalized embeddings.
///
/// Input: `embeddings` of shape `[N, D]`. Embeddings are L2-normalized
/// internally before computing the metric.
///
/// Returns a value typically in the range `[-4, 0]`. More negative means
/// better uniformity (representations more evenly spread on the hypersphere).
///
/// # Errors
/// Returns `AnalysisError` if there are fewer than 2 embeddings or values are non-finite.
pub fn uniformity(embeddings: &Array2<f32>) -> Result<f32, AnalysisError> {
    uniformity_with_temperature(embeddings, DEFAULT_T)
}

/// Compute uniformity with a custom temperature parameter `t`.
///
/// # Errors
/// Returns `AnalysisError` if data is insufficient or non-finite.
pub fn uniformity_with_temperature(embeddings: &Array2<f32>, t: f32) -> Result<f32, AnalysisError> {
    let n = embeddings.shape()[0];
    if n < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "Uniformity requires at least 2 embeddings, got {n}"
        )));
    }
    ensure_finite_2d(embeddings, "embeddings for uniformity")?;

    // L2-normalize all embeddings
    let normalized = l2_normalize_rows(embeddings);

    // Collect exponent values: -t * ||f(x) - f(y)||^2 for all pairs.
    // We then use log-sum-exp for numerical stability:
    //   log(mean(exp(x_i))) = max_x + log(mean(exp(x_i - max_x)))
    let mut exponents = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        let row_i = normalized.row(i);
        for j in (i + 1)..n {
            let row_j = normalized.row(j);
            let sq_dist: f32 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            exponents.push((-t * sq_dist) as f64);
        }
    }

    if exponents.is_empty() {
        return Err(AnalysisError::InsufficientData(
            "No valid pairs for uniformity computation".into(),
        ));
    }

    // Log-sum-exp trick: shift by max to prevent overflow/underflow
    let max_exp = exponents.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_shifted: f64 = exponents.iter().map(|&x| (x - max_exp).exp()).sum();
    let log_mean = max_exp + (sum_shifted / exponents.len() as f64).ln();

    Ok(log_mean as f32)
}

/// L2-normalize each row of the input matrix.
fn l2_normalize_rows(data: &Array2<f32>) -> Array2<f32> {
    let mut out = data.to_owned();
    for mut row in out.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        row.mapv_inplace(|v| v / norm);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn uniformity_of_identical_vectors_is_zero() {
        // All vectors identical => sq_dist = 0 => exp(0) = 1 => log(1) = 0
        let data = Array2::from_shape_fn((20, 8), |(_i, j)| (j + 1) as f32);
        let score = uniformity(&data).unwrap();
        approx::assert_relative_eq!(score, 0.0, epsilon = 0.01);
    }

    #[test]
    fn uniformity_of_spread_vectors_is_negative() {
        // Orthogonal unit vectors should be well-spread
        let mut data = Array2::zeros((16, 16));
        for i in 0..16 {
            data[[i, i]] = 1.0;
        }
        let score = uniformity(&data).unwrap();
        assert!(
            score < -1.0,
            "Expected strongly negative uniformity, got {score}"
        );
    }

    #[test]
    fn uniformity_requires_two_samples() {
        let data = Array2::from_elem((1, 4), 1.0_f32);
        assert!(uniformity(&data).is_err());
    }

    #[test]
    fn uniformity_rejects_non_finite() {
        let mut data = Array2::from_elem((4, 4), 1.0_f32);
        data[[2, 3]] = f32::INFINITY;
        assert!(uniformity(&data).is_err());
    }

    #[test]
    fn custom_temperature_affects_magnitude() {
        let data = Array2::from_shape_fn((10, 8), |(i, j)| ((i * 7 + j * 3) % 11) as f32 / 11.0);
        let low_t = uniformity_with_temperature(&data, 1.0).unwrap();
        let high_t = uniformity_with_temperature(&data, 4.0).unwrap();
        // Higher temperature should produce more negative values for spread data
        assert!(
            high_t < low_t,
            "Expected higher t to give more negative uniformity: t=1 => {low_t}, t=4 => {high_t}"
        );
    }
}
