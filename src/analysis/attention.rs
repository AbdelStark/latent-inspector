//! Attention concentration metrics (Gini coefficient).

use crate::errors::AnalysisError;
use ndarray::{Array1, Array4};

/// Compute the Gini coefficient of a 1-D distribution.
///
/// Returns a value in `[0, 1]`: 0 = uniform, 1 = fully concentrated.
pub fn gini(values: &Array1<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f32;
    let sum: f32 = values.sum();
    if sum.abs() < 1e-10 {
        return 0.0;
    }

    let mut sorted: Vec<f32> = values.iter().copied().map(|v| v.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut gini_sum = 0.0_f32;
    for (i, &v) in sorted.iter().enumerate() {
        gini_sum += (2.0 * (i as f32 + 1.0) - n - 1.0) * v;
    }
    gini_sum / (n * sum)
}

/// Per-head Gini coefficients averaged over patches and layers.
///
/// `weights`: `[layers, heads, N, N]`
/// Returns `[heads]`.
pub fn per_head_gini(weights: &Array4<f32>) -> Result<Array1<f32>, AnalysisError> {
    let shape = weights.shape();
    if shape.len() != 4 {
        return Err(AnalysisError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let (layers, heads, n, _) = (shape[0], shape[1], shape[2], shape[3]);
    let mut result = Array1::<f32>::zeros(heads);

    for h in 0..heads {
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for l in 0..layers {
            for q in 0..n {
                let row: Array1<f32> = (0..n)
                    .map(|k| weights[[l, h, q, k]])
                    .collect::<Array1<f32>>();
                sum += gini(&row);
                count += 1;
            }
        }
        result[h] = if count > 0 { sum / count as f32 } else { 0.0 };
    }

    Ok(result)
}

/// Mean Gini coefficient across all heads, patches, and layers.
pub fn mean_gini(weights: &Array4<f32>) -> Result<f32, AnalysisError> {
    let per_head = per_head_gini(weights)?;
    Ok(per_head.mean().unwrap_or(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_gini_uniform() {
        let v = array![0.25, 0.25, 0.25, 0.25];
        assert_relative_eq!(gini(&v), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gini_concentrated() {
        // All mass on one token → Gini close to 1
        let v = array![0.0, 0.0, 0.0, 1.0];
        let g = gini(&v);
        assert!(g > 0.6, "expected high Gini, got {g}");
    }

    #[test]
    fn test_gini_range() {
        let v = array![0.1, 0.2, 0.3, 0.4];
        let g = gini(&v);
        assert!(g >= 0.0 && g <= 1.0);
    }

    #[test]
    fn test_per_head_gini_shape() {
        let weights = Array4::from_elem((2, 4, 8, 8), 0.125_f32);
        let result = per_head_gini(&weights).unwrap();
        assert_eq!(result.len(), 4);
    }
}
