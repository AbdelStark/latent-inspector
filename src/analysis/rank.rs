//! Representation rank: effective dimensionality via singular value thresholding.

use crate::analysis::pca;
use crate::errors::AnalysisError;
use ndarray::Array2;

/// Compute the effective rank of `data` `[N, D]`.
///
/// The rank is the number of singular values (PCA eigenvalues) above
/// `threshold_ratio * max_eigenvalue`.
pub fn effective_rank(
    data: &Array2<f32>,
    threshold_ratio: f32,
    max_components: usize,
) -> Result<usize, AnalysisError> {
    let k = max_components.min(data.shape()[0] - 1).min(data.shape()[1]);
    let result = pca::pca(data, k, 500)?;
    let max_ev = result
        .explained_variance
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if max_ev <= 0.0 {
        return Ok(0);
    }
    let threshold = threshold_ratio * max_ev;
    let rank = result
        .explained_variance
        .iter()
        .filter(|&&ev| ev > threshold)
        .count();
    Ok(rank)
}

/// Count dimensions with near-zero variance across patches (dead dimensions).
///
/// A dimension is "dead" if its variance across all patches is below `threshold`.
pub fn dead_dimensions(data: &Array2<f32>, threshold: f32) -> usize {
    let d = data.shape()[1];
    let n = data.shape()[0] as f32;

    (0..d)
        .filter(|&j| {
            let col = data.column(j);
            let mean = col.sum() / n;
            let var: f32 = col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            var < threshold
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_effective_rank_full() {
        // Diagonal data: identity-like, all dimensions active
        let data = Array2::from_shape_fn((64, 8), |(i, j)| {
            if i % 8 == j { 1.0 } else { 0.1 }
        });
        let rank = effective_rank(&data, 0.01, 8).unwrap();
        assert!(rank >= 1);
    }

    #[test]
    fn test_effective_rank_low() {
        // Data confined to 1 direction
        let data = Array2::from_shape_fn((32, 8), |(i, _j)| i as f32);
        let rank = effective_rank(&data, 0.01, 8).unwrap();
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_dead_dimensions_none() {
        let data = Array2::from_shape_fn((10, 4), |(i, j)| (i + j * 3) as f32);
        let dead = dead_dimensions(&data, 1e-6);
        assert_eq!(dead, 0);
    }

    #[test]
    fn test_dead_dimensions_all() {
        // All columns constant → all dead
        let data = Array2::from_elem((10, 4), 1.0_f32);
        let dead = dead_dimensions(&data, 1e-6);
        assert_eq!(dead, 4);
    }
}
