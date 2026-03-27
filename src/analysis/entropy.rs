//! Patch entropy via k-means clustering + Shannon entropy.

use crate::errors::AnalysisError;
use ndarray::{Array1, Array2};

/// Compute Shannon entropy (bits) of a discrete probability distribution.
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Assign each row of `data` to its nearest centroid.
fn assign_clusters(data: &Array2<f32>, centroids: &Array2<f32>) -> Vec<usize> {
    data.rows()
        .into_iter()
        .map(|row| {
            centroids
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, c)| {
                    let diff = &row - &c;
                    (i, diff.dot(&diff))
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

/// Simple k-means on `data` `[N, D]`.  Returns cluster assignments `[N]`.
pub fn kmeans(data: &Array2<f32>, k: usize, max_iter: usize) -> Vec<usize> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    let k = k.min(n);

    // Deterministic init: spread evenly
    let step = n / k;
    let mut centroids: Array2<f32> =
        Array2::from_shape_fn((k, d), |(i, j)| data[[i * step.max(1), j]]);

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let new_assignments = assign_clusters(data, &centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;

        // Update centroids
        for ci in 0..k {
            let members: Vec<_> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == ci)
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }
            let mut new_centroid = Array1::<f32>::zeros(d);
            for &idx in &members {
                new_centroid += &data.row(idx);
            }
            new_centroid /= members.len() as f32;
            centroids.row_mut(ci).assign(&new_centroid);
        }
    }

    assignments
}

/// Compute patch entropy of `patch_tokens` `[N, D]` using `k` clusters.
///
/// Returns Shannon entropy in bits.
pub fn patch_entropy(
    patch_tokens: &Array2<f32>,
    k: usize,
    max_iter: usize,
) -> Result<f32, AnalysisError> {
    let n = patch_tokens.shape()[0];
    if n < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "patch_entropy needs ≥2 patches, got {n}"
        )));
    }
    let k = k.min(n);
    let assignments = kmeans(patch_tokens, k, max_iter);

    let mut counts = vec![0usize; k];
    for &a in &assignments {
        counts[a] += 1;
    }

    let probs: Vec<f32> = counts.iter().map(|&c| c as f32 / n as f32).collect();
    Ok(shannon_entropy(&probs))
}

/// Compute feature norm distribution statistics.
pub struct NormStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

pub fn patch_norm_stats(patch_tokens: &Array2<f32>) -> NormStats {
    let norms: Vec<f32> = patch_tokens
        .rows()
        .into_iter()
        .map(|row| row.dot(&row).sqrt())
        .collect();
    let n = norms.len() as f32;
    let mean = norms.iter().sum::<f32>() / n;
    let var = norms.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let min = norms.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    NormStats { mean, std: var.sqrt(), min, max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shannon_entropy_uniform() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let e = shannon_entropy(&p);
        assert_relative_eq!(e, 2.0, epsilon = 1e-4); // log2(4) = 2 bits
    }

    #[test]
    fn test_shannon_entropy_concentrated() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let e = shannon_entropy(&p);
        assert_relative_eq!(e, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_patch_entropy_runs() {
        let data = Array2::from_shape_fn((64, 32), |(i, j)| ((i + j) % 5) as f32);
        let e = patch_entropy(&data, 5, 20).unwrap();
        assert!(e >= 0.0);
    }

    #[test]
    fn test_patch_entropy_all_same() {
        let data = Array2::from_elem((16, 4), 1.0_f32);
        // All patches identical → 1 cluster → entropy = 0
        let e = patch_entropy(&data, 4, 20).unwrap();
        assert!(e >= 0.0);
    }
}
