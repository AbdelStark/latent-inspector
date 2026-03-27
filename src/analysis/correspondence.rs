//! Patch correspondence via optimal assignment (Hungarian algorithm).

use crate::errors::AnalysisError;
use ndarray::Array2;
use pathfinding::matrix::Matrix;
use pathfinding::prelude::kuhn_munkres_min;

/// Compute cosine similarity between each pair of patches from `a` and `b`.
///
/// `a`: `[Na, D]`, `b`: `[Nb, D]` → returns `[Na, Nb]`.
pub fn patch_cosine_similarity(
    a: &Array2<f32>,
    b: &Array2<f32>,
) -> Result<Array2<f32>, AnalysisError> {
    let (na, da) = (a.shape()[0], a.shape()[1]);
    let (nb, db) = (b.shape()[0], b.shape()[1]);

    if da != db {
        return Err(AnalysisError::ShapeMismatch {
            expected: vec![na, da],
            actual: vec![nb, db],
        });
    }

    // Normalise
    let norm_a = normalize_rows(a);
    let norm_b = normalize_rows(b);

    // [Na, Nb] = [Na, D] * [D, Nb]
    Ok(norm_a.dot(&norm_b.t()))
}

fn normalize_rows(m: &Array2<f32>) -> Array2<f32> {
    let mut out = m.clone();
    for mut row in out.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        row.mapv_inplace(|v| v / norm);
    }
    out
}

/// Result of patch correspondence matching.
#[derive(Debug, Clone)]
pub struct CorrespondenceResult {
    /// Assignment: `assignments[i]` = index in `b` matched to patch `i` in `a`.
    pub assignments: Vec<usize>,
    /// Mean cosine similarity of matched pairs.
    pub mean_similarity: f32,
    /// Cosine similarity for each matched pair.
    pub pair_similarities: Vec<f32>,
}

/// Find optimal patch correspondence between `a` and `b` using Hungarian matching
/// on cosine similarity.
///
/// When `a` and `b` have different numbers of patches, only `min(Na, Nb)` are matched.
pub fn patch_correspondence(
    a: &Array2<f32>,
    b: &Array2<f32>,
) -> Result<CorrespondenceResult, AnalysisError> {
    let na = a.shape()[0];
    let nb = b.shape()[0];

    let sim = patch_cosine_similarity(a, b)?;

    // Hungarian algorithm minimizes cost; convert similarity to cost (1 - sim)
    let n = na.min(nb);
    let scale = 10000i64;

    // Pre-build the cost matrix as a flat Vec to avoid borrow issues
    let mut cost_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            cost_flat.push(((1.0 - sim[[i, j]]) * scale as f32) as i64);
        }
    }
    let cost_matrix = Matrix::from_vec(n, n, cost_flat).unwrap_or_else(|_| Matrix::new(n, n, 0i64));

    let (_total_cost, assignments): (i64, Vec<usize>) = kuhn_munkres_min(&cost_matrix);

    let pair_similarities: Vec<f32> = assignments
        .iter()
        .enumerate()
        .take(n)
        .map(|(i, &j)| sim[[i, j]])
        .collect();

    let mean_similarity = if pair_similarities.is_empty() {
        0.0
    } else {
        pair_similarities.iter().sum::<f32>() / pair_similarities.len() as f32
    };

    Ok(CorrespondenceResult {
        assignments: assignments.into_iter().take(n).collect(),
        mean_similarity,
        pair_similarities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cosine_sim_shape() {
        let a = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((4, 8), |(i, j)| (i * 2 + j) as f32);
        let sim = patch_cosine_similarity(&a, &b).unwrap();
        assert_eq!(sim.shape(), &[4, 4]);
    }

    #[test]
    fn test_correspondence_identical() {
        let a = Array2::from_shape_fn((4, 8), |(i, j)| if j == i { 1.0 } else { 0.0 });
        let result = patch_correspondence(&a, &a).unwrap();
        // Each patch should match itself
        for (i, &j) in result.assignments.iter().enumerate() {
            assert_eq!(i, j);
        }
        assert_relative_eq!(result.mean_similarity, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_similarity_range() {
        let a = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((4, 8), |(i, j)| (i * j + 1) as f32);
        let result = patch_correspondence(&a, &b).unwrap();
        assert!(result.mean_similarity >= -1.0 && result.mean_similarity <= 1.0 + 1e-5);
    }
}
