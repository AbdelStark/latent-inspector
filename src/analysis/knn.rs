//! Mutual k-NN overlap between two representation sets.

use crate::analysis::finite::ensure_finite_2d;
use crate::errors::AnalysisError;
use ndarray::Array2;

/// Compute pairwise cosine similarity matrix `[N, N]` for rows of `data`.
pub fn cosine_similarity_matrix(data: &Array2<f32>) -> Array2<f32> {
    let n = data.shape()[0];
    let d = data.shape()[1];

    // Normalize rows
    let mut normed = data.clone();
    for mut row in normed.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        row.mapv_inplace(|v| v / norm);
    }

    let mut sim = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let dot: f32 = (0..d).map(|k| normed[[i, k]] * normed[[j, k]]).sum();
            sim[[i, j]] = dot;
            sim[[j, i]] = dot;
        }
    }
    sim
}

/// Find k nearest neighbors for each row in the similarity matrix.
/// Returns `[N, k]` indices.
pub fn top_k_neighbors(sim: &Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let n = sim.shape()[0];
    let k = k.min(n.saturating_sub(1));

    (0..n)
        .map(|i| {
            let mut indexed: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, sim[[i, j]]))
                .collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            indexed.truncate(k);
            indexed.into_iter().map(|(j, _)| j).collect()
        })
        .collect()
}

/// Mutual k-NN overlap: fraction of k nearest neighbors that agree between two representations.
///
/// `a` and `b`: `[N, D_a]` and `[N, D_b]` — same samples, different representations.
pub fn knn_overlap(a: &Array2<f32>, b: &Array2<f32>, k: usize) -> Result<f32, AnalysisError> {
    let na = a.shape()[0];
    let nb = b.shape()[0];

    if na != nb {
        return Err(AnalysisError::ShapeMismatch {
            expected: vec![na],
            actual: vec![nb],
        });
    }
    if na < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "knn_overlap requires ≥2 samples, got {na}"
        )));
    }
    ensure_finite_2d(a, "left representation for k-NN overlap")?;
    ensure_finite_2d(b, "right representation for k-NN overlap")?;

    let sim_a = cosine_similarity_matrix(a);
    let sim_b = cosine_similarity_matrix(b);

    let nn_a = top_k_neighbors(&sim_a, k);
    let nn_b = top_k_neighbors(&sim_b, k);

    let total_overlap: usize = nn_a
        .iter()
        .zip(nn_b.iter())
        .map(|(na, nb)| {
            let set_b: std::collections::HashSet<_> = nb.iter().copied().collect();
            na.iter().filter(|&&x| set_b.contains(&x)).count()
        })
        .sum();

    let effective_k = k.min(na - 1);
    if effective_k == 0 {
        return Ok(0.0);
    }
    Ok(total_overlap as f32 / (na * effective_k) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_knn_overlap_identical() {
        let data = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as f32);
        let overlap = knn_overlap(&data, &data, 3).unwrap();
        assert_relative_eq!(overlap, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_knn_overlap_range() {
        let a = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((10, 4), |(i, j)| (i * j + 1) as f32);
        let overlap = knn_overlap(&a, &b, 3).unwrap();
        assert!((0.0..=1.0 + 1e-5).contains(&overlap));
    }

    #[test]
    fn test_cosine_sim_diagonal() {
        // Square identity: each row is a basis vector; self-similarity = 1.
        let data = Array2::from_shape_fn((4, 4), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let sim = cosine_similarity_matrix(&data);
        for i in 0..4 {
            assert_relative_eq!(sim[[i, i]], 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_knn_overlap_rejects_non_finite_values() {
        let a = Array2::from_elem((10, 4), 1.0_f32);
        let mut b = Array2::from_elem((10, 4), 1.0_f32);
        b[[4, 2]] = f32::NAN;

        let error = knn_overlap(&a, &b, 3).unwrap_err();

        assert!(matches!(error, AnalysisError::NonFiniteValues { .. }));
    }
}
