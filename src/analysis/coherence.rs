//! Spatial coherence analysis for patch representations.
//!
//! Spatial coherence measures how smoothly a model's representations vary
//! across neighboring patches in the spatial grid. High coherence (close to 1)
//! means adjacent patches have similar representations — characteristic of
//! models like DINOv2 that produce unsupervised segmentation-like features.
//! Low coherence means each patch encodes distinct information regardless of
//! spatial proximity — characteristic of I-JEPA's latent prediction objective.
//!
//! This directly quantifies a fundamental difference between SSL training
//! objectives that existing metrics (isotropy, uniformity, entropy) treat only
//! indirectly, since they ignore spatial arrangement.

use crate::analysis::finite::{ensure_finite_2d, square_grid_side};
use crate::errors::AnalysisError;
use ndarray::Array2;

/// Compute the spatial coherence score of patch embeddings arranged on a grid.
///
/// For each patch, computes the cosine similarity to its immediate spatial
/// neighbors (up, down, left, right — 4-connected grid) and returns the
/// mean across all neighbor pairs. The patch tokens must form a perfect
/// square grid (e.g., 256 patches = 16×16).
///
/// # Returns
///
/// A score in `[-1, 1]` where:
/// - **~1.0** — adjacent patches are nearly identical (strong spatial smoothness)
/// - **~0.0** — no spatial correlation (patches are spatially independent)
/// - **< 0** — anti-correlated neighbors (rare in practice)
///
/// # Errors
///
/// - `AnalysisError::InsufficientData` if fewer than 4 patches
/// - `AnalysisError::InvalidPatchGrid` if patch count is not a perfect square
/// - `AnalysisError::NonFiniteValues` if embeddings contain NaN/Inf
pub fn spatial_coherence(patches: &Array2<f32>) -> Result<f32, AnalysisError> {
    let n = patches.shape()[0];
    if n < 4 {
        return Err(AnalysisError::InsufficientData(format!(
            "Spatial coherence requires at least 4 patches, got {n}"
        )));
    }
    ensure_finite_2d(patches, "patches for spatial coherence")?;

    let grid = square_grid_side(n, "spatial coherence")?;

    // Precompute L2 norms for cosine similarity
    let norms: Vec<f32> = patches
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10))
        .collect();

    let mut total_sim = 0.0_f64;
    let mut pair_count = 0_u64;

    // 4-connected neighbors: right and down only (each pair counted once)
    for row in 0..grid {
        for col in 0..grid {
            let idx = row * grid + col;

            // Right neighbor
            if col + 1 < grid {
                let neighbor = row * grid + (col + 1);
                let sim = cosine_sim_rows(patches, idx, neighbor, &norms);
                total_sim += sim as f64;
                pair_count += 1;
            }

            // Down neighbor
            if row + 1 < grid {
                let neighbor = (row + 1) * grid + col;
                let sim = cosine_sim_rows(patches, idx, neighbor, &norms);
                total_sim += sim as f64;
                pair_count += 1;
            }
        }
    }

    if pair_count == 0 {
        return Ok(0.0);
    }

    Ok((total_sim / pair_count as f64) as f32)
}

/// Compute per-patch spatial coherence: the mean cosine similarity of each
/// patch with its immediate 4-connected spatial neighbors.
///
/// Returns a 1-D array of length `n_patches`. Useful for generating spatial
/// coherence heatmaps over the image.
///
/// # Errors
///
/// Same as [`spatial_coherence`].
pub fn spatial_coherence_map(patches: &Array2<f32>) -> Result<Vec<f32>, AnalysisError> {
    let n = patches.shape()[0];
    if n < 4 {
        return Err(AnalysisError::InsufficientData(format!(
            "Spatial coherence map requires at least 4 patches, got {n}"
        )));
    }
    ensure_finite_2d(patches, "patches for spatial coherence map")?;

    let grid = square_grid_side(n, "spatial coherence map")?;

    let norms: Vec<f32> = patches
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10))
        .collect();

    let mut per_patch = vec![0.0_f32; n];

    for row in 0..grid {
        for col in 0..grid {
            let idx = row * grid + col;
            let mut sum = 0.0_f64;
            let mut count = 0_u32;

            // Up
            if row > 0 {
                let neighbor = (row - 1) * grid + col;
                sum += cosine_sim_rows(patches, idx, neighbor, &norms) as f64;
                count += 1;
            }
            // Down
            if row + 1 < grid {
                let neighbor = (row + 1) * grid + col;
                sum += cosine_sim_rows(patches, idx, neighbor, &norms) as f64;
                count += 1;
            }
            // Left
            if col > 0 {
                let neighbor = row * grid + (col - 1);
                sum += cosine_sim_rows(patches, idx, neighbor, &norms) as f64;
                count += 1;
            }
            // Right
            if col + 1 < grid {
                let neighbor = row * grid + (col + 1);
                sum += cosine_sim_rows(patches, idx, neighbor, &norms) as f64;
                count += 1;
            }

            per_patch[idx] = if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            };
        }
    }

    Ok(per_patch)
}

/// Cosine similarity between two rows of a matrix using precomputed norms.
#[inline]
fn cosine_sim_rows(data: &Array2<f32>, i: usize, j: usize, norms: &[f32]) -> f32 {
    let row_i = data.row(i);
    let row_j = data.row(j);
    let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
    (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Identical patches should have coherence of 1.0.
    #[test]
    fn identical_patches_have_perfect_coherence() {
        // 4x4 grid, all patches identical
        let patches = Array2::from_shape_fn((16, 32), |(_i, j)| j as f32 + 1.0);
        let score = spatial_coherence(&patches).unwrap();
        approx::assert_relative_eq!(score, 1.0, epsilon = 1e-5);
    }

    /// Orthogonal neighboring patches should have coherence near 0.
    #[test]
    fn orthogonal_patches_have_zero_coherence() {
        // 4x4 grid where each patch is a different one-hot basis vector
        let dim = 16;
        let mut patches = Array2::zeros((16, dim));
        for i in 0..16 {
            patches[[i, i]] = 1.0;
        }
        let score = spatial_coherence(&patches).unwrap();
        approx::assert_abs_diff_eq!(score, 0.0, epsilon = 1e-5);
    }

    /// Smooth gradient across the grid should produce positive coherence.
    #[test]
    fn smooth_gradient_has_positive_coherence() {
        // 4x4 grid with embeddings that vary smoothly
        let patches = Array2::from_shape_fn((16, 8), |(i, j)| {
            let row = i / 4;
            let col = i % 4;
            (row as f32 * 0.1 + col as f32 * 0.1 + j as f32).sin()
        });
        let score = spatial_coherence(&patches).unwrap();
        assert!(
            score > 0.5,
            "Smooth gradient should have high coherence, got {score}"
        );
    }

    /// Non-square patch count should error.
    #[test]
    fn non_square_patch_count_errors() {
        let patches = Array2::zeros((15, 8));
        let result = spatial_coherence(&patches);
        assert!(result.is_err());
    }

    /// Fewer than 4 patches should error.
    #[test]
    fn too_few_patches_errors() {
        let patches = Array2::zeros((2, 8));
        let result = spatial_coherence(&patches);
        assert!(matches!(result, Err(AnalysisError::InsufficientData(_))));
    }

    /// Per-patch coherence map has correct length.
    #[test]
    fn coherence_map_correct_length() {
        let patches = Array2::from_shape_fn((16, 8), |(i, j)| (i + j) as f32);
        let map = spatial_coherence_map(&patches).unwrap();
        assert_eq!(map.len(), 16);
    }

    /// Corner patches have fewer neighbors than interior patches.
    #[test]
    fn corner_vs_interior_neighbor_counts() {
        // All-ones patches: every cosine similarity is 1.0, so all per-patch values should be 1.0
        let patches = Array2::from_shape_fn((16, 8), |(_i, j)| j as f32 + 1.0);
        let map = spatial_coherence_map(&patches).unwrap();
        for &val in &map {
            approx::assert_relative_eq!(val, 1.0, epsilon = 1e-5);
        }
    }

    /// Global and per-patch coherence are consistent.
    #[test]
    fn global_matches_per_patch_average() {
        let patches = Array2::from_shape_fn((16, 8), |(i, j)| ((i * 7 + j * 13) as f32).sin());
        let global = spatial_coherence(&patches).unwrap();
        let map = spatial_coherence_map(&patches).unwrap();

        // The global score is the mean of unique adjacent pairs,
        // while per-patch averages each patch's neighbors (double-counting pairs).
        // They should be close but not necessarily identical.
        let per_patch_mean: f32 = map.iter().sum::<f32>() / map.len() as f32;
        approx::assert_abs_diff_eq!(global, per_patch_mean, epsilon = 0.05);
    }
}
