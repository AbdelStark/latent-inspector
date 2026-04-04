//! RankMe: smooth effective rank via Shannon entropy of normalized singular values.
//!
//! Reference: Garrido et al., "RankMe: Assessing the Downstream Performance of
//! Pretrained Self-Supervised Representations by Their Rank" (ICML 2023).
//!
//! Unlike threshold-based effective rank, RankMe provides a smooth, continuous
//! measure of the effective dimensionality by computing `exp(H(p))` where `H(p)`
//! is the Shannon entropy of the normalized singular value distribution.

use crate::analysis::variance::VarianceSpectrum;

/// Compute RankMe from a pre-computed variance spectrum.
///
/// RankMe = exp(H(p)) where p_i = sigma_i / sum(sigma_j) and H is Shannon entropy.
/// The singular values are derived from the PCA eigenvalues (sigma = sqrt(eigenvalue)).
///
/// Returns a value in `[1, k]` where `k` is the number of retained components.
/// A value of 1 means total collapse (all variance in one dimension).
/// A value near `k` means variance is spread uniformly across all dimensions.
pub fn rankme_from_spectrum(spectrum: &VarianceSpectrum) -> f32 {
    let eigenvalues = &spectrum.explained_variance;
    if eigenvalues.is_empty() {
        return 0.0;
    }

    // Compute singular values from eigenvalues (sigma = sqrt(lambda))
    // Filter out non-positive eigenvalues to avoid NaN from sqrt
    let singular_values: Vec<f32> = eigenvalues
        .iter()
        .filter(|&&ev| ev > 0.0)
        .map(|&ev| ev.sqrt())
        .collect();

    if singular_values.is_empty() {
        return 0.0;
    }

    let total: f32 = singular_values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    // Normalize to a probability distribution and compute Shannon entropy
    let entropy: f32 = singular_values
        .iter()
        .map(|&s| {
            let p = s / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    // RankMe = exp(entropy)
    entropy.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn spectrum_from_eigenvalues(eigenvalues: &[f32]) -> VarianceSpectrum {
        let ev = Array1::from_vec(eigenvalues.to_vec());
        let total: f32 = ev.iter().sum();
        let ratios = if total > 0.0 {
            ev.mapv(|v| v / total)
        } else {
            Array1::zeros(ev.len())
        };
        let mut cumulative = Array1::zeros(ev.len());
        let mut cum = 0.0_f32;
        for (i, &r) in ratios.iter().enumerate() {
            cum += r;
            cumulative[i] = cum;
        }
        VarianceSpectrum {
            explained_variance: ev,
            ratios,
            cumulative,
            components_90pct: eigenvalues.len(),
            components_99pct: eigenvalues.len(),
            top10_concentration: 1.0,
        }
    }

    #[test]
    fn uniform_eigenvalues_give_max_rankme() {
        // All eigenvalues equal → RankMe should equal k
        let spec = spectrum_from_eigenvalues(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let rm = rankme_from_spectrum(&spec);
        approx::assert_relative_eq!(rm, 5.0, epsilon = 1e-4);
    }

    #[test]
    fn single_dominant_eigenvalue_gives_rankme_near_one() {
        // One huge eigenvalue, rest negligible → RankMe ≈ 1
        let spec = spectrum_from_eigenvalues(&[1000.0, 0.001, 0.001, 0.001]);
        let rm = rankme_from_spectrum(&spec);
        assert!(rm >= 1.0);
        assert!(rm < 1.5);
    }

    #[test]
    fn empty_spectrum_returns_zero() {
        let spec = spectrum_from_eigenvalues(&[]);
        assert_eq!(rankme_from_spectrum(&spec), 0.0);
    }

    #[test]
    fn all_zero_eigenvalues_returns_zero() {
        let spec = spectrum_from_eigenvalues(&[0.0, 0.0, 0.0]);
        assert_eq!(rankme_from_spectrum(&spec), 0.0);
    }

    #[test]
    fn rankme_is_between_one_and_k() {
        let spec = spectrum_from_eigenvalues(&[10.0, 5.0, 2.0, 1.0, 0.5, 0.1]);
        let rm = rankme_from_spectrum(&spec);
        assert!(rm >= 1.0);
        assert!(rm <= 6.0);
    }

    #[test]
    fn two_equal_eigenvalues_give_rankme_two() {
        let spec = spectrum_from_eigenvalues(&[4.0, 4.0]);
        let rm = rankme_from_spectrum(&spec);
        approx::assert_relative_eq!(rm, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn negative_eigenvalues_are_filtered() {
        // Numerical noise can produce negative eigenvalues
        let spec = spectrum_from_eigenvalues(&[10.0, 5.0, -0.01, -0.1]);
        let rm = rankme_from_spectrum(&spec);
        assert!(rm.is_finite());
        assert!(rm >= 1.0);
    }
}
