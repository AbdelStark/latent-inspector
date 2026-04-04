//! Spectral decay rate: characterizes how quickly eigenvalue magnitude drops.
//!
//! Fits a power-law decay `lambda_i ~ i^(-beta)` to the PCA eigenvalue spectrum
//! via linear regression in log-log space. A higher beta indicates faster decay
//! (more concentrated representations), while a lower beta indicates slower decay
//! (variance spread more uniformly).
//!
//! This metric complements RankMe by providing a shape descriptor: RankMe tells
//! you *how many* effective dimensions exist, spectral decay tells you *how
//! sharply* the representation transitions from dominant to minor dimensions.

use crate::analysis::variance::VarianceSpectrum;

/// Compute the spectral decay exponent (beta) from a variance spectrum.
///
/// Fits `log(lambda_i) = -beta * log(i) + c` via least-squares linear regression
/// on the positive eigenvalues. Returns `None` if fewer than 2 positive eigenvalues
/// exist (regression requires at least 2 points).
///
/// Typical ranges:
/// - beta < 1.0: Slow decay, representation uses many dimensions evenly
/// - beta ~ 1.0-2.0: Moderate decay, typical for well-trained SSL models
/// - beta > 2.0: Rapid decay, representation dominated by few dimensions
pub fn spectral_decay_from_spectrum(spectrum: &VarianceSpectrum) -> Option<f32> {
    let eigenvalues = &spectrum.explained_variance;

    // Collect (log(index), log(eigenvalue)) pairs for positive eigenvalues
    let points: Vec<(f32, f32)> = eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, &ev)| ev > 0.0)
        .map(|(i, &ev)| ((i as f32 + 1.0).ln(), ev.ln()))
        .collect();

    if points.len() < 2 {
        return None;
    }

    // Least-squares linear regression: y = slope * x + intercept
    // where x = log(index), y = log(eigenvalue)
    // slope = -beta (negative because eigenvalues decrease with index)
    let n = points.len() as f32;
    let sum_x: f32 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f32 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f32 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f32 = points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // beta = -slope (power law decay exponent is positive)
    let beta = -slope;
    Some(beta)
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
    fn exact_power_law_recovers_exponent() {
        // lambda_i = i^(-2) for i=1..20
        let eigenvalues: Vec<f32> = (1..=20).map(|i| (i as f32).powi(-2)).collect();
        let spec = spectrum_from_eigenvalues(&eigenvalues);
        let beta = spectral_decay_from_spectrum(&spec).unwrap();
        approx::assert_relative_eq!(beta, 2.0, epsilon = 0.01);
    }

    #[test]
    fn uniform_eigenvalues_give_zero_decay() {
        let spec = spectrum_from_eigenvalues(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let beta = spectral_decay_from_spectrum(&spec).unwrap();
        approx::assert_abs_diff_eq!(beta, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn fewer_than_two_points_returns_none() {
        let spec = spectrum_from_eigenvalues(&[5.0]);
        assert!(spectral_decay_from_spectrum(&spec).is_none());
    }

    #[test]
    fn empty_spectrum_returns_none() {
        let spec = spectrum_from_eigenvalues(&[]);
        assert!(spectral_decay_from_spectrum(&spec).is_none());
    }

    #[test]
    fn all_zero_returns_none() {
        let spec = spectrum_from_eigenvalues(&[0.0, 0.0, 0.0]);
        assert!(spectral_decay_from_spectrum(&spec).is_none());
    }

    #[test]
    fn steep_decay_gives_high_beta() {
        // lambda_i = i^(-4)
        let eigenvalues: Vec<f32> = (1..=10).map(|i| (i as f32).powi(-4)).collect();
        let spec = spectrum_from_eigenvalues(&eigenvalues);
        let beta = spectral_decay_from_spectrum(&spec).unwrap();
        assert!(beta > 3.5, "Expected beta > 3.5, got {beta}");
    }

    #[test]
    fn moderate_decay_gives_moderate_beta() {
        // lambda_i = i^(-1)
        let eigenvalues: Vec<f32> = (1..=10).map(|i| 1.0 / i as f32).collect();
        let spec = spectrum_from_eigenvalues(&eigenvalues);
        let beta = spectral_decay_from_spectrum(&spec).unwrap();
        approx::assert_relative_eq!(beta, 1.0, epsilon = 0.01);
    }

    #[test]
    fn negative_eigenvalues_are_filtered() {
        let spec = spectrum_from_eigenvalues(&[10.0, 5.0, 2.0, -0.01, -0.5]);
        let beta = spectral_decay_from_spectrum(&spec);
        assert!(beta.is_some());
        assert!(beta.unwrap().is_finite());
    }
}
