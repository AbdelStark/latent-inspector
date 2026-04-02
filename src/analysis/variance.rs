//! Feature variance spectrum: distribution of information across PCA components.

use crate::analysis::pca;
use crate::errors::AnalysisError;
use ndarray::{s, Array1, Array2};

/// Variance spectrum result.
#[derive(Debug, Clone)]
pub struct VarianceSpectrum {
    /// Raw explained variance (eigenvalues) for each retained component.
    pub explained_variance: Array1<f32>,
    /// Explained variance ratio for each component (sorted descending), length `k`.
    pub ratios: Array1<f32>,
    /// Cumulative explained variance, length `k`.
    pub cumulative: Array1<f32>,
    /// Index at which cumulative variance first exceeds 0.90.
    pub components_90pct: usize,
    /// Index at which cumulative variance first exceeds 0.99.
    pub components_99pct: usize,
    /// Concentration of variance in top-10 components (fraction of total variance).
    pub top10_concentration: f32,
}

impl VarianceSpectrum {
    /// Return a new spectrum keeping only the first `k` components, recomputing
    /// cumulative ratios and concentration fields.
    pub fn truncated(&self, k: usize) -> Self {
        let k = k.max(1).min(self.ratios.len());
        let explained_variance = self.explained_variance.slice(s![..k]).to_owned();
        let ratios = self.ratios.slice(s![..k]).to_owned();
        let mut cumulative = Array1::<f32>::zeros(k);
        let mut running = 0.0_f32;
        for (index, ratio) in ratios.iter().enumerate() {
            running += *ratio;
            cumulative[index] = running;
        }

        let components_90pct = cumulative
            .iter()
            .position(|&c| c >= 0.90)
            .map(|i| i + 1)
            .unwrap_or(k);

        let components_99pct = cumulative
            .iter()
            .position(|&c| c >= 0.99)
            .map(|i| i + 1)
            .unwrap_or(k);

        let top10_concentration = ratios.iter().take(10).sum();

        Self {
            explained_variance,
            ratios,
            cumulative,
            components_90pct,
            components_99pct,
            top10_concentration,
        }
    }
}

/// Compute the variance spectrum of `data` `[N, D]` using `k` PCA components.
pub fn variance_spectrum(data: &Array2<f32>, k: usize) -> Result<VarianceSpectrum, AnalysisError> {
    let result = pca::pca(data, k, 500)?;
    let explained_variance = result.explained_variance;
    let ratios = result.explained_variance_ratio;

    let mut cumulative = Array1::<f32>::zeros(ratios.len());
    let mut cum = 0.0_f32;
    for (i, &r) in ratios.iter().enumerate() {
        cum += r;
        cumulative[i] = cum;
    }

    let components_90pct = cumulative
        .iter()
        .position(|&c| c >= 0.90)
        .map(|i| i + 1)
        .unwrap_or(ratios.len());

    let components_99pct = cumulative
        .iter()
        .position(|&c| c >= 0.99)
        .map(|i| i + 1)
        .unwrap_or(ratios.len());

    let top10_concentration = ratios.iter().take(10).sum();

    Ok(VarianceSpectrum {
        explained_variance,
        ratios,
        cumulative,
        components_90pct,
        components_99pct,
        top10_concentration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_shape() {
        let data = Array2::from_shape_fn((50, 16), |(i, j)| (i + j) as f32);
        let spec = variance_spectrum(&data, 8).unwrap();
        assert_eq!(spec.ratios.len(), 8);
        assert_eq!(spec.cumulative.len(), 8);
    }

    #[test]
    fn test_cumulative_monotone() {
        let data = Array2::from_shape_fn((40, 8), |(i, j)| (i * 2 + j) as f32);
        let spec = variance_spectrum(&data, 6).unwrap();
        for i in 1..spec.cumulative.len() {
            assert!(spec.cumulative[i] >= spec.cumulative[i - 1] - 1e-5);
        }
    }

    #[test]
    fn test_top10_concentration_range() {
        let data = Array2::from_shape_fn((60, 16), |(i, j)| (i + j) as f32);
        let spec = variance_spectrum(&data, 16).unwrap();
        assert!(spec.top10_concentration >= 0.0);
        assert!(spec.top10_concentration <= 1.0 + 1e-5);
    }

    #[test]
    fn truncation_recomputes_cumulative_fields() {
        let data = Array2::from_shape_fn((60, 16), |(i, j)| (i + j * 2) as f32);
        let spec = variance_spectrum(&data, 12).unwrap();
        let truncated = spec.truncated(5);

        assert_eq!(truncated.explained_variance.len(), 5);
        assert_eq!(truncated.ratios.len(), 5);
        assert_eq!(truncated.cumulative.len(), 5);
        assert!(truncated.cumulative[4] <= 1.0 + 1e-5);
    }
}
