//! Centered Kernel Alignment (CKA) for comparing representations.

use crate::errors::AnalysisError;
use ndarray::Array2;

/// Linear kernel: K = X * X^T.
fn linear_kernel(x: &Array2<f32>) -> Array2<f32> {
    x.dot(&x.t())
}

/// Centre a kernel matrix: K_c = H * K * H, where H = I - (1/n)*11^T.
fn centre_kernel(k: &Array2<f32>) -> Array2<f32> {
    let n = k.shape()[0] as f32;
    let row_mean = k.mean_axis(ndarray::Axis(0)).unwrap();
    let grand_mean = row_mean.mean().unwrap();

    let mut kc = k.clone();
    for mut row in kc.rows_mut() {
        row -= &row_mean;
    }
    for (j, mut col) in kc.columns_mut().into_iter().enumerate() {
        col -= &(ndarray::Array1::from_elem(col.len(), row_mean[j]));
    }
    kc += grand_mean;
    // Scale by 1/n (optional: HSIC uses 1/(n-1)^2, so we keep raw HSIC)
    let _ = n;
    kc
}

/// HSIC (Hilbert-Schmidt Independence Criterion): trace(Kc * Lc) / (n-1)^2.
fn hsic(kc: &Array2<f32>, lc: &Array2<f32>) -> f32 {
    let n = kc.shape()[0];
    if n < 2 {
        return 0.0;
    }
    // trace(Kc * Lc) = sum_{ij} Kc[i,j] * Lc[i,j]   (element-wise then sum)
    let trace: f32 = kc.iter().zip(lc.iter()).map(|(a, b)| a * b).sum();
    trace / ((n - 1) * (n - 1)) as f32
}

/// Linear CKA between representations `x` `[N, D1]` and `y` `[N, D2]`.
///
/// Returns a value in `[0, 1]`: 1 = identical representation geometry.
pub fn linear_cka(x: &Array2<f32>, y: &Array2<f32>) -> Result<f32, AnalysisError> {
    let nx = x.shape()[0];
    let ny = y.shape()[0];

    if nx != ny {
        return Err(AnalysisError::ShapeMismatch {
            expected: vec![nx, x.shape()[1]],
            actual: vec![ny, y.shape()[1]],
        });
    }
    if nx < 2 {
        return Err(AnalysisError::InsufficientData(format!(
            "CKA requires ≥2 samples, got {nx}"
        )));
    }

    let kx = linear_kernel(x);
    let ky = linear_kernel(y);

    let kxc = centre_kernel(&kx);
    let kyc = centre_kernel(&ky);

    let hsic_xy = hsic(&kxc, &kyc);
    let hsic_xx = hsic(&kxc, &kxc);
    let hsic_yy = hsic(&kyc, &kyc);

    let denom = (hsic_xx * hsic_yy).sqrt();
    if denom < 1e-10 {
        return Ok(0.0);
    }

    Ok((hsic_xy / denom).clamp(0.0, 1.0))
}

/// CLS cosine similarity between two CLS vectors.
pub fn cls_cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cka_identical() {
        let x = Array2::from_shape_fn((20, 8), |(i, j)| (i + j) as f32);
        let cka = linear_cka(&x, &x).unwrap();
        assert_relative_eq!(cka, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_cka_range() {
        let x = Array2::from_shape_fn((20, 8), |(i, j)| (i + j) as f32);
        let y = Array2::from_shape_fn((20, 8), |(i, j)| (i * j) as f32);
        let cka = linear_cka(&x, &y).unwrap();
        assert!(cka >= 0.0 && cka <= 1.0 + 1e-5);
    }

    #[test]
    fn test_cka_shape_mismatch() {
        let x = Array2::from_elem((10, 4), 1.0_f32);
        let y = Array2::from_elem((12, 4), 1.0_f32);
        assert!(linear_cka(&x, &y).is_err());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = ndarray::array![1.0f32, 2.0, 3.0];
        let b = ndarray::array![1.0f32, 2.0, 3.0];
        assert_relative_eq!(cls_cosine_similarity(&a, &b), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = ndarray::array![1.0f32, 0.0];
        let b = ndarray::array![0.0f32, 1.0];
        assert_relative_eq!(cls_cosine_similarity(&a, &b), 0.0, epsilon = 1e-5);
    }
}
