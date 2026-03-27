use crate::errors::ModelError;
use image::{DynamicImage, RgbImage};
use ndarray::Array4;

/// Preprocessing parameters for a specific model.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub input_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PreprocessConfig {
    pub fn new(input_size: u32, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            input_size,
            mean,
            std,
        }
    }
}

/// Resize an image to `(input_size, input_size)` and normalize it into a
/// `[1, 3, H, W]` float32 tensor using the given mean/std.
pub fn preprocess(img: &DynamicImage, cfg: &PreprocessConfig) -> Result<Array4<f32>, ModelError> {
    if cfg.input_size == 0 {
        return Err(ModelError::Preprocessing(
            "input_size must be greater than zero".to_string(),
        ));
    }
    if cfg.std.iter().any(|value| value.abs() <= f32::EPSILON) {
        return Err(ModelError::Preprocessing(
            "normalization std must be non-zero for every channel".to_string(),
        ));
    }

    let size = cfg.input_size;

    // Resize to square
    let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
    let rgb: RgbImage = resized.to_rgb8();

    let h = size as usize;
    let w = size as usize;

    // Build [1, 3, H, W] tensor
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let raw = pixel[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (raw - cfg.mean[c]) / cfg.std[c];
            }
        }
    }

    Ok(tensor)
}

/// Load an image from disk.
pub fn load_image(path: &std::path::Path) -> Result<DynamicImage, ModelError> {
    image::open(path).map_err(|e| {
        ModelError::Preprocessing(format!("Failed to open image {}: {}", path.display(), e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_shape() {
        let img = DynamicImage::new_rgb8(400, 300);
        let cfg = PreprocessConfig::new(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        let tensor = preprocess(&img, &cfg).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_preprocess_normalization() {
        // White image (255,255,255) → normalized value should be (1.0 - mean) / std
        let img = DynamicImage::new_rgb8(32, 32);
        // new_rgb8 creates a black image
        let cfg = PreprocessConfig::new(32, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let tensor = preprocess(&img, &cfg).unwrap();
        // Black pixels → 0.0 normalized
        assert!((tensor[[0, 0, 0, 0]] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_preprocess_rejects_zero_std() {
        let img = DynamicImage::new_rgb8(32, 32);
        let cfg = PreprocessConfig::new(32, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]);
        let err = preprocess(&img, &cfg).unwrap_err();
        assert!(err.to_string().contains("non-zero"));
    }
}
