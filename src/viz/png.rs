//! PNG export: attention overlays, PCA RGB projections, heatmaps.

use crate::errors::VizError;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::Array2;
use std::path::Path;

/// Normalise an array to `[0, 1]`.
fn normalize(data: &Array2<f32>) -> Array2<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.mapv(|v| (v - min) / range)
}

/// Map a value in `[0, 1]` to a heatmap colour (blue→green→red).
fn heatmap_color(v: f32) -> [u8; 3] {
    let v = v.clamp(0.0, 1.0);
    let r = (v * 2.0 - 1.0).max(0.0) * 255.0;
    let g = (1.0 - (v * 2.0 - 1.0).abs()) * 255.0;
    let b = (1.0 - v * 2.0).max(0.0) * 255.0;
    [r as u8, g as u8, b as u8]
}

/// Save a 2-D attention map `[H_grid, W_grid]` overlaid on the original image.
pub fn save_attention_overlay(
    original: &DynamicImage,
    attention_map: &Array2<f32>,
    output_path: &Path,
    alpha: f32,
) -> Result<(), VizError> {
    let (ow, oh) = (original.width(), original.height());
    let rgb = original.to_rgb8();
    let (ah, aw) = (attention_map.shape()[0], attention_map.shape()[1]);

    let norm = normalize(attention_map);
    let mut out: RgbImage = ImageBuffer::new(ow, oh);

    for py in 0..oh {
        for px in 0..ow {
            let ax = (px as f32 / ow as f32 * aw as f32) as usize;
            let ay = (py as f32 / oh as f32 * ah as f32) as usize;
            let attn_val = norm[[ay.min(ah - 1), ax.min(aw - 1)]];
            let heat = heatmap_color(attn_val);

            let orig = rgb.get_pixel(px, py);
            let r = (orig[0] as f32 * (1.0 - alpha) + heat[0] as f32 * alpha) as u8;
            let g = (orig[1] as f32 * (1.0 - alpha) + heat[1] as f32 * alpha) as u8;
            let b = (orig[2] as f32 * (1.0 - alpha) + heat[2] as f32 * alpha) as u8;
            out.put_pixel(px, py, Rgb([r, g, b]));
        }
    }

    out.save(output_path)
        .map_err(|e| VizError::Png(format!("Failed to save {}: {e}", output_path.display())))?;
    Ok(())
}

/// Save PCA 3-component projection as an RGB image.
///
/// `projections`: `[N_patches, 3]` — first 3 PCA components per patch.
/// `grid_size`: number of patches along each axis (assumes square grid).
pub fn save_pca_rgb(
    projections: &Array2<f32>,
    grid_size: usize,
    output_path: &Path,
) -> Result<(), VizError> {
    let n = projections.shape()[0];
    if n < grid_size * grid_size {
        return Err(VizError::Png(format!(
            "Expected {}×{}={} patches, got {}",
            grid_size,
            grid_size,
            grid_size * grid_size,
            n
        )));
    }

    // Normalize each channel independently to [0, 255]
    let make_channel = |c: usize| -> Vec<u8> {
        let vals: Vec<f32> = (0..grid_size * grid_size)
            .map(|i| projections[[i, c]])
            .collect();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        vals.iter()
            .map(|&v| ((v - min) / range * 255.0) as u8)
            .collect()
    };

    let r_ch = make_channel(0);
    let g_ch = make_channel(1);
    let b_ch = make_channel(2.min(projections.shape()[1] - 1));

    let mut img: RgbImage = ImageBuffer::new(grid_size as u32, grid_size as u32);
    for (i, pixel) in img.pixels_mut().enumerate() {
        *pixel = Rgb([r_ch[i], g_ch[i], b_ch[i]]);
    }

    img.save(output_path)
        .map_err(|e| VizError::Png(format!("Failed to save {}: {e}", output_path.display())))?;
    Ok(())
}

/// Save a square similarity matrix `[N, N]` as a heatmap PNG.
pub fn save_similarity_heatmap(matrix: &Array2<f32>, output_path: &Path) -> Result<(), VizError> {
    let n = matrix.shape()[0];
    let norm = normalize(matrix);
    let mut img: RgbImage = ImageBuffer::new(n as u32, n as u32);

    for (i, pixel) in img.pixels_mut().enumerate() {
        let row = i / n;
        let col = i % n;
        let c = heatmap_color(norm[[row, col]]);
        *pixel = Rgb(c);
    }

    img.save(output_path)
        .map_err(|e| VizError::Png(format!("Failed to save {}: {e}", output_path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_heatmap_color_clamps_to_expected_extremes() {
        assert_eq!(heatmap_color(-1.0), [0, 0, 255]);
        assert_eq!(heatmap_color(0.5), [0, 255, 0]);
        assert_eq!(heatmap_color(2.0), [255, 0, 0]);
    }

    #[test]
    fn test_save_similarity_heatmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sim.png");
        let matrix = Array2::from_shape_fn((8, 8), |(i, j)| if i == j { 1.0 } else { 0.0 });
        save_similarity_heatmap(&matrix, &path).unwrap();
        assert!(path.exists());
    }
}
