//! PNG export: attention overlays, PCA RGB projections, heatmaps.

use crate::errors::VizError;
use crate::viz::report::PairwiseMatrix;
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

/// Minimum output dimension for PCA RGB images. Each patch is upscaled so the
/// final image is at least this many pixels on each side.
const PCA_MIN_OUTPUT_SIZE: u32 = 448;

/// Save PCA 3-component projection as an RGB image.
///
/// Each patch's top-3 PCA components are mapped to RGB channels and painted
/// over its spatial region, producing a full-resolution visualization where
/// same-color regions have similar representations. This is the standard
/// ViT patch embedding visualization used in DINOv2 and related papers.
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
            "Expected {}x{}={} patches, got {}",
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

    // Upscale: each patch covers (scale x scale) pixels so the output is
    // at least PCA_MIN_OUTPUT_SIZE on each side.
    let scale = (PCA_MIN_OUTPUT_SIZE as usize / grid_size).max(1);
    let out_size = (grid_size * scale) as u32;

    let mut img: RgbImage = ImageBuffer::new(out_size, out_size);
    for py in 0..out_size {
        for px in 0..out_size {
            let gx = (px as usize / scale).min(grid_size - 1);
            let gy = (py as usize / scale).min(grid_size - 1);
            let idx = gy * grid_size + gx;
            img.put_pixel(px, py, Rgb([r_ch[idx], g_ch[idx], b_ch[idx]]));
        }
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

pub fn save_pairwise_heatmap(matrix: &PairwiseMatrix, output_path: &Path) -> Result<(), VizError> {
    if matrix.is_empty() {
        return Err(VizError::Png(
            "Cannot render an empty pairwise heatmap".to_string(),
        ));
    }

    let present_values = matrix
        .rows
        .iter()
        .flat_map(|row| row.iter().flatten().copied())
        .collect::<Vec<_>>();
    let min = present_values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(1.0);
    let max = present_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);
    let range = (max - min).max(1e-8);
    let cell_size = 48u32;
    let gap = 2u32;
    let n = matrix.len() as u32;
    let width = n * cell_size + n.saturating_sub(1) * gap;
    let height = width;
    let mut img = RgbImage::from_pixel(width, height, Rgb([18, 23, 33]));

    for (row_idx, row) in matrix.rows.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            let color = value
                .map(|value| heatmap_color((value - min) / range))
                .unwrap_or([72, 78, 88]);
            let x0 = col_idx as u32 * (cell_size + gap);
            let y0 = row_idx as u32 * (cell_size + gap);

            for y in y0..(y0 + cell_size) {
                for x in x0..(x0 + cell_size) {
                    img.put_pixel(x, y, Rgb(color));
                }
            }
        }
    }

    img.save(output_path)
        .map_err(|e| VizError::Png(format!("Failed to save {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn save_series_chart(values: &[f32], output_path: &Path) -> Result<(), VizError> {
    let values = if values.is_empty() {
        vec![0.0]
    } else {
        values.to_vec()
    };

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    let bar_width = 22u32;
    let gap = 8u32;
    let padding = 16u32;
    let chart_height = 180u32;
    let width =
        padding * 2 + values.len() as u32 * bar_width + values.len().saturating_sub(1) as u32 * gap;
    let height = chart_height + padding * 2;
    let mut img = RgbImage::from_pixel(width, height, Rgb([13, 17, 23]));

    for (index, value) in values.iter().enumerate() {
        let normalized = ((*value - min) / range).clamp(0.0, 1.0);
        let bar_height = (normalized * chart_height as f32).round() as u32;
        let x0 = padding + index as u32 * (bar_width + gap);
        let y0 = padding + chart_height.saturating_sub(bar_height);
        let color = heatmap_color((index as f32 + 1.0) / values.len() as f32);

        for y in y0..(padding + chart_height) {
            for x in x0..(x0 + bar_width) {
                img.put_pixel(x, y, Rgb(color));
            }
        }
    }

    img.save(output_path)
        .map_err(|e| VizError::Png(format!("Failed to save {}: {e}", output_path.display())))?;
    Ok(())
}

pub fn save_variance_spectrum_chart(ratios: &[f32], output_path: &Path) -> Result<(), VizError> {
    if ratios.is_empty() {
        return Err(VizError::Png(
            "Cannot render a variance spectrum without ratios".to_string(),
        ));
    }

    let bar_width = 18u32;
    let gap = 6u32;
    let padding = 16u32;
    let chart_height = 180u32;
    let width =
        padding * 2 + ratios.len() as u32 * bar_width + ratios.len().saturating_sub(1) as u32 * gap;
    let height = chart_height + padding * 2;
    let mut img = RgbImage::from_pixel(width, height, Rgb([13, 17, 23]));

    for (index, ratio) in ratios.iter().enumerate() {
        let bar_height = (ratio.clamp(0.0, 1.0) * chart_height as f32).round() as u32;
        let x0 = padding + index as u32 * (bar_width + gap);
        let y0 = padding + chart_height.saturating_sub(bar_height);
        let color = heatmap_color((index as f32 + 1.0) / ratios.len() as f32);

        for y in y0..(padding + chart_height) {
            for x in x0..(x0 + bar_width) {
                img.put_pixel(x, y, Rgb(color));
            }
        }
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

    #[test]
    fn test_save_pairwise_heatmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pairwise.png");
        let matrix = PairwiseMatrix {
            labels: vec!["a".into(), "b".into()],
            rows: vec![vec![Some(1.0), Some(0.5)], vec![Some(0.5), Some(1.0)]],
        };

        save_pairwise_heatmap(&matrix, &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_save_variance_spectrum_chart() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("variance.png");

        save_variance_spectrum_chart(&[0.4, 0.25, 0.2, 0.1], &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_save_series_chart() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("series.png");

        save_series_chart(&[0.2, 0.6, 0.4], &path).unwrap();
        assert!(path.exists());
    }
}
