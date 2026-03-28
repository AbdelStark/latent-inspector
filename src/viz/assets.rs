use crate::errors::VizError;
use crate::viz::html::VisualAsset;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use std::path::Path;

const PREVIEW_MAX_DIMENSION: u32 = 384;

pub fn slugify_filename(label: &str) -> String {
    let mut slug = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if matches!(ch, '-' | '_') {
            slug.push(ch);
        } else {
            slug.push('_');
        }
    }
    slug.trim_matches('_').to_string()
}

pub fn visual_asset(
    path: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
) -> VisualAsset {
    let title = title.into();
    VisualAsset {
        path: path.into(),
        alt: title.clone(),
        title,
        description: description.into(),
    }
}

pub fn write_preview_image(
    image: &DynamicImage,
    outdir: &Path,
    filename: &str,
    title: impl Into<String>,
    description: impl Into<String>,
) -> Result<VisualAsset, VizError> {
    let preview = resized_preview(image);
    let path = outdir.join(filename);
    preview.save(&path).map_err(|err| {
        VizError::Png(format!("Failed to write preview {}: {err}", path.display()))
    })?;
    Ok(visual_asset(filename, title, description))
}

pub fn write_preview_from_path(
    source: &Path,
    outdir: &Path,
    filename: &str,
    title: impl Into<String>,
    description: impl Into<String>,
) -> Result<VisualAsset, VizError> {
    let image = image::open(source).map_err(|err| {
        VizError::Png(format!(
            "Failed to load preview source {}: {err}",
            source.display()
        ))
    })?;
    write_preview_image(&image, outdir, filename, title, description)
}

fn resized_preview(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let max_dimension = width.max(height);
    if max_dimension <= PREVIEW_MAX_DIMENSION {
        image.clone()
    } else {
        image.resize(
            PREVIEW_MAX_DIMENSION,
            PREVIEW_MAX_DIMENSION,
            FilterType::Triangle,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn slugify_filename_replaces_non_filename_characters() {
        assert_eq!(slugify_filename("dinov2-vit-l14#2"), "dinov2-vit-l14_2");
        assert_eq!(slugify_filename("class-a/leaf"), "class-a_leaf");
    }

    #[test]
    fn write_preview_image_resizes_large_inputs() {
        let dir = tempdir().unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            1024,
            512,
            image::Rgb([42, 84, 126]),
        ));

        let asset = write_preview_image(
            &image,
            dir.path(),
            "preview.png",
            "Preview",
            "Resized image preview.",
        )
        .unwrap();

        assert_eq!(asset.path, "preview.png");
        let saved = image::open(dir.path().join("preview.png")).unwrap();
        assert_eq!(saved.width(), PREVIEW_MAX_DIMENSION);
        assert_eq!(saved.height(), PREVIEW_MAX_DIMENSION / 2);
    }
}
