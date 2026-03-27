use crate::errors::DatasetError;
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use tracing::debug;

const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "webp"];

/// A single image entry in a dataset.
#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub stem: String,
}

/// Scan `dir` for image files and return them sorted by path.
pub fn scan_images(dir: &Path) -> Result<Vec<ImageEntry>, DatasetError> {
    if !dir.exists() {
        return Err(DatasetError::DirectoryNotFound(dir.display().to_string()));
    }

    let mut entries: Vec<ImageEntry> = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    entries.push(ImageEntry { path, stem });
                }
            }
        }
    }

    if entries.is_empty() {
        return Err(DatasetError::NoImages(dir.display().to_string()));
    }

    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

/// Load an image from disk.
pub fn load_image(path: &Path) -> Result<DynamicImage, DatasetError> {
    debug!("Loading image: {}", path.display());
    image::open(path).map_err(|e| DatasetError::ImageLoad {
        path: path.display().to_string(),
        reason: e.to_string(),
    })
}

/// Iterator that loads images from a dataset directory with a progress bar.
pub struct DatasetIterator {
    entries: Vec<ImageEntry>,
    index: usize,
    progress: Option<ProgressBar>,
}

impl DatasetIterator {
    pub fn new(dir: &Path, show_progress: bool) -> Result<Self, DatasetError> {
        let entries = scan_images(dir)?;
        let progress = if show_progress {
            let pb = ProgressBar::new(entries.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            pb.set_message("Loading images");
            Some(pb)
        } else {
            None
        };

        Ok(Self {
            entries,
            index: 0,
            progress,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Iterator for DatasetIterator {
    type Item = Result<(ImageEntry, DynamicImage), DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.entries.len() {
            if let Some(pb) = &self.progress {
                pb.finish_with_message("Done");
            }
            return None;
        }

        let entry = self.entries[self.index].clone();
        self.index += 1;

        if let Some(pb) = &self.progress {
            pb.inc(1);
        }

        let img = match load_image(&entry.path) {
            Ok(img) => img,
            Err(e) => return Some(Err(e)),
        };

        Some(Ok((entry, img)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_scan_empty_dir() {
        let dir = tempdir().unwrap();
        let result = scan_images(dir.path());
        assert!(matches!(result, Err(DatasetError::NoImages(_))));
    }

    #[test]
    fn test_scan_nonexistent_dir() {
        let result = scan_images(Path::new("/nonexistent/path/12345"));
        assert!(matches!(result, Err(DatasetError::DirectoryNotFound(_))));
    }

    #[test]
    fn test_scan_finds_images() {
        let dir = tempdir().unwrap();
        // Create a dummy PNG file
        let img = image::RgbImage::new(4, 4);
        let path = dir.path().join("test.png");
        img.save(&path).unwrap();

        let entries = scan_images(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].stem, "test");
    }
}
