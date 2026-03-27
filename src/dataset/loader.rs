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
    pub relative_path: PathBuf,
    pub stem: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedImage {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetProcessingSummary {
    pub discovered: usize,
    pub loaded: usize,
    pub skipped: usize,
    pub skipped_examples: Vec<SkippedImage>,
}

impl DatasetProcessingSummary {
    pub fn new(discovered: usize) -> Self {
        Self {
            discovered,
            loaded: 0,
            skipped: 0,
            skipped_examples: Vec::new(),
        }
    }

    pub fn record_loaded(&mut self) {
        self.loaded += 1;
    }

    pub fn record_skipped(&mut self, path: impl Into<String>, reason: impl Into<String>) {
        self.skipped += 1;
        if self.skipped_examples.len() < 5 {
            self.skipped_examples.push(SkippedImage {
                path: path.into(),
                reason: reason.into(),
            });
        }
    }

    pub fn has_loaded_images(&self) -> bool {
        self.loaded > 0
    }
}

/// Scan `dir` for image files and return them sorted by path.
pub fn scan_images(dir: &Path) -> Result<Vec<ImageEntry>, DatasetError> {
    if !dir.exists() || !dir.is_dir() {
        return Err(DatasetError::DirectoryNotFound(dir.display().to_string()));
    }

    let mut entries: Vec<ImageEntry> = Vec::new();
    scan_images_recursive(dir, dir, &mut entries)?;

    if entries.is_empty() {
        return Err(DatasetError::NoImages(dir.display().to_string()));
    }

    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
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

/// Visit each readable image in a dataset, skipping corrupt image files while
/// preserving a summary of what was loaded vs skipped.
pub fn for_each_image<F, E>(
    dir: &Path,
    show_progress: bool,
    mut visit: F,
) -> Result<DatasetProcessingSummary, E>
where
    F: FnMut(ImageEntry, DynamicImage) -> Result<(), E>,
    E: From<DatasetError>,
{
    let dataset = DatasetIterator::new(dir, show_progress)?;
    let mut summary = DatasetProcessingSummary::new(dataset.len());

    for result in dataset {
        match result {
            Ok((entry, image)) => {
                summary.record_loaded();
                visit(entry, image)?;
            }
            Err(DatasetError::ImageLoad { path, reason }) => {
                summary.record_skipped(path, reason);
            }
            Err(error) => return Err(error.into()),
        }
    }

    Ok(summary)
}

fn scan_images_recursive(
    root: &Path,
    current: &Path,
    entries: &mut Vec<ImageEntry>,
) -> Result<(), DatasetError> {
    let mut directory_entries = std::fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    directory_entries.sort_by_key(|entry| entry.path());

    for entry in directory_entries {
        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            scan_images_recursive(root, &path, entries)?;
            continue;
        }

        if file_type.is_file() && is_supported_image_path(&path) {
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(path.as_path())
                .to_path_buf();
            entries.push(ImageEntry {
                stem: relative_stem(&relative_path),
                path,
                relative_path,
            });
        }
    }

    Ok(())
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn relative_stem(relative_path: &Path) -> String {
    relative_path
        .with_extension("")
        .to_string_lossy()
        .replace('\\', "/")
}

fn load_entry_image(entry: &ImageEntry) -> Result<DynamicImage, DatasetError> {
    debug!("Loading image: {}", entry.path.display());
    image::open(&entry.path).map_err(|e| DatasetError::ImageLoad {
        path: entry.relative_path.display().to_string(),
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

        let img = match load_entry_image(&entry) {
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

    #[test]
    fn test_scan_recurses_and_sorts_by_relative_path() {
        let dir = tempdir().unwrap();
        let nested_a = dir.path().join("class-b");
        let nested_b = dir.path().join("class-a").join("deep");
        std::fs::create_dir_all(&nested_a).unwrap();
        std::fs::create_dir_all(&nested_b).unwrap();

        image::RgbImage::new(4, 4)
            .save(dir.path().join("root.png"))
            .unwrap();
        image::RgbImage::new(4, 4)
            .save(nested_a.join("beta.png"))
            .unwrap();
        image::RgbImage::new(4, 4)
            .save(nested_b.join("alpha.png"))
            .unwrap();

        let entries = scan_images(dir.path()).unwrap();
        let relative_paths = entries
            .iter()
            .map(|entry| entry.relative_path.display().to_string())
            .collect::<Vec<_>>();
        let stems = entries
            .iter()
            .map(|entry| entry.stem.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            relative_paths,
            vec![
                "class-a/deep/alpha.png".to_string(),
                "class-b/beta.png".to_string(),
                "root.png".to_string()
            ]
        );
        assert_eq!(stems, vec!["class-a/deep/alpha", "class-b/beta", "root"]);
    }

    #[test]
    fn test_for_each_image_skips_corrupt_supported_files() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();

        image::RgbImage::new(4, 4)
            .save(dir.path().join("good.png"))
            .unwrap();
        std::fs::write(nested.join("broken.png"), b"not a real image").unwrap();

        let mut visited = Vec::new();
        let summary = for_each_image(dir.path(), false, |entry, _image| {
            visited.push(entry.stem);
            Ok::<(), DatasetError>(())
        })
        .unwrap();

        assert_eq!(visited, vec!["good".to_string()]);
        assert_eq!(summary.discovered, 2);
        assert_eq!(summary.loaded, 1);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.skipped_examples.len(), 1);
        assert_eq!(summary.skipped_examples[0].path, "nested/broken.png");
    }
}
