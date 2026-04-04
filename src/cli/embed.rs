use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use clap::Args;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::info;

/// Embedding export level: what to include in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum EmbedLevel {
    /// Export only the global image embedding (CLS or mean-patch).
    Global,
    /// Export full patch-level embeddings (one vector per spatial patch).
    Patches,
    /// Export both global and patch embeddings.
    Full,
}

impl std::fmt::Display for EmbedLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedLevel::Global => write!(f, "global"),
            EmbedLevel::Patches => write!(f, "patches"),
            EmbedLevel::Full => write!(f, "full"),
        }
    }
}

#[derive(Args, Debug)]
pub struct EmbedArgs {
    /// Image file or directory of images to embed.
    pub input: PathBuf,

    /// Model to use for embedding.
    #[arg(short, long, default_value = "dinov2-vit-l14")]
    pub model: String,

    /// What to export: global embedding, patch embeddings, or both.
    #[arg(short, long, default_value = "global")]
    pub level: EmbedLevel,

    /// Output file path. Writes JSON Lines (one JSON object per image).
    /// Use "-" or omit to write to stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// One line of JSONL output for global-level embeddings.
#[derive(Debug, Serialize)]
struct GlobalEmbedRecord {
    image: String,
    model: String,
    basis: String,
    embed_dim: usize,
    embedding: Vec<f32>,
}

/// One line of JSONL output for patch-level embeddings.
#[derive(Debug, Serialize)]
struct PatchEmbedRecord {
    image: String,
    model: String,
    n_patches: usize,
    embed_dim: usize,
    /// Flattened patch matrix [n_patches * embed_dim], row-major.
    patches: Vec<f32>,
}

/// One line of JSONL output for full (global + patches) embeddings.
#[derive(Debug, Serialize)]
struct FullEmbedRecord {
    image: String,
    model: String,
    basis: String,
    embed_dim: usize,
    embedding: Vec<f32>,
    n_patches: usize,
    /// Flattened patch matrix [n_patches * embed_dim], row-major.
    patches: Vec<f32>,
}

/// Execute the `embed` subcommand: export embeddings as JSON Lines.
pub fn run(args: EmbedArgs) -> Result<(), Error> {
    let mut session = ModelSession::load_for_analysis(&args.model)?;

    // Collect input images
    let images = collect_images(&args.input)?;
    if images.is_empty() {
        return Err(Error::Analysis(
            crate::errors::AnalysisError::InsufficientData(format!(
                "No valid images found at {:?}",
                args.input
            )),
        ));
    }

    info!(
        "Embedding {} images with {} (level: {})",
        images.len(),
        args.model,
        args.level
    );

    // Open output writer
    let mut writer: Box<dyn Write> = match &args.output {
        Some(path) if path.to_string_lossy() != "-" => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Box::new(std::io::BufWriter::new(std::fs::File::create(path)?))
        }
        _ => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };

    let mut processed = 0usize;
    let mut skipped = 0usize;

    for image_path in &images {
        let img = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                tracing::warn!("Skipping {}: {e}", image_path.display());
                skipped += 1;
                continue;
            }
        };

        let output = match session.infer(&img) {
            Ok(output) => output,
            Err(e) => {
                tracing::warn!("Inference failed for {}: {e}", image_path.display());
                skipped += 1;
                continue;
            }
        };

        let features = ExtractedFeatures::from_output(output)?;
        let image_str = image_path.display().to_string();

        match args.level {
            EmbedLevel::Global => {
                let (basis, embedding) = features.preferred_global_embedding();
                let record = GlobalEmbedRecord {
                    image: image_str,
                    model: args.model.clone(),
                    basis: basis.label().to_string(),
                    embed_dim: embedding.len(),
                    embedding: embedding.to_vec(),
                };
                serde_json::to_writer(&mut writer, &record)?;
                writeln!(writer)?;
            }
            EmbedLevel::Patches => {
                let record = PatchEmbedRecord {
                    image: image_str,
                    model: args.model.clone(),
                    n_patches: features.n_patches,
                    embed_dim: features.embed_dim,
                    patches: features.patch_tokens.as_slice().unwrap_or(&[]).to_vec(),
                };
                serde_json::to_writer(&mut writer, &record)?;
                writeln!(writer)?;
            }
            EmbedLevel::Full => {
                let (basis, embedding) = features.preferred_global_embedding();
                let record = FullEmbedRecord {
                    image: image_str,
                    model: args.model.clone(),
                    basis: basis.label().to_string(),
                    embed_dim: features.embed_dim,
                    embedding: embedding.to_vec(),
                    n_patches: features.n_patches,
                    patches: features.patch_tokens.as_slice().unwrap_or(&[]).to_vec(),
                };
                serde_json::to_writer(&mut writer, &record)?;
                writeln!(writer)?;
            }
        }

        processed += 1;
    }

    writer.flush()?;

    // Summary to stderr (so it doesn't pollute JSONL output on stdout)
    eprintln!(
        "Embedded {processed} images ({skipped} skipped) with {} at {level} level",
        args.model,
        level = args.level
    );

    Ok(())
}

/// Collect image paths from a file or directory.
fn collect_images(input: &Path) -> Result<Vec<PathBuf>, Error> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }

    if input.is_dir() {
        let mut images = Vec::new();
        collect_images_recursive(input, &mut images);
        images.sort();
        return Ok(images);
    }

    Err(Error::Analysis(
        crate::errors::AnalysisError::InsufficientData(format!(
            "Input path does not exist: {}",
            input.display()
        )),
    ))
}

fn collect_images_recursive(dir: &std::path::Path, images: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_images_recursive(&path, images);
        } else if is_image_file(&path) {
            images.push(path);
        }
    }
}

fn is_image_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tiff" | "tif" | "webp"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_image_recognizes_common_extensions() {
        assert!(is_image_file(std::path::Path::new("photo.jpg")));
        assert!(is_image_file(std::path::Path::new("photo.JPEG")));
        assert!(is_image_file(std::path::Path::new("photo.png")));
        assert!(is_image_file(std::path::Path::new("photo.webp")));
        assert!(!is_image_file(std::path::Path::new("readme.txt")));
        assert!(!is_image_file(std::path::Path::new("model.onnx")));
    }

    #[test]
    fn embed_level_display() {
        assert_eq!(EmbedLevel::Global.to_string(), "global");
        assert_eq!(EmbedLevel::Patches.to_string(), "patches");
        assert_eq!(EmbedLevel::Full.to_string(), "full");
    }
}
