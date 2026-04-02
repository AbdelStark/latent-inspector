//! CLI handler for the `tui` subcommand.

use crate::errors::Error;
use crate::tui;
use crate::tui::app::App;
use clap::Args;
use std::path::PathBuf;
use tracing::info;

#[cfg(feature = "onnx-inference")]
use crate::analysis::{compute_metrics, variance_spectrum};
#[cfg(feature = "onnx-inference")]
use crate::extract::ExtractedFeatures;
#[cfg(feature = "onnx-inference")]
use crate::models::ModelSession;

/// Launch the interactive terminal UI.
#[derive(Args, Debug)]
pub struct TuiArgs {
    /// Path to an image to analyse (uses demo data if omitted).
    pub image: Option<PathBuf>,

    /// Comma-separated list of models (defaults to all registered models).
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,
}

pub fn run(args: TuiArgs) -> Result<(), Error> {
    let app = if let Some(ref path) = args.image {
        build_app_with_image(path, args.models)?
    } else {
        info!("No image provided — launching with demo data");
        App::demo()
    };
    tui::run(app).map_err(Error::Io)
}

/// When ONNX inference is NOT compiled, use demo metrics but show the real image.
#[cfg(not(feature = "onnx-inference"))]
fn build_app_with_image(
    path: &std::path::Path,
    _models: Option<Vec<String>>,
) -> Result<App, Error> {
    info!("ONNX inference not available — showing demo metrics with image preview");
    let mut app = App::demo();
    app.image_path = Some(path.to_path_buf());
    if let Ok(img) = image::open(path) {
        app.image_thumbnail = Some(
            img.resize(400, 400, image::imageops::FilterType::Triangle)
                .to_rgb8(),
        );
    }
    Ok(app)
}

/// When ONNX inference IS compiled, run real analysis.
#[cfg(feature = "onnx-inference")]
fn build_app_with_image(path: &std::path::Path, models: Option<Vec<String>>) -> Result<App, Error> {
    info!("Loading image: {:?}", path);
    let img = image::open(path)?;
    let thumbnail = img
        .resize(400, 400, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let model_names = models.unwrap_or_else(crate::models::registry::model_names);

    let mut all_metrics = Vec::new();
    let mut all_features: Vec<(String, ExtractedFeatures)> = Vec::new();
    let mut all_spectra = Vec::new();

    for name in &model_names {
        info!("Loading model: {}", name);
        let session = ModelSession::load(name)?;
        let output = session.infer(&img)?;
        let features = ExtractedFeatures::from_output(output)?;
        let metrics = compute_metrics(&features, name)?;
        let spectrum = variance_spectrum(&features.patch_tokens, 32)?;
        all_metrics.push(metrics);
        all_spectra.push(spectrum);
        all_features.push((name.clone(), features));
    }

    let mut all_comparisons = Vec::new();
    for i in 0..all_features.len() {
        for j in (i + 1)..all_features.len() {
            let (na, fa) = &all_features[i];
            let (nb, fb) = &all_features[j];
            all_comparisons.push(crate::analysis::compute_comparison(fa, fb, na, nb)?);
        }
    }

    let mut app = App::new(
        Some(path.to_path_buf()),
        all_metrics,
        all_comparisons,
        all_spectra,
    );
    app.image_thumbnail = Some(thumbnail);
    Ok(app)
}
