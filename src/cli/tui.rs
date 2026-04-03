//! CLI handler for the `tui` subcommand.

use crate::errors::Error;
use crate::tui;
use crate::tui::app::App;
use crate::{analysis, extract, models};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

/// Launch the interactive terminal UI.
#[derive(Args, Debug)]
pub struct TuiArgs {
    /// Path to an image to analyse (uses demo data if omitted).
    pub image: Option<PathBuf>,

    /// Comma-separated list of models (defaults to all registered models).
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,
}

/// Launch the interactive terminal UI with ratatui.
pub fn run(args: TuiArgs) -> Result<(), Error> {
    let app = if let Some(ref path) = args.image {
        build_app_with_image(path, args.models)?
    } else {
        info!("No image provided; launching the TUI with demo data");
        App::demo()
    };
    tui::run(app).map_err(Error::Io)
}

fn build_app_with_image(
    path: &std::path::Path,
    selected_models: Option<Vec<String>>,
) -> Result<App, Error> {
    info!("Loading image for live TUI analysis: {:?}", path);
    let img = image::open(path)?;
    let thumbnail = img
        .resize(400, 400, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let model_names = selected_models.unwrap_or_else(models::registry::ready_model_names);

    let mut all_metrics = Vec::new();
    let mut all_features: Vec<(String, extract::ExtractedFeatures)> = Vec::new();
    let mut all_spectra = Vec::new();

    for name in &model_names {
        info!("Loading model: {}", name);
        let mut session = models::ModelSession::load(name)?;
        let output = session.infer(&img)?;
        let features = extract::ExtractedFeatures::from_output(output)?;
        let metrics = analysis::compute_metrics(&features, name)?;
        let spectrum =
            analysis::variance_spectrum(&features.patch_tokens, analysis::TUI_PCA_COMPONENTS)?;
        all_metrics.push(metrics);
        all_spectra.push((name.clone(), spectrum));
        all_features.push((name.clone(), features));
    }

    let mut all_comparisons = Vec::new();
    for i in 0..all_features.len() {
        for j in (i + 1)..all_features.len() {
            let (na, fa) = &all_features[i];
            let (nb, fb) = &all_features[j];
            all_comparisons.push(analysis::compute_comparison(fa, fb, na, nb)?);
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
