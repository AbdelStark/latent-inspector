use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Analysis error: {0}")]
    Analysis(#[from] AnalysisError),

    #[error("Visualization error: {0}")]
    Viz(#[from] VizError),

    #[error("Dataset error: {0}")]
    Dataset(#[from] DatasetError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found in registry: {0}")]
    NotFound(String),

    #[error("Model download failed for '{name}': {reason}")]
    DownloadFailed { name: String, reason: String },

    #[error("Hash verification failed for '{name}': expected {expected}, got {actual}")]
    VerificationFailed {
        name: String,
        expected: String,
        actual: String,
    },

    #[error("ONNX inference error: {0}")]
    InferenceFailed(String),

    #[error("Session creation failed: {0}")]
    SessionCreation(String),

    #[error("Preprocessing error: {0}")]
    Preprocessing(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Convergence failed after {iterations} iterations: {reason}")]
    ConvergenceFailed { iterations: usize, reason: String },

    #[error("Empty input: {0}")]
    EmptyInput(String),
}

#[derive(Error, Debug)]
pub enum VizError {
    #[error("Terminal render error: {0}")]
    Terminal(String),

    #[error("PNG export error: {0}")]
    Png(String),

    #[error("HTML generation error: {0}")]
    Html(String),
}

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Dataset directory not found: {0}")]
    DirectoryNotFound(String),

    #[error("No images found in: {0}")]
    NoImages(String),

    #[error("Image load error for '{path}': {reason}")]
    ImageLoad { path: String, reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
