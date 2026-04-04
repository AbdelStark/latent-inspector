use clap::{Parser, Subcommand};

pub mod benchmark;
pub mod compare;
pub mod drift;
pub mod embed;
pub mod inspect;
pub mod models;
pub mod neighbors;
pub mod profile;
pub mod similarity;
pub mod tui;
pub mod validate;

/// Fast CLI for inspecting and comparing SSL vision model representations.
#[derive(Parser, Debug)]
#[command(
    name = "latent-inspector",
    version,
    about = "Inspect and compare learned representations across self-supervised vision models",
    long_about = None,
)]
pub struct Cli {
    /// Enable verbose logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Benchmark inference latency for a model.
    Benchmark(benchmark::BenchmarkArgs),
    /// Compare representations across multiple models for a single image.
    Compare(compare::CompareArgs),
    /// Export embeddings as JSON Lines for downstream use.
    Embed(embed::EmbedArgs),
    /// Deep-dive analysis of a single model's representation.
    Inspect(inspect::InspectArgs),
    /// Find nearest neighbors in a dataset for a query image.
    Neighbors(neighbors::NeighborsArgs),
    /// Profile a model's representation space over a dataset.
    Profile(profile::ProfileArgs),
    /// Measure representation similarity between two models across a dataset.
    Similarity(similarity::SimilarityArgs),
    /// Track representation changes across model checkpoints.
    Drift(drift::DriftArgs),
    /// List and download available models.
    Models(models::ModelsArgs),
    /// Validate preprocessing, tensor semantics, and reference parity.
    Validate(validate::ValidateArgs),
    /// Launch the interactive terminal UI.
    Tui(tui::TuiArgs),
}
