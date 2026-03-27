mod cli;

// Re-export library modules so CLI subcommands can use `crate::` paths.
pub use latent_inspector::analysis;
pub use latent_inspector::dataset;
pub use latent_inspector::errors;
pub use latent_inspector::extract;
pub use latent_inspector::models;
pub use latent_inspector::viz;

use clap::Parser;
use tracing_subscriber::EnvFilter;

fn main() {
    let args = cli::Cli::parse();

    // Initialise tracing
    let filter = if args.verbose {
        "latent_inspector=debug,info"
    } else {
        "latent_inspector=info,warn"
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(false)
        .without_time()
        .init();

    let result = match args.command {
        cli::Commands::Compare(a) => cli::compare::run(a),
        cli::Commands::Inspect(a) => cli::inspect::run(a),
        cli::Commands::Neighbors(a) => cli::neighbors::run(a),
        cli::Commands::Similarity(a) => cli::similarity::run(a),
        cli::Commands::Drift(a) => cli::drift::run(a),
        cli::Commands::Models(a) => cli::models::run(a),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
