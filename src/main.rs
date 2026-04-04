mod cli;

// Re-export library modules so CLI subcommands can use `crate::` paths.
pub use latent_inspector::analysis;
pub use latent_inspector::dataset;
pub use latent_inspector::errors;
pub use latent_inspector::extract;
pub use latent_inspector::models;
pub use latent_inspector::tui;
pub use latent_inspector::validation;
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
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    let result = match args.command {
        cli::Commands::Compare(a) => cli::compare::run(a),
        cli::Commands::Embed(a) => cli::embed::run(a),
        cli::Commands::Inspect(a) => cli::inspect::run(a),
        cli::Commands::Neighbors(a) => cli::neighbors::run(a),
        cli::Commands::Profile(a) => cli::profile::run(a),
        cli::Commands::Similarity(a) => cli::similarity::run(a),
        cli::Commands::Drift(a) => cli::drift::run(a),
        cli::Commands::Models(a) => cli::models::run(a),
        cli::Commands::Validate(a) => cli::validate::run(a),
        cli::Commands::Tui(a) => cli::tui::run(a),
    };

    if let Err(e) = result {
        if e.should_print() {
            eprintln!("error: {e}");
        }
        std::process::exit(e.exit_code());
    }
}
