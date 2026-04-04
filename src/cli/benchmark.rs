use crate::errors::Error;
use crate::models::ModelSession;
use clap::Args;
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Image to benchmark with.
    pub image: PathBuf,

    /// Model to benchmark.
    #[arg(short, long, default_value = "dinov2-vit-l14")]
    pub model: String,

    /// Number of warmup iterations (not timed).
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Number of timed iterations.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub iterations: usize,

    /// Output format: terminal or json.
    #[arg(short, long, default_value = "terminal")]
    pub format: BenchmarkFormat,

    /// Output file path (writes to stdout if omitted).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BenchmarkFormat {
    Terminal,
    Json,
}

impl std::fmt::Display for BenchmarkFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkFormat::Terminal => write!(f, "terminal"),
            BenchmarkFormat::Json => write!(f, "json"),
        }
    }
}

/// Latency statistics for a series of measurements.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub count: usize,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
    pub std_ms: f64,
}

/// Full benchmark result.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub model: String,
    pub image: String,
    pub backend: String,
    pub warmup_iterations: usize,
    pub timed_iterations: usize,
    pub total: LatencyStats,
    pub throughput_img_per_sec: f64,
}

impl LatencyStats {
    fn from_durations(durations: &[Duration]) -> Self {
        let mut ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = ms.len();
        let min_ms = ms.first().copied().unwrap_or(0.0);
        let max_ms = ms.last().copied().unwrap_or(0.0);
        let mean_ms = ms.iter().sum::<f64>() / count.max(1) as f64;
        let median_ms = if count == 0 {
            0.0
        } else if count % 2 == 0 {
            (ms[count / 2 - 1] + ms[count / 2]) / 2.0
        } else {
            ms[count / 2]
        };
        let p95_idx = ((count as f64 * 0.95).ceil() as usize).min(count.saturating_sub(1));
        let p95_ms = ms.get(p95_idx).copied().unwrap_or(max_ms);
        let variance =
            ms.iter().map(|v| (v - mean_ms).powi(2)).sum::<f64>() / (count as f64 - 1.0).max(1.0);
        let std_ms = variance.sqrt();

        LatencyStats {
            count,
            min_ms,
            mean_ms,
            median_ms,
            p95_ms,
            max_ms,
            std_ms,
        }
    }
}

/// Execute the `benchmark` subcommand.
pub fn run(args: BenchmarkArgs) -> Result<(), Error> {
    info!(
        "Benchmarking {} on {:?} ({} warmup + {} timed)",
        args.model, args.image, args.warmup, args.iterations
    );

    let mut session = ModelSession::load_for_analysis(&args.model)?;
    let img = image::open(&args.image)?;
    let backend = session.backend().label().to_string();

    // Warmup
    for i in 0..args.warmup {
        session.infer(&img)?;
        if i == 0 {
            info!("Warmup iteration 1/{} complete", args.warmup);
        }
    }

    // Timed iterations
    let mut total_durations = Vec::with_capacity(args.iterations);

    for _ in 0..args.iterations {
        let start = Instant::now();
        session.infer(&img)?;
        total_durations.push(start.elapsed());
    }

    let total_stats = LatencyStats::from_durations(&total_durations);
    let throughput = if total_stats.mean_ms > 0.0 {
        1000.0 / total_stats.mean_ms
    } else {
        0.0
    };

    let result = BenchmarkResult {
        model: args.model.clone(),
        image: args.image.display().to_string(),
        backend,
        warmup_iterations: args.warmup,
        timed_iterations: args.iterations,
        total: total_stats,
        throughput_img_per_sec: throughput,
    };

    match args.format {
        BenchmarkFormat::Terminal => print_terminal(&result),
        BenchmarkFormat::Json => {
            let json = serde_json::to_string_pretty(&result)?;
            if let Some(path) = &args.output {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(path, &json)?;
                println!("Benchmark results written to {}", path.display());
            } else {
                println!("{json}");
            }
        }
    }

    Ok(())
}

fn print_terminal(result: &BenchmarkResult) {
    println!();
    println!("Benchmark Results");
    println!("{}", "=".repeat(60));
    println!("  Model:      {}", result.model);
    println!("  Image:      {}", result.image);
    println!("  Backend:    {}", result.backend);
    println!(
        "  Iterations: {} warmup + {} timed",
        result.warmup_iterations, result.timed_iterations
    );
    println!("{}", "-".repeat(60));
    println!("  Inference Latency (preprocess + inference):");
    print_stats("    ", &result.total);
    println!("{}", "-".repeat(60));
    println!(
        "  Throughput: {:.1} images/sec",
        result.throughput_img_per_sec
    );
    println!("{}", "=".repeat(60));
    println!();
}

fn print_stats(prefix: &str, stats: &LatencyStats) {
    println!("{prefix}Min:    {:>8.2} ms", stats.min_ms);
    println!("{prefix}Mean:   {:>8.2} ms", stats.mean_ms);
    println!("{prefix}Median: {:>8.2} ms", stats.median_ms);
    println!("{prefix}P95:    {:>8.2} ms", stats.p95_ms);
    println!("{prefix}Max:    {:>8.2} ms", stats.max_ms);
    println!(
        "{prefix}Std:    {:>8.2} ms  ({} iterations)",
        stats.std_ms, stats.count
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_stats_from_durations() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(150),
            Duration::from_millis(180),
            Duration::from_millis(120),
        ];
        let stats = LatencyStats::from_durations(&durations);

        assert_eq!(stats.count, 5);
        approx::assert_abs_diff_eq!(stats.min_ms, 100.0, epsilon = 0.1);
        approx::assert_abs_diff_eq!(stats.max_ms, 200.0, epsilon = 0.1);
        approx::assert_abs_diff_eq!(stats.mean_ms, 150.0, epsilon = 0.1);
        approx::assert_abs_diff_eq!(stats.median_ms, 150.0, epsilon = 0.1);
        assert!(stats.p95_ms >= 180.0);
        assert!(stats.std_ms > 0.0);
    }

    #[test]
    fn latency_stats_single_duration() {
        let durations = vec![Duration::from_millis(42)];
        let stats = LatencyStats::from_durations(&durations);

        assert_eq!(stats.count, 1);
        approx::assert_abs_diff_eq!(stats.min_ms, 42.0, epsilon = 0.1);
        approx::assert_abs_diff_eq!(stats.max_ms, 42.0, epsilon = 0.1);
        approx::assert_abs_diff_eq!(stats.median_ms, 42.0, epsilon = 0.1);
    }

    #[test]
    fn latency_stats_empty() {
        let stats = LatencyStats::from_durations(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.min_ms, 0.0);
        assert_eq!(stats.mean_ms, 0.0);
    }
}
