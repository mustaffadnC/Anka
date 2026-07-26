//! Anka command-line interface.
//!
//! Phase 0 provides `anka load <dataset>`; `build`, `query`, `bench` and `checkpoint` follow
//! in later phases.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anka_core::{dataset, mem};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "anka",
    version,
    about = "Anka — a vector search engine built from scratch"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Load a dataset, verify its shape, and report load time and memory cost.
    Load(LoadArgs),
}

#[derive(Args, Debug)]
struct LoadArgs {
    /// Dataset to load: siftsmall, sift1m or glove100.
    dataset: String,

    /// Directory holding the datasets. Defaults to $ANKA_DATASETS, then ~/anka-datasets.
    #[arg(long, env = "ANKA_DATASETS")]
    datasets_dir: Option<PathBuf>,

    /// Load only the base vectors, skipping queries and ground truth.
    #[arg(long)]
    base_only: bool,
}

/// Where a dataset's files live and what shape they are required to have.
///
/// The expected counts are asserted, not merely printed. Phase 0's definition of done is that
/// SIFT1M loads as exactly 1 000 000 x 128; a loader that quietly accepts 999 999 would
/// invalidate every recall figure measured on top of it.
struct DatasetSpec {
    dir: &'static str,
    prefix: &'static str,
    dim: usize,
    base: usize,
    queries: usize,
    neighbours: usize,
}

const SIFTSMALL: DatasetSpec = DatasetSpec {
    dir: "siftsmall",
    prefix: "siftsmall",
    dim: 128,
    base: 10_000,
    queries: 100,
    neighbours: 100,
};

const SIFT1M: DatasetSpec = DatasetSpec {
    dir: "sift",
    prefix: "sift",
    dim: 128,
    base: 1_000_000,
    queries: 10_000,
    neighbours: 100,
};

const GLOVE100: DatasetSpec = DatasetSpec {
    dir: "glove100",
    prefix: "glove100",
    dim: 100,
    base: 1_183_514,
    queries: 10_000,
    neighbours: 100,
};

fn main() -> Result<()> {
    init_tracing();
    match Cli::parse().command {
        Command::Load(args) => load(args),
    }
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}

fn load(args: LoadArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = resolve_root(args.datasets_dir)?.join(spec.dir);

    if !dir.is_dir() {
        bail!(
            "{} does not exist — run ./scripts/download_datasets.sh {}",
            dir.display(),
            args.dataset
        );
    }
    tracing::info!("loading {} from {}", args.dataset, dir.display());

    let base_path = dir.join(format!("{}_base.fvecs", spec.prefix));
    let (base, elapsed) = timed(|| dataset::read_fvecs(&base_path))
        .with_context(|| format!("reading {}", base_path.display()))?;
    check_shape("base", base.len(), base.dim(), spec.base, spec.dim)?;
    report("base", base.len(), base.dim(), base.data_bytes(), elapsed);

    if !args.base_only {
        let query_path = dir.join(format!("{}_query.fvecs", spec.prefix));
        let (queries, elapsed) = timed(|| dataset::read_fvecs(&query_path))
            .with_context(|| format!("reading {}", query_path.display()))?;
        check_shape(
            "query",
            queries.len(),
            queries.dim(),
            spec.queries,
            spec.dim,
        )?;
        report(
            "query",
            queries.len(),
            queries.dim(),
            queries.data_bytes(),
            elapsed,
        );

        let gt_path = dir.join(format!("{}_groundtruth.ivecs", spec.prefix));
        let (ground_truth, elapsed) = timed(|| dataset::read_ivecs(&gt_path))
            .with_context(|| format!("reading {}", gt_path.display()))?;
        check_shape(
            "groundtruth",
            ground_truth.len(),
            ground_truth.dim(),
            spec.queries,
            spec.neighbours,
        )?;
        report(
            "groundtruth",
            ground_truth.len(),
            ground_truth.dim(),
            ground_truth.data_bytes(),
            elapsed,
        );
    }

    match mem::resident_set_size() {
        Some(usage) => tracing::info!(
            "resident set size: {} (peak {})",
            mem::human_bytes(usage.current_bytes),
            mem::human_bytes(usage.peak_bytes),
        ),
        // Not an error: measurements are taken on Linux, and refusing to invent a number
        // elsewhere is the point. See docs/DESIGN.md, section 9.
        None => tracing::warn!("resident set size unavailable on this platform (Linux only)"),
    }

    Ok(())
}

fn spec_for(name: &str) -> Result<&'static DatasetSpec> {
    Ok(match name {
        "siftsmall" => &SIFTSMALL,
        "sift1m" | "sift" => &SIFT1M,
        "glove100" | "glove" => &GLOVE100,
        other => bail!("unknown dataset '{other}' (expected: siftsmall, sift1m, glove100)"),
    })
}

fn resolve_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        return Ok(dir);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("pass --datasets-dir or set $ANKA_DATASETS: no home directory is available")?;
    Ok(PathBuf::from(home).join("anka-datasets"))
}

fn timed<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<(T, Duration), E> {
    let start = Instant::now();
    let value = f()?;
    Ok((value, start.elapsed()))
}

fn check_shape(
    what: &str,
    rows: usize,
    columns: usize,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<()> {
    if rows != expected_rows || columns != expected_columns {
        bail!("{what}: expected {expected_rows} x {expected_columns}, found {rows} x {columns}");
    }
    Ok(())
}

fn report(what: &str, rows: usize, columns: usize, bytes: usize, elapsed: Duration) {
    tracing::info!(
        "{what:<12} {rows:>9} x {columns:<4} {:>10}  in {:.2?}",
        mem::human_bytes(bytes as u64),
        elapsed
    );
}
