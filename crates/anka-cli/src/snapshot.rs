//! `anka snapshot`: write an index to disk, load it back, and prove nothing changed.
//!
//! This is the phase 3 definition of done, run at full scale. Two things are being established.
//!
//! **The round trip is exact.** Not "recall is unchanged" — every query returns the same ids in
//! the same order with bit-identical distances. Recall is an average and would hide a graph that
//! came back subtly different; comparing whole result lists cannot.
//!
//! **Mapping and reading are a real trade, so both are measured.** `load` maps the file and
//! returns immediately, deferring the cost of every byte to whichever query first touches it.
//! `read` pays for the whole file up front. Which one wins depends on how much of the collection a
//! workload visits, and a graph search visits almost none of it — that is the argument for
//! mapping, and this command is where it stops being an argument.
//!
//! The load figures are therefore reported alongside a **cold query pass**, because a load time
//! that excludes the page faults it postponed is not a load time.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anka_core::{
    Candidate, Cosine, DotProduct, L2Squared, Metric, MetricKind, VectorStore, mem, preprocess_all,
};
use anka_index::hnsw::{DistanceCounter, HnswIndex, HnswParams, SelectionPolicy};
use anka_store::Verify;
use anyhow::{Result, bail};
use clap::Args;

use crate::{DatasetSpec, dataset_dir, spec_for, take_vectors};

#[derive(Args, Debug)]
pub struct SnapshotArgs {
    /// Dataset to index: siftsmall, sift1m or glove100.
    pub dataset: String,

    /// Directory holding the datasets. Defaults to $ANKA_DATASETS, then ~/anka-datasets.
    #[arg(long, env = "ANKA_DATASETS")]
    pub datasets_dir: Option<PathBuf>,

    /// Where to write the snapshot. Defaults to a temporary file next to the datasets.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Keep the snapshot afterwards instead of deleting it.
    #[arg(long)]
    pub keep: bool,

    /// Connections per node per layer.
    #[arg(long, default_value_t = 16)]
    pub m: usize,

    /// Candidate list size during construction.
    #[arg(long, default_value_t = 200)]
    pub ef_construction: usize,

    /// Seed for layer assignment.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Index only the first N base vectors.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Queries used for the comparison. Fewer is faster; every one of them must match exactly.
    #[arg(long, default_value_t = 1_000)]
    pub queries: usize,

    /// Neighbours per query.
    #[arg(long, default_value_t = 10)]
    pub k: usize,

    /// Beam size for the comparison queries.
    #[arg(long, default_value_t = 80)]
    pub ef: usize,
}

pub fn run(args: SnapshotArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = dataset_dir(&args.dataset, spec, args.datasets_dir.clone())?;

    let mut base = anka_core::dataset::read_fvecs(dir.join(format!("{}_base.fvecs", spec.prefix)))?;
    let mut queries =
        anka_core::dataset::read_fvecs(dir.join(format!("{}_query.fvecs", spec.prefix)))?;

    if let Some(limit) = args.limit {
        if limit == 0 || limit > base.len() {
            bail!("--limit must be between 1 and {}", base.len());
        }
        base = take_vectors(&base, limit)?;
    }
    if args.queries == 0 || args.queries > queries.len() {
        bail!("--queries must be between 1 and {}", queries.len());
    }
    queries = take_vectors(&queries, args.queries)?;

    if args.k == 0 || args.ef == 0 {
        bail!("--k and --ef must be at least 1");
    }

    let path = args
        .out
        .clone()
        .unwrap_or_else(|| dir.join("snapshot.anka"));

    tracing::info!(
        "{}: {} base x {} queries, dim {}, metric {}",
        args.dataset,
        base.len(),
        queries.len(),
        base.dim(),
        spec.metric.name()
    );
    tracing::info!("snapshot path: {}", path.display());

    let outcome = match spec.metric {
        MetricKind::L2Squared => measure::<L2Squared>(base, queries, spec, &args, &path),
        MetricKind::Cosine => measure::<Cosine>(base, queries, spec, &args, &path),
        MetricKind::Dot => measure::<DotProduct>(base, queries, spec, &args, &path),
    };

    if !args.keep && path.exists() {
        let _ = std::fs::remove_file(&path);
    }
    outcome
}

fn measure<M: Metric>(
    mut base: VectorStore,
    mut queries: VectorStore,
    spec: &DatasetSpec,
    args: &SnapshotArgs,
    path: &std::path::Path,
) -> Result<()> {
    preprocess_all::<M>(&mut base)?;
    preprocess_all::<M>(&mut queries)?;

    let params = HnswParams::new(args.m)?
        .with_ef_construction(args.ef_construction)?
        .with_seed(args.seed)?
        .with_selection(SelectionPolicy::default())?;

    let mut index = HnswIndex::with_capacity(base.dim(), spec.metric, params, base.len())?;
    let mut searcher = index.searcher();
    let mut counter = DistanceCounter::new();

    let view = base.view();
    let start = Instant::now();
    for position in 0..base.len() {
        index.insert::<M>(&mut searcher, view.get(position), &mut counter)?;
    }
    let build_time = start.elapsed();
    tracing::info!(
        "build: {:.2?} ({:.0} vectors/s)",
        build_time,
        index.len() as f64 / build_time.as_secs_f64()
    );

    let before = answers::<M>(&index, &queries, args)?;

    // ---- write ------------------------------------------------------------------------------
    let start = Instant::now();
    anka_store::write(&index, path, 0)?;
    let write_time = start.elapsed();
    let file_bytes = std::fs::metadata(path)?.len();
    tracing::info!(
        "write: {:.2?}  {} ({:.0} MiB/s), including fsync of the file and its directory",
        write_time,
        mem::human_bytes(file_bytes),
        file_bytes as f64 / (1 << 20) as f64 / write_time.as_secs_f64()
    );

    // The in-memory index is dropped before loading so the two do not coexist in the RSS figure.
    drop(index);
    drop(base);

    // ---- load: mapped -----------------------------------------------------------------------
    //
    // `Verify::Header` on purpose. Checksumming the body reads every byte of the file, which is
    // exactly what mapping exists to avoid — timing a lazy load with full verification turned on
    // measures the opposite of the thing being claimed. What that verification costs is measured
    // separately below, where it answers its own question instead of contaminating this one.
    let start = Instant::now();
    let mapped = anka_store::load(path, Verify::Header)?;
    let mapped_load = start.elapsed();
    let mapped_rss = current_rss();

    let start = Instant::now();
    let after_mapped = answers::<M>(&mapped, &queries, args)?;
    let mapped_cold = start.elapsed();

    let start = Instant::now();
    answers::<M>(&mapped, &queries, args)?;
    let mapped_warm = start.elapsed();

    let stats = mapped.graph_stats();
    mapped
        .validate()
        .map_err(|violation| anyhow::anyhow!("the loaded graph is invalid: {violation}"))?;
    tracing::info!(
        "loaded graph: {} nodes, {} edges over {} layers — invariants hold",
        stats.nodes,
        stats.edges,
        stats.per_layer.len()
    );
    drop(mapped);

    // ---- load: fully read -------------------------------------------------------------------
    let start = Instant::now();
    let owned = anka_store::read(path, Verify::Header)?;
    let owned_load = start.elapsed();
    let owned_rss = current_rss();

    let start = Instant::now();
    let after_owned = answers::<M>(&owned, &queries, args)?;
    let owned_cold = start.elapsed();
    drop(owned);

    // ---- load: mapped, with the body checksummed --------------------------------------------
    // What full verification costs, as its own line rather than hidden inside the one above.
    let start = Instant::now();
    let verified = anka_store::load(path, Verify::Body)?;
    let verified_load = start.elapsed();
    let verified_rss = current_rss();
    drop(verified);

    // ---- results ----------------------------------------------------------------------------
    report_table(&[
        Row::new(
            "mapped",
            mapped_load,
            Some(mapped_cold),
            Some(mapped_warm),
            mapped_rss,
        ),
        Row::new("fully read", owned_load, Some(owned_cold), None, owned_rss),
        Row::new("mapped +crc", verified_load, None, None, verified_rss),
    ]);

    let mut failures = 0;
    for (label, after) in [("mapped", &after_mapped), ("fully read", &after_owned)] {
        match first_difference(&before, after) {
            None => tracing::info!(
                "{label}: all {} queries identical, ids and distances",
                before.len()
            ),
            Some((query, detail)) => {
                failures += 1;
                tracing::error!("{label}: query {query} differs — {detail}");
            }
        }
    }

    crate::report_memory();
    if failures > 0 {
        bail!("FAIL a reloaded index answered differently from the one that was written");
    }
    tracing::info!(
        "PASS {} queries answered bit-identically after a round trip through disk",
        before.len()
    );
    Ok(())
}

/// Every query's full result list, which is what the comparison is against.
fn answers<M: Metric>(
    index: &HnswIndex,
    queries: &VectorStore,
    args: &SnapshotArgs,
) -> Result<Vec<Vec<Candidate>>> {
    let mut searcher = index.searcher();
    let mut counter = DistanceCounter::new();
    let view = queries.view();
    (0..view.len())
        .map(|position| {
            Ok(index.search::<M>(
                &mut searcher,
                view.get(position),
                args.k,
                args.ef,
                &mut counter,
            )?)
        })
        .collect()
}

/// The first query whose result list changed, and how.
///
/// Reported rather than counted: one concrete disagreement — which rank, which id, which distance
/// — says what a failure count cannot.
fn first_difference(
    before: &[Vec<Candidate>],
    after: &[Vec<Candidate>],
) -> Option<(usize, String)> {
    if before.len() != after.len() {
        return Some((
            before.len().min(after.len()),
            format!(
                "the reloaded index answered {} queries against {}",
                after.len(),
                before.len()
            ),
        ));
    }
    for (query, (left, right)) in before.iter().zip(after).enumerate() {
        if left.len() != right.len() {
            return Some((
                query,
                format!("{} neighbours against {}", left.len(), right.len()),
            ));
        }
        for (rank, (a, b)) in left.iter().zip(right).enumerate() {
            if a != b {
                return Some((
                    query,
                    format!(
                        "rank {rank}: id {} at {:e} became id {} at {:e}",
                        a.id, a.dist, b.id, b.dist
                    ),
                ));
            }
        }
    }
    None
}

struct Row {
    label: &'static str,
    load: Duration,
    cold: Option<Duration>,
    warm: Option<Duration>,
    rss: Option<u64>,
}

impl Row {
    fn new(
        label: &'static str,
        load: Duration,
        cold: Option<Duration>,
        warm: Option<Duration>,
        rss: Option<u64>,
    ) -> Self {
        Self {
            label,
            load,
            cold,
            warm,
            rss,
        }
    }
}

fn report_table(rows: &[Row]) {
    tracing::info!("");
    tracing::info!(
        "  {:<12} {:>12} {:>14} {:>14} {:>12}",
        "load",
        "open",
        "first queries",
        "repeat",
        "RSS"
    );
    for row in rows {
        tracing::info!(
            "  {:<12} {:>12} {:>14} {:>14} {:>12}",
            row.label,
            format!("{:.2?}", row.load),
            duration(row.cold),
            duration(row.warm),
            row.rss.map_or_else(|| "-".to_string(), mem::human_bytes)
        );
    }
    tracing::info!("");
    tracing::info!(
        "\"open\" excludes whatever the mapping deferred; \"first queries\" is where that lands"
    );
}

fn duration(value: Option<Duration>) -> String {
    value.map_or_else(|| "-".to_string(), |d| format!("{d:.2?}"))
}

fn current_rss() -> Option<u64> {
    mem::resident_set_size().map(|usage| usage.current_bytes)
}
