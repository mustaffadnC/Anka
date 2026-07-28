//! `anka bench`: build an index, then sweep `ef` measuring recall and throughput.
//!
//! Build and query live in one command because an index cannot be persisted until phase 3, so
//! splitting them would mean rebuilding for every measurement.
//!
//! Measurement rules, from spec section 6, and each one is here for a reason:
//!
//! - **Queries are timed single-threaded.** With several threads the memory bandwidth saturates
//!   and cache sharing adds noise, and an algorithmic improvement becomes indistinguishable from
//!   a scheduling accident. Phase 1 measured that ceiling at 49.4 GB/s.
//! - **Warm-up uses a different slice than the measurement.** Warming with the queries about to be
//!   timed leaves their result pages in cache and inflates QPS.
//! - **A limited base set gets fresh ground truth.** The published list describes the whole
//!   collection; against a subset it is simply the wrong answer.

use std::time::{Duration, Instant};

use anka_core::dataset::IntMatrix;
use anka_core::{
    Cosine, DotProduct, L2Squared, Metric, MetricKind, VectorStore, mem, preprocess_all,
};
use anka_index::brute_force::Kernel;
use anka_index::ground_truth;
use anka_index::hnsw::{DistanceCounter, HnswIndex, HnswParams, SelectionPolicy};
use anyhow::{Result, bail};
use clap::Args;

use crate::{DatasetSpec, dataset_dir, spec_for, take_rows, take_vectors};

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Dataset to measure: siftsmall, sift1m or glove100.
    pub dataset: String,

    /// Directory holding the datasets. Defaults to $ANKA_DATASETS, then ~/anka-datasets.
    #[arg(long, env = "ANKA_DATASETS")]
    pub datasets_dir: Option<std::path::PathBuf>,

    /// Connections per node per layer.
    #[arg(long, default_value_t = 16)]
    pub m: usize,

    /// Candidate list size during construction.
    #[arg(long, default_value_t = 200)]
    pub ef_construction: usize,

    /// Seed for layer assignment.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Use only the first N base vectors. Ground truth is recomputed when set.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Use only the first N queries.
    #[arg(long)]
    pub queries: Option<usize>,

    /// Neighbours per query.
    #[arg(long, default_value_t = 10)]
    pub k: usize,

    /// Query-time beam sizes to sweep.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = vec![10usize, 20, 40, 80, 160, 320, 512, 800]
    )]
    pub ef: Vec<usize>,

    /// Queries used to warm caches before timing. Taken from the tail of the query set so they are
    /// not the ones being measured.
    #[arg(long, default_value_t = 1000)]
    pub warmup: usize,

    /// Ablation: pick the nearest M candidates instead of running the heuristic.
    #[arg(long)]
    pub no_heuristic: bool,

    /// Ablation: do not refill pruned candidates back up to M.
    #[arg(long)]
    pub no_keep_pruned: bool,
}

impl BenchArgs {
    fn selection(&self) -> SelectionPolicy {
        SelectionPolicy {
            heuristic: !self.no_heuristic,
            keep_pruned: !self.no_keep_pruned,
        }
    }
}

/// Everything a sweep step needs that does not change between beam sizes.
struct SweepContext<'a> {
    queries: &'a VectorStore,
    truth: &'a IntMatrix,
    k: usize,
    /// Number of tail queries used to warm caches.
    warmup: usize,
    /// Number of leading queries actually timed.
    measured: usize,
}

/// One row of the sweep.
struct SweepRow {
    ef: usize,
    recall: f64,
    qps: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    distances_per_query: Option<f64>,
}

pub fn run(args: BenchArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = dataset_dir(&args.dataset, spec, args.datasets_dir.clone())?;

    let mut base = anka_core::dataset::read_fvecs(dir.join(format!("{}_base.fvecs", spec.prefix)))?;
    let mut queries =
        anka_core::dataset::read_fvecs(dir.join(format!("{}_query.fvecs", spec.prefix)))?;
    let mut published =
        anka_core::dataset::read_ivecs(dir.join(format!("{}_groundtruth.ivecs", spec.prefix)))?;

    if args.k == 0 {
        bail!("--k must be at least 1");
    }
    if args.ef.is_empty() {
        bail!("--ef must list at least one beam size");
    }

    // A subset invalidates the published ground truth: it describes the whole collection.
    let mut recompute_truth = false;
    if let Some(limit) = args.limit {
        if limit == 0 || limit > base.len() {
            bail!("--limit must be between 1 and {}", base.len());
        }
        if limit < base.len() {
            base = take_vectors(&base, limit)?;
            recompute_truth = true;
        }
    }
    if let Some(n) = args.queries {
        if n == 0 || n > queries.len() {
            bail!("--queries must be between 1 and {}", queries.len());
        }
        queries = take_vectors(&queries, n)?;
        published = take_rows(&published, n)?;
    }
    if args.k > base.len() {
        bail!(
            "--k ({}) exceeds the base set size ({})",
            args.k,
            base.len()
        );
    }

    let truth = if recompute_truth {
        None
    } else {
        Some(published)
    };

    tracing::info!(
        "{}: {} base x {} queries, dim {}, metric {}",
        args.dataset,
        base.len(),
        queries.len(),
        base.dim(),
        spec.metric.name()
    );
    tracing::info!(
        "params: M={} ef_construction={} seed={} heuristic={} keep_pruned={}",
        args.m,
        args.ef_construction,
        args.seed,
        !args.no_heuristic,
        !args.no_keep_pruned
    );

    match spec.metric {
        MetricKind::L2Squared => measure::<L2Squared>(base, queries, truth, spec, &args),
        MetricKind::Cosine => measure::<Cosine>(base, queries, truth, spec, &args),
        MetricKind::Dot => measure::<DotProduct>(base, queries, truth, spec, &args),
    }
}

fn measure<M: Metric>(
    mut base: VectorStore,
    mut queries: VectorStore,
    published: Option<IntMatrix>,
    spec: &DatasetSpec,
    args: &BenchArgs,
) -> Result<()> {
    preprocess_all::<M>(&mut base)?;
    preprocess_all::<M>(&mut queries)?;

    let truth = match published {
        Some(list) => {
            if list.dim() < args.k {
                bail!(
                    "the published ground truth holds {} neighbours, fewer than --k {}",
                    list.dim(),
                    args.k
                );
            }
            tracing::info!("recall measured against the published ground truth");
            list
        }
        None => {
            tracing::info!("base set was limited — recomputing exact ground truth");
            let start = Instant::now();
            let list = ground_truth::compute::<M>(&base, &queries, args.k, Kernel::Reference)?;
            tracing::info!("ground truth computed in {:.2?}", start.elapsed());
            list
        }
    };

    // ---- build ----------------------------------------------------------------------------
    let params = HnswParams::new(args.m)?
        .with_ef_construction(args.ef_construction)?
        .with_seed(args.seed)?
        .with_selection(args.selection())?;

    let mut index = HnswIndex::with_capacity(base.dim(), spec.metric, params, base.len())?;
    let mut searcher = index.searcher();
    let mut build_counter = DistanceCounter::new();

    let start = Instant::now();
    for vector in base.as_slice().chunks_exact(base.dim()) {
        index.insert::<M>(&mut searcher, vector, &mut build_counter)?;
    }
    let build_time = start.elapsed();

    tracing::info!(
        "build: {:.2?} ({:.0} vectors/s)",
        build_time,
        index.len() as f64 / build_time.as_secs_f64()
    );
    if let Some(count) = build_counter.count() {
        tracing::info!(
            "build distance computations: {count} ({:.0} per vector)",
            count as f64 / index.len() as f64
        );
    }

    report_graph(&index);

    // ---- sweep ----------------------------------------------------------------------------
    // Warm-up comes from the tail of the query set so it never overlaps the measured slice.
    let warmup = args.warmup.min(queries.len() / 2);
    let measured = queries.len() - warmup;
    if measured == 0 {
        bail!("--warmup leaves no queries to measure");
    }
    tracing::info!(
        "sweeping ef over {:?}: {measured} queries measured, {warmup} used for warm-up",
        args.ef
    );

    let context = SweepContext {
        queries: &queries,
        truth: &truth,
        k: args.k,
        warmup,
        measured,
    };

    let mut rows = Vec::new();
    for &ef in &args.ef {
        rows.push(sweep_one::<M>(&index, &mut searcher, &context, ef)?);
    }

    println!();
    println!(
        "  {:>5}  {:>9}  {:>10}  {:>9}  {:>9}  {:>9}  {:>12}",
        "ef", "recall@k", "QPS", "p50", "p95", "p99", "dist/query"
    );
    for row in &rows {
        println!(
            "  {:>5}  {:>9.4}  {:>10.1}  {:>9}  {:>9}  {:>9}  {:>12}",
            row.ef,
            row.recall,
            row.qps,
            format!("{:.1?}", row.p50),
            format!("{:.1?}", row.p95),
            format!("{:.1?}", row.p99),
            row.distances_per_query
                .map_or_else(|| "-".to_string(), |d| format!("{d:.0}")),
        );
    }
    println!();

    crate::report_memory();

    // The definition-of-done threshold for this dataset, checked rather than eyeballed.
    if let Some(target) = spec.recall_target {
        let best = rows.iter().map(|r| r.recall).fold(0.0f64, f64::max);
        if best + 1e-9 >= target {
            tracing::info!(
                "PASS best recall@{} is {best:.4}, target {target:.2}",
                args.k
            );
        } else {
            tracing::error!(
                "FAIL best recall@{} is {best:.4}, below the target of {target:.2}",
                args.k
            );
            bail!("recall target not met");
        }
    }

    Ok(())
}

fn sweep_one<M: Metric>(
    index: &HnswIndex,
    searcher: &mut anka_index::hnsw::Searcher,
    context: &SweepContext<'_>,
    ef: usize,
) -> Result<SweepRow> {
    let SweepContext {
        queries,
        truth,
        k,
        warmup,
        measured,
    } = *context;

    // Warm up on the tail of the query set, and throw the counts away with it.
    let mut discard = DistanceCounter::new();
    for position in (queries.len() - warmup)..queries.len() {
        index.search::<M>(searcher, queries.get(position), k, ef, &mut discard)?;
    }

    let mut counter = DistanceCounter::new();
    let mut latencies = Vec::with_capacity(measured);
    let mut hits = 0usize;

    let start = Instant::now();
    for position in 0..measured {
        let query = queries.get(position);
        let query_start = Instant::now();
        let found = index.search::<M>(searcher, query, k, ef, &mut counter)?;
        latencies.push(query_start.elapsed());

        let expected = &truth.row(position)[..k];
        hits += found
            .iter()
            .filter(|c| expected.contains(&(c.id as i32)))
            .count();
    }
    let elapsed = start.elapsed();
    latencies.sort_unstable();

    Ok(SweepRow {
        ef,
        recall: hits as f64 / (measured * k) as f64,
        qps: measured as f64 / elapsed.as_secs_f64(),
        p50: percentile(&latencies, 0.50),
        p95: percentile(&latencies, 0.95),
        p99: percentile(&latencies, 0.99),
        distances_per_query: counter.count().map(|c| c as f64 / measured as f64),
    })
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() as f64 * fraction).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn report_graph(index: &HnswIndex) {
    match index.validate() {
        Ok(()) => tracing::info!("graph invariants hold"),
        Err(violation) => tracing::error!("graph invariant violated: {violation}"),
    }

    let stats = index.graph_stats();
    tracing::info!(
        "graph: {} edges over {} layers, {} ({:.1} bytes/vector)",
        stats.edges,
        stats.max_layer + 1,
        mem::human_bytes(stats.graph_bytes as u64),
        stats.graph_bytes as f64 / stats.nodes as f64,
    );
    // Asymmetry is reported, not asserted: the pruning step legitimately produces one-way edges.
    tracing::info!(
        "one-way edges: {} of {} ({:.2}%)",
        stats.asymmetric_edges,
        stats.edges,
        stats.asymmetry_ratio() * 100.0,
    );

    println!();
    println!(
        "  {:>5}  {:>9}  {:>9}  {:>4}  {:>4}  {:>7}  {:>8}  {:>10}",
        "layer", "nodes", "edges", "cap", "max", "mean", "isolated", "bytes"
    );
    for layer in &stats.per_layer {
        println!(
            "  {:>5}  {:>9}  {:>9}  {:>4}  {:>4}  {:>7.2}  {:>8}  {:>10}",
            layer.layer,
            layer.nodes,
            layer.edges,
            layer.degree_cap,
            layer.max_degree,
            layer.mean_degree,
            layer.isolated,
            mem::human_bytes(layer.bytes as u64),
        );
    }
    println!();
}
