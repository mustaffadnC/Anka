//! Anka command-line interface.
//!
//! Phase 0 provides `anka load`; phase 1 adds `anka gt`, which is where the project's arithmetic
//! gets checked against something it did not produce itself. `build`, `query` and `checkpoint`
//! follow in later phases.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anka_core::dataset::{self, IntMatrix};
use anka_core::{
    Cosine, DotProduct, L2Squared, Metric, MetricKind, VectorError, VectorStore, mem,
    preprocess_all,
};
use anka_index::brute_force::Kernel;
use anka_index::ground_truth::{self, Agreement, DistanceAgreement};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

mod bench;

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
    /// Compute exact ground truth and check it against the dataset's published list.
    Gt(GroundTruthArgs),
    /// Build an HNSW index, then sweep `ef` measuring recall and throughput.
    Bench(bench::BenchArgs),
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

#[derive(Args, Debug)]
struct GroundTruthArgs {
    /// Dataset to check: siftsmall, sift1m or glove100.
    dataset: String,

    /// Directory holding the datasets. Defaults to $ANKA_DATASETS, then ~/anka-datasets.
    #[arg(long, env = "ANKA_DATASETS")]
    datasets_dir: Option<PathBuf>,

    /// Neighbours per query. Both SIFT and GloVe publish 100.
    #[arg(long, default_value_t = 100)]
    k: usize,

    /// Use only the first N queries. The whole set when omitted.
    #[arg(long)]
    queries: Option<usize>,

    /// Relative tolerance for calling two neighbours equidistant.
    #[arg(long, default_value_t = 1e-6)]
    rtol: f64,

    /// Write the reference-kernel list to this path as `.ivecs`.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Where a dataset's files live, what shape they must have, and which metric it is scored with.
///
/// The expected counts are asserted, not merely printed. Phase 0's definition of done is that
/// SIFT1M loads as exactly 1 000 000 x 128; a loader that quietly accepted 999 999 would
/// invalidate every recall figure measured on top of it.
pub(crate) struct DatasetSpec {
    pub dir: &'static str,
    pub prefix: &'static str,
    pub dim: usize,
    pub base: usize,
    pub queries: usize,
    pub neighbours: usize,
    pub metric: MetricKind,
    /// `recall@10` the phase 2 definition of done requires on this dataset, where the spec sets
    /// one. Checked by `anka bench` rather than left to be eyeballed off a table.
    pub recall_target: Option<f64>,
}

const SIFTSMALL: DatasetSpec = DatasetSpec {
    dir: "siftsmall",
    prefix: "siftsmall",
    dim: 128,
    base: 10_000,
    queries: 100,
    neighbours: 100,
    metric: MetricKind::L2Squared,
    // Not a spec threshold: 10 000 points with M=16 should be effectively exact, so anything
    // below this means something is broken rather than merely approximate.
    recall_target: Some(0.99),
};

const SIFT1M: DatasetSpec = DatasetSpec {
    dir: "sift",
    prefix: "sift",
    dim: 128,
    base: 1_000_000,
    queries: 10_000,
    neighbours: 100,
    metric: MetricKind::L2Squared,
    recall_target: Some(0.95),
};

const GLOVE100: DatasetSpec = DatasetSpec {
    dir: "glove100",
    prefix: "glove100",
    dim: 100,
    base: 1_183_514,
    queries: 10_000,
    neighbours: 100,
    metric: MetricKind::Cosine,
    // Lower than SIFT on purpose: GloVe is where graph methods separate, and the spec sets the
    // bar accordingly.
    recall_target: Some(0.90),
};

fn main() -> Result<()> {
    init_tracing();
    match Cli::parse().command {
        Command::Load(args) => load(args),
        Command::Gt(args) => check_ground_truth(args),
        Command::Bench(args) => bench::run(args),
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

// ---------------------------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------------------------

fn load(args: LoadArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = dataset_dir(&args.dataset, spec, args.datasets_dir)?;
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

    report_memory();
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// gt
// ---------------------------------------------------------------------------------------------

fn check_ground_truth(args: GroundTruthArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = dataset_dir(&args.dataset, spec, args.datasets_dir.clone())?;

    let base = dataset::read_fvecs(dir.join(format!("{}_base.fvecs", spec.prefix)))?;
    let mut queries = dataset::read_fvecs(dir.join(format!("{}_query.fvecs", spec.prefix)))?;
    let mut published =
        dataset::read_ivecs(dir.join(format!("{}_groundtruth.ivecs", spec.prefix)))?;

    check_shape("base", base.len(), base.dim(), spec.base, spec.dim)?;
    check_shape(
        "groundtruth",
        published.len(),
        published.dim(),
        spec.queries,
        spec.neighbours,
    )?;

    if let Some(n) = args.queries {
        if n == 0 || n > queries.len() {
            bail!("--queries must be between 1 and {}", queries.len());
        }
        queries = take_vectors(&queries, n)?;
        published = take_rows(&published, n)?;
        tracing::warn!("restricted to the first {n} queries — not a full verification run");
    }

    tracing::info!(
        "{}: {} base x {} queries, k={}, metric={}",
        args.dataset,
        base.len(),
        queries.len(),
        args.k,
        spec.metric.name()
    );

    // The dispatch boundary described in docs/DESIGN.md, section 4: the metric arrives as data,
    // and from here down every kernel is monomorphised.
    match spec.metric {
        MetricKind::L2Squared => verify::<L2Squared>(base, queries, &published, &args),
        MetricKind::Cosine => verify::<Cosine>(base, queries, &published, &args),
        MetricKind::Dot => verify::<DotProduct>(base, queries, &published, &args),
    }
}

fn verify<M: Metric>(
    mut base: VectorStore,
    mut queries: VectorStore,
    published: &IntMatrix,
    args: &GroundTruthArgs,
) -> Result<()> {
    // Cosine is only equivalent to a dot product on unit vectors, and GloVe does not ship
    // normalised. Skipping this does not fail loudly — it produces plausible, meaningless recall.
    let (_, elapsed) = timed(|| -> Result<(), VectorError> {
        preprocess_all::<M>(&mut base)?;
        preprocess_all::<M>(&mut queries)?;
        Ok(())
    })?;
    tracing::info!("preprocessed with {} in {:.2?}", M::NAME, elapsed);

    let (reference_list, reference_time) =
        timed(|| ground_truth::compute::<M>(&base, &queries, args.k, Kernel::Reference))?;
    tracing::info!(
        "reference kernel: {:.2?} ({:.1} queries/s)",
        reference_time,
        queries.len() as f64 / reference_time.as_secs_f64()
    );

    let published_agreement = ground_truth::compare(&reference_list, published)?;
    report_agreement("reference vs published", &published_agreement);

    // Id equality alone cannot tell "our distances are wrong" apart from "the published list
    // broke a tie differently". Only recomputing the distances at the differing positions can,
    // and the difference decides whether there is a bug to fix or a convention to document.
    let published_equivalence = ground_truth::distance_equivalence::<M>(
        &base,
        &queries,
        &reference_list,
        published,
        args.rtol,
    )?;
    report_equivalence(&published_equivalence, args.rtol);

    let (fast_list, fast_time) =
        timed(|| ground_truth::compute::<M>(&base, &queries, args.k, Kernel::Fast))?;
    tracing::info!(
        "simd kernel:      {:.2?} ({:.1} queries/s), speed-up {:.2}x",
        fast_time,
        queries.len() as f64 / fast_time.as_secs_f64(),
        reference_time.div_duration_f64(fast_time)
    );

    let kernel_agreement = ground_truth::compare(&fast_list, &reference_list)?;
    report_agreement("simd vs reference", &kernel_agreement);

    let equivalence = ground_truth::distance_equivalence::<M>(
        &base,
        &queries,
        &fast_list,
        &reference_list,
        args.rtol,
    )?;
    report_equivalence(&equivalence, args.rtol);

    if let Some(path) = &args.out {
        dataset::write_ivecs(path, args.k, reference_list.as_slice())?;
        tracing::info!("wrote the reference list to {}", path.display());
    }

    report_memory();

    // The verdict is an exit code, so this can gate a script or a CI job.
    //
    // The criterion is that our **distance profile** matches the published one: for every query
    // and every rank, the distance to our neighbour equals the distance to theirs. Since both
    // lists are sorted ascending by distance, that makes the two sorted distance sequences
    // identical — and a list of k items whose distance sequence equals that of a known-exact
    // top-k *is* an exact top-k. Anything less than exact would have to contain an item beyond
    // the true k-th distance, which would show up here.
    //
    // Neither id equality nor set equality can be required, and neither implies correctness:
    //
    // - Where two neighbours are exactly equidistant, the published list's order comes from an
    //   undocumented tie-break. No rule of ours reproduces it, and none is more correct.
    // - Where a tie *straddles the k-th position*, the top-k set is not uniquely defined at all:
    //   several vectors are tied for the last slot and any choice is an exact answer. So the
    //   sets can differ while both lists are exactly right.
    //
    // Both are reported, because a sudden change in either is a signal worth noticing.
    let mut failed = false;

    if published_equivalence.is_equivalent() {
        tracing::info!(
            "PASS reference kernel is exact: distances match the published list at all {} \
             positions ({} of them via a different but equidistant neighbour)",
            published_agreement.queries * published_agreement.k,
            published_equivalence.differing_positions,
        );
        if published_agreement.same_set_rows < published_agreement.queries {
            tracing::info!(
                "      {} queries select different ids among equidistant candidates — ties \
                 straddling rank {} leave the top-k set undefined, and both lists are exact",
                published_agreement.queries - published_agreement.same_set_rows,
                published_agreement.k,
            );
        }
    } else {
        tracing::error!(
            "FAIL reference kernel is not exact: {} of {} differing positions are at genuinely \
             different distances (largest relative gap {:.3e})",
            published_equivalence.differing_positions - published_equivalence.equivalent_positions,
            published_equivalence.differing_positions,
            published_equivalence.max_relative_gap,
        );
        failed = true;
    }

    if equivalence.is_equivalent() {
        tracing::info!("PASS simd kernel is distance-equivalent to the reference");
    } else {
        tracing::error!("FAIL simd kernel disagrees on distances, not just on ties");
        failed = true;
    }

    if failed {
        bail!("ground truth verification failed");
    }
    Ok(())
}

fn report_agreement(label: &str, agreement: &Agreement) {
    tracing::info!(
        "{label}: rows {}/{} ({:.4}%), same-set {}/{}, positions {}/{} ({:.4}%)",
        agreement.identical_rows,
        agreement.queries,
        agreement.row_ratio() * 100.0,
        agreement.same_set_rows,
        agreement.queries,
        agreement.identical_positions,
        agreement.queries * agreement.k,
        agreement.position_ratio() * 100.0,
    );
    for example in &agreement.examples {
        tracing::warn!(
            "  query {} first differs at rank {}: ours {:?} vs reference {:?}",
            example.query,
            example.first_difference,
            window(&example.ours, example.first_difference),
            window(&example.reference, example.first_difference),
        );
    }
}

fn report_equivalence(agreement: &DistanceAgreement, rtol: f64) {
    if agreement.differing_positions == 0 {
        tracing::info!(
            "distance equivalence: ids identical at all {} positions",
            agreement.queries * agreement.k
        );
        return;
    }
    tracing::info!(
        "distance equivalence: {}/{} differing positions are equidistant within {:e} \
         (largest relative gap {:.3e})",
        agreement.equivalent_positions,
        agreement.differing_positions,
        rtol,
        agreement.max_relative_gap,
    );
    for example in &agreement.examples {
        tracing::warn!(
            "  query {} rank {}: ours id={} d={} vs reference id={} d={} gap={:.3e}",
            example.query,
            example.rank,
            example.our_id,
            example.our_distance,
            example.reference_id,
            example.reference_distance,
            example.relative_gap,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------------------------

pub(crate) fn spec_for(name: &str) -> Result<&'static DatasetSpec> {
    Ok(match name {
        "siftsmall" => &SIFTSMALL,
        "sift1m" | "sift" => &SIFT1M,
        "glove100" | "glove" => &GLOVE100,
        other => bail!("unknown dataset '{other}' (expected: siftsmall, sift1m, glove100)"),
    })
}

pub(crate) fn dataset_dir(
    name: &str,
    spec: &DatasetSpec,
    explicit: Option<PathBuf>,
) -> Result<PathBuf> {
    let dir = resolve_root(explicit)?.join(spec.dir);
    if !dir.is_dir() {
        bail!(
            "{} does not exist — run ./scripts/download_datasets.sh {name}",
            dir.display()
        );
    }
    Ok(dir)
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

pub(crate) fn take_vectors(store: &VectorStore, count: usize) -> Result<VectorStore> {
    let dim = store.dim();
    let view = store.view();
    let mut data = Vec::with_capacity(count * dim);
    for position in 0..count {
        data.extend_from_slice(view.get(position));
    }
    Ok(VectorStore::from_flat(dim, data)?)
}

pub(crate) fn take_rows(matrix: &IntMatrix, count: usize) -> Result<IntMatrix> {
    let dim = matrix.dim();
    Ok(IntMatrix::new(
        dim,
        matrix.as_slice()[..count * dim].to_vec(),
    )?)
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

pub(crate) fn report_memory() {
    match mem::resident_set_size() {
        Some(usage) => tracing::info!(
            "resident set size: {} (peak {})",
            mem::human_bytes(usage.current_bytes),
            mem::human_bytes(usage.peak_bytes),
        ),
        // Not an error: measurements are taken on Linux, and refusing to invent a number
        // elsewhere is the point. See docs/DESIGN.md, section 10.
        None => tracing::warn!("resident set size unavailable on this platform (Linux only)"),
    }
}

/// A few ids either side of `rank`.
///
/// Printing the head of the list instead would show two identical prefixes, since disagreements
/// cluster at deep ranks where neighbours bunch together.
fn window(ids: &[i32], rank: usize) -> Vec<i32> {
    let start = rank.saturating_sub(1);
    let end = (rank + 4).min(ids.len());
    ids[start..end].to_vec()
}
