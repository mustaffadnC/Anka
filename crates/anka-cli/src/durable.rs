//! `anka ingest` and `anka checkpoint`: a collection that outlives the process.
//!
//! `ingest` builds one from a dataset with the log switched on, which is the only way to find out
//! what durability costs. `checkpoint` folds the log into a snapshot and starts a new one, and
//! reports what opening the collection had to repair on the way in — the number an operator
//! actually wants after a crash.

use std::path::PathBuf;
use std::time::Instant;

use anka_core::{Cosine, DotProduct, L2Squared, Metric, MetricKind, VectorStore, mem};
use anka_index::hnsw::{DistanceCounter, HnswParams, SelectionPolicy};
use anka_store::collection::{Collection, log_path, snapshot_path};
use anka_store::wal::SyncPolicy;
use anka_store::{Tail, Verify};
use anyhow::{Result, bail};
use clap::Args;

use crate::{DatasetSpec, dataset_dir, spec_for, take_vectors};

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// Dataset to ingest: siftsmall, sift1m or glove100.
    pub dataset: String,

    /// Directory the collection lives in. Created if it does not exist.
    #[arg(long)]
    pub root: PathBuf,

    /// Directory holding the datasets. Defaults to $ANKA_DATASETS, then ~/anka-datasets.
    #[arg(long, env = "ANKA_DATASETS")]
    pub datasets_dir: Option<PathBuf>,

    /// When the log is forced to disk: `always`, `every:N`, or `never`.
    ///
    /// `always` is the only setting under which an acknowledged record is claimed to survive.
    #[arg(long, default_value = "always")]
    pub fsync: String,

    /// Ingest only the first N vectors. `always` costs an fsync per record, so the whole of
    /// SIFT1M is a long wait — which is itself the measurement.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Connections per node per layer.
    #[arg(long, default_value_t = 16)]
    pub m: usize,

    /// Candidate list size during construction.
    #[arg(long, default_value_t = 200)]
    pub ef_construction: usize,

    /// Seed for layer assignment.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Checkpoint every N inserts. Off by default.
    #[arg(long)]
    pub checkpoint_every: Option<usize>,
}

#[derive(Args, Debug)]
pub struct CheckpointArgs {
    /// Directory the collection lives in.
    pub root: PathBuf,

    /// When the log is forced to disk: `always`, `every:N`, or `never`.
    #[arg(long, default_value = "always")]
    pub fsync: String,

    /// Checksum the snapshot body on the way in. Reads the whole file.
    #[arg(long)]
    pub verify: bool,
}

/// Parses `always`, `every:N` or `never`.
fn sync_policy(spec: &str) -> Result<SyncPolicy> {
    match spec {
        "always" => Ok(SyncPolicy::Always),
        "never" => Ok(SyncPolicy::Never),
        other => match other.strip_prefix("every:") {
            Some(n) => {
                let n: u32 = n.parse().map_err(|_| {
                    anyhow::anyhow!("--fsync every:N needs a number, got '{other}'")
                })?;
                let n = std::num::NonZeroU32::new(n)
                    .ok_or_else(|| anyhow::anyhow!("--fsync every:0 is not a policy"))?;
                Ok(SyncPolicy::EveryN(n))
            }
            None => bail!("--fsync must be always, every:N or never, got '{other}'"),
        },
    }
}

fn describe(policy: SyncPolicy) -> String {
    match policy {
        SyncPolicy::Always => "always".to_string(),
        SyncPolicy::EveryN(n) => format!("every {n} records"),
        SyncPolicy::Never => "never".to_string(),
    }
}

pub fn ingest(args: IngestArgs) -> Result<()> {
    let spec = spec_for(&args.dataset)?;
    let dir = dataset_dir(&args.dataset, spec, args.datasets_dir.clone())?;
    let policy = sync_policy(&args.fsync)?;

    let mut base = anka_core::dataset::read_fvecs(dir.join(format!("{}_base.fvecs", spec.prefix)))?;
    if let Some(limit) = args.limit {
        if limit == 0 || limit > base.len() {
            bail!("--limit must be between 1 and {}", base.len());
        }
        base = take_vectors(&base, limit)?;
    }

    tracing::info!(
        "{}: ingesting {} vectors of dim {} into {}",
        args.dataset,
        base.len(),
        base.dim(),
        args.root.display()
    );
    tracing::info!(
        "fsync {}, checkpoint {}",
        describe(policy),
        args.checkpoint_every
            .map_or_else(|| "never".to_string(), |n| format!("every {n} inserts"))
    );

    match spec.metric {
        MetricKind::L2Squared => write_all::<L2Squared>(base, spec, &args, policy),
        MetricKind::Cosine => write_all::<Cosine>(base, spec, &args, policy),
        MetricKind::Dot => write_all::<DotProduct>(base, spec, &args, policy),
    }
}

fn write_all<M: Metric>(
    mut base: VectorStore,
    spec: &DatasetSpec,
    args: &IngestArgs,
    policy: SyncPolicy,
) -> Result<()> {
    anka_core::preprocess_all::<M>(&mut base)?;

    let params = HnswParams::new(args.m)?
        .with_ef_construction(args.ef_construction)?
        .with_seed(args.seed)?
        .with_selection(SelectionPolicy::default())?;

    let mut collection = Collection::create(&args.root, base.dim(), spec.metric, params, policy)?;
    let mut searcher = collection.searcher();
    let mut counter = DistanceCounter::new();

    let view = base.view();
    let start = Instant::now();
    let mut checkpoints = 0usize;

    for position in 0..base.len() {
        collection.insert::<M>(
            position as u64,
            view.get(position),
            &mut searcher,
            &mut counter,
        )?;
        if let Some(every) = args.checkpoint_every
            && (position + 1).is_multiple_of(every)
        {
            collection.checkpoint()?;
            checkpoints += 1;
        }
    }
    let elapsed = start.elapsed();

    tracing::info!(
        "ingested {} vectors in {:.2?} ({:.0}/s), {checkpoints} checkpoints",
        collection.len(),
        elapsed,
        collection.len() as f64 / elapsed.as_secs_f64()
    );
    report_files(&args.root)?;
    crate::report_memory();
    Ok(())
}

pub fn checkpoint(args: CheckpointArgs) -> Result<()> {
    let policy = sync_policy(&args.fsync)?;
    let verify = if args.verify {
        Verify::Body
    } else {
        Verify::Header
    };

    let start = Instant::now();
    let (mut collection, opened) = Collection::open(&args.root, policy, verify)?;
    let open_time = start.elapsed();

    tracing::info!(
        "opened {} in {:.2?}: {} vectors, {} records replayed, {} already in the snapshot",
        args.root.display(),
        open_time,
        collection.len(),
        opened.replayed,
        opened.skipped
    );
    match opened.tail {
        Tail::Clean => tracing::info!("the log ended cleanly — nothing was discarded"),
        // Not a warning about a bug: this is what a crash looks like, and the point of the log is
        // that it is recoverable. Reported so an operator knows a restart was not graceful.
        Tail::Torn(torn) => tracing::warn!("the log was torn and has been repaired: {torn:?}"),
        Tail::SequenceGap { expected, found } => tracing::warn!(
            "the log jumped from {expected} to {found}; everything from there was discarded"
        ),
    }

    let before = std::fs::metadata(log_path(&args.root))
        .map(|m| m.len())
        .unwrap_or(0);
    let start = Instant::now();
    let seq = collection.checkpoint()?;
    let elapsed = start.elapsed();

    tracing::info!(
        "checkpointed at sequence {seq} in {:.2?}; the log went from {} to {}",
        elapsed,
        mem::human_bytes(before),
        mem::human_bytes(std::fs::metadata(log_path(&args.root))?.len())
    );
    report_files(&args.root)?;
    Ok(())
}

fn report_files(root: &std::path::Path) -> Result<()> {
    let snapshot = std::fs::metadata(snapshot_path(root))?.len();
    let log = std::fs::metadata(log_path(root))
        .map(|m| m.len())
        .unwrap_or(0);
    tracing::info!(
        "on disk: snapshot {}, log {}",
        mem::human_bytes(snapshot),
        mem::human_bytes(log)
    );
    Ok(())
}
