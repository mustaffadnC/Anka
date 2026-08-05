//! A durable collection: an index, its snapshot, and its log, kept in step.
//!
//! ```text
//! root/
//!   collection.anka   the snapshot
//!   wal.log           records written since it
//! ```
//!
//! **The ordering rule, which is the reason this type exists.** A durable insert goes: draw the
//! level, write the record, sync it to the extent the policy promises, *then* update the index.
//! Reversed, a search can return a record that never reached disk — the failure that makes a
//! database untrustworthy rather than merely lossy. Putting the two operations behind one method
//! is what stops the order from being a thing anyone has to remember.
//!
//! **The level is drawn before the record is written, not by the insert.** The log has to carry
//! the level the index will actually use, so replay reproduces the same graph. That is why
//! [`anka_index::HnswIndex::draw_level`] is separate from `insert`.
//!
//! **Checkpointing.** The snapshot is written with `wal_seq` set to the sequence number the
//! CHECKPOINT record is *about* to take, then that record is appended to the old log, then a fresh
//! log is started after it. A crash at any point leaves a consistent pair:
//!
//! | crash after | recovery finds |
//! |---|---|
//! | nothing | old snapshot + old log — unchanged |
//! | the snapshot lands | new snapshot + old log; every record it names is skipped |
//! | the checkpoint record | same, and the marker is in the log an operator can read |
//! | the log is replaced | new snapshot + a log that starts right after it |

use std::path::{Path, PathBuf};

use anka_core::{ExternalId, Metric, MetricKind, NodeId, VectorStore};
use anka_index::{DistanceCounter, HnswIndex, HnswParams, Searcher};

use crate::error::RecoveryError;
use crate::recovery::{self, Tail};
use crate::snapshot::{self, Verify};
use crate::wal::{Record, SyncPolicy, WalWriter};

pub const SNAPSHOT_FILE: &str = "collection.anka";
pub const LOG_FILE: &str = "wal.log";

/// An index that survives a restart.
pub struct Collection {
    index: HnswIndex,
    log: WalWriter,
    root: PathBuf,
}

/// What opening a collection had to do to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Opened {
    pub tail: Tail,
    pub replayed: usize,
    pub skipped: usize,
}

impl Collection {
    /// Creates an empty collection at `root`, replacing anything already there.
    pub fn create(
        root: &Path,
        dim: usize,
        metric: MetricKind,
        params: HnswParams,
        policy: SyncPolicy,
    ) -> Result<Self, RecoveryError> {
        std::fs::create_dir_all(root)
            .map_err(|e| crate::error::WalError::io(root, e))
            .map_err(RecoveryError::Wal)?;

        let index = HnswIndex::new(dim, metric, params).map_err(crate::error::WalError::Index)?;
        // Snapshot first, so a collection is never a log with nothing to replay onto.
        snapshot::write(&index, &snapshot_path(root), 0)?;
        let log = WalWriter::create(&log_path(root), policy, 1)?;

        Ok(Self {
            index,
            log,
            root: root.to_path_buf(),
        })
    }

    /// Opens the collection at `root`, replaying whatever the log holds past the snapshot.
    ///
    /// The torn tail, if there is one, is cut off before this returns — so the first append after
    /// opening lands on a log with no hole in it. Leaving it would give every later recovery a
    /// place to stop, silently losing everything written after the crash.
    pub fn open(
        root: &Path,
        policy: SyncPolicy,
        verify: Verify,
    ) -> Result<(Self, Opened), RecoveryError> {
        let recovered = recovery::open(&snapshot_path(root), &log_path(root), verify)?;
        let report = Opened {
            tail: recovered.tail,
            replayed: recovered.replayed,
            skipped: recovered.skipped,
        };

        let log = if recovered.valid_bytes == 0 {
            // No log at all: checkpointed and untouched since.
            WalWriter::create(&log_path(root), policy, recovered.next_seq)?
        } else {
            WalWriter::reopen(
                &log_path(root),
                policy,
                recovered.next_seq,
                recovered.valid_bytes,
            )?
        };

        Ok((
            Self {
                index: recovered.index,
                log,
                root: root.to_path_buf(),
            },
            report,
        ))
    }

    pub fn index(&self) -> &HnswIndex {
        &self.index
    }

    pub fn vectors(&self) -> &VectorStore {
        self.index.vectors()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn searcher(&self) -> Searcher {
        self.index.searcher()
    }

    /// The sequence number the next durable write will take.
    pub fn next_seq(&self) -> u64 {
        self.log.next_seq()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Inserts a vector durably.
    ///
    /// Returns once the record has reached disk to the extent [`SyncPolicy`] promises. The index
    /// is only updated after that, so nothing this collection can return has failed to be logged.
    pub fn insert<M: Metric>(
        &mut self,
        external_id: ExternalId,
        vector: &[f32],
        searcher: &mut Searcher,
        counter: &mut DistanceCounter,
    ) -> Result<NodeId, RecoveryError> {
        // Drawn here so the record carries the level the index will use. The draw counts even if
        // the write below fails, which is why the snapshot records the count rather than deriving
        // it from the node total.
        let level = self.index.draw_level();

        self.log.append(&Record::Insert {
            external_id,
            level: level as u8,
            vector: vector.to_vec(),
            metadata: Vec::new(),
        })?;

        Ok(self
            .index
            .insert_at_level::<M>(searcher, vector, level, counter)
            .map_err(crate::error::WalError::Index)?)
    }

    /// Writes a snapshot of the current state and starts a fresh log after it.
    ///
    /// See this module's header for what a crash at each step leaves behind.
    pub fn checkpoint(&mut self) -> Result<u64, RecoveryError> {
        // The snapshot claims to contain the checkpoint record that has not been written yet.
        // That is deliberate: it makes the new log start at `seq + 1` with no gap, so recovery
        // does not see the marker's number missing and conclude records were lost.
        let seq = self.log.next_seq();
        snapshot::write(&self.index, &snapshot_path(&self.root), seq)?;

        // Appended to the *old* log. If the replacement below never happens, a reader still sees
        // where the checkpoint was, and recovery skips everything up to it anyway.
        self.log.append(&Record::Checkpoint {
            snapshot_wal_seq: seq,
        })?;

        self.log = WalWriter::create(&log_path(&self.root), self.log.policy(), seq + 1)?;
        Ok(seq)
    }

    /// Forces the log to disk regardless of policy.
    pub fn sync(&mut self) -> Result<(), RecoveryError> {
        Ok(self.log.sync()?)
    }
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("root", &self.root)
            .field("len", &self.index.len())
            .field("next_seq", &self.log.next_seq())
            .finish_non_exhaustive()
    }
}

pub fn snapshot_path(root: &Path) -> PathBuf {
    root.join(SNAPSHOT_FILE)
}

pub fn log_path(root: &Path) -> PathBuf {
    root.join(LOG_FILE)
}

#[cfg(test)]
mod tests {
    use anka_core::L2Squared;
    use tempfile::TempDir;

    use super::*;

    const DIM: usize = 4;

    fn vector(id: u64) -> Vec<f32> {
        (0..DIM).map(|i| id as f32 * 3.0 + i as f32).collect()
    }

    fn fill(collection: &mut Collection, ids: impl IntoIterator<Item = u64>) {
        let mut searcher = collection.searcher();
        let mut counter = DistanceCounter::new();
        for id in ids {
            collection
                .insert::<L2Squared>(id, &vector(id), &mut searcher, &mut counter)
                .unwrap();
        }
    }

    fn answers(collection: &Collection) -> Vec<Vec<anka_core::Candidate>> {
        let mut searcher = collection.searcher();
        let mut counter = DistanceCounter::new();
        (0..10u64)
            .map(|id| {
                collection
                    .index()
                    .search::<L2Squared>(&mut searcher, &vector(id * 2), 5, 32, &mut counter)
                    .unwrap()
            })
            .collect()
    }

    fn open(root: &Path) -> (Collection, Opened) {
        Collection::open(root, SyncPolicy::Always, Verify::Body).unwrap()
    }

    #[test]
    fn a_collection_survives_being_closed_and_reopened() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut collection, 1..=40);
        let before = answers(&collection);
        let seq = collection.next_seq();
        drop(collection);

        let (reopened, report) = open(dir.path());
        assert_eq!(report.tail, Tail::Clean);
        assert_eq!(report.replayed, 40);
        assert_eq!(reopened.len(), 40);
        assert_eq!(reopened.next_seq(), seq);
        assert_eq!(answers(&reopened), before);
        reopened.index().validate().unwrap();
    }

    /// A checkpoint changes where the state lives, not what it is.
    #[test]
    fn checkpointing_does_not_change_the_answers() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut collection, 1..=30);
        let before = answers(&collection);

        let seq = collection.checkpoint().unwrap();
        assert_eq!(answers(&collection), before, "in memory");
        assert_eq!(collection.next_seq(), seq + 1);

        // The log now holds only its header: everything before it is in the snapshot.
        let log = std::fs::read(log_path(dir.path())).unwrap();
        assert_eq!(log.len(), crate::wal::HEADER_BYTES);
        assert_eq!(snapshot::wal_seq(&snapshot_path(dir.path())).unwrap(), seq);

        drop(collection);
        let (reopened, report) = open(dir.path());
        assert_eq!(report.replayed, 0, "everything came from the snapshot");
        assert_eq!(report.skipped, 0);
        assert_eq!(reopened.len(), 30);
        assert_eq!(answers(&reopened), before);
    }

    /// The case a checkpoint exists for: snapshot, keep writing, restart. The two halves have to
    /// join up into the same index as one that never stopped.
    #[test]
    fn writes_after_a_checkpoint_are_replayed_onto_it() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut collection, 1..=20);
        collection.checkpoint().unwrap();
        fill(&mut collection, 21..=35);
        let before = answers(&collection);
        drop(collection);

        let (reopened, report) = open(dir.path());
        assert_eq!(report.replayed, 15);
        assert_eq!(report.skipped, 0);
        assert_eq!(reopened.len(), 35);
        assert_eq!(answers(&reopened), before);

        // And the same index as one built without ever stopping.
        let straight = TempDir::new().unwrap();
        let mut plain = Collection::create(
            straight.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut plain, 1..=35);
        assert_eq!(answers(&plain), before);
    }

    /// Checkpointing repeatedly must not drift: each one starts a log the next recovery reads
    /// cleanly, with sequence numbers that keep climbing.
    #[test]
    fn repeated_checkpoints_stay_consistent() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();

        let mut last_seq = 0;
        for round in 0..4u64 {
            fill(&mut collection, round * 10 + 1..=round * 10 + 10);
            let seq = collection.checkpoint().unwrap();
            assert!(seq > last_seq, "sequence numbers climb across checkpoints");
            last_seq = seq;
        }
        let before = answers(&collection);
        drop(collection);

        let (reopened, report) = open(dir.path());
        assert_eq!(report.tail, Tail::Clean);
        assert_eq!(reopened.len(), 40);
        assert_eq!(answers(&reopened), before);
        reopened.index().validate().unwrap();
    }

    /// A crash between the snapshot landing and the log being replaced. The old log is still
    /// there, full of records the new snapshot already contains, and they must be skipped rather
    /// than applied a second time.
    #[test]
    fn a_crash_between_snapshot_and_log_replacement_recovers() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut collection, 1..=25);
        let before = answers(&collection);
        let log_before = std::fs::read(log_path(dir.path())).unwrap();

        let seq = collection.checkpoint().unwrap();
        drop(collection);

        // Put the pre-checkpoint log back: this is the state a crash right after the rename of
        // the snapshot leaves behind.
        std::fs::write(log_path(dir.path()), &log_before).unwrap();
        assert_eq!(snapshot::wal_seq(&snapshot_path(dir.path())).unwrap(), seq);

        let (reopened, report) = open(dir.path());
        assert_eq!(report.replayed, 0);
        assert_eq!(report.skipped, 25, "the snapshot already had them");
        assert_eq!(reopened.len(), 25);
        assert_eq!(answers(&reopened), before);
    }

    /// Opening cuts a torn tail off, so the next append does not land after a hole.
    #[test]
    fn opening_repairs_a_torn_log_before_writing_to_it() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();
        fill(&mut collection, 1..=20);
        drop(collection);

        let path = log_path(dir.path());
        let whole = std::fs::read(&path).unwrap();
        std::fs::write(&path, &whole[..whole.len() - 9]).unwrap();

        let (mut reopened, report) = open(dir.path());
        assert!(report.tail.truncated());
        assert_eq!(report.replayed, 19, "the last record was incomplete");
        assert_eq!(reopened.len(), 19);

        fill(&mut reopened, 100..=105);
        let after = answers(&reopened);
        drop(reopened);

        // Reading it back finds one log with no hole in it.
        let (again, report) = open(dir.path());
        assert_eq!(report.tail, Tail::Clean);
        assert_eq!(again.len(), 25);
        assert_eq!(answers(&again), after);
    }

    /// Under `Always`, a record that `insert` returned from has been synced. This is the property
    /// the crash tests exercise for real; here it is pinned at the level of the ordering rule.
    #[test]
    fn the_log_is_written_before_the_index_is_updated() {
        let dir = TempDir::new().unwrap();
        let mut collection = Collection::create(
            dir.path(),
            DIM,
            MetricKind::L2Squared,
            HnswParams::default(),
            SyncPolicy::Always,
        )
        .unwrap();

        let mut searcher = collection.searcher();
        let mut counter = DistanceCounter::new();
        for id in 1..=12u64 {
            collection
                .insert::<L2Squared>(id, &vector(id), &mut searcher, &mut counter)
                .unwrap();

            // After every acknowledged insert, the log on disk already describes the index in
            // memory — read it without touching the live collection.
            let recovered = recovery::replay(
                HnswIndex::new(DIM, MetricKind::L2Squared, HnswParams::default()).unwrap(),
                0,
                &log_path(dir.path()),
            )
            .unwrap();
            assert_eq!(recovered.replayed as u64, id);
            assert_eq!(recovered.index.len(), collection.len());
            assert_eq!(recovered.tail, Tail::Clean);
        }
    }
}
