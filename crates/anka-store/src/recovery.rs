//! Rebuilding an index from a snapshot plus whatever the log holds after it.
//!
//! ```text
//! 1. load the snapshot; S = header.wal_seq  (no snapshot: S = 0 and an empty index)
//! 2. validate the log header — a mismatch is an error, and the file is left alone
//! 3. read records in order:  seq <= S  skip, the snapshot already has them
//!                            seq >  S  replay
//! 4. stop cleanly at the first of: fewer than 8 bytes left, payload short of `len`,
//!    checksum mismatch, or a sequence number that is not the previous plus one
//! 5. the caller truncates to that offset and appends from there
//! ```
//!
//! **Stopping is not failing.** A log is *expected* to end mid-record: that is what a process
//! dying during a write leaves behind. Recovery reports where the intact records end and why,
//! and the tail is discarded. What it will not do is skip past a bad record and keep going —
//! resurrecting data on the far side of a hole produces an index nobody can reason about, and
//! the records after a gap were never acknowledged in an order that survived.
//!
//! **Never draw a level.** Replay inserts at the level in the record. Drawing a fresh one would
//! rebuild a different graph from the same log, which is exactly what makes "identical results
//! after a restart" untestable — so [`anka_index::HnswIndex::insert_at_level`] exists, and the
//! generator's draw count is carried in the snapshot header so it stays in step.

use std::path::Path;

use anka_core::{Cosine, DotProduct, L2Squared, Metric, MetricKind};
use anka_index::hnsw::MAX_LEVEL;
use anka_index::{DistanceCounter, HnswIndex, IndexError};

use crate::error::{RecoveryError, WalError};
use crate::snapshot::{self, Verify};
use crate::wal::{self, Next, Record, Torn};

/// How the log ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// The last record ended exactly where the file does. A clean shutdown looks like this.
    Clean,
    /// The log stops partway through a record — the shape a crash during a write leaves.
    Torn(Torn),
    /// A sequence number is not the previous plus one. Everything from here on is discarded:
    /// the records are intact, but the log they belong to is not the one that was being written.
    SequenceGap { expected: u64, found: u64 },
}

impl Tail {
    /// Whether anything was discarded.
    pub fn truncated(self) -> bool {
        !matches!(self, Tail::Clean)
    }
}

/// An index brought back up to date, and what it took.
#[derive(Debug)]
pub struct Recovered {
    pub index: HnswIndex,
    /// Where the intact records end. The log is truncated here before anything is appended.
    pub valid_bytes: u64,
    /// The sequence number the next append takes.
    pub next_seq: u64,
    pub tail: Tail,
    /// Records replayed into the index.
    pub replayed: usize,
    /// Records the snapshot already contained.
    pub skipped: usize,
}

/// Loads a snapshot and replays the log on top of it.
pub fn open(snapshot: &Path, log: &Path, verify: Verify) -> Result<Recovered, RecoveryError> {
    let index = snapshot::load(snapshot, verify)?;
    let contains = snapshot::wal_seq(snapshot)?;
    replay(index, contains, log)
}

/// Replays `log` onto `index`, which already contains every record up to `contains`.
///
/// Split out from [`open`] so an index that came from somewhere else — a fresh empty one, a test
/// fixture — can be brought up to date the same way.
pub fn replay(index: HnswIndex, contains: u64, log: &Path) -> Result<Recovered, RecoveryError> {
    let bytes = match std::fs::read(log) {
        Ok(bytes) => bytes,
        // No log at all is a valid state: a collection that has been checkpointed and not written
        // to since. Anything else is a real I/O failure and is reported.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Recovered {
                index,
                valid_bytes: 0,
                next_seq: contains + 1,
                tail: Tail::Clean,
                replayed: 0,
                skipped: 0,
            });
        }
        Err(e) => return Err(WalError::io(log, e).into()),
    };

    let mut offset = wal::read_header(&bytes)?;
    let dim = index.dim();

    match index.metric() {
        MetricKind::L2Squared => run::<L2Squared>(index, contains, &bytes, &mut offset, dim),
        MetricKind::Cosine => run::<Cosine>(index, contains, &bytes, &mut offset, dim),
        MetricKind::Dot => run::<DotProduct>(index, contains, &bytes, &mut offset, dim),
    }
}

fn run<M: Metric>(
    mut index: HnswIndex,
    contains: u64,
    bytes: &[u8],
    offset: &mut usize,
    dim: usize,
) -> Result<Recovered, RecoveryError> {
    let mut searcher = index.searcher();
    let mut counter = DistanceCounter::new();

    let mut expected: Option<u64> = None;
    let mut last_seq = contains;
    let mut replayed = 0usize;
    let mut skipped = 0usize;
    let tail;

    loop {
        // The offset only advances past a record that was fully read *and* accepted, so it is
        // always a safe place to truncate to.
        let framed = match wal::read_record(&bytes[*offset..], dim)? {
            Next::Record(framed) => framed,
            Next::Torn(torn) => {
                tail = if torn == (Torn::ShortFrame { remaining: 0 }) {
                    Tail::Clean
                } else {
                    Tail::Torn(torn)
                };
                break;
            }
        };

        if let Some(want) = expected
            && framed.seq != want
        {
            tail = Tail::SequenceGap {
                expected: want,
                found: framed.seq,
            };
            break;
        }
        // The first record read has nothing before it in this file, so the check that applies is
        // against the snapshot: it must not start past the record that follows what the snapshot
        // holds, or the records in between were lost with the log they were written to.
        if expected.is_none() && framed.seq > contains + 1 {
            return Err(RecoveryError::MissingRecords {
                contains,
                found: framed.seq,
            });
        }

        if framed.seq > contains {
            apply::<M>(
                &mut index,
                &mut searcher,
                &mut counter,
                &framed.record,
                framed.seq,
            )?;
            replayed += 1;
        } else {
            skipped += 1;
        }

        expected = Some(framed.seq + 1);
        last_seq = framed.seq.max(last_seq);
        *offset += framed.bytes;
    }

    Ok(Recovered {
        index,
        valid_bytes: *offset as u64,
        next_seq: last_seq + 1,
        tail,
        replayed,
        skipped,
    })
}

fn apply<M: Metric>(
    index: &mut HnswIndex,
    searcher: &mut anka_index::Searcher,
    counter: &mut DistanceCounter,
    record: &Record,
    seq: u64,
) -> Result<(), RecoveryError> {
    match record {
        Record::Insert { level, vector, .. } => {
            // Clamping would quietly build a different graph from the one the log describes, so
            // a level past the guard is refused instead.
            if *level as usize > MAX_LEVEL {
                return Err(WalError::LevelOutOfRange { seq, level: *level }.into());
            }
            index.insert_at_level::<M>(searcher, vector, *level as usize, counter)?;
            Ok(())
        }
        // Deletion needs the tombstone set and the id map, both of which arrive in phase 4.
        // Nothing in this build writes such a record, so one can only have come from a build that
        // does — and dropping it would lose an acknowledged deletion.
        Record::Delete { .. } => Err(RecoveryError::CannotReplayDelete { seq }),
        // Purely informational here: the snapshot's own header is what recovery trusts about how
        // far it goes. The record exists so an operator reading the log can see where a
        // checkpoint happened.
        Record::Checkpoint { .. } => Ok(()),
    }
}

impl From<IndexError> for RecoveryError {
    fn from(error: IndexError) -> Self {
        RecoveryError::Wal(WalError::Index(error))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use anka_core::{MetricKind, NodeId};
    use anka_index::HnswParams;
    use tempfile::TempDir;

    use super::*;
    use crate::wal::{SyncPolicy, WalWriter};

    const DIM: usize = 4;

    fn vector(id: u64) -> Vec<f32> {
        (0..DIM).map(|i| id as f32 * 3.0 + i as f32).collect()
    }

    fn insert_record(id: u64, level: u8) -> Record {
        Record::Insert {
            external_id: id,
            level,
            vector: vector(id),
            metadata: Vec::new(),
        }
    }

    fn empty_index() -> HnswIndex {
        HnswIndex::new(DIM, MetricKind::L2Squared, HnswParams::default()).unwrap()
    }

    /// Builds the index the log describes, by inserting the same vectors at the same levels
    /// directly. This is the answer recovery has to reproduce.
    fn expected_index(levels: &[(u64, u8)]) -> HnswIndex {
        let mut index = empty_index();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();
        for &(id, level) in levels {
            index
                .insert_at_level::<L2Squared>(
                    &mut searcher,
                    &vector(id),
                    level as usize,
                    &mut counter,
                )
                .unwrap();
        }
        index
    }

    fn assert_same_graph(a: &HnswIndex, b: &HnswIndex) {
        assert_eq!(a.len(), b.len(), "node count");
        assert_eq!(a.max_layer(), b.max_layer(), "max layer");
        assert_eq!(a.entry_point(), b.entry_point(), "entry point");
        for layer in 0..=a.max_layer() {
            let (left, right) = (&a.layers()[layer], &b.layers()[layer]);
            assert_eq!(
                left.nodes().collect::<Vec<_>>(),
                right.nodes().collect::<Vec<_>>(),
                "layer {layer} membership"
            );
            for node in left.nodes() {
                assert_eq!(
                    left.neighbors(node),
                    right.neighbors(node),
                    "layer {layer}, node {node}"
                );
            }
        }
        for node in 0..a.len() as NodeId {
            assert_eq!(a.level_of(node), b.level_of(node), "level of {node}");
        }
    }

    fn assert_same_answers(a: &HnswIndex, b: &HnswIndex) {
        let mut sa = a.searcher();
        let mut sb = b.searcher();
        let mut counter = DistanceCounter::new();
        for id in 0..12u64 {
            let query = vector(id * 2);
            assert_eq!(
                a.search::<L2Squared>(&mut sa, &query, 5, 32, &mut counter)
                    .unwrap(),
                b.search::<L2Squared>(&mut sb, &query, 5, 32, &mut counter)
                    .unwrap(),
                "query {id}"
            );
        }
    }

    /// Writes `records` to a fresh log starting at `first_seq`.
    fn write_log(dir: &TempDir, records: &[Record], first_seq: u64) -> std::path::PathBuf {
        let path = dir.path().join("wal.log");
        let mut writer = WalWriter::create(&path, SyncPolicy::Always, first_seq).unwrap();
        for record in records {
            writer.append(record).unwrap();
        }
        path
    }

    const LEVELS: [(u64, u8); 8] = [
        (1, 0),
        (2, 1),
        (3, 0),
        (4, 0),
        (5, 2),
        (6, 0),
        (7, 1),
        (8, 0),
    ];

    /// The phase 3 claim for the WAL path: replay reproduces the graph exactly, not approximately.
    #[test]
    fn replaying_a_log_rebuilds_the_same_index() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 1);

        let recovered = replay(empty_index(), 0, &path).unwrap();

        assert_eq!(recovered.replayed, LEVELS.len());
        assert_eq!(recovered.skipped, 0);
        assert_eq!(recovered.tail, Tail::Clean);
        assert_eq!(recovered.next_seq, LEVELS.len() as u64 + 1);
        assert!(!recovered.tail.truncated());

        let expected = expected_index(&LEVELS);
        assert_same_graph(&expected, &recovered.index);
        assert_same_answers(&expected, &recovered.index);
    }

    /// Records the snapshot already holds are skipped, not applied twice. Applying them again
    /// would duplicate every vector and quietly double the collection.
    #[test]
    fn records_already_in_the_snapshot_are_skipped() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 1);

        // A snapshot holding the first five records, and a log that still carries all eight.
        let base = expected_index(&LEVELS[..5]);
        let recovered = replay(base, 5, &path).unwrap();

        assert_eq!(recovered.skipped, 5);
        assert_eq!(recovered.replayed, 3);
        assert_eq!(recovered.index.len(), LEVELS.len());

        let expected = expected_index(&LEVELS);
        assert_same_graph(&expected, &recovered.index);
        assert_same_answers(&expected, &recovered.index);
    }

    /// Scenario 4 of four: the sequence numbers jump. Everything from the jump on is discarded,
    /// including records that are individually intact.
    #[test]
    fn a_sequence_gap_truncates_from_the_gap() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wal.log");
        let mut writer = WalWriter::create(&path, SyncPolicy::Always, 1).unwrap();
        for &(id, level) in &LEVELS[..3] {
            writer.append(&insert_record(id, level)).unwrap();
        }
        let valid = writer.len();
        // Skip a number, then keep writing perfectly good records.
        let mut writer = WalWriter::reopen(&path, SyncPolicy::Always, 5, valid).unwrap();
        for &(id, level) in &LEVELS[3..] {
            writer.append(&insert_record(id, level)).unwrap();
        }

        let recovered = replay(empty_index(), 0, &path).unwrap();

        assert_eq!(
            recovered.tail,
            Tail::SequenceGap {
                expected: 4,
                found: 5
            }
        );
        assert_eq!(recovered.replayed, 3);
        assert_eq!(recovered.valid_bytes, valid);
        assert_eq!(recovered.next_seq, 4);
        assert_same_graph(&expected_index(&LEVELS[..3]), &recovered.index);
    }

    /// Scenarios 1–3, end to end: whatever a crash left behind, recovery stops at the last intact
    /// record and reports where. Every truncation point of the last record is covered.
    #[test]
    fn a_torn_tail_stops_at_the_last_intact_record() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 1);
        let whole = std::fs::read(&path).unwrap();

        let complete = replay(empty_index(), 0, &path).unwrap();
        assert_eq!(complete.tail, Tail::Clean);
        let full_len = complete.valid_bytes as usize;
        let last_record_len = insert_record(8, 0).encoded_len();
        let boundary = full_len - last_record_len;

        // Truncated at every byte of the final record. `kept = 0` is the boundary case and is
        // deliberately included: a crash exactly between two records leaves a log that is *clean*,
        // seven records long, with nothing to discard. Anything past that byte is a torn tail.
        for kept in 0..last_record_len {
            std::fs::write(&path, &whole[..boundary + kept]).unwrap();
            let recovered = replay(empty_index(), 0, &path).unwrap();

            assert_eq!(
                recovered.replayed, 7,
                "kept {kept} bytes of the last record"
            );
            assert_eq!(recovered.valid_bytes, boundary as u64);
            assert_eq!(recovered.next_seq, 8);
            assert_eq!(
                recovered.tail.truncated(),
                kept > 0,
                "kept {kept} bytes of the last record"
            );
            if kept > 0 && kept < wal::FRAME_BYTES {
                assert_eq!(
                    recovered.tail,
                    Tail::Torn(Torn::ShortFrame { remaining: kept })
                );
            }
            assert_same_graph(&expected_index(&LEVELS[..7]), &recovered.index);
        }

        // A single flipped bit anywhere in the final record, which the checksum has to catch.
        for byte in boundary..full_len {
            let mut bytes = whole.clone();
            bytes[byte] ^= 0x01;
            std::fs::write(&path, &bytes).unwrap();
            let recovered = replay(empty_index(), 0, &path).unwrap();

            assert_eq!(recovered.replayed, 7, "flipped bit at byte {byte}");
            assert_eq!(recovered.valid_bytes, boundary as u64);
            assert!(recovered.tail.truncated());
        }
    }

    /// Recovery has to be idempotent: running it, truncating, and running it again must not
    /// change the answer. Restarting twice after one crash is an ordinary thing to happen.
    #[test]
    fn recovering_twice_gives_the_same_index() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 1);

        let whole = std::fs::read(&path).unwrap();
        std::fs::write(&path, &whole[..whole.len() - 5]).unwrap();

        let first = replay(empty_index(), 0, &path).unwrap();
        assert!(first.tail.truncated());

        // What a real restart does next: cut the tail off, then append.
        WalWriter::reopen(&path, SyncPolicy::Always, first.next_seq, first.valid_bytes).unwrap();

        let second = replay(empty_index(), 0, &path).unwrap();
        assert_eq!(second.tail, Tail::Clean);
        assert_eq!(second.replayed, first.replayed);
        assert_eq!(second.valid_bytes, first.valid_bytes);
        assert_same_graph(&first.index, &second.index);
    }

    /// Writing after a crash continues where recovery said, and the next recovery sees one log.
    #[test]
    fn writing_after_recovery_extends_the_same_log() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS[..4]
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 1);

        let recovered = replay(empty_index(), 0, &path).unwrap();
        let mut writer = WalWriter::reopen(
            &path,
            SyncPolicy::Always,
            recovered.next_seq,
            recovered.valid_bytes,
        )
        .unwrap();
        for &(id, level) in &LEVELS[4..] {
            writer.append(&insert_record(id, level)).unwrap();
        }

        let again = replay(empty_index(), 0, &path).unwrap();
        assert_eq!(again.tail, Tail::Clean);
        assert_eq!(again.replayed, LEVELS.len());
        assert_same_graph(&expected_index(&LEVELS), &again.index);
    }

    /// A missing log is a collection that was checkpointed and not written to since, not a
    /// failure. A log that exists but is not ours is a failure, and the file is left alone.
    #[test]
    fn a_missing_log_is_fine_and_a_foreign_one_is_not() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("nothing.log");
        let recovered = replay(expected_index(&LEVELS), 8, &missing).unwrap();
        assert_eq!(recovered.tail, Tail::Clean);
        assert_eq!(recovered.replayed, 0);
        assert_eq!(recovered.next_seq, 9);

        let foreign = dir.path().join("foreign.log");
        std::fs::write(&foreign, b"not a log, just some bytes here").unwrap();
        assert!(matches!(
            replay(empty_index(), 0, &foreign),
            Err(RecoveryError::Wal(WalError::BadMagic { .. }))
        ));
        // Reported, not repaired: the file is still there for an operator to look at.
        assert!(foreign.exists());
    }

    /// A log that starts past the snapshot means the records in between are gone. Proceeding
    /// would produce an index missing acknowledged data and no sign that anything was lost.
    #[test]
    fn a_log_that_skips_past_the_snapshot_is_an_error() {
        let dir = TempDir::new().unwrap();
        let records: Vec<Record> = LEVELS[..3]
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        let path = write_log(&dir, &records, 10);

        assert!(matches!(
            replay(empty_index(), 4, &path),
            Err(RecoveryError::MissingRecords {
                contains: 4,
                found: 10
            })
        ));
        // Starting exactly where the snapshot ends is the normal post-checkpoint case.
        assert!(replay(empty_index(), 9, &path).is_ok());
    }

    /// Deletion arrives in phase 4. Until then a delete record can only have come from a build
    /// that implements it, and skipping one would lose an acknowledged deletion.
    #[test]
    fn a_delete_record_is_refused_rather_than_skipped() {
        let dir = TempDir::new().unwrap();
        let path = write_log(
            &dir,
            &[insert_record(1, 0), Record::Delete { external_id: 1 }],
            1,
        );

        assert!(matches!(
            replay(empty_index(), 0, &path),
            Err(RecoveryError::CannotReplayDelete { seq: 2 })
        ));
    }

    /// Checkpoint records carry no state recovery needs — the snapshot's own header is what it
    /// trusts — so they replay as no-ops without interrupting the records around them.
    #[test]
    fn checkpoint_records_replay_as_nothing() {
        let dir = TempDir::new().unwrap();
        let mut records: Vec<Record> = LEVELS[..3]
            .iter()
            .map(|&(id, level)| insert_record(id, level))
            .collect();
        records.push(Record::Checkpoint {
            snapshot_wal_seq: 3,
        });
        records.extend(
            LEVELS[3..]
                .iter()
                .map(|&(id, level)| insert_record(id, level)),
        );
        let path = write_log(&dir, &records, 1);

        let recovered = replay(empty_index(), 0, &path).unwrap();
        assert_eq!(recovered.tail, Tail::Clean);
        assert_eq!(recovered.replayed, records.len());
        assert_eq!(recovered.index.len(), LEVELS.len());
        assert_same_graph(&expected_index(&LEVELS), &recovered.index);
    }

    /// The end-to-end shape: snapshot on disk, log on top of it, one call to bring both back.
    #[test]
    fn a_snapshot_and_its_log_recover_together() {
        let dir = TempDir::new().unwrap();
        let snapshot_path = dir.path().join("collection.anka");
        let log_path = dir.path().join("wal.log");

        // Five records committed and checkpointed, three more logged after it.
        let checkpointed = expected_index(&LEVELS[..5]);
        snapshot::write(&checkpointed, &snapshot_path, 5).unwrap();

        let mut writer = WalWriter::create(&log_path, SyncPolicy::Always, 6).unwrap();
        for &(id, level) in &LEVELS[5..] {
            writer.append(&insert_record(id, level)).unwrap();
        }

        let recovered = open(&snapshot_path, &log_path, Verify::Body).unwrap();

        assert_eq!(recovered.skipped, 0, "the log starts after the snapshot");
        assert_eq!(recovered.replayed, 3);
        assert_eq!(recovered.tail, Tail::Clean);
        assert_eq!(recovered.next_seq, 9);

        let expected = expected_index(&LEVELS);
        assert_same_graph(&expected, &recovered.index);
        assert_same_answers(&expected, &recovered.index);
        recovered.index.validate().unwrap();

        // The vectors the snapshot supplied are still mapped; the replayed ones are not.
        assert!(recovered.index.vectors().is_mapped());
        assert_eq!(recovered.index.vectors().mapped_count(), 5);
    }

    /// The sync policy changes durability, never the recovered index. This is what lets the crash
    /// tests vary the policy and still compare against one expected answer.
    #[test]
    fn the_sync_policy_does_not_change_what_recovery_produces() {
        let expected = expected_index(&LEVELS);
        for policy in [
            SyncPolicy::Always,
            SyncPolicy::EveryN(NonZeroU32::new(3).unwrap()),
            SyncPolicy::Never,
        ] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("wal.log");
            let mut writer = WalWriter::create(&path, policy, 1).unwrap();
            for &(id, level) in &LEVELS {
                writer.append(&insert_record(id, level)).unwrap();
            }
            drop(writer);

            let recovered = replay(empty_index(), 0, &path).unwrap();
            assert_eq!(recovered.tail, Tail::Clean, "{policy:?}");
            assert_same_graph(&expected, &recovered.index);
        }
    }
}
