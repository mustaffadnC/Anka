//! Persistence errors.
//!
//! A snapshot is an untrusted file. Spec section 7 requires that a corrupt one produce an error
//! rather than a panic, so every variant here exists because some byte pattern can reach it — and
//! each carries the numbers needed to tell "this file is not ours" apart from "this file is ours
//! and damaged".

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file holds {got} bytes, fewer than the {needed}-byte header")]
    TooShort { needed: usize, got: usize },

    #[error("not an Anka snapshot: magic is {found:?}, expected \"ANKA\"")]
    BadMagic { found: [u8; 4] },

    #[error("snapshot format version {found}, this build reads version {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },

    #[error("header checksum mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    HeaderChecksumMismatch { stored: u32, computed: u32 },

    #[error("body checksum mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    BodyChecksumMismatch { stored: u32, computed: u32 },

    #[error("unknown metric tag {tag}")]
    UnknownMetric { tag: u8 },

    /// Unknown flags are rejected rather than masked off. A flag this build does not understand
    /// describes a graph whose shape it cannot reason about, and proceeding would produce
    /// plausible, wrong results.
    #[error("snapshot sets flag bits {bits:#010x} that this build does not understand")]
    UnknownFlags { bits: u32 },

    #[error("invalid dimension {dim}")]
    InvalidDim { dim: u32 },

    #[error("live_count {live_count} exceeds count {count}")]
    LiveCountAboveCount { live_count: u32, count: u32 },

    #[error("entry point {entry_point} is outside a collection of {count} vectors")]
    EntryPointOutOfRange { entry_point: u32, count: u32 },

    #[error("snapshot holds {count} vectors but records no entry point")]
    MissingEntryPoint { count: u32 },

    #[error("invalid degree parameters: M={m}, max_degree0={max_degree0}")]
    InvalidDegrees { m: u16, max_degree0: u16 },

    #[error("ef_construction must be at least 1")]
    InvalidEfConstruction,

    #[error("section {section} starts at {offset}, before the previous section's {previous}")]
    SectionsOutOfOrder {
        section: &'static str,
        offset: u64,
        previous: u64,
    },

    /// Sections are cast straight out of the mapping, and a misaligned cast panics. Rejecting the
    /// offset is what keeps a corrupt file from taking the process down.
    #[error("section {section} starts at {offset}, which is not {align}-byte aligned")]
    SectionMisaligned {
        section: &'static str,
        offset: u64,
        align: usize,
    },

    #[error("section {section} starts at {offset}, past the {body_len}-byte body")]
    SectionOutOfRange {
        section: &'static str,
        offset: u64,
        body_len: u64,
    },

    #[error(
        "file holds {file_len} bytes but the header describes a {body_len}-byte body after a \
         {header_len}-byte header"
    )]
    BodyTruncated {
        file_len: u64,
        body_len: u64,
        header_len: usize,
    },

    #[error("section {section} is {found} bytes, expected {expected}")]
    SectionSizeMismatch {
        section: &'static str,
        found: usize,
        expected: usize,
    },

    /// A section's bytes are not a whole number of the elements it holds, or sit at an address
    /// the target cannot read them from. Reachable only from a corrupt file, since the writer
    /// pads every section — but "unreachable" is not something a reader of untrusted bytes gets
    /// to assume.
    #[error("section {section} cannot be read as {element}: {reason}")]
    SectionNotCastable {
        section: &'static str,
        element: &'static str,
        reason: bytemuck::PodCastError,
    },

    #[error(transparent)]
    Vector(#[from] anka_core::VectorError),

    #[error(transparent)]
    Index(#[from] anka_index::IndexError),
}

impl SnapshotError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// A write-ahead log could not be written or read.
///
/// Separate from [`SnapshotError`] because the two failure modes barely overlap: a snapshot is
/// wrong or it is not, while a log is expected to end mid-record and the interesting question is
/// whether that ending is a torn tail — which is normal — or a log this build cannot read, which
/// is not. Only the second kind reaches this type; see [`crate::wal::Torn`] for the first.
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file holds {got} bytes, fewer than the {needed}-byte log header")]
    TooShort { needed: usize, got: usize },

    #[error("not an Anka write-ahead log: magic is {found:?}, expected \"AWAL\"")]
    BadMagic { found: [u8; 4] },

    #[error("log format version {found}, this build reads version {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },

    /// The record's checksum was valid, so its bytes are intact — they simply describe an
    /// operation this build does not implement. Skipping it would drop a committed record.
    #[error("record {seq} carries operation {op:#04x}, which this build does not understand")]
    UnknownOp { seq: u64, op: u8 },

    /// Intact bytes that do not parse. The usual cause is a log written against a different
    /// vector dimension, which is why the arithmetic is required to be exact rather than
    /// generous.
    #[error("record {seq} (op {op:#04x}) is malformed: {reason}")]
    MalformedPayload {
        seq: u64,
        op: u8,
        reason: &'static str,
    },

    #[error("record of {bytes} bytes exceeds the {limit}-byte limit")]
    RecordTooLarge { bytes: usize, limit: usize },

    /// Replay records a level rather than drawing one, so a level past the index's guard means
    /// the log and this build disagree about the graph's shape.
    #[error("record {seq} assigns level {level}, above the supported maximum")]
    LevelOutOfRange { seq: u64, level: u8 },

    #[error(transparent)]
    Vector(#[from] anka_core::VectorError),

    #[error(transparent)]
    Index(#[from] anka_index::IndexError),
}

impl WalError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// A collection could not be brought back up to date.
///
/// Distinct from a torn tail, which is not an error at all — see [`crate::recovery::Tail`]. What
/// reaches this type is a snapshot and a log that cannot be reconciled, and in every case the
/// files are reported rather than repaired: an operator can look at them, and nothing has been
/// silently discarded on their behalf.
#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),

    #[error(transparent)]
    Wal(#[from] WalError),

    /// The log begins after the point the snapshot reaches, so the records in between exist
    /// nowhere. Proceeding would produce an index missing acknowledged data and no sign of it.
    #[error(
        "the snapshot contains records up to {contains} and the log starts at {found}: \
         everything in between is missing"
    )]
    MissingRecords { contains: u64, found: u64 },

    /// Deletion arrives in phase 4. Nothing in this build writes such a record, so one can only
    /// have come from a build that implements it — and skipping it loses an acknowledged delete.
    #[error("record {seq} is a delete, which this build cannot replay (deletion is phase 4)")]
    CannotReplayDelete { seq: u64 },
}
