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
