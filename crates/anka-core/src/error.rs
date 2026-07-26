//! Error types.
//!
//! Two rules apply throughout: malformed input is an error, never a panic, and every error
//! carries enough context to locate the problem without re-running under a debugger. A
//! truncated dataset should say *which record* was short.

use std::path::PathBuf;

/// Rejected vector data.
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("dimension must be non-zero")]
    ZeroDim,

    #[error("dimension {dim} exceeds the supported maximum of {max}")]
    DimTooLarge { dim: usize, max: usize },

    #[error("expected dimension {expected}, got {found}")]
    DimMismatch { expected: usize, found: usize },

    #[error("a flat buffer of {len} values is not a whole number of {dim}-dimensional vectors")]
    RaggedBuffer { len: usize, dim: usize },

    /// A single NaN or infinity poisons everything downstream: heap ordering quietly stops
    /// being a total order, and the search results become unexplainable. Rejecting at the
    /// boundary is what keeps the hot path free of defensive checks.
    #[error("vector {vector}, component {component} is not finite ({value})")]
    NonFinite {
        vector: usize,
        component: usize,
        value: f32,
    },

    #[error("{count} vectors exceeds the NodeId space (u32::MAX)")]
    TooManyVectors { count: usize },

    #[error("memory-mapped storage is read-only")]
    ReadOnlyStorage,

    #[error("memory-mapped storage needs a 4-byte-aligned offset, got {offset}")]
    MisalignedOffset { offset: usize },

    #[error(
        "mapping is too small: {dim} x {count} vectors need {needed} bytes from offset \
         {offset}, but the mapping is {available} bytes"
    )]
    MappingTooSmall {
        dim: usize,
        count: usize,
        offset: usize,
        needed: usize,
        available: usize,
    },
}

/// Failures while reading or writing a dataset file.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path}: file is empty")]
    Empty { path: PathBuf },

    #[error(
        "{path}: {len} bytes is not a whole number of {record_bytes}-byte records \
         (dimension {dim}); the file is truncated or is not a {format} file"
    )]
    Ragged {
        path: PathBuf,
        len: u64,
        record_bytes: usize,
        dim: usize,
        format: &'static str,
    },

    /// Every record in a `.fvecs`/`.ivecs` file repeats its dimension. A mismatch means the
    /// file is corrupt or the stride was misread — either way, continuing would produce a
    /// plausible-looking but wrong dataset.
    #[error(
        "{path}: record {record} declares dimension {found}, but the file opened with {expected}"
    )]
    InconsistentDim {
        path: PathBuf,
        record: usize,
        expected: usize,
        found: usize,
    },

    #[error("{path}: unexpected end of file inside record {record}")]
    Truncated { path: PathBuf, record: usize },

    #[error("{path}: {source}")]
    Vector {
        path: PathBuf,
        #[source]
        source: VectorError,
    },
}

impl DatasetError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
