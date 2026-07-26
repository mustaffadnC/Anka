//! Core building blocks shared by the rest of Anka.
//!
//! Contents as of phase 0:
//! - [`VectorStore`] / [`Storage`] — flat, row-major fp32 storage, either owned or memory-mapped
//! - [`dataset`] — readers and writers for the `.fvecs` / `.ivecs` formats
//! - [`mem`] — resident set size, for the memory numbers in `docs/RESULTS.md`
//!
//! Phase 1 adds the `Metric` trait and the scalar and AVX2 distance kernels.
//!
//! See `docs/DESIGN.md` for the reasoning behind the layout choices.

// The .fvecs/.ivecs readers reinterpret a little-endian byte buffer as f32/i32 directly.
// Every target this project runs on is little-endian; fail the build rather than silently
// misparse if that ever stops being true.
#[cfg(target_endian = "big")]
compile_error!("anka-core assumes a little-endian target (see dataset.rs)");

pub mod dataset;
pub mod error;
pub mod mem;
pub mod vector_store;

pub use error::{DatasetError, VectorError};
pub use vector_store::{MAX_DIM, Storage, VectorStore};

/// Identifier supplied by the caller. Stable for the lifetime of a vector.
///
/// Deliberately distinct from [`NodeId`]: compaction rewrites the graph's node numbering
/// while these must not move. See `docs/DESIGN.md`, section 2.
pub type ExternalId = u64;

/// Index of a node inside the graph. Reassigned by compaction.
pub type NodeId = u32;

/// Which distance function a collection uses.
///
/// Runtime metric selection stops here: a snapshot header and an HTTP request both carry the
/// metric as data, while the `Metric` trait (phase 1) is static so the inner loops can
/// monomorphise. Dispatch happens once, at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Squared euclidean distance. Squared because the square root is monotonic and adds
    /// nothing but cost to a comparison.
    L2Squared,
    /// Cosine distance over vectors normalised at insert time.
    Cosine,
    /// Negated inner product, so that smaller still means closer.
    Dot,
}

impl MetricKind {
    /// Wire and on-disk encoding. Kept explicit rather than derived from declaration order,
    /// because a snapshot written by an older build must keep meaning the same thing.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::L2Squared => 0,
            Self::Cosine => 1,
            Self::Dot => 2,
        }
    }

    /// Inverse of [`Self::as_u8`]. Returns `None` for an unknown tag, so a corrupt or
    /// future-version snapshot is an error rather than a silently wrong metric.
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::L2Squared),
            1 => Some(Self::Cosine),
            2 => Some(Self::Dot),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_tags_round_trip() {
        for metric in [MetricKind::L2Squared, MetricKind::Cosine, MetricKind::Dot] {
            assert_eq!(MetricKind::from_u8(metric.as_u8()), Some(metric));
        }
    }

    #[test]
    fn unknown_metric_tag_is_rejected() {
        assert_eq!(MetricKind::from_u8(3), None);
        assert_eq!(MetricKind::from_u8(u8::MAX), None);
    }
}
