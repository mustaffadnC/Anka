//! Core building blocks shared by the rest of Anka.
//!
//! Contents:
//! - [`VectorStore`] / [`Storage`] — flat, row-major fp32 storage, either owned or memory-mapped
//! - [`metric`] — the distance contract, `f64` reference kernels and AVX2 fast paths
//! - [`Candidate`] — a scored neighbour with a total, deterministic ordering
//! - [`dataset`] — readers and writers for the `.fvecs` / `.ivecs` formats
//! - [`mem`] — resident set size, for the memory numbers in `docs/RESULTS.md`
//!
//! See `docs/DESIGN.md` for the reasoning behind the layout and metric choices.

// The .fvecs/.ivecs readers reinterpret a little-endian byte buffer as f32/i32 directly.
// Every target this project runs on is little-endian; fail the build rather than silently
// misparse if that ever stops being true.
#[cfg(target_endian = "big")]
compile_error!("anka-core assumes a little-endian target (see dataset.rs)");

pub mod candidate;
pub mod dataset;
pub mod error;
pub mod mem;
pub mod metric;
pub mod vector_store;

pub use candidate::Candidate;
pub use error::{DatasetError, VectorError};
pub use metric::{Cosine, DotProduct, L2Squared, Metric, MetricKind, preprocess_all};
pub use vector_store::{MAX_DIM, Storage, VectorStore, Vectors};

/// Identifier supplied by the caller. Stable for the lifetime of a vector.
///
/// Deliberately distinct from [`NodeId`]: compaction rewrites the graph's node numbering while
/// these must not move. See `docs/DESIGN.md`, section 2.
pub type ExternalId = u64;

/// Index of a node inside the graph. Reassigned by compaction.
pub type NodeId = u32;
