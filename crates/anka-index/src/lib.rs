//! Index implementations.
//!
//! - [`brute_force`] — exhaustive scan, the source of truth every later phase measures its
//!   recall against
//! - [`ground_truth`] — exact neighbour lists, plus the checks that hold ours to a published one
//! - HNSW (phase 2), quantization (phase 5)
//!
//! See `docs/DESIGN.md`, section 6, for the parts of the HNSW paper where a naive reading
//! produces a working-but-wrong index.

pub mod brute_force;
pub mod ground_truth;

pub use brute_force::{BruteForceIndex, Kernel};
pub use ground_truth::{Agreement, DistanceAgreement};
