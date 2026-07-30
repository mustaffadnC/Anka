//! Persistence: snapshots, the write-ahead log, and recovery (phase 3).
//!
//! The on-disk formats are specified in `docs/DESIGN.md`, section 3. Two invariants there are
//! easy to get wrong and expensive to fix afterwards:
//!
//! - snapshot writes go `tmp → fsync(file) → rename → fsync(directory)`; skipping the
//!   directory `fsync` can lose the rename across a crash
//! - every WAL record carries a `seq`, and every INSERT carries the assigned HNSW `level`,
//!   so replay is both well-defined and deterministic
//!
//! Built in order: [`header`] first, because the format has to be pinned down before anything can
//! be written into it.

pub mod error;
pub mod header;

pub use error::SnapshotError;
pub use header::{FORMAT_VERSION, HEADER_BYTES, HeaderFlags, Section, SnapshotHeader};
