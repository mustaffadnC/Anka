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
//! be written into it, then [`snapshot`] on top of it, then [`wal`].

// Sections are cast straight out of the mapping, which reads them in the host's byte order. The
// format is little-endian by specification, so a big-endian host would read every array wrong and
// checksum it right. Refusing to build is honest; a byte-swapping path nobody can test is not.
#[cfg(target_endian = "big")]
compile_error!("the snapshot format is little-endian; this target is big-endian");

pub mod error;
mod fsync;
pub mod header;
pub mod snapshot;
pub mod wal;

pub use error::{SnapshotError, WalError};
pub use header::{
    FORMAT_VERSION, HEADER_BYTES, HeaderFlags, SECTION_ALIGN, Section, SnapshotHeader,
};
pub use snapshot::{Verify, load, read, write};
pub use wal::{Framed, Next, Record, SyncPolicy, Torn, WalWriter};
