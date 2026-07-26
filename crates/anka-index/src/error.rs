//! Index-level errors.
//!
//! [`crate::brute_force`] and [`crate::ground_truth`] return `anka_core::VectorError` directly,
//! because the only way they can fail is on vector-shaped input: a wrong dimension, an id out of
//! range, a `k` that makes no sense. The HNSW index adds failure modes of its own — parameters
//! that cannot build a usable graph — so it needs a type that covers both, and `IndexError`
//! converts from `VectorError` so the two compose with `?`.

use anka_core::VectorError;

/// Largest supported `M`.
///
/// A sanity bound rather than a real limit. `M` above ~100 makes no sense for any published
/// dataset, and a bound turns a nonsense parameter into an error instead of an allocation the
/// size of the machine.
pub const MAX_M: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error(transparent)]
    Vector(#[from] VectorError),

    #[error("M must be at least 1")]
    ZeroM,

    #[error("M is {m}, which exceeds the supported maximum of {MAX_M}")]
    MTooLarge { m: usize },

    /// `Mmax0` below `M` would force layer 0 — the layer that has to stay connected — to hold
    /// fewer edges than the upper layers.
    #[error("max_degree0 ({max_degree0}) must be at least M ({m})")]
    MaxDegreeTooSmall { m: usize, max_degree0: usize },

    #[error("ef_construction must be at least 1")]
    ZeroEfConstruction,

    #[error("the index was built with dimension {expected}, got {found}")]
    DimMismatch { expected: usize, found: usize },
}
