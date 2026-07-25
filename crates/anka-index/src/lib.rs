//! Index implementations.
//!
//! - `BruteForceIndex` (phase 1) — exhaustive scan, the source of truth every later phase
//!   measures its recall against
//! - `HnswIndex` (phase 2) — the graph index
//! - quantization (phase 5)
//!
//! See `docs/DESIGN.md`, section 5, for the parts of the HNSW paper where a naive reading
//! produces a working-but-wrong index.
