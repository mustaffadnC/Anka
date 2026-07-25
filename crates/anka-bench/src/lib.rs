//! Benchmark harness crate.
//!
//! This exists as a workspace member rather than a top-level `benches/` directory because
//! Cargo only discovers `benches/` inside a package. Criterion benchmarks land here from
//! phase 1 onwards (distance kernels first, then HNSW search).
//!
//! Benchmarks are measured with `RUSTFLAGS="-C target-cpu=native"`; see
//! `docs/DESIGN.md`, section 9.
