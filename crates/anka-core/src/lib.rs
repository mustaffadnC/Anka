//! Core building blocks shared by the rest of Anka.
//!
//! Contents as of phase 0:
//! - `VectorStore` / `Storage` — flat, row-major fp32 storage, either owned or memory-mapped
//! - dataset readers for the `.fvecs` / `.ivecs` formats
//!
//! Phase 1 adds the `Metric` trait and the scalar and AVX2 distance kernels.
//!
//! See `docs/DESIGN.md` for the reasoning behind the layout choices.
