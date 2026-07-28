//! Hierarchical Navigable Small World index.
//!
//! Follows Malkov & Yashunin, arXiv:1603.09320, algorithms 1–5. The layout and the places where
//! a straightforward reading of the paper yields a working-but-wrong index are written up in
//! `docs/DESIGN.md`, section 6.
//!
//! Built bottom-up, one piece per commit:
//!
//! - [`layer`] — flat adjacency, sparse above layer 0
//! - [`visited`] — epoch-stamped per-query membership set
//! - [`params`] — build parameters and seeded layer assignment
//! - [`stats`] — distance-computation counter, behind the `stats` feature
//! - [`search`] — Algorithm 2, beam search within one layer
//! - [`select`] — Algorithm 4, the neighbour heuristic the index rests on
//! - insert and the top-level search follow

pub mod layer;
pub mod params;
pub mod search;
pub mod select;
pub mod stats;
pub mod visited;

pub use layer::Layer;
pub use params::{HnswParams, LevelGenerator, MAX_LEVEL};
pub use search::Searcher;
pub use select::select_neighbors;
pub use stats::DistanceCounter;
pub use visited::VisitedList;
