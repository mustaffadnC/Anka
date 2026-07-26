//! Exhaustive search.
//!
//! This is the slowest thing in the repository and the most important: every recall figure the
//! project publishes is measured against what this returns. If it is wrong, nothing downstream
//! means anything, so it is written to be obviously correct rather than fast.

use std::collections::BinaryHeap;

use anka_core::{Candidate, Metric, NodeId, VectorError, VectorStore};

/// Which distance kernel a scan uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `f64` accumulator, no SIMD. Slow, and the definition of correct — ground truth is
    /// generated with this one.
    Reference,
    /// AVX2 where the CPU has it. What every measured query path uses.
    Fast,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Fast => "fast",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "reference" | "scalar" => Some(Self::Reference),
            "fast" | "simd" => Some(Self::Fast),
            _ => None,
        }
    }
}

/// Exhaustive nearest-neighbour search over a [`VectorStore`].
pub struct BruteForceIndex<'a> {
    vectors: &'a VectorStore,
}

impl<'a> BruteForceIndex<'a> {
    pub fn new(vectors: &'a VectorStore) -> Self {
        Self { vectors }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    /// The `k` nearest neighbours of `query`, ascending by `(distance, id)`.
    ///
    /// Fewer than `k` results when the index holds fewer vectors; that is not an error. An
    /// empty index and `k == 0` both yield an empty list.
    pub fn search<M: Metric>(
        &self,
        query: &[f32],
        k: usize,
        kernel: Kernel,
    ) -> Result<Vec<Candidate>, VectorError> {
        let dim = self.vectors.dim();
        if query.len() != dim {
            return Err(VectorError::DimMismatch {
                expected: dim,
                found: query.len(),
            });
        }
        if k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Hoisted: for a mapped store this is where the alignment check lives, and it has no
        // business running once per candidate.
        let data = self.vectors.as_slice();

        // Two call sites rather than a branch inside the loop, so each one monomorphises with
        // its kernel inlined.
        Ok(match kernel {
            Kernel::Reference => scan(data, dim, query, k, M::distance_scalar),
            Kernel::Fast => scan(data, dim, query, k, M::distance),
        })
    }

    /// Distances from `query` to specific ids, in the order given.
    ///
    /// Used to ask whether two differing neighbour lists are nevertheless equidistant.
    pub fn distances_to<M: Metric>(
        &self,
        query: &[f32],
        ids: &[NodeId],
        kernel: Kernel,
    ) -> Result<Vec<f32>, VectorError> {
        let dim = self.vectors.dim();
        if query.len() != dim {
            return Err(VectorError::DimMismatch {
                expected: dim,
                found: query.len(),
            });
        }
        ids.iter()
            .map(|id| {
                let vector = self.vectors.try_get(*id as usize).ok_or({
                    VectorError::IdOutOfRange {
                        id: *id,
                        count: self.vectors.len(),
                    }
                })?;
                Ok(match kernel {
                    Kernel::Reference => M::distance_scalar(query, vector),
                    Kernel::Fast => M::distance(query, vector),
                })
            })
            .collect()
    }
}

/// Bounded top-`k` selection over a flat buffer.
///
/// The heap holds at most `k` entries with the *furthest* at the top, so the scan is
/// `O(n log k)` in `O(k)` space. Collecting all distances and sorting would be `O(n log n)` and
/// would need 4 MB of scratch per query on SIFT1M.
fn scan(
    data: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
    distance: impl Fn(&[f32], &[f32]) -> f32,
) -> Vec<Candidate> {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k);

    for (index, vector) in data.chunks_exact(dim).enumerate() {
        let candidate = Candidate::new(distance(query, vector), index as NodeId);

        if heap.len() < k {
            heap.push(candidate);
            continue;
        }
        // `peek_mut` sifts once when the value is replaced; pop-then-push sifts twice.
        if let Some(mut furthest) = heap.peek_mut() {
            if candidate < *furthest {
                *furthest = candidate;
            }
        }
    }

    heap.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    use anka_core::{Cosine, DotProduct, L2Squared};

    use super::*;

    /// Four points on a line: distances from `[0.0]` are 1, 4, 9, 16.
    fn line() -> VectorStore {
        VectorStore::from_flat(1, vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    }

    fn ids(candidates: &[Candidate]) -> Vec<NodeId> {
        candidates.iter().map(|c| c.id).collect()
    }

    #[test]
    fn returns_the_nearest_k_in_order() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        let found = index
            .search::<L2Squared>(&[0.0], 3, Kernel::Reference)
            .unwrap();

        assert_eq!(ids(&found), vec![0, 1, 2]);
        assert_eq!(found[0].dist, 1.0);
        assert_eq!(found[1].dist, 4.0);
        assert_eq!(found[2].dist, 9.0);
    }

    #[test]
    fn both_kernels_agree() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        let reference = index
            .search::<L2Squared>(&[0.0], 4, Kernel::Reference)
            .unwrap();
        let fast = index.search::<L2Squared>(&[0.0], 4, Kernel::Fast).unwrap();
        assert_eq!(reference, fast);
    }

    /// Spec section 14: asking for more than exists returns what exists.
    #[test]
    fn k_larger_than_the_index_returns_everything() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        let found = index.search::<L2Squared>(&[0.0], 99, Kernel::Fast).unwrap();
        assert_eq!(found.len(), 4);
        assert_eq!(ids(&found), vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_k_and_empty_index_return_nothing() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        assert!(
            index
                .search::<L2Squared>(&[0.0], 0, Kernel::Fast)
                .unwrap()
                .is_empty()
        );

        let empty = VectorStore::empty(1).unwrap();
        let index = BruteForceIndex::new(&empty);
        assert!(index.is_empty());
        assert!(
            index
                .search::<L2Squared>(&[0.0], 5, Kernel::Fast)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn wrong_query_dimension_is_an_error() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        assert!(matches!(
            index.search::<L2Squared>(&[0.0, 1.0], 1, Kernel::Fast),
            Err(VectorError::DimMismatch {
                expected: 1,
                found: 2
            })
        ));
    }

    /// Deterministic tie-breaking is what makes an exact ground-truth match achievable. With
    /// four vectors all at distance 1, the answer must be the four smallest ids.
    #[test]
    fn equidistant_neighbours_are_broken_by_id() {
        let store = VectorStore::from_flat(1, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
        let index = BruteForceIndex::new(&store);
        let found = index
            .search::<L2Squared>(&[0.0], 4, Kernel::Reference)
            .unwrap();
        assert!(found.iter().all(|c| c.dist == 1.0));
        assert_eq!(ids(&found), vec![0, 1, 2, 3]);
    }

    /// A brute-force scan is only meaningful if the metric's direction is respected. For dot
    /// product the "nearest" vector is the one with the *largest* inner product.
    #[test]
    fn dot_product_ranks_the_largest_inner_product_first() {
        let store = VectorStore::from_flat(2, vec![1.0, 0.0, 5.0, 0.0, 2.0, 0.0]).unwrap();
        let index = BruteForceIndex::new(&store);
        let found = index
            .search::<DotProduct>(&[1.0, 0.0], 3, Kernel::Fast)
            .unwrap();
        assert_eq!(ids(&found), vec![1, 2, 0]);
        assert_eq!(found[0].dist, -5.0);
    }

    #[test]
    fn cosine_ignores_magnitude_after_normalisation() {
        let mut store = VectorStore::from_flat(2, vec![1.0, 0.0, 100.0, 0.0, 0.0, 3.0]).unwrap();
        anka_core::preprocess_all::<Cosine>(&mut store).unwrap();
        let index = BruteForceIndex::new(&store);

        let mut query = vec![7.0, 0.0];
        Cosine::preprocess(&mut query).unwrap();
        let found = index.search::<Cosine>(&query, 3, Kernel::Fast).unwrap();

        // Vectors 0 and 1 point the same way and are equidistant; the id tie-break orders them.
        assert_eq!(ids(&found), vec![0, 1, 2]);
        assert!(found[0].dist.abs() < 1e-6);
        assert!(found[1].dist.abs() < 1e-6);
        assert!((found[2].dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distances_to_specific_ids() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        let d = index
            .distances_to::<L2Squared>(&[0.0], &[3, 0], Kernel::Reference)
            .unwrap();
        assert_eq!(d, vec![16.0, 1.0]);
    }

    #[test]
    fn distances_to_an_unknown_id_is_an_error() {
        let store = line();
        let index = BruteForceIndex::new(&store);
        assert!(matches!(
            index.distances_to::<L2Squared>(&[0.0], &[9], Kernel::Fast),
            Err(VectorError::IdOutOfRange { id: 9, count: 4 })
        ));
    }

    /// The heap keeps exactly k entries, so a large scan must not depend on insertion order.
    /// Feeding the same data in reverse must produce the same answer.
    #[test]
    fn result_is_independent_of_storage_order() {
        let forward: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let backward: Vec<f32> = forward.iter().rev().copied().collect();

        let a = VectorStore::from_flat(1, forward).unwrap();
        let b = VectorStore::from_flat(1, backward).unwrap();

        let da = BruteForceIndex::new(&a)
            .search::<L2Squared>(&[250.0], 5, Kernel::Fast)
            .unwrap();
        let db = BruteForceIndex::new(&b)
            .search::<L2Squared>(&[250.0], 5, Kernel::Fast)
            .unwrap();

        let dists_a: Vec<f32> = da.iter().map(|c| c.dist).collect();
        let dists_b: Vec<f32> = db.iter().map(|c| c.dist).collect();
        assert_eq!(dists_a, dists_b);
    }
}
