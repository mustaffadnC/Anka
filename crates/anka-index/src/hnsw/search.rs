//! Algorithm 2: beam search within one layer.
//!
//! The scratch state — visited stamps and the two heaps — lives in [`Searcher`] and is reused
//! across calls. A query allocating three containers before it can start would spend a
//! meaningful fraction of its time in the allocator, and construction calls this once per layer
//! per insert.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anka_core::{Candidate, Metric, NodeId, VectorStore};

use crate::hnsw::layer::Layer;
use crate::hnsw::stats::DistanceCounter;
use crate::hnsw::visited::VisitedList;

/// Reusable scratch for one searching thread.
///
/// Not `Sync`, deliberately: a shared visited list between threads would corrupt both searches.
/// Parallel search gives each thread its own (spec section 12, stretch goals).
pub struct Searcher {
    visited: VisitedList,
    /// Candidates left to expand, nearest first.
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Best results so far, furthest first, capped at `ef`.
    results: BinaryHeap<Candidate>,
}

impl Searcher {
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: VisitedList::new(capacity),
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    pub fn ensure_capacity(&mut self, capacity: usize) {
        self.visited.ensure_capacity(capacity);
    }

    pub fn capacity(&self) -> usize {
        self.visited.capacity()
    }

    /// Bytes held by the reusable scratch.
    pub fn memory_bytes(&self) -> usize {
        self.visited.memory_bytes()
            + self.candidates.capacity() * size_of::<Candidate>()
            + self.results.capacity() * size_of::<Candidate>()
    }

    /// Algorithm 2: the `ef` nearest nodes to `query` reachable from `entry_points` within
    /// `layer`, ascending by `(distance, id)`.
    ///
    /// `entry_points` is a slice rather than a single node because Algorithm 1 hands the whole
    /// result set of one layer to the next as its entry points. Duplicates in it are harmless;
    /// they are filtered by the visited stamps.
    pub fn search_layer<M: Metric>(
        &mut self,
        vectors: &VectorStore,
        layer: &Layer,
        query: &[f32],
        entry_points: &[NodeId],
        ef: usize,
        counter: &mut DistanceCounter,
    ) -> Vec<Candidate> {
        self.visited.begin();
        self.candidates.clear();
        self.results.clear();

        if ef == 0 || entry_points.is_empty() {
            return Vec::new();
        }

        // Resolved once. Every access below indexes this rather than matching on the storage
        // variant, which matters because a hybrid store — a snapshot with replayed vectors after
        // it — is not one contiguous buffer.
        let view = vectors.view();

        for &entry in entry_points {
            if !self.visited.visit(entry) {
                continue;
            }
            let candidate = Candidate::new(M::distance(query, view.get(entry as usize)), entry);
            counter.record(1);
            self.candidates.push(Reverse(candidate));
            push_bounded(&mut self.results, candidate, ef);
        }

        while let Some(Reverse(current)) = self.candidates.pop() {
            // Stop once the nearest unexplored candidate is further than the worst result held,
            // *and* the result set is full. Dropping the fullness check would cut the search off
            // early while there is still room to improve, which looks like poor recall rather
            // than like a bug.
            if self.results.len() >= ef
                && let Some(furthest) = self.results.peek()
                && current.dist > furthest.dist
            {
                break;
            }

            for &neighbor in layer.neighbors(current.id) {
                if !self.visited.visit(neighbor) {
                    continue;
                }

                let candidate =
                    Candidate::new(M::distance(query, view.get(neighbor as usize)), neighbor);
                counter.record(1);

                // Order matters: the length test comes first so an empty result set is never
                // asked for its worst element. Comparison is by `(dist, id)`, which is stricter
                // than the paper's distance-only test — an equidistant candidate with a smaller
                // id displaces the incumbent, and that is what makes the output deterministic.
                let accept = self.results.len() < ef
                    || self.results.peek().is_some_and(|worst| candidate < *worst);

                if accept {
                    self.candidates.push(Reverse(candidate));
                    push_bounded(&mut self.results, candidate, ef);
                }
            }
        }

        let mut found: Vec<Candidate> = self.results.drain().collect();
        found.sort_unstable();
        found
    }
}

impl std::fmt::Debug for Searcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Searcher")
            .field("capacity", &self.visited.capacity())
            .finish_non_exhaustive()
    }
}

/// Keeps `results` at no more than `ef` entries, dropping the furthest.
///
/// `peek_mut` replaces in one sift; pop-then-push sifts twice.
fn push_bounded(results: &mut BinaryHeap<Candidate>, candidate: Candidate, ef: usize) {
    if results.len() < ef {
        results.push(candidate);
    } else if let Some(mut worst) = results.peek_mut() {
        if candidate < *worst {
            *worst = candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    use anka_core::L2Squared;

    use super::*;

    /// `n` points on a line at 0, 1, 2, ... connected to their immediate neighbours.
    ///
    /// A path graph is the clearest thing to reason about: greedy search has exactly one way to
    /// travel, so the expected answer can be derived by hand rather than asserted from whatever
    /// the code happened to produce.
    fn path_graph(n: usize) -> (VectorStore, Layer) {
        let store =
            VectorStore::from_flat(1, (0..n).map(|i| i as f32).collect()).expect("valid store");
        let mut layer = Layer::dense(2, n);
        for node in 0..n as NodeId {
            layer.add(node);
        }
        for node in 0..n as NodeId {
            let mut neighbors = Vec::new();
            if node > 0 {
                neighbors.push(node - 1);
            }
            if (node as usize) + 1 < n {
                neighbors.push(node + 1);
            }
            layer.set_neighbors(node, &neighbors);
        }
        (store, layer)
    }

    fn ids(found: &[Candidate]) -> Vec<NodeId> {
        found.iter().map(|c| c.id).collect()
    }

    /// Walks the path from node 0 to the query at position 5. Hand-derived: the beam ends holding
    /// 5 at distance 0 and 4 and 6 both at distance 1, ordered by the id tie-break.
    #[test]
    fn beam_search_walks_to_the_query() {
        let (store, layer) = path_graph(12);
        let mut searcher = Searcher::new(12);
        let mut counter = DistanceCounter::new();

        let found =
            searcher.search_layer::<L2Squared>(&store, &layer, &[5.0], &[0], 3, &mut counter);

        assert_eq!(ids(&found), vec![5, 4, 6]);
        assert_eq!(found[0].dist, 0.0);
        assert_eq!(found[1].dist, 1.0);
        assert_eq!(found[2].dist, 1.0);
    }

    /// `ef = 1` is the pure-greedy descent Algorithm 1 uses on the upper layers.
    #[test]
    fn ef_of_one_is_greedy_and_lands_on_the_nearest() {
        let (store, layer) = path_graph(20);
        let mut searcher = Searcher::new(20);
        let mut counter = DistanceCounter::new();

        for target in [0.0f32, 7.0, 19.0] {
            let found = searcher.search_layer::<L2Squared>(
                &store,
                &layer,
                &[target],
                &[0],
                1,
                &mut counter,
            );
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].id, target as NodeId);
        }
    }

    #[test]
    fn a_larger_beam_returns_more_neighbours() {
        let (store, layer) = path_graph(30);
        let mut searcher = Searcher::new(30);
        let mut counter = DistanceCounter::new();

        for ef in [1usize, 2, 5, 9] {
            let found =
                searcher.search_layer::<L2Squared>(&store, &layer, &[15.0], &[0], ef, &mut counter);
            assert_eq!(found.len(), ef, "ef = {ef}");
            // Ascending by distance, so the first is the closest.
            assert_eq!(found[0].id, 15);
            assert!(found.windows(2).all(|w| w[0].dist <= w[1].dist));
        }
    }

    #[test]
    fn no_entry_points_or_zero_ef_returns_nothing() {
        let (store, layer) = path_graph(5);
        let mut searcher = Searcher::new(5);
        let mut counter = DistanceCounter::new();

        assert!(
            searcher
                .search_layer::<L2Squared>(&store, &layer, &[1.0], &[], 3, &mut counter)
                .is_empty()
        );
        assert!(
            searcher
                .search_layer::<L2Squared>(&store, &layer, &[1.0], &[0], 0, &mut counter)
                .is_empty()
        );
    }

    #[test]
    fn repeated_entry_points_are_deduplicated() {
        let (store, layer) = path_graph(8);
        let mut searcher = Searcher::new(8);
        let mut counter = DistanceCounter::new();

        let found = searcher.search_layer::<L2Squared>(
            &store,
            &layer,
            &[3.0],
            &[0, 0, 0, 7],
            3,
            &mut counter,
        );
        let mut unique = ids(&found);
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), found.len());
    }

    /// Two components with no edge between them. A search started in one can never see the other,
    /// which is exactly why filtered search must not remove non-matching nodes from *traversal* —
    /// doing so fragments the graph and recall collapses.
    #[test]
    fn search_cannot_leave_its_connected_component() {
        let store = VectorStore::from_flat(1, vec![0.0, 1.0, 100.0, 101.0]).unwrap();
        let mut layer = Layer::dense(2, 4);
        for node in 0..4 {
            layer.add(node);
        }
        layer.set_neighbors(0, &[1]);
        layer.set_neighbors(1, &[0]);
        layer.set_neighbors(2, &[3]);
        layer.set_neighbors(3, &[2]);

        let mut searcher = Searcher::new(4);
        let mut counter = DistanceCounter::new();

        // The query sits on top of node 2, but the search starts in the other component.
        let found =
            searcher.search_layer::<L2Squared>(&store, &layer, &[100.0], &[0], 4, &mut counter);
        assert_eq!(ids(&found), vec![1, 0]);

        // Entering the right component finds it immediately.
        let found =
            searcher.search_layer::<L2Squared>(&store, &layer, &[100.0], &[2], 4, &mut counter);
        assert_eq!(found[0].id, 2);
    }

    /// Successive searches must not inherit the previous one's visited stamps.
    #[test]
    fn consecutive_searches_are_independent() {
        let (store, layer) = path_graph(16);
        let mut searcher = Searcher::new(16);
        let mut counter = DistanceCounter::new();

        let first =
            searcher.search_layer::<L2Squared>(&store, &layer, &[8.0], &[0], 3, &mut counter);
        let second =
            searcher.search_layer::<L2Squared>(&store, &layer, &[8.0], &[0], 3, &mut counter);
        assert_eq!(first, second);
    }

    /// A node is never scored twice within one search: the visited set is what bounds the work.
    #[test]
    fn each_node_is_scored_at_most_once_per_search() {
        let (store, layer) = path_graph(10);
        let mut searcher = Searcher::new(10);
        let mut counter = DistanceCounter::new();

        searcher.search_layer::<L2Squared>(&store, &layer, &[9.0], &[0], 10, &mut counter);

        if let Some(count) = counter.count() {
            assert!(
                count <= 10,
                "scored {count} times over a 10-node graph — a node was revisited"
            );
        }
    }
}
