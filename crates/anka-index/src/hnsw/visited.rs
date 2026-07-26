//! Per-query visited set.
//!
//! A `HashSet` is the obvious choice and the wrong one: this is the hottest structure in the
//! search, touched once per neighbour examined, and hashing plus clearing it between queries
//! dominates the profile. The spec puts the cost at 30%+ of query time.
//!
//! Instead every node has a `u16` stamp and each query gets a new epoch. Marking is a store,
//! testing is a load and a compare, and clearing is `epoch += 1` — O(1). The array is 2 bytes
//! per node, so 2 MB for a million vectors, allocated once per searching thread.

use anka_core::NodeId;

/// Epoch-stamped membership set over a fixed node range.
pub struct VisitedList {
    marks: Vec<u16>,
    epoch: u16,
}

impl VisitedList {
    /// Room for `capacity` nodes.
    ///
    /// The epoch starts at 0 and [`Self::begin`] increments *before* the first use, so the first
    /// query runs at epoch 1. That ordering is load-bearing: `marks` is zero-initialised, so a
    /// query at epoch 0 would find every node already visited and the search would terminate
    /// immediately, returning the entry point and nothing else.
    pub fn new(capacity: usize) -> Self {
        Self {
            marks: vec![0; capacity],
            epoch: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.marks.len()
    }

    /// Grows to hold at least `capacity` nodes. Existing stamps are preserved.
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.marks.len() < capacity {
            self.marks.resize(capacity, 0);
        }
    }

    /// Starts a new query, discarding everything the previous one marked.
    pub fn begin(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped. Stamps from 65 535 queries ago would now be mistaken for this query's,
            // so the array is cleared — O(n) once every 65 535 queries, amortising to nothing.
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Marks `node` visited and reports whether it was *not* already marked.
    ///
    /// One call does the test and the set, because in the search loop they always happen
    /// together.
    #[inline]
    pub fn visit(&mut self, node: NodeId) -> bool {
        let mark = &mut self.marks[node as usize];
        if *mark == self.epoch {
            false
        } else {
            *mark = self.epoch;
            true
        }
    }

    #[inline]
    pub fn is_visited(&self, node: NodeId) -> bool {
        self.marks[node as usize] == self.epoch
    }

    pub fn memory_bytes(&self) -> usize {
        self.marks.capacity() * size_of::<u16>()
    }
}

impl std::fmt::Debug for VisitedList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisitedList")
            .field("capacity", &self.marks.len())
            .field("epoch", &self.epoch)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_and_reports_first_visit() {
        let mut visited = VisitedList::new(8);
        visited.begin();

        assert!(visited.visit(3));
        assert!(!visited.visit(3));
        assert!(visited.is_visited(3));
        assert!(!visited.is_visited(4));
    }

    /// The off-by-one that makes an HNSW search return exactly one result and look like a graph
    /// connectivity bug: at epoch 0 the zero-initialised array reports everything as visited.
    #[test]
    fn the_first_query_does_not_start_at_epoch_zero() {
        let mut visited = VisitedList::new(4);
        visited.begin();
        for node in 0..4 {
            assert!(
                visited.visit(node),
                "node {node} must be unvisited on the first query"
            );
        }
    }

    #[test]
    fn a_new_query_forgets_the_previous_one() {
        let mut visited = VisitedList::new(4);
        visited.begin();
        visited.visit(1);
        assert!(visited.is_visited(1));

        visited.begin();
        assert!(!visited.is_visited(1));
        assert!(visited.visit(1));
    }

    /// After 65 535 queries the `u16` epoch wraps. Without clearing, stamps left by the query
    /// that used this epoch number last would be read as belonging to the current one — nodes
    /// would appear visited before being reached, and recall would drop for reasons no amount
    /// of staring at the algorithm would explain.
    #[test]
    fn wrapping_the_epoch_clears_stale_stamps() {
        let mut visited = VisitedList::new(4);

        visited.begin();
        visited.visit(2);

        // Cycle all the way around back to the same epoch value.
        for _ in 0..u16::MAX as u32 {
            visited.begin();
        }

        assert!(
            !visited.is_visited(2),
            "a stamp from a wrapped-around epoch must not count as visited"
        );
        assert!(visited.visit(2));
    }

    #[test]
    fn every_epoch_in_a_full_cycle_starts_clean() {
        let mut visited = VisitedList::new(2);
        for query in 0..(u16::MAX as u32 + 3) {
            visited.begin();
            assert!(
                visited.visit(0),
                "node 0 was already marked at query {query}"
            );
            assert!(!visited.visit(0));
        }
    }

    #[test]
    fn growing_preserves_the_current_query() {
        let mut visited = VisitedList::new(2);
        visited.begin();
        visited.visit(1);

        visited.ensure_capacity(10);
        assert_eq!(visited.capacity(), 10);
        assert!(visited.is_visited(1));
        assert!(!visited.is_visited(9));
        assert!(visited.visit(9));
    }

    #[test]
    fn shrinking_is_not_attempted() {
        let mut visited = VisitedList::new(10);
        visited.ensure_capacity(2);
        assert_eq!(visited.capacity(), 10);
    }

    #[test]
    fn memory_is_two_bytes_per_node() {
        let visited = VisitedList::new(1_000_000);
        assert_eq!(visited.memory_bytes(), 2_000_000);
    }
}
