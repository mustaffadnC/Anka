//! A scored neighbour.

use std::cmp::Ordering;

use crate::NodeId;

/// A neighbour and its distance from the query.
///
/// The ordering is `(dist, id)` lexicographic over `f32::total_cmp`, and both halves of that
/// matter:
///
/// - `f32` has no `Ord`, and `partial_cmp().unwrap()` is a panic waiting for the first NaN to
///   reach a heap. `total_cmp` is a genuine total order over every bit pattern.
/// - Ties break towards the smaller id. Without a deterministic rule, two runs over the same
///   data can return different equidistant neighbours, which turns a 100% match against a
///   published ground truth into a 99.9% one and sends you hunting for a bug that is not
///   there.
///
/// `PartialEq` is written by hand rather than derived so that it agrees with `Ord`: a derived
/// `PartialEq` would say `NaN != NaN` while `total_cmp` says they are equal, and `Eq` would
/// then be a lie.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub dist: f32,
    pub id: NodeId,
}

impl Candidate {
    pub fn new(dist: f32, id: NodeId) -> Self {
        Self { dist, id }
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use super::*;

    #[test]
    fn orders_by_distance_then_id() {
        let mut v = vec![
            Candidate::new(2.0, 1),
            Candidate::new(1.0, 9),
            Candidate::new(2.0, 0),
            Candidate::new(1.0, 3),
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                Candidate::new(1.0, 3),
                Candidate::new(1.0, 9),
                Candidate::new(2.0, 0),
                Candidate::new(2.0, 1),
            ]
        );
    }

    /// The tie-break is the whole reason this type exists rather than a bare tuple.
    #[test]
    fn equal_distances_break_towards_the_smaller_id() {
        assert!(Candidate::new(1.0, 4) < Candidate::new(1.0, 5));
        assert_eq!(Candidate::new(1.0, 4), Candidate::new(1.0, 4));
    }

    /// `Eq` demands reflexivity. A derived `PartialEq` over `f32` would break it here, and
    /// anything relying on `Eq` (`BTreeMap`, `dedup`, `assert_eq!`) would misbehave for NaN.
    #[test]
    fn nan_compares_equal_to_itself() {
        let nan = Candidate::new(f32::NAN, 7);
        assert_eq!(nan, nan);
        assert_eq!(nan.cmp(&nan), Ordering::Equal);
    }

    /// NaN sorts above every real distance under `total_cmp`, so a poisoned entry loses the
    /// top-k race instead of silently winning it. Vectors are rejected at insert, but the
    /// ordering should still fail safe.
    #[test]
    fn nan_sorts_after_finite_distances() {
        assert!(Candidate::new(f32::INFINITY, 0) < Candidate::new(f32::NAN, 0));
        assert!(Candidate::new(1e30, 0) < Candidate::new(f32::NAN, 0));
    }

    /// A max-heap of `Candidate` keeps the furthest neighbour at the top, which is what makes
    /// bounded top-k selection a constant-space operation.
    #[test]
    fn binary_heap_exposes_the_furthest_candidate() {
        let mut heap = BinaryHeap::new();
        for (dist, id) in [(3.0, 0), (1.0, 1), (2.0, 2)] {
            heap.push(Candidate::new(dist, id));
        }
        assert_eq!(heap.peek(), Some(&Candidate::new(3.0, 0)));
        assert_eq!(
            heap.into_sorted_vec(),
            vec![
                Candidate::new(1.0, 1),
                Candidate::new(2.0, 2),
                Candidate::new(3.0, 0),
            ]
        );
    }
}
