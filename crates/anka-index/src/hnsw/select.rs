//! Algorithm 4: neighbour selection.
//!
//! **This is the part that decides whether the index works at all.** Taking the nearest `M`
//! candidates builds a graph that passes every structural check and searches badly: every edge is
//! short-range, no bridge survives between distant clusters, and greedy descent settles into
//! local minima it cannot climb out of. Recall stalls in the seventies and raising `ef` does not
//! help, because the beam is exploring a neighbourhood that has no route to the answer.
//!
//! The heuristic keeps a candidate only when it is closer to the query than to anything already
//! selected. That one condition spreads the chosen neighbours across directions and preserves the
//! long edges, which is what produces the "express lane" effect the paper borrows from skip lists.
//!
//! Two flags from the paper, both settled here rather than left open:
//!
//! - `extendCandidates` — **off.** Growing the candidate pool with the candidates' own neighbours
//!   slows construction considerably and the paper reports it helping mainly on clustered data.
//!   It stays a stretch goal to measure, not a default to assume.
//! - `keepPrunedConnections` — **on.** Without it a node whose candidates were mostly pruned ends
//!   up below `M` edges, the graph thins out, and recall drops for a reason that looks like a
//!   parameter problem rather than a code one.

use anka_core::{Candidate, Metric, NodeId, VectorStore};

use crate::hnsw::search::vector_at;
use crate::hnsw::stats::DistanceCounter;

/// Picks up to `m` neighbours out of `candidates`.
///
/// `candidates` is sorted in place, nearest first — a slice rather than a `Vec` because nothing
/// here grows it. Each entry's `dist` is its distance to the query, already computed by the search
/// that produced it, so no query distance is recomputed. What this does compute is
/// candidate-to-candidate distances, which is where its cost sits.
///
/// Returns ids in selection order: the diverse picks first, then the pruned ones added back if
/// `keep_pruned` is set.
pub fn select_neighbors<M: Metric>(
    vectors: &VectorStore,
    candidates: &mut [Candidate],
    m: usize,
    keep_pruned: bool,
    counter: &mut DistanceCounter,
) -> Vec<NodeId> {
    if m == 0 {
        return Vec::new();
    }

    // Nearest first, ties by id, so the whole procedure is deterministic.
    candidates.sort_unstable();

    let data = vectors.as_slice();
    let dim = vectors.dim();

    let mut selected: Vec<NodeId> = Vec::with_capacity(m);
    let mut pruned: Vec<NodeId> = Vec::new();

    for candidate in candidates.iter() {
        if selected.len() >= m {
            break;
        }

        let candidate_vector = vector_at(data, dim, candidate.id);

        // Discard when some already-selected neighbour `r` is *closer to this candidate* than the
        // query is. Note what is being compared: dist(e, r) against dist(e, q) — not against
        // dist(r, q). Comparing distances to the query would be meaningless here, since the list
        // is already sorted by exactly that.
        let dominated = selected.iter().any(|&r| {
            counter.record(1);
            M::distance(candidate_vector, vector_at(data, dim, r)) < candidate.dist
        });

        if dominated {
            pruned.push(candidate.id);
        } else {
            selected.push(candidate.id);
        }
    }

    if keep_pruned {
        // `pruned` is already nearest-first, so the refill takes the best of what was dropped.
        for id in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(id);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use anka_core::L2Squared;

    use super::*;

    /// Four points around the origin: three strung out along +x at 1.0, 1.05 and 1.1, and one off
    /// at (0, 1.2) in a different direction and slightly further away.
    ///
    /// Naive top-2 takes the two nearest, which are 5 hundredths apart and point the same way.
    fn clustered() -> (VectorStore, Vec<Candidate>) {
        let store = VectorStore::from_flat(
            2,
            vec![
                1.0, 0.0, // 0
                1.05, 0.0, // 1
                1.1, 0.0, // 2
                0.0, 1.2, // 3
            ],
        )
        .unwrap();

        // Squared L2 from the origin.
        let candidates = vec![
            Candidate::new(1.0, 0),
            Candidate::new(1.05 * 1.05, 1),
            Candidate::new(1.1 * 1.1, 2),
            Candidate::new(1.2 * 1.2, 3),
        ];
        (store, candidates)
    }

    /// The property the whole index rests on: given a cluster and one distant direction, the
    /// heuristic takes one from the cluster and the distant one — not the two nearest.
    #[test]
    fn selection_spreads_across_directions() {
        let (store, mut candidates) = clustered();
        let mut counter = DistanceCounter::new();

        let selected =
            select_neighbors::<L2Squared>(&store, &mut candidates, 2, false, &mut counter);

        assert_eq!(
            selected,
            vec![0, 3],
            "expected the nearest plus the one pointing elsewhere, not the two nearest"
        );
    }

    /// What the naive alternative would have produced, asserted so the difference is on the
    /// record rather than described.
    #[test]
    fn the_naive_choice_would_have_been_the_two_nearest() {
        let (_store, mut candidates) = clustered();
        candidates.sort_unstable();
        let naive: Vec<NodeId> = candidates.iter().take(2).map(|c| c.id).collect();
        assert_eq!(naive, vec![0, 1]);
    }

    /// With `keep_pruned` the list is topped back up to `m` from the discarded candidates, so
    /// degree does not silently fall below `M`.
    #[test]
    fn keep_pruned_refills_to_m() {
        let (store, mut candidates) = clustered();
        let mut counter = DistanceCounter::new();

        let kept = select_neighbors::<L2Squared>(&store, &mut candidates, 3, true, &mut counter);
        assert_eq!(
            kept,
            vec![0, 3, 1],
            "diverse picks first, then the nearest pruned one"
        );

        let dropped =
            select_neighbors::<L2Squared>(&store, &mut candidates, 3, false, &mut counter);
        assert_eq!(dropped, vec![0, 3], "without the flag the list stays short");
    }

    #[test]
    fn the_first_candidate_is_always_kept() {
        let (store, mut candidates) = clustered();
        let mut counter = DistanceCounter::new();
        let selected =
            select_neighbors::<L2Squared>(&store, &mut candidates, 1, false, &mut counter);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn m_of_zero_selects_nothing() {
        let (store, mut candidates) = clustered();
        let mut counter = DistanceCounter::new();
        assert!(
            select_neighbors::<L2Squared>(&store, &mut candidates, 0, true, &mut counter)
                .is_empty()
        );
    }

    #[test]
    fn no_candidates_selects_nothing() {
        let (store, _) = clustered();
        let mut empty = Vec::new();
        let mut counter = DistanceCounter::new();
        assert!(
            select_neighbors::<L2Squared>(&store, &mut empty, 8, true, &mut counter).is_empty()
        );
    }

    #[test]
    fn fewer_candidates_than_m_returns_all_of_them() {
        let (store, mut candidates) = clustered();
        let mut counter = DistanceCounter::new();
        let selected =
            select_neighbors::<L2Squared>(&store, &mut candidates, 10, true, &mut counter);
        assert_eq!(selected.len(), 4);
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    /// Well-separated points in different directions dominate nobody, so the heuristic keeps them
    /// all and behaves like plain nearest-`m`. That equivalence is the sanity check that it is not
    /// pruning indiscriminately.
    #[test]
    fn well_spread_candidates_are_all_kept_in_distance_order() {
        let store = VectorStore::from_flat(
            2,
            vec![
                1.0, 0.0, //
                0.0, 2.0, //
                -3.0, 0.0, //
                0.0, -4.0,
            ],
        )
        .unwrap();
        let mut candidates = vec![
            Candidate::new(1.0, 0),
            Candidate::new(4.0, 1),
            Candidate::new(9.0, 2),
            Candidate::new(16.0, 3),
        ];
        let mut counter = DistanceCounter::new();

        let selected =
            select_neighbors::<L2Squared>(&store, &mut candidates, 3, false, &mut counter);
        assert_eq!(selected, vec![0, 1, 2]);
    }

    /// Duplicates of one point: the first survives, the rest are dominated at distance zero.
    #[test]
    fn identical_candidates_collapse_to_one() {
        let store = VectorStore::from_flat(1, vec![1.0, 1.0, 1.0]).unwrap();
        let mut candidates = vec![
            Candidate::new(1.0, 0),
            Candidate::new(1.0, 1),
            Candidate::new(1.0, 2),
        ];
        let mut counter = DistanceCounter::new();

        let selected =
            select_neighbors::<L2Squared>(&store, &mut candidates, 3, false, &mut counter);
        assert_eq!(selected, vec![0], "co-located points add no reachability");

        let refilled =
            select_neighbors::<L2Squared>(&store, &mut candidates, 3, true, &mut counter);
        assert_eq!(refilled, vec![0, 1, 2]);
    }

    /// Selection is deterministic: the same candidates in any input order give the same answer.
    #[test]
    fn input_order_does_not_matter() {
        let (store, candidates) = clustered();
        let mut counter = DistanceCounter::new();

        let mut forward = candidates.clone();
        let mut backward: Vec<Candidate> = candidates.into_iter().rev().collect();

        let a = select_neighbors::<L2Squared>(&store, &mut forward, 3, true, &mut counter);
        let b = select_neighbors::<L2Squared>(&store, &mut backward, 3, true, &mut counter);
        assert_eq!(a, b);
    }
}
