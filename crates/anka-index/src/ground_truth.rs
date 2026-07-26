//! Ground truth: exact neighbour lists, and the machinery for holding ours to a published one.
//!
//! The published SIFT1M ground truth is the only external check this project has on its own
//! arithmetic. Matching it exactly is not a nicety — every recall number in `docs/RESULTS.md`
//! is `|ours ∩ truth| / k`, so a truth that is 99% right silently caps and distorts everything
//! measured against it.
//!
//! Two questions get asked, and they are not the same question:
//!
//! 1. Does the **reference** kernel reproduce the published list exactly? It must. Anything
//!    less means the distance function, the ordering, or the tie-break is wrong.
//! 2. Does the **SIMD** kernel produce a list that is *distance-equivalent* to the reference?
//!    Exact id equality is the wrong bar here: SIMD sums in a different order, so two genuinely
//!    equidistant neighbours can swap places. What must hold is that the distances at every
//!    rank agree.

use anka_core::dataset::IntMatrix;
use anka_core::{Metric, NodeId, VectorError, VectorStore};
use rayon::prelude::*;

use crate::brute_force::{BruteForceIndex, Kernel};

/// How many differing rows or positions to keep as examples. Enough to debug with, few enough
/// to print.
const MAX_EXAMPLES: usize = 5;

/// Computes exact top-`k` neighbours for every query.
///
/// Parallel over queries: each worker scans the whole base set for its own queries, which needs
/// no synchronisation and no shared mutable state. Construction time is reported, but this is
/// not a throughput measurement — spec section 6 requires query benchmarks to be
/// single-threaded.
pub fn compute<M: Metric>(
    base: &VectorStore,
    queries: &VectorStore,
    k: usize,
    kernel: Kernel,
) -> Result<IntMatrix, VectorError> {
    if base.dim() != queries.dim() {
        return Err(VectorError::DimMismatch {
            expected: base.dim(),
            found: queries.dim(),
        });
    }
    // Every row must have exactly k entries or the result is not a matrix. Asking for more
    // neighbours than exist is a caller error rather than something to silently truncate.
    if k == 0 || k > base.len() {
        return Err(VectorError::InvalidK {
            k,
            available: base.len(),
        });
    }

    let index = BruteForceIndex::new(base);
    let rows: Vec<Vec<i32>> = queries
        .as_slice()
        .par_chunks_exact(queries.dim())
        .map(|query| {
            index
                .search::<M>(query, k, kernel)
                .map(|found| found.iter().map(|c| c.id as i32).collect())
        })
        .collect::<Result<_, _>>()?;

    IntMatrix::new(k, rows.into_iter().flatten().collect())
}

/// A row of one list that differs from the corresponding row of the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDifference {
    pub query: usize,
    /// First rank at which the two rows disagree.
    ///
    /// Recorded because disagreements cluster at *deep* ranks, where neighbours bunch together
    /// and ties become common. Printing the head of such a row shows two identical prefixes and
    /// tells the reader nothing.
    pub first_difference: usize,
    pub ours: Vec<i32>,
    pub reference: Vec<i32>,
}

/// Id-level agreement between two neighbour lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agreement {
    pub queries: usize,
    pub k: usize,
    /// Rows identical id-for-id, in order.
    pub identical_rows: usize,
    /// Rows holding the same set of ids, in any order.
    pub same_set_rows: usize,
    /// Positions at which both lists name the same id.
    pub identical_positions: usize,
    pub examples: Vec<RowDifference>,
}

impl Agreement {
    /// Whether the two lists are identical everywhere. This is the bar the reference kernel has
    /// to clear against the published ground truth.
    pub fn is_exact(&self) -> bool {
        self.identical_rows == self.queries
    }

    pub fn row_ratio(&self) -> f64 {
        ratio(self.identical_rows, self.queries)
    }

    pub fn same_set_ratio(&self) -> f64 {
        ratio(self.same_set_rows, self.queries)
    }

    pub fn position_ratio(&self) -> f64 {
        ratio(self.identical_positions, self.queries * self.k)
    }
}

/// Compares two neighbour lists position by position.
///
/// `ours` sets the width: when the published list carries more neighbours than were computed
/// (SIFT1M ships 100), only its first `k` columns take part. Comparing a top-10 list against a
/// top-100 one is otherwise meaningless.
pub fn compare(ours: &IntMatrix, reference: &IntMatrix) -> Result<Agreement, VectorError> {
    let k = ours.dim();
    if ours.len() != reference.len() || reference.dim() < k {
        return Err(VectorError::ShapeMismatch {
            left: "ours",
            left_rows: ours.len(),
            left_cols: ours.dim(),
            right: "reference",
            right_rows: reference.len(),
            right_cols: reference.dim(),
        });
    }

    let mut agreement = Agreement {
        queries: ours.len(),
        k,
        identical_rows: 0,
        same_set_rows: 0,
        identical_positions: 0,
        examples: Vec::new(),
    };

    for query in 0..ours.len() {
        let mine = ours.row(query);
        let theirs = &reference.row(query)[..k];

        let matching = mine.iter().zip(theirs).filter(|(a, b)| a == b).count();
        agreement.identical_positions += matching;

        if matching == k {
            agreement.identical_rows += 1;
            agreement.same_set_rows += 1;
            continue;
        }

        let mut a = mine.to_vec();
        let mut b = theirs.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        if a == b {
            agreement.same_set_rows += 1;
        }

        if agreement.examples.len() < MAX_EXAMPLES {
            agreement.examples.push(RowDifference {
                query,
                first_difference: mine
                    .iter()
                    .zip(theirs)
                    .position(|(a, b)| a != b)
                    .unwrap_or(k),
                ours: mine.to_vec(),
                reference: theirs.to_vec(),
            });
        }
    }

    Ok(agreement)
}

/// A position where two lists name different ids, with the distances involved.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionDifference {
    pub query: usize,
    pub rank: usize,
    pub our_id: i32,
    pub our_distance: f32,
    pub reference_id: i32,
    pub reference_distance: f32,
    pub relative_gap: f64,
}

/// Distance-level agreement between two neighbour lists.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceAgreement {
    pub queries: usize,
    pub k: usize,
    /// Positions where the ids differ.
    pub differing_positions: usize,
    /// Of those, positions where the distances nevertheless agree within tolerance.
    pub equivalent_positions: usize,
    /// Largest relative distance gap seen at any differing position.
    pub max_relative_gap: f64,
    pub examples: Vec<PositionDifference>,
}

impl DistanceAgreement {
    /// Whether every disagreement is merely a reordering of equidistant neighbours.
    pub fn is_equivalent(&self) -> bool {
        self.differing_positions == self.equivalent_positions
    }
}

/// Asks whether two neighbour lists are distance-equivalent.
///
/// Only positions whose ids differ are examined; where the ids agree there is nothing to check.
/// For each such position the distance to our id and to theirs are recomputed with the
/// **reference** kernel — using the fast kernel here would be asking the suspect to vouch for
/// itself.
pub fn distance_equivalence<M: Metric>(
    base: &VectorStore,
    queries: &VectorStore,
    ours: &IntMatrix,
    reference: &IntMatrix,
    relative_tolerance: f64,
) -> Result<DistanceAgreement, VectorError> {
    let k = ours.dim();
    if ours.len() != reference.len() || reference.dim() < k || queries.len() != ours.len() {
        return Err(VectorError::ShapeMismatch {
            left: "ours",
            left_rows: ours.len(),
            left_cols: ours.dim(),
            right: "reference",
            right_rows: reference.len(),
            right_cols: reference.dim(),
        });
    }

    let index = BruteForceIndex::new(base);
    let mut result = DistanceAgreement {
        queries: ours.len(),
        k,
        differing_positions: 0,
        equivalent_positions: 0,
        max_relative_gap: 0.0,
        examples: Vec::new(),
    };

    for query_index in 0..ours.len() {
        let query = queries.get(query_index);
        let mine = ours.row(query_index);
        let theirs = &reference.row(query_index)[..k];

        for (rank, (our_id, reference_id)) in mine.iter().zip(theirs).enumerate() {
            if our_id == reference_id {
                continue;
            }
            result.differing_positions += 1;

            let pair = [*our_id as NodeId, *reference_id as NodeId];
            let distances = index.distances_to::<M>(query, &pair, Kernel::Reference)?;
            let (ours_d, theirs_d) = (distances[0], distances[1]);

            let gap = relative_gap(ours_d, theirs_d);
            result.max_relative_gap = result.max_relative_gap.max(gap);

            if gap <= relative_tolerance {
                result.equivalent_positions += 1;
            } else if result.examples.len() < MAX_EXAMPLES {
                result.examples.push(PositionDifference {
                    query: query_index,
                    rank,
                    our_id: *our_id,
                    our_distance: ours_d,
                    reference_id: *reference_id,
                    reference_distance: theirs_d,
                    relative_gap: gap,
                });
            }
        }
    }

    Ok(result)
}

/// Relative difference between two distances, scaled by the larger magnitude.
///
/// Two exactly-equal values give zero regardless of magnitude, which is the common case here:
/// a swap between neighbours whose distances are bit-identical.
fn relative_gap(a: f32, b: f32) -> f64 {
    let (a, b) = (f64::from(a), f64::from(b));
    let difference = (a - b).abs();
    if difference == 0.0 {
        return 0.0;
    }
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        return difference;
    }
    difference / scale
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        1.0
    } else {
        part as f64 / whole as f64
    }
}

#[cfg(test)]
mod tests {
    use anka_core::L2Squared;

    use super::*;

    fn store(dim: usize, data: Vec<f32>) -> VectorStore {
        VectorStore::from_flat(dim, data).unwrap()
    }

    #[test]
    fn computes_exact_neighbours_for_every_query() {
        let base = store(1, vec![0.0, 10.0, 20.0, 30.0]);
        let queries = store(1, vec![1.0, 29.0]);

        let gt = compute::<L2Squared>(&base, &queries, 2, Kernel::Reference).unwrap();
        assert_eq!(gt.len(), 2);
        assert_eq!(gt.dim(), 2);
        assert_eq!(gt.row(0), &[0, 1]);
        assert_eq!(gt.row(1), &[3, 2]);
    }

    #[test]
    fn both_kernels_produce_the_same_ground_truth() {
        let base: Vec<f32> = (0..2000).map(|i| (i % 97) as f32 * 1.5).collect();
        let base = store(4, base);
        let queries = store(4, vec![1.0, 2.0, 3.0, 4.0, 40.0, 41.0, 42.0, 43.0]);

        let reference = compute::<L2Squared>(&base, &queries, 10, Kernel::Reference).unwrap();
        let fast = compute::<L2Squared>(&base, &queries, 10, Kernel::Fast).unwrap();
        assert_eq!(reference, fast);
    }

    #[test]
    fn k_outside_the_valid_range_is_an_error() {
        let base = store(1, vec![0.0, 1.0]);
        let queries = store(1, vec![0.0]);

        assert!(matches!(
            compute::<L2Squared>(&base, &queries, 0, Kernel::Fast),
            Err(VectorError::InvalidK { k: 0, available: 2 })
        ));
        assert!(matches!(
            compute::<L2Squared>(&base, &queries, 3, Kernel::Fast),
            Err(VectorError::InvalidK { k: 3, available: 2 })
        ));
    }

    #[test]
    fn mismatched_dimensions_are_an_error() {
        let base = store(2, vec![0.0, 1.0]);
        let queries = store(3, vec![0.0, 1.0, 2.0]);
        assert!(matches!(
            compute::<L2Squared>(&base, &queries, 1, Kernel::Fast),
            Err(VectorError::DimMismatch { .. })
        ));
    }

    #[test]
    fn identical_lists_agree_completely() {
        let a = IntMatrix::new(3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let agreement = compare(&a, &a).unwrap();

        assert!(agreement.is_exact());
        assert_eq!(agreement.identical_rows, 2);
        assert_eq!(agreement.same_set_rows, 2);
        assert_eq!(agreement.identical_positions, 6);
        assert_eq!(agreement.row_ratio(), 1.0);
        assert_eq!(agreement.position_ratio(), 1.0);
        assert!(agreement.examples.is_empty());
    }

    /// A reordering is not an exact match but is a set match — the distinction that tells a
    /// tie-break bug apart from a distance bug.
    #[test]
    fn a_reordered_row_matches_as_a_set_but_not_exactly() {
        let ours = IntMatrix::new(3, vec![1, 3, 2]).unwrap();
        let reference = IntMatrix::new(3, vec![1, 2, 3]).unwrap();
        let agreement = compare(&ours, &reference).unwrap();

        assert!(!agreement.is_exact());
        assert_eq!(agreement.identical_rows, 0);
        assert_eq!(agreement.same_set_rows, 1);
        assert_eq!(agreement.identical_positions, 1);
        assert_eq!(agreement.examples.len(), 1);
        assert_eq!(agreement.examples[0].query, 0);
        assert_eq!(agreement.examples[0].first_difference, 1);
    }

    /// SIFT1M ships 100 neighbours per query; a top-10 run compares against the first ten.
    #[test]
    fn a_narrower_list_compares_against_the_leading_columns() {
        let ours = IntMatrix::new(2, vec![1, 2]).unwrap();
        let reference = IntMatrix::new(5, vec![1, 2, 3, 4, 5]).unwrap();
        assert!(compare(&ours, &reference).unwrap().is_exact());
    }

    #[test]
    fn a_wider_list_than_the_reference_is_an_error() {
        let ours = IntMatrix::new(5, vec![1, 2, 3, 4, 5]).unwrap();
        let reference = IntMatrix::new(2, vec![1, 2]).unwrap();
        assert!(matches!(
            compare(&ours, &reference),
            Err(VectorError::ShapeMismatch { .. })
        ));
    }

    /// Two vectors sit at exactly the same distance from the query. Whichever one a list names,
    /// the lists are distance-equivalent even though the ids differ.
    #[test]
    fn swapping_equidistant_neighbours_is_distance_equivalent() {
        let base = store(1, vec![-1.0, 1.0, 5.0]);
        let queries = store(1, vec![0.0]);

        let ours = IntMatrix::new(1, vec![1]).unwrap();
        let reference = IntMatrix::new(1, vec![0]).unwrap();

        let result =
            distance_equivalence::<L2Squared>(&base, &queries, &ours, &reference, 1e-6).unwrap();
        assert!(result.is_equivalent());
        assert_eq!(result.differing_positions, 1);
        assert_eq!(result.equivalent_positions, 1);
        assert_eq!(result.max_relative_gap, 0.0);
    }

    /// A genuinely wrong neighbour is not equivalent, and the report says how far off it was.
    #[test]
    fn a_genuinely_different_neighbour_is_not_equivalent() {
        let base = store(1, vec![1.0, 100.0]);
        let queries = store(1, vec![0.0]);

        let ours = IntMatrix::new(1, vec![1]).unwrap();
        let reference = IntMatrix::new(1, vec![0]).unwrap();

        let result =
            distance_equivalence::<L2Squared>(&base, &queries, &ours, &reference, 1e-6).unwrap();
        assert!(!result.is_equivalent());
        assert_eq!(result.differing_positions, 1);
        assert_eq!(result.equivalent_positions, 0);
        assert!(result.max_relative_gap > 0.99);
        assert_eq!(result.examples.len(), 1);
        assert_eq!(result.examples[0].our_id, 1);
        assert_eq!(result.examples[0].reference_id, 0);
    }

    #[test]
    fn identical_lists_have_nothing_to_examine() {
        let base = store(1, vec![1.0, 2.0]);
        let queries = store(1, vec![0.0]);
        let list = IntMatrix::new(2, vec![0, 1]).unwrap();

        let result =
            distance_equivalence::<L2Squared>(&base, &queries, &list, &list, 1e-6).unwrap();
        assert!(result.is_equivalent());
        assert_eq!(result.differing_positions, 0);
    }

    #[test]
    fn an_out_of_range_id_in_a_list_is_an_error() {
        let base = store(1, vec![1.0, 2.0]);
        let queries = store(1, vec![0.0]);
        let ours = IntMatrix::new(1, vec![7]).unwrap();
        let reference = IntMatrix::new(1, vec![0]).unwrap();

        assert!(matches!(
            distance_equivalence::<L2Squared>(&base, &queries, &ours, &reference, 1e-6),
            Err(VectorError::IdOutOfRange { id: 7, count: 2 })
        ));
    }
}
