//! Algorithms 1 and 5: insert, and the top-level search.

use anka_core::{Candidate, Metric, MetricKind, NodeId, VectorStore};

use crate::error::IndexError;
use crate::hnsw::layer::Layer;
use crate::hnsw::params::{HnswParams, MAX_LEVEL};
use crate::hnsw::search::Searcher;
use crate::hnsw::select::select_neighbors;
use crate::hnsw::stats::DistanceCounter;

/// A hierarchical navigable small world graph over an owned [`VectorStore`].
///
/// The index owns its vectors: inserting appends to the store, so a node's id is its position.
/// That is what lets layer 0 address neighbour slots by node id with no indirection, and it is the
/// shape phase 3 will snapshot.
pub struct HnswIndex {
    vectors: VectorStore,
    params: HnswParams,
    metric: MetricKind,
    /// Layer 0 is dense; every layer above it is sparse.
    layers: Vec<Layer>,
    /// Highest layer each node reaches, indexed by node id.
    node_levels: Vec<u8>,
    entry_point: Option<NodeId>,
    max_layer: usize,
    levels: crate::hnsw::params::LevelGenerator,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: MetricKind, params: HnswParams) -> Result<Self, IndexError> {
        Self::with_capacity(dim, metric, params, 0)
    }

    pub fn with_capacity(
        dim: usize,
        metric: MetricKind,
        params: HnswParams,
        capacity: usize,
    ) -> Result<Self, IndexError> {
        Ok(Self {
            vectors: VectorStore::empty(dim)?,
            layers: vec![Layer::dense(params.max_degree(0), capacity)],
            node_levels: Vec::with_capacity(capacity),
            levels: params.level_generator(),
            params,
            metric,
            entry_point: None,
            max_layer: 0,
        })
    }

    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    pub fn metric(&self) -> MetricKind {
        self.metric
    }

    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    pub fn vectors(&self) -> &VectorStore {
        &self.vectors
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Highest layer `node` reaches.
    pub fn level_of(&self, node: NodeId) -> Option<usize> {
        self.node_levels.get(node as usize).map(|l| *l as usize)
    }

    /// Bytes held by the graph, excluding the vectors.
    pub fn graph_bytes(&self) -> usize {
        self.layers.iter().map(Layer::memory_bytes).sum::<usize>() + self.node_levels.capacity()
    }

    /// A searcher sized for this index.
    pub fn searcher(&self) -> Searcher {
        Searcher::new(self.len().max(1))
    }

    /// Inserts `vector`, drawing its level from the seeded generator.
    pub fn insert<M: Metric>(
        &mut self,
        searcher: &mut Searcher,
        vector: &[f32],
        counter: &mut DistanceCounter,
    ) -> Result<NodeId, IndexError> {
        let level = self.levels.next_level();
        self.insert_at_level::<M>(searcher, vector, level, counter)
    }

    /// Inserts `vector` at an explicit level.
    ///
    /// Exposed because phase 3 needs it: the write-ahead log records the level a node was given,
    /// and replay has to reproduce it rather than draw a fresh one. Redrawing would rebuild a
    /// *different* graph from the same log, which makes "identical results after restart"
    /// untestable.
    pub fn insert_at_level<M: Metric>(
        &mut self,
        searcher: &mut Searcher,
        vector: &[f32],
        level: usize,
        counter: &mut DistanceCounter,
    ) -> Result<NodeId, IndexError> {
        debug_assert_eq!(
            M::NAME,
            self.metric.name(),
            "the metric passed to insert must match the one the index was built with"
        );
        if vector.len() != self.dim() {
            return Err(IndexError::DimMismatch {
                expected: self.dim(),
                found: vector.len(),
            });
        }

        let level = level.min(MAX_LEVEL);
        let node = self.vectors.len() as NodeId;
        self.vectors.push(vector)?;
        self.node_levels.push(level as u8);
        searcher.ensure_capacity(self.vectors.len());

        // A node present at `level` is present on every layer below it too.
        while self.layers.len() <= level {
            let lc = self.layers.len();
            self.layers.push(Layer::sparse(self.params.max_degree(lc)));
        }
        for lc in 0..=level {
            self.layers[lc].add(node);
        }

        let Some(entry) = self.entry_point else {
            // The very first node. There is nothing to search and no entry point to search from —
            // the case a direct reading of Algorithm 1 skips straight past.
            self.entry_point = Some(node);
            self.max_layer = level;
            return Ok(node);
        };

        // Greedy descent through the layers above `level`, beam of one.
        let mut entry_points = vec![entry];
        for lc in (level + 1..=self.max_layer).rev() {
            let found = searcher.search_layer::<M>(
                &self.vectors,
                &self.layers[lc],
                vector,
                &entry_points,
                1,
                counter,
            );
            if let Some(nearest) = found.first() {
                entry_points = vec![nearest.id];
            }
        }

        // Link on every layer from min(max_layer, level) down to 0.
        let ef_construction = self.params.ef_construction();
        let policy = self.params.selection();
        for lc in (0..=self.max_layer.min(level)).rev() {
            let mut candidates = searcher.search_layer::<M>(
                &self.vectors,
                &self.layers[lc],
                vector,
                &entry_points,
                ef_construction,
                counter,
            );

            let selected = select_neighbors::<M>(
                &self.vectors,
                &mut candidates,
                self.params.m(),
                policy,
                counter,
            );

            self.layers[lc].set_neighbors(node, &selected);
            for &neighbor in &selected {
                if !self.layers[lc].push_neighbor(neighbor, node) {
                    // The neighbour is at Mmax(lc). Re-select its whole list through the
                    // heuristic instead of dropping the edge.
                    //
                    // Skipping this step is the mistake that makes an index degrade invisibly:
                    // degree grows without bound as construction proceeds, memory inflates, and
                    // recall falls. The paper has it, and it is easy to read past.
                    self.reselect_neighbors::<M>(lc, neighbor, node, counter);
                }
            }

            // Algorithm 1's `ep <- W`: the whole result set seeds the next layer, not just the
            // nearest. Passing one node would make the descent a single greedy chain and lose the
            // breadth ef_construction just paid for.
            entry_points = candidates.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points = vec![node];
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(node);
        }
        Ok(node)
    }

    /// Re-selects `node`'s neighbour list on `layer`, including `extra`, capped at `Mmax(layer)`.
    ///
    /// Distances are measured from `node`, not from the vector being inserted: this is
    /// `SELECT_NEIGHBORS_HEURISTIC(e, eConn, Mmax)` from Algorithm 1, where `e` is the anchor.
    fn reselect_neighbors<M: Metric>(
        &mut self,
        layer: usize,
        node: NodeId,
        extra: NodeId,
        counter: &mut DistanceCounter,
    ) {
        let max_degree = self.params.max_degree(layer);

        let mut ids: Vec<NodeId> = self.layers[layer].neighbors(node).to_vec();
        ids.push(extra);

        let mut candidates = {
            let view = self.vectors.view();
            let anchor = view.get(node as usize);
            ids.iter()
                .map(|&id| {
                    counter.record(1);
                    Candidate::new(M::distance(anchor, view.get(id as usize)), id)
                })
                .collect::<Vec<_>>()
        };

        let selected = select_neighbors::<M>(
            &self.vectors,
            &mut candidates,
            max_degree,
            self.params.selection(),
            counter,
        );
        self.layers[layer].set_neighbors(node, &selected);
    }

    /// Algorithm 5: the `k` nearest neighbours of `query`.
    ///
    /// `ef` controls the beam on layer 0 and is the knob that trades recall against speed. Takes
    /// `&self` and a caller-supplied [`Searcher`] so that concurrent readers each bring their own
    /// scratch instead of contending for one inside the index.
    pub fn search<M: Metric>(
        &self,
        searcher: &mut Searcher,
        query: &[f32],
        k: usize,
        ef: usize,
        counter: &mut DistanceCounter,
    ) -> Result<Vec<Candidate>, IndexError> {
        debug_assert_eq!(
            M::NAME,
            self.metric.name(),
            "the metric passed to search must match the one the index was built with"
        );
        if query.len() != self.dim() {
            return Err(IndexError::DimMismatch {
                expected: self.dim(),
                found: query.len(),
            });
        }
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        // A beam narrower than k cannot return k results. Clamping here rather than trusting the
        // caller means a low ef degrades recall, which is the intent, instead of silently
        // returning short lists.
        let ef = ef.max(k);
        searcher.ensure_capacity(self.len());

        let mut entry_points = vec![entry];
        for lc in (1..=self.max_layer).rev() {
            let found = searcher.search_layer::<M>(
                &self.vectors,
                &self.layers[lc],
                query,
                &entry_points,
                1,
                counter,
            );
            if let Some(nearest) = found.first() {
                entry_points = vec![nearest.id];
            }
        }

        let mut found = searcher.search_layer::<M>(
            &self.vectors,
            &self.layers[0],
            query,
            &entry_points,
            ef,
            counter,
        );
        found.truncate(k);
        Ok(found)
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("len", &self.len())
            .field("dim", &self.dim())
            .field("metric", &self.metric)
            .field("layers", &self.layers.len())
            .field("max_layer", &self.max_layer)
            .field("entry_point", &self.entry_point)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use anka_core::L2Squared;

    use super::*;
    use crate::brute_force::{BruteForceIndex, Kernel};
    use crate::hnsw::select::SelectionPolicy;

    /// Deterministic pseudo-random points, so a failure is reproducible from its seed.
    fn points(seed: u64, count: usize, dim: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 100.0
            })
            .collect()
    }

    fn build(params: HnswParams, data: &[f32], dim: usize) -> HnswIndex {
        let count = data.len() / dim;
        let mut index =
            HnswIndex::with_capacity(dim, MetricKind::L2Squared, params, count).unwrap();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();
        for vector in data.chunks_exact(dim) {
            index
                .insert::<L2Squared>(&mut searcher, vector, &mut counter)
                .unwrap();
        }
        index
    }

    /// Fraction of the exact top-`k` that the index returned, averaged over queries.
    fn recall(index: &HnswIndex, queries: &[f32], dim: usize, k: usize, ef: usize) -> f64 {
        let contiguous = index
            .vectors()
            .as_contiguous()
            .expect("an index built in memory is contiguous")
            .to_vec();
        let store = VectorStore::from_flat(dim, contiguous).unwrap();
        let exact = BruteForceIndex::new(&store);

        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();
        let mut total = 0.0;
        let mut queried = 0;

        for query in queries.chunks_exact(dim) {
            let truth: Vec<NodeId> = exact
                .search::<L2Squared>(query, k, Kernel::Reference)
                .unwrap()
                .iter()
                .map(|c| c.id)
                .collect();
            let found = index
                .search::<L2Squared>(&mut searcher, query, k, ef, &mut counter)
                .unwrap();
            let hits = found.iter().filter(|c| truth.contains(&c.id)).count();
            total += hits as f64 / truth.len() as f64;
            queried += 1;
        }
        total / queried as f64
    }

    #[test]
    fn an_empty_index_returns_nothing() {
        let index = HnswIndex::new(4, MetricKind::L2Squared, HnswParams::default()).unwrap();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();

        assert!(index.is_empty());
        assert_eq!(index.entry_point(), None);
        assert!(
            index
                .search::<L2Squared>(&mut searcher, &[0.0; 4], 5, 10, &mut counter)
                .unwrap()
                .is_empty()
        );
    }

    /// The first insert has no entry point to search from — the case Algorithm 1 walks past.
    #[test]
    fn the_first_node_becomes_the_entry_point() {
        let mut index = HnswIndex::new(2, MetricKind::L2Squared, HnswParams::default()).unwrap();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();

        let node = index
            .insert::<L2Squared>(&mut searcher, &[1.0, 2.0], &mut counter)
            .unwrap();
        assert_eq!(node, 0);
        assert_eq!(index.entry_point(), Some(0));
        assert_eq!(index.max_layer(), index.level_of(0).unwrap());

        let found = index
            .search::<L2Squared>(&mut searcher, &[1.0, 2.0], 1, 10, &mut counter)
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, 0);
        assert_eq!(found[0].dist, 0.0);
    }

    #[test]
    fn a_small_index_is_exact() {
        let dim = 4;
        let data = points(1, 200, dim);
        let index = build(HnswParams::default(), &data, dim);

        assert_eq!(index.len(), 200);
        // 200 points, M=16: the graph is dense enough that search should be exact.
        assert_eq!(recall(&index, &points(2, 50, dim), dim, 10, 64), 1.0);
    }

    /// The headline behaviour: high recall on a graph big enough that brute force is not what is
    /// being measured.
    #[test]
    fn recall_is_high_on_a_larger_index() {
        let dim = 8;
        let data = points(3, 5_000, dim);
        let index = build(HnswParams::default(), &data, dim);

        let queries = points(4, 100, dim);
        let measured = recall(&index, &queries, dim, 10, 64);
        assert!(measured >= 0.95, "recall@10 was {measured:.4}");
    }

    /// Raising `ef` must not lower recall. A monotonic curve is what makes the phase 2 sweep
    /// meaningful; a non-monotonic one means the beam or the stopping condition is wrong.
    #[test]
    fn recall_does_not_fall_as_ef_grows() {
        let dim = 8;
        let data = points(5, 3_000, dim);
        let index = build(HnswParams::default(), &data, dim);
        let queries = points(6, 60, dim);

        let mut previous = 0.0;
        for ef in [10usize, 20, 40, 80, 160] {
            let measured = recall(&index, &queries, dim, 10, ef);
            assert!(
                measured >= previous - 1e-9,
                "recall fell from {previous:.4} to {measured:.4} when ef reached {ef}"
            );
            previous = measured;
        }
        assert!(previous >= 0.99, "recall at ef=160 was only {previous:.4}");
    }

    /// `ef` below `k` is clamped, so the index still returns `k` results.
    #[test]
    fn ef_is_clamped_up_to_k() {
        let dim = 4;
        let data = points(7, 500, dim);
        let index = build(HnswParams::default(), &data, dim);
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();

        let found = index
            .search::<L2Squared>(&mut searcher, &[50.0; 4], 20, 1, &mut counter)
            .unwrap();
        assert_eq!(found.len(), 20);
    }

    #[test]
    fn k_larger_than_the_index_returns_everything() {
        let dim = 2;
        let data = points(8, 30, dim);
        let index = build(HnswParams::default(), &data, dim);
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();

        let found = index
            .search::<L2Squared>(&mut searcher, &[10.0, 10.0], 500, 500, &mut counter)
            .unwrap();
        assert_eq!(found.len(), 30);
    }

    #[test]
    fn a_wrong_dimension_is_an_error() {
        let mut index = HnswIndex::new(3, MetricKind::L2Squared, HnswParams::default()).unwrap();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();

        assert!(matches!(
            index.insert::<L2Squared>(&mut searcher, &[1.0, 2.0], &mut counter),
            Err(IndexError::DimMismatch {
                expected: 3,
                found: 2
            })
        ));
        assert!(matches!(
            index.search::<L2Squared>(&mut searcher, &[1.0; 5], 1, 10, &mut counter),
            Err(IndexError::DimMismatch { .. })
        ));
    }

    /// **The pruning step this project could most easily have shipped without.** No node may
    /// exceed Mmax(layer); if the re-selection in insert were skipped, degrees would grow without
    /// bound and this would fail.
    #[test]
    fn no_node_exceeds_its_layer_degree_cap() {
        let dim = 4;
        let data = points(9, 4_000, dim);
        let params = HnswParams::default();
        let index = build(params, &data, dim);

        for (lc, layer) in index.layers().iter().enumerate() {
            let cap = params.max_degree(lc);
            assert_eq!(layer.max_degree(), cap);
            for node in layer.nodes() {
                assert!(
                    layer.degree(node) <= cap,
                    "node {node} on layer {lc} has degree {} > {cap}",
                    layer.degree(node)
                );
            }
        }
    }

    /// A node on layer `l` must be on every layer beneath it, or the descent drops through a hole.
    #[test]
    fn layer_membership_is_contiguous() {
        let dim = 4;
        let data = points(10, 2_000, dim);
        let index = build(HnswParams::default(), &data, dim);

        for node in 0..index.len() as NodeId {
            let level = index.level_of(node).unwrap();
            for lc in 0..=level {
                assert!(
                    index.layers()[lc].contains(node),
                    "node {node} has level {level} but is missing from layer {lc}"
                );
            }
            for (lc, layer) in index.layers().iter().enumerate().skip(level + 1) {
                assert!(
                    !layer.contains(node),
                    "node {node} has level {level} but appears on layer {lc}"
                );
            }
        }
    }

    #[test]
    fn the_entry_point_sits_on_the_top_layer() {
        let dim = 4;
        let data = points(11, 2_000, dim);
        let index = build(HnswParams::default(), &data, dim);

        let entry = index
            .entry_point()
            .expect("non-empty index has an entry point");
        assert_eq!(index.level_of(entry), Some(index.max_layer()));
        assert_eq!(index.layers().len(), index.max_layer() + 1);
    }

    #[test]
    fn no_node_is_isolated_on_layer_zero() {
        let dim = 4;
        let data = points(12, 1_000, dim);
        let index = build(HnswParams::default(), &data, dim);

        for node in 0..index.len() as NodeId {
            assert!(
                index.layers()[0].degree(node) > 0,
                "node {node} is unreachable on layer 0"
            );
        }
    }

    /// Same seed, same data, same graph — the reproducibility contract in one assertion.
    #[test]
    fn building_twice_gives_the_same_index() {
        let dim = 4;
        let data = points(13, 800, dim);

        let first = build(HnswParams::default(), &data, dim);
        let second = build(HnswParams::default(), &data, dim);

        assert_eq!(first.max_layer(), second.max_layer());
        assert_eq!(first.entry_point(), second.entry_point());
        for lc in 0..first.layers().len() {
            for node in first.layers()[lc].nodes() {
                assert_eq!(
                    first.layers()[lc].neighbors(node),
                    second.layers()[lc].neighbors(node),
                    "layer {lc}, node {node}"
                );
            }
        }
    }

    /// The ablation the phase 2 DoD asks for, as a test rather than a claim: dropping the
    /// heuristic costs recall on clustered data.
    #[test]
    fn the_heuristic_beats_nearest_m_on_clustered_data() {
        let dim = 8;
        // Twenty tight clusters, so short-range edges alone leave no route between them.
        let mut data = Vec::new();
        let mut state = 99u64;
        for cluster in 0..20 {
            let centre = cluster as f32 * 200.0;
            for _ in 0..150 {
                for _ in 0..dim {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let jitter = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
                    data.push(centre + jitter);
                }
            }
        }

        let queries = {
            let mut q = Vec::new();
            for cluster in 0..20 {
                for _ in 0..dim {
                    q.push(cluster as f32 * 200.0 + 0.5);
                }
            }
            q
        };

        let with = build(HnswParams::default(), &data, dim);
        let without = build(
            HnswParams::default()
                .with_selection(SelectionPolicy::naive())
                .unwrap(),
            &data,
            dim,
        );

        let recall_with = recall(&with, &queries, dim, 10, 32);
        let recall_without = recall(&without, &queries, dim, 10, 32);

        assert!(
            recall_with >= recall_without,
            "heuristic {recall_with:.4} should not be worse than nearest-m {recall_without:.4}"
        );
        assert!(
            recall_with >= 0.95,
            "heuristic recall was only {recall_with:.4}"
        );
    }

    /// The index exists to compute fewer distances than a scan. On 5 000 points a brute-force
    /// query costs 5 000; the graph should need a small fraction of that.
    #[test]
    #[cfg(feature = "stats")]
    fn search_computes_far_fewer_distances_than_a_scan() {
        let dim = 8;
        let count = 5_000;
        let data = points(14, count, dim);
        let index = build(HnswParams::default(), &data, dim);

        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();
        index
            .search::<L2Squared>(&mut searcher, &[50.0; 8], 10, 64, &mut counter)
            .unwrap();

        let computed = counter.count().unwrap();
        assert!(
            computed < count as u64 / 4,
            "one query cost {computed} distance computations against {count} vectors"
        );
    }
}
