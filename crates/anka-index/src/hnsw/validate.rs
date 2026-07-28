//! Graph invariants and measured graph properties.
//!
//! Two different things live here, and keeping them apart is the point.
//!
//! [`GraphViolation`] covers properties the algorithm **guarantees**. Any of them failing is a
//! bug, so `validate` is a gate.
//!
//! [`GraphStats`] covers properties worth **measuring**. Degree distribution belongs here because
//! the phase 2 ablations are about how it changes, and so does edge asymmetry — which the spec
//! originally listed as an invariant. See [`GraphStats::asymmetric_edges`] for why it is not one.

use std::collections::HashSet;

use anka_core::NodeId;

use crate::hnsw::index::HnswIndex;
use crate::hnsw::layer::Layer;

/// A property the algorithm guarantees, and which was found broken.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphViolation {
    #[error("node {node} on layer {layer} has degree {degree}, above the cap of {cap}")]
    DegreeAboveCap {
        layer: usize,
        node: NodeId,
        degree: usize,
        cap: usize,
    },

    /// A node reaching layer `l` must exist on every layer below it, or the descent falls through
    /// a hole and the lower layers become unreachable from above.
    #[error("node {node} reaches layer {level} but is missing from layer {missing}")]
    LayerGap {
        node: NodeId,
        level: usize,
        missing: usize,
    },

    #[error("node {node} records level {recorded} but also appears on layer {found}")]
    LevelMismatch {
        node: NodeId,
        recorded: usize,
        found: usize,
    },

    #[error("entry point {node} is on layer {level}, but max_layer is {max_layer}")]
    EntryPointBelowTop {
        node: NodeId,
        level: usize,
        max_layer: usize,
    },

    #[error("index holds {nodes} nodes but has no entry point")]
    MissingEntryPoint { nodes: usize },

    #[error("node {node} on layer {layer} lists itself as a neighbour")]
    SelfLoop { layer: usize, node: NodeId },

    #[error("node {node} on layer {layer} lists neighbour {neighbor} more than once")]
    DuplicateNeighbor {
        layer: usize,
        node: NodeId,
        neighbor: NodeId,
    },

    #[error("node {node} on layer {layer} lists {neighbor}, which is not in the index")]
    NeighborOutOfRange {
        layer: usize,
        node: NodeId,
        neighbor: NodeId,
    },

    /// An edge may only point at a node that exists on the same layer, or traversal would read a
    /// slot that was never allocated.
    #[error("node {node} on layer {layer} lists {neighbor}, which is not on that layer")]
    NeighborNotOnLayer {
        layer: usize,
        node: NodeId,
        neighbor: NodeId,
    },

    /// Layer 0 carries every node, so an isolated one there can never be returned by a search.
    #[error("node {node} is isolated on layer 0 and can never be reached")]
    IsolatedOnLayerZero { node: NodeId },
}

/// Measured properties of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStats {
    pub layer: usize,
    pub nodes: usize,
    pub edges: usize,
    pub degree_cap: usize,
    pub min_degree: usize,
    pub max_degree: usize,
    pub mean_degree: f64,
    pub isolated: usize,
    /// Directed edges on this layer whose reverse is absent.
    pub asymmetric_edges: usize,
    pub bytes: usize,
}

/// Measured properties of the whole graph.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    pub nodes: usize,
    pub max_layer: usize,
    pub edges: usize,
    /// Directed edges whose reverse is absent.
    ///
    /// **Not zero, and not a defect.** Insert adds edges in both directions, but the pruning step
    /// that follows re-selects a saturated neighbour's list through the heuristic — and that
    /// re-selection can drop the edge back to the node that just arrived, while the forward edge
    /// remains. The graph HNSW builds is directed; hnswlib's is too.
    ///
    /// The spec listed bidirectionality as an invariant. It is measured here instead of asserted,
    /// because asserting it would fail on a correct implementation.
    pub asymmetric_edges: usize,
    pub graph_bytes: usize,
    pub per_layer: Vec<LayerStats>,
}

impl GraphStats {
    /// Share of directed edges missing their reverse.
    pub fn asymmetry_ratio(&self) -> f64 {
        if self.edges == 0 {
            0.0
        } else {
            self.asymmetric_edges as f64 / self.edges as f64
        }
    }
}

impl HnswIndex {
    /// Checks every guaranteed invariant, returning the first violation found.
    ///
    /// Walks the whole graph, so it is a debug and test tool rather than something to run per
    /// insert.
    pub fn validate(&self) -> Result<(), GraphViolation> {
        if self.is_empty() {
            return Ok(());
        }

        let Some(entry) = self.entry_point() else {
            return Err(GraphViolation::MissingEntryPoint { nodes: self.len() });
        };
        let entry_level = self.level_of(entry).unwrap_or(0);
        if entry_level != self.max_layer() {
            return Err(GraphViolation::EntryPointBelowTop {
                node: entry,
                level: entry_level,
                max_layer: self.max_layer(),
            });
        }

        // Layer membership matches the recorded level, in both directions.
        for node in 0..self.len() as NodeId {
            let level = self.level_of(node).expect("node is in range");
            for lc in 0..=level {
                if !self.layers()[lc].contains(node) {
                    return Err(GraphViolation::LayerGap {
                        node,
                        level,
                        missing: lc,
                    });
                }
            }
            for (lc, layer) in self.layers().iter().enumerate().skip(level + 1) {
                if layer.contains(node) {
                    return Err(GraphViolation::LevelMismatch {
                        node,
                        recorded: level,
                        found: lc,
                    });
                }
            }
        }

        let mut seen: HashSet<NodeId> = HashSet::new();
        for (lc, layer) in self.layers().iter().enumerate() {
            let cap = layer.max_degree();
            for node in layer.nodes() {
                let neighbors = layer.neighbors(node);
                if neighbors.len() > cap {
                    return Err(GraphViolation::DegreeAboveCap {
                        layer: lc,
                        node,
                        degree: neighbors.len(),
                        cap,
                    });
                }
                if lc == 0 && self.len() > 1 && neighbors.is_empty() {
                    return Err(GraphViolation::IsolatedOnLayerZero { node });
                }

                seen.clear();
                for &neighbor in neighbors {
                    if neighbor == node {
                        return Err(GraphViolation::SelfLoop { layer: lc, node });
                    }
                    if neighbor as usize >= self.len() {
                        return Err(GraphViolation::NeighborOutOfRange {
                            layer: lc,
                            node,
                            neighbor,
                        });
                    }
                    if !layer.contains(neighbor) {
                        return Err(GraphViolation::NeighborNotOnLayer {
                            layer: lc,
                            node,
                            neighbor,
                        });
                    }
                    if !seen.insert(neighbor) {
                        return Err(GraphViolation::DuplicateNeighbor {
                            layer: lc,
                            node,
                            neighbor,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Measures the graph: degree distribution per layer, memory, and edge asymmetry.
    ///
    /// This is what the phase 2 ablations report. The `M` sweep needs memory and degree per layer;
    /// the keep-pruned ablation is precisely a claim about the degree distribution.
    pub fn graph_stats(&self) -> GraphStats {
        let per_layer: Vec<LayerStats> = self
            .layers()
            .iter()
            .enumerate()
            .map(|(lc, layer)| layer_stats(lc, layer))
            .collect();

        GraphStats {
            nodes: self.len(),
            max_layer: self.max_layer(),
            edges: per_layer.iter().map(|l| l.edges).sum(),
            asymmetric_edges: per_layer.iter().map(|l| l.asymmetric_edges).sum(),
            graph_bytes: self.graph_bytes(),
            per_layer,
        }
    }
}

fn layer_stats(index: usize, layer: &Layer) -> LayerStats {
    let mut edges = 0usize;
    let mut min_degree = usize::MAX;
    let mut max_degree = 0usize;
    let mut isolated = 0usize;
    let mut asymmetric_edges = 0usize;

    for node in layer.nodes() {
        let neighbors = layer.neighbors(node);
        edges += neighbors.len();
        min_degree = min_degree.min(neighbors.len());
        max_degree = max_degree.max(neighbors.len());
        if neighbors.is_empty() {
            isolated += 1;
        }
        for &neighbor in neighbors {
            if !layer.neighbors(neighbor).contains(&node) {
                asymmetric_edges += 1;
            }
        }
    }

    let nodes = layer.len();
    LayerStats {
        layer: index,
        nodes,
        edges,
        degree_cap: layer.max_degree(),
        min_degree: if nodes == 0 { 0 } else { min_degree },
        max_degree,
        mean_degree: if nodes == 0 {
            0.0
        } else {
            edges as f64 / nodes as f64
        },
        isolated,
        asymmetric_edges,
        bytes: layer.memory_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use anka_core::{L2Squared, MetricKind};

    use super::*;
    use crate::hnsw::params::HnswParams;
    use crate::hnsw::select::SelectionPolicy;
    use crate::hnsw::stats::DistanceCounter;

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

    #[test]
    fn an_empty_index_is_valid() {
        let index = HnswIndex::new(4, MetricKind::L2Squared, HnswParams::default()).unwrap();
        assert_eq!(index.validate(), Ok(()));

        let stats = index.graph_stats();
        assert_eq!(stats.nodes, 0);
        assert_eq!(stats.edges, 0);
        assert_eq!(stats.asymmetry_ratio(), 0.0);
    }

    #[test]
    fn a_single_node_index_is_valid() {
        let index = build(HnswParams::default(), &[1.0, 2.0], 2);
        assert_eq!(index.validate(), Ok(()));
    }

    #[test]
    fn a_built_index_satisfies_every_invariant() {
        for (seed, count, dim) in [(1u64, 200usize, 4usize), (2, 2_000, 8), (3, 5_000, 4)] {
            let index = build(HnswParams::default(), &points(seed, count, dim), dim);
            assert_eq!(
                index.validate(),
                Ok(()),
                "seed {seed}, {count} nodes, dim {dim}"
            );
        }
    }

    /// Both ablations have to produce structurally valid graphs too, or an ablation measurement
    /// would be comparing against something broken rather than against something worse.
    #[test]
    fn ablated_indexes_are_still_valid() {
        let dim = 4;
        let data = points(4, 1_500, dim);

        for policy in [
            SelectionPolicy::naive(),
            SelectionPolicy {
                heuristic: true,
                keep_pruned: false,
            },
        ] {
            let params = HnswParams::default().with_selection(policy).unwrap();
            let index = build(params, &data, dim);
            assert_eq!(index.validate(), Ok(()), "policy {policy:?}");
        }
    }

    #[test]
    fn stats_describe_the_layers() {
        let dim = 4;
        let count = 3_000;
        let params = HnswParams::default();
        let index = build(params, &points(5, count, dim), dim);
        let stats = index.graph_stats();

        assert_eq!(stats.nodes, count);
        assert_eq!(stats.per_layer.len(), stats.max_layer + 1);
        assert_eq!(stats.per_layer[0].nodes, count);
        assert_eq!(stats.per_layer[0].degree_cap, params.max_degree0());
        assert_eq!(stats.per_layer[0].isolated, 0);
        assert!(stats.per_layer[0].mean_degree > 1.0);

        // Layers thin out geometrically, so each one holds fewer nodes than the one below.
        for pair in stats.per_layer.windows(2) {
            assert!(
                pair[1].nodes < pair[0].nodes,
                "layer {} holds {} nodes, layer {} holds {}",
                pair[0].layer,
                pair[0].nodes,
                pair[1].layer,
                pair[1].nodes
            );
        }

        for layer in &stats.per_layer {
            assert!(layer.max_degree <= layer.degree_cap);
        }
    }

    /// `keep_pruned` is a claim about the degree distribution, so it is checked as one: turning it
    /// off must leave layer 0 with a lower mean degree.
    #[test]
    fn keep_pruned_raises_the_mean_degree() {
        let dim = 8;
        let data = points(6, 2_000, dim);

        let with = build(HnswParams::default(), &data, dim).graph_stats();
        let without = build(
            HnswParams::default()
                .with_selection(SelectionPolicy {
                    heuristic: true,
                    keep_pruned: false,
                })
                .unwrap(),
            &data,
            dim,
        )
        .graph_stats();

        assert!(
            with.per_layer[0].mean_degree > without.per_layer[0].mean_degree,
            "keep_pruned on: {:.2}, off: {:.2}",
            with.per_layer[0].mean_degree,
            without.per_layer[0].mean_degree
        );
    }

    /// The spec listed "all edges bidirectional" as an invariant. It is not one: the pruning step
    /// re-selects a saturated neighbour's list, and that can drop the edge back to the node that
    /// just arrived while the forward edge stays. This test records that the asymmetry exists and
    /// stays a small minority, which is the honest form of the claim.
    #[test]
    fn edge_asymmetry_exists_and_is_a_small_minority() {
        let dim = 8;
        let index = build(HnswParams::default(), &points(7, 4_000, dim), dim);
        let stats = index.graph_stats();

        assert!(stats.edges > 0);
        assert!(
            stats.asymmetric_edges > 0,
            "expected pruning to leave some one-way edges; measured {} of {}",
            stats.asymmetric_edges,
            stats.edges
        );
        assert!(
            stats.asymmetry_ratio() < 0.2,
            "asymmetry ratio {:.4} is higher than a pruning side effect should be",
            stats.asymmetry_ratio()
        );
        // Asymmetric or not, every edge still points at a node that exists on its layer.
        assert_eq!(index.validate(), Ok(()));
    }
}
