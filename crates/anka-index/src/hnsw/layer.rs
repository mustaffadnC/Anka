//! Neighbour storage for one layer of the graph.
//!
//! Adjacency lives in a single flat `Vec`, one fixed-capacity slot per node:
//!
//! ```text
//! [ degree, n0, n1, ..., n_{max_degree-1} ][ degree, n0, ... ] ...
//! ```
//!
//! One allocation per layer, contiguous access while walking a neighbour list, and it
//! serialises to disk unchanged — offsets survive a restart where pointers would not.
//!
//! The part that is easy to get wrong is that **upper layers are sparse.** With `M = 16`, about
//! one node in 16 exists at layer 1, one in 256 at layer 2, and so on. Giving every layer a slot
//! per node would waste roughly 68 MB per layer on a million vectors — several hundred megabytes
//! for a graph whose useful content is 132 MB. Layers above 0 therefore keep a `NodeId → slot`
//! table and allocate slots only for the nodes they actually hold.

use anka_core::NodeId;

/// Marks a node absent from a sparse layer.
///
/// Safe as a sentinel: slot indices are bounded by the number of nodes in the layer, which is
/// bounded by `NodeId::MAX`, so a *slot* can never legitimately be `u32::MAX`.
const NO_SLOT: u32 = u32::MAX;

/// A layer's two arrays do not describe a layer.
///
/// Separate from [`crate::hnsw::validate::GraphViolation`], which is about a graph being wrong.
/// These are about the bytes not being a graph at all, which is what a corrupt file produces and
/// has to be caught before anything indexes into them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LayerShapeError {
    #[error("adjacency of {len} entries does not divide into slots of {stride}")]
    RaggedAdjacency { len: usize, stride: usize },

    #[error("dense layer carries a slot table of {len} entries; layer 0's slot is its node id")]
    DenseLayerHasSlotTable { len: usize },

    #[error("{nodes} nodes for {slots} adjacency slots")]
    SlotCountMismatch { nodes: usize, slots: usize },

    #[error("node {node} occupies two slots")]
    DuplicateNode { node: NodeId },
}

/// Adjacency for a single layer.
pub struct Layer {
    /// `NodeId → slot` for sparse layers. `None` on layer 0, which holds every node: there the
    /// slot *is* the node id, and a table would be 4 MB of identity function.
    slot_of: Option<Vec<u32>>,
    /// `slot → NodeId`. Empty on layer 0, where it is the identity. Needed to iterate and to
    /// serialise a sparse layer.
    nodes: Vec<NodeId>,
    /// Flat adjacency, `max_degree + 1` entries per slot.
    neighbors: Vec<NodeId>,
    max_degree: usize,
}

impl Layer {
    /// Layer 0: every node is present, slot index equals node id.
    pub fn dense(max_degree: usize, capacity: usize) -> Self {
        Self {
            slot_of: None,
            nodes: Vec::new(),
            neighbors: Vec::with_capacity(capacity * (max_degree + 1)),
            max_degree,
        }
    }

    /// Layers above 0: only the nodes assigned to this layer get a slot.
    pub fn sparse(max_degree: usize) -> Self {
        Self {
            slot_of: Some(Vec::new()),
            nodes: Vec::new(),
            neighbors: Vec::new(),
            max_degree,
        }
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Number of nodes present in this layer.
    pub fn len(&self) -> usize {
        if self.slot_of.is_some() {
            self.nodes.len()
        } else {
            self.neighbors.len() / self.stride()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_dense(&self) -> bool {
        self.slot_of.is_none()
    }

    #[inline]
    fn stride(&self) -> usize {
        self.max_degree + 1
    }

    #[inline]
    fn slot(&self, node: NodeId) -> Option<usize> {
        match &self.slot_of {
            // Dense: present iff a slot has been allocated for it.
            None => {
                let slot = node as usize;
                (slot * self.stride() < self.neighbors.len()).then_some(slot)
            }
            Some(map) => match map.get(node as usize).copied() {
                None | Some(NO_SLOT) => None,
                Some(slot) => Some(slot as usize),
            },
        }
    }

    pub fn contains(&self, node: NodeId) -> bool {
        self.slot(node).is_some()
    }

    /// Adds `node` with an empty neighbour list and returns its slot.
    ///
    /// # Panics
    ///
    /// On a dense layer, if `node` is not the next id in sequence — layer 0's slot index *is*
    /// the node id, so ids have to arrive in order. Callers insert vectors sequentially, and a
    /// gap would silently misalign every slot after it.
    pub fn add(&mut self, node: NodeId) -> usize {
        debug_assert!(!self.contains(node), "node {node} is already in this layer");

        let slot = match &mut self.slot_of {
            None => {
                let expected = self.neighbors.len() / (self.max_degree + 1);
                assert_eq!(
                    node as usize, expected,
                    "a dense layer requires sequential node ids"
                );
                expected
            }
            Some(map) => {
                let needed = node as usize + 1;
                if map.len() < needed {
                    map.resize(needed, NO_SLOT);
                }
                let slot = self.nodes.len();
                map[node as usize] = slot as u32;
                self.nodes.push(node);
                slot
            }
        };

        // Degree 0 plus max_degree unused entries. The padding is what makes the slot layout
        // addressable by multiplication instead of by chasing offsets.
        self.neighbors.push(0);
        self.neighbors
            .resize(self.neighbors.len() + self.max_degree, 0);
        slot
    }

    /// Neighbours of `node`.
    ///
    /// # Panics
    ///
    /// If `node` is not in this layer. Use [`Self::contains`] when that is in question.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let start = self.slot(node).expect("node is not in this layer") * self.stride();
        let degree = self.neighbors[start] as usize;
        debug_assert!(degree <= self.max_degree);
        &self.neighbors[start + 1..start + 1 + degree]
    }

    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        let start = self.slot(node).expect("node is not in this layer") * self.stride();
        self.neighbors[start] as usize
    }

    /// Replaces the neighbour list of `node`.
    ///
    /// # Panics
    ///
    /// If `node` is absent, or if `neighbors` is longer than `max_degree` — exceeding the cap
    /// means a pruning step was skipped, and silently truncating would hide that.
    pub fn set_neighbors(&mut self, node: NodeId, neighbors: &[NodeId]) {
        assert!(
            neighbors.len() <= self.max_degree,
            "{} neighbours exceeds max_degree {}",
            neighbors.len(),
            self.max_degree
        );
        let start = self.slot(node).expect("node is not in this layer") * self.stride();
        self.neighbors[start] = neighbors.len() as NodeId;
        self.neighbors[start + 1..start + 1 + neighbors.len()].copy_from_slice(neighbors);
    }

    /// Appends one neighbour if there is room.
    ///
    /// Returns `false` when the list is already at `max_degree`, which is the signal for the
    /// caller to re-select the whole list through the heuristic rather than to drop the edge.
    pub fn push_neighbor(&mut self, node: NodeId, neighbor: NodeId) -> bool {
        let start = self.slot(node).expect("node is not in this layer") * self.stride();
        let degree = self.neighbors[start] as usize;
        if degree >= self.max_degree {
            return false;
        }
        self.neighbors[start + 1 + degree] = neighbor;
        self.neighbors[start] = (degree + 1) as NodeId;
        true
    }

    /// Every node in this layer, in insertion order.
    pub fn nodes(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        match &self.slot_of {
            None => Box::new(0..self.len() as NodeId),
            Some(_) => Box::new(self.nodes.iter().copied()),
        }
    }

    /// The flat adjacency array, exactly as it is laid out in memory.
    ///
    /// This is what goes to disk. The layout survives the round trip unchanged because it holds
    /// slot offsets rather than pointers, which is why it was chosen.
    pub fn raw_neighbors(&self) -> &[NodeId] {
        &self.neighbors
    }

    /// `slot → NodeId` for a sparse layer; empty for a dense one, where it is the identity.
    pub fn slot_nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Rebuilds a layer from the two arrays [`Self::raw_neighbors`] and [`Self::slot_nodes`]
    /// return.
    ///
    /// Used by snapshot loading, where the arrays come from an untrusted file, so the shape is
    /// checked here rather than trusted: the adjacency array has to divide evenly into slots, and
    /// there has to be exactly one node per slot. Whether the *contents* make a valid graph — ids
    /// in range, no dangling edges — is [`crate::hnsw::HnswIndex::validate`]'s job, since that
    /// costs a pass over every edge and the caller decides when to pay it.
    pub fn from_parts(
        max_degree: usize,
        dense: bool,
        nodes: Vec<NodeId>,
        neighbors: Vec<NodeId>,
    ) -> Result<Self, LayerShapeError> {
        let stride = max_degree + 1;
        if !neighbors.len().is_multiple_of(stride) {
            return Err(LayerShapeError::RaggedAdjacency {
                len: neighbors.len(),
                stride,
            });
        }
        let slots = neighbors.len() / stride;

        if dense {
            if !nodes.is_empty() {
                return Err(LayerShapeError::DenseLayerHasSlotTable { len: nodes.len() });
            }
            return Ok(Self {
                slot_of: None,
                nodes,
                neighbors,
                max_degree,
            });
        }

        if nodes.len() != slots {
            return Err(LayerShapeError::SlotCountMismatch {
                nodes: nodes.len(),
                slots,
            });
        }

        // The `NodeId → slot` table is derived rather than stored: it is the inverse of `nodes`,
        // so writing it would be redundant bytes that could disagree with what they mirror.
        let mut slot_of = Vec::new();
        for (slot, &node) in nodes.iter().enumerate() {
            let needed = node as usize + 1;
            if slot_of.len() < needed {
                slot_of.resize(needed, NO_SLOT);
            }
            if slot_of[node as usize] != NO_SLOT {
                return Err(LayerShapeError::DuplicateNode { node });
            }
            slot_of[node as usize] = slot as u32;
        }

        Ok(Self {
            slot_of: Some(slot_of),
            nodes,
            neighbors,
            max_degree,
        })
    }

    /// Bytes held by this layer's tables.
    ///
    /// Reported per layer because the sparse-vs-dense decision is justified by these numbers,
    /// and a claim about memory should be checkable.
    pub fn memory_bytes(&self) -> usize {
        let map = self
            .slot_of
            .as_ref()
            .map_or(0, |m| m.capacity() * size_of::<u32>());
        map + self.nodes.capacity() * size_of::<NodeId>()
            + self.neighbors.capacity() * size_of::<NodeId>()
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("kind", &if self.is_dense() { "dense" } else { "sparse" })
            .field("nodes", &self.len())
            .field("max_degree", &self.max_degree)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dense_layer_addresses_slots_by_node_id() {
        let mut layer = Layer::dense(4, 8);
        assert!(layer.is_dense());
        assert!(layer.is_empty());

        for node in 0..3 {
            layer.add(node);
        }
        assert_eq!(layer.len(), 3);
        assert!(layer.contains(0));
        assert!(layer.contains(2));
        assert!(!layer.contains(3));
        assert_eq!(layer.nodes().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn a_sparse_layer_only_allocates_for_present_nodes() {
        let mut layer = Layer::sparse(4);
        assert!(!layer.is_dense());

        layer.add(100);
        layer.add(5);
        assert_eq!(layer.len(), 2);
        assert!(layer.contains(100));
        assert!(layer.contains(5));
        assert!(!layer.contains(0));
        assert!(!layer.contains(50));
        assert!(!layer.contains(1_000_000));
        assert_eq!(layer.nodes().collect::<Vec<_>>(), vec![100, 5]);
    }

    #[test]
    fn neighbours_start_empty_and_can_be_appended() {
        let mut layer = Layer::sparse(3);
        layer.add(7);
        assert_eq!(layer.degree(7), 0);
        assert_eq!(layer.neighbors(7), &[] as &[NodeId]);

        assert!(layer.push_neighbor(7, 1));
        assert!(layer.push_neighbor(7, 2));
        assert_eq!(layer.neighbors(7), &[1, 2]);
        assert_eq!(layer.degree(7), 2);
    }

    /// Reaching `max_degree` must be reported, not absorbed: it is the trigger for re-selecting
    /// the list through the heuristic. Dropping the edge instead is how a graph quietly
    /// degrades.
    #[test]
    fn appending_past_max_degree_is_refused() {
        let mut layer = Layer::sparse(2);
        layer.add(0);
        assert!(layer.push_neighbor(0, 1));
        assert!(layer.push_neighbor(0, 2));
        assert!(!layer.push_neighbor(0, 3));
        assert_eq!(layer.neighbors(0), &[1, 2]);
    }

    #[test]
    fn set_neighbors_replaces_the_whole_list() {
        let mut layer = Layer::sparse(4);
        layer.add(0);
        layer.set_neighbors(0, &[9, 8, 7]);
        assert_eq!(layer.neighbors(0), &[9, 8, 7]);

        layer.set_neighbors(0, &[1]);
        assert_eq!(layer.neighbors(0), &[1]);
        assert_eq!(layer.degree(0), 1);

        layer.set_neighbors(0, &[]);
        assert_eq!(layer.degree(0), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds max_degree")]
    fn set_neighbors_beyond_max_degree_panics() {
        let mut layer = Layer::sparse(2);
        layer.add(0);
        layer.set_neighbors(0, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "sequential node ids")]
    fn a_dense_layer_rejects_a_gap_in_node_ids() {
        let mut layer = Layer::dense(4, 8);
        layer.add(0);
        layer.add(2);
    }

    #[test]
    fn slots_stay_independent() {
        let mut layer = Layer::dense(3, 4);
        for node in 0..3 {
            layer.add(node);
        }
        layer.set_neighbors(0, &[10, 11, 12]);
        layer.set_neighbors(1, &[20]);
        layer.set_neighbors(2, &[30, 31]);

        assert_eq!(layer.neighbors(0), &[10, 11, 12]);
        assert_eq!(layer.neighbors(1), &[20]);
        assert_eq!(layer.neighbors(2), &[30, 31]);
    }

    /// What a snapshot round trip has to preserve: the same neighbours for the same nodes, and
    /// the `NodeId → slot` table rebuilt from the slot list rather than stored.
    #[test]
    fn a_layer_survives_a_round_trip_through_its_raw_arrays() {
        for dense in [true, false] {
            let mut layer = if dense {
                Layer::dense(4, 6)
            } else {
                Layer::sparse(4)
            };
            let members: Vec<NodeId> = if dense {
                (0..6).collect()
            } else {
                vec![3, 11, 40]
            };
            for &node in &members {
                layer.add(node);
            }
            layer.set_neighbors(members[0], &[members[1], members[2]]);
            layer.set_neighbors(members[2], &[members[0]]);

            let restored = Layer::from_parts(
                layer.max_degree(),
                dense,
                layer.slot_nodes().to_vec(),
                layer.raw_neighbors().to_vec(),
            )
            .expect("well-shaped");

            assert_eq!(restored.is_dense(), dense);
            assert_eq!(restored.len(), layer.len());
            assert_eq!(
                restored.nodes().collect::<Vec<_>>(),
                layer.nodes().collect::<Vec<_>>()
            );
            for &node in &members {
                assert!(restored.contains(node));
                assert_eq!(
                    restored.neighbors(node),
                    layer.neighbors(node),
                    "node {node}"
                );
            }
            // A node that was never added must still read as absent after the rebuild.
            assert!(!restored.contains(99));
        }
    }

    #[test]
    fn malformed_layer_arrays_are_rejected() {
        // 7 entries cannot divide into slots of 5.
        assert_eq!(
            Layer::from_parts(4, true, Vec::new(), vec![0; 7]).unwrap_err(),
            LayerShapeError::RaggedAdjacency { len: 7, stride: 5 }
        );
        // Layer 0's slot is its node id, so a slot table there is a contradiction.
        assert_eq!(
            Layer::from_parts(4, true, vec![0, 1], vec![0; 10]).unwrap_err(),
            LayerShapeError::DenseLayerHasSlotTable { len: 2 }
        );
        // Two slots of adjacency, three nodes claiming them.
        assert_eq!(
            Layer::from_parts(4, false, vec![0, 1, 2], vec![0; 10]).unwrap_err(),
            LayerShapeError::SlotCountMismatch { nodes: 3, slots: 2 }
        );
        // The same node in two slots would make the derived table ambiguous.
        assert_eq!(
            Layer::from_parts(4, false, vec![5, 5], vec![0; 10]).unwrap_err(),
            LayerShapeError::DuplicateNode { node: 5 }
        );
    }

    /// The reason sparse layers exist. A dense layer over a million nodes costs 4 bytes per
    /// slot entry times the stride; a sparse layer holding 60 of them costs its remap table
    /// plus 60 slots — orders of magnitude apart.
    #[test]
    fn a_sparse_layer_is_far_smaller_than_a_dense_one() {
        let node_count = 100_000;
        let max_degree = 16;

        let mut dense = Layer::dense(max_degree, node_count);
        for node in 0..node_count as NodeId {
            dense.add(node);
        }

        let mut sparse = Layer::sparse(max_degree);
        for node in (0..node_count as NodeId).step_by(16) {
            sparse.add(node);
        }

        assert_eq!(dense.len(), node_count);
        assert_eq!(sparse.len(), node_count / 16);
        assert!(
            sparse.memory_bytes() * 4 < dense.memory_bytes(),
            "sparse {} vs dense {}",
            sparse.memory_bytes(),
            dense.memory_bytes()
        );
    }
}
